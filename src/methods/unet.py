"""Rung 4: U-Net whose first layer is a Bayar-Stamm constrained convolution.

The constrained layer computes "pixel minus a weighted average of its neighbours",
which suppresses scene content and keeps local residuals (like ELA's evidence).
norm="sum" (used) divides the surround weights by their sum; "l1" by their absolute sum.
upsample="resize" (used): nearest-neighbour x2 resize, then a 3x3 conv. Every output pixel
uses the same kernel, so the decoder cannot learn a checkerboard pattern (Odena et al. 2016).
upsample="transpose": the original U-Net's 2x2 transposed conv; kept only to reload old models.
"""
import torch                           # PyTorch
import torch.nn as nn                  # building blocks for networks (layers)
import torch.nn.functional as F        # functions such as loss functions


# ---- Constrained first layer ----

class ConstrainedConv2d(nn.Conv2d):
    """Conv layer with centre weight fixed at -1 and the other weights summing to 1."""
    def __init__(self, in_ch, out_ch, k=5, norm="sum"):
        # an ordinary k x k convolution; padding keeps the output the same size; no bias term
        super().__init__(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.k = k                              # kernel size (5)
        self.norm = norm                        # how to normalise the surrounding weights

    def constrain(self):
        """Re-apply the constraint; called after every optimiser step."""
        with torch.no_grad():                   # changing weights directly, not as part of training maths
            w = self.weight.data                # shape (out_channels, in_channels, k, k)
            c = self.k // 2                         # index of the centre weight
            w[:, :, c, c] = 0.0                     # exclude the centre from the normalisation
            if self.norm == "sum":
                w /= w.sum(dim=(2, 3), keepdim=True) + 1e-8   # scale so the surrounding weights add up to 1
            else:                                   # "l1"
                w /= w.abs().sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)   # scale so their absolute values add to 1
            w[:, :, c, c] = -1.0                    # fix the centre at -1
            # result: output = (weighted average of neighbours) - (centre pixel) = a prediction error

    @torch.no_grad()
    def pre_norm_sum(self):
        """Smallest |surround sum| before normalisation (call BEFORE constrain(); used to diagnose instability)."""
        w = self.weight.data.clone()            # copy, so the real weights aren't changed
        c = self.k // 2
        w[:, :, c, c] = 0.0                     # ignore the centre
        return w.sum(dim=(2, 3)).abs().min().item()   # the smallest number constrain() would divide by


# ---- U-Net ----

def block(i, o):
    """Two (3x3 conv, BatchNorm, ReLU) layers."""
    return nn.Sequential(                       # layers applied one after another
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),   # conv -> normalise -> ReLU
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))   # and again


# ---- Decoder upsampling ----
UPSAMPLE = ("resize", "transpose")              # the two allowed schemes


def up(i, o, scheme):
    """Double the resolution and map i channels to o. See the module docstring."""
    if scheme == "resize":                      # used for every reported model
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),   # copy each pixel into a 2x2 block
                             nn.Conv2d(i, o, 3, padding=1))                  # then an ordinary 3x3 conv
    if scheme == "transpose":                   # old decoder, only to reload old models
        return nn.ConvTranspose2d(i, o, 2, stride=2)                         # 2x2 transposed conv, stride 2
    raise ValueError(f"upsample must be one of {UPSAMPLE}, not {scheme!r}")  # typo in the setting


class UNet(nn.Module):
    def __init__(self, base=32, constrained=True, norm="sum", upsample="resize"):
        super().__init__()                      # set up the PyTorch module machinery
        self.constrained = constrained          # whether to use the constrained first layer
        self.upsample = upsample                # decoder upsampling scheme (recorded in each model's sidecar)
        self.pre = ConstrainedConv2d(3, 3, 5, norm=norm) if constrained else nn.Identity()   # 3 residual filters

        # Encoder: each block doubles the channels (32, 64, 128); pooling between blocks halves the size
        self.e1, self.e2, self.e3 = block(3, base), block(base, base*2), block(base*2, base*4)  # encoder
        self.bott = block(base*4, base*8)                                                       # bottleneck
        # Decoder: upsample, join with the matching encoder map (skip connection), then a block
        self.u3 = up(base*8, base*4, upsample)                         # decoder: upsample x2 ...
        self.d3 = block(base*8, base*4)                                # ... then convs on [upsampled | skip]
        self.u2 = up(base*4, base*2, upsample)                         # upsample x2 (64 x 64 -> 128 x 128 on a crop)
        self.d2 = block(base*4, base*2)                                # convs on [upsampled | e2]
        self.u1 = up(base*2, base, upsample)                           # upsample x2 back to full size
        self.d1 = block(base*2, base)                                  # convs on [upsampled | e1]
        self.out = nn.Conv2d(base, 1, 1)                               # one logit per pixel
        self.pool = nn.MaxPool2d(2)                                    # halves the resolution

    def forward(self, x):
        """Run the network: x is a batch of images (batch, 3, H, W); returns logits (batch, 1, H, W)."""
        x = self.pre(x)                         # constrained layer: residuals instead of raw pixels
        e1 = self.e1(x)                         # full resolution, 32 channels
        e2 = self.e2(self.pool(e1))             # 1/2 resolution, 64 channels
        e3 = self.e3(self.pool(e2))             # 1/4 resolution, 128 channels
        b = self.bott(self.pool(e3))            # 1/8 resolution, 256 channels
        # decoder: upsample, concatenate the encoder map of the same size (skip), convolve
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))    # back to 1/4 resolution
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))   # back to 1/2 resolution
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))   # back to full resolution
        return self.out(d1)                            # logits, one per pixel

    def constrain(self):
        """Re-apply the first layer's constraint (called after every weight update)."""
        if self.constrained:
            self.pre.constrain()

    @torch.no_grad()
    def pre_norm_sum(self):
        """See ConstrainedConv2d.pre_norm_sum."""
        return self.pre.pre_norm_sum() if self.constrained else float("nan")

    @torch.no_grad()
    def weight_norm(self):
        """L2 norm of all parameters (logged in the stability experiment)."""
        return sum(p.norm().item() ** 2 for p in self.parameters()) ** 0.5   # sqrt of the sum of squared weights


def dice_bce(logits, target, eps=1.0):
    """Loss = binary cross-entropy + Dice (Dice copes with the few spliced pixels)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)    # per-pixel classification loss
    p = torch.sigmoid(logits)                                   # logits -> probabilities between 0 and 1
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps     # soft Dice, per image in the batch
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps   # predicted area + true area
    return bce + (1 - num / den).mean()                 # Dice is 1 for a perfect match, so 1 - Dice is a loss
