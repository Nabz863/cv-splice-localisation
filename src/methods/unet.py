"""Rung 4: U-Net whose first layer is a Bayar-Stamm constrained convolution.

The constrained layer computes "pixel minus a weighted average of its neighbours",
which suppresses scene content and keeps local residuals (like ELA's evidence).
norm="sum" (used) divides the surround weights by their sum; "l1" by their absolute sum.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Constrained first layer ----

class ConstrainedConv2d(nn.Conv2d):
    """Conv layer with centre weight fixed at -1 and the other weights summing to 1."""
    def __init__(self, in_ch, out_ch, k=5, norm="sum"):
        super().__init__(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.k = k
        self.norm = norm

    def constrain(self):
        """Re-apply the constraint; called after every optimiser step."""
        with torch.no_grad():
            w = self.weight.data
            c = self.k // 2                         # index of the centre weight
            w[:, :, c, c] = 0.0                     # exclude the centre from the normalisation
            if self.norm == "sum":
                w /= w.sum(dim=(2, 3), keepdim=True) + 1e-8
            else:                                   # "l1"
                w /= w.abs().sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
            w[:, :, c, c] = -1.0                    # fix the centre at -1

    @torch.no_grad()
    def pre_norm_sum(self):
        """Smallest |surround sum| before normalisation (call BEFORE constrain(); used to diagnose instability)."""
        w = self.weight.data.clone()
        c = self.k // 2
        w[:, :, c, c] = 0.0
        return w.sum(dim=(2, 3)).abs().min().item()


# ---- U-Net ----

def block(i, o):
    """Two (3x3 conv, BatchNorm, ReLU) layers."""
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, constrained=True, norm="sum"):
        super().__init__()
        self.constrained = constrained
        self.pre = ConstrainedConv2d(3, 3, 5, norm=norm) if constrained else nn.Identity()   # 3 residual filters

        self.e1, self.e2, self.e3 = block(3, base), block(base, base*2), block(base*2, base*4)  # encoder
        self.bott = block(base*4, base*8)                                                       # bottleneck
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)      # decoder: 2x2 transposed conv, stride 2 (upsample x2) ...
        self.d3 = block(base*8, base*4)                                # ... then convs on [upsampled | skip]
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = block(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = block(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)                               # one logit per pixel
        self.pool = nn.MaxPool2d(2)                                    # halves the resolution

    def forward(self, x):
        x = self.pre(x)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        # decoder: upsample, concatenate the encoder map of the same size (skip), convolve
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)                            # logits, one per pixel

    def constrain(self):
        if self.constrained:
            self.pre.constrain()

    @torch.no_grad()
    def pre_norm_sum(self):
        """See ConstrainedConv2d.pre_norm_sum."""
        return self.pre.pre_norm_sum() if self.constrained else float("nan")

    @torch.no_grad()
    def weight_norm(self):
        """L2 norm of all parameters (logged in the stability experiment)."""
        return sum(p.norm().item() ** 2 for p in self.parameters()) ** 0.5


def dice_bce(logits, target, eps=1.0):
    """Loss = binary cross-entropy + Dice (Dice copes with the few spliced pixels)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps     # soft Dice, per image in the batch
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - num / den).mean()
