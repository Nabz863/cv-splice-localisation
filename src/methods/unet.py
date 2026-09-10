"""Rung 4: U-Net with a constrained first layer.

The constraint (Bayar & Stamm) forces conv1 to compute a residual: the centre
weight is fixed at -1 and the surround is normalised, so whatever the layer
learns, its output is always "pixel minus a weighted combination of its
neighbours". The network is pushed structurally toward noise and compression
statistics rather than semantic content.

Two normalisation schemes:

  "sum"  -- w /= w.sum().  The paper's formulation, and the default. UNBOUNDED:
            the surround weights can drift toward cancelling, sending the sum
            toward zero and amplifying the division without limit. Observed to
            produce non-finite losses from epoch ~18-22 in three of five folds.

  "l1"   -- w /= |w|.sum().  Bounded by construction -- the denominator is a sum
            of magnitudes, so it is only small when every weight is small. Stable,
            but measurably worse validation loss in the stabiliser comparison,
            because it changes the filter's effective scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, k=5, norm="sum"):
        super().__init__(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.k = k
        self.norm = norm

    def constrain(self):
        """Re-impose the constraint. Call after every optimiser step."""
        with torch.no_grad():
            w = self.weight.data
            c = self.k // 2
            w[:, :, c, c] = 0.0
            if self.norm == "sum":
                w /= w.sum(dim=(2, 3), keepdim=True) + 1e-8
            else:                                   # "l1"
                w /= w.abs().sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
            w[:, :, c, c] = -1.0

    @torch.no_grad()
    def pre_norm_sum(self):
        """Smallest |surround sum| BEFORE normalisation.

        This is the quantity that drives the instability: constrain() divides by
        it, so as it approaches zero the weights are amplified without bound.
        MUST be called after the optimiser step and BEFORE constrain(); calling
        it afterwards returns 1 by construction and measures nothing.
        """
        w = self.weight.data.clone()
        c = self.k // 2
        w[:, :, c, c] = 0.0
        return w.sum(dim=(2, 3)).abs().min().item()


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, constrained=True, norm="sum"):
        super().__init__()
        self.constrained = constrained
        self.pre = ConstrainedConv2d(3, 3, 5, norm=norm) if constrained else nn.Identity()

        self.e1, self.e2, self.e3 = block(3, base), block(base, base*2), block(base*2, base*4)
        self.bott = block(base*4, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = block(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = block(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = block(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pre(x)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        # Skip connections: encoder features concatenated at matching resolution.
        # This is what recovers sharp boundaries; upsampling alone cannot.
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)                            # logits, one per pixel

    def constrain(self):
        if self.constrained:
            self.pre.constrain()

    @torch.no_grad()
    def pre_norm_sum(self):
        """See ConstrainedConv2d.pre_norm_sum. Call BEFORE constrain()."""
        return self.pre.pre_norm_sum() if self.constrained else float("nan")

    @torch.no_grad()
    def weight_norm(self):
        return sum(p.norm().item() ** 2 for p in self.parameters()) ** 0.5


def dice_bce(logits, target, eps=1.0):
    """Dice + BCE. Plain BCE is swamped by the authentic majority class; Dice
    optimises overlap directly and is insensitive to that imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - num / den).mean()
