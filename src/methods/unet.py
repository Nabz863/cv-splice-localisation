"""Rung 4: U-Net with a constrained first layer.

The constraint (Bayar & Stamm) forces conv1 to compute a residual: centre weight
fixed at -1, surrounding weights normalised to sum to +1. Whatever it learns, its
output is always "pixel minus a weighted average of its neighbours", so the network
is structurally pushed toward noise/compression statistics rather than semantic
content -- the right prior for a forensic detector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_SUM = 1e-3      # floor on |surround sum| before normalising; see constrain()


class ConstrainedConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.k = k

    def constrain(self):
        """Re-impose the constraint. Called after every optimiser step.

        The surround is divided by its own sum, so if that sum drifts toward zero
        the weights explode -- this produced NaN losses at epochs 18-20 in two
        folds before the floor below was added. Clamping |sum| away from zero
        bounds the amplification at 1/EPS_SUM.
        """
        with torch.no_grad():
            w = self.weight.data
            c = self.k // 2
            w[:, :, c, c] = 0.0
            s = w.sum(dim=(2, 3), keepdim=True)
            sign = torch.where(s >= 0, 1.0, -1.0)
            s = sign * torch.clamp(s.abs(), min=EPS_SUM)
            w /= s
            w[:, :, c, c] = -1.0


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, constrained=True):
        super().__init__()
        self.constrained = constrained
        self.pre = ConstrainedConv2d(3, 3, 5) if constrained else nn.Identity()

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
        # Skip connections: concatenate encoder features at matching resolution.
        # This is what recovers sharp boundaries -- upsampling alone cannot.
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)                            # logits, one per pixel

    def constrain(self):
        if self.constrained:
            self.pre.constrain()


def dice_bce(logits, target, eps=1.0):
    """Dice + BCE. Plain BCE is swamped by the authentic majority class; Dice
    optimises overlap directly and is insensitive to that imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - num / den).mean()
