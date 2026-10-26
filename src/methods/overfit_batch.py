"""Wiring check before real training: the U-Net should be able to overfit one fixed batch.

No augmentation and no weight decay. Usage:  python src/methods/overfit_batch.py
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet, dice_bce
from patch_data import SpliceCrops, load_rows

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

STEPS = int(os.environ.get("STEPS", 2000))               # max optimiser steps
CONSTRAINED = bool(int(os.environ.get("CONSTRAINED", 1)))
NORM = os.environ.get("NORM", "l1")
TARGET = float(os.environ.get("TARGET", 0.05))           # loss that counts as "overfitted"

rows = load_rows()[:8]
x, y = next(iter(DataLoader(SpliceCrops(rows, train=False), batch_size=8)))   # one fixed batch, no augmentation
x, y = x.to(DEV), y.to(DEV)

model = UNet(constrained=CONSTRAINED, norm=NORM).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)      # no weight decay

print(f"device {DEV} | constrained={CONSTRAINED} norm={NORM} | "
      f"target loss {TARGET} in {STEPS} steps")

model.train()                       # BatchNorm in training mode, as in a real run
model.constrain()
for step in range(STEPS):
    opt.zero_grad()
    loss = dice_bce(model(x), y)
    if not torch.isfinite(loss):
        print(f"NON-FINITE loss at step {step} - weight norm {model.weight_norm():.4g}")
        break
    loss.backward()
    opt.step()
    model.constrain()                                     # re-apply the first-layer constraint

    if loss.item() < TARGET:
        print(f"PASS — reached {loss.item():.4f} at step {step}")
        break
    if step % 200 == 0:
        print(f"  step {step:>5}  loss {loss.item():.4f}  weight norm {model.weight_norm():.4g}")
else:
    print(f"FAILED — final loss {loss.item():.4f} after {STEPS} steps. "
          f"Investigate before training on the full dataset.")
