"""Wiring check before real training: the U-Net should be able to overfit one fixed batch.

No augmentation and no weight decay. Usage:  python src/methods/overfit_batch.py
"""
import os, sys                                    # paths and import path
import numpy as np                                # arrays and maths
import torch                                      # deep-learning library
from torch.utils.data import DataLoader           # feeds batches of crops to the model

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from unet import UNet, dice_bce                   # the model and its loss
from patch_data import SpliceCrops, load_rows     # crops and the image list

DEV = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):  # refuse to silently run on CPU
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

STEPS = int(os.environ.get("STEPS", 2000))               # max optimiser steps
CONSTRAINED = bool(int(os.environ.get("CONSTRAINED", 1)))   # use the Bayar first layer (1 = yes)
NORM = os.environ.get("NORM", "l1")                      # how the Bayar constraint normalises
TARGET = float(os.environ.get("TARGET", 0.05))           # loss that counts as "overfitted"

rows = load_rows()[:8]                                   # first 8 images
x, y = next(iter(DataLoader(SpliceCrops(rows, train=False), batch_size=8)))   # one fixed batch, no augmentation
x, y = x.to(DEV), y.to(DEV)                              # move it to the GPU

model = UNet(constrained=CONSTRAINED, norm=NORM).to(DEV) # fresh model
opt = torch.optim.Adam(model.parameters(), lr=1e-3)      # no weight decay

print(f"device {DEV} | constrained={CONSTRAINED} norm={NORM} | "
      f"target loss {TARGET} in {STEPS} steps")

# ---- Train on the same batch over and over; the loss should fall close to 0 ----
model.train()                       # BatchNorm in training mode, as in a real run
model.constrain()                   # apply the first-layer constraint before starting
for step in range(STEPS):
    opt.zero_grad()                                       # clear old gradients
    loss = dice_bce(model(x), y)                          # predict and score
    if not torch.isfinite(loss):                          # NaN or infinity: stop and report
        print(f"NON-FINITE loss at step {step} - weight norm {model.weight_norm():.4g}")
        break
    loss.backward()                                       # compute gradients
    opt.step()                                            # update the weights
    model.constrain()                                     # re-apply the first-layer constraint

    if loss.item() < TARGET:                              # low enough: the wiring works
        print(f"PASS — reached {loss.item():.4f} at step {step}")
        break
    if step % 200 == 0:                                   # progress every 200 steps
        print(f"  step {step:>5}  loss {loss.item():.4f}  weight norm {model.weight_norm():.4g}")
else:                                                     # runs only if the loop never hit "break"
    print(f"FAILED — final loss {loss.item():.4f} after {STEPS} steps. "
          f"Investigate before training on the full dataset.")
