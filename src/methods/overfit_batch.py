"""The book's ritual (sec 2.6.3): overfit one fixed batch before any real run.

Per the book, augmentation and regularisation are disabled -- this is a capacity
and wiring check, not a generalisation test. train=False on the dataset turns off
flips and rotations; weight decay is off; the model is in train() mode so
BatchNorm uses batch statistics as it would during real training.

Reaching a low loss proves only that the data, model, loss and optimiser are
connected and that capacity is sufficient. It does not prove the model will
generalise.
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

STEPS = int(os.environ.get("STEPS", 2000))
CONSTRAINED = bool(int(os.environ.get("CONSTRAINED", 1)))
NORM = os.environ.get("NORM", "l1")
TARGET = float(os.environ.get("TARGET", 0.05))

rows = load_rows()[:8]
x, y = next(iter(DataLoader(SpliceCrops(rows, train=False), batch_size=8)))
x, y = x.to(DEV), y.to(DEV)

model = UNet(constrained=CONSTRAINED, norm=NORM).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)      # no weight decay: ritual

print(f"device {DEV} | constrained={CONSTRAINED} norm={NORM} | "
      f"target loss {TARGET} in {STEPS} steps")

model.train()                       # BatchNorm in training mode, as in a real run
model.constrain()
for step in range(STEPS):
    opt.zero_grad()
    loss = dice_bce(model(x), y)
    if not torch.isfinite(loss):
        print(f"NON-FINITE loss at step {step} — {model.diagnostics()}")
        break
    loss.backward()
    opt.step()
    model.constrain()

    if loss.item() < TARGET:
        print(f"PASS — reached {loss.item():.4f} at step {step}")
        break
    if step % 200 == 0:
        d = model.diagnostics()
        extra = "  ".join(f"{k} {v:.4g}" for k, v in d.items())
        print(f"  step {step:>5}  loss {loss.item():.4f}  {extra}")
else:
    print(f"FAILED — final loss {loss.item():.4f} after {STEPS} steps. "
          f"Investigate before training on the full dataset.")
