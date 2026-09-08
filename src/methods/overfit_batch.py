"""Book's ritual (sec 2.6.3): before a real run, overfit one fixed batch."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet, dice_bce
from patch_data import SpliceCrops, load_rows
from torch.utils.data import DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("GPU unavailable — refusing CPU fallback. Set ALLOW_CPU=1 to override.")
rows = load_rows()[:8]
x, y = next(iter(DataLoader(SpliceCrops(rows, train=False), batch_size=8)))
x, y = x.to(DEV), y.to(DEV)

model = UNet(constrained=bool(int(os.environ.get("CONSTRAINED", 1)))).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.constrain()
for step in range(int(os.environ.get("STEPS", 600))):
    opt.zero_grad(); loss = dice_bce(model(x), y)
    loss.backward(); opt.step(); model.constrain()
    if loss.item() < 0.05:
        print(f"reached loss {loss.item():.4f} at step {step}"); break
    if step % 100 == 0:
        print(f"  step {step}  loss {loss.item():.4f}")
else:
    print(f"FAILED to overfit — final loss {loss.item():.4f}. Fix before training.")
