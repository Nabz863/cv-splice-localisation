"""Rung 4: train per fold, evaluate at full resolution with the shared harness."""
import csv, os, sys, time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seed
from unet import UNet, dice_bce
from patch_data import SpliceCrops, load_rows

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", 12))
BATCH = int(os.environ.get("BATCH", 8))
N_FOLDS = int(os.environ.get("N_FOLDS", 5))
N_TRAIN = int(os.environ.get("N_TRAIN", 0))      # 0 = all; else data-size curve


@torch.no_grad()
def predict_full(model, path):
    """Full-resolution inference. Pad to a multiple of 8 so the three pools and
    three upsamples land back on the original size exactly."""
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()[:h, :w]


def run_fold(k, rows):
    set_seed(k)
    tr = [r for r in rows if int(r["fold"]) != k]
    te = [r for r in rows if int(r["fold"]) == k]
    if N_TRAIN:
        tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]

    dl = DataLoader(SpliceCrops(tr, train=True, seed=k), batch_size=BATCH,
                    shuffle=True, num_workers=4, drop_last=True)
    model = UNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    model.constrain()
    for ep in range(EPOCHS):
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.constrain()          # re-impose after every step
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        print(f"  fold {k} ep {ep+1}/{EPOCHS}  loss {tot/n:.4f}  ({time.time()-t0:.0f}s)")

    # threshold chosen on training folds, applied to the held-out fold
    model.eval()
    THR = np.arange(0.05, 0.96, 0.05)
    cal = list(np.random.default_rng(1).permutation(tr))[:150]
    c = np.zeros((len(THR), 3), np.int64)
    for r in cal:
        p = predict_full(model, r["image"])
        t = np.array(Image.open(r["mask"]).convert("L")) > 127
        for j, th in enumerate(THR):
            m = p >= th
            c[j] += ((m & t).sum(), (m & ~t).sum(), (~m & t).sum())
    f1s = [2*a / (2*a + b + d) if (a+b+d) else 0 for a, b, d in c]
    thr = THR[int(np.argmax(f1s))]

    tp = fp = fn = 0
    for r in te:
        p = predict_full(model, r["image"]) >= thr
        t = np.array(Image.open(r["mask"]).convert("L")) > 127
        tp += (p & t).sum(); fp += (p & ~t).sum(); fn += (~p & t).sum()
    return thr, 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)


if __name__ == "__main__":
    rows = load_rows()
    print(f"device {DEV} · {len(rows)} pairs · epochs {EPOCHS} · n_train {N_TRAIN or 'all'}")
    out = []
    for k in range(N_FOLDS):
        thr, f1, iou = run_fold(k, rows)
        print(f"fold {k}: thr {thr:.2f}  F1 {f1:.4f}  IoU {iou:.4f}\n")
        out.append((k, thr, f1, iou))

    f1s = [o[2] for o in out]
    print(f"mean F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    os.makedirs("results", exist_ok=True)
    tag = f"_n{N_TRAIN}" if N_TRAIN else ""
    with open(f"results/rung4_unet{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "thr", "f1", "iou"]); w.writerows(out)
