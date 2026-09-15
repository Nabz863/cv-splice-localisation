"""Rung 5, phase 1: train (or load) each pair's U-Net and cache its logits.

GPU ONLY. No multiprocessing anywhere in this file -- a CUDA context and a
process pool in the same interpreter produced repeated illegal-memory-access
faults, and separating the phases removes the interaction entirely rather than
trying to work around it.

Writes results/logits/{tag}.npz holding the validation-calibration and test
logits for one (test, val) pair, as float16 to keep the files small. Phase 2
(sweep_cached.py) consumes these and touches no GPU.
"""
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
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

EPOCHS = int(os.environ.get("EPOCHS", 90))
PATIENCE = int(os.environ.get("PATIENCE", 10))
BATCH = int(os.environ.get("BATCH", 8))
N_FOLDS = int(os.environ.get("N_FOLDS", 5))
WORKERS = int(os.environ.get("WORKERS", 2))
NORM = os.environ.get("NORM", "sum")
WD = float(os.environ.get("WD", 1e-4))
CLIP = float(os.environ.get("CLIP", 1.0))
N_CAL = int(os.environ.get("N_CAL", 150))
NESTED = bool(int(os.environ.get("NESTED", 1)))
AMP_DTYPE = torch.bfloat16


@torch.no_grad()
def logits_full(model, path):
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    return model(x)[0, 0].float().cpu().numpy()[:h, :w]


def train_one(test_fold, val_fold, rows, tag):
    set_seed(test_fold * 10 + val_fold)
    tr = [r for r in rows if int(r["fold"]) not in (test_fold, val_fold)]
    va = [r for r in rows if int(r["fold"]) == val_fold]

    os.makedirs("results/ckpt_refine", exist_ok=True)
    ck = f"results/ckpt_refine/{tag}.pt"

    dl = DataLoader(SpliceCrops(tr, train=True, seed=test_fold), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,
                     shuffle=False, num_workers=WORKERS)

    model = UNet(norm=NORM).to(DEV)
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD) if WD
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    best, best_state, bad, start_ep = float("inf"), None, 0, 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        best, best_state, bad, start_ep = st["best"], st["best_state"], st["bad"], st["epoch"]
        print(f"  resumed {tag} from epoch {start_ep}", flush=True)

    model.constrain()
    for ep in range(start_ep, EPOCHS):
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(DEV, dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            if CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt); scaler.update()
            model.constrain()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        if n == 0:
            break

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xv, yv in vdl:
                xv, yv = xv.to(DEV), yv.to(DEV)
                l = dice_bce(model(xv), yv)
                if torch.isfinite(l):
                    vl += l.item() * xv.size(0); vn += xv.size(0)
        vl = vl / vn if vn else float("inf")
        print(f"  {tag} ep {ep+1}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {ep+1} (best val {best:.4f})", flush=True)
                break

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_state": best_state, "bad": bad,
                    "epoch": ep + 1}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if os.path.exists(ck):
        os.remove(ck)

    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), f"results/models/refine_{tag}.pt")
    return model


def cache_pair(test_fold, val_fold, rows):
    tag = f"t{test_fold}v{val_fold}"
    out = f"results/logits/{tag}.npz"
    if os.path.exists(out):
        print(f"{tag}: cached already", flush=True)
        return

    saved = f"results/models/refine_{tag}.pt"
    if os.path.exists(saved):
        model = UNet(norm=NORM).to(DEV)
        model.load_state_dict(torch.load(saved, map_location=DEV))
        model.eval()
        print(f"{tag}: loaded {saved}", flush=True)
    else:
        print(f"{tag}: training", flush=True)
        model = train_one(test_fold, val_fold, rows, tag)

    va = [r for r in rows if int(r["fold"]) == val_fold]
    te = [r for r in rows if int(r["fold"]) == test_fold]
    cal = list(np.random.default_rng(1).permutation(va))[:N_CAL]

    t0 = time.time()
    blob = {}
    for split, subset in (("cal", cal), ("test", te)):
        for i, r in enumerate(subset):
            lg = logits_full(model, r["image"]).astype(np.float16)
            m = (np.array(Image.open(r["mask"]).convert("L")) > 127)
            blob[f"{split}_lg_{i}"] = lg
            blob[f"{split}_gt_{i}"] = np.packbits(m)      # 1 bit/pixel
            blob[f"{split}_shape_{i}"] = np.array(m.shape, np.int32)
        blob[f"n_{split}"] = np.array(len(subset), np.int32)

    os.makedirs("results/logits", exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", **blob)
    os.replace(out + ".tmp.npz", out)
    mb = os.path.getsize(out) / 1e6
    print(f"{tag}: cached {len(cal)} cal + {len(te)} test "
          f"({mb:.0f} MB, {time.time()-t0:.0f}s)\n", flush=True)


if __name__ == "__main__":
    rows = load_rows()
    pairs = ([(k, v) for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
             if NESTED else [(k, (k + 1) % N_FOLDS) for k in range(N_FOLDS)])
    print(f"device {DEV} | {len(pairs)} pairs | no multiprocessing in this phase",
          flush=True)
    for k, v in pairs:
        cache_pair(k, v, rows)
    print("phase 1 done. now run: python3 src/methods/sweep_cached.py", flush=True)
