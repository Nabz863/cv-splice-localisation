"""Rung 4: U-Net with a constrained first layer, 5-fold CV with early stopping.

Split protocol (matched to rungs 1-2 via reselect_mrf.py):
  test fold      = k
  validation     = (k+1) % 5   -- early stopping and threshold calibration
  training       = the other three folds
The test fold is never consulted before the final evaluation.
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

EPOCHS = int(os.environ.get("EPOCHS", 60))        # ceiling; early stopping decides
PATIENCE = int(os.environ.get("PATIENCE", 8))
BATCH = int(os.environ.get("BATCH", 8))
N_FOLDS = int(os.environ.get("N_FOLDS", 5))
FOLD_START = int(os.environ.get("FOLD_START", 0))
N_TRAIN = int(os.environ.get("N_TRAIN", 0))       # 0 = all; else data-size curve
WORKERS = int(os.environ.get("WORKERS", 2))
CLIP = float(os.environ.get("CLIP", 1.0))         # gradient-norm clip; 0 disables
USE_WANDB = bool(int(os.environ.get("WANDB", 1)))

if USE_WANDB:
    import wandb


@torch.no_grad()
def predict_full(model, path):
    """Full-resolution inference. Pad to a multiple of 8 so three pools and three
    upsamples land back on the original size exactly."""
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    return torch.sigmoid(model(x))[0, 0].float().cpu().numpy()[:h, :w]


def counts_at(model, subset, thresholds):
    c = np.zeros((len(thresholds), 3), np.int64)
    for r in subset:
        p = predict_full(model, r["image"])
        t = np.array(Image.open(r["mask"]).convert("L")) > 127
        for j, th in enumerate(thresholds):
            m = p >= th
            c[j] += ((m & t).sum(), (m & ~t).sum(), (~m & t).sum())
    return c


def run_fold(k, rows):
    set_seed(k)
    val_fold = (k + 1) % N_FOLDS
    tr = [r for r in rows if int(r["fold"]) not in (k, val_fold)]
    va = [r for r in rows if int(r["fold"]) == val_fold]
    te = [r for r in rows if int(r["fold"]) == k]
    if N_TRAIN:
        tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]

    if USE_WANDB:
        wandb.init(project="cv-splice-localisation", name=f"unet-fold{k}",
                   group=f"rung4{'-n' + str(N_TRAIN) if N_TRAIN else ''}",
                   config={"fold": k, "val_fold": val_fold, "epochs_max": EPOCHS,
                           "patience": PATIENCE, "batch": BATCH, "lr": 1e-3,
                           "grad_clip": CLIP, "n_train": N_TRAIN or len(tr),
                           "constrained": True, "patch": 256,
                           "crops_per_image": 8, "seed": k})

    dl = DataLoader(SpliceCrops(tr, train=True, seed=k), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,
                     shuffle=False, num_workers=WORKERS)

    model = UNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    best, best_state, bad, stopped_at, nan_epoch = float("inf"), None, 0, EPOCHS, None
    model.constrain()

    for ep in range(EPOCHS):
        model.train()
        tot, n, t0, skipped = 0.0, 0, time.time(), 0
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)

            # Skip a non-finite batch rather than letting it poison the weights.
            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            if CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            model.constrain()          # re-impose the constraint after every step
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        if n == 0:
            print(f"  fold {k} ep {ep+1}: every batch non-finite - aborting", flush=True)
            nan_epoch = ep + 1
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

        msg = (f"  fold {k} ep {ep+1}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f}  "
               f"({time.time()-t0:.0f}s)")
        if skipped:
            msg += f"  [skipped {skipped} non-finite batches]"
            if nan_epoch is None:
                nan_epoch = ep + 1
        print(msg, flush=True)

        if USE_WANDB:
            wandb.log({"epoch": ep + 1, "train/loss": tot / n, "val/loss": vl,
                       "skipped_batches": skipped})

        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                stopped_at = ep + 1
                print(f"  early stop at epoch {stopped_at} (best val {best:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)      # the best epoch, not the last
    model.eval()

    # Threshold chosen on the validation fold, then applied unchanged to test.
    THR = np.round(np.arange(0.05, 0.96, 0.05), 2)
    cal = list(np.random.default_rng(1).permutation(va))[:150]
    cc = counts_at(model, cal, THR)
    f1s = [2*a/(2*a+b+d) if (a+b+d) else 0.0 for a, b, d in cc]
    thr = float(THR[int(np.argmax(f1s))])

    ct = counts_at(model, te, [thr])
    tp, fp, fn = ct[0]
    f1, iou = 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)

    if USE_WANDB:
        wandb.log({"test/f1": f1, "test/iou": iou, "test/threshold": thr,
                   "stopped_at_epoch": stopped_at, "best_val_loss": best,
                   "first_nan_epoch": nan_epoch or -1})
        wandb.finish()

    os.makedirs("results", exist_ok=True)
    tag = f"_n{N_TRAIN}" if N_TRAIN else ""
    with open(f"results/rung4_fold{k}{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "thr", "f1", "iou", "stopped_at", "best_val", "first_nan_epoch"])
        w.writerow([k, thr, f"{f1:.6f}", f"{iou:.6f}", stopped_at,
                    f"{best:.6f}", nan_epoch or ""])
    return thr, f1, iou, stopped_at, nan_epoch


if __name__ == "__main__":
    rows = load_rows()
    print(f"device {DEV} | {len(rows)} pairs | epochs<={EPOCHS} patience={PATIENCE} "
          f"clip={CLIP} | n_train {N_TRAIN or 'all'}", flush=True)

    out = []
    for k in range(FOLD_START, N_FOLDS):
        thr, f1, iou, ep, nan_ep = run_fold(k, rows)
        note = f"  [first non-finite batch at ep {nan_ep}]" if nan_ep else ""
        print(f"fold {k}: thr {thr:.2f}  F1 {f1:.4f}  IoU {iou:.4f}  "
              f"(stopped ep {ep}){note}\n", flush=True)
        out.append((k, thr, f1, iou, ep, nan_ep or ""))

    f1s = [o[2] for o in out]; ious = [o[3] for o in out]
    print(f"mean F1  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"mean IoU {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
    n_nan = sum(1 for o in out if o[5])
    print(f"folds with any non-finite batch: {n_nan}/{len(out)}")

    tag = f"_n{N_TRAIN}" if N_TRAIN else ""
    with open(f"results/rung4_unet{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "thr", "f1", "iou", "stopped_at", "first_nan_epoch"])
        w.writerows(out)
    print(f"wrote results/rung4_unet{tag}.csv")
