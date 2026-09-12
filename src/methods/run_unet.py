"""Rung 4: U-Net with a constrained first layer, cross-validated.

Split protocol. Each run trains on three folds, early-stops and calibrates its
threshold on a fourth (the validation fold), and reports on the fifth (the test
fold), which is never consulted before the final evaluation.

Two modes:
  default        val_fold = (k+1) % 5, five runs, one per test fold.
  nested (all)   every (test, val) pair, twenty runs.

The default pairing is a systematic rule rather than a random one, so any
correlation between adjacent fold indices propagates identically through every
fold. Enumerating all twenty pairs for the MRF (nested_pairs.py) showed this
biases the mean upward by 0.0055 F1 -- the (k+1) pairing scored above its fold's
all-pairs mean in five of five folds. The nested mode removes that.

Numerics: bfloat16 autocast, not fp16. bf16 carries fp32's exponent range, so
the 65,504 overflow that produced non-finite losses in earlier runs cannot occur.
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
N_TRAIN = int(os.environ.get("N_TRAIN", 0))       # 0 = all; else data-size curve
WORKERS = int(os.environ.get("WORKERS", 2))
NORM = os.environ.get("NORM", "sum")
WD = float(os.environ.get("WD", 0.0))
CLIP = float(os.environ.get("CLIP", 1.0))         # selected by compare_stabilisers.py
NESTED = bool(int(os.environ.get("NESTED", 0)))   # 1 = all 20 (test, val) pairs
USE_WANDB = bool(int(os.environ.get("WANDB", 1)))
AMP_DTYPE = torch.bfloat16

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


def run_pair(test_fold, val_fold, rows):
    """Train on every fold except test_fold and val_fold; stop and calibrate on
    val_fold; evaluate once on test_fold."""
    assert test_fold != val_fold, "test and validation folds must differ"
    set_seed(test_fold * 10 + val_fold)          # distinct but reproducible per pair

    tr = [r for r in rows if int(r["fold"]) not in (test_fold, val_fold)]
    va = [r for r in rows if int(r["fold"]) == val_fold]
    te = [r for r in rows if int(r["fold"]) == test_fold]
    if N_TRAIN:
        tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]

    tag = f"t{test_fold}v{val_fold}"
    if USE_WANDB:
        wandb.init(project="cv-splice-localisation", name=f"unet-{tag}",
                   group=f"rung4{'-nested' if NESTED else ''}"
                         f"{'-n' + str(N_TRAIN) if N_TRAIN else ''}",
                   config={"test_fold": test_fold, "val_fold": val_fold,
                           "epochs_max": EPOCHS, "patience": PATIENCE,
                           "batch": BATCH, "lr": 1e-3, "constraint_norm": NORM,
                           "weight_decay": WD, "grad_clip": CLIP,
                           "amp_dtype": "bfloat16", "n_train": N_TRAIN or len(tr),
                           "n_val": len(va), "n_test": len(te),
                           "constrained": True, "patch": 256, "crops_per_image": 8})

    dl = DataLoader(SpliceCrops(tr, train=True, seed=test_fold), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,
                     shuffle=False, num_workers=WORKERS)

    model = UNet(norm=NORM).to(DEV)
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD) if WD
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    best, best_state, bad, stopped_at, nan_epoch = float("inf"), None, 0, EPOCHS, None
    model.constrain()

    # Mid-pair checkpointing. Power here is unreliable and a pair takes over an
    # hour, so without this an outage costs the entire pair. With it, one epoch.
    os.makedirs("results/ckpt", exist_ok=True)
    ck = f"results/ckpt/{tag}.pt"
    start_ep = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        best, best_state, bad = st["best"], st["best_state"], st["bad"]
        nan_epoch, start_ep = st["nan_epoch"], st["epoch"]
        print(f"  resumed {tag} from epoch {start_ep} (best val {best:.4f})", flush=True)

    for ep in range(start_ep, EPOCHS):
        model.train()
        tot, n, t0, skipped = 0.0, 0, time.time(), 0
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(DEV, dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)
            if not torch.isfinite(loss):
                skipped += 1
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
            print(f"  {tag} ep {ep+1}: every batch non-finite - aborting", flush=True)
            nan_epoch = nan_epoch or ep + 1
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

        msg = (f"  {tag} ep {ep+1}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f}  "
               f"({time.time()-t0:.0f}s)")
        if skipped:
            msg += f"  [skipped {skipped} non-finite]"
            nan_epoch = nan_epoch or ep + 1
        print(msg, flush=True)

        if USE_WANDB:
            wandb.log({"epoch": ep + 1, "train/loss": tot / n, "val/loss": vl,
                       "skipped_batches": skipped,
                       "diag/weight_norm": model.weight_norm(),
                       "diag/pre_norm_sum": model.pre_norm_sum()})

        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1


            if bad >= PATIENCE:
                stopped_at = ep + 1
                print(f"  early stop at epoch {stopped_at} (best val {best:.4f})", flush=True)
                break

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_state": best_state, "bad": bad,
                    "nan_epoch": nan_epoch, "epoch": ep + 1}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)      # atomic: a cut mid-write cannot corrupt it

    if best_state is not None:
        model.load_state_dict(best_state)      # the best epoch, not the last
    model.eval()
    if os.path.exists(ck):
        os.remove(ck)                          # pair done; checkpoint no longer needed

    THR = np.round(np.arange(0.05, 0.96, 0.05), 2)
    cal = list(np.random.default_rng(1).permutation(va))[:150]
    cc = counts_at(model, cal, THR)
    f1s = [2*a/(2*a+b+d) if (a+b+d) else 0.0 for a, b, d in cc]
    thr = float(THR[int(np.argmax(f1s))])

    tp, fp, fn = counts_at(model, te, [thr])[0]
    f1, iou = 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)

    if USE_WANDB:
        wandb.log({"test/f1": f1, "test/iou": iou, "test/threshold": thr,
                   "stopped_at_epoch": stopped_at, "best_val_loss": best,
                   "first_nan_epoch": nan_epoch or -1})
        wandb.finish()

    os.makedirs("results", exist_ok=True)
    suffix = f"_n{N_TRAIN}" if N_TRAIN else ""
    with open(f"results/rung4_pair_{tag}{suffix}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_fold", "val_fold", "thr", "f1", "iou", "stopped_at",
                    "best_val", "first_nan_epoch", "norm", "weight_decay", "clip"])
        w.writerow([test_fold, val_fold, thr, f"{f1:.6f}", f"{iou:.6f}", stopped_at,
                    f"{best:.6f}", nan_epoch or "", NORM, WD, CLIP])
    return thr, f1, iou, stopped_at, nan_epoch


def pairs_to_run():
    if NESTED:
        return [(k, v) for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
    return [(k, (k + 1) % N_FOLDS) for k in range(N_FOLDS)]


if __name__ == "__main__":
    rows = load_rows()
    todo = pairs_to_run()
    suffix = f"_n{N_TRAIN}" if N_TRAIN else ""

    # resume: skip pairs already written
    pending = [(k, v) for k, v in todo
               if not os.path.exists(f"results/rung4_pair_t{k}v{v}{suffix}.csv")]

    print(f"device {DEV} | {len(rows)} pairs | epochs<={EPOCHS} patience={PATIENCE} "
          f"| norm={NORM} wd={WD} clip={CLIP} amp=bf16 | n_train {N_TRAIN or 'all'}",
          flush=True)
    print(f"{len(todo)} (test, val) pairs | {len(todo)-len(pending)} already done | "
          f"{len(pending)} to run\n", flush=True)

    for k, v in pending:
        t0 = time.time()
        thr, f1, iou, ep, nan_ep = run_pair(k, v, rows)
        note = f"  [first non-finite at ep {nan_ep}]" if nan_ep else ""
        print(f"t{k}v{v}: thr {thr:.2f}  F1 {f1:.4f}  IoU {iou:.4f}  "
              f"(stopped ep {ep}, {time.time()-t0:.0f}s){note}\n", flush=True)

    print("run: python3 src/methods/merge_nested.py", flush=True)
