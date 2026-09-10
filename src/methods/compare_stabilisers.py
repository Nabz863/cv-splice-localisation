"""Select a stabilisation method by controlled comparison, logged to W&B.

BACKGROUND. Three of five folds in the first full rung-4 run produced non-finite
losses from epoch 18-22. Because early stopping restores the best-validation
weights, the reported scores were unaffected (fold 1 diverged at epoch 22 and
still scored the highest F1 of any fold, 0.7304) -- but training halts early and
the runs are not clean.

MECHANISM UNDER TEST. constrain() divides the surround weights by their own
signed sum. That sum is unbounded below: if the weights drift toward cancelling
it approaches zero and the division amplifies without limit. pre_norm_sum()
measures it directly, before normalisation, so this experiment can confirm or
refute the diagnosis rather than assume it.

SIZING. Divergence appeared at ~24,000 optimiser steps in the full run (1093
training images, 8 crops each, batch 8, epoch 22). A shorter comparison will not
reproduce it: an earlier attempt at 12,000 steps saw no divergence in any
candidate, which was a flaw in the test, not evidence of stability. At
N_TRAIN=400 one epoch is 400 steps, so EPOCHS must be >= 80 to reach the same
regime. Budget roughly 33 minutes per candidate.
"""
import csv, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seed
from unet import UNet, dice_bce
from patch_data import SpliceCrops, load_rows

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

EPOCHS = int(os.environ.get("EPOCHS", 90))       # >= 80 to reach the divergence regime
N_TRAIN = int(os.environ.get("N_TRAIN", 400))
WORKERS = int(os.environ.get("WORKERS", 2))
BATCH = 8
USE_WANDB = bool(int(os.environ.get("WANDB", 1)))

if USE_WANDB:
    import wandb

# name, norm, weight_decay, clip
ALL = {
    "baseline":  ("sum", 0.0,  0.0),    # control: must reproduce the divergence
    "clip":      ("sum", 0.0,  1.0),    # bounds gradient norm, not the weight update
    "wd1e-3":    ("sum", 1e-3, 0.0),    # best-performing decay in the short run
    "l1norm":    ("l1",  0.0,  0.0),    # bounded by construction
}
ONLY = os.environ.get("ONLY", "")
CANDIDATES = ([(n, *ALL[n]) for n in ONLY.split(",")] if ONLY
              else [(n, *v) for n, v in ALL.items()])


def run(name, norm, wd, clip, tr, va):
    set_seed(0)                                   # identical init across candidates
    dl = DataLoader(SpliceCrops(tr, train=True, seed=0), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,
                     shuffle=False, num_workers=WORKERS)

    model = UNet(norm=norm).to(DEV)
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd) if wd
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    if USE_WANDB:
        wandb.init(project="cv-splice-localisation", name=f"stab-{name}",
                   group="stabiliser-comparison",
                   config={"candidate": name, "constraint_norm": norm,
                           "weight_decay": wd, "grad_clip": clip, "lr": 1e-3,
                           "n_train": len(tr), "batch": BATCH,
                           "epochs": EPOCHS, "steps_per_epoch": len(dl), "seed": 0})

    best, first_nan, min_s, survived, total_bad = float("inf"), None, float("inf"), 0, 0
    model.constrain()

    for ep in range(EPOCHS):
        model.train()
        tot, n, bad, ep_min_s = 0.0, 0, 0, float("inf")
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)
            if not torch.isfinite(loss):
                bad += 1
                continue
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update()
            # measured AFTER the step, BEFORE constrain() -- this is the
            # denominator constrain() is about to divide by.
            ep_min_s = min(ep_min_s, model.pre_norm_sum())
            model.constrain()
            tot += loss.item() * x.size(0); n += x.size(0)

        min_s = min(min_s, ep_min_s)
        total_bad += bad
        if bad and first_nan is None:
            first_nan = ep + 1
        if n == 0:
            print(f"  {name}: every batch non-finite at epoch {ep+1} - aborting", flush=True)
            break
        survived = ep + 1

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xv, yv in vdl:
                xv, yv = xv.to(DEV), yv.to(DEV)
                l = dice_bce(model(xv), yv)
                if torch.isfinite(l):
                    vl += l.item() * xv.size(0); vn += xv.size(0)
        vl = vl / vn if vn else float("inf")
        best = min(best, vl)

        if USE_WANDB:
            wandb.log({"epoch": ep + 1, "train/loss": tot / n, "val/loss": vl,
                       "diag/min_pre_norm_sum": ep_min_s,
                       "diag/weight_norm": model.weight_norm(),
                       "non_finite_batches": bad})

        if (ep + 1) % 5 == 0 or bad:
            print(f"  {name:<10} ep {ep+1:>3}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f}  "
                  f"min|s| {ep_min_s:.3g}" + (f"  [{bad} non-finite]" if bad else ""),
                  flush=True)

    if USE_WANDB:
        wandb.log({"summary/survived": survived, "summary/first_nan": first_nan or -1,
                   "summary/best_val": best, "summary/min_pre_norm_sum": min_s,
                   "summary/total_non_finite": total_bad})
        wandb.finish()
    return survived, first_nan, best, min_s, total_bad


if __name__ == "__main__":
    rows = load_rows()
    tr = [r for r in rows if int(r["fold"]) not in (0, 1)]
    tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]
    va = [r for r in rows if int(r["fold"]) == 1][:120]
    steps = (len(tr) * 8) // BATCH

    print(f"device {DEV} | {len(tr)} train, {len(va)} val | {EPOCHS} epochs "
          f"({steps} steps/epoch, {steps*EPOCHS:,} total) | "
          f"{len(CANDIDATES)} candidates\n", flush=True)

    results = []
    for name, norm, wd, clip in CANDIDATES:
        print(f"=== {name} (norm={norm}, wd={wd}, clip={clip}) ===", flush=True)
        t0 = time.time()
        s, fn, b, ms, tb = run(name, norm, wd, clip, tr, va)
        results.append((name, norm, wd, clip, s, fn or "", b, ms, tb))
        print(f"  -> survived {s}/{EPOCHS}, first non-finite {fn or 'none'}, "
              f"best val {b:.4f}, min|s| {ms:.3g}, {tb} bad batches "
              f"[{time.time()-t0:.0f}s]\n", flush=True)

    print(f"{'candidate':<12}{'epochs':>8}{'1st NaN':>9}{'best val':>10}"
          f"{'min|s|':>12}{'bad':>7}")
    for name, _, _, _, s, fn, b, ms, tb in results:
        print(f"{name:<12}{s:>8}{str(fn or '-'):>9}{b:>10.4f}{ms:>12.3g}{tb:>7}")

    os.makedirs("results", exist_ok=True)
    with open("results/stabiliser_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "norm", "weight_decay", "clip", "epochs_survived",
                    "first_nan_epoch", "best_val", "min_pre_norm_sum",
                    "total_non_finite_batches"])
        w.writerows(results)
    print("\nwrote results/stabiliser_comparison.csv")
