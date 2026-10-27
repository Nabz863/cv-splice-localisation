"""Why did fp16 training blow up? Train 3-4 configurations from the same seed and compare.

Candidates: no stabiliser, gradient clipping, weight decay 1e-3, L1-normalised constraint.
Logs weight norm and the constraint's denominator per epoch (W&B) -> results/stabiliser_comparison.csv.
Needs >= 80 epochs at N_TRAIN=400 (~36,000 steps) to reach the step count where divergence appeared.
Usage:  ONLY=baseline,clip,wd1e-3 python src/methods/compare_stabilisers.py
"""
import csv, os, sys, time                         # standard-library helpers
import numpy as np                                # arrays and maths
import torch                                      # deep-learning library
from torch.utils.data import DataLoader           # feeds batches of crops to the model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so src/ packages can be imported
sys.path.insert(0, os.path.dirname(__file__))                       # so files in this folder can be imported
from utils.seed import set_seed                   # fixes random number generators
from unet import UNet, dice_bce                   # the model and its loss
from patch_data import SpliceCrops, load_rows     # training crops and the image list

DEV = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):  # refuse to silently train on CPU
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

# ---- Settings ----
EPOCHS = int(os.environ.get("EPOCHS", 90))       # >= 80 to reach the divergence regime
N_TRAIN = int(os.environ.get("N_TRAIN", 400))    # training images
WORKERS = int(os.environ.get("WORKERS", 2))      # data-loading processes
BATCH = 8                                        # crops per batch
USE_WANDB = bool(int(os.environ.get("WANDB", 1)))   # log to Weights & Biases (1 = yes)

if USE_WANDB:
    import wandb                                 # only imported when logging is on

# name, norm, weight_decay, clip
ALL = {
    "baseline":  ("sum", 0.0,  0.0),    # control: must reproduce the divergence
    "clip":      ("sum", 0.0,  1.0),    # bounds gradient norm, not the weight update
    "wd1e-3":    ("sum", 1e-3, 0.0),    # best-performing decay in the short run
    "l1norm":    ("l1",  0.0,  0.0),    # bounded by construction
}
ONLY = os.environ.get("ONLY", "")                # e.g. "baseline,clip" to run just those
CANDIDATES = ([(n, *ALL[n]) for n in ONLY.split(",")] if ONLY
              else [(n, *v) for n, v in ALL.items()])   # otherwise run all four


# ---- Train one candidate ----

def run(name, norm, wd, clip, tr, va):
    """Train one candidate; return epochs survived, first non-finite epoch, best val loss, min denominator, bad batches."""
    set_seed(0)                                   # identical init across candidates
    dl = DataLoader(SpliceCrops(tr, train=True, seed=0), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)   # training batches (random crops)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,
                     shuffle=False, num_workers=WORKERS)                 # validation batches (fixed crops)

    model = UNet(norm=norm).to(DEV)               # fresh model on the GPU
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd) if wd
           else torch.optim.Adam(model.parameters(), lr=1e-3))   # AdamW only when weight decay is used
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))  # rescales the loss so fp16 gradients don't underflow

    if USE_WANDB:                                 # start a W&B run recording this candidate's settings
        wandb.init(project="cv-splice-localisation", name=f"stab-{name}",
                   group="stabiliser-comparison",
                   config={"candidate": name, "constraint_norm": norm,
                           "weight_decay": wd, "grad_clip": clip, "lr": 1e-3,
                           "n_train": len(tr), "batch": BATCH,
                           "epochs": EPOCHS, "steps_per_epoch": len(dl), "seed": 0})

    best, first_nan, min_s, survived, total_bad = float("inf"), None, float("inf"), 0, 0   # running totals
    model.constrain()                             # apply the Bayar constraint before the first step

    for ep in range(EPOCHS):
        # ---- Training pass ----
        model.train()                             # training mode
        tot, n, bad, ep_min_s = 0.0, 0, 0, float("inf")   # loss sum, crops seen, bad batches, smallest denominator
        for x, y in dl:                           # x = image crops, y = mask crops
            x, y = x.to(DEV), y.to(DEV)           # move the batch to the GPU
            opt.zero_grad()                       # clear the previous step's gradients
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):   # fp16 autocast (the setting that failed)
                loss = dice_bce(model(x), y)      # predict and score
            if not torch.isfinite(loss):          # NaN or infinity: skip this batch and count it
                bad += 1
                continue
            scaler.scale(loss).backward()         # compute gradients (on the scaled loss)
            if clip > 0:                          # gradient clipping candidate
                scaler.unscale_(opt)              # undo the scaling so the clip limit means what it says
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # cap the total gradient size
            scaler.step(opt); scaler.update()     # update the weights; adjust the scale factor
            # the denominator constrain() is about to divide by (measured before constrain)
            ep_min_s = min(ep_min_s, model.pre_norm_sum())
            model.constrain()                     # re-apply the Bayar constraint after the step
            tot += loss.item() * x.size(0); n += x.size(0)   # running loss and crop count

        min_s = min(min_s, ep_min_s)              # smallest denominator over all epochs
        total_bad += bad
        if bad and first_nan is None:             # remember the first epoch with a bad batch
            first_nan = ep + 1
        if n == 0:                                # nothing trained this epoch: the run has failed
            print(f"  {name}: every batch non-finite at epoch {ep+1} - aborting", flush=True)
            break
        survived = ep + 1

        # ---- Validation pass ----
        model.eval()                              # evaluation mode
        vl, vn = 0.0, 0                           # validation loss sum and crop count
        with torch.no_grad():                     # no gradients needed
            for xv, yv in vdl:
                xv, yv = xv.to(DEV), yv.to(DEV)
                l = dice_bce(model(xv), yv)
                if torch.isfinite(l):             # ignore non-finite batches
                    vl += l.item() * xv.size(0); vn += xv.size(0)
        vl = vl / vn if vn else float("inf")      # mean validation loss
        best = min(best, vl)                      # best so far

        if USE_WANDB:                             # log this epoch
            wandb.log({"epoch": ep + 1, "train/loss": tot / n, "val/loss": vl,
                       "diag/min_pre_norm_sum": ep_min_s,
                       "diag/weight_norm": model.weight_norm(),
                       "non_finite_batches": bad})

        if (ep + 1) % 5 == 0 or bad:              # print every 5 epochs, or when something went wrong
            print(f"  {name:<10} ep {ep+1:>3}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f}  "
                  f"min|s| {ep_min_s:.3g}" + (f"  [{bad} non-finite]" if bad else ""),
                  flush=True)

    if USE_WANDB:                                 # final summary for this candidate
        wandb.log({"summary/survived": survived, "summary/first_nan": first_nan or -1,
                   "summary/best_val": best, "summary/min_pre_norm_sum": min_s,
                   "summary/total_non_finite": total_bad})
        wandb.finish()
    return survived, first_nan, best, min_s, total_bad


if __name__ == "__main__":
    # ---- Same data for every candidate: 400 training images, 120 validation images ----
    rows = load_rows()                            # every splicing image
    tr = [r for r in rows if int(r["fold"]) not in (0, 1)]   # training pool: folds 2-4
    tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]   # seeded random subset
    va = [r for r in rows if int(r["fold"]) == 1][:120]      # validation: first 120 of fold 1
    steps = (len(tr) * 8) // BATCH                # 8 crops per image per epoch

    print(f"device {DEV} | {len(tr)} train, {len(va)} val | {EPOCHS} epochs "
          f"({steps} steps/epoch, {steps*EPOCHS:,} total) | "
          f"{len(CANDIDATES)} candidates\n", flush=True)

    # ---- Train each candidate in turn ----
    results = []
    for name, norm, wd, clip in CANDIDATES:
        print(f"=== {name} (norm={norm}, wd={wd}, clip={clip}) ===", flush=True)
        t0 = time.time()
        s, fn, b, ms, tb = run(name, norm, wd, clip, tr, va)
        results.append((name, norm, wd, clip, s, fn or "", b, ms, tb))
        print(f"  -> survived {s}/{EPOCHS}, first non-finite {fn or 'none'}, "
              f"best val {b:.4f}, min|s| {ms:.3g}, {tb} bad batches "
              f"[{time.time()-t0:.0f}s]\n", flush=True)

    # ---- Summary table ----
    print(f"{'candidate':<12}{'epochs':>8}{'1st NaN':>9}{'best val':>10}"
          f"{'min|s|':>12}{'bad':>7}")
    for name, _, _, _, s, fn, b, ms, tb in results:
        print(f"{name:<12}{s:>8}{str(fn or '-'):>9}{b:>10.4f}{ms:>12.3g}{tb:>7}")

    # ---- Save ----
    os.makedirs("results", exist_ok=True)
    with open("results/stabiliser_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "norm", "weight_decay", "clip", "epochs_survived",
                    "first_nan_epoch", "best_val", "min_pre_norm_sum",
                    "total_non_finite_batches"])
        w.writerows(results)
    print("\nwrote results/stabiliser_comparison.csv")
