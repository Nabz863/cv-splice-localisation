"""Rung 5, step 2 (CPU): choose (tau, beta) on each pair's validation logits, score on test.

Also records the best beta = 0 setting (= rung 4), so the prior's gain is measured
against its own control. tau is a logit threshold (0 = probability 0.5).
SUFFIX=_n50 etc. sweeps a data-size run. Writes results/rung5_pair_<tag>.csv.
"""
import csv, glob, os, sys, time                   # standard-library helpers
import multiprocessing as mp                      # for starting worker processes
from concurrent.futures import ProcessPoolExecutor   # runs a function on many images in parallel
import numpy as np                                # arrays and maths

sys.path.insert(0, os.path.dirname(__file__))     # so mrf.py in this folder can be imported
from mrf import solve_graphcut, energy            # the exact MRF solver and the energy function

# ---- Search grid ----
TAUS = [float(x) for x in os.environ.get("TAUS", "-2,-1,-0.5,0,0.5,1,2").split(",")]     # logit thresholds to try
BETAS = [float(x) for x in os.environ.get("BETAS", "0,4,16,32,64,128,256").split(",")]   # prior strengths to try
GRID = [(t, b) for t in TAUS for b in BETAS]      # every combination: 7 x 7 = 49 settings
WORKERS = int(os.environ.get("CUT_WORKERS", 10))  # CPU processes running graph cuts in parallel
SUFFIX = os.environ.get("SUFFIX", "")             # "" = full-data run, "_n50" etc. = data-size runs


def load_split(z, split):
    """(logits, ground-truth mask) for every image of one split ("cal" or "test") in a cache file."""
    n = int(z[f"n_{split}"])                      # how many images in this split
    out = []
    for i in range(n):
        shape = tuple(z[f"{split}_shape_{i}"])    # the mask's height and width
        gt = np.unpackbits(z[f"{split}_gt_{i}"])[:shape[0]*shape[1]].reshape(shape).astype(bool)  # masks stored 1 bit/pixel
        out.append((z[f"{split}_lg_{i}"].astype(np.float32), gt))   # (logit map, true mask)
    return out


def cuts_one(item):
    """tp/fp/fn and energy for one image at every grid point."""
    lg, gt = item                                 # the image's logits and true mask
    c = np.zeros((len(GRID), 3), np.int64)        # one row of (tp, fp, fn) per grid point
    e = np.zeros(len(GRID))                       # the MRF energy reached at each grid point
    for gi, (tau, beta) in enumerate(GRID):       # try every (tau, beta)
        x = solve_graphcut(lg, tau, beta)         # predicted mask (beta = 0 is plain thresholding)
        c[gi] = ((x & gt).sum(), (x & ~gt).sum(), (~x & gt).sum())   # true positives, false positives, false negatives
        e[gi] = energy(lg, x, tau, beta)
    return c, e


def counts_over(items, ex, label):
    """Counts pooled over all images of a split (images processed in parallel)."""
    t0 = time.time()
    c = np.zeros((len(GRID), 3), np.int64)        # running totals of (tp, fp, fn)
    e = np.zeros(len(GRID))                       # running total of energies
    for i, (ci, ei) in enumerate(ex.map(cuts_one, items, chunksize=2)):   # results arrive image by image
        c += ci; e += ei                          # add this image's counts to the totals ("pooling")
        if (i + 1) % 100 == 0:                    # progress every 100 images
            print(f"    {label} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    return c, e / max(len(items), 1)              # totals, and the mean energy


def f1_of(c, gi):
    """Pooled F1 at grid point gi."""
    tp, fp, fn = c[gi]
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0


def run_pair(path, ex):
    """Select on validation, evaluate on test, for one (test, validation) pair."""
    tag = os.path.basename(path).replace(".npz", "")   # e.g. "t0v1"
    out = f"results/rung5_pair_{tag}.csv"         # this pair's one-row result
    if os.path.exists(out):                       # finished earlier: skip
        print(f"{tag}: done already", flush=True)
        return
    t0 = time.time()
    with np.load(path) as z:                      # open the cached logits
        cal, te = load_split(z, "cal"), load_split(z, "test")   # validation images and test images

    # ---- Select on the validation images ----
    cc, _ = counts_over(cal, ex, f"{tag} cal")    # pooled counts at every grid point
    gi = int(np.argmax([f1_of(cc, g) for g in range(len(GRID))]))      # best (tau, beta)
    tau, beta = GRID[gi]
    zero = [g for g, (t, b) in enumerate(GRID) if b == 0.0]   # grid points with the prior switched off
    gi0 = zero[int(np.argmax([f1_of(cc, g) for g in zero]))]           # best with beta = 0 (rung 4)

    # ---- Evaluate both choices on the test images ----
    ct, _ = counts_over(te, ex, f"{tag} test")    # pooled test counts at every grid point
    tp, fp, fn = ct[gi]                           # test counts at the chosen (tau, beta)
    f1, iou = 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)    # rung 5's test F1 and IoU
    f1_0 = f1_of(ct, gi0)                         # rung 4's test F1

    # ---- Save one row for this pair ----
    os.makedirs("results", exist_ok=True)
    k, v = tag[1], tag[3]                       # tag "t0v1" -> test fold 0, validation fold 1
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_fold", "val_fold", "tau", "beta", "f1", "iou",
                    "tau_beta0", "f1_beta0", "prior_gain"])          # column names
        w.writerow([k, v, tau, beta, f"{f1:.6f}", f"{iou:.6f}",
                    GRID[gi0][0], f"{f1_0:.6f}", f"{f1-f1_0:.6f}"])   # prior_gain = rung 5 minus rung 4
    print(f"{tag}: tau {tau} beta {beta}  F1 {f1:.4f} (beta=0: {f1_0:.4f}, "
          f"gain {f1-f1_0:+.4f})  IoU {iou:.4f}  [{time.time()-t0:.0f}s]\n", flush=True)


if __name__ == "__main__":
    paths = sorted(glob.glob(f"results/logits/t?v?{SUFFIX}.npz"))   # every cached pair, e.g. t0v1.npz ... t4v3.npz
    if not paths:
        raise SystemExit("no cached logits - run cache_logits.py first")
    print(f"{len(paths)} cached pairs | {len(GRID)} (tau, beta) points | "
          f"{WORKERS} workers | CPU only", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=mp.get_context("spawn")) as ex:   # "spawn": fresh worker processes
        for p in paths:                           # one pair at a time; images within a pair run in parallel
            run_pair(p, ex)
    print(f"summarise: SUFFIX={SUFFIX} python3 src/methods/merge_refine.py", flush=True)
