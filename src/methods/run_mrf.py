"""Rung 2: TP/FP/FN of the graph-cut MRF for every image at every (tau, beta) -> results/rung2_grid.npz.

Usage:  python src/methods/run_mrf.py
"""
import csv, os, sys, time                         # standard-library helpers
from concurrent.futures import ProcessPoolExecutor   # runs a function on many images in parallel
import numpy as np                                # arrays and maths
from PIL import Image                             # reads image files

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so "utils" (in src/) can be imported
sys.path.insert(0, os.path.dirname(__file__))                       # so files in this folder can be imported
from utils.seed import set_seed                   # fixes random number generators
from noise_variants import v_ela                  # rung 1's score map
from mrf import solve_icm, solve_graphcut, energy # MRF solvers and the energy function

set_seed(0)                                       # same randomness every run
ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
CACHE = os.path.join(ROOT, "ela_cache")           # where ELA maps are saved so they're computed once
# ---- Search grid ----
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]  # thresholds to try (on the ELA z-score)
BETAS = [0.0, 0.5, 2.0, 4.0, 8.0, 16.0]           # prior strengths to try (0 = no prior = rung 1)
GRID = [(t, b) for t in TAUS for b in BETAS]      # every combination: 8 x 6 = 48 settings
SOLVERS = {"graphcut": solve_graphcut}   # ICM on the same grid: icm_vs_graphcut.py

rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))   # every splicing image
os.makedirs(CACHE, exist_ok=True)


def score_for(r):
    """ELA map for one image, cached on disk (it is reused at every grid point)."""
    key = os.path.join(CACHE, os.path.basename(r["image"]).rsplit(".", 1)[0] + ".npy")   # cache file name
    if os.path.exists(key):                       # computed before: just load it
        return np.load(key)
    s = v_ela(np.array(Image.open(r["image"]).convert("RGB")))   # compute ELA
    np.save(key, s.astype(np.float32))            # save for next time
    return s


def one(r):
    """Counts and energy for one image at every grid point."""
    s = score_for(r)                              # ELA score map
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127   # true mask (True = spliced)
    out = np.zeros((len(SOLVERS), len(GRID), 3), np.int64)       # (tp, fp, fn) per solver and grid point
    ener = np.zeros((len(SOLVERS), len(GRID)))                   # energy reached
    for si, (_, solve) in enumerate(SOLVERS.items()):
        for gi, (tau, beta) in enumerate(GRID):
            x = solve(s, tau, beta)                                       # predicted mask
            out[si, gi] = ((x & truth).sum(), (x & ~truth).sum(), (~x & truth).sum())   # tp, fp, fn
            ener[si, gi] = energy(s, x, tau, beta)
    return out, ener


if __name__ == "__main__":
    t0 = time.time()                                    # start a timer
    counts = np.zeros((len(rows), len(SOLVERS), len(GRID), 3), np.int64)   # all images' counts
    energies = np.zeros((len(rows), len(SOLVERS), len(GRID)))
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:   # up to 12 processes
        for i, (c, e) in enumerate(ex.map(one, rows, chunksize=4)):   # results arrive in manifest order
            counts[i], energies[i] = c, e
            if (i + 1) % 200 == 0:                # progress every 200 images
                print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)")

    # ---- Save per-image counts (build_tables.py does the 20-pair selection from these) ----
    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/rung2_grid.npz", counts=counts,  # saved compressed
                        energies=energies, grid=np.array(GRID),
                        solvers=list(SOLVERS))
    print("saved results/rung2_grid.npz")

    # ---- Quick printed summary: pick (tau, beta) on the other 4 folds, score the 5th ----
    folds = np.array([int(r["fold"]) for r in rows])   # each image's fold

    def f1_of(idx, si, gi):
        """Pooled F1 over images idx, for solver si at grid point gi."""
        tp, fp, fn = counts[idx, si, gi].sum(0)   # add up counts over those images
        return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0

    results = []                                        # one row per (solver, fold)
    for si, name in enumerate(SOLVERS):
        print(f"\n=== {name} ===")
        print(f"{'fold':<6}{'tau*':>7}{'beta*':>8}{'test F1':>10}{'test IoU':>10}")
        per_fold = []                                   # this solver's 5 folds
        for k in range(5):                        # each fold takes a turn as the test fold
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]    # other folds / this fold
            gi = int(np.argmax([f1_of(tr, si, g) for g in range(len(GRID))]))   # best setting on the others
            tp, fp, fn = counts[te, si, gi].sum(0)                       # applied to this fold
            f1, iou = 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)
            tau, beta = GRID[gi]                        # the chosen setting
            per_fold.append((tau, beta, f1, iou))       # keep it
            print(f"{k:<6}{tau:>7}{beta:>8}{f1:>10.4f}{iou:>10.4f}")
            results.append((name, k, tau, beta, f1, iou))  # and in the CSV list
        f1s = [p[2] for p in per_fold]                  # F1 of each fold
        print(f"mean F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # ---- Mean energy per solver at fixed (tau, beta); beta = 0 skipped (every solver ties) ----
    print("\nmean energy at fixed (tau, beta), lower = better optimisation")
    print(f"{'tau':>6}{'beta':>7}" + "".join(f"{n:>12}" for n in SOLVERS))
    for gi, (tau, beta) in enumerate(GRID):
        if beta == 0 or tau != 1.0:               # only show tau = 1.0 with the prior on
            continue
        me = energies[:, :, gi].mean(0)           # mean energy per solver
        print(f"{tau:>6}{beta:>7}" + "".join(f"{v:>12.1f}" for v in me))
    if len(SOLVERS) > 1:                          # only when two solvers were run together
        gi_pos = [g for g, (t, b) in enumerate(GRID) if b > 0]
        gaps = energies[:, 1, gi_pos] - energies[:, 0, gi_pos]  # how much higher ICM's energy is, per image
        print(f"\ngraphcut energy <= icm energy in {(gaps <= 1e-6).mean():.1%} of cases")

    # ---- Save the printed summary as a CSV ----
    os.makedirs("results", exist_ok=True)
    with open("results/rung2_mrf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "fold", "tau", "beta", "f1", "iou"])
        w.writerows(results)
    print("\nwrote results/rung2_mrf.csv")
