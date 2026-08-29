"""Rung 2: nested (tau, beta) grid search per fold, ICM and graph cuts."""
import csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seed
from noise_variants import v_ela
from mrf import solve_icm, solve_graphcut, energy

set_seed(0)
ROOT = os.path.join(os.environ["DATASETS"], "casia2")
CACHE = os.path.join(ROOT, "ela_cache")
TAUS = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
GRID = [(t, b) for t in TAUS for b in BETAS]
SOLVERS = {"graphcut": solve_graphcut}   # ICM result recorded in results/rung2_mrf.csv

rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
os.makedirs(CACHE, exist_ok=True)


def score_for(r):
    """ELA map, cached -- it is deterministic and reused across the whole grid."""
    key = os.path.join(CACHE, os.path.basename(r["image"]).rsplit(".", 1)[0] + ".npy")
    if os.path.exists(key):
        return np.load(key)
    s = v_ela(np.array(Image.open(r["image"]).convert("RGB")))
    np.save(key, s.astype(np.float32))
    return s


def one(r):
    s = score_for(r)
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    out = np.zeros((len(SOLVERS), len(GRID), 3), np.int64)
    ener = np.zeros((len(SOLVERS), len(GRID)))
    for si, (_, solve) in enumerate(SOLVERS.items()):
        for gi, (tau, beta) in enumerate(GRID):
            x = solve(s, tau, beta)
            out[si, gi] = ((x & truth).sum(), (x & ~truth).sum(), (~x & truth).sum())
            ener[si, gi] = energy(s, x, tau, beta)
    return out, ener


if __name__ == "__main__":
    t0 = time.time()
    counts = np.zeros((len(rows), len(SOLVERS), len(GRID), 3), np.int64)
    energies = np.zeros((len(rows), len(SOLVERS), len(GRID)))
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        for i, (c, e) in enumerate(ex.map(one, rows, chunksize=4)):
            counts[i], energies[i] = c, e
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)")

    folds = np.array([int(r["fold"]) for r in rows])

    def f1_of(idx, si, gi):
        tp, fp, fn = counts[idx, si, gi].sum(0)
        return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0

    results = []
    for si, name in enumerate(SOLVERS):
        print(f"\n=== {name} ===")
        print(f"{'fold':<6}{'tau*':>7}{'beta*':>8}{'test F1':>10}{'test IoU':>10}")
        per_fold = []
        for k in range(5):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            gi = int(np.argmax([f1_of(tr, si, g) for g in range(len(GRID))]))
            tp, fp, fn = counts[te, si, gi].sum(0)
            f1, iou = 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)
            tau, beta = GRID[gi]
            per_fold.append((tau, beta, f1, iou))
            print(f"{k:<6}{tau:>7}{beta:>8}{f1:>10.4f}{iou:>10.4f}")
            results.append((name, k, tau, beta, f1, iou))
        f1s = [p[2] for p in per_fold]
        print(f"mean F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Energy comparison must hold (tau, beta) FIXED and use beta > 0:
    # at beta = 0 thresholding attains energy 0 exactly, so any solver ties.
    print("\nmean energy at fixed (tau, beta), lower = better optimisation")
    print(f"{'tau':>6}{'beta':>7}" + "".join(f"{n:>12}" for n in SOLVERS))
    for gi, (tau, beta) in enumerate(GRID):
        if beta == 0 or tau != 1.0:
            continue
        me = energies[:, :, gi].mean(0)
        print(f"{tau:>6}{beta:>7}" + "".join(f"{v:>12.1f}" for v in me))
    gi_pos = [g for g, (t, b) in enumerate(GRID) if b > 0]
    gaps = energies[:, 1, gi_pos] - energies[:, 0, gi_pos]   # graphcut - icm
    print(f"\ngraphcut energy <= icm energy in {(gaps <= 1e-6).mean():.1%} of cases")

    os.makedirs("results", exist_ok=True)
    with open("results/rung2_mrf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "fold", "tau", "beta", "f1", "iou"])
        w.writerows(results)
    print("\nwrote results/rung2_mrf.csv")
