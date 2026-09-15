"""Rung 5, phase 2: the (tau, beta) sweep over cached logits.

CPU ONLY. Imports no torch and touches no GPU, so it cannot hit the CUDA faults
that dogged the combined script, and it runs unchanged on the cluster.

For each pair: grid-search (tau, beta) on the validation-calibration logits,
apply the winner to the test logits, and separately find the best beta = 0
configuration the same way. beta = 0 reduces exactly to thresholding the U-Net's
logits, so the difference isolates the spatial prior.

tau is in LOGIT space (0 = p 0.5), unlike rung 2's z-scored ELA map.
"""
import csv, glob, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mrf import solve_graphcut, energy

TAUS = [float(x) for x in os.environ.get("TAUS", "-2,-1,-0.5,0,0.5,1,2").split(",")]
BETAS = [float(x) for x in os.environ.get("BETAS", "0,4,16,32,64,128,256").split(",")]
GRID = [(t, b) for t in TAUS for b in BETAS]
WORKERS = int(os.environ.get("CUT_WORKERS", 10))


def load_split(z, split):
    n = int(z[f"n_{split}"])
    out = []
    for i in range(n):
        shape = tuple(z[f"{split}_shape_{i}"])
        gt = np.unpackbits(z[f"{split}_gt_{i}"])[:shape[0]*shape[1]].reshape(shape).astype(bool)
        out.append((z[f"{split}_lg_{i}"].astype(np.float32), gt))
    return out


def cuts_one(item):
    lg, gt = item
    c = np.zeros((len(GRID), 3), np.int64)
    e = np.zeros(len(GRID))
    for gi, (tau, beta) in enumerate(GRID):
        x = solve_graphcut(lg, tau, beta)
        c[gi] = ((x & gt).sum(), (x & ~gt).sum(), (~x & gt).sum())
        e[gi] = energy(lg, x, tau, beta)
    return c, e


def counts_over(items, ex, label):
    t0 = time.time()
    c = np.zeros((len(GRID), 3), np.int64)
    e = np.zeros(len(GRID))
    for i, (ci, ei) in enumerate(ex.map(cuts_one, items, chunksize=2)):
        c += ci; e += ei
        if (i + 1) % 100 == 0:
            print(f"    {label} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    return c, e / max(len(items), 1)


def f1_of(c, gi):
    tp, fp, fn = c[gi]
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0


def run_pair(path, ex):
    tag = os.path.basename(path).replace(".npz", "")
    out = f"results/rung5_pair_{tag}.csv"
    if os.path.exists(out):
        print(f"{tag}: done already", flush=True)
        return
    t0 = time.time()
    with np.load(path) as z:
        cal, te = load_split(z, "cal"), load_split(z, "test")

    cc, _ = counts_over(cal, ex, f"{tag} cal")
    gi = int(np.argmax([f1_of(cc, g) for g in range(len(GRID))]))
    tau, beta = GRID[gi]
    zero = [g for g, (t, b) in enumerate(GRID) if b == 0.0]
    gi0 = zero[int(np.argmax([f1_of(cc, g) for g in zero]))]

    ct, _ = counts_over(te, ex, f"{tag} test")
    tp, fp, fn = ct[gi]
    f1, iou = 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)
    f1_0 = f1_of(ct, gi0)

    os.makedirs("results", exist_ok=True)
    k, v = tag[1:].split("v")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_fold", "val_fold", "tau", "beta", "f1", "iou",
                    "tau_beta0", "f1_beta0", "prior_gain"])
        w.writerow([k, v, tau, beta, f"{f1:.6f}", f"{iou:.6f}",
                    GRID[gi0][0], f"{f1_0:.6f}", f"{f1-f1_0:.6f}"])
    print(f"{tag}: tau {tau} beta {beta}  F1 {f1:.4f} (beta=0: {f1_0:.4f}, "
          f"gain {f1-f1_0:+.4f})  IoU {iou:.4f}  [{time.time()-t0:.0f}s]\n", flush=True)


if __name__ == "__main__":
    paths = sorted(glob.glob("results/logits/t*v*.npz"))
    if not paths:
        raise SystemExit("no cached logits - run cache_logits.py first")
    print(f"{len(paths)} cached pairs | {len(GRID)} (tau, beta) points | "
          f"{WORKERS} workers | CPU only", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=mp.get_context("spawn")) as ex:
        for p in paths:
            run_pair(p, ex)
    print("summarise: python3 src/methods/merge_refine.py", flush=True)
