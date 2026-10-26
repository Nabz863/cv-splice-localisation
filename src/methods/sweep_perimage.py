"""Rung 5: keep every image's tp/fp/fn at every (tau, beta), so build_tables.py can select by per-image F1.

Same grid and solver as sweep_cached.py. Writes results/rung5_counts/<tag>.npz.
Usage:  CUT_WORKERS=12 python src/methods/sweep_perimage.py
"""
import glob, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mrf import solve_graphcut
from sweep_cached import GRID, WORKERS, load_split

OUT = "results/rung5_counts"


def counts_one(item):
    """tp/fp/fn for one image at every grid point."""
    lg, gt = item
    c = np.zeros((len(GRID), 3), np.int64)
    for gi, (tau, beta) in enumerate(GRID):
        x = solve_graphcut(lg, tau, beta)
        c[gi] = ((x & gt).sum(), (x & ~gt).sum(), (~x & gt).sum())
    return c


def per_image(items, ex, label):
    """Stack of per-image counts, shape (images, grid, 3)."""
    t0, out = time.time(), []
    for i, c in enumerate(ex.map(counts_one, items, chunksize=2)):
        out.append(c)
        if (i + 1) % 100 == 0:
            print(f"    {label} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(out)


if __name__ == "__main__":
    paths = sorted(glob.glob("results/logits/t?v?.npz"))      # full-data pairs only
    if not paths:
        raise SystemExit("no cached logits - run cache_logits.py first")
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(paths)} cached pairs | {len(GRID)} (tau, beta) points | "
          f"{WORKERS} workers | CPU only", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=mp.get_context("spawn")) as ex:
        for p in paths:
            tag = os.path.basename(p).replace(".npz", "")
            out = os.path.join(OUT, f"{tag}.npz")
            if os.path.exists(out):                                    # resumable: skip finished pairs
                print(f"{tag}: done already", flush=True)
                continue
            t0 = time.time()
            with np.load(p) as z:
                cal, te = load_split(z, "cal"), load_split(z, "test")
            c_cal = per_image(cal, ex, f"{tag} cal")
            c_te = per_image(te, ex, f"{tag} test")
            np.savez_compressed(out + ".tmp.npz", cal=c_cal, test=c_te, grid=np.array(GRID))
            os.replace(out + ".tmp.npz", out)                          # write-then-rename: no half files
            print(f"{tag}: {len(cal)} cal + {len(te)} test images [{time.time()-t0:.0f}s]",
                  flush=True)
    print("next: python3 src/report/build_tables.py", flush=True)
