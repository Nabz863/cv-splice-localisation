"""Rung 2 with the greedy ICM solver instead of the graph cut, on the same grid -> results/rung2_grid_icm.npz.

Shows that an inexact solver would hide the prior's gain. Usage:  python src/methods/icm_vs_graphcut.py
"""
import csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import v_ela
from mrf import solve_icm, energy

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
CACHE = os.path.join(ROOT, "ela_cache")
# ---- Same grid as run_mrf.py (checked below) ----
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
BETAS = [0.0, 0.5, 2.0, 4.0, 8.0, 16.0]
GRID = [(t, b) for t in TAUS for b in BETAS]
ICM_ITERS = 8


def score_for(r):
    """ELA map for one image, from the same cache run_mrf.py uses."""
    key = os.path.join(CACHE, os.path.basename(r["image"]).rsplit(".", 1)[0] + ".npy")
    if os.path.exists(key):
        return np.load(key)
    s = v_ela(np.array(Image.open(r["image"]).convert("RGB")))
    os.makedirs(CACHE, exist_ok=True)
    np.save(key, s.astype(np.float32))
    return s


def one(r):
    """ICM counts and energy for one image at every grid point."""
    s = score_for(r)
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    out = np.zeros((len(GRID), 3), np.int64)
    ener = np.zeros(len(GRID))
    for gi, (tau, beta) in enumerate(GRID):
        x = solve_icm(s, tau, beta, iters=ICM_ITERS)
        out[gi] = ((x & truth).sum(), (x & ~truth).sum(), (~x & truth).sum())
        ener[gi] = energy(s, x, tau, beta)
    return out, ener


if __name__ == "__main__":
    ref = np.load("results/rung2_grid.npz", allow_pickle=True)
    if [tuple(map(float, g)) for g in ref["grid"]] != GRID:
        raise SystemExit("GRID differs from rung2_grid.npz - keep the two in step")

    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
    if len(rows) != ref["counts"].shape[0]:
        raise SystemExit(f"manifest has {len(rows)} rows, rung2_grid.npz {ref['counts'].shape[0]}")

    t0 = time.time()
    counts = np.zeros((len(rows), len(GRID), 3), np.int64)
    energies = np.zeros((len(rows), len(GRID)))
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        for i, (c, e) in enumerate(ex.map(one, rows, chunksize=4)):
            counts[i], energies[i] = c, e
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- Check: at beta = 0 both solvers just threshold, so their counts must match ----
    gc = ref["counts"][:, 0]
    b0 = [g for g, (_, b) in enumerate(GRID) if b == 0.0]
    if not np.array_equal(counts[:, b0], gc[:, b0]):
        raise SystemExit("ICM and graph cuts disagree at beta = 0, where both must threshold")

    np.savez_compressed("results/rung2_grid_icm.npz", counts=counts, energies=energies,
                        grid=np.array(GRID), iters=ICM_ITERS)
    print(f"wrote results/rung2_grid_icm.npz ({time.time()-t0:.0f}s)")
