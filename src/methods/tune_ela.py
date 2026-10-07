"""Rung 1, step 2: tune ELA's two knobs.

Re-compression quality sets which DCT coefficients get requantised; the window
sets the spatial scale of the evidence. Sweeps both by mean pixel AUC on 300
images from folds 1-4 (fold 0 untouched), fixed seed.

CPU, a few minutes. Writes results/ela_tuning.csv, which build_tables.py reads.

Usage (from the repo root):
    python src/methods/tune_ela.py
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from noise_variants import v_ela
from eval.metrics import pixel_auc

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
GRID = [(q, w) for q in (70, 80, 90, 95, 98) for w in (8, 16, 32)]


def sample_rows():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
    rows = [r for r in rows if int(r["fold"]) != 0]
    return list(np.random.default_rng(0).permutation(rows))[:300]


def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    return [pixel_auc(v_ela(img, quality=q, w=w), truth) for q, w in GRID]


if __name__ == "__main__":
    rows = sample_rows()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        out = list(ex.map(one, rows, chunksize=4))

    print(f"{'quality':>8}{'window':>8}{'mean AUC':>11}{'median':>9}")
    best = None
    os.makedirs("results", exist_ok=True)
    with open("results/ela_tuning.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["quality", "window", "n_images", "mean_auc", "median_auc"])
        for i, (q, w) in enumerate(GRID):
            a = np.array([r[i] for r in out if r[i] is not None])
            print(f"{q:>8}{w:>8}{a.mean():>11.4f}{np.median(a):>9.4f}")
            w_.writerow([q, w, len(a), f"{a.mean():.6f}", f"{np.median(a):.6f}"])
            if best is None or a.mean() > best[0]:
                best = (a.mean(), q, w)
    print(f"\nbest: quality={best[1]}, window={best[2]}  (AUC {best[0]:.4f})")
    print("wrote results/ela_tuning.csv")
