"""ELA has two knobs: re-compression quality and smoothing window. Both are
physically meaningful -- quality sets which DCT coefficients get requantised,
window sets the spatial scale of the evidence. Sweep both."""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import v_ela
from compare_variants import pixel_auc

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
rows = list(np.random.default_rng(0).permutation(rows))[:300]
GRID = [(q, w) for q in (70, 80, 90, 95, 98) for w in (8, 16, 32)]


def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    return [pixel_auc(v_ela(img, quality=q, w=w), truth) for q, w in GRID]


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        out = np.array([o for o in ex.map(one, rows, chunksize=4)], dtype=object)

    print(f"{'quality':>8}{'window':>8}{'mean AUC':>11}{'median':>9}")
    best = None
    for i, (q, w) in enumerate(GRID):
        a = np.array([r[i] for r in out if r[i] is not None])
        print(f"{q:>8}{w:>8}{a.mean():>11.4f}{np.median(a):>9.4f}")
        if best is None or a.mean() > best[0]:
            best = (a.mean(), q, w)
    print(f"\nbest: quality={best[1]}, window={best[2]}  (AUC {best[0]:.4f})")
