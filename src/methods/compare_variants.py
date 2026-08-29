"""Rank candidate score maps by pixel AUC on a subset. Threshold-free."""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import VARIANTS

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
N = 400
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
rows = [r for r in rows if int(r["fold"]) != 0]
rows = list(np.random.default_rng(0).permutation(rows))[:N]


def pixel_auc(s, t):
    idx = np.random.default_rng(0).choice(t.size, size=min(20000, t.size), replace=False)
    s, t = s.ravel()[idx], t.ravel()[idx]
    npos, nneg = int(t.sum()), int((~t).sum())
    if npos == 0 or nneg == 0:
        return None
    order = s.argsort()
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[t].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    return {k: pixel_auc(f(img), truth) for k, f in VARIANTS.items()}


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        out = list(ex.map(one, rows, chunksize=4))

    print(f"pixel AUC over {len(out)} images (0.5 = no signal)\n")
    print(f"{'variant':<18}{'mean':>8}{'median':>9}{'>0.5':>8}")
    for k in VARIANTS:
        a = np.array([o[k] for o in out if o[k] is not None])
        print(f"{k:<18}{a.mean():>8.4f}{np.median(a):>9.4f}{(a > 0.5).mean():>8.1%}")
