"""Does the rung-1 score map carry signal at all, independent of threshold?"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from noise_residual import score_map

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))


def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    s = score_map(img).ravel()
    t = truth.ravel()
    if t.all() or not t.any():
        return None
    # Pixel-level ROC AUC via rank identity, subsampled for tractability
    idx = np.random.default_rng(0).choice(len(t), size=min(20000, len(t)), replace=False)
    s, t = s[idx], t[idx]
    order = s.argsort()
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = int(t.sum()), int((~t).sum())
    if npos == 0 or nneg == 0:
        return None
    auc = (ranks[t].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return auc, truth.mean()


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        out = [o for o in ex.map(one, rows, chunksize=8) if o is not None]

    aucs = np.array([o[0] for o in out])
    rates = np.array([o[1] for o in out])

    print(f"images scored: {len(aucs)}")
    print(f"\nmean tampered-pixel rate: {rates.mean():.4f}")
    p = rates.mean()
    print(f"F1 if we predict EVERY pixel tampered: {2*p/(1+p):.4f}   <-- the number to beat")
    print(f"\npixel-level AUC of the score map")
    print(f"  mean   {aucs.mean():.4f}   (0.5 = no signal)")
    print(f"  median {np.median(aucs):.4f}")
    print(f"  frac of images above 0.5: {(aucs > 0.5).mean():.3f}")
