"""Is the noise-residual cue tracking texture rather than noise?

For the first 200 manifest images, correlates the rung-1 noise-residual score
map (noise_residual.score_map) with local texture energy (Sobel gradient
magnitude averaged over a 32 x 32 window). A high correlation means the cue is
finding busy regions, not foreign noise.

CPU, about a minute. Writes results/texture_corr.csv (one row per image), which
build_tables.py summarises.

Usage (from the repo root):
    python src/methods/diagnose_texture.py
"""
import csv, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from noise_residual import score_map

ROOT = os.path.join(os.environ["DATASETS"], "casia2")

if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))[:200]
    cors = []
    for r in rows:
        g = np.array(Image.open(r["image"]).convert("L"), float) / 255.0
        gx = ndimage.sobel(g, 0); gy = ndimage.sobel(g, 1)
        texture = ndimage.uniform_filter(np.hypot(gx, gy), size=32)
        s = score_map(np.array(Image.open(r["image"]).convert("RGB")))
        cors.append(np.corrcoef(s.ravel(), texture.ravel())[0, 1])

    os.makedirs("results", exist_ok=True)
    with open("results/texture_corr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "corr_score_texture"])
        for r, c in zip(rows, cors):
            w.writerow([os.path.basename(r["image"]), f"{c:.6f}"])

    cors = np.array(cors)
    print(f"corr(score, local texture energy): mean {np.nanmean(cors):+.3f}, "
          f"median {np.nanmedian(cors):+.3f}")
    print(f"|corr| > 0.3 in {(np.abs(cors) > 0.3).mean():.1%} of images")
    print("wrote results/texture_corr.csv")
