"""Does the noise cue just find texture? Correlate its score with local edge strength (200 images).

Writes results/texture_corr.csv.   Usage:  python src/methods/diagnose_texture.py
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
        g = np.array(Image.open(r["image"]).convert("L"), float) / 255.0     # greyscale [0, 1]
        gx = ndimage.sobel(g, 0); gy = ndimage.sobel(g, 1)                    # image gradients
        texture = ndimage.uniform_filter(np.hypot(gx, gy), size=32)           # local edge strength
        s = score_map(np.array(Image.open(r["image"]).convert("RGB")))       # noise-cue score
        cors.append(np.corrcoef(s.ravel(), texture.ravel())[0, 1])            # Pearson correlation

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
