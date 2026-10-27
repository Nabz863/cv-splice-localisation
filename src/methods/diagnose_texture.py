"""Does the noise cue just find texture? Correlate its score with local edge strength (200 images).

Writes results/texture_corr.csv.   Usage:  python src/methods/diagnose_texture.py
"""
import csv, os, sys   # standard-library helpers
import numpy as np   # arrays and maths
from PIL import Image   # reads image files
from scipy import ndimage   # image filters (Sobel, box average)

sys.path.insert(0, os.path.dirname(__file__))   # so files in this folder can be imported
from noise_residual import score_map   # the sensor-noise score map being checked

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder

if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))[:200]   # first 200 splicing images
    # ---- Correlate the noise score with plain texture, image by image ----
    cors = []   # one correlation per image
    for r in rows:   # each image
        g = np.array(Image.open(r["image"]).convert("L"), float) / 255.0     # greyscale [0, 1]
        gx = ndimage.sobel(g, 0); gy = ndimage.sobel(g, 1)                    # image gradients
        texture = ndimage.uniform_filter(np.hypot(gx, gy), size=32)           # local edge strength
        s = score_map(np.array(Image.open(r["image"]).convert("RGB")))       # noise-cue score
        cors.append(np.corrcoef(s.ravel(), texture.ravel())[0, 1])            # Pearson correlation

    # ---- Save and summarise ----
    os.makedirs("results", exist_ok=True)   # create results/ if missing
    with open("results/texture_corr.csv", "w", newline="") as f:
        w = csv.writer(f)   # CSV writer
        w.writerow(["image", "corr_score_texture"])   # header
        for r, c in zip(rows, cors):   # each image with its correlation
            w.writerow([os.path.basename(r["image"]), f"{c:.6f}"])   # one row per image

    cors = np.array(cors)   # list -> array for the stats below
    print(f"corr(score, local texture energy): mean {np.nanmean(cors):+.3f}, "   # nanmean skips images where correlation is undefined
          f"median {np.nanmedian(cors):+.3f}")
    print(f"|corr| > 0.3 in {(np.abs(cors) > 0.3).mean():.1%} of images")   # share of images where the cue mostly tracks texture
    print("wrote results/texture_corr.csv")   # report what was written
