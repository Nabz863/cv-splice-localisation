"""Copy-move transfer test: rungs 1-2 with the splicing-selected (tau, beta), unchanged.

Scored against the all-tampered chance line pooled the same way as the metric.
Writes results/copymove_matched.csv and results/copymove_counts.npz.
Usage:  python src/methods/copymove_matched.py
"""
import csv, os, sys, time                         # standard-library helpers
from concurrent.futures import ProcessPoolExecutor   # runs a function on many images in parallel
import numpy as np                                # arrays and maths
from PIL import Image                             # reads image files

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from noise_variants import v_ela                  # rung 1's score map
from mrf import solve_graphcut                    # rung 2's solver

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
MANIFEST = os.environ.get("CM_MANIFEST", os.path.join(ROOT, "copymove_manifest.csv"))   # copy-move images
TAU = float(os.environ.get("TAU", 0.75))       # splicing-selected operating point, not re-tuned
BETA = float(os.environ.get("BETA", 8.0))      # prior strength at that point
WORKERS = int(os.environ.get("WORKERS", 10))   # parallel processes


# ---- Helpers ----

def counts_one(r):
    """tp/fp/fn for rung 1 and rung 2 on one copy-move image, plus its pixel count."""
    img = np.asarray(Image.open(r["image"]).convert("RGB"))  # load the image
    gt = np.asarray(Image.open(r["mask"]).convert("L")) > 127   # true mask (True = tampered)
    z = v_ela(img)                                            # ELA score map
    x1 = z >= TAU                                             # rung 1: threshold
    x2 = np.asarray(solve_graphcut(z, TAU, BETA), bool)       # rung 2: MRF
    c = lambda x: ((x & gt).sum(), (x & ~gt).sum(), (~x & gt).sum())   # (tp, fp, fn) of a prediction
    return c(x1), c(x2), gt.size                              # gt.size = number of pixels


def f1_micro(c):
    """Pooled F1."""
    tp, fp, fn = c.sum(0).astype(float)           # add up over all images first
    return 2 * tp / (2 * tp + fp + fn)


def f1_img(c):
    """Mean per-image F1."""
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)   # one value per image
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())   # empty prediction = 0


# ---- Main ----

def main():
    rows = list(csv.DictReader(open(MANIFEST)))   # every copy-move image
    print(f"{len(rows)} copy-move pairs, tau={TAU}, beta={BETA}, workers={WORKERS}")
    c1 = np.zeros((len(rows), 3), np.int64)       # rung 1 counts per image
    c2 = np.zeros((len(rows), 3), np.int64)       # rung 2 counts per image
    area = np.zeros(len(rows))                    # pixels per image
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (a, b, n) in enumerate(ex.map(counts_one, rows, chunksize=4)):   # results in manifest order
            c1[i], c2[i], area[i] = a, b, n
            if (i + 1) % 500 == 0:                # progress every 500 images
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- chance lines: predict-all F1, pooled and per-image ----
    pos = (c1[:, 0] + c1[:, 2]).astype(float)                  # spliced pixels per image
    P = pos.sum() / area.sum()                                 # pooled tampered rate
    frac = pos / area                                          # tampered fraction per image
    ch_micro = 2 * P / (1 + P)                                 # pooled chance F1
    ch_img = float((2 * frac / (1 + frac)).mean())             # mean per-image chance F1

    res = [                                                    # (convention, method, F1, chance)
        ("micro", "rung 1 (ELA)", f1_micro(c1), ch_micro),
        ("micro", "rung 2 (ELA+MRF)", f1_micro(c2), ch_micro),
        ("per-image", "rung 1 (ELA)", f1_img(c1), ch_img),
        ("per-image", "rung 2 (ELA+MRF)", f1_img(c2), ch_img),
    ]
    print(f"\n{'convention':<11}{'method':<18}{'F1':>8}{'chance':>9}{'vs ch.':>9}")
    print("-" * 55)
    for conv, m, f, ch in res:
        print(f"{conv:<11}{m:<18}{f:>8.4f}{ch:>9.4f}{f - ch:>+9.4f}")   # last column = margin over chance
    print(f"\nold results/copymove.csv: ELA 0.1278, ELA+MRF 0.1391 "
          f"-> whichever row above matches those is the convention it used")

    # ---- Save ----
    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/copymove_counts.npz", ela=c1, mrf=c2, area=area)   # raw counts
    with open("results/copymove_matched.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["convention", "method", "f1", "chance_f1", "margin", "tau", "beta", "n"])
        for conv, m, fv, ch in res:
            w.writerow([conv, m, round(fv, 4), round(ch, 4), round(fv - ch, 4), TAU, BETA, len(rows)])
    print("wrote results/copymove_matched.csv, results/copymove_counts.npz")


if __name__ == "__main__":
    main()
