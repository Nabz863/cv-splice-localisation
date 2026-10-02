"""Copy-move transfer test, scored against matched chance lines.

The original copy-move result compared F1 against 2p/(1+p) at the MEAN per-image
tampered rate. That is the right chance line for neither convention:
  micro F1 (pooled counts)  ->  2P/(1+P), P = pooled tampered-pixel rate
  per-image mean F1         ->  mean over images of 2p_i/(1+p_i)
This recomputes rungs 1 and 2 on every copy-move pair, keeps per-image counts, and
reports both, each against its own line. tau and beta are the splicing-selected
values, applied unchanged, as before: the question is transfer, not refitting.

CPU only. Imports no torch.

Usage (from the repo root):
    CM_MANIFEST=/home/dell/datasets/casia2/copymove_manifest.csv \
        python src/methods/copymove_matched.py

Writes results/copymove_matched.csv (summary) and results/copymove_counts.npz
(per-image tp/fp/fn, so nothing needs recomputing later).
"""
import csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import v_ela
from mrf import solve_graphcut
from image_auc_classical import zscore

MANIFEST = os.environ.get("CM_MANIFEST", "/home/dell/datasets/casia2/copymove_manifest.csv")
TAU = float(os.environ.get("TAU", 0.75))
BETA = float(os.environ.get("BETA", 8.0))
WORKERS = int(os.environ.get("WORKERS", 10))


def counts_one(r):
    img = np.asarray(Image.open(r["image"]).convert("RGB"))   # uint8, as run_mrf.py
    gt = np.asarray(Image.open(r["mask"]).convert("L")) > 127
    z = zscore(np.asarray(v_ela(img), np.float32))
    x1 = z >= TAU
    x2 = np.asarray(solve_graphcut(z, TAU, BETA), bool)
    c = lambda x: ((x & gt).sum(), (x & ~gt).sum(), (~x & gt).sum())
    return c(x1), c(x2), gt.size


def f1_micro(c):
    tp, fp, fn = c.sum(0).astype(float)
    return 2 * tp / (2 * tp + fp + fn)


def f1_img(c):
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    print(f"{len(rows)} copy-move pairs, tau={TAU}, beta={BETA}, workers={WORKERS}")
    c1 = np.zeros((len(rows), 3), np.int64)
    c2 = np.zeros((len(rows), 3), np.int64)
    area = np.zeros(len(rows))
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (a, b, n) in enumerate(ex.map(counts_one, rows, chunksize=4)):
            c1[i], c2[i], area[i] = a, b, n
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    pos = (c1[:, 0] + c1[:, 2]).astype(float)
    P = pos.sum() / area.sum()
    frac = pos / area
    ch_micro = 2 * P / (1 + P)
    ch_img = float((2 * frac / (1 + frac)).mean())

    res = [
        ("micro", "rung 1 (ELA)", f1_micro(c1), ch_micro),
        ("micro", "rung 2 (ELA+MRF)", f1_micro(c2), ch_micro),
        ("per-image", "rung 1 (ELA)", f1_img(c1), ch_img),
        ("per-image", "rung 2 (ELA+MRF)", f1_img(c2), ch_img),
    ]
    print(f"\n{'convention':<11}{'method':<18}{'F1':>8}{'chance':>9}{'vs ch.':>9}")
    print("-" * 55)
    for conv, m, f, ch in res:
        print(f"{conv:<11}{m:<18}{f:>8.4f}{ch:>9.4f}{f - ch:>+9.4f}")
    print(f"\nold results/copymove.csv: ELA 0.1278, ELA+MRF 0.1391 "
          f"-> whichever row above matches those is the convention it used")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/copymove_counts.npz", ela=c1, mrf=c2, area=area)
    with open("results/copymove_matched.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["convention", "method", "f1", "chance_f1", "margin", "tau", "beta", "n"])
        for conv, m, fv, ch in res:
            w.writerow([conv, m, round(fv, 4), round(ch, 4), round(fv - ch, 4), TAU, BETA, len(rows)])
    print("wrote results/copymove_matched.csv, results/copymove_counts.npz")


if __name__ == "__main__":
    main()
