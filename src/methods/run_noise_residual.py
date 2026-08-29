import csv, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.seed import set_seed
from eval.metrics import f1_iou
from noise_residual import score_map

set_seed(0)
ROOT = os.path.join(os.environ["DATASETS"], "casia2")
TAUS = np.round(np.arange(0.5, 10.01, 0.25), 2)
N_FOLDS = 5

rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
counts = np.zeros((len(rows), len(TAUS), 3), np.int64)   # tp, fp, fn

t0 = time.time()
for i, r in enumerate(rows):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    s = score_map(img)
    for j, tau in enumerate(TAUS):
        m = f1_iou(s >= tau, truth)
        counts[i, j] = (m["tp"], m["fp"], m["fn"])
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)")

folds = np.array([int(r["fold"]) for r in rows])

def dataset_f1(idx, j):
    """Micro-averaged F1 over a set of images at threshold index j."""
    tp, fp, fn = counts[idx, j].sum(0)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0

print(f"\n{'fold':<6}{'tau*':>7}{'test F1':>10}{'test IoU':>10}")
results = []
for k in range(N_FOLDS):
    train, test = np.where(folds != k)[0], np.where(folds == k)[0]
    j = int(np.argmax([dataset_f1(train, j) for j in range(len(TAUS))]))
    tp, fp, fn = counts[test, j].sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    results.append((TAUS[j], f1, iou))
    print(f"{k:<6}{TAUS[j]:>7}{f1:>10.4f}{iou:>10.4f}")

f1s = [r[1] for r in results]
ious = [r[2] for r in results]
print(f"\nmean F1  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"mean IoU {np.mean(ious):.4f} +/- {np.std(ious):.4f}")

os.makedirs("results", exist_ok=True)
with open("results/rung1_noise_residual.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "tau", "f1", "iou"])
    for k, (tau, f1, iou) in enumerate(results):
        w.writerow([k, tau, f"{f1:.6f}", f"{iou:.6f}"])
print("\nwrote results/rung1_noise_residual.csv")
