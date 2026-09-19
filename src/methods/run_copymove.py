"""Predicted failure: the noise-residual baseline cannot detect copy-move.

A copy-move forgery pastes a region from the SAME image, so the pasted pixels
carry the host's own sensor noise and compression history by construction. ELA
measures exactly that history, so it has nothing to key on. This is not a
tuning failure -- it is structural, and the point of the experiment is to show
it quantitatively rather than assert it.

Protocol: the tau selected on the SPLICING training folds is applied unchanged.
Refitting tau on copy-move would test whether ELA can be coaxed into something;
applying the splicing operating point tests whether the method transfers, which
is the question a forensic analyst actually faces.
"""
import csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import v_ela
from mrf import solve_graphcut

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
# Modal selection from the splicing folds (results/rung2_mrf_valselect.csv)
TAU, BETA = 0.75, 8.0
WORKERS = int(os.environ.get("WORKERS", 8))


def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    s = v_ela(img)
    out = []
    for x in (s >= TAU, solve_graphcut(s, TAU, BETA)):      # rung 1, then rung 2
        out.append(((x & truth).sum(), (x & ~truth).sum(), (~x & truth).sum()))
    return out, truth.mean()


if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(ROOT, "copymove_manifest.csv"))))
    print(f"{len(rows)} copy-move pairs | tau={TAU} beta={BETA} "
          f"(selected on splicing) | {WORKERS} workers", flush=True)

    t0 = time.time()
    counts = np.zeros((2, 3), np.int64)
    rates = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (c, rate) in enumerate(ex.map(one, rows, chunksize=8)):
            counts += np.array(c)
            rates.append(rate)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    p = float(np.mean(rates))
    chance = 2 * p / (1 + p)      # F1 of predicting every pixel tampered
    print(f"\nmean tampered-pixel rate {p:.4f}")
    print(f"chance F1 (predict all)  {chance:.4f}\n")

    print(f"{'method':<22}{'F1':>9}{'IoU':>9}{'vs chance':>12}")
    res = []
    for name, (tp, fp, fn) in zip(["rung 1 (ELA)", "rung 2 (ELA+MRF)"], counts):
        f1 = 2*tp / (2*tp + fp + fn)
        iou = tp / (tp + fp + fn)
        print(f"{name:<22}{f1:>9.4f}{iou:>9.4f}{f1-chance:>+12.4f}")
        res.append((name, f1, iou, chance))

    os.makedirs("results", exist_ok=True)
    with open("results/copymove.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "f1", "iou", "chance_f1", "tau", "beta", "n_images"])
        for name, f1, iou, ch in res:
            w.writerow([name, f"{f1:.6f}", f"{iou:.6f}", f"{ch:.6f}",
                        TAU, BETA, len(rows)])
    print("\nwrote results/copymove.csv")
