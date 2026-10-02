"""Choose the U-Net's image-level summary on held-out data, then report it on test.

image_auc_unet.py reports five summaries of the same logit map. The pre-registered
one (area labelled tampered, at the (tau, beta) chosen for PIXEL F1) is a poor
detector, and the best of the five was only identifiable after looking at the test
AUCs. Quoting that one directly would be selection on the test set.

So, same nested discipline as everywhere else:
  select  on  validation-fold tampered (the 'cal' split already cached in
              results/logits/{tag}.npz)  vs  half A of the authentic images
  report  on  test-fold tampered  vs  half B of the authentic images
The two authentic halves are disjoint, and neither tampered set saw the other.

CPU only. Imports no torch. Reads image_auc_unet_{tag}.csv for the test side, so
run that first.

Usage (from the repo root):
    python src/methods/select_summary_unet.py
    TAG=t0v2 python src/methods/select_summary_unet.py
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from image_auc_unet import scores_one, auc, boot_ci, read_pair_params

TAG = os.environ.get("TAG", "t0v1")
WORKERS = int(os.environ.get("WORKERS", 10))
KEYS = ["unet_area", "mrf_area", "prob_mean", "prob_max", "prob_p99"]
PREREG = {"unet_area", "mrf_area"}


def main():
    tau, beta = read_pair_params(TAG)
    z = np.load(f"results/logits/{TAG}.npz", allow_pickle=False)
    cal = [z[f"cal_lg_{i}"].astype(np.float32) for i in range(int(z["n_cal"]))]

    p = f"results/image_auc_unet_{TAG}.csv"
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; run image_auc_unet.py first")
    rows = list(csv.DictReader(open(p)))
    test_t = [{k: float(r[k]) for k in KEYS} for r in rows if r["label"] == "1"]
    auth = [{k: float(r[k]) for k in KEYS} for r in rows if r["label"] == "0"]
    perm = np.random.default_rng(0).permutation(len(auth))
    h = len(auth) // 2
    auth_sel = [auth[i] for i in perm[:h]]
    auth_rep = [auth[i] for i in perm[h:]]

    with ProcessPoolExecutor(WORKERS) as ex:
        cal_s = list(ex.map(partial(scores_one, TAU=tau, BETA=beta), cal, chunksize=2))

    print(f"{TAG}: tau={tau}, beta={beta}")
    print(f"select: {len(cal_s)} val-fold tampered vs {len(auth_sel)} authentic (half A)")
    print(f"report: {len(test_t)} test-fold tampered vs {len(auth_rep)} authentic (half B)\n")

    sel = {k: auc([s[k] for s in cal_s], [a[k] for a in auth_sel]) for k in KEYS}
    best = max(KEYS, key=lambda k: sel[k])

    print(f"{'summary':<11}{'select AUC':>12}{'report AUC':>12}{'95% CI':>20}")
    print("-" * 57)
    for k in KEYS:
        t = [s[k] for s in test_t]; a = [s[k] for s in auth_rep]
        v = auc(t, a); lo, hi = boot_ci(t, a)
        tag = "  <- selected" if k == best else ("  (pre-registered)" if k in PREREG else "")
        print(f"{k:<11}{sel[k]:>12.4f}{v:>12.4f}   [{lo:.4f}, {hi:.4f}]{tag}")

    print(f"\nquote '{best}' at its REPORT AUC: it was chosen without seeing the test fold.")


if __name__ == "__main__":
    main()
