"""Image-level AUC, step 3: pick the best of the five scores without touching the test fold.

Select on validation-fold tampered vs half A of the authentic images; report on test-fold
tampered vs the other half B. Writes results/image_auc_select_<tag>.csv.
Usage:  TAG=t0v1 python src/methods/select_summary_unet.py
"""
import csv, os, sys                               # CSV files, paths
from concurrent.futures import ProcessPoolExecutor   # parallel processing
from functools import partial                     # fixes some arguments of a function in advance
import numpy as np                                # arrays and maths

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so "eval" (in src/) can be imported
sys.path.insert(0, os.path.dirname(__file__))                       # so files in this folder can be imported
from eval.metrics import image_auc, bootstrap_auc_ci                # AUC and its confidence interval
from image_auc_unet import scores_one, read_pair_params             # the five scores; the pair's (tau, beta)

TAG = os.environ.get("TAG", "t0v1")               # which (test, validation) pair
WORKERS = int(os.environ.get("WORKERS", 10))      # parallel CPU processes
KEYS = ["unet_area", "mrf_area", "prob_mean", "prob_max", "prob_p99"]   # the five candidate scores
PREREG = {"unet_area", "mrf_area"}         # the scores planned before seeing any results


def auc(pos, neg):
    """AUC of tampered scores pos vs authentic scores neg."""
    return image_auc(list(pos) + list(neg), [1] * len(pos) + [0] * len(neg))   # label 1 = tampered


def main():
    tau, beta = read_pair_params(TAG)             # (tau, beta) this pair chose on validation
    z = np.load(f"results/logits/{TAG}.npz", allow_pickle=False)
    cal = [z[f"cal_lg_{i}"].astype(np.float32) for i in range(int(z["n_cal"]))]   # validation-fold tampered logits

    # ---- Read the scores image_auc_unet.py computed for test-fold tampered and authentic images ----
    p = f"results/image_auc_unet_{TAG}.csv"
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; run image_auc_unet.py first")
    rows = list(csv.DictReader(open(p)))                # per-image scores
    test_t = [{k: float(r[k]) for k in KEYS} for r in rows if r["label"] == "1"]   # tampered (test fold)
    auth = [{k: float(r[k]) for k in KEYS} for r in rows if r["label"] == "0"]     # authentic
    # ---- split the authentic images into two fixed halves ----
    perm = np.random.default_rng(0).permutation(len(auth))   # shuffled order, fixed seed
    h = len(auth) // 2
    auth_sel = [auth[i] for i in perm[:h]]        # half A: used for choosing the score
    auth_rep = [auth[i] for i in perm[h:]]        # half B: used for reporting it

    with ProcessPoolExecutor(WORKERS) as ex:      # compute the five scores for the validation images
        cal_s = list(ex.map(partial(scores_one, TAU=tau, BETA=beta), cal, chunksize=2))

    print(f"{TAG}: tau={tau}, beta={beta}")
    print(f"select: {len(cal_s)} val-fold tampered vs {len(auth_sel)} authentic (half A)")
    print(f"report: {len(test_t)} test-fold tampered vs {len(auth_rep)} authentic (half B)\n")

    # ---- select on validation + half A, then report every score on test + half B ----
    sel = {k: auc([s[k] for s in cal_s], [a[k] for a in auth_sel]) for k in KEYS}   # selection AUC of each score
    best = max(KEYS, key=lambda k: sel[k])        # the score with the highest selection AUC

    print(f"{'summary':<11}{'select AUC':>12}{'report AUC':>12}{'95% CI':>20}")
    print("-" * 57)
    out = []                                            # one row per summary score
    for k in KEYS:                                # report all five, marking the one selected
        t = [s[k] for s in test_t]; a = [s[k] for s in auth_rep]
        v = auc(t, a); lo, hi = bootstrap_auc_ci(t, a)      # report AUC and its 95% interval
        out.append(dict(tag=TAG, summary=k, select_auc=round(sel[k], 4),
                        report_auc=round(v, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        selected=int(k == best), preregistered=int(k in PREREG),
                        n_tampered=len(t), n_authentic=len(a), tau=tau, beta=beta))
        tag = "  <- selected" if k == best else ("  (pre-registered)" if k in PREREG else "")  # mark the chosen and the pre-registered scores
        print(f"{k:<11}{sel[k]:>12.4f}{v:>12.4f}   [{lo:.4f}, {hi:.4f}]{tag}")

    # ---- Save (build_tables.py reads this) ----
    print(f"\nquote '{best}' at its REPORT AUC: it was chosen without seeing the test fold.")
    path = f"results/image_auc_select_{TAG}.csv"        # output file
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))  # columns = keys of the first row
        w.writeheader(); w.writerows(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
