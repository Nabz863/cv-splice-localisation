"""Image-level AUC for rungs 4 and 5, phase 2: the cuts and the AUC.

Tampered side: the test-split logits already in results/logits/{tag}.npz.
Authentic side: results/logits/auth_{tag}.npz from cache_auth_logits.py.

(tau, beta) are not re-tuned here. They are read from results/rung5_pair_{tag}.csv,
i.e. the values that pair's own validation-calibration split selected, so the
image-level number inherits the same nested discipline as the pixel-level one and
nothing is fitted on either test set.

Rung 4 and rung 5 differ only in beta, exactly as in the pixel-level comparison:
beta = 0 reduces the cut to thresholding the logits, so the gap between the two AUC
rows is attributable to the spatial prior and nothing else.

CPU ONLY. Imports no torch.

Usage (from the repo root):
    python src/methods/image_auc_unet.py
    TAG=t0v2 WORKERS=8 python src/methods/image_auc_unet.py

Writes results/image_auc_unet_{tag}.csv
"""
import csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from eval.metrics import image_auc, bootstrap_auc_ci
from mrf import solve_graphcut

TAG = os.environ.get("TAG", "t0v1")
WORKERS = int(os.environ.get("WORKERS", 10))
_TAU = os.environ.get("TAU")
_BETA = os.environ.get("BETA")


def read_pair_params(tag):
    """The (tau, beta) this pair's validation split chose, not a value picked here."""
    if _TAU is not None and _BETA is not None:
        print(f"using TAU={_TAU}, BETA={_BETA} from the environment")
        return float(_TAU), float(_BETA)
    p = f"results/rung5_pair_{tag}.csv"
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found, so the selected (tau, beta) is unknown.\n"
                         f"either run the rung 5 sweep for {tag} first, or pass them "
                         f"explicitly: TAU=.. BETA=.. python src/methods/image_auc_unet.py")
    r = list(csv.DictReader(open(p)))[0]
    return float(r["tau"]), float(r["beta"])


def load_tampered(tag):
    z = np.load(f"results/logits/{tag}.npz", allow_pickle=False)
    n = int(z["n_test"])
    return [z[f"test_lg_{i}"].astype(np.float32) for i in range(n)]


def load_authentic(tag):
    p = f"results/logits/auth_{tag}.npz"
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; run cache_auth_logits.py first (it needs the GPU)")
    z = np.load(p, allow_pickle=False)
    n = int(z["n"])
    return [z[f"lg_{i}"].astype(np.float32) for i in range(n)]


def scores_one(lg, TAU, BETA):
    p = 1.0 / (1.0 + np.exp(-lg))
    x4 = solve_graphcut(lg, TAU, 0.0)      # rung 4: the prior switched off
    x5 = solve_graphcut(lg, TAU, BETA)     # rung 5
    return dict(
        prob_mean=float(p.mean()),
        prob_max=float(p.max()),
        prob_p99=float(np.percentile(p, 99)),
        unet_area=float(np.asarray(x4, bool).mean()),
        mrf_area=float(np.asarray(x5, bool).mean()),
    )


def run(maps, label, tau, beta):
    t0, out = time.time(), []
    fn = partial(scores_one, TAU=tau, BETA=beta)
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(fn, maps, chunksize=2)):
            out.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {label} {i+1}/{len(maps)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    TAU, BETA = read_pair_params(TAG)
    tam, aut = load_tampered(TAG), load_authentic(TAG)
    print(f"{TAG}: {len(tam)} tampered (test fold), {len(aut)} authentic")
    print(f"tau={TAU}, beta={BETA} (selected on this pair's validation split)\n")

    t = run(tam, "tampered", TAU, BETA)
    a = run(aut, "authentic", TAU, BETA)

    os.makedirs("results", exist_ok=True)
    keys = ["prob_mean", "prob_max", "prob_p99", "unet_area", "mrf_area"]
    with open(f"results/image_auc_unet_{TAG}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label"] + keys)
        w.writeheader()
        for lab, g in ((1, t), (0, a)):
            for r in g:
                w.writerow({"label": lab, **r})

    print(f"\n{'summary':<12}{'rung':>6}{'AUC':>8}{'95% CI':>18}")
    print("-" * 44)
    rows = []
    for name, rung in (("unet_area", 4), ("mrf_area", 5),
                       ("prob_mean", "4*"), ("prob_max", "4*"), ("prob_p99", "4*")):
        p = [r[name] for r in t]; n = [r[name] for r in a]
        v = image_auc(p + n, [1] * len(p) + [0] * len(n)); lo, hi = bootstrap_auc_ci(p, n)
        rows.append((name, v))
        print(f"{name:<12}{str(rung):>6}{v:>8.4f}   [{lo:.4f}, {hi:.4f}]")

    d = dict(rows)
    print(f"\nprior contributes {d['mrf_area'] - d['unet_area']:+.4f} AUC "
          f"(rung 5 - rung 4, same tau, beta 0 vs {BETA})")
    print("* alternative summaries of the raw logit map, for transparency.")
    print(f"wrote results/image_auc_unet_{TAG}.csv")


if __name__ == "__main__":
    main()
