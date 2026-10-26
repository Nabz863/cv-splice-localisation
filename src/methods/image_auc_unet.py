"""Image-level AUC for rungs 4/5, step 2 (CPU): score tampered vs authentic images.

Uses the (tau, beta) this pair chose on validation (rung5_pair_<tag>.csv), so nothing
is tuned on test. Rung 4 = beta 0, rung 5 = chosen beta. Writes results/image_auc_unet_<tag>.csv.
Usage:  TAG=t0v1 python src/methods/image_auc_unet.py
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
    """(tau, beta) chosen on this pair's validation fold (or TAU/BETA from the environment)."""
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
    """This pair's test-fold logits (already cached by cache_logits.py)."""
    z = np.load(f"results/logits/{tag}.npz", allow_pickle=False)
    n = int(z["n_test"])
    return [z[f"test_lg_{i}"].astype(np.float32) for i in range(n)]


def load_authentic(tag):
    """This pair's authentic-image logits (from cache_auth_logits.py)."""
    p = f"results/logits/auth_{tag}.npz"
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; run cache_auth_logits.py first (it needs the GPU)")
    z = np.load(p, allow_pickle=False)
    n = int(z["n"])
    return [z[f"lg_{i}"].astype(np.float32) for i in range(n)]


def scores_one(lg, TAU, BETA):
    """Five candidate image-level scores from one logit map."""
    p = 1.0 / (1.0 + np.exp(-lg))           # logits -> probabilities
    x4 = solve_graphcut(lg, TAU, 0.0)      # rung 4: the prior switched off
    x5 = solve_graphcut(lg, TAU, BETA)     # rung 5
    return dict(
        prob_mean=float(p.mean()),
        prob_max=float(p.max()),
        prob_p99=float(np.percentile(p, 99)),
        unet_area=float(np.asarray(x4, bool).mean()),   # fraction of image labelled spliced, rung 4
        mrf_area=float(np.asarray(x5, bool).mean()),    # same for rung 5
    )


def run(maps, label, tau, beta):
    """scores_one over a list of maps, in parallel."""
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
