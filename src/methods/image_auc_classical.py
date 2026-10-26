"""Image-level AUC for rungs 1-2: can the methods tell tampered images from authentic ones?

Score per image = fraction of pixels labelled tampered (ela_area = rung 1, mrf_area = rung 2);
ela_mean and ela_p99 are reported for comparison. Writes results/image_auc_classical.csv.
Usage:  python src/methods/image_auc_classical.py
"""
import csv, os, sys, time, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from eval.metrics import image_auc, bootstrap_auc_ci
from noise_variants import v_ela
from mrf import solve_graphcut

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
MANIFEST = os.environ.get("MANIFEST", os.path.join(ROOT, "splice_manifest.csv"))
AUTH_DIR = os.environ.get("AUTH_DIR", os.path.join(ROOT, "images", "CASIA2.0_revised", "Au"))
TAU = float(os.environ.get("TAU", 0.75))
BETA = float(os.environ.get("BETA", 8.0))
WORKERS = int(os.environ.get("WORKERS", 10))
N_AUTH = int(os.environ.get("N_AUTH", 0))       # 0 = every authentic image


def scores_one(path):
    """Image-level scores for one image (rung 1 and rung 2 areas, plus two alternatives)."""
    try:
        img = np.asarray(Image.open(path).convert("RGB"))
        z = v_ela(img)
        x = solve_graphcut(z, TAU, BETA)
        return dict(
            path=path,
            ela_mean=float(z.mean()),
            ela_p99=float(np.percentile(z, 99)),
            ela_area=float((z >= TAU).mean()),               # rung 1: fraction labelled tampered
            mrf_area=float(np.asarray(x, bool).mean()),      # rung 2: same, after the MRF
        )
    except Exception as e:                      # one unreadable file must not kill the run
        return dict(path=path, error=f"{type(e).__name__}: {e}")


def run(paths, label):
    """scores_one over many images in parallel; drops (and reports) failures."""
    t0, out = time.time(), []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(scores_one, paths, chunksize=4)):
            out.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {label} {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    bad = [r for r in out if "error" in r]
    if bad:
        print(f"  {label}: {len(bad)} failed, e.g. {bad[0]['path']} -> {bad[0]['error']}")
    return [r for r in out if "error" not in r]


def main():
    # ---- tampered images from the manifest, authentic images from the Au folder ----
    rows = list(csv.DictReader(open(MANIFEST)))
    tam = [r["image"] for r in rows]
    auth_dir = AUTH_DIR
    auth = sorted(sum((glob.glob(os.path.join(auth_dir, e))
                       for e in ("*.jpg", "*.JPG", "*.tif", "*.TIF",
                                 "*.png", "*.PNG", "*.bmp", "*.BMP")), []))
    if not auth:
        raise SystemExit(f"no images found in {auth_dir}")
    if N_AUTH and N_AUTH < len(auth):
        auth = list(np.random.default_rng(0).permutation(auth))[:N_AUTH]

    print(f"tampered  {len(tam)} from {MANIFEST}")
    print(f"authentic {len(auth)} from {auth_dir}")
    print(f"tau={TAU}, beta={BETA}, workers={WORKERS}\n")

    t = run(tam, "tampered")
    a = run(auth, "authentic")

    os.makedirs("results", exist_ok=True)
    keys = ["path", "label", "ela_mean", "ela_p99", "ela_area", "mrf_area"]
    with open("results/image_auc_classical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for lab, group in ((1, t), (0, a)):
            for r in group:
                w.writerow({**{k: r[k] for k in keys if k != "label"}, "label": lab})

    print(f"\n{'summary':<12}{'rung':>6}{'AUC':>8}{'95% CI':>18}")
    print("-" * 44)
    for name, rung in (("ela_area", 1), ("mrf_area", 2),
                       ("ela_mean", "1*"), ("ela_p99", "1*")):
        p = [r[name] for r in t]; n = [r[name] for r in a]
        v = image_auc(p + n, [1] * len(p) + [0] * len(n)); lo, hi = bootstrap_auc_ci(p, n)
        print(f"{name:<12}{str(rung):>6}{v:>8.4f}   [{lo:.4f}, {hi:.4f}]")
    print("\n* alternative summaries of the same rung 1 map, shown for transparency;")
    print("  ela_area / mrf_area are the pre-registered pair, since they are the same")
    print("  statistic computed before and after the prior.")
    print("\n0.5 is chance. AUC is invariant to the tampered:authentic ratio.")
    print("wrote results/image_auc_classical.csv")


if __name__ == "__main__":
    main()
