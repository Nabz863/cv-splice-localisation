"""Rungs 1-2 F1 per CASIA host category, each against its own chance line (from stored counts).

Usage:  TAU=0.75 BETA=8.0 python src/methods/per_category.py      Writes results/per_category.csv
"""
import csv, os, sys
import numpy as np
from PIL import Image

GRID = "results/rung2_grid.npz"
MANIFEST = os.environ.get("MANIFEST", os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv"))
TAU = float(os.environ.get("TAU", 0.75))       # single operating point (best pooled F1 over all images)
BETA = float(os.environ.get("BETA", 8.0))


def load_grid(path):
    """Load rung2_grid.npz; return the (tau, beta) list and graph-cut counts (grid, image, tp/fp/fn)."""
    z = np.load(path, allow_pickle=False)
    grid = [(float(t), float(b)) for t, b in z["grid"]]
    solvers = [str(x) for x in z["solvers"]]
    want = os.environ.get("SOLVER", "graph")
    hits = [i for i, n in enumerate(solvers) if want in n.lower()]
    if hits:
        si = hits[0]
    elif len(solvers) == 1:
        si = 0
    else:
        raise SystemExit(f"no solver matching {want!r} in {solvers}; set SOLVER=")
    print(f"solver: {solvers[si]!r}   "
          f"taus: {sorted({t for t, _ in grid})}   betas: {sorted({b for _, b in grid})}")
    c = np.asarray(z["counts"][:, si], np.int64).transpose(1, 0, 2)
    return grid, c


def grid_index(grid, tau, beta, label):
    """Index of the grid point nearest to (tau, beta)."""
    best = min(range(len(grid)), key=lambda i: (abs(grid[i][0] - tau), abs(grid[i][1] - beta)))
    t, b = grid[best]
    if abs(t - tau) > 1e-6 or abs(b - beta) > 1e-6:
        print(f"note: {label} requested (tau={tau}, beta={beta}), "
              f"nearest on the grid is (tau={t}, beta={b})")
    return best


def f1_per_image(c):
    """Per-image F1 from (tp, fp, fn); an empty prediction scores 0."""
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def f1_micro(c):
    """Pooled F1 over a group of images."""
    tp, fp, fn = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum()
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den else 0.0


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    grid, counts = load_grid(GRID)
    n_img = counts.shape[1]
    if n_img != len(rows):
        raise SystemExit(f"{GRID} has {n_img} images, {MANIFEST} has {len(rows)}; "
                         f"they must be in the same order")

    gi_mrf = grid_index(grid, TAU, BETA, "ELA+MRF")
    gi_ela = grid_index(grid, TAU, 0.0, "ELA (beta=0)")

    # ---- tampered fraction of each image (tp + fn = spliced pixels) ----
    pos = (counts[gi_mrf][:, 0] + counts[gi_mrf][:, 2]).astype(float)
    area = np.array([np.prod(Image.open(r["mask"]).size) for r in rows], float)
    frac = pos / area
    chance_each = 2 * frac / (1 + frac)          # predict-all F1, image by image
    per_mrf, per_ela = f1_per_image(counts[gi_mrf]), f1_per_image(counts[gi_ela])

    # ---- group images by host category ----
    cats = {}
    for i, r in enumerate(rows):
        cats.setdefault(r["host_category"], []).append(i)
    groups = [(c, np.array(cats[c])) for c in sorted(cats, key=lambda c: -len(cats[c]))]
    groups.append(("ALL", np.arange(n_img)))

    # chance lines must match the metric: pooled F1 vs 2P/(1+P) with P pooled;
    # per-image mean F1 vs the mean of 2p_i/(1+p_i)
    out = []
    for cat, idx in groups:
        P = float(pos[idx].sum() / area[idx].sum())
        out.append(dict(
            category=cat, n=len(idx),
            pooled_frac=P, mean_frac=float(frac[idx].mean()),
            chance_micro=2 * P / (1 + P),
            ela_micro=f1_micro(counts[gi_ela][idx]),
            mrf_micro=f1_micro(counts[gi_mrf][idx]),
            chance_img=float(chance_each[idx].mean()),
            ela_img=float(per_ela[idx].mean()),
            mrf_img=float(per_mrf[idx].mean()),
        ))

    def table(title, ch, e, m):
        hdr = (f"{'category':<10}{'n':>6}{'chance':>9}{'ELA':>9}"
               f"{'+MRF':>9}{'gain':>9}{'vs ch.':>9}")
        print()
        print(title)
        print(hdr)
        print("-" * len(hdr))
        for r in out:
            if r["category"] == "ALL":
                print("-" * len(hdr))
            print(f"{r['category']:<10}{r['n']:>6}{r[ch]:>9.4f}{r[e]:>9.4f}{r[m]:>9.4f}"
                  f"{r[m] - r[e]:>+9.4f}{r[m] - r[ch]:>+9.4f}")
        beat = [r for r in out if r["category"] != "ALL" and r[m] > r[ch]]
        gain = [r for r in out if r["category"] != "ALL" and r[m] > r[e]]
        print(f"ELA+MRF above its own chance line in {len(beat)}/{len(out)-1} categories; "
              f"prior helps in {len(gain)}/{len(out)-1}")

    print(f"\ntau={TAU}, beta={BETA} applied to all {n_img} images "
          f"(not nested, so close to Table 1, not equal)")
    table("MICRO F1 (pooled counts; same convention as Table 1)",
          "chance_micro", "ela_micro", "mrf_micro")
    table("PER-IMAGE MEAN F1", "chance_img", "ela_img", "mrf_img")

    a = out[-1]
    pb = a["mean_frac"]
    print("\nchance-line check for Table 1:")
    print(f"  pooled tampered rate P    = {a['pooled_frac']:.4f} -> 2P/(1+P) = {a['chance_micro']:.4f}")
    print(f"  mean per-image rate p-bar = {pb:.4f} -> 2p/(1+p) = {2*pb/(1+pb):.4f}"
          f"   (what Table 1 uses now)")
    print(f"  mean of per-image chance  = {a['chance_img']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/per_category.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nwrote results/per_category.csv")


if __name__ == "__main__":
    main()
