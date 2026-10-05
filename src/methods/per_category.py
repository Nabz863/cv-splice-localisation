"""Per-category breakdown of rungs 1 and 2, from stored counts only.

The folds were stratified by the host image's content category, but no result was
ever reported at that granularity, so the stratification currently appears in the
methods with nothing attached to it. This fixes that.

Nothing is recomputed. results/rung2_grid.npz already holds the per-image
confusion counts at every (tau, beta); this regroups them by the host_category
column in the manifest, and gives each category its own chance line, since a
category whose tampered regions are larger has an easier chance baseline and
comparing it against the global 0.2351 would be misleading.

CPU, seconds, imports no torch.

Usage (from the repo root):
    python src/methods/per_category.py
    TAU=0.75 BETA=8.0 python src/methods/per_category.py

Writes results/per_category.csv
"""
import csv, os, sys
import numpy as np
from PIL import Image

GRID = "results/rung2_grid.npz"
MANIFEST = os.environ.get("MANIFEST", os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv"))
TAU = float(os.environ.get("TAU", 0.75))
BETA = float(os.environ.get("BETA", 8.0))


def load_grid(path):
    """rung2_grid.npz: counts (n_img, n_solver, n_grid, 3) as (tp, fp, fn),
    grid (n_grid, 2) as (tau, beta), solvers (n_solver,) names.
    Returns the (tau, beta) list and the graph-cut counts as (n_grid, n_img, 3)."""
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
    best = min(range(len(grid)), key=lambda i: (abs(grid[i][0] - tau), abs(grid[i][1] - beta)))
    t, b = grid[best]
    if abs(t - tau) > 1e-6 or abs(b - beta) > 1e-6:
        print(f"note: {label} requested (tau={tau}, beta={beta}), "
              f"nearest on the grid is (tau={t}, beta={b})")
    return best


def f1_per_image(c):
    """Per-image F1 from (tp, fp, fn); every ground-truth mask here is non-empty,
    so the only degenerate case is an empty prediction, which scores 0."""
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def f1_micro(c):
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

    # tp+fn is the positive count, constant across the grid. Area from the mask
    # header, so no pixels are decoded.
    pos = (counts[gi_mrf][:, 0] + counts[gi_mrf][:, 2]).astype(float)
    area = np.array([np.prod(Image.open(r["mask"]).size) for r in rows], float)
    frac = pos / area
    chance_each = 2 * frac / (1 + frac)          # predict-all F1, image by image
    per_mrf, per_ela = f1_per_image(counts[gi_mrf]), f1_per_image(counts[gi_ela])

    cats = {}
    for i, r in enumerate(rows):
        cats.setdefault(r["host_category"], []).append(i)
    groups = [(c, np.array(cats[c])) for c in sorted(cats, key=lambda c: -len(cats[c]))]
    groups.append(("ALL", np.arange(n_img)))

    # Each F1 is compared against the chance line computed THE SAME WAY:
    #   micro F1 (pooled counts)   vs  2P/(1+P), P = pooled tampered-pixel rate
    #   per-image mean F1          vs  mean over images of 2p_i/(1+p_i)
    # Mixing them (per-image F1 against 2*mean(p)/(1+mean(p))) overstates chance,
    # because 2p/(1+p) is concave, so its mean is below its value at the mean.
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
