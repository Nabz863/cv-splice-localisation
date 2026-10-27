"""Rungs 1-2 F1 per CASIA host category, each against its own chance line (from stored counts).

Usage:  TAU=0.75 BETA=8.0 python src/methods/per_category.py      Writes results/per_category.csv
"""
import csv, os, sys                               # standard-library helpers
import numpy as np                                # arrays and maths
from PIL import Image                             # reads image files (only to get mask sizes)

GRID = "results/rung2_grid.npz"                   # per-image counts from run_mrf.py
MANIFEST = os.environ.get("MANIFEST", os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv"))
TAU = float(os.environ.get("TAU", 0.75))       # single operating point (best pooled F1 over all images)
BETA = float(os.environ.get("BETA", 8.0))      # prior strength at that point


# ---- Helpers ----

def load_grid(path):
    """Load rung2_grid.npz; return the (tau, beta) list and graph-cut counts (grid, image, tp/fp/fn)."""
    z = np.load(path, allow_pickle=False)
    grid = [(float(t), float(b)) for t, b in z["grid"]]     # list of (tau, beta) pairs
    solvers = [str(x) for x in z["solvers"]]                 # solver names stored in the file
    want = os.environ.get("SOLVER", "graph")                 # which solver to use (default: graph cut)
    hits = [i for i, n in enumerate(solvers) if want in n.lower()]   # solvers whose name matches
    if hits:
        si = hits[0]                              # use the first match
    elif len(solvers) == 1:
        si = 0                                    # only one solver stored: use it
    else:
        raise SystemExit(f"no solver matching {want!r} in {solvers}; set SOLVER=")
    print(f"solver: {solvers[si]!r}   "
          f"taus: {sorted({t for t, _ in grid})}   betas: {sorted({b for _, b in grid})}")
    c = np.asarray(z["counts"][:, si], np.int64).transpose(1, 0, 2)   # reorder to (grid, image, 3)
    return grid, c


def grid_index(grid, tau, beta, label):
    """Index of the grid point nearest to (tau, beta)."""
    best = min(range(len(grid)), key=lambda i: (abs(grid[i][0] - tau), abs(grid[i][1] - beta)))   # closest tau, then beta
    t, b = grid[best]
    if abs(t - tau) > 1e-6 or abs(b - beta) > 1e-6:          # warn if not an exact match
        print(f"note: {label} requested (tau={tau}, beta={beta}), "
              f"nearest on the grid is (tau={t}, beta={b})")
    return best


def f1_per_image(c):
    """Per-image F1 from (tp, fp, fn); an empty prediction scores 0."""
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)   # one value per image
    den = 2 * tp + fp + fn                        # F1 denominator
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)   # max(den, 1) avoids dividing by 0


def f1_micro(c):
    """Pooled F1 over a group of images."""
    tp, fp, fn = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum()     # add up counts first
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den else 0.0


# ---- Main ----

def main():
    rows = list(csv.DictReader(open(MANIFEST)))   # every splicing image (same order as the counts)
    grid, counts = load_grid(GRID)
    n_img = counts.shape[1]                       # number of images in the counts file
    if n_img != len(rows):                        # the two files must line up image for image
        raise SystemExit(f"{GRID} has {n_img} images, {MANIFEST} has {len(rows)}; "
                         f"they must be in the same order")

    gi_mrf = grid_index(grid, TAU, BETA, "ELA+MRF")          # rung 2's grid point
    gi_ela = grid_index(grid, TAU, 0.0, "ELA (beta=0)")      # rung 1 = same tau, no prior

    # ---- tampered fraction of each image (tp + fn = spliced pixels) ----
    pos = (counts[gi_mrf][:, 0] + counts[gi_mrf][:, 2]).astype(float)   # spliced pixels per image
    area = np.array([np.prod(Image.open(r["mask"]).size) for r in rows], float)   # total pixels per image
    frac = pos / area                             # tampered fraction per image
    chance_each = 2 * frac / (1 + frac)          # predict-all F1, image by image
    per_mrf, per_ela = f1_per_image(counts[gi_mrf]), f1_per_image(counts[gi_ela])   # per-image F1s

    # ---- group images by host category ----
    cats = {}                                     # category -> list of image indices
    for i, r in enumerate(rows):
        cats.setdefault(r["host_category"], []).append(i)
    groups = [(c, np.array(cats[c])) for c in sorted(cats, key=lambda c: -len(cats[c]))]   # largest first
    groups.append(("ALL", np.arange(n_img)))      # plus a row for every image together

    # chance lines must match the metric: pooled F1 vs 2P/(1+P) with P pooled;
    # per-image mean F1 vs the mean of 2p_i/(1+p_i)
    out = []                                      # one dict of results per category
    for cat, idx in groups:
        P = float(pos[idx].sum() / area[idx].sum())          # pooled tampered rate of this group
        out.append(dict(
            category=cat, n=len(idx),
            pooled_frac=P, mean_frac=float(frac[idx].mean()),
            chance_micro=2 * P / (1 + P),                    # pooled chance line
            ela_micro=f1_micro(counts[gi_ela][idx]),         # rung 1 pooled F1
            mrf_micro=f1_micro(counts[gi_mrf][idx]),         # rung 2 pooled F1
            chance_img=float(chance_each[idx].mean()),       # per-image chance line
            ela_img=float(per_ela[idx].mean()),              # rung 1 mean per-image F1
            mrf_img=float(per_mrf[idx].mean()),              # rung 2 mean per-image F1
        ))

    def table(title, ch, e, m):
        """Print one table; ch, e, m name the chance / ELA / MRF columns to use."""
        hdr = (f"{'category':<10}{'n':>6}{'chance':>9}{'ELA':>9}"
               f"{'+MRF':>9}{'gain':>9}{'vs ch.':>9}")
        print()
        print(title)
        print(hdr)
        print("-" * len(hdr))                     # underline as wide as the header
        for r in out:
            if r["category"] == "ALL":
                print("-" * len(hdr))             # separator before the ALL row
            print(f"{r['category']:<10}{r['n']:>6}{r[ch]:>9.4f}{r[e]:>9.4f}{r[m]:>9.4f}"
                  f"{r[m] - r[e]:>+9.4f}{r[m] - r[ch]:>+9.4f}")   # gain from the prior; margin over chance
        beat = [r for r in out if r["category"] != "ALL" and r[m] > r[ch]]   # categories above chance
        gain = [r for r in out if r["category"] != "ALL" and r[m] > r[e]]    # categories where the prior helps
        print(f"ELA+MRF above its own chance line in {len(beat)}/{len(out)-1} categories; "
              f"prior helps in {len(gain)}/{len(out)-1}")

    print(f"\ntau={TAU}, beta={BETA} applied to all {n_img} images "
          f"(not nested, so close to Table 1, not equal)")
    table("MICRO F1 (pooled counts; same convention as Table 1)",
          "chance_micro", "ela_micro", "mrf_micro")
    table("PER-IMAGE MEAN F1", "chance_img", "ela_img", "mrf_img")

    # ---- Show the difference between the two ways of computing the chance line ----
    a = out[-1]                                   # the ALL row
    pb = a["mean_frac"]                           # mean of the per-image tampered fractions
    print("\nchance-line check for Table 1:")
    print(f"  pooled tampered rate P    = {a['pooled_frac']:.4f} -> 2P/(1+P) = {a['chance_micro']:.4f}")
    print(f"  mean per-image rate p-bar = {pb:.4f} -> 2p/(1+p) = {2*pb/(1+pb):.4f}"
          f"   (what Table 1 uses now)")
    print(f"  mean of per-image chance  = {a['chance_img']:.4f}")

    # ---- Save ----
    os.makedirs("results", exist_ok=True)
    with open("results/per_category.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))   # columns = the dict keys
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})   # 4 decimals
    print("\nwrote results/per_category.csv")


if __name__ == "__main__":
    main()
