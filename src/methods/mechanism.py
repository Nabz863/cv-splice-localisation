"""Where does the prior's gain over the U-Net come from? (Fig. 6 and the region counts)

Every pixel the prior changes is classed as FP removed / FN filled (gains) or TP lost /
FP added (losses), and binned by distance to the true boundary. Also counts connected
regions before/after and per-image F1. Uses each pair's selected (tau, beta); TAGS=t0v1
restricts to one pair. Writes results/tables/mechanism*.csv and perimage_f1.csv.
Usage:  python src/methods/mechanism.py
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from mrf import solve_graphcut

# ---- Settings ----
_ALL = ",".join(f"t{k}v{v}" for k in range(5) for v in range(5) if k != v)
TAGS = os.environ.get("TAGS", _ALL).split(",")
B1, B2 = (float(x) for x in os.environ.get("BANDS", "3,15").split(","))   # distance bands, pixels
WORKERS = int(os.environ.get("WORKERS", 10))
BANDS = [f"<= {B1:g} px", f"{B1:g}-{B2:g} px", f"> {B2:g} px"]
KINDS = ["fp_removed", "fn_filled", "tp_lost", "fp_added"]

def read_pair_params(tag):
    """(tau, beta) this pair selected on validation."""
    r = list(csv.DictReader(open(f"results/rung5_pair_{tag}.csv")))[0]
    return float(r["tau"]), float(r["beta"])


def load_test(tag):
    """(logits, mask) for each test image of a pair."""
    z = np.load(f"results/logits/{tag}.npz", allow_pickle=False)
    out = []
    for i in range(int(z["n_test"])):
        sh = tuple(z[f"test_shape_{i}"])
        gt = np.unpackbits(z[f"test_gt_{i}"])[:sh[0] * sh[1]].reshape(sh).astype(bool)
        out.append((z[f"test_lg_{i}"].astype(np.float32), gt))
    return out


def f1(x, gt):
    """Per-image F1."""
    tp = (x & gt).sum(); fp = (x & ~gt).sum(); fn = (~x & gt).sum()
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 1.0


def one(item, tau, beta):
    """For one image: what the prior changed, by distance band; region counts; F1 before/after."""
    lg, gt = item
    x4 = np.asarray(solve_graphcut(lg, tau, 0.0), bool)     # rung 4 (no prior)
    x5 = np.asarray(solve_graphcut(lg, tau, beta), bool)    # rung 5 (with prior)

    # ---- distance of every pixel to the true splice boundary -> band 0, 1 or 2 ----
    edge = gt ^ ndimage.binary_erosion(gt)                  # boundary pixels of the true mask
    d = ndimage.distance_transform_edt(~edge) if edge.any() else np.full(gt.shape, np.inf)
    band = np.where(d <= B1, 0, np.where(d <= B2, 1, 2))

    # ---- classify each pixel whose label the prior changed ----
    kinds = {"fp_removed": x4 & ~x5 & ~gt, "fn_filled": ~x4 & x5 & gt,
             "tp_lost": x4 & ~x5 & gt, "fp_added": ~x4 & x5 & ~gt}
    c = np.zeros((3, 4), np.int64)
    for k, (name, m) in enumerate(kinds.items()):
        c[:, k] = np.bincount(band[m], minlength=3)[:3]     # count per band
    # far-band corrections split: inside the tampered region vs background
    far = band == 2
    inside = np.array([(kinds["fn_filled"] & far & gt).sum(),
                       (kinds["fp_removed"] & far & ~gt).sum()])

    # ---- connected regions, and how many touch no spliced pixel ("spurious") ----
    st = np.ones((3, 3), bool)                              # 8-connectivity
    def comps(x):
        lab, n = ndimage.label(x, structure=st)
        if n == 0:
            return 0, 0
        hit = ndimage.maximum(gt, lab, index=np.arange(1, n + 1))   # does each region overlap the truth?
        return n, int((np.asarray(hit) == 0).sum())
    return c, inside, comps(x4), comps(x5), f1(x4, gt), f1(x5, gt)


def main():
    # ---- run every pair, accumulating totals ----
    tot = np.zeros((3, 4), np.int64); far_split = np.zeros(2, np.int64)
    comp4, comp5, f4s, f5s, per_img = [], [], [], [], []
    for tag in TAGS:
        tau, beta = read_pair_params(tag)
        items = load_test(tag)
        print(f"{tag}: {len(items)} test images, tau={tau}, beta={beta}", flush=True)
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, (c, ins, k4, k5, a, b) in enumerate(
                    ex.map(partial(one, tau=tau, beta=beta), items, chunksize=2)):
                tot += c; far_split += ins
                comp4.append(k4); comp5.append(k5); f4s.append(a); f5s.append(b)
                per_img.append(dict(tag=tag, index=i, f1_rung4=float(a), f1_rung5=float(b)))

    # ---- print and save ----
    gain = tot[:, 0] + tot[:, 1]; loss = tot[:, 2] + tot[:, 3]; net = gain - loss
    print(f"\nboundary bands: distance to the nearest ground-truth boundary pixel")
    print(f"{'band':<12}{'FP removed':>12}{'FN filled':>11}{'TP lost':>10}{'FP added':>10}"
          f"{'net':>11}{'share':>8}")
    print("-" * 74)
    for i, b in enumerate(BANDS):
        print(f"{b:<12}{tot[i,0]:>12,}{tot[i,1]:>11,}{tot[i,2]:>10,}{tot[i,3]:>10,}"
              f"{net[i]:>11,}{net[i]/max(net.sum(),1):>8.1%}")
    print(f"{'total':<12}{tot[:,0].sum():>12,}{tot[:,1].sum():>11,}{tot[:,2].sum():>10,}"
          f"{tot[:,3].sum():>10,}{net.sum():>11,}")
    print(f"\nfar band ({BANDS[2]}): {far_split[1]:,} background FPs removed, "
          f"{far_split[0]:,} interior holes filled")

    c4 = np.array(comp4); c5 = np.array(comp5)
    print(f"\npredicted components per image: {c4[:,0].mean():.2f} -> {c5[:,0].mean():.2f}")
    print(f"spurious (no GT overlap):       {c4[:,1].mean():.2f} -> {c5[:,1].mean():.2f}")
    f4s, f5s = np.array(f4s), np.array(f5s)
    print(f"\nper-image F1 rung 4 / rung 5: mean {f4s.mean():.3f} / {f5s.mean():.3f}, "
          f"median {np.median(f4s):.3f} / {np.median(f5s):.3f}")
    print(f"images with F1 < 0.1:          {(f4s < .1).mean():.1%} / {(f5s < .1).mean():.1%}")
    print(f"images the prior improves:     {(f5s > f4s).mean():.1%}, "
          f"worsens: {(f5s < f4s).mean():.1%}")

    out = "results/tables"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/mechanism.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["band"] + KINDS + ["net", "share"])
        for i, b in enumerate(BANDS):
            w.writerow([b] + [int(v) for v in tot[i]] + [int(net[i]), round(net[i] / max(net.sum(), 1), 6)])
    summary = dict(tags="+".join(TAGS), n_images=len(f4s), band_edges=f"{B1:g},{B2:g}",
                   far_background_fp_removed=int(far_split[1]), far_interior_filled=int(far_split[0]),
                   components_rung4=float(c4[:, 0].mean()), components_rung5=float(c5[:, 0].mean()),
                   spurious_rung4=float(c4[:, 1].mean()), spurious_rung5=float(c5[:, 1].mean()),
                   f1_mean_rung4=float(f4s.mean()), f1_mean_rung5=float(f5s.mean()),
                   f1_median_rung4=float(np.median(f4s)), f1_median_rung5=float(np.median(f5s)),
                   frac_below_0p1_rung4=float((f4s < .1).mean()), frac_below_0p1_rung5=float((f5s < .1).mean()),
                   frac_improved=float((f5s > f4s).mean()), frac_worsened=float((f5s < f4s).mean()))
    with open(f"{out}/mechanism_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(summary.items())
    with open(f"{out}/perimage_f1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys())); w.writeheader(); w.writerows(per_img)
    print(f"\nwrote {out}/mechanism.csv, mechanism_summary.csv, perimage_f1.csv")


if __name__ == "__main__":
    main()
