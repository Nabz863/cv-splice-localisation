"""Where does the prior's gain over the U-Net come from? (Fig. 6 and the region counts)

Every pixel the prior changes is classed as FP removed / FN filled (gains) or TP lost /
FP added (losses), and binned by distance to the true boundary. Also counts connected
regions before/after and per-image F1. Uses each pair's selected (tau, beta); TAGS=t0v1
restricts to one pair. Writes results/tables/mechanism*.csv and perimage_f1.csv.
Usage:  python src/methods/mechanism.py
"""
import csv, os, sys                               # CSV files, paths
from concurrent.futures import ProcessPoolExecutor   # parallel processing
from functools import partial                     # fixes some arguments of a function in advance
import numpy as np                                # arrays and maths
from scipy import ndimage                         # image operations: erosion, distances, connected regions
sys.path.insert(0, os.path.dirname(__file__))     # so mrf.py can be imported
from mrf import solve_graphcut                    # the MRF solver

# ---- Settings ----
_ALL = ",".join(f"t{k}v{v}" for k in range(5) for v in range(5) if k != v)   # "t0v1,t0v2,...,t4v3" (all 20 pairs)
TAGS = os.environ.get("TAGS", _ALL).split(",")    # which pairs to analyse
B1, B2 = (float(x) for x in os.environ.get("BANDS", "3,15").split(","))   # distance bands, pixels
WORKERS = int(os.environ.get("WORKERS", 10))      # parallel CPU processes
BANDS = [f"<= {B1:g} px", f"{B1:g}-{B2:g} px", f"> {B2:g} px"]   # band names for printing
KINDS = ["fp_removed", "fn_filled", "tp_lost", "fp_added"]       # the four kinds of change

def read_pair_params(tag):
    """(tau, beta) this pair selected on validation."""
    r = list(csv.DictReader(open(f"results/rung5_pair_{tag}.csv")))[0]   # the file's single row
    return float(r["tau"]), float(r["beta"])


def load_test(tag):
    """(logits, mask) for each test image of a pair."""
    z = np.load(f"results/logits/{tag}.npz", allow_pickle=False)  # cached logits and masks for this pair
    out = []                                            # one item per test image
    for i in range(int(z["n_test"])):
        sh = tuple(z[f"test_shape_{i}"])          # mask height and width
        gt = np.unpackbits(z[f"test_gt_{i}"])[:sh[0] * sh[1]].reshape(sh).astype(bool)   # unpack the 1-bit mask
        out.append((z[f"test_lg_{i}"].astype(np.float32), gt))
    return out


def f1(x, gt):
    """Per-image F1."""
    tp = (x & gt).sum(); fp = (x & ~gt).sum(); fn = (~x & gt).sum()   # pixel counts
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 1.0               # 1.0 if there was nothing to find and nothing found


def one(item, tau, beta):
    """For one image: what the prior changed, by distance band; region counts; F1 before/after."""
    lg, gt = item                                 # logits and true mask
    x4 = np.asarray(solve_graphcut(lg, tau, 0.0), bool)     # rung 4 (no prior)
    x5 = np.asarray(solve_graphcut(lg, tau, beta), bool)    # rung 5 (with prior)

    # ---- distance of every pixel to the true splice boundary -> band 0, 1 or 2 ----
    edge = gt ^ ndimage.binary_erosion(gt)                  # boundary pixels of the true mask
    # distance from every pixel to the nearest boundary pixel (infinite if the mask is empty)
    d = ndimage.distance_transform_edt(~edge) if edge.any() else np.full(gt.shape, np.inf)
    band = np.where(d <= B1, 0, np.where(d <= B2, 1, 2))    # 0: within 3 px, 1: 3-15 px, 2: further

    # ---- classify each pixel whose label the prior changed ----
    kinds = {"fp_removed": x4 & ~x5 & ~gt, "fn_filled": ~x4 & x5 & gt,      # gains
             "tp_lost": x4 & ~x5 & gt, "fp_added": ~x4 & x5 & ~gt}         # losses
    c = np.zeros((3, 4), np.int64)                          # rows: bands; columns: the four kinds
    for k, (name, m) in enumerate(kinds.items()):
        c[:, k] = np.bincount(band[m], minlength=3)[:3]     # count per band
    # far-band corrections split: inside the tampered region vs background
    far = band == 2
    inside = np.array([(kinds["fn_filled"] & far & gt).sum(),       # holes filled inside a splice
                       (kinds["fp_removed"] & far & ~gt).sum()])    # false positives removed from the background

    # ---- connected regions, and how many touch no spliced pixel ("spurious") ----
    st = np.ones((3, 3), bool)                              # 8-connectivity
    def comps(x):
        """(number of predicted regions, number of them that touch no spliced pixel)."""
        lab, n = ndimage.label(x, structure=st)             # give each connected region its own number
        if n == 0:
            return 0, 0
        hit = ndimage.maximum(gt, lab, index=np.arange(1, n + 1))   # does each region overlap the truth?
        return n, int((np.asarray(hit) == 0).sum())
    return c, inside, comps(x4), comps(x5), f1(x4, gt), f1(x5, gt)


def main():
    # ---- run every pair, accumulating totals ----
    tot = np.zeros((3, 4), np.int64); far_split = np.zeros(2, np.int64)   # totals over all images
    comp4, comp5, f4s, f5s, per_img = [], [], [], [], []     # per-image region counts and F1s
    for tag in TAGS:
        tau, beta = read_pair_params(tag)               # rung 5's selected setting
        items = load_test(tag)                          # test images' logits and masks
        print(f"{tag}: {len(items)} test images, tau={tau}, beta={beta}", flush=True)
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, (c, ins, k4, k5, a, b) in enumerate(
                    ex.map(partial(one, tau=tau, beta=beta), items, chunksize=2)):
                tot += c; far_split += ins                   # add this image to the totals
                comp4.append(k4); comp5.append(k5); f4s.append(a); f5s.append(b)
                per_img.append(dict(tag=tag, index=i, f1_rung4=float(a), f1_rung5=float(b)))  # per-image F1 without / with the prior

    # ---- print and save ----
    gain = tot[:, 0] + tot[:, 1]; loss = tot[:, 2] + tot[:, 3]; net = gain - loss   # per band
    print(f"\nboundary bands: distance to the nearest ground-truth boundary pixel")
    print(f"{'band':<12}{'FP removed':>12}{'FN filled':>11}{'TP lost':>10}{'FP added':>10}"
          f"{'net':>11}{'share':>8}")
    print("-" * 74)
    for i, b in enumerate(BANDS):                 # one line per distance band
        print(f"{b:<12}{tot[i,0]:>12,}{tot[i,1]:>11,}{tot[i,2]:>10,}{tot[i,3]:>10,}"
              f"{net[i]:>11,}{net[i]/max(net.sum(),1):>8.1%}")
    print(f"{'total':<12}{tot[:,0].sum():>12,}{tot[:,1].sum():>11,}{tot[:,2].sum():>10,}"
          f"{tot[:,3].sum():>10,}{net.sum():>11,}")
    print(f"\nfar band ({BANDS[2]}): {far_split[1]:,} background FPs removed, "
          f"{far_split[0]:,} interior holes filled")

    c4 = np.array(comp4); c5 = np.array(comp5)    # columns: (regions, spurious regions)
    print(f"\npredicted components per image: {c4[:,0].mean():.2f} -> {c5[:,0].mean():.2f}")
    print(f"spurious (no GT overlap):       {c4[:,1].mean():.2f} -> {c5[:,1].mean():.2f}")
    f4s, f5s = np.array(f4s), np.array(f5s)             # lists -> arrays
    print(f"\nper-image F1 rung 4 / rung 5: mean {f4s.mean():.3f} / {f5s.mean():.3f}, "
          f"median {np.median(f4s):.3f} / {np.median(f5s):.3f}")
    print(f"images with F1 < 0.1:          {(f4s < .1).mean():.1%} / {(f5s < .1).mean():.1%}")
    print(f"images the prior improves:     {(f5s > f4s).mean():.1%}, "
          f"worsens: {(f5s < f4s).mean():.1%}")

    out = "results/tables"                              # output folder
    os.makedirs(out, exist_ok=True)                     # create it if missing
    with open(f"{out}/mechanism.csv", "w", newline="") as f:          # Fig. 6's numbers
        w = csv.writer(f); w.writerow(["band"] + KINDS + ["net", "share"])
        for i, b in enumerate(BANDS):
            w.writerow([b] + [int(v) for v in tot[i]] + [int(net[i]), round(net[i] / max(net.sum(), 1), 6)])
    # one-number summaries quoted in the report (region counts, per-image F1, ...)
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
    with open(f"{out}/perimage_f1.csv", "w", newline="") as f:       # every image's F1 before/after
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys())); w.writeheader(); w.writerows(per_img)
    print(f"\nwrote {out}/mechanism.csv, mechanism_summary.csv, perimage_f1.csv")


if __name__ == "__main__":
    main()
