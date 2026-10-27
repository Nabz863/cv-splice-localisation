"""Compute every number quoted in the report from the results files, with cross-checks.

Reads results/*.csv / *.npz written by the method scripts (each function below names its
input) and writes results/tables/*.csv. make_figures.py reads only these tables.
  main.csv      Table I (all rungs, 20 pairs, + chance)    auc.csv        Table II (image AUC)
  gains.csv     Fig. 4 (gain per pair)                     datasize.csv   Fig. 8
  selection*.csv Fig. 7 / Sec. V-B                         sam2.csv       Fig. 9
  category.csv  Sec. V-F                                   copymove.csv   Sec. VI
  stabiliser.csv Fig. 3                                    icm*.csv, cues.csv, splits.csv, training.csv: Sec. III-IV
Usage:  python src/report/build_tables.py
"""
import csv, glob, itertools, json, os, re, struct, sys  # standard-library helpers (struct reads binary data, re = text patterns)
from collections import Counter, defaultdict            # value counter; dict with default values
import numpy as np                                      # arrays and maths

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # so src/ packages can be imported
from eval.metrics import image_auc, bootstrap_auc_ci    # image-level AUC and its bootstrap 95% interval

RES = "results"                                         # where the method scripts wrote their results
OUT = os.path.join(RES, "tables")                       # where this script writes its tables
N_FOLDS = 5                                             # number of folds
ALL_TAGS = [f"t{k}v{v}" for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]  # all 20 ordered pairs: "t0v1", "t0v2", ...
_t = os.environ.get("TAGS", "all")                      # optionally only some pairs (comma-separated)
TAGS = ALL_TAGS if _t == "all" else _t.split(",")       # the pairs to use


# --------------------------------------------------------------------------- utils
def read_csv(path):
    """List of dict rows."""
    if not os.path.exists(path):                        # stop with a hint if a results file is missing
        raise SystemExit(f"missing {path} - see the docstring for which script writes it")
    return list(csv.DictReader(open(path)))             # one dict per row


def write_csv(name, rows):
    """Write rows to results/tables/<name>."""
    os.makedirs(OUT, exist_ok=True)                     # create results/tables/ if missing
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))  # columns = keys of the first row
        w.writeheader()                                 # header row
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})  # floats rounded to 6 decimals
    print(f"  wrote {path} ({len(rows)} rows)")


def f1_of(c):
    """F1 from one (tp, fp, fn) triple."""
    tp, fp, fn = (float(x) for x in c)                  # as floats, so the division below is exact
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0  # nothing to find and nothing predicted = perfect


def iou_from_f1(f):
    """Exact for a single pooled confusion: IoU = tp/(tp+fp+fn) = F1/(2-F1)."""
    return f / (2 - f)


def fold_mean(records, key):
    """Mean over test folds of the per-fold mean over val folds, and the SD across test folds."""
    by = defaultdict(list)                              # test fold -> list of values (one per val fold)
    for r in records:
        by[int(r["test_fold"])].append(float(r[key]))   # file each value under its test fold
    m = np.array([np.mean(by[k]) for k in sorted(by)])  # mean per test fold
    return float(m.mean()), float(m.std())              # mean of those, and their spread


def chance(p):
    """All-tampered F1 at tampered rate P: 2P / (1 + P)."""
    return 2 * p / (1 + p)


# ------------------------------------------------------------------- the manifest
def load_manifest():
    """Manifest rows, their folds, and image areas."""
    path = os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv")  # same manifest the method scripts used
    rows = read_csv(path)
    folds = np.array([int(r["fold"]) for r in rows])    # each image's fold
    sizes = [tuple(int(v) for v in re.findall(r"\d+", r["img_size"])) for r in rows]  # "(800, 600)" -> (800, 600)
    area = np.array([w * h for w, h in sizes], float)   # pixels per image
    return rows, folds, area


# ------------------------------------------------------------------ rungs 1 and 2
def select_pairs(counts, grid, folds):
    """For each (test, val) pair: pick (tau, beta) on val (rung 2) and tau with beta = 0 (rung 1); score on test."""
    beta0 = [g for g, (t, b) in enumerate(grid) if b == 0.0]  # grid points with no prior

    def pooled(idx, gi):                                # pooled (tp, fp, fn) of images idx at grid point gi
        return counts[idx, gi].sum(0)

    pairs = []
    for k, v in itertools.product(range(N_FOLDS), repeat=2):  # every (test, val) combination
        if k == v:                                      # a fold can't be both
            continue
        vi, ti = np.where(folds == v)[0], np.where(folds == k)[0]  # validation / test image indices
        g2 = max(range(len(grid)), key=lambda g: (f1_of(pooled(vi, g)), -g))  # rung 2: best grid point on validation (ties -> earliest)
        g1 = max(beta0, key=lambda g: (f1_of(pooled(vi, g)), -g))  # rung 1: best beta = 0 point on validation
        f2, f1 = f1_of(pooled(ti, g2)), f1_of(pooled(ti, g1))  # both scored on the test fold
        pairs.append(dict(test_fold=k, val_fold=v,      # one record per pair
                          tau1=grid[g1][0], f1_rung1=f1, iou_rung1=iou_from_f1(f1),
                          tau2=grid[g2][0], beta2=grid[g2][1], f1_rung2=f2,
                          iou_rung2=iou_from_f1(f2), gain=f2 - f1))
    return pairs


def rung12(folds):
    """Rungs 1-2 under the 20-pair protocol, from rung2_grid.npz."""
    z = np.load(os.path.join(RES, "rung2_grid.npz"), allow_pickle=True)  # per-image counts from run_mrf.py
    solvers = [str(s) for s in z["solvers"]]            # solver names stored in the file
    counts = z["counts"][:, solvers.index("graphcut")]          # (images, grid, 3)
    grid = [tuple(map(float, g)) for g in z["grid"]]    # list of (tau, beta)
    if counts.shape[0] != len(folds):                   # counts and manifest must line up image for image
        raise SystemExit(f"rung2_grid has {counts.shape[0]} images, manifest {len(folds)}")
    pairs = select_pairs(counts, grid, folds)           # rung 1-2 selection for all 20 pairs

    # cross-check against nested_pairs.py's independent selection
    ref = {(int(r["test_fold"]), int(r["val_fold"])): float(r["test_f1"])  # pair -> F1 from nested_pairs.py
           for r in read_csv(os.path.join(RES, "rung2_nested_pairs.csv"))}
    worst = max(abs(ref[(p["test_fold"], p["val_fold"])] - p["f1_rung2"]) for p in pairs)  # largest disagreement
    if worst > 1e-9:                                    # must agree exactly
        raise SystemExit(f"rung 2 recomputation disagrees with rung2_nested_pairs.csv "
                         f"by {worst:.2e}: the manifest is not in rung2_grid.npz's order")
    print(f"  rung 2 reproduces rung2_nested_pairs.csv on all 20 pairs (max diff {worst:.1e})")
    return pairs, counts, grid, z["energies"][:, solvers.index("graphcut")]  # also return the energies for icm_tables()


def fixed_operating_point(counts, grid):
    """Single (tau, beta) with the best pooled F1 on all splicing images (used for copy-move, AUC, categories)."""
    f = [f1_of(counts[:, g].sum(0)) for g in range(len(grid))]  # pooled F1 at every grid point, all images
    tau, beta = grid[int(np.argmax(f))]                 # the best one
    if (tau, beta) != (0.75, 8.0):                      # the scripts listed below were run at (0.75, 8.0); stop if that changed
        raise SystemExit(f"operating point is now tau={tau}, beta={beta}: rerun per_category.py, "
                         f"copymove_matched.py and image_auc_classical.py with TAU/BETA set to it")
    print(f"  fixed operating point tau={tau}, beta={beta} (best pooled F1 on all splicing images)")
    return tau, beta


# ------------------------------------------------------------------ rungs 4 and 5
def rung45():
    """Rungs 4-5 per pair, from rung5_mrf_refine.csv."""
    rows = read_csv(os.path.join(RES, "rung5_mrf_refine.csv"))  # written by merge_refine.py
    for r in rows:
        r["iou_beta0"] = iou_from_f1(float(r["f1_beta0"]))  # rung 4 IoU from its F1
        check = iou_from_f1(float(r["f1"]))             # rung 5 IoU recomputed from its F1 ...
        if abs(check - float(r["iou"])) > 1e-5:         # ... must match the stored IoU
            raise SystemExit(f"IoU identity fails on {r['test_fold']},{r['val_fold']}: "
                             f"{check:.6f} vs {r['iou']}")
    return rows


def rung3():
    """Rung 3 per pair, from rung3_sam2_20pairs.csv (from run_sam2_auc.py)."""
    path = os.path.join(RES, "rung3_sam2_20pairs.csv")  # written by run_sam2_auc.py
    if not os.path.exists(path):                        # rung 3 optional: table still builds without it
        print("  rung 3: results/rung3_sam2_20pairs.csv not present - marked pending")
        return []

    rows = read_csv(path)                               # one row per pair

    if len(rows) != 20:                                 # must be complete
        raise SystemExit(f"rung 3: expected 20 pairs, found {len(rows)}")

    got = {                                             # pairs present
        (int(r["test_fold"]), int(r["val_fold"]))
        for r in rows
    }
    expected = {                                        # pairs that should be present
        (k, v)
        for k in range(N_FOLDS)
        for v in range(N_FOLDS)
        if k != v
    }

    if got != expected:
        raise SystemExit("rung 3 does not contain all 20 ordered (test, val) pairs")

    return rows


# ------------------------------------------------------- validation-pairing bias
def splits_table(pairs, r45):
    """Error from using one fixed pairing (val = test + 1) instead of averaging all four val folds."""
    fams = [("MRF (rung 2)", {(p["test_fold"], p["val_fold"]): p["f1_rung2"] for p in pairs}),  # pair -> F1, for each model
            ("U-Net (rung 4)", {(int(r["test_fold"]), int(r["val_fold"])):
                                                 float(r["f1_beta0"]) for r in r45}),
            ("U-Net + MRF (rung 5)", {(int(r["test_fold"]), int(r["val_fold"])): float(r["f1"])
                                      for r in r45})]
    out = []                                            # one row per model
    for name, d in fams:
        fixed = np.mean([d[(k, (k + 1) % N_FOLDS)] for k in range(N_FOLDS)])  # mean F1 using only val = test + 1
        per_k = [[d[(k, v)] for v in range(N_FOLDS) if v != k] for k in range(N_FOLDS)]  # for each test fold: its 4 val-fold F1s
        allp = np.mean([np.mean(x) for x in per_k])     # mean over all 20 pairs (fold-averaged)
        rng_ = [max(x) - min(x) for x in per_k]         # how much F1 moves with the val fold, per test fold
        out.append(dict(model=name, f1_fixed_pairing=float(fixed), f1_all_pairs=float(allp),  # save both, their difference, and the range
                        fixed_minus_all=float(fixed - allp),
                        val_range_min=float(min(rng_)), val_range_max=float(max(rng_))))
    return out


# ------------------------------------------------------------------ ICM vs cuts
def icm_tables(folds, grid, gc_counts, gc_energy, tau_fixed):
    """ICM vs graph cut: selected F1 and energies on identical problems."""
    path = os.path.join(RES, "rung2_grid_icm.npz")      # written by icm_vs_graphcut.py
    if not os.path.exists(path):
        print("  icm: results/rung2_grid_icm.npz not present - run icm_vs_graphcut.py")
        return None, None
    z = np.load(path)                                   # ICM counts and energies
    if [tuple(map(float, g)) for g in z["grid"]] != grid:  # same grid required
        raise SystemExit("rung2_grid_icm.npz grid differs from rung2_grid.npz")
    ic, ie = z["counts"], z["energies"]                 # counts (images, grid, 3); energies (images, grid)
    # graph cuts is exact, so it can never end above ICM on the same problem
    worst = float((gc_energy - ie).max())               # largest amount the graph cut is ABOVE ICM (should be <= 0)
    if worst > 1e-6 * max(1.0, float(np.abs(ie).max())):  # allow tiny rounding error
        raise SystemExit(f"graph cut energy exceeds ICM's by up to {worst:.3g}: not exact?")

    rows = []                                           # selected F1 per solver
    for solver, c in (("graph cut", gc_counts), ("ICM", ic)):  # run the same 20-pair selection for each solver
        p = select_pairs(c, grid, folds)
        f, fs = fold_mean(p, "f1_rung2")                # fold-averaged F1
        betas = Counter(x["beta2"] for x in p)          # how often each beta was chosen
        rows.append(dict(solver=solver, f1=f, f1_sd=fs, n_pairs=len(p),
                         beta_counts=" ".join(f"{b:g}:{n}" for b, n in sorted(betas.items())),
                         beta0_selected=betas.get(0.0, 0)))
    erows = []                                          # mean energies at the fixed tau
    for gi, (tau, beta) in enumerate(grid):
        if tau != tau_fixed or beta == 0.0:             # skip other taus and beta = 0 (solvers tie there)
            continue
        g, i = gc_energy[:, gi], ie[:, gi]              # graph-cut / ICM energy per image
        erows.append(dict(tau=tau, beta=beta, mean_energy_gc=float(g.mean()),
                          mean_energy_icm=float(i.mean()), ratio_icm_over_gc=float(i.sum() / g.sum()),
                          frac_images_gc_lower=float((g < i - 1e-9).mean())))
    return rows, erows


# ------------------------------------------- selection criterion: pooled vs per image
def _img_f1(c):
    """Per-image F1 from (tp, fp, fn) counts (1.0 if nothing to find and nothing predicted)."""
    tp, fp, fn = (c[..., k].astype(float) for k in range(3))  # tp, fp, fn arrays (last axis = 3 counts)
    d = 2 * tp + fp + fn                                # F1 denominator
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 1.0)  # inner where avoids dividing by 0


def _pooled_f1(c):
    """Pooled F1 at every grid point."""
    tp, fp, fn = (c[..., k].sum(0).astype(float) for k in range(3))  # same, but summed over images first
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def _select_eval(cal, test, grid, criterion):
    """Choose a grid point on cal by criterion (whole grid and beta = 0 only); score both on test."""
    score = _pooled_f1(cal) if criterion == "pooled" else _img_f1(cal).mean(0)  # validation score of every grid point
    zero = [g for g, (t, b) in enumerate(grid) if b == 0.0]  # grid points with no prior
    g1 = int(np.argmax(score))                          # best grid point overall
    g0 = zero[int(np.argmax(score[zero]))]              # best beta = 0 grid point
    out = {}                                            # results, keyed by name
    for name, g in (("beta0", g0), ("prior", g1)):      # score both choices on the test images
        f = _img_f1(test[:, g])                         # per-image F1 on test
        out[f"pooled_{name}"] = float(_pooled_f1(test[:, g]))  # pooled F1 on test
        out[f"img_{name}"] = float(f.mean())            # mean per-image F1 on test
        out[f"zero_{name}"] = float((test[:, g, 0] == 0).mean())  # fraction of images with zero correct pixels
        out[f"_f_{name}"] = f                       # per-image F1s, for selection_detail
    out["tau"], out["beta"] = grid[g1]                  # selected setting
    out["tau0"] = grid[g0][0]                           # selected beta = 0 threshold
    return out


def selection_table(folds, ela_counts, ela_grid):
    """Prior selected by pooled F1 vs by mean per-image F1 (Fig. 7), from rung5_counts/."""
    paths = sorted(glob.glob(os.path.join(RES, "rung5_counts", "t*v*.npz")))  # one file per pair, written by sweep_perimage.py
    if len(paths) != len(ALL_TAGS):                     # all 20 needed
        print(f"  selection: {len(paths)}/20 pairs in results/rung5_counts - run "
              f"sweep_perimage.py")
        return None
    per = []                                            # one record per (unary, criterion, pair)
    for k, v in itertools.product(range(N_FOLDS), repeat=2):        # ELA, rungs 1-2
        if k == v:
            continue
        cal, te = ela_counts[folds == v], ela_counts[folds == k]  # validation / test images' counts
        for crit in ("pooled", "per-image"):            # both selection rules
            per.append(dict(unary="ELA", criterion=crit, test_fold=k, val_fold=v,
                            **_select_eval(cal, te, ela_grid, crit)))
    ref = {(int(r["test_fold"]), int(r["val_fold"])): r  # sweep_cached.py's selections
           for r in read_csv(os.path.join(RES, "rung5_mrf_refine.csv"))}
    for p in paths:                                                  # U-Net, rungs 4-5
        tag = os.path.basename(p)[:-4]                  # "t0v1"
        k, v = int(tag[1]), int(tag[3])                 # test and val fold from the tag
        z = np.load(p)                                  # cal / test per-image counts for this pair
        grid = [tuple(map(float, g)) for g in z["grid"]]  # list of (tau, beta)
        for crit in ("pooled", "per-image"):            # both selection rules
            r = _select_eval(z["cal"], z["test"], grid, crit)  # select on cal, score on test
            if crit == "pooled":         # must reproduce sweep_cached.py's choice exactly
                a = ref[(k, v)]                         # sweep_cached.py's row for this pair
                same = (r["tau"], r["beta"]) == (float(a["tau"]), float(a["beta"]))  # same (tau, beta) chosen?
                if not same or abs(r["pooled_prior"] - float(a["f1"])) > 1e-5:  # and the same F1?
                    raise SystemExit(f"{tag}: per-image counts do not reproduce "
                                     f"rung5_mrf_refine.csv")
            per.append(dict(unary="U-Net", criterion=crit, test_fold=k, val_fold=v, **r))  # keep this pair's results
    print("  selection: pooled selection from per-image counts reproduces "
          "rung5_mrf_refine.csv on all 20 pairs")
    rows = []                                           # summary: one row per (unary, criterion)
    for unary in ("ELA", "U-Net"):                      # ELA (rungs 1-2) then U-Net (rungs 4-5)
        for crit in ("pooled", "per-image"):
            recs = [r for r in per if r["unary"] == unary and r["criterion"] == crit]  # the 20 matching records
            row = dict(unary=unary, criterion=crit)
            for key in ("pooled_beta0", "pooled_prior", "img_beta0", "img_prior",  # each quantity: fold-averaged mean and SD
                        "zero_beta0", "zero_prior"):
                row[key], row[key + "_sd"] = fold_mean(recs, key)  # fold-averaged value and SD
            row["img_gain"] = row["img_prior"] - row["img_beta0"]  # what the prior changes, per image
            row["pooled_gain"] = row["pooled_prior"] - row["pooled_beta0"]  # what the prior changes, pooled
            row["pairs_img_improved"] = sum(r["img_prior"] > r["img_beta0"] for r in recs)  # pairs where the prior helped per image
            row["beta_counts"] = " ".join(f"{b:g}:{n}" for b, n in  # how often each beta was chosen
                                          sorted(Counter(r["beta"] for r in recs).items()))
            rows.append(row)
    detail = []                                         # detail: what happens to individual images
    for unary in ("ELA", "U-Net"):
        for crit in ("pooled", "per-image"):
            recs = [r for r in per if r["unary"] == unary and r["criterion"] == crit]  # the 20 matching records
            a = np.concatenate([r["_f_beta0"] for r in recs])  # every test image's F1 without the prior
            b = np.concatenate([r["_f_prior"] for r in recs])  # ... and with the prior
            lost, kept = (b == 0) & (a > 0), b > 0      # lost = prior wiped out every correct pixel; kept = still has some
            detail.append(dict(unary=unary, criterion=crit, evaluations=len(a),
                               frac_lost=float(lost.mean()),
                               f1_beta0_where_lost=float(a[lost].mean()) if lost.any() else "",
                               f1_beta0_where_kept=float(a[kept].mean()),
                               f1_prior_where_kept=float(b[kept].mean()),
                               frac_improved=float((b > a).mean()),
                               frac_worsened=float((b < a).mean())))
    return rows, detail


# ------------------------------------------------------------------- rung 1 cues
def cue_table():
    """Rung 1 cue AUCs, chosen ELA setting and texture correlation."""
    need = [os.path.join(RES, f) for f in ("cue_auc.csv", "ela_tuning.csv", "texture_corr.csv")]  # the three input files
    missing = [p for p in need if not os.path.exists(p)]  # which are missing
    if missing:
        print(f"  cues: {', '.join(missing)} not present - run compare_variants.py, "
              f"tune_ela.py, diagnose_texture.py")
        return None
    rows = [dict(item=f"auc {r['variant']}", value=float(r["mean_auc"]), n=int(r["n_images"]))  # one row per cue's mean AUC
            for r in read_csv(need[0])]
    tun = read_csv(need[1])                             # ELA setting search
    best = max(tun, key=lambda r: float(r["mean_auc"]))  # best mean AUC
    rows += [dict(item="ela best quality", value=float(best["quality"]), n=int(best["n_images"])),
             dict(item="ela best window", value=float(best["window"]), n=int(best["n_images"])),
             dict(item="ela best auc", value=float(best["mean_auc"]), n=int(best["n_images"]))]
    # the same choice with each other test fold held out instead of fold 0
    for k in range(1, N_FOLDS):                         # folds 1-4
        pa = os.path.join(RES, f"cue_auc_holdout{k}.csv")  # cue comparison with fold k held out
        pb = os.path.join(RES, f"ela_tuning_holdout{k}.csv")  # ELA tuning with fold k held out
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  cues: holdout {k} files missing - run HOLDOUT={k} compare_variants.py "
                  f"and tune_ela.py")
            continue
        cue = max(read_csv(pa), key=lambda r: float(r["mean_auc"]))  # best cue with fold k held out
        tb = max(read_csv(pb), key=lambda r: float(r["mean_auc"]))  # best ELA setting with fold k held out
        same = (cue["variant"] == "E_ela" and (float(tb["quality"]), float(tb["window"]))  # same choice as with fold 0 held out?
                == (float(best["quality"]), float(best["window"])))
        rows.append(dict(item=f"holdout {k}: same cue and settings", value=float(same),  # 1.0 = same, 0.0 = different
                         n=int(cue["n_images"])))
        if not same:
            print(f"  WARNING: holdout {k} chose {cue['variant']} q={tb['quality']} "
                  f"w={tb['window']}")
    c = np.array([float(r["corr_score_texture"]) for r in read_csv(need[2])])  # per-image correlation with texture
    c = c[np.isfinite(c)]                               # drop undefined values
    rows += [dict(item="texture corr mean", value=float(c.mean()), n=len(c)),  # mean, and share above 0.3
             dict(item="texture |corr|>0.3 frac", value=float((np.abs(c) > 0.3).mean()), n=len(c))]
    return rows


# --------------------------------------------------------------------------- main
def main():
    """Build every table in order."""
    print("building tables")
    rows, folds, area = load_manifest()                 # manifest rows, folds, image areas
    pairs, counts, grid, gc_energy = rung12(folds)      # rungs 1-2
    r45 = rung45()                                      # rungs 4-5
    r3 = rung3()                                        # rung 3 (empty list if missing)
    cat = read_csv(os.path.join(RES, "per_category.csv"))  # from per_category.py
    allrow = next(r for r in cat if r["category"] == "ALL")  # its ALL row
    P = float(allrow["pooled_frac"])                    # pooled tampered rate of all splicing images

    # ---- main.csv (Table 1)
    b2 = sum(p["beta2"] == 0.0 for p in pairs)          # pairs where rung 2 chose beta = 0
    b5 = sum(float(r["beta"]) == 0.0 for r in r45)      # pairs where rung 5 chose beta = 0
    main_rows = [dict(key="chance", rung="", label="Chance (predict all tampered)",  # first row: the chance line
                      f1=chance(P), f1_sd="", iou=P, n_pairs="", beta0_selected="",
                      source="per_category.csv pooled rate")]
    for key, rung, label, recs, fk, ik, b0, src in (    # one row per rung: (key, rung, label, records, F1 column, IoU column, beta=0 count, source)
            ("rung1", 1, "ELA", pairs, "f1_rung1", "iou_rung1", "", "rung2_grid.npz, beta=0 only"),
            ("rung2", 2, "ELA + MRF", pairs, "f1_rung2", "iou_rung2", b2, "rung2_grid.npz"),
            ("rung4", 4, "U-Net", r45, "f1_beta0", "iou_beta0", "", "rung5_mrf_refine.csv f1_beta0"),
            ("rung5", 5, "U-Net + MRF", r45, "f1", "iou", b5, "rung5_mrf_refine.csv")):
        f, fs = fold_mean(recs, fk)                     # fold-averaged F1 and SD
        i, _ = fold_mean(recs, ik)                      # fold-averaged IoU
        main_rows.append(dict(key=key, rung=rung, label=label, f1=f, f1_sd=fs, iou=i,
                              n_pairs=len(recs), beta0_selected=b0, source=src))
    if r3:                                              # rung 3, if present ...
        f, fs = fold_mean(r3, "f1")
        i, _ = fold_mean(r3, "iou")                     # fold-averaged IoU
        main_rows.insert(3, dict(key="rung3", rung=3, label="SAM2, zero-shot",  # ... goes in position 3 (after rung 2)
                                 f1=f, f1_sd=fs, iou=i, n_pairs=len(r3),
                                 beta0_selected="", source="rung3_sam2_20pairs.csv"))
    write_csv("main.csv", main_rows)                    # Table I

    # ---- sam2.csv: rung 3's size-filtered union vs the ground-truth oracle that
    # picks each image's best single proposal (a diagnostic, never a method)
    if r3:
        fa, fsa = fold_mean(r3, "f1")                   # union's F1
        fo, fso = fold_mean(r3, "oracle_f1")            # oracle's F1
        write_csv("sam2.csv", [dict(variant="automatic", f1=fa, f1_sd=fsa, n_pairs=len(r3)),
                               dict(variant="oracle", f1=fo, f1_sd=fso, n_pairs=len(r3))])

    # ---- gains.csv (per pair)
    gains = [dict(unary="ELA", test_fold=p["test_fold"], val_fold=p["val_fold"],  # rung 2 gain per pair
                  gain=p["gain"], beta=p["beta2"]) for p in pairs]
    gains += [dict(unary="U-Net", test_fold=int(r["test_fold"]), val_fold=int(r["val_fold"]),  # rung 5 gain per pair
                   gain=float(r["prior_gain"]), beta=float(r["beta"])) for r in r45]
    write_csv("gains.csv", gains)

    # ---- datasize.csv: the rung 4/5 pipeline trained on subsets, full data = Table 1
    tr_sizes = [int((folds != k).sum() - (folds == v).sum())  # training-set size of each pair
                for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
    n_full = int(round(np.mean(tr_sizes)))              # their average (the full-data point)
    ds = []                                             # one row per training-set size
    for suffix, n in (("_n50", 50), ("_n150", 150), ("_n400", 400), ("_n800", 800), ("", n_full)):  # file suffix and training-set size
        path = os.path.join(RES, f"rung5_mrf_refine{suffix}.csv")  # that size's results
        if not os.path.exists(path):
            print(f"  datasize: {path} not present - run slurm/rerun_datasize.sbatch")
            continue
        recs = read_csv(path)                           # 20 pairs
        f0, f0s = fold_mean(recs, "f1_beta0")           # U-Net alone
        f1, f1s = fold_mean(recs, "f1")                 # U-Net + MRF
        ds.append(dict(n_train=n, f1=f0, f1_sd=f0s, f1_prior=f1, f1_prior_sd=f1s,
                       gain=f1 - f0, n_pairs=len(recs)))
    if ds:
        write_csv("datasize.csv", ds)

    # ---- training.csv: every rung 4/5 model, including any non-finite batches it skipped
    tr_rows = []                                        # one row per training-set size
    for suffix, n in (("_n50", 50), ("_n150", 150), ("_n400", 400), ("_n800", 800), ("", n_full)):
        path = os.path.join(RES, f"rung5_training{suffix}.csv")  # per-model training log from cache_logits.py
        if not os.path.exists(path):
            continue
        recs = read_csv(path)                           # one row per model
        if "nonfinite_train" not in recs[0]:            # older logs lack this column
            raise SystemExit(f"{path} does not record non-finite batches - retrain with the "
                             f"current cache_logits.py")
        stop = np.array([int(r["stopped_at"]) for r in recs])  # epoch each model stopped at
        nt = np.array([int(r["nonfinite_train"]) for r in recs])  # skipped training batches per model
        nv = np.array([int(r["nonfinite_val"]) for r in recs])  # skipped validation batches per model
        tr_rows.append(dict(n_train=n, models=len(recs), stopped_min=int(stop.min()),
                            stopped_median=float(np.median(stop)), stopped_max=int(stop.max()),
                            nonfinite_train=int(nt.sum()), nonfinite_val=int(nv.sum()),
                            models_with_nonfinite=int(((nt + nv) > 0).sum())))
        if nt.sum() or nv.sum():                        # warn if any batch was skipped
            print(f"  WARNING: {path}: {nt.sum()} training and {nv.sum()} validation "
                  f"batches were non-finite and skipped")
    if tr_rows:
        write_csv("training.csv", tr_rows)

    # ---- category.csv
    write_csv("category.csv", [dict(category=r["category"], n=int(r["n"]),  # Sec. V-F: one row per category
                                    pooled_frac=float(r["pooled_frac"]),
                                    chance=float(r["chance_micro"]),
                                    ela=float(r["ela_micro"]), mrf=float(r["mrf_micro"]))
                               for r in cat if r["category"] != "ALL"])

    # ---- auc.csv
    auc_rows = []                                       # one row per (rung, score, pair)
    cl = read_csv(os.path.join(RES, "image_auc_classical.csv"))  # rungs 1-2 per-image scores
    jpg = lambda p: p.lower().endswith((".jpg", ".jpeg"))  # is the file a JPEG?
    for subset, keep in (("all", lambda r: True), ("jpeg", lambda r: jpg(r["path"]))):  # all images, then JPEG only
        for score, rung, label in (("ela_area", 1, "ELA"), ("mrf_area", 2, "ELA + MRF")):  # rung 1 and rung 2 scores
            t = [float(r[score]) for r in cl if r["label"] == "1" and keep(r)]  # tampered images' scores
            a = [float(r[score]) for r in cl if r["label"] == "0" and keep(r)]  # authentic images' scores
            lo, hi = bootstrap_auc_ci(t, a)             # bootstrap 95% interval
            auc_rows.append(dict(rung=rung, label=label, score="area", tag="all", subset=subset,
                                 auc=image_auc(t + a, [1] * len(t) + [0] * len(a)),
                                 ci_lo=lo, ci_hi=hi, n_tampered=len(t), n_authentic=len(a),
                                 selected_on_heldout=0))
    for tag in TAGS:                                    # rungs 4-5, per pair
        sel = read_csv(os.path.join(RES, f"image_auc_select_{tag}.csv"))  # which summary score was chosen on held-out data
        chosen = next(r["summary"] for r in sel if r["selected"] == "1")  # the chosen one
        ur = read_csv(os.path.join(RES, f"image_auc_unet_{tag}.csv"))  # per-image scores for this pair
        t_ = [r for r in ur if r["label"] == "1"]       # tampered
        a_ = [r for r in ur if r["label"] == "0"]       # authentic
        perm = np.random.default_rng(0).permutation(len(a_))       # as select_summary_unet.py
        half_b = [a_[i] for i in perm[len(a_) // 2:]]   # the second half (the first half was used to choose the summary)
        scores = [("unet_area", 4, "U-Net"), ("mrf_area", 5, "U-Net + MRF")]  # always report both area scores
        if chosen not in ("unet_area", "mrf_area"):     # else already a row, flagged selected
            scores.append((chosen, 4, "U-Net"))
        for score, rung, label in scores:
            t = [float(r[score]) for r in t_]           # tampered scores
            a = [float(r[score]) for r in half_b]       # authentic scores (second half)
            lo, hi = bootstrap_auc_ci(t, a)
            v = image_auc(t + a, [1] * len(t) + [0] * len(a))  # AUC
            ref = next(r for r in sel if r["summary"] == score)  # same number select_summary_unet.py reported
            if abs(v - float(ref["report_auc"])) > 1e-3:  # must agree
                raise SystemExit(f"{tag} {score}: {v:.4f} here vs {ref['report_auc']} in "
                                 f"image_auc_select_{tag}.csv")
            auc_rows.append(dict(rung=rung, label=label,  # one row per score
                                 score="area" if score.endswith("area") else score,
                                 tag=tag, subset="all", auc=v, ci_lo=lo, ci_hi=hi,
                                 n_tampered=len(t), n_authentic=len(a),
                                 selected_on_heldout=int(score == chosen)))
    # With every pair cached, add the cross-validated summary: mean over test folds
    # of the per-fold mean over val folds, as for pixel F1, and its SD across folds.
    if set(TAGS) == set(ALL_TAGS):                      # only when every pair is present
        per = [r for r in auc_rows if r["tag"] in ALL_TAGS]  # per-pair rows
        groups = (("U-Net", 4, lambda r: r["rung"] == 4 and r["score"] == "area"),  # (label, rung, which rows): area scores, then the held-out choice
                  ("U-Net + MRF", 5, lambda r: r["rung"] == 5 and r["score"] == "area"),
                  ("U-Net", 4, lambda r: r["selected_on_heldout"] == 1))
        for i, (label, rung, keep) in enumerate(groups):
            by_tag = {r["tag"]: r for r in per if keep(r)}       # one row per pair
            assert len(by_tag) == len(ALL_TAGS), f"auc group {i}: {len(by_tag)} pairs"
            m, sd = fold_mean([dict(test_fold=int(t[1]), auc=r["auc"])  # fold-averaged AUC and SD
                               for t, r in by_tag.items()], "auc")
            score = "area"                              # what to write in the score column
            if i == 2:                                  # third group: list which summary each pair chose
                chosen = Counter(r["score"] for r in by_tag.values())
                score = "held-out choice: " + ", ".join(f"{k} {n}/20" for k, n in chosen.most_common())
            auc_rows.append(dict(rung=rung, label=label, score=score, tag="all20", subset="all",  # one summary row per group
                                 auc=m, ci_lo="", ci_hi="", n_tampered="", n_authentic="",
                                 selected_on_heldout=int(i == 2), auc_sd=sd))
    sam_auc_path = os.path.join(RES, "image_auc_sam2_summary.csv")  # from run_sam2_auc.py
    if os.path.exists(sam_auc_path):                    # rung 3 optional
        sam_auc = read_csv(sam_auc_path)

        if len(sam_auc) != 20:                          # 20 pairs needed
            raise SystemExit(f"SAM2 AUC: expected 20 pairs, found {len(sam_auc)}")

        if any(int(r["n_authentic"]) != 7491 for r in sam_auc):  # all authentic images needed
            raise SystemExit("SAM2 AUC does not use all 7,491 authentic images")

        m, sd = fold_mean(sam_auc, "auc")               # fold-averaged AUC and SD

        auc_rows.append(dict(                           # one summary row for SAM2
            rung=3,
            label="SAM2, zero-shot",
            score="area",
            tag="all20",
            subset="all",
            auc=m,
            ci_lo="",
            ci_hi="",
            n_tampered="",
            n_authentic=7491,
            selected_on_heldout=0,
            auc_sd=sd
        ))

    for r in auc_rows:                                  # give every row an auc_sd column (blank if none)
        r.setdefault("auc_sd", "")

    write_csv("auc.csv", auc_rows)                      # Table II

    # ---- copymove.csv: margin over each dataset's own pooled chance line
    tau, beta = fixed_operating_point(counts, grid)     # (0.75, 8.0)
    gi = grid.index((tau, beta)); gi0 = grid.index((tau, 0.0))  # its grid index, and the same tau with beta = 0
    cm = np.load(os.path.join(RES, "copymove_counts.npz"))  # from copymove_matched.py
    sets = {"splicing": (counts[:, gi0], counts[:, gi], area),  # dataset -> (rung 1 counts, rung 2 counts, image areas)
            "copy-move": (cm["ela"], cm["mrf"], cm["area"].astype(float))}
    cm_rows = []                                        # one row per (dataset, method)
    for name, (c_ela, c_mrf, ar) in sets.items():
        pos = (c_ela[:, 0] + c_ela[:, 2]).astype(float)  # tampered pixels per image
        if name == "splicing" and abs(pos.sum() / ar.sum() - P) > 1e-4:  # splicing rate must match per_category.csv
            raise SystemExit(f"pooled tampered rate from manifest sizes {pos.sum()/ar.sum():.4f} "
                             f"!= per_category.csv {P:.4f}")
        rng, b1, b2_ = np.random.default_rng(0), [], []  # seeded generator; bootstrap margins for rung 1 and rung 2
        for _ in range(2000):                       # resample images, chance line included
            i = rng.integers(0, len(ar), len(ar))       # random image indices, with repeats
            ch = chance(pos[i].sum() / ar[i].sum())     # chance line of that resample
            b1.append(f1_of(c_ela[i].sum(0)) - ch); b2_.append(f1_of(c_mrf[i].sum(0)) - ch)  # margins over chance on the resample
        ch = chance(pos.sum() / ar.sum())               # chance line of the real set
        for method, c, b in (("ELA", c_ela, b1), ("ELA + MRF", c_mrf, b2_)):  # rung 1 then rung 2
            f = f1_of(c.sum(0))                         # pooled F1
            cm_rows.append(dict(dataset=name, method=method, n=len(ar), f1=f, chance=ch,  # F1, chance, margin and its 95% interval
                                margin=f - ch, ci_lo=float(np.percentile(b, 2.5)),
                                ci_hi=float(np.percentile(b, 97.5)), tau=tau, beta=beta))
    write_csv("copymove.csv", cm_rows)                  # Sec. VI

    # ---- splits.csv
    write_csv("splits.csv", splits_table(pairs, r45))   # validation-pairing effect

    # ---- icm.csv, icm_energy.csv
    icm_rows, icm_e = icm_tables(folds, grid, counts, gc_energy, tau)  # needs icm_vs_graphcut.py
    if icm_rows:                                        # skipped if missing
        write_csv("icm.csv", icm_rows)
        write_csv("icm_energy.csv", icm_e)

    # ---- selection.csv
    sel = selection_table(folds, counts, grid)          # needs sweep_perimage.py
    if sel:
        write_csv("selection.csv", sel[0])
        write_csv("selection_detail.csv", sel[1])

    # ---- cues.csv
    cues = cue_table()                                  # needs compare_variants.py, tune_ela.py, diagnose_texture.py
    if cues:
        write_csv("cues.csv", cues)

    # ---- stabiliser.csv
    st = stabiliser_histories()                         # needs the wandb/ folder
    if st:
        write_csv("stabiliser.csv", st)
    print("done")


# -------------------------------------------------------------- W&B history files
def _wandb_records(path):
    """Yield raw records from a .wandb log file (7-byte header, then 32 KiB blocks of chunks)."""
    b = open(path, "rb").read()                         # whole file as bytes
    pos, buf, block = 7, b"", 32768                     # start after the 7-byte header; partial record; block size
    while pos + 7 <= len(b):                            # while a 7-byte chunk header still fits
        left = block - pos % block                      # bytes left in this block
        if left < 7:                                    # too few for a header: skip to the next block
            pos += left
            continue
        _, ln, typ = struct.unpack("<IHB", b[pos:pos + 7]); pos += 7  # chunk header: checksum, length, type
        if typ == 0 and ln == 0:                        # padding to the end of the block
            pos += left - 7
            continue
        data = b[pos:pos + ln]; pos += ln               # the chunk's data
        if typ == 1:                                    # type 1: a whole record
            yield data
        elif typ == 2:                                  # type 2: first piece of a record
            buf = data
        elif typ == 3:                                  # type 3: middle piece
            buf += data
        elif typ == 4:                                  # type 4: last piece
            yield buf + data; buf = b""                 # return the joined record; reset


def stabiliser_histories():
    """Per-epoch weight norms of the stabiliser runs, read from wandb/ (Fig. 3)."""
    try:
        from wandb.proto import wandb_internal_pb2 as pb  # W&B's record format
    except ImportError:
        print("  stabiliser: wandb not installed - skipped")
        return []
    runs = {}                                           # candidate name -> (file, epoch records)
    files = glob.glob("wandb/run-*/run-*.wandb") + glob.glob("wandb/offline-run-*/run-*.wandb")  # every W&B run log
    stamp = lambda f: re.search(r"run-(\d{8}_\d{6})", os.path.dirname(f)).group(1)  # date-time from the folder name, e.g. "20260912_143015"
    for f in sorted(files, key=stamp):                  # oldest first
        cfg, hist = {}, []                              # this run's settings and per-epoch logs
        for d in _wandb_records(f):                     # every record in the file
            r = pb.Record(); r.ParseFromString(d)       # decode it
            kind = r.WhichOneof("record_type")          # what kind of record it is
            if kind in ("config", "run"):
                for it in (r.config.update if kind == "config" else r.run.config.update):  # settings can come in either kind
                    try:
                        cfg[it.key] = json.loads(it.value_json)  # value stored as JSON
                    except ValueError:
                        pass
            elif kind == "history":                     # one logged step
                hist.append({(it.key or "/".join(it.nested_key)): json.loads(it.value_json)  # key -> value
                             for it in r.history.item})
        name = cfg.get("candidate")                     # which candidate this run was
        name = name.get("value") if isinstance(name, dict) else name  # sometimes stored as {"value": ...}
        ep = [h for h in hist if "diag/weight_norm" in h and "epoch" in h]  # epoch records that have the weight norm
        if not name or not ep:                          # not a stabiliser run
            continue
        runs[name] = (f, ep)            # files are in start order: the latest run wins
    # The histories must be the runs that wrote stabiliser_comparison.csv (the rerun
    # under the current loader), not an older run of the same candidate.
    summ = os.path.join(RES, "stabiliser_comparison.csv")  # from compare_stabilisers.py
    if os.path.exists(summ):
        listed = {r["candidate"] for r in read_csv(summ)}  # candidates in that file
        runs = {k: v for k, v in runs.items() if k in listed}    # only the rerun's candidates
        for r in read_csv(summ):                        # each candidate must match its history
            if r["candidate"] not in runs:
                raise SystemExit(f"stabiliser: no W&B history for {r['candidate']}")
            ep = runs[r["candidate"]][1]
            first = next((int(h["epoch"]) for h in ep if h.get("non_finite_batches", 0)), None)  # first epoch with a skipped batch
            want = int(r["first_nan_epoch"]) if r["first_nan_epoch"] else None  # what the CSV says
            if first != want or len(ep) != int(r["epochs_survived"]):  # both must agree, and the epoch count too
                raise SystemExit(f"stabiliser {r['candidate']}: W&B history ({len(ep)} epochs, "
                                 f"first non-finite {first}) does not match "
                                 f"stabiliser_comparison.csv - commit the rerun's wandb runs")
    out = []                                            # one row per (candidate, epoch)
    for name, (f, ep) in sorted(runs.items()):
        print(f"  stabiliser {name}: {len(ep)} epochs from {os.path.dirname(f)}")
        for h in ep:
            out.append(dict(candidate=name, epoch=int(h["epoch"]),  # one row per epoch
                            weight_norm=float(h["diag/weight_norm"]),
                            non_finite=int(h.get("non_finite_batches", 0))))
    return out


if __name__ == "__main__":
    main()
