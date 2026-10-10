"""Every number the report and the figures quote, computed from committed files.

Nothing here is typed in by hand. Each table is derived from a results file that
an earlier script wrote, and wherever two files should agree, this script checks
that they do and stops if they do not. make_figures.py then reads ONLY the tables
this writes, so a figure can never show a number the tables do not contain.

Inputs
  $DATASETS/casia2/splice_manifest.csv  folds and image sizes; its row order is
                                        the row order of rung2_grid.npz
  results/rung2_grid.npz                rungs 1-2: per-image (tp, fp, fn) at every
                                        (tau, beta), written by run_mrf.py
  results/rung2_nested_pairs.csv        cross-check for the rung 2 recomputation
  results/rung5_mrf_refine.csv          rungs 4-5, all 20 pairs (merge_refine.py)
  results/rung5_mrf_refine_n{50,..}.csv data-size curve: the same pipeline trained
                                        on 50/150/400/800 masks (merge_refine.py)
  results/per_category.csv              per-category results (per_category.py)
  results/copymove_counts.npz           copy-move per-image counts (copymove_matched.py)
  results/image_auc_classical.csv       per-image scores, rungs 1-2
  results/image_auc_unet_{tag}.csv      per-image scores, rungs 4-5
  results/image_auc_select_{tag}.csv    which U-Net summary was chosen on held-out
                                        data (select_summary_unet.py). TAGS=all (the
                                        default) reads all 20 pairs and adds the
                                        cross-validated rows; TAGS=t0v1,t0v2 reads two
  results/rung3_sam2.csv                OPTIONAL. columns: variant,f1,iou with
                                        variant in {automatic, oracle}
  results/rung2_grid_icm.npz            ICM on the same grid (icm_vs_graphcut.py)
  results/cue_auc.csv                   rung 1 cue comparison (compare_variants.py)
  results/ela_tuning.csv                ELA quality/window sweep (tune_ela.py)
  results/texture_corr.csv              noise cue vs texture (diagnose_texture.py)
  results/rung5_counts/{tag}.npz        per-image counts at every (tau, beta) for
                                        rungs 4-5 (sweep_perimage.py), OPTIONAL
  wandb/run-*/run-*.wandb               stabiliser training histories

Outputs (results/tables/)
  main.csv        Table 1: every rung under the same 20-pair protocol, + chance
  gains.csv       per-pair F1 added by the prior, over each unary
  datasize.csv    rungs 4 and 5 F1 against training-set size
  category.csv    per-category F1, each against its own chance line
  auc.csv         image-level AUC with bootstrap CIs (Table 2)
  copymove.csv    margin over chance, splicing vs copy-move, with CIs
  stabiliser.csv  per-epoch weight norm for each stabiliser candidate
  splits.csv      fixed (k, k+1 mod 5) pairing vs all 20 pairs, per model family
  icm.csv         ICM vs graph cuts: selected F1 and beta under the 20-pair protocol
  icm_energy.csv  ICM vs graph-cut energy on identical problems, per beta
  cues.csv        rung 1 cue AUCs, the chosen ELA setting, texture correlation
  selection.csv   the prior's effect when (tau, beta) is selected by pooled F1 vs
                  by mean per-image F1, for both unaries (needs rung5_counts/)
  selection_detail.csv  per-image detail behind selection.csv, over all 20 pairs

Usage (from the repo root, CPU only, about a minute):
    python src/report/build_tables.py
"""
import csv, glob, itertools, json, os, re, struct, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.metrics import image_auc, bootstrap_auc_ci

RES = "results"
OUT = os.path.join(RES, "tables")
N_FOLDS = 5
ALL_TAGS = [f"t{k}v{v}" for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
_t = os.environ.get("TAGS", "all")
TAGS = ALL_TAGS if _t == "all" else _t.split(",")


# --------------------------------------------------------------------------- utils
def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - see the docstring for which script writes it")
    return list(csv.DictReader(open(path)))


def write_csv(name, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {path} ({len(rows)} rows)")


def f1_of(c):
    tp, fp, fn = (float(x) for x in c)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0


def iou_from_f1(f):
    """Exact for a single pooled confusion: IoU = tp/(tp+fp+fn) = F1/(2-F1)."""
    return f / (2 - f)


def fold_mean(records, key):
    """Mean over test folds of the per-fold mean over val folds, and its spread.
    Every test fold weighs equally -- the convention of merge_refine."""
    by = defaultdict(list)
    for r in records:
        by[int(r["test_fold"])].append(float(r[key]))
    m = np.array([np.mean(by[k]) for k in sorted(by)])
    return float(m.mean()), float(m.std())


def chance(p):
    return 2 * p / (1 + p)


# ------------------------------------------------------------------- the manifest
def load_manifest():
    path = os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv")
    rows = read_csv(path)
    folds = np.array([int(r["fold"]) for r in rows])
    sizes = [tuple(int(v) for v in re.findall(r"\d+", r["img_size"])) for r in rows]
    area = np.array([w * h for w, h in sizes], float)
    return rows, folds, area


# ------------------------------------------------------------------ rungs 1 and 2
def select_pairs(counts, grid, folds):
    """For every ordered (test, val) pair: choose (tau, beta) on the val fold by
    pooled F1, over the whole grid (rung 2) and over beta = 0 only (rung 1), and
    score both on the test fold. Ties go to the earlier grid entry."""
    beta0 = [g for g, (t, b) in enumerate(grid) if b == 0.0]

    def pooled(idx, gi):
        return counts[idx, gi].sum(0)

    pairs = []
    for k, v in itertools.product(range(N_FOLDS), repeat=2):
        if k == v:
            continue
        vi, ti = np.where(folds == v)[0], np.where(folds == k)[0]
        g2 = max(range(len(grid)), key=lambda g: (f1_of(pooled(vi, g)), -g))
        g1 = max(beta0, key=lambda g: (f1_of(pooled(vi, g)), -g))
        f2, f1 = f1_of(pooled(ti, g2)), f1_of(pooled(ti, g1))
        pairs.append(dict(test_fold=k, val_fold=v,
                          tau1=grid[g1][0], f1_rung1=f1, iou_rung1=iou_from_f1(f1),
                          tau2=grid[g2][0], beta2=grid[g2][1], f1_rung2=f2,
                          iou_rung2=iou_from_f1(f2), gain=f2 - f1))
    return pairs


def rung12(folds):
    z = np.load(os.path.join(RES, "rung2_grid.npz"), allow_pickle=True)
    solvers = [str(s) for s in z["solvers"]]
    counts = z["counts"][:, solvers.index("graphcut")]          # (images, grid, 3)
    grid = [tuple(map(float, g)) for g in z["grid"]]
    if counts.shape[0] != len(folds):
        raise SystemExit(f"rung2_grid has {counts.shape[0]} images, manifest {len(folds)}")
    pairs = select_pairs(counts, grid, folds)

    # Cross-check: run_mrf.py / nested_pairs.py already selected rung 2 on all 20
    # pairs. If the manifest's order differs from the grid's, this is where it shows.
    ref = {(int(r["test_fold"]), int(r["val_fold"])): float(r["test_f1"])
           for r in read_csv(os.path.join(RES, "rung2_nested_pairs.csv"))}
    worst = max(abs(ref[(p["test_fold"], p["val_fold"])] - p["f1_rung2"]) for p in pairs)
    if worst > 1e-9:
        raise SystemExit(f"rung 2 recomputation disagrees with rung2_nested_pairs.csv "
                         f"by {worst:.2e}: the manifest is not in rung2_grid.npz's order")
    print(f"  rung 2 reproduces rung2_nested_pairs.csv on all 20 pairs (max diff {worst:.1e})")
    return pairs, counts, grid, z["energies"][:, solvers.index("graphcut")]


def fixed_operating_point(counts, grid):
    """The single (tau, beta) applied unchanged to other data (copy-move, the
    authentic images, the per-category split): the grid point with the highest
    pooled F1 over all 1,822 splicing images. Those other sets share no image with
    the splicing set, so selecting on all of it leaks nothing into them.
    per_category.py, copymove_matched.py and image_auc_classical.py take the
    operating point as TAU/BETA (default 0.75, 8.0); this stops if they disagree."""
    f = [f1_of(counts[:, g].sum(0)) for g in range(len(grid))]
    tau, beta = grid[int(np.argmax(f))]
    if (tau, beta) != (0.75, 8.0):
        raise SystemExit(f"operating point is now tau={tau}, beta={beta}: rerun per_category.py, "
                         f"copymove_matched.py and image_auc_classical.py with TAU/BETA set to it")
    print(f"  fixed operating point tau={tau}, beta={beta} (best pooled F1 on all splicing images)")
    return tau, beta


# ------------------------------------------------------------------ rungs 4 and 5
def rung45():
    rows = read_csv(os.path.join(RES, "rung5_mrf_refine.csv"))
    for r in rows:
        r["iou_beta0"] = iou_from_f1(float(r["f1_beta0"]))
        check = iou_from_f1(float(r["f1"]))
        if abs(check - float(r["iou"])) > 1e-5:
            raise SystemExit(f"IoU identity fails on {r['test_fold']},{r['val_fold']}: "
                             f"{check:.6f} vs {r['iou']}")
    return rows


def rung3():
    path = os.path.join(RES, "rung3_sam2.csv")
    if not os.path.exists(path):
        print("  rung 3: results/rung3_sam2.csv not present - marked pending")
        return {}
    return {r["variant"]: r for r in read_csv(path)}


# ------------------------------------------------------- validation-pairing bias
def splits_table(pairs, r45):
    """How much would the usual single pairing (val = test + 1 mod 5) have
    misstated each result, relative to averaging over all four val folds?"""
    fams = [("MRF (rung 2)", {(p["test_fold"], p["val_fold"]): p["f1_rung2"] for p in pairs}),
            ("U-Net (rung 4)", {(int(r["test_fold"]), int(r["val_fold"])):
                                                 float(r["f1_beta0"]) for r in r45}),
            ("U-Net + MRF (rung 5)", {(int(r["test_fold"]), int(r["val_fold"])): float(r["f1"])
                                      for r in r45})]
    out = []
    for name, d in fams:
        fixed = np.mean([d[(k, (k + 1) % N_FOLDS)] for k in range(N_FOLDS)])
        per_k = [[d[(k, v)] for v in range(N_FOLDS) if v != k] for k in range(N_FOLDS)]
        allp = np.mean([np.mean(x) for x in per_k])
        rng_ = [max(x) - min(x) for x in per_k]
        out.append(dict(model=name, f1_fixed_pairing=float(fixed), f1_all_pairs=float(allp),
                        fixed_minus_all=float(fixed - allp),
                        val_range_min=float(min(rng_)), val_range_max=float(max(rng_))))
    return out


# ------------------------------------------------------------------ ICM vs cuts
def icm_tables(folds, grid, gc_counts, gc_energy, tau_fixed):
    path = os.path.join(RES, "rung2_grid_icm.npz")
    if not os.path.exists(path):
        print("  icm: results/rung2_grid_icm.npz not present - run icm_vs_graphcut.py")
        return None, None
    z = np.load(path)
    if [tuple(map(float, g)) for g in z["grid"]] != grid:
        raise SystemExit("rung2_grid_icm.npz grid differs from rung2_grid.npz")
    ic, ie = z["counts"], z["energies"]
    # graph cuts is exact, so it can never end above ICM on the same problem
    worst = float((gc_energy - ie).max())
    if worst > 1e-6 * max(1.0, float(np.abs(ie).max())):
        raise SystemExit(f"graph cut energy exceeds ICM's by up to {worst:.3g}: not exact?")

    rows = []
    for solver, c in (("graph cut", gc_counts), ("ICM", ic)):
        p = select_pairs(c, grid, folds)
        f, fs = fold_mean(p, "f1_rung2")
        betas = Counter(x["beta2"] for x in p)
        rows.append(dict(solver=solver, f1=f, f1_sd=fs, n_pairs=len(p),
                         beta_counts=" ".join(f"{b:g}:{n}" for b, n in sorted(betas.items())),
                         beta0_selected=betas.get(0.0, 0)))
    erows = []
    for gi, (tau, beta) in enumerate(grid):
        if tau != tau_fixed or beta == 0.0:
            continue
        g, i = gc_energy[:, gi], ie[:, gi]
        erows.append(dict(tau=tau, beta=beta, mean_energy_gc=float(g.mean()),
                          mean_energy_icm=float(i.mean()), ratio_icm_over_gc=float(i.sum() / g.sum()),
                          frac_images_gc_lower=float((g < i - 1e-9).mean())))
    return rows, erows


# ------------------------------------------- selection criterion: pooled vs per image
def _img_f1(c):
    """Per-image F1 from (..., 3) counts; an image with nothing to find and
    nothing predicted scores 1."""
    tp, fp, fn = (c[..., k].astype(float) for k in range(3))
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 1.0)


def _pooled_f1(c):
    tp, fp, fn = (c[..., k].sum(0).astype(float) for k in range(3))
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def _select_eval(cal, test, grid, criterion):
    """cal/test: (images, grid, 3). Choose a grid point on cal by `criterion`
    over the whole grid and over beta = 0 only (first index wins ties, as in
    sweep_cached.py), then score both on test."""
    score = _pooled_f1(cal) if criterion == "pooled" else _img_f1(cal).mean(0)
    zero = [g for g, (t, b) in enumerate(grid) if b == 0.0]
    g1 = int(np.argmax(score))
    g0 = zero[int(np.argmax(score[zero]))]
    out = {}
    for name, g in (("beta0", g0), ("prior", g1)):
        f = _img_f1(test[:, g])
        out[f"pooled_{name}"] = float(_pooled_f1(test[:, g]))
        out[f"img_{name}"] = float(f.mean())
        out[f"zero_{name}"] = float((test[:, g, 0] == 0).mean())
        out[f"_f_{name}"] = f                       # per-image F1s, for selection_detail
    out["tau"], out["beta"] = grid[g1]
    out["tau0"] = grid[g0][0]
    return out


def selection_table(folds, ela_counts, ela_grid):
    paths = sorted(glob.glob(os.path.join(RES, "rung5_counts", "t*v*.npz")))
    if len(paths) != len(ALL_TAGS):
        print(f"  selection: {len(paths)}/20 pairs in results/rung5_counts - run "
              f"sweep_perimage.py")
        return None
    per = []
    for k, v in itertools.product(range(N_FOLDS), repeat=2):        # ELA, rungs 1-2
        if k == v:
            continue
        cal, te = ela_counts[folds == v], ela_counts[folds == k]
        for crit in ("pooled", "per-image"):
            per.append(dict(unary="ELA", criterion=crit, test_fold=k, val_fold=v,
                            **_select_eval(cal, te, ela_grid, crit)))
    ref = {(int(r["test_fold"]), int(r["val_fold"])): r
           for r in read_csv(os.path.join(RES, "rung5_mrf_refine.csv"))}
    for p in paths:                                                  # U-Net, rungs 4-5
        tag = os.path.basename(p)[:-4]
        k, v = int(tag[1]), int(tag[3])
        z = np.load(p)
        grid = [tuple(map(float, g)) for g in z["grid"]]
        for crit in ("pooled", "per-image"):
            r = _select_eval(z["cal"], z["test"], grid, crit)
            if crit == "pooled":         # must reproduce sweep_cached.py's choice exactly
                a = ref[(k, v)]
                same = (r["tau"], r["beta"]) == (float(a["tau"]), float(a["beta"]))
                if not same or abs(r["pooled_prior"] - float(a["f1"])) > 1e-5:
                    raise SystemExit(f"{tag}: per-image counts do not reproduce "
                                     f"rung5_mrf_refine.csv")
            per.append(dict(unary="U-Net", criterion=crit, test_fold=k, val_fold=v, **r))
    print("  selection: pooled selection from per-image counts reproduces "
          "rung5_mrf_refine.csv on all 20 pairs")
    rows = []
    for unary in ("ELA", "U-Net"):
        for crit in ("pooled", "per-image"):
            recs = [r for r in per if r["unary"] == unary and r["criterion"] == crit]
            row = dict(unary=unary, criterion=crit)
            for key in ("pooled_beta0", "pooled_prior", "img_beta0", "img_prior",
                        "zero_beta0", "zero_prior"):
                row[key], row[key + "_sd"] = fold_mean(recs, key)
            row["img_gain"] = row["img_prior"] - row["img_beta0"]
            row["pooled_gain"] = row["pooled_prior"] - row["pooled_beta0"]
            row["pairs_img_improved"] = sum(r["img_prior"] > r["img_beta0"] for r in recs)
            row["beta_counts"] = " ".join(f"{b:g}:{n}" for b, n in
                                          sorted(Counter(r["beta"] for r in recs).items()))
            rows.append(row)
    detail = []
    for unary in ("ELA", "U-Net"):
        for crit in ("pooled", "per-image"):
            recs = [r for r in per if r["unary"] == unary and r["criterion"] == crit]
            a = np.concatenate([r["_f_beta0"] for r in recs])
            b = np.concatenate([r["_f_prior"] for r in recs])
            lost, kept = (b == 0) & (a > 0), b > 0
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
    need = [os.path.join(RES, f) for f in ("cue_auc.csv", "ela_tuning.csv", "texture_corr.csv")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print(f"  cues: {', '.join(missing)} not present - run compare_variants.py, "
              f"tune_ela.py, diagnose_texture.py")
        return None
    rows = [dict(item=f"auc {r['variant']}", value=float(r["mean_auc"]), n=int(r["n_images"]))
            for r in read_csv(need[0])]
    tun = read_csv(need[1])
    best = max(tun, key=lambda r: float(r["mean_auc"]))
    rows += [dict(item="ela best quality", value=float(best["quality"]), n=int(best["n_images"])),
             dict(item="ela best window", value=float(best["window"]), n=int(best["n_images"])),
             dict(item="ela best auc", value=float(best["mean_auc"]), n=int(best["n_images"]))]
    # the same choice with each other test fold held out instead of fold 0
    for k in range(1, N_FOLDS):
        pa = os.path.join(RES, f"cue_auc_holdout{k}.csv")
        pb = os.path.join(RES, f"ela_tuning_holdout{k}.csv")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  cues: holdout {k} files missing - run HOLDOUT={k} compare_variants.py "
                  f"and tune_ela.py")
            continue
        cue = max(read_csv(pa), key=lambda r: float(r["mean_auc"]))
        tb = max(read_csv(pb), key=lambda r: float(r["mean_auc"]))
        same = (cue["variant"] == "E_ela" and (float(tb["quality"]), float(tb["window"]))
                == (float(best["quality"]), float(best["window"])))
        rows.append(dict(item=f"holdout {k}: same cue and settings", value=float(same),
                         n=int(cue["n_images"])))
        if not same:
            print(f"  WARNING: holdout {k} chose {cue['variant']} q={tb['quality']} "
                  f"w={tb['window']}")
    c = np.array([float(r["corr_score_texture"]) for r in read_csv(need[2])])
    c = c[np.isfinite(c)]
    rows += [dict(item="texture corr mean", value=float(c.mean()), n=len(c)),
             dict(item="texture |corr|>0.3 frac", value=float((np.abs(c) > 0.3).mean()), n=len(c))]
    return rows


# --------------------------------------------------------------------------- main
def main():
    print("building tables")
    rows, folds, area = load_manifest()
    pairs, counts, grid, gc_energy = rung12(folds)
    r45 = rung45()
    r3 = rung3()
    cat = read_csv(os.path.join(RES, "per_category.csv"))
    allrow = next(r for r in cat if r["category"] == "ALL")
    P = float(allrow["pooled_frac"])

    # ---- main.csv (Table 1)
    b2 = sum(p["beta2"] == 0.0 for p in pairs)
    b5 = sum(float(r["beta"]) == 0.0 for r in r45)
    main_rows = [dict(key="chance", rung="", label="Chance (predict all tampered)",
                      f1=chance(P), f1_sd="", iou=P, n_pairs="", beta0_selected="",
                      source="per_category.csv pooled rate")]
    for key, rung, label, recs, fk, ik, b0, src in (
            ("rung1", 1, "ELA", pairs, "f1_rung1", "iou_rung1", "", "rung2_grid.npz, beta=0 only"),
            ("rung2", 2, "ELA + MRF", pairs, "f1_rung2", "iou_rung2", b2, "rung2_grid.npz"),
            ("rung4", 4, "U-Net", r45, "f1_beta0", "iou_beta0", "", "rung5_mrf_refine.csv f1_beta0"),
            ("rung5", 5, "U-Net + MRF", r45, "f1", "iou", b5, "rung5_mrf_refine.csv")):
        f, fs = fold_mean(recs, fk)
        i, _ = fold_mean(recs, ik)
        main_rows.append(dict(key=key, rung=rung, label=label, f1=f, f1_sd=fs, iou=i,
                              n_pairs=len(recs), beta0_selected=b0, source=src))
    if "automatic" in r3:
        main_rows.insert(3, dict(key="rung3", rung=3, label="SAM2, zero-shot",
                                 f1=float(r3["automatic"]["f1"]), f1_sd="",
                                 iou=float(r3["automatic"]["iou"]), n_pairs="",
                                 beta0_selected="", source="rung3_sam2.csv"))
    write_csv("main.csv", main_rows)

    # ---- gains.csv (per pair)
    gains = [dict(unary="ELA", test_fold=p["test_fold"], val_fold=p["val_fold"],
                  gain=p["gain"], beta=p["beta2"]) for p in pairs]
    gains += [dict(unary="U-Net", test_fold=int(r["test_fold"]), val_fold=int(r["val_fold"]),
                   gain=float(r["prior_gain"]), beta=float(r["beta"])) for r in r45]
    write_csv("gains.csv", gains)

    # ---- datasize.csv: the rung 4/5 pipeline trained on subsets, full data = Table 1
    tr_sizes = [int((folds != k).sum() - (folds == v).sum())
                for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
    n_full = int(round(np.mean(tr_sizes)))
    ds = []
    for suffix, n in (("_n50", 50), ("_n150", 150), ("_n400", 400), ("_n800", 800), ("", n_full)):
        path = os.path.join(RES, f"rung5_mrf_refine{suffix}.csv")
        if not os.path.exists(path):
            print(f"  datasize: {path} not present - run slurm/rerun_datasize.sbatch")
            continue
        recs = read_csv(path)
        f0, f0s = fold_mean(recs, "f1_beta0")
        f1, f1s = fold_mean(recs, "f1")
        ds.append(dict(n_train=n, f1=f0, f1_sd=f0s, f1_prior=f1, f1_prior_sd=f1s,
                       gain=f1 - f0, n_pairs=len(recs)))
    if ds:
        write_csv("datasize.csv", ds)

    # ---- category.csv
    write_csv("category.csv", [dict(category=r["category"], n=int(r["n"]),
                                    pooled_frac=float(r["pooled_frac"]),
                                    chance=float(r["chance_micro"]),
                                    ela=float(r["ela_micro"]), mrf=float(r["mrf_micro"]))
                               for r in cat if r["category"] != "ALL"])

    # ---- auc.csv
    auc_rows = []
    cl = read_csv(os.path.join(RES, "image_auc_classical.csv"))
    jpg = lambda p: p.lower().endswith((".jpg", ".jpeg"))
    for subset, keep in (("all", lambda r: True), ("jpeg", lambda r: jpg(r["path"]))):
        for score, rung, label in (("ela_area", 1, "ELA"), ("mrf_area", 2, "ELA + MRF")):
            t = [float(r[score]) for r in cl if r["label"] == "1" and keep(r)]
            a = [float(r[score]) for r in cl if r["label"] == "0" and keep(r)]
            lo, hi = bootstrap_auc_ci(t, a)
            auc_rows.append(dict(rung=rung, label=label, score="area", tag="all", subset=subset,
                                 auc=image_auc(t + a, [1] * len(t) + [0] * len(a)),
                                 ci_lo=lo, ci_hi=hi, n_tampered=len(t), n_authentic=len(a),
                                 selected_on_heldout=0))
    for tag in TAGS:
        sel = read_csv(os.path.join(RES, f"image_auc_select_{tag}.csv"))
        chosen = next(r["summary"] for r in sel if r["selected"] == "1")
        ur = read_csv(os.path.join(RES, f"image_auc_unet_{tag}.csv"))
        t_ = [r for r in ur if r["label"] == "1"]
        a_ = [r for r in ur if r["label"] == "0"]
        perm = np.random.default_rng(0).permutation(len(a_))       # as select_summary_unet.py
        half_b = [a_[i] for i in perm[len(a_) // 2:]]
        scores = [("unet_area", 4, "U-Net"), ("mrf_area", 5, "U-Net + MRF")]
        if chosen not in ("unet_area", "mrf_area"):     # else already a row, flagged selected
            scores.append((chosen, 4, "U-Net"))
        for score, rung, label in scores:
            t = [float(r[score]) for r in t_]
            a = [float(r[score]) for r in half_b]
            lo, hi = bootstrap_auc_ci(t, a)
            v = image_auc(t + a, [1] * len(t) + [0] * len(a))
            ref = next(r for r in sel if r["summary"] == score)
            if abs(v - float(ref["report_auc"])) > 1e-3:
                raise SystemExit(f"{tag} {score}: {v:.4f} here vs {ref['report_auc']} in "
                                 f"image_auc_select_{tag}.csv")
            auc_rows.append(dict(rung=rung, label=label,
                                 score="area" if score.endswith("area") else score,
                                 tag=tag, subset="all", auc=v, ci_lo=lo, ci_hi=hi,
                                 n_tampered=len(t), n_authentic=len(a),
                                 selected_on_heldout=int(score == chosen)))
    # With every pair cached, add the cross-validated summary: mean over test folds
    # of the per-fold mean over val folds, as for pixel F1, and its SD across folds.
    if set(TAGS) == set(ALL_TAGS):
        per = [r for r in auc_rows if r["tag"] in ALL_TAGS]
        groups = (("U-Net", 4, lambda r: r["rung"] == 4 and r["score"] == "area"),
                  ("U-Net + MRF", 5, lambda r: r["rung"] == 5 and r["score"] == "area"),
                  ("U-Net", 4, lambda r: r["selected_on_heldout"] == 1))
        for i, (label, rung, keep) in enumerate(groups):
            by_tag = {r["tag"]: r for r in per if keep(r)}       # one row per pair
            assert len(by_tag) == len(ALL_TAGS), f"auc group {i}: {len(by_tag)} pairs"
            m, sd = fold_mean([dict(test_fold=int(t[1]), auc=r["auc"])
                               for t, r in by_tag.items()], "auc")
            score = "area"
            if i == 2:
                chosen = Counter(r["score"] for r in by_tag.values())
                score = "held-out choice: " + ", ".join(f"{k} {n}/20" for k, n in chosen.most_common())
            auc_rows.append(dict(rung=rung, label=label, score=score, tag="all20", subset="all",
                                 auc=m, ci_lo="", ci_hi="", n_tampered="", n_authentic="",
                                 selected_on_heldout=int(i == 2), auc_sd=sd))
        for r in auc_rows:
            r.setdefault("auc_sd", "")
    write_csv("auc.csv", auc_rows)

    # ---- copymove.csv: margin over each dataset's own pooled chance line
    tau, beta = fixed_operating_point(counts, grid)
    gi = grid.index((tau, beta)); gi0 = grid.index((tau, 0.0))
    cm = np.load(os.path.join(RES, "copymove_counts.npz"))
    sets = {"splicing": (counts[:, gi0], counts[:, gi], area),
            "copy-move": (cm["ela"], cm["mrf"], cm["area"].astype(float))}
    cm_rows = []
    for name, (c_ela, c_mrf, ar) in sets.items():
        pos = (c_ela[:, 0] + c_ela[:, 2]).astype(float)
        if name == "splicing" and abs(pos.sum() / ar.sum() - P) > 1e-4:
            raise SystemExit(f"pooled tampered rate from manifest sizes {pos.sum()/ar.sum():.4f} "
                             f"!= per_category.csv {P:.4f}")
        rng, b1, b2_ = np.random.default_rng(0), [], []
        for _ in range(2000):                       # resample images, chance line included
            i = rng.integers(0, len(ar), len(ar))
            ch = chance(pos[i].sum() / ar[i].sum())
            b1.append(f1_of(c_ela[i].sum(0)) - ch); b2_.append(f1_of(c_mrf[i].sum(0)) - ch)
        ch = chance(pos.sum() / ar.sum())
        for method, c, b in (("ELA", c_ela, b1), ("ELA + MRF", c_mrf, b2_)):
            f = f1_of(c.sum(0))
            cm_rows.append(dict(dataset=name, method=method, n=len(ar), f1=f, chance=ch,
                                margin=f - ch, ci_lo=float(np.percentile(b, 2.5)),
                                ci_hi=float(np.percentile(b, 97.5)), tau=tau, beta=beta))
    write_csv("copymove.csv", cm_rows)

    # ---- splits.csv
    write_csv("splits.csv", splits_table(pairs, r45))

    # ---- icm.csv, icm_energy.csv
    icm_rows, icm_e = icm_tables(folds, grid, counts, gc_energy, tau)
    if icm_rows:
        write_csv("icm.csv", icm_rows)
        write_csv("icm_energy.csv", icm_e)

    # ---- selection.csv
    sel = selection_table(folds, counts, grid)
    if sel:
        write_csv("selection.csv", sel[0])
        write_csv("selection_detail.csv", sel[1])

    # ---- cues.csv
    cues = cue_table()
    if cues:
        write_csv("cues.csv", cues)

    # ---- stabiliser.csv
    st = stabiliser_histories()
    if st:
        write_csv("stabiliser.csv", st)
    print("done")


# -------------------------------------------------------------- W&B history files
def _wandb_records(path):
    """Yield the raw protobuf records of a .wandb file. The format is a 7-byte
    header, then 32 KiB blocks of chunks, each with a 7-byte header
    (crc32 u32, length u16, type u8; type 1 = whole record, 2/3/4 = first/middle/last)."""
    b = open(path, "rb").read()
    pos, buf, block = 7, b"", 32768
    while pos + 7 <= len(b):
        left = block - pos % block
        if left < 7:
            pos += left
            continue
        _, ln, typ = struct.unpack("<IHB", b[pos:pos + 7]); pos += 7
        if typ == 0 and ln == 0:
            pos += left - 7
            continue
        data = b[pos:pos + ln]; pos += ln
        if typ == 1:
            yield data
        elif typ == 2:
            buf = data
        elif typ == 3:
            buf += data
        elif typ == 4:
            yield buf + data; buf = b""


def stabiliser_histories():
    try:
        from wandb.proto import wandb_internal_pb2 as pb
    except ImportError:
        print("  stabiliser: wandb not installed - skipped")
        return []
    runs = {}
    for f in sorted(glob.glob("wandb/run-*/run-*.wandb")):
        cfg, hist = {}, []
        for d in _wandb_records(f):
            r = pb.Record(); r.ParseFromString(d)
            kind = r.WhichOneof("record_type")
            if kind in ("config", "run"):
                for it in (r.config.update if kind == "config" else r.run.config.update):
                    try:
                        cfg[it.key] = json.loads(it.value_json)
                    except ValueError:
                        pass
            elif kind == "history":
                hist.append({(it.key or "/".join(it.nested_key)): json.loads(it.value_json)
                             for it in r.history.item})
        name = cfg.get("candidate")
        name = name.get("value") if isinstance(name, dict) else name
        ep = [h for h in hist if "diag/weight_norm" in h and "epoch" in h]
        if not name or not ep:
            continue
        # keep the longest run per candidate; ties go to the latest
        if name not in runs or len(ep) >= len(runs[name][1]):
            runs[name] = (f, ep)
    out = []
    for name, (f, ep) in sorted(runs.items()):
        print(f"  stabiliser {name}: {len(ep)} epochs from {os.path.dirname(f)}")
        for h in ep:
            out.append(dict(candidate=name, epoch=int(h["epoch"]),
                            weight_norm=float(h["diag/weight_norm"]),
                            non_finite=int(h.get("non_finite_batches", 0))))
    return out


if __name__ == "__main__":
    main()
