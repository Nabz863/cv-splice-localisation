"""Draw every report and slide figure from results/tables/, and check each one.

This script reads nothing but the tables build_tables.py (and mechanism.py)
write, so every mark on every figure traces back to a results file. Each figure
is checked by overlap.py at output resolution before it is saved; a figure with
any collision is not written, and the script exits non-zero after listing all
of them.

Colour roles, fixed across every figure:
  blue    no spatial prior (beta = 0)
  orange  with the spatial prior
  aqua    a different score (the U-Net's max probability)
  gray    chance and other reference lines
The stabiliser figure has no prior in it, so it uses blue / aqua / violet,
validated as a set, with every line labelled directly.

Usage (from the repo root):
    python src/report/build_tables.py
    python src/report/make_figures.py          -> results/figures/*.pdf, *.png
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from overlap import save_checked

TAB = "results/tables"
OUT = os.environ.get("FIG_DIR", "results/figures")
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID, PALE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#c3c2b7"

# Figures are drawn at their printed size in the IEEE two-column layout, so a
# font size here is the size on the page (body text is 10 pt).
COL, FULL = 3.5, 7.16          # IEEEtran column and text widths, inches
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def table(name, required=True):
    path = os.path.join(TAB, name)
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"missing {path}: run src/report/build_tables.py first")
        return None
    return list(csv.DictReader(open(path)))


def num(r, k):
    return float(r[k]) if r.get(k, "") != "" else None


def grid_x(ax):
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def above_axes(ax):
    """x in data, y in axes fraction: for labels that sit just above the plot."""
    return mtrans.blended_transform_factory(ax.transData, ax.transAxes)


def swatch(c):
    return plt.Rectangle((0, 0), 1, 1, color=c)


# --------------------------------------------------------------------- 0. method
def fig_method():
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 6.15); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=0.9, gid="box"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, linespacing=1.15)

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(
            arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=8))

    tint = {"ela": ("#dce9f9", BLUE), "unet": ("#dce9f9", BLUE), "sam": ("#d6f1e6", AQUA),
            "mrf": ("#fbe2d7", ORANGE), "thr": ("#f2f1ee", MUTED)}
    rows = [(5.00, 1, "ELA\nJPEG re-compression", "ela", "threshold τ", "thr",
             "compression physics; 1 parameter"),
            (4.00, 2, "ELA\nJPEG re-compression", "ela", "MRF (τ, β)\nexact graph cut", "mrf",
             "+ spatial prior; 2 parameters"),
            (3.00, 3, "SAM2\nunprompted", "sam", "size filter", "thr",
             "general segmentation; no forensics"),
            (2.00, 4, "U-Net\nconstrained 1st layer", "unet", "threshold τ", "thr",
             "~1,093 labelled masks"),
            (1.00, 5, "U-Net\nconstrained 1st layer", "unet", "MRF (τ, β)\nexact graph cut", "mrf",
             "learned evidence + spatial prior")]
    h = 0.80
    for y, r, ft, fk, dt, dk, know in rows:
        ax.text(0.1, y + h / 2, f"Rung {r}", va="center", fontsize=7, weight="bold")
        box(1.3, y, 0.9, h, "image", "white", MUTED)
        arrow(2.2, 2.45, y + h / 2)
        box(2.45, y, 2.45, h, ft, *tint[fk], fs=7.4)
        arrow(4.9, 5.15, y + h / 2)
        box(5.15, y, 2.0, h, dt, *tint[dk])
        arrow(7.15, 7.7, y + h / 2)
        ax.text(7.78, y + h / 2, know, va="center", fontsize=7, color=INK2)
    for top, bot in ((5.00, 4.00), (2.00, 1.00)):          # the beta = 0 reductions
        x = 7.32
        ax.annotate("", xy=(x, top + 0.03), xytext=(x, bot + h - 0.03), arrowprops=dict(
            arrowstyle="<->", color=ORANGE, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(x + 0.1, (top + bot + h) / 2, "β = 0", va="center", fontsize=7, color=ORANGE)
    ax.text(7.78, 6.03, "where the knowledge comes from", fontsize=7, color=MUTED,
            style="italic", va="center")
    ax.text(0.1, 0.35, "Same image in, same binary mask out, same folds, same evaluation code. "
            "At β = 0, rung 2 reduces exactly to rung 1, and rung 5 to rung 4.",
            fontsize=7, color=INK2, va="center")
    return fig


# ------------------------------------------------------------------- 1. main F1
def fig_main():
    t = {r["key"]: r for r in table("main.csv")}
    chance = num(t["chance"], "f1")
    order = [("rung1", "Rung 1  ELA", BLUE), ("rung2", "Rung 2  ELA + MRF", ORANGE),
             ("rung3", "Rung 3  SAM2, zero-shot", INK2), ("rung4", "Rung 4  U-Net", BLUE),
             ("rung5", "Rung 5  U-Net + MRF", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(order))[::-1]
    xmax = max(num(t[k], "f1") + (num(t[k], "f1_sd") or 0) for k, *_ in order if k in t)
    for yi, (k, lab, c) in zip(y, order):
        if k not in t:
            ax.text(0.01, yi, "pending", va="center", color=MUTED, style="italic")
            continue
        v, sd = num(t[k], "f1"), num(t[k], "f1_sd")
        ax.barh(yi, v, height=0.6, color=c, edgecolor="white", lw=2)
        if sd:
            ax.errorbar(v, yi, xerr=sd, color=INK, lw=0.9, capsize=2.5)
        ax.text(v + (sd or 0) + 0.015, yi, f"{v:.3f}", va="center")
    ax.axvline(chance, color=MUTED, ls=(0, (3, 2)), lw=1)
    ax.text(chance, 1.0, f"chance {chance:.3f}", transform=above_axes(ax), ha="center",
            va="bottom", color=INK2, fontsize=7)
    ax.set_yticks(y, [o[1] for o in order]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, xmax * 1.16); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(f"Pixel F1 (micro) over {t['rung2']['n_pairs']} nested pairs; "
                  "bars ±1 SD across test folds")
    grid_x(ax)
    ax.legend([swatch(BLUE), swatch(ORANGE)], ["no spatial prior (β = 0)", "with spatial prior"],
              loc="center right", fontsize=7, handlelength=1, handleheight=1)
    return fig


# ------------------------------------------------------------- 2. prior's gain
def fig_gain():
    g = table("gains.csv"); t = {r["key"]: r for r in table("main.csv")}
    by = defaultdict(list)
    for r in g:
        by[r["unary"]].append(float(r["gain"]))
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    labels = []
    for x, (u, key, desc) in enumerate((("ELA", "rung2", "hand-crafted"),
                                        ("U-Net", "rung5", "learned"))):
        v = np.array(by[u])
        jit = ((np.arange(len(v)) * 7) % len(v) / max(len(v) - 1, 1) - 0.5) * 0.3
        ax.plot(x + jit, v, "o", ms=4.5, color=ORANGE, alpha=0.75, mec="white", mew=0.6)
        m = num(t[key], "f1") - num(t["rung1" if key == "rung2" else "rung4"], "f1")
        ax.plot([x - 0.24, x + 0.24], [m, m], color=INK, lw=1.6)
        ax.text(x + 0.27, m, f"mean +{m:.3f}", va="center", fontsize=7)
        b0 = t[key]["beta0_selected"]
        labels.append(f"over {u}\n({desc})\nβ = 0 chosen {b0}/{len(v)}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([0, 1], labels); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 1.75)
    lo = min(min(v) for v in by.values()); hi = max(max(v) for v in by.values())
    ax.set_ylim(min(lo * 1.3, -0.004), hi * 1.12)
    ax.set_ylabel("F1 added by the MRF prior\n(one dot per nested pair)")
    grid_y(ax)
    return fig


# ----------------------------------------------------------------- 3. data size
def fig_datasize():
    d = table("datasize.csv", required=False); t = {r["key"]: r for r in table("main.csv")}
    if not d:
        return None
    n = np.array([int(r["n_train"]) for r in d])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for key, sdk, c, lab, below in (("f1", "f1_sd", BLUE, "U-Net", True),
                                    ("f1_prior", "f1_prior_sd", ORANGE, "U-Net + MRF", False)):
        f = np.array([float(r[key]) for r in d]); sd = np.array([float(r[sdk]) for r in d])
        ax.fill_between(n, f - sd, f + sd, color=c, alpha=0.12, lw=0)
        ax.plot(n, f, color=c, lw=2, marker="o", ms=4.5, mec="white", mew=1.0, zorder=3,
                label=lab)
        for i, (x, v) in enumerate(zip(n, f)):     # label the ends only: the right-hand
            if i not in (0, len(n) - 1):           # points sit too close on a log axis
                continue
            if below:                              # below the lower line, above the upper
                ax.text(x, v - sd[i] - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=7,
                        color=c)
            else:
                ax.text(x, v + sd[i] + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                        color=c)
    for key, lab, ls in (("rung2", "ELA + MRF", (0, (3, 2))), ("chance", "chance", (0, (1, 2)))):
        v = num(t[key], "f1")
        ax.axhline(v, color=MUTED, ls=ls, lw=1)
        ax.text(n[-1] * 1.4, v + 0.008, f"{lab} {v:.3f}", fontsize=7, color=INK2, va="bottom",
                ha="right")
    ax.set_xscale("log"); ax.set_xticks(n, [str(v) for v in n]); ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7)
    lab = ax.get_xticklabels()                   # the last two ticks are close on a
    lab[-2].set_ha("right"); lab[-1].set_ha("left")   # log axis: push them apart
    hi = max(float(r["f1_prior"]) + float(r["f1_prior_sd"]) for r in d)
    ax.set_xlim(n[0] * 0.8, n[-1] * 1.5); ax.set_ylim(0.18, hi + 0.12)
    ax.set_xlabel("Labelled training masks (log scale)")
    ax.set_ylabel("Pixel F1 (±1 SD across folds)")
    ax.legend(loc="upper left", fontsize=7)
    grid_y(ax)
    return fig


# --------------------------------------------------------------- 4. categories
def fig_category():
    d = sorted(table("category.csv"), key=lambda r: float(r["pooled_frac"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(d))[::-1]
    split = None
    for yi, r in zip(y, d):
        ch, e, m = float(r["chance"]), float(r["ela"]), float(r["mrf"])
        ax.plot([e, m], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        ax.plot(ch, yi, marker="|", ms=13, mew=1.8, color=MUTED, ls="", zorder=2)
        ax.plot(m, yi, "o", ms=8, color=ORANGE, mec="white", mew=1.2, ls="", zorder=3)
        ax.plot(e, yi, "o", ms=5.5, color=BLUE, mec="white", mew=1.0, ls="", zorder=4)
        if split is None and float(r["pooled_frac"]) > 0.15:
            split = yi + 0.5
    ax.set_yticks(y, [f"{r['category']}  (n={r['n']}, {float(r['pooled_frac'])*100:.0f}% tampered)"
                      for r in d])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    if split is not None:
        ax.axhline(split, color=PALE, lw=0.8)
    xs = [float(r[k]) for r in d for k in ("chance", "ela", "mrf")]
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.04); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Pixel F1 (micro), τ = 0.75, β = 8; sorted by share of pixels tampered")
    grid_x(ax)
    h = [plt.Line2D([], [], marker="|", ls="", ms=11, mew=1.8, color=MUTED),
         plt.Line2D([], [], marker="o", ls="", ms=6, color=BLUE),
         plt.Line2D([], [], marker="o", ls="", ms=7, color=ORANGE)]
    ax.legend(h, ["own chance line", "ELA", "ELA + MRF"], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7)
    return fig


# ------------------------------------------------------------- 5. image AUC
def fig_auc():
    rows = [r for r in table("auc.csv") if r["subset"] == "all"]
    cv = any(r["tag"] == "all20" for r in rows)
    if cv:   # all 20 pairs cached: show the cross-validated rows, interval = +-1 SD over folds
        rows = [dict(r) for r in rows if r["tag"] in ("all", "all20")]
        for r in rows:
            if r["tag"] == "all20":
                a, sd = float(r["auc"]), float(r["auc_sd"])
                r["ci_lo"], r["ci_hi"] = a - sd, a + sd
    groups = [("ELA  (area)", "1", "area", BLUE), ("ELA + MRF  (area)", "2", "area", ORANGE),
              ("U-Net  (area)", "4", "area", BLUE), ("U-Net + MRF  (area)", "5", "area", ORANGE)]
    sel = [r for r in rows if r["selected_on_heldout"] == "1"]
    if sel:
        name = "held-out choice" if cv else sel[0]["score"].replace("_", " ") + ", held-out"
        groups.append((f"U-Net  ({name})", "4", sel[0]["score"], AQUA))
    tags = [] if cv else sorted({r["tag"] for r in rows if r["tag"] != "all"})
    marks = {"all": "o", "all20": "o", **{tg: m for tg, m in zip(tags, "os^")}}
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(groups))[::-1].astype(float)
    for yi, (lab, rung, score, c) in zip(y, groups):
        pts = [r for r in rows if r["rung"] == rung and r["score"] == score]
        offs = [0.0] if len(pts) == 1 else np.linspace(0.15, -0.15, len(pts))
        for r, o in zip(pts, offs):
            ax.plot([float(r["ci_lo"]), float(r["ci_hi"])], [yi + o] * 2, color=c, lw=1.6,
                    solid_capstyle="round")
            ax.plot(float(r["auc"]), yi + o, marks[r["tag"]], color=c, ms=6, mec="white", mew=1.2)
        if c == AQUA:
            ax.text(min(float(r["ci_lo"]) for r in pts) - 0.012, yi,
                    " / ".join(f"{float(r['auc']):.3f}" for r in pts), ha="right", va="center",
                    fontsize=7)
    ax.axvline(0.5, color=MUTED, ls=(0, (3, 2)), lw=1)
    ax.text(0.5, 1.0, "chance", transform=above_axes(ax), ha="center", va="bottom",
            color=INK2, fontsize=7)
    ax.set_yticks(y, [g[0] for g in groups]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    lo = min(float(r["ci_lo"]) for r in rows)
    ax.set_xlim(min(lo - 0.03, 0.45), 1.0); ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("Image-level AUC, tampered vs. authentic ("
                  + ("classical: 95% bootstrap CI; U-Net: \u00b11 SD over test folds)" if cv
                     else "95% bootstrap CI)"))
    grid_x(ax)
    if tags:
        h = [plt.Line2D([], [], marker=marks[tg], ls="", ms=5.5, color=INK2) for tg in tags]
        ax.legend(h, [f"pair {tg}" for tg in tags], loc="upper right", fontsize=7)
    return fig


# --------------------------------------------------------------- 6. copy-move
def fig_copymove():
    d = table("copymove.csv")
    sets = list(dict.fromkeys(r["dataset"] for r in d))
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    w = 0.34
    for i, s in enumerate(sets):
        for method, c, dx in (("ELA", BLUE, -w / 2), ("ELA + MRF", ORANGE, w / 2)):
            r = next(r for r in d if r["dataset"] == s and r["method"] == method)
            v, lo, hi = float(r["margin"]), float(r["ci_lo"]), float(r["ci_hi"])
            ax.bar(i + dx, v, width=w, color=c, edgecolor="white", lw=2)
            ax.errorbar(i + dx, v, yerr=[[v - lo], [hi - v]], color=INK, lw=0.9, capsize=2.5)
            ax.text(i + dx, hi + 0.002, f"{v:+.3f}", ha="center", va="bottom", fontsize=7)
    n = {r["dataset"]: r["n"] for r in d}
    ax.set_xticks(range(len(sets)), [f"{s.capitalize()}\n({int(n[s]):,} images)" for s in sets])
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color=INK2, lw=0.8)
    top = max(float(r["ci_hi"]) for r in d)
    ax.set_ylim(min(0, min(float(r["ci_lo"]) for r in d)) - 0.002, top * 1.45)
    ax.set_ylabel("F1 above own chance line\n(95% bootstrap CI)")
    grid_y(ax)
    ax.legend([swatch(BLUE), swatch(ORANGE)], ["ELA", "ELA + MRF"], loc="upper left", ncol=2,
              fontsize=7, handlelength=1, handleheight=1)
    return fig


# -------------------------------------------------------------- 7. stabilisers
def fig_stabiliser():
    """Weight norm per epoch, with each run's first non-finite loss marked.

    Deliberately no fp16-maximum line: 65,504 is the largest single value fp16
    holds, while this is an L2 norm over every weight, so a line at 65,504 would
    imply a threshold the data does not show (both divergences began below it)."""
    d = table("stabiliser.csv", required=False)
    if not d:
        return None
    by = defaultdict(list)
    for r in d:
        by[r["candidate"]].append(r)
    colours = dict(zip(sorted(by), (BLUE, AQUA, VIOLET)))
    names = {"baseline": "no stabiliser", "clip": "gradient clipping", "wd1e-3": "weight decay 1e-3"}
    fig, ax = plt.subplots(figsize=(FULL - 0.95, 2.3))   # end labels overhang the axes
    ends = []
    for cand, rs in sorted(by.items()):
        ep = np.array([int(r["epoch"]) for r in rs]); wn = np.array([float(r["weight_norm"]) for r in rs])
        ax.plot(ep, wn, color=colours[cand], lw=1.8)
        bad = [i for i, r in enumerate(rs) if int(r["non_finite"]) > 0]
        if bad:
            i = bad[0]
            ax.plot(ep[i], wn[i], "X", ms=8, color=colours[cand], mec="white", mew=0.8, zorder=4)
            tag = f"non-finite from ep {ep[i]}\nat norm {wn[i]:,.0f}"
        else:
            j = int(wn.argmax())
            tag = f"peak {wn[j]:,.0f} (ep {ep[j]}),\nnever non-finite"
        ends.append([wn[-1], f"{names.get(cand, cand)}: {tag}", ep[-1]])
    all_wn = [float(r["weight_norm"]) for r in d]
    ax.set_yscale("log")
    ax.set_ylim(min(all_wn) * 0.6, max(all_wn) * 2.5)
    # direct labels at the right edge; three-line labels need a minimum gap,
    # converted from points to decades using the axes' real height.
    fig.canvas.draw()
    h_pt = ax.get_window_extent().height * 72 / fig.dpi
    decades = np.log10(ax.get_ylim()[1] / ax.get_ylim()[0])
    min_gap = 3 * 7.2 * 1.3 / (h_pt / decades)
    ends.sort()
    ypos = []
    for wn, *_ in ends:
        v = np.log10(wn)
        if ypos and v - ypos[-1] < min_gap:
            v = ypos[-1] + min_gap
        ypos.append(v)
    for (wn, text, last), v in zip(ends, ypos):
        ax.text(last + 2, 10 ** v, text, va="center", fontsize=7, color=INK)
    ax.set_xlim(0, max(e[2] for e in ends) + 2)
    ax.set_xlabel("Epoch (400 optimiser steps each; fp16 autocast)")
    ax.set_ylabel("Weight norm, all parameters (log)")
    grid_y(ax)
    ax.plot([], [], "X", color=INK2, ms=7, ls="", label="first non-finite loss")
    ax.legend(loc="lower right", fontsize=7)
    return fig


# -------------------------------------------------------------- 8. mechanism
def fig_mechanism():
    d = table("mechanism.csv", required=False)
    if not d:
        return None
    k = 1e3
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    y = np.arange(len(d))[::-1]
    v = {key: np.array([float(r[key]) for r in d]) / k
         for key in ("fp_removed", "fn_filled", "tp_lost", "fp_added")}
    ax.barh(y, v["fp_removed"], height=0.58, color=ORANGE, edgecolor="white", lw=2,
            label="false positives removed")
    ax.barh(y, v["fn_filled"], left=v["fp_removed"], height=0.58, color=AQUA, edgecolor="white",
            lw=2, label="missed pixels filled")
    ax.barh(y, -v["tp_lost"], height=0.58, color=MUTED, edgecolor="white", lw=2,
            label="correct pixels lost")
    ax.barh(y, -v["fp_added"], left=-v["tp_lost"], height=0.58, color=PALE, edgecolor="white",
            lw=2, label="false positives added")
    gain = v["fp_removed"] + v["fn_filled"]
    for yi, gv, r in zip(y, gain, d):
        ax.text(gv + gain.max() * 0.02, yi, f"net {float(r['share']):.0%}", va="center", fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r["band"].replace("<=", "≤").replace("-", "–") for r in d])
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("distance to true boundary")
    ax.set_xlim(-max((v["tp_lost"] + v["fp_added"]).max(), gain.max() * 0.05) * 1.3,
                gain.max() * 1.22)
    ax.set_xlabel("pixels the prior changes (thousands); losses left of zero, gains right")
    grid_x(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7,
              handlelength=1, handleheight=1)
    return fig


# ------------------------------------------------------------ 9. per-image F1
def fig_perimage():
    d = table("perimage_f1.csv", required=False)
    if not d:
        return None
    f4 = np.array([float(r["f1_rung4"]) for r in d]); f5 = np.array([float(r["f1_rung5"]) for r in d])
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(f4, bins=bins, histtype="step", lw=2, color=BLUE, label="U-Net (β = 0)")
    ax.hist(f5, bins=bins, histtype="step", lw=2, color=ORANGE, label="U-Net + MRF")
    n_img = len({r["index"] for r in d}); n_mod = len({r["tag"] for r in d})
    ax.set_xlabel(f"per-image F1: {n_img} fold-0 test images × {n_mod} model(s)")
    ax.set_ylabel("images")
    top = max(np.histogram(f4, bins)[0].max(), np.histogram(f5, bins)[0].max())
    ax.set_ylim(0, top * 1.35)
    grid_y(ax)
    ax.legend(loc="upper center", ncol=2, fontsize=7)
    return fig


FIGURES = [("fig0_method", fig_method), ("fig1_main_f1", fig_main), ("fig2_prior_gain", fig_gain),
           ("fig3_datasize", fig_datasize), ("fig4_per_category", fig_category),
           ("fig5_image_auc", fig_auc), ("fig6_copymove", fig_copymove),
           ("fig7_stabiliser", fig_stabiliser), ("fig8_mechanism", fig_mechanism),
           ("fig9_perimage_f1", fig_perimage)]


def main():
    os.makedirs(OUT, exist_ok=True)
    failed = []
    for name, make in FIGURES:
        fig = make()
        if fig is None:
            print(f"  {name}: skipped (its table is not there yet)")
            continue
        try:
            save_checked(fig, os.path.join(OUT, name))
            print(f"  {name}: ok")
        except RuntimeError as e:
            failed.append(str(e))
            print(f"  {name}: NOT SAVED")
        plt.close(fig)
    if failed:
        print("\n" + "\n\n".join(failed))
        raise SystemExit(f"{len(failed)} figure(s) have collisions")
    print(f"all figures written to {OUT}/")


if __name__ == "__main__":
    main()
