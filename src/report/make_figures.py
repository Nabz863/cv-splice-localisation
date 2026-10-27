"""Draw every figure from results/tables/ (written by build_tables.py) into results/figures/.

Each figure is checked for overlapping labels (overlap.py) before saving.
Colours: blue = no prior (beta 0), orange = with prior, aqua = other score, grey = chance/reference.
Usage:  python src/report/make_figures.py
"""
import csv, os, sys                                     # standard-library helpers
from collections import defaultdict                     # dict with default values
import numpy as np                                      # arrays and maths
import matplotlib                                       # plotting library
matplotlib.use("Agg")                                   # draw to files, no screen needed
import matplotlib.pyplot as plt                         # plotting
import matplotlib.transforms as mtrans                  # coordinate systems (for labels above a plot)
from matplotlib.patches import FancyBboxPatch           # rounded boxes for diagrams

sys.path.insert(0, os.path.dirname(__file__))           # so overlap.py can be imported
from overlap import save_checked                        # saves a figure only if nothing overlaps

TAB = "results/tables"                                  # tables written by build_tables.py
OUT = os.environ.get("FIG_DIR", "results/figures")      # output folder (can be changed with FIG_DIR)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"  # series colours
INK, INK2, MUTED, GRID, PALE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#c3c2b7"  # text, lines and background greys

# Figures are drawn at their printed size in the IEEE two-column layout, so a
# font size here is the size on the page (body text is 10 pt).
COL, FULL = 3.5, 7.16          # IEEEtran column and text widths, inches
plt.rcParams.update({                                   # default styles for every figure
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


# ---- Small helpers ----
def table(name, required=True):
    """Rows of results/tables/<name> (None if missing and not required)."""
    path = os.path.join(TAB, name)                      # full path
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"missing {path}: run src/report/build_tables.py first")  # stop with a hint
        return None
    return list(csv.DictReader(open(path)))             # one dict per row


def num(r, k):
    """Field k of row r as a float (None if blank)."""
    return float(r[k]) if r.get(k, "") != "" else None  # blank cells become None


def grid_x(ax):
    """Light vertical grid lines."""
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)  # grid behind the data


def grid_y(ax):
    """Light horizontal grid lines."""
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)  # grid behind the data


def above_axes(ax):
    """x in data, y in axes fraction: for labels that sit just above the plot."""
    return mtrans.blended_transform_factory(ax.transData, ax.transAxes)  # mixed coordinates


def swatch(c):
    """Legend handle: a coloured square."""
    return plt.Rectangle((0, 0), 1, 1, color=c)         # a 1x1 square


# --------------------------------------------------------------------- 0. method
def fig_method():
    """Fig. 1: the five rungs and where their knowledge comes from."""
    fig, ax = plt.subplots(figsize=(FULL, 2.9))         # full page width, 2.9 in tall
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 6.15); ax.axis("off")  # coordinate range; no axes drawn

    def box(x, y, w, h, text, fc, ec, fs=7.4):          # rounded box with centred text
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",  # gid="box" lets overlap.py check it
                                    fc=fc, ec=ec, lw=0.9, gid="box"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,  # centred text
                color=INK, linespacing=1.15)

    def arrow(x0, x1, y):                               # horizontal arrow from x0 to x1
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(
            arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=8))

    tint = {"ela": ("#dce9f9", BLUE), "unet": ("#dce9f9", BLUE), "sam": ("#d6f1e6", AQUA),  # (fill, edge) colour per kind of box
            "mrf": ("#fbe2d7", ORANGE), "thr": ("#f2f1ee", MUTED)}
    rows = [(5.00, 1, "ELA\nJPEG re-compression", "ela", "threshold τ", "thr",  # (y, rung, first box, its kind, second box, its kind, note on the right)
             "compression physics; 1 parameter"),
            (4.00, 2, "ELA\nJPEG re-compression", "ela", "MRF (τ, β)\nexact graph cut", "mrf",
             "+ spatial prior; 2 parameters"),
            (3.00, 3, "SAM2\nunprompted", "sam", "size filter", "thr",
             "general segmentation; no forensics"),
            (2.00, 4, "U-Net\nconstrained 1st layer", "unet", "threshold τ", "thr",
             "~1,093 labelled masks"),
            (1.00, 5, "U-Net\nconstrained 1st layer", "unet", "MRF (τ, β)\nexact graph cut", "mrf",
             "learned evidence + spatial prior")]
    h = 0.80                                            # box height
    for y, r, ft, fk, dt, dk, know in rows:             # one row per rung
        ax.text(0.1, y + h / 2, f"Rung {r}", va="center", fontsize=7, weight="bold")  # rung number at the left
        box(1.3, y, 0.9, h, "image", "white", MUTED)    # input box
        arrow(2.2, 2.45, y + h / 2)                     # arrow
        box(2.45, y, 2.45, h, ft, *tint[fk], fs=7.4)    # evidence box
        arrow(4.9, 5.15, y + h / 2)                     # arrow
        box(5.15, y, 2.0, h, dt, *tint[dk])             # decision box
        arrow(7.15, 7.7, y + h / 2)                     # arrow
        ax.text(7.78, y + h / 2, know, va="center", fontsize=7, color=INK2)  # note
    for top, bot in ((5.00, 4.00), (2.00, 1.00)):          # the beta = 0 reductions
        x = 7.32                                        # x position of the double arrow
        ax.annotate("", xy=(x, top + 0.03), xytext=(x, bot + h - 0.03), arrowprops=dict(  # double arrow between the two rows
            arrowstyle="<->", color=ORANGE, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(x + 0.1, (top + bot + h) / 2, "β = 0", va="center", fontsize=7, color=ORANGE)  # its label
    ax.text(7.78, 6.03, "where the knowledge comes from", fontsize=7, color=MUTED,  # column heading over the notes
            style="italic", va="center")
    ax.text(0.1, 0.35, "Same image in, same binary mask out, same folds, same evaluation code. "  # caption line at the bottom
            "At β = 0, rung 2 reduces exactly to rung 1, and rung 5 to rung 4.",
            fontsize=7, color=INK2, va="center")
    return fig


# ------------------------------------------------------------------- 1. main F1
def fig_main():
    """Main F1 bar chart (slides; Table I in the report)."""
    t = {r["key"]: r for r in table("main.csv")}        # key -> row of Table I
    chance = num(t["chance"], "f1")                     # chance F1
    order = [("rung1", "Rung 1  ELA", BLUE), ("rung2", "Rung 2  ELA + MRF", ORANGE),  # (key, label, colour) per bar, top to bottom
             ("rung3", "Rung 3  SAM2, zero-shot", INK2), ("rung4", "Rung 4  U-Net", BLUE),
             ("rung5", "Rung 5  U-Net + MRF", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))          # figure size in inches
    y = np.arange(len(order))[::-1]                     # y positions, top to bottom
    xmax = max(num(t[k], "f1") + (num(t[k], "f1_sd") or 0) for k, *_ in order if k in t)  # right end of the longest bar + error bar
    for yi, (k, lab, c) in zip(y, order):               # one bar per rung
        if k not in t:                                  # rung missing from the table
            ax.text(0.01, yi, "pending", va="center", color=MUTED, style="italic")
            continue
        v, sd = num(t[k], "f1"), num(t[k], "f1_sd")     # F1 and its SD
        ax.barh(yi, v, height=0.6, color=c, edgecolor="white", lw=2)  # horizontal bar
        if sd:                                          # error bar if there is an SD
            ax.errorbar(v, yi, xerr=sd, color=INK, lw=0.9, capsize=2.5)
        ax.text(v + (sd or 0) + 0.015, yi, f"{v:.3f}", va="center")  # value label just past the error bar
    ax.axvline(chance, color=MUTED, ls=(0, (3, 2)), lw=1)  # dashed chance line
    ax.text(chance, 1.0, f"chance {chance:.3f}", transform=above_axes(ax), ha="center",  # its label above the plot
            va="bottom", color=INK2, fontsize=7)
    ax.set_yticks(y, [o[1] for o in order]); ax.tick_params(axis="y", length=0)  # rung names as y labels; no tick marks
    ax.spines["left"].set_visible(False)                # hide the left axis line
    ax.set_xlim(0, xmax * 1.16); ax.set_ylim(-0.6, len(order) - 0.4)  # room for the value labels
    ax.set_xlabel(f"Pixel F1 (micro) over {t['rung2']['n_pairs']} nested pairs; "  # x-axis label
                  "bars ±1 SD across test folds")
    grid_x(ax)                                          # grid lines
    ax.legend([swatch(BLUE), swatch(ORANGE)], ["no spatial prior (β = 0)", "with spatial prior"],  # colour key
              loc="center right", fontsize=7, handlelength=1, handleheight=1)
    return fig


# ------------------------------------------------------------- 2. prior's gain
def fig_gain():
    """Fig. 4: F1 added by the prior in each of the 20 pairs."""
    g = table("gains.csv"); t = {r["key"]: r for r in table("main.csv")}  # per-pair gains; Table I
    by = defaultdict(list)                              # unary -> list of gains
    for r in g:
        by[r["unary"]].append(float(r["gain"]))         # group gains by unary
    fig, ax = plt.subplots(figsize=(COL, 2.5))          # one-column figure
    labels = []                                         # x-axis labels
    for x, (u, key, desc) in enumerate((("ELA", "rung2", "hand-crafted"),  # x = 0 for ELA, 1 for U-Net
                                        ("U-Net", "rung5", "learned"))):
        v = np.array(by[u])                             # its 20 gains
        jit = ((np.arange(len(v)) * 7) % len(v) / max(len(v) - 1, 1) - 0.5) * 0.3  # small fixed sideways offsets so dots don't sit on top of each other
        ax.plot(x + jit, v, "o", ms=4.5, color=ORANGE, alpha=0.75, mec="white", mew=0.6)  # one dot per pair
        m = num(t[key], "f1") - num(t["rung1" if key == "rung2" else "rung4"], "f1")  # mean gain from Table I (rung with prior minus rung without)
        ax.plot([x - 0.24, x + 0.24], [m, m], color=INK, lw=1.6)  # short horizontal line at the mean
        ax.text(x + 0.27, m, f"mean +{m:.3f}", va="center", fontsize=7)  # its label
        b0 = t[key]["beta0_selected"]                   # how often beta = 0 was chosen
        labels.append(f"over {u}\n({desc})\nβ = 0 chosen {b0}/{len(v)}")  # x label for this group
    ax.axhline(0, color=INK2, lw=0.8)                   # zero line
    ax.set_xticks([0, 1], labels); ax.tick_params(axis="x", length=0)  # group labels
    ax.set_xlim(-0.45, 1.75)                            # room for the mean labels
    lo = min(min(v) for v in by.values()); hi = max(max(v) for v in by.values())  # smallest and largest gain
    ax.set_ylim(min(lo * 1.3, -0.004), hi * 1.12)       # y range with a little margin
    ax.set_ylabel("F1 added by the MRF prior\n(one dot per nested pair)")  # y-axis label
    grid_y(ax)
    return fig


# ----------------------------------------------------------------- 3. data size
def fig_datasize():
    """Fig. 8: rungs 4-5 F1 vs number of training masks."""
    d = table("datasize.csv", required=False); t = {r["key"]: r for r in table("main.csv")}  # datasize.csv may be missing
    if not d:                                           # skip this figure
        return None
    n = np.array([int(r["n_train"]) for r in d])        # training-set sizes
    fig, ax = plt.subplots(figsize=(COL, 2.6))          # one-column figure
    for key, sdk, c, lab, below in (("f1", "f1_sd", BLUE, "U-Net", True),  # U-Net, then U-Net + MRF
                                    ("f1_prior", "f1_prior_sd", ORANGE, "U-Net + MRF", False)):
        f = np.array([float(r[key]) for r in d]); sd = np.array([float(r[sdk]) for r in d])  # F1 and SD at each size
        ax.fill_between(n, f - sd, f + sd, color=c, alpha=0.12, lw=0)  # shaded band = +-1 SD
        ax.plot(n, f, color=c, lw=2, marker="o", ms=4.5, mec="white", mew=1.0, zorder=3,  # line with dots
                label=lab)
        for i, (x, v) in enumerate(zip(n, f)):     # label the ends only: the right-hand
            if i not in (0, len(n) - 1):           # points sit too close on a log axis
                continue
            if below:                              # below the lower line, above the upper
                ax.text(x, v - sd[i] - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=7,  # label below the band
                        color=c)
            else:
                ax.text(x, v + sd[i] + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7,  # label above the band
                        color=c)
    for key, lab, ls in (("rung2", "ELA + MRF", (0, (3, 2))), ("chance", "chance", (0, (1, 2)))):  # two reference lines: rung 2 and chance
        v = num(t[key], "f1")                           # its F1
        ax.axhline(v, color=MUTED, ls=ls, lw=1)         # horizontal line
        ax.text(n[-1] * 1.4, v + 0.008, f"{lab} {v:.3f}", fontsize=7, color=INK2, va="bottom",  # its label at the right end
                ha="right")
    ax.set_xscale("log"); ax.set_xticks(n, [str(v) for v in n]); ax.minorticks_off()  # log x axis with a tick at each size
    ax.tick_params(axis="x", labelsize=7)               # smaller tick labels
    lab = ax.get_xticklabels()                   # the last two ticks are close on a
    lab[-2].set_ha("right"); lab[-1].set_ha("left")   # log axis: push them apart
    hi = max(float(r["f1_prior"]) + float(r["f1_prior_sd"]) for r in d)  # top of the highest band
    ax.set_xlim(n[0] * 0.8, n[-1] * 1.5); ax.set_ylim(0.18, hi + 0.12)  # axis ranges
    ax.set_xlabel("Labelled training masks (log scale)")  # x-axis label
    ax.set_ylabel("Pixel F1 (±1 SD across folds)")      # y-axis label
    ax.legend(loc="upper left", fontsize=7)             # legend
    grid_y(ax)
    return fig


# --------------------------------------------------------------- 4. categories
def fig_category():
    """Per-category F1 (not in the report; Sec. V-F text)."""
    d = sorted(table("category.csv"), key=lambda r: float(r["pooled_frac"]))  # categories, lowest tampered share first
    fig, ax = plt.subplots(figsize=(6.4, 3.4))          # figure size
    y = np.arange(len(d))[::-1]                         # y positions, top to bottom
    split = None                                        # y where the line between small and large tampered shares goes
    for yi, r in zip(y, d):                             # one row per category
        ch, e, m = float(r["chance"]), float(r["ela"]), float(r["mrf"])  # chance, rung 1 and rung 2 F1
        ax.plot([e, m], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)  # grey bar from rung 1 to rung 2
        ax.plot(ch, yi, marker="|", ms=13, mew=1.8, color=MUTED, ls="", zorder=2)  # chance tick
        ax.plot(m, yi, "o", ms=8, color=ORANGE, mec="white", mew=1.2, ls="", zorder=3)  # rung 2 dot
        ax.plot(e, yi, "o", ms=5.5, color=BLUE, mec="white", mew=1.0, ls="", zorder=4)  # rung 1 dot
        if split is None and float(r["pooled_frac"]) > 0.15:  # first category above 15% tampered
            split = yi + 0.5                            # put the line just above it
    ax.set_yticks(y, [f"{r['category']}  (n={r['n']}, {float(r['pooled_frac'])*100:.0f}% tampered)"  # row labels with size and tampered share
                      for r in d])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)  # no tick marks; no left axis line
    if split is not None:
        ax.axhline(split, color=PALE, lw=0.8)           # thin dividing line
    xs = [float(r[k]) for r in d for k in ("chance", "ela", "mrf")]  # every plotted value
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.04); ax.set_ylim(-0.6, len(d) - 0.4)  # axis ranges
    ax.set_xlabel("Pixel F1 (micro), τ = 0.75, β = 8; sorted by share of pixels tampered")  # x-axis label
    grid_x(ax)
    h = [plt.Line2D([], [], marker="|", ls="", ms=11, mew=1.8, color=MUTED),  # legend symbols
         plt.Line2D([], [], marker="o", ls="", ms=6, color=BLUE),
         plt.Line2D([], [], marker="o", ls="", ms=7, color=ORANGE)]
    ax.legend(h, ["own chance line", "ELA", "ELA + MRF"], loc="lower left",  # legend above the plot
              bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7)
    return fig


# ------------------------------------------------------------- 5. image AUC
def fig_auc():
    """Image-level AUC chart (Table II in the report)."""
    rows = [r for r in table("auc.csv") if r["subset"] == "all"]  # all-images rows only (not the JPEG subset)
    cv = any(r["tag"] == "all20" for r in rows)         # cross-validated rows present?
    if cv:   # all 20 pairs cached: show the cross-validated rows, interval = +-1 SD over folds
        rows = [dict(r) for r in rows if r["tag"] in ("all", "all20")]  # keep the classical rows and the 20-pair summaries
        for r in rows:
            if r["tag"] == "all20":                     # 20-pair summary rows ...
                a, sd = float(r["auc"]), float(r["auc_sd"])  # ... draw +-1 SD instead of a bootstrap interval
                r["ci_lo"], r["ci_hi"] = a - sd, a + sd
    groups = [("ELA  (area)", "1", "area", BLUE), ("ELA + MRF  (area)", "2", "area", ORANGE),  # (label, rung, score, colour) per row
              ("U-Net  (area)", "4", "area", BLUE), ("U-Net + MRF  (area)", "5", "area", ORANGE)]
    sel = [r for r in rows if r["selected_on_heldout"] == "1"]  # the score chosen on held-out data
    if sel:
        name = "held-out choice" if cv else sel[0]["score"].replace("_", " ") + ", held-out"  # its row label
        groups.append((f"U-Net  ({name})", "4", sel[0]["score"], AQUA))  # extra row for it
    tags = [] if cv else sorted({r["tag"] for r in rows if r["tag"] != "all"})  # single pairs shown separately (only without 20-pair summaries)
    marks = {"all": "o", "all20": "o", **{tg: m for tg, m in zip(tags, "os^")}}  # marker per tag
    fig, ax = plt.subplots(figsize=(6.4, 2.9))          # figure size
    y = np.arange(len(groups))[::-1].astype(float)      # y positions, top to bottom
    for yi, (lab, rung, score, c) in zip(y, groups):    # one row per group
        pts = [r for r in rows if r["rung"] == rung and r["score"] == score]  # matching table rows
        offs = [0.0] if len(pts) == 1 else np.linspace(0.15, -0.15, len(pts))  # small vertical offsets if there are several
        for r, o in zip(pts, offs):
            ax.plot([float(r["ci_lo"]), float(r["ci_hi"])], [yi + o] * 2, color=c, lw=1.6,  # interval line
                    solid_capstyle="round")
            ax.plot(float(r["auc"]), yi + o, marks[r["tag"]], color=c, ms=6, mec="white", mew=1.2)  # AUC dot
        if c == AQUA:                                   # the held-out choice row ...
            ax.text(min(float(r["ci_lo"]) for r in pts) - 0.012, yi,  # ... gets its AUC written at the left
                    " / ".join(f"{float(r['auc']):.3f}" for r in pts), ha="right", va="center",
                    fontsize=7)
    ax.axvline(0.5, color=MUTED, ls=(0, (3, 2)), lw=1)  # dashed line at 0.5 (chance)
    ax.text(0.5, 1.0, "chance", transform=above_axes(ax), ha="center", va="bottom",  # its label above the plot
            color=INK2, fontsize=7)
    ax.set_yticks(y, [g[0] for g in groups]); ax.tick_params(axis="y", length=0)  # row labels; no tick marks
    ax.spines["left"].set_visible(False)                # hide the left axis line
    lo = min(float(r["ci_lo"]) for r in rows)           # lowest interval end
    ax.set_xlim(min(lo - 0.03, 0.45), 1.0); ax.set_ylim(-0.6, len(groups) - 0.4)  # axis ranges
    ax.set_xlabel("Image-level AUC, tampered vs. authentic ("  # x-axis label depends on the mode
                  + ("classical: 95% bootstrap CI; U-Net: \u00b11 SD over test folds)" if cv
                     else "95% bootstrap CI)"))
    grid_x(ax)
    if tags:                                            # legend only when single pairs are shown
        h = [plt.Line2D([], [], marker=marks[tg], ls="", ms=5.5, color=INK2) for tg in tags]  # one symbol per pair
        ax.legend(h, [f"pair {tg}" for tg in tags], loc="upper right", fontsize=7)
    return fig


# --------------------------------------------------------------- 6. copy-move
def fig_copymove():
    """Copy-move margins over chance (Sec. VI)."""
    d = table("copymove.csv")                           # from build_tables.py
    sets = list(dict.fromkeys(r["dataset"] for r in d))  # dataset names in table order
    fig, ax = plt.subplots(figsize=(4.8, 2.9))          # figure size
    w = 0.34                                            # bar width
    for i, s in enumerate(sets):                        # one group per dataset
        for method, c, dx in (("ELA", BLUE, -w / 2), ("ELA + MRF", ORANGE, w / 2)):  # two bars per group
            r = next(r for r in d if r["dataset"] == s and r["method"] == method)  # its table row
            v, lo, hi = float(r["margin"]), float(r["ci_lo"]), float(r["ci_hi"])  # margin and its interval
            ax.bar(i + dx, v, width=w, color=c, edgecolor="white", lw=2)  # bar
            ax.errorbar(i + dx, v, yerr=[[v - lo], [hi - v]], color=INK, lw=0.9, capsize=2.5)  # error bar
            ax.text(i + dx, hi + 0.002, f"{v:+.3f}", ha="center", va="bottom", fontsize=7)  # value above it
    n = {r["dataset"]: r["n"] for r in d}               # images per dataset
    ax.set_xticks(range(len(sets)), [f"{s.capitalize()}\n({int(n[s]):,} images)" for s in sets])  # group labels
    ax.tick_params(axis="x", length=0)                  # no tick marks
    ax.axhline(0, color=INK2, lw=0.8)                   # zero line
    top = max(float(r["ci_hi"]) for r in d)             # top of the highest interval
    ax.set_ylim(min(0, min(float(r["ci_lo"]) for r in d)) - 0.002, top * 1.45)  # y range
    ax.set_ylabel("F1 above own chance line\n(95% bootstrap CI)")  # y-axis label
    grid_y(ax)
    ax.legend([swatch(BLUE), swatch(ORANGE)], ["ELA", "ELA + MRF"], loc="upper left", ncol=2,  # legend
              fontsize=7, handlelength=1, handleheight=1)
    return fig


# -------------------------------------------------------------- 7. stabilisers
def fig_stabiliser():
    """Fig. 3: weight norm per epoch for the fp16 runs, first non-finite loss marked."""
    d = table("stabiliser.csv", required=False)         # stabiliser.csv may be missing
    if not d:
        return None
    by = defaultdict(list)                              # candidate -> its epoch rows
    for r in d:
        by[r["candidate"]].append(r)                    # group rows by candidate
    colours = dict(zip(sorted(by), (BLUE, AQUA, VIOLET)))  # one colour per candidate
    names = {"baseline": "no stabiliser", "clip": "gradient clipping", "wd1e-3": "weight decay 1e-3"}  # readable names
    fig, ax = plt.subplots(figsize=(FULL - 0.95, 2.3))   # end labels overhang the axes
    ends = []                                           # right-hand labels: [last value, text, last epoch]
    for cand, rs in sorted(by.items()):                 # one line per candidate
        ep = np.array([int(r["epoch"]) for r in rs]); wn = np.array([float(r["weight_norm"]) for r in rs])  # epochs and weight norms
        ax.plot(ep, wn, color=colours[cand], lw=1.8)    # the line
        bad = [i for i, r in enumerate(rs) if int(r["non_finite"]) > 0]  # epochs with a skipped batch
        if bad:                                         # if any ...
            i = bad[0]                                  # ... mark the first with an X
            ax.plot(ep[i], wn[i], "X", ms=8, color=colours[cand], mec="white", mew=0.8, zorder=4)
            tag = f"non-finite from ep {ep[i]}\nat norm {wn[i]:,.0f}"  # label text
        else:
            j = int(wn.argmax())                        # otherwise label the peak
            tag = f"peak {wn[j]:,.0f} (ep {ep[j]}),\nnever non-finite"
        ends.append([wn[-1], f"{names.get(cand, cand)}: {tag}", ep[-1], colours[cand]])  # value, label, last epoch, colour
    all_wn = [float(r["weight_norm"]) for r in d]       # every weight norm
    ax.set_yscale("log")                                # log y axis
    ax.set_ylim(min(all_wn) * 0.6, max(all_wn) * 2.5)   # y range with margin
    # direct labels at the right edge; three-line labels need a minimum gap,
    # converted from points to decades using the axes' real height.
    fig.canvas.draw()                                   # lay out the figure so sizes can be measured
    h_pt = ax.get_window_extent().height * 72 / fig.dpi  # axes height in points (72 per inch)
    decades = np.log10(ax.get_ylim()[1] / ax.get_ylim()[0])  # number of powers of ten shown
    min_gap = 3 * 7.2 * 1.3 / (h_pt / decades)          # 3 lines of 7.2-pt text, as a distance in powers of ten
    ends.sort()                                         # labels in order of height
    ypos = []                                           # label positions (log10 units)
    for wn, *_ in ends:                                 # for each label, bottom to top
        v = np.log10(wn)                                # natural position
        if ypos and v - ypos[-1] < min_gap:             # too close to the one below ...
            v = ypos[-1] + min_gap                      # ... push it up
        ypos.append(v)
    # every label goes in one column right of the longest run, in its line's colour:
    # a run that stopped early would otherwise put its label across the other lines
    x_lab = max(e[2] for e in ends) + 2                 # x of the label column
    for (wn, text, last, c), v in zip(ends, ypos):      # draw the labels
        ax.text(x_lab, 10 ** v, text, va="center", fontsize=7, color=c)
    ax.set_xlim(0, x_lab)                               # x range
    ax.set_xlabel("Epoch (400 optimiser steps each; fp16 autocast)")  # x-axis label
    ax.set_ylabel("Weight norm, all parameters (log)")  # y-axis label
    grid_y(ax)
    ax.plot([], [], "X", color=INK2, ms=7, ls="", label="first non-finite loss")  # invisible point, only for the legend
    ax.legend(loc="lower right", fontsize=7)            # legend
    return fig


# -------------------------------------------------------------- 8. mechanism
def fig_mechanism():
    """Fig. 6: pixels the prior changes, by distance to the true boundary."""
    d = table("mechanism.csv", required=False)          # mechanism.csv may be missing
    if not d:
        return None
    k = 1e6                                             # show pixels in millions
    fig, ax = plt.subplots(figsize=(COL, 1.8))          # one-column figure
    y = np.arange(len(d))[::-1]                         # one row per distance band
    v = {key: np.array([float(r[key]) for r in d]) / k  # each count, in millions
         for key in ("fp_removed", "fn_filled", "tp_lost", "fp_added")}
    h = 0.62                                            # bar height
    ax.barh(y, v["fp_removed"], height=h, color=ORANGE, edgecolor="white", lw=1.2,  # right side: what the prior fixed
            label="false positives removed")
    ax.barh(y, v["fn_filled"], left=v["fp_removed"], height=h, color=AQUA, edgecolor="white",  # stacked on the first
            lw=1.2, label="missed pixels filled")
    ax.barh(y, -v["tp_lost"], height=h, color=MUTED, edgecolor="white", lw=1.2,  # left side (negative): what it broke
            label="correct pixels lost")
    ax.barh(y, -v["fp_added"], left=-v["tp_lost"], height=h, color=PALE, edgecolor="white",  # stacked on the third
            lw=1.2, label="false positives added")
    gain = v["fp_removed"] + v["fn_filled"]             # total fixed per band
    for yi, gv, r in zip(y, gain, d):                   # one label per band
        ax.text(gv + 0.6, yi, f"net {float(r['net']) / k:+.1f}M".replace("-", "\u2212"),  # net change, with a proper minus sign
                va="center", fontsize=6.8)
    ax.axvline(0, color=INK2, lw=0.8)                   # zero line
    ax.set_yticks(y, [r["band"].replace("<=", "≤").replace("-", "–") for r in d])  # band labels with nicer symbols
    ax.tick_params(axis="y", length=0)                  # no tick marks
    ax.set_ylabel("distance to\ntrue boundary")         # y-axis label
    lo = (v["tp_lost"] + v["fp_added"]).max()           # total broken in the worst band
    ax.set_xlim(-lo * 1.15, gain.max() * 1.32)          # x range
    ax.set_xlabel("pixels changed by the prior (millions)")  # x-axis label
    grid_x(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.5,  # legend above the plot
              handlelength=1, handleheight=1, columnspacing=1.0)
    return fig


# ------------------------------------------------------------ 9. per-image F1
def fig_perimage():
    """Histogram of per-image F1, rungs 4 and 5."""
    d = table("perimage_f1.csv", required=False)        # perimage_f1.csv may be missing
    if not d:
        return None
    f4 = np.array([float(r["f1_rung4"]) for r in d]); f5 = np.array([float(r["f1_rung5"]) for r in d])  # per-image F1 without and with the prior
    fig, ax = plt.subplots(figsize=(4.8, 2.8))          # figure size
    bins = np.linspace(0, 1, 21)                        # 20 bins from 0 to 1
    ax.hist(f4, bins=bins, histtype="step", lw=2, color=BLUE, label="U-Net (β = 0)")  # histogram outline
    ax.hist(f5, bins=bins, histtype="step", lw=2, color=ORANGE, label="U-Net + MRF")  # histogram outline
    n_img = len({r["index"] for r in d}); n_mod = len({r["tag"] for r in d})  # how many images and models
    ax.set_xlabel(f"per-image F1: {n_img} fold-0 test images × {n_mod} model(s)")  # x-axis label
    ax.set_ylabel("images")                             # y-axis label
    top = max(np.histogram(f4, bins)[0].max(), np.histogram(f5, bins)[0].max())  # tallest bar
    ax.set_ylim(0, top * 1.35)                          # y range
    grid_y(ax)
    ax.legend(loc="upper center", ncol=2, fontsize=7)   # legend
    return fig


# ------------------------------------------------------------- 10. the U-Net
def fig_unet():
    """Fig. 2: the U-Net drawn as feature-map slabs; sizes are read from unet.py."""
    from matplotlib.patches import Polygon, Rectangle   # shapes for the 3-D slabs
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methods"))  # so methods/unet.py can be imported
    from unet import UNet                               # the real model
    from patch_data import PATCH                        # crop size used in training
    m = UNet()                                          # build it (only to read its sizes)
    ch = [m.e1[0].out_channels, m.e2[0].out_channels, m.e3[0].out_channels,  # channels at each level
          m.bott[0].out_channels]
    k, c_in = m.pre.kernel_size[0], m.pre.out_channels  # first layer's kernel size and channels
    res = [PATCH // 2 ** i for i in range(4)]           # resolution at each level (halves each time)

    # (front, top, side): the side face is the map's spatial extent, so it gets
    # the main tint; the front (channel thickness) is a shade darker.
    BLUE_F, BLUE_T, BLUE_S = "#b9d2f2", "#e9f1fc", "#d6e5f8"
    GOLD_F, GOLD_T, GOLD_S = "#efcd7a", "#fbf0d2", "#f6e1aa"  # gold: skip connections
    VIO_F, VIO_T, VIO_S = "#c4bbec", "#eeebf9", "#ddd7f4"  # violet: constrained layer
    GREY_F, GREY_T, GREY_S = "#d9d7d0", "#f6f5f2", "#ebe9e4"  # grey: input and output
    GOLD = "#c8930f"                                    # arrow colour for skips

    fig = plt.figure(figsize=(FULL, 2.55))              # full page width
    ax = fig.add_axes([0, 0, 1, 1])                     # axes filling the whole figure
    ax.set_xlim(-0.75, 15.6); ax.set_ylim(-0.95, 4.85); ax.axis("off")  # coordinate range; no axes drawn
    ax.set_aspect("auto")                               # don't force equal x/y scale

    H = {res[0]: 1.55, res[1]: 1.05, res[2]: 0.70, res[3]: 0.46}  # slab height per resolution
    W = lambda c: 0.07 + 0.0016 * c                     # slab thickness grows with channels
    Y = [3.55, 2.25, 1.15, 0.25]                        # y centre of each level

    def slab(x, yc, w, h, cols):                        # draw a 3-D slab: front rectangle, top and side faces
        f, t, s = cols                                  # front, top, side colours
        d = 0.22 * h                                    # depth of the 3-D effect
        dx, dy = d, 0.55 * d                            # sideways and upward offsets
        b, top = yc - h / 2, yc + h / 2                 # bottom and top of the front face
        ax.add_patch(Polygon([(x, top), (x + w, top), (x + w + dx, top + dy), (x + dx, top + dy)],  # top face
                             closed=True, fc=t, ec=INK2, lw=0.5))
        ax.add_patch(Polygon([(x + w, b), (x + w + dx, b + dy), (x + w + dx, top + dy), (x + w, top)],  # side face
                             closed=True, fc=s, ec=INK2, lw=0.5))
        ax.add_patch(Rectangle((x, b), w, h, fc=f, ec=INK2, lw=0.5))  # front face
        return dict(x0=x, x1=x + w + dx, xm=x + w / 2, b=b, top=top + dy, yc=yc, xr=x + w)  # handy positions for arrows and labels

    def arrow(p0, p1, c=INK2, lw=0.8):                  # arrow from p0 to p1
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                    shrinkA=0, shrinkB=0, mutation_scale=7))

    def label(x, y, s, **kw):                           # text with default size, centring and colour
        kw.setdefault("fontsize", 7); kw.setdefault("ha", "center")  # defaults
        ax.text(x, y, s, color=kw.pop("color", INK), **kw)  # draw it

    # input and the constrained layer
    inp = slab(0.0, Y[0], 0.07, H[res[0]], (GREY_F, GREY_T, GREY_S))  # input image
    label(inp["xm"] + 0.1, inp["top"] + 0.1, "RGB")     # its label
    con = slab(1.15, Y[0], 0.07, H[res[0]], (VIO_F, VIO_T, VIO_S))  # constrained first layer
    arrow((inp["x1"] + 0.05, Y[0]), (con["x0"] - 0.05, Y[0]))  # arrow between them
    label(con["xm"] + 0.1, con["top"] + 0.1, f"{c_in}", color=VIOLET)  # its channel count
    label(con["xm"] + 0.1, con["b"] - 0.12, f"constrained\n{k}$\\times${k}: centre $-1$,\nsurround sums to 1",  # description below it
          va="top", color=VIOLET, fontsize=6.5, linespacing=1.15)

    # encoder
    ex = [2.35, 3.35, 4.25]                             # x position of each encoder level
    enc = []                                            # encoder slabs
    for i in range(3):                                  # three levels
        e = slab(ex[i], Y[i], W(ch[i]), H[res[i]], (BLUE_F, BLUE_T, BLUE_S))  # slab
        label(e["xm"] + 0.06, e["top"] + 0.08, f"{ch[i]}")  # channel count
        enc.append(e)
    arrow((con["x1"] + 0.05, Y[0]), (enc[0]["x0"] - 0.05, Y[0]))  # arrow from first layer to encoder
    bot = slab(5.25, Y[3], W(ch[3]), H[res[3]], (BLUE_F, BLUE_T, BLUE_S))  # bottleneck
    label(bot["xm"] + 0.05, bot["top"] + 0.08, f"{ch[3]}")  # its channel count
    for i in range(4):                                   # resolution of each level, at left
        label(-0.2, Y[i], f"{res[i]}$\\times${res[i]}", ha="right", va="center", fontsize=6.5,
              color=INK2)
    lower = enc[1:] + [bot]                             # the slab below each encoder level
    for i in range(3):                                   # max-pool, down a level
        a, b = enc[i], lower[i]
        arrow((a["xm"] + 0.02, a["b"] - 0.05), (b["x0"] - 0.06, b["yc"]), c=INK2)  # diagonal arrow down

    # decoder: concat [skip copy | upsampled], then two convs
    dx = {2: 6.55, 1: 8.55, 0: 10.75}                   # x position of each decoder level
    prev = bot                                          # start from the bottleneck
    dec = {}                                            # decoder output slab per level
    for i in (2, 1, 0):                                 # bottom level first
        g = slab(dx[i], Y[i], W(ch[i]), H[res[i]], (GOLD_F, GOLD_T, GOLD_S))  # gold = copied skip features
        u = slab(g["xr"], Y[i], W(ch[i]), H[res[i]], (BLUE_F, BLUE_T, BLUE_S))  # blue = upsampled features beside it
        label((g["x0"] + u["x1"]) / 2, u["top"] + 0.08, f"{ch[i]}+{ch[i]}")  # channel counts
        arrow((prev["x1"] + 0.06, prev["yc"]), (u["xm"], u["b"] - 0.05))                 # upsample
        o = slab(u["x1"] + 0.45, Y[i], W(ch[i]), H[res[i]], (BLUE_F, BLUE_T, BLUE_S))  # output of the two convs
        label(o["xm"] + 0.06, o["top"] + 0.08, f"{ch[i]}")  # its channel count
        arrow((u["x1"] + 0.05, Y[i]), (o["x0"] - 0.05, Y[i]))                             # convs
        e = enc[i]                                                                         # skip
        arrow((e["x1"] + 0.06, Y[i] + (0.1 if i == 0 else 0)), (g["x0"] - 0.06, Y[i] + (0.1 if i == 0 else 0)),
              c=GOLD, lw=1.0)
        dec[i] = o; prev = o                            # remember it; next level starts here

    # output: 1x1 conv to one logit per pixel, then threshold or MRF
    o = dec[0]                                          # top decoder output
    lg = slab(o["x1"] + 0.55, Y[0], 0.07, H[res[0]], (GREY_F, GREY_T, GREY_S))  # logit map
    arrow((o["x1"] + 0.05, Y[0]), (lg["x0"] - 0.05, Y[0]))  # arrow
    label(lg["xm"] + 0.1, lg["top"] + 0.08, "1")        # one channel
    label(lg["xm"] + 0.1, lg["b"] - 0.12, "logits", va="top", color=INK2, fontsize=6.5)  # label
    bx, bw, bh = lg["x1"] + 0.45, 1.65, 0.95            # box position and size
    ax.add_patch(FancyBboxPatch((bx, Y[0] - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.1",  # final decision box
                                fc="#fbe2d7", ec=ORANGE, lw=0.9, gid="box"))
    label(bx + bw / 2, Y[0], "threshold $\\tau$\n(rung 4), or\nMRF $(\\tau,\\beta)$\n(rung 5)",  # its text
          va="center", fontsize=6.6, linespacing=1.1)
    arrow((lg["x1"] + 0.05, Y[0]), (bx - 0.05, Y[0]))   # arrow into it

    # key, written once
    kx, ky = 6.6, 4.62                                  # key position
    arrow((kx, ky), (kx + 0.45, ky), c=GOLD, lw=1.0)    # gold arrow sample
    label(kx + 0.55, ky, "skip: copy and concatenate", ha="left", va="center", fontsize=6.5)  # its meaning
    up_txt = {"resize": "nearest $\\times$2 resize, then 3$\\times$3 conv",
              "transpose": "2$\\times$2 transposed conv"}[m.upsample]   # describes the model actually drawn
    ax.text(-0.7, -0.38, "Slab height: resolution (left). Slab thickness: channels (above). "  # caption line 1
            f"Down: 2$\\times$2 max-pool. Up: {up_txt}.", fontsize=6.5,
            color=INK2, va="center", ha="left")
    ax.text(-0.7, -0.72, "Right: 2$\\times$(3$\\times$3 conv, BatchNorm, ReLU), same padding; "  # caption line 2 (parameter count from the real model)
            f"the last is a 1$\\times$1 conv. {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M "
            "parameters.", fontsize=6.5, color=INK2, va="center", ha="left")
    return fig


# ------------------------------------------------------------------ 11. SAM2
def fig_sam2():
    """Fig. 9: SAM2 among the other rungs, plus the ground-truth oracle."""
    s = table("sam2.csv", required=False)               # sam2.csv may be missing
    if not s:
        return None
    s = {r["variant"]: r for r in s}                    # variant -> row
    t = {r["key"]: r for r in table("main.csv")}        # Table I
    ds = {int(r["n_train"]): r for r in table("datasize.csv")}  # training size -> row
    rows = [("SAM2, best proposal\n(picked with ground truth)", s["oracle"], INK2, "hollow"),  # (label, row, colour, style), top to bottom
            ("SAM2, size-filtered union\n(rung 3)", s["automatic"], INK2, "full"),
            ("U-Net + MRF (rung 5)", t["rung5"], ORANGE, "full"),
            ("U-Net (rung 4)", t["rung4"], BLUE, "full"),
            ("U-Net, 50 training masks", dict(f1=ds[min(ds)]["f1"], f1_sd=ds[min(ds)]["f1_sd"]), BLUE,
             "full"),
            ("ELA + MRF (rung 2)", t["rung2"], ORANGE, "full")]
    fig, ax = plt.subplots(figsize=(COL, 2.35))         # one-column figure
    y = np.arange(len(rows))[::-1].astype(float)        # y positions
    for yi, (lab, r, c, style) in zip(y, rows):         # one row per entry
        v, sd = float(r["f1"]), float(r["f1_sd"])       # F1 and SD
        ax.plot([v - sd, v + sd], [yi, yi], color=c, lw=1.4, solid_capstyle="round")  # SD line
        ax.plot(v, yi, "o", ms=5.5, color=c, mfc="white" if style == "hollow" else c, mew=1.3,  # dot (hollow for the oracle)
                mec=c if style == "hollow" else "white")
        ax.text(v + sd + 0.02, yi, f"{v:.3f}", va="center", fontsize=6.8)  # value label
    ch = num(t["chance"], "f1")                         # chance F1
    ax.axvline(ch, color=MUTED, ls=(0, (3, 2)), lw=0.9)  # dashed chance line
    ax.text(ch, 1.0, f"chance {ch:.3f}", transform=above_axes(ax), ha="center", va="bottom",  # its label above the plot
            fontsize=6.5, color=INK2)
    a, o = float(s["automatic"]["f1"]), float(s["oracle"]["f1"])  # union and oracle F1
    ax.annotate("", xy=(o - 0.015, y[0] - 0.12), xytext=(a + 0.15, y[1] + 0.12),  # dotted arrow from union to oracle (the gap)
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=6,
                                shrinkA=0, shrinkB=0, ls=(0, (2, 1.5))))
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.6)  # row labels
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)  # no tick marks; no left axis line
    ax.set_xlim(0.15, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4)  # axis ranges
    ax.set_xlabel("Pixel F1 (micro), $\\pm$1 SD across test folds")  # x-axis label
    grid_x(ax)
    return fig


# --------------------------------------------- 12. selection: pixels vs images
def fig_selection():
    """Fig. 7: the prior when (tau, beta) are chosen by pooled F1 vs per-image F1."""
    sel = table("selection.csv", required=False)        # selection.csv may be missing
    if not sel:
        return None
    sel = {(r["unary"], r["criterion"]): r for r in sel}  # (unary, criterion) -> row
    det = {(r["unary"], r["criterion"]): r for r in table("selection_detail.csv")}  # same, per-image detail
    fig, ax = plt.subplots(figsize=(COL, 2.2))          # one-column figure
    ys, labels = [], []                                 # y positions and labels
    groups = (("pooled", "chosen for pooled F1"), ("per-image", "chosen for per-image F1"))  # (criterion, heading)
    y = 4.6                                             # starting height
    for crit, head in groups:                           # one group per criterion
        r = sel[("U-Net", crit)]                        # U-Net rows only
        betas = sorted(float(x.split(":")[0]) for x in r["beta_counts"].split())  # betas that were chosen
        lost = float(det[("U-Net", crit)]["frac_lost"])  # fraction of images that lost every detection
        pct = f"{100 * lost:.0f}" if lost >= 0.1 else f"{100 * lost:.1f}"  # as a percentage
        ax.text(0.502, y + 0.62, f"{head} ($\\beta$ {betas[0]:g}\u2013{betas[-1]:g}): "  # group heading
                f"{pct}% of images lose every detection", fontsize=6.2, color=INK2,
                va="center", ha="left")
        for meas, k0, k1 in (("pooled F1", "pooled_beta0", "pooled_prior"),  # two measures per group
                             ("per-image F1", "img_beta0", "img_prior")):
            a, b = float(r[k0]), float(r[k1])           # F1 without and with the prior
            ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(  # arrow from one to the other
                arrowstyle="-|>", color=ORANGE if b > a else MUTED, lw=1.3, mutation_scale=7,
                shrinkA=3.5, shrinkB=2.5))
            ax.plot(a, y, "o", ms=5, color=BLUE, mec="white", mew=0.8, zorder=3)  # blue dot: no prior
            ax.plot(b, y, "o", ms=5, color=ORANGE, mec="white", mew=0.8, zorder=3)  # orange dot: with prior
            lx = max(a, b) + 0.012                      # label position
            ax.text(lx, y, f"{b - a:+.3f}".replace("-", "−"), va="center", fontsize=6.6,  # change, with a proper minus sign
                    color=ORANGE if b > a else INK2)
            ys.append(y); labels.append(meas)           # remember the row
            y -= 0.9                                    # next row down
        y -= 0.75                                       # gap between groups
    ax.set_yticks(ys, labels, fontsize=6.6)             # row labels
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)  # no tick marks; no left axis line
    ax.set_xlim(0.5, 0.8); ax.set_ylim(y + 0.9 - 0.55, 5.55)  # axis ranges
    ax.set_xlabel("U-Net F1, mean over 20 pairs")       # x-axis label
    grid_x(ax)
    h = [plt.Line2D([], [], marker="o", ls="", ms=5, color=BLUE),  # legend symbols
         plt.Line2D([], [], marker="o", ls="", ms=5, color=ORANGE)]
    ax.legend(h, ["$\\beta = 0$", "with the prior"], loc="lower right", fontsize=6.5,  # legend
              handletextpad=0.2, borderaxespad=0.2)
    return fig

FIGURES = [("fig0_method", fig_method), ("fig1_main_f1", fig_main), ("fig2_prior_gain", fig_gain),  # (file name, function) for every figure
           ("fig3_datasize", fig_datasize), ("fig4_per_category", fig_category),
           ("fig5_image_auc", fig_auc), ("fig6_copymove", fig_copymove),
           ("fig7_stabiliser", fig_stabiliser), ("fig8_mechanism", fig_mechanism),
           ("fig9_perimage_f1", fig_perimage), ("fig10_unet", fig_unet),
           ("fig11_sam2", fig_sam2), ("fig12_selection", fig_selection)]


def main():
    """Draw, check and save every figure in FIGURES."""
    os.makedirs(OUT, exist_ok=True)                     # create the output folder
    failed = []                                         # error messages of figures that failed
    for name, make in FIGURES:                          # each figure
        fig = make()                                    # draw it
        if fig is None:                                 # its table is missing
            print(f"  {name}: skipped (its table is not there yet)")
            continue
        try:
            exts = ("pdf", "png", "svg") if name == "fig10_unet" else ("pdf", "png")  # Fig. 2 also as SVG
            save_checked(fig, os.path.join(OUT, name), exts=exts)  # check and save
            print(f"  {name}: ok")
        except RuntimeError as e:                       # something overlaps
            failed.append(str(e))                       # keep the message
            print(f"  {name}: NOT SAVED")
        plt.close(fig)                                  # free memory
    if failed:                                          # any failures ...
        print("\n" + "\n\n".join(failed))               # ... print them all
        raise SystemExit(f"{len(failed)} figure(s) have collisions")  # ... and stop with an error
    print(f"all figures written to {OUT}/")


if __name__ == "__main__":
    main()
