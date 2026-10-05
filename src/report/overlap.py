"""Refuse to save a figure whose labels collide with anything.

Eyeballing a figure misses collisions at some sizes and catches them at others.
This measures them instead. After the figure is drawn, every visible piece of
text, every legend, every bar, marker, line and arrow has a bounding box (or a
path) in screen pixels, and any two that should not touch are tested:

  text      vs text, bars/boxes, markers, line paths, arrows
  legend    vs every data artist and every text outside the legend itself
  box       vs box (diagram boxes must not overlap each other)

The one permitted overlap is text that sits entirely inside a box drawn for it
(the labels in the method diagram). Tag such a box with gid="box".

Usage:
    problems = collisions(fig)          # list of human-readable strings
    save_checked(fig, "results/figures/fig1")   # raises if problems is non-empty
"""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

SHRINK = 1.0     # px trimmed from every box first, so merely touching is not a collision
_NO_MARK = {None, "None", "none", "", " "}
_NO_LINE = {"None", "none", "", " "}


def _shrunk(bb, d=SHRINK):
    return Bbox.from_extents(bb.x0 + d, bb.y0 + d, bb.x1 - d, bb.y1 - d)


def _hit(a, b):
    return a.overlaps(b) and a.width > 0 and a.height > 0 and b.width > 0 and b.height > 0


def _inside(inner, outer):
    return (inner.x0 >= outer.x0 and inner.x1 <= outer.x1
            and inner.y0 >= outer.y0 and inner.y1 <= outer.y1)


def _label(a):
    if isinstance(a, Text):
        return f'text "{a.get_text()[:40]}"'
    if isinstance(a, Legend):
        return "legend"
    gid = a.get_gid() or a.get_label() or type(a).__name__
    return f"{type(a).__name__}({gid})"


def _undrawn_ticklabels(fig):
    """Tick labels matplotlib keeps but does not draw because their tick lies
    outside the axis limits (common on log axes). They have text and a position,
    so without this they would count as phantom text."""
    out = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            span = hi - lo
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span:
                    out.update({id(tick.label1), id(tick.label2)})
    return out


def _texts(fig, r):
    skip = _undrawn_ticklabels(fig)
    out = []
    for t in fig.findobj(Text):
        if id(t) in skip or not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb))
    return out


def collisions(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legends = [lg for ax in fig.axes for lg in [ax.get_legend()] if lg is not None]
    legends += [lg for lg in fig.legends]
    legend_texts = {id(t) for lg in legends for t in list(lg.get_texts()) + [lg.get_title()]}
    legend_handles = {id(h) for lg in legends for h in lg.legend_handles}

    # ---- texts (including drawn tick labels, titles, legend entries)
    texts = [(t, _shrunk(bb)) for t, bb in _texts(fig, r)]

    # ---- data artists
    boxes, bars, marks, paths, arrows = [], [], [], [], []
    for ax in fig.axes:
        grid = set(map(id, ax.get_xgridlines() + ax.get_ygridlines()))
        for p in ax.patches:
            if not p.get_visible() or id(p) in legend_handles:
                continue
            bb = _shrunk(p.get_window_extent(r))
            if isinstance(p, FancyArrowPatch):
                arrows.append((p, bb))
            elif isinstance(p, FancyBboxPatch) and p.get_gid() == "box":
                boxes.append((p, bb))
            elif isinstance(p, (Rectangle, FancyBboxPatch)):
                bars.append((p, bb))
        for ln in ax.lines:
            if not ln.get_visible() or id(ln) in grid or id(ln) in legend_handles:
                continue
            xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
            xy = xy[np.isfinite(xy).all(1)]
            if ln.get_marker() not in _NO_MARK:
                h = ln.get_markersize() * fig.dpi / 72 / 2 + ln.get_markeredgewidth() * fig.dpi / 144
                for x, y in xy:
                    marks.append((ln, _shrunk(Bbox.from_extents(x - h, y - h, x + h, y + h))))
            if ln.get_linestyle() not in _NO_LINE and len(xy) > 1:
                paths.append((ln, Path(xy)))
        for c in ax.collections:
            if isinstance(c, LineCollection) and c.get_visible():
                tr = c.get_transform()
                for seg in c.get_segments():
                    seg = tr.transform(np.asarray(seg, float))
                    if len(seg) > 1:
                        paths.append((c, Path(seg)))
        for t in ax.texts:
            if isinstance(t, Annotation) and t.arrow_patch is not None:
                arrows.append((t.arrow_patch, _shrunk(t.arrow_patch.get_window_extent(r))))

    out = []

    def report(a, b):
        out.append(f"{_label(a)} overlaps {_label(b)}")

    # text vs text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (a, ba), (b, bb) = texts[i], texts[j]
            if _hit(ba, bb):
                report(a, b)

    # text vs data artists (legend entries are checked as part of the legend)
    for t, tb in texts:
        if id(t) in legend_texts:
            continue
        for p, pb in bars + arrows:
            if _hit(tb, pb):
                report(t, p)
        for p, pb in boxes:
            if _hit(tb, pb) and not _inside(tb, pb):
                report(t, p)
        for ln, mb in marks:
            if _hit(tb, mb):
                report(t, ln); break
        for ln, path in paths:
            if path.intersects_bbox(tb, filled=False):
                report(t, ln); break

    # legend vs everything outside it
    for lg in legends:
        lb = _shrunk(lg.get_window_extent(r))
        for t, tb in texts:
            if id(t) not in legend_texts and _hit(lb, tb):
                report(lg, t)
        for p, pb in bars + boxes + arrows + marks:
            if _hit(lb, pb):
                report(lg, p); break
        for ln, path in paths:
            if path.intersects_bbox(lb, filled=False):
                report(lg, ln); break

    # diagram boxes vs each other
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _hit(boxes[i][1], boxes[j][1]):
                report(boxes[i][0], boxes[j][0])

    return sorted(set(out))


def save_checked(fig, stem, dpi=220, exts=("pdf", "png"), pad=0.06):
    """Check at the output resolution, then save. Raises on any collision.

    The crop box is the union of every visible text and every axes, computed
    here rather than left to bbox_inches="tight", which can drop the overhanging
    end of a long axis label and silently clip it from the saved file."""
    fig.set_dpi(dpi)
    problems = collisions(fig)
    if problems:
        raise RuntimeError(f"{stem}: {len(problems)} collision(s)\n  " + "\n  ".join(problems))
    r = fig.canvas.get_renderer()
    parts = []
    parts += [bb for _, bb in _texts(fig, r)]
    parts += [lg.get_window_extent(r) for ax in fig.axes for lg in [ax.get_legend()] if lg]
    parts += [ax.get_window_extent(r) for ax in fig.axes]
    box = Bbox.union(parts).transformed(fig.dpi_scale_trans.inverted()).padded(pad)
    for ext in exts:
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches=box)
