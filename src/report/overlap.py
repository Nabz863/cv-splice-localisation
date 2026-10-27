"""Find overlapping labels/legends/markers in a matplotlib figure; save_checked() refuses to save if any."""
import numpy as np                                # arrays and maths
from matplotlib.collections import LineCollection # many line segments drawn as one object
from matplotlib.legend import Legend              # legend box
from matplotlib.lines import Line2D               # a plotted line (with or without markers)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle   # arrows, rounded boxes, bars
from matplotlib.path import Path                  # a line as a list of points (can test crossing a box)
from matplotlib.text import Annotation, Text      # text, and text with an arrow
from matplotlib.transforms import Bbox            # a rectangle (x0, y0, x1, y1) in pixels

SHRINK = 1.0     # px trimmed from every box first, so merely touching is not a collision
_NO_MARK = {None, "None", "none", "", " "}        # marker settings that mean "no marker"
_NO_LINE = {"None", "none", "", " "}              # line-style settings that mean "no line"


# ---- Small box helpers (all boxes are in screen pixels) ----

def _shrunk(bb, d=SHRINK):
    """The box made d pixels smaller on every side."""
    return Bbox.from_extents(bb.x0 + d, bb.y0 + d, bb.x1 - d, bb.y1 - d)


def _hit(a, b):
    """True if two non-empty boxes overlap."""
    return a.overlaps(b) and a.width > 0 and a.height > 0 and b.width > 0 and b.height > 0


def _inside(inner, outer):
    """True if box inner lies completely inside box outer."""
    return (inner.x0 >= outer.x0 and inner.x1 <= outer.x1
            and inner.y0 >= outer.y0 and inner.y1 <= outer.y1)


def _label(a):
    """Readable name of a drawn object, for the error message."""
    if isinstance(a, Text):
        return f'text "{a.get_text()[:40]}"'      # first 40 characters of the text
    if isinstance(a, Legend):
        return "legend"
    gid = a.get_gid() or a.get_label() or type(a).__name__   # its id, its label, or its type
    return f"{type(a).__name__}({gid})"


# ---- Collect every visible piece of text ----

def _undrawn_ticklabels(fig):
    """Tick labels matplotlib keeps but does not draw because their tick lies
    outside the axis limits (common on log axes). They have text and a position,
    so without this they would count as phantom text."""
    out = set()                                   # ids of the tick labels to ignore
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):   # x then y axis
            lo, hi = sorted(lim)                  # axis range, low to high
            span = hi - lo
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span:    # tick outside the range
                    out.update({id(tick.label1), id(tick.label2)})
    return out


def _texts(fig, r):
    """(text object, its box) for every visible, non-empty text in the figure."""
    skip = _undrawn_ticklabels(fig)                     # tick labels to ignore
    out = []                                            # results
    for t in fig.findobj(Text):                   # every Text object anywhere in the figure
        if id(t) in skip or not t.get_visible() or not t.get_text().strip():   # hidden or blank
            continue
        bb = t.get_window_extent(r)               # where it is drawn, in pixels
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb))
    return out


# ---- The check itself ----

def collisions(fig):
    """List of 'X overlaps Y' messages for the figure (empty list = clean)."""
    fig.canvas.draw()                             # lay the figure out so positions are final
    r = fig.canvas.get_renderer()                 # needed to measure sizes
    legends = [lg for ax in fig.axes for lg in [ax.get_legend()] if lg is not None]   # per-axes legends
    legends += [lg for lg in fig.legends]         # plus figure-level legends
    legend_texts = {id(t) for lg in legends for t in list(lg.get_texts()) + [lg.get_title()]}   # text inside legends
    legend_handles = {id(h) for lg in legends for h in lg.legend_handles}   # sample markers/lines inside legends

    # ---- texts (including drawn tick labels, titles, legend entries)
    texts = [(t, _shrunk(bb)) for t, bb in _texts(fig, r)]

    # ---- data artists
    boxes, bars, marks, paths, arrows = [], [], [], [], []   # sorted by kind
    for ax in fig.axes:
        grid = set(map(id, ax.get_xgridlines() + ax.get_ygridlines()))   # grid lines are ignored
        for p in ax.patches:                      # filled shapes
            if not p.get_visible() or id(p) in legend_handles:
                continue
            bb = _shrunk(p.get_window_extent(r))        # its box
            if isinstance(p, FancyArrowPatch):
                arrows.append((p, bb))
            elif isinstance(p, FancyBboxPatch) and p.get_gid() == "box":   # diagram boxes we drew (gid="box")
                boxes.append((p, bb))
            elif isinstance(p, (Rectangle, FancyBboxPatch)):               # bars and other rectangles
                bars.append((p, bb))
        for ln in ax.lines:                       # plotted lines
            if not ln.get_visible() or id(ln) in grid or id(ln) in legend_handles:
                continue
            xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))   # data -> pixel coordinates
            xy = xy[np.isfinite(xy).all(1)]       # drop NaN / infinite points
            if ln.get_marker() not in _NO_MARK:   # has markers: one small box per point
                h = ln.get_markersize() * fig.dpi / 72 / 2 + ln.get_markeredgewidth() * fig.dpi / 144   # half-width in px (72 points per inch)
                for x, y in xy:
                    marks.append((ln, _shrunk(Bbox.from_extents(x - h, y - h, x + h, y + h))))
            if ln.get_linestyle() not in _NO_LINE and len(xy) > 1:   # has a line: keep its path
                paths.append((ln, Path(xy)))
        for c in ax.collections:                  # groups of line segments (e.g. error bars)
            if isinstance(c, LineCollection) and c.get_visible():
                tr = c.get_transform()                  # data -> pixel conversion
                for seg in c.get_segments():
                    seg = tr.transform(np.asarray(seg, float))   # to pixel coordinates
                    if len(seg) > 1:
                        paths.append((c, Path(seg)))
        for t in ax.texts:                        # annotation arrows
            if isinstance(t, Annotation) and t.arrow_patch is not None:
                arrows.append((t.arrow_patch, _shrunk(t.arrow_patch.get_window_extent(r))))

    out = []                                      # the messages

    def report(a, b):
        out.append(f"{_label(a)} overlaps {_label(b)}")

    # text vs text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):        # each pair once
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
            if _hit(tb, pb) and not _inside(tb, pb):   # text fully inside its box is fine
                report(t, p)
        for ln, mb in marks:
            if _hit(tb, mb):
                report(t, ln); break              # one message per line is enough
        for ln, path in paths:
            if path.intersects_bbox(tb, filled=False):   # the line crosses the text's box
                report(t, ln); break

    # legend vs everything outside it
    for lg in legends:
        lb = _shrunk(lg.get_window_extent(r))     # the legend's box
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

    return sorted(set(out))                       # remove duplicates, sort for stable output


# ---- Save only if clean ----

def save_checked(fig, stem, dpi=220, exts=("pdf", "png"), pad=0.06):
    """Check at the output resolution, then save. Raises on any collision.

    The crop box is the union of every visible text and every axes, computed
    here rather than left to bbox_inches="tight", which can drop the overhanging
    end of a long axis label and silently clip it from the saved file."""
    fig.set_dpi(dpi)                              # measure at the resolution we will save at
    problems = collisions(fig)
    if problems:                                  # stop with a list of every overlap
        raise RuntimeError(f"{stem}: {len(problems)} collision(s)\n  " + "\n  ".join(problems))
    r = fig.canvas.get_renderer()                       # needed to measure sizes
    parts = []                                    # boxes that must all be inside the saved image
    parts += [bb for _, bb in _texts(fig, r)]
    parts += [lg.get_window_extent(r) for ax in fig.axes for lg in [ax.get_legend()] if lg]
    parts += [ax.get_window_extent(r) for ax in fig.axes]
    box = Bbox.union(parts).transformed(fig.dpi_scale_trans.inverted()).padded(pad)   # pixels -> inches, plus a margin
    for ext in exts:                              # save as PDF and PNG
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches=box)
