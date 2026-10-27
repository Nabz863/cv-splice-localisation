"""Tests for overlap.py: each kind of collision is caught and a clean figure passes."""
import matplotlib                                 # plotting library
matplotlib.use("Agg")                             # draw to memory, no screen needed
import matplotlib.pyplot as plt                   # plotting
from matplotlib.patches import FancyBboxPatch     # boxes used in diagrams
from overlap import collisions                    # the function being tested


def fig_with(draw):
    """A small figure with whatever draw(ax) puts on it."""
    fig, ax = plt.subplots(figsize=(4, 3)); draw(ax); fig.set_dpi(220)  # draw, then set the resolution used for saving
    return fig


# ---- Each case deliberately makes one kind of overlap ----
cases = {
    "text on text":   lambda ax: (ax.text(0.5, 0.5, "first label"), ax.text(0.52, 0.5, "second label")),
    "text on bar":    lambda ax: (ax.bar([0], [1]), ax.text(0, 0.5, "inside", ha="center")),
    "text on marker": lambda ax: (ax.plot([0.5], [0.5], "o", ms=12), ax.text(0.5, 0.5, "x", ha="center")),
    "text on line":   lambda ax: (ax.axvline(0.5), ax.text(0.5, 0.5, "chance", ha="center")),
    "legend on data": lambda ax: (ax.plot([0, 1], [1, 1], "o-", label="a"), ax.set_ylim(0, 1.1),
                                  ax.legend(loc="upper left")),
    "tick labels":    lambda ax: (ax.set_xticks([800, 820], ["800", "1093"]), ax.set_xlim(0, 1200)),   # two ticks too close
    "box on box":     lambda ax: (ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.4, 0.3, gid="box")),
                                  ax.add_patch(FancyBboxPatch((0.3, 0.2), 0.4, 0.3, gid="box"))),
    "text out of box": lambda ax: (ax.add_patch(FancyBboxPatch((0.4, 0.4), 0.1, 0.1, boxstyle="square,pad=0", gid="box")),
                                   ax.text(0.45, 0.45, "a label far too wide for it", ha="center")),
}
for name, draw in cases.items():
    found = collisions(fig_with(draw))                  # list of overlaps found
    assert found, f"missed: {name}"               # every case must be caught
    plt.close("all")                              # free the figure

# ---- A figure with nothing overlapping must pass ----
clean = fig_with(lambda ax: (ax.bar([0, 1], [1, 2]), ax.text(0, 1.1, "1", ha="center"),
                             ax.text(1, 2.1, "2", ha="center"), ax.set_ylim(0, 2.6),
                             ax.add_patch(FancyBboxPatch((1.8, 0.3), 1.0, 0.6, boxstyle="square,pad=0",
                                                         gid="box")),
                             ax.text(2.3, 0.6, "fits", ha="center", va="center"), ax.set_xlim(-0.5, 3)))
assert not collisions(clean), collisions(clean)         # must find none
print(f"all {len(cases)} collision types caught; clean figure passes")
