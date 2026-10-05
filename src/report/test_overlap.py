"""The overlap check must catch each kind of collision and pass a clean figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from overlap import collisions


def fig_with(draw):
    fig, ax = plt.subplots(figsize=(4, 3)); draw(ax); fig.set_dpi(220)
    return fig


cases = {
    "text on text":   lambda ax: (ax.text(0.5, 0.5, "first label"), ax.text(0.52, 0.5, "second label")),
    "text on bar":    lambda ax: (ax.bar([0], [1]), ax.text(0, 0.5, "inside", ha="center")),
    "text on marker": lambda ax: (ax.plot([0.5], [0.5], "o", ms=12), ax.text(0.5, 0.5, "x", ha="center")),
    "text on line":   lambda ax: (ax.axvline(0.5), ax.text(0.5, 0.5, "chance", ha="center")),
    "legend on data": lambda ax: (ax.plot([0, 1], [1, 1], "o-", label="a"), ax.set_ylim(0, 1.1),
                                  ax.legend(loc="upper left")),
    "tick labels":    lambda ax: (ax.set_xticks([800, 820], ["800", "1093"]), ax.set_xlim(0, 1200)),
    "box on box":     lambda ax: (ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.4, 0.3, gid="box")),
                                  ax.add_patch(FancyBboxPatch((0.3, 0.2), 0.4, 0.3, gid="box"))),
    "text out of box": lambda ax: (ax.add_patch(FancyBboxPatch((0.4, 0.4), 0.1, 0.1, boxstyle="square,pad=0", gid="box")),
                                   ax.text(0.45, 0.45, "a label far too wide for it", ha="center")),
}
for name, draw in cases.items():
    found = collisions(fig_with(draw))
    assert found, f"missed: {name}"
    plt.close("all")

clean = fig_with(lambda ax: (ax.bar([0, 1], [1, 2]), ax.text(0, 1.1, "1", ha="center"),
                             ax.text(1, 2.1, "2", ha="center"), ax.set_ylim(0, 2.6),
                             ax.add_patch(FancyBboxPatch((1.8, 0.3), 1.0, 0.6, boxstyle="square,pad=0",
                                                         gid="box")),
                             ax.text(2.3, 0.6, "fits", ha="center", va="center"), ax.set_xlim(-0.5, 3)))
assert not collisions(clean), collisions(clean)
print(f"all {len(cases)} collision types caught; clean figure passes")
