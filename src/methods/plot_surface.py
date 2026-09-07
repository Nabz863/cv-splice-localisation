"""Characterise the (tau, beta) objective surface."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("results/rung2_grid.npz", allow_pickle=True)
counts, grid, solvers = d["counts"], d["grid"], list(d["solvers"])
taus = sorted(set(grid[:, 0])); betas = sorted(set(grid[:, 1]))

fig, axes = plt.subplots(1, len(solvers), figsize=(6.5 * len(solvers), 5), squeeze=False)
for si, name in enumerate(solvers):
    S = np.zeros((len(taus), len(betas)))
    for gi, (t, b) in enumerate(grid):
        tp, fp, fn = counts[:, si, gi].sum(0)
        S[taus.index(t), betas.index(b)] = 2 * tp / (2 * tp + fp + fn)
    ax = axes[0][si]
    im = ax.imshow(S, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels(betas, rotation=45)
    ax.set_yticks(range(len(taus)));  ax.set_yticklabels(taus)
    ax.set_xlabel("beta"); ax.set_ylabel("tau"); ax.set_title(f"dataset F1 - {name}")
    plt.colorbar(im, ax=ax)

    i, j = np.unravel_index(S.argmax(), S.shape)
    print(f"\n{name}: F1 {S.min():.4f} to {S.max():.4f}  (spread {S.max()-S.min():.4f})")
    print(f"  best tau={taus[i]}, beta={betas[j]}")
    print(f"  on boundary: tau {'YES' if i in (0, len(taus)-1) else 'no'}, "
          f"beta {'YES' if j in (0, len(betas)-1) else 'no'}")
    print(f"  F1 at beta=0 (rung 1): {S[:, 0].max():.4f}")

plt.tight_layout(); plt.savefig("results/rung2_surface.png", dpi=110)
print("\nwrote results/rung2_surface.png")
