"""Plot rung 2's dataset F1 over the (tau, beta) grid -> results/rung2_surface.png."""
import numpy as np                                # arrays and maths
import matplotlib; matplotlib.use("Agg")          # draw to files, no screen needed
import matplotlib.pyplot as plt                   # plotting

d = np.load("results/rung2_grid.npz", allow_pickle=True)   # per-image counts from run_mrf.py
counts, grid, solvers = d["counts"], d["grid"], list(d["solvers"])
taus = sorted(set(grid[:, 0])); betas = sorted(set(grid[:, 1]))   # the distinct tau and beta values

fig, axes = plt.subplots(1, len(solvers), figsize=(6.5 * len(solvers), 5), squeeze=False)   # one panel per solver
for si, name in enumerate(solvers):
    # ---- Fill a tau x beta table of pooled F1 ----
    S = np.zeros((len(taus), len(betas)))
    for gi, (t, b) in enumerate(grid):
        tp, fp, fn = counts[:, si, gi].sum(0)                        # pool counts over all images
        S[taus.index(t), betas.index(b)] = 2 * tp / (2 * tp + fp + fn)   # pooled F1
    # ---- Draw it as a heat map ----
    ax = axes[0][si]
    im = ax.imshow(S, cmap="viridis", aspect="auto", origin="lower")   # rows = tau, columns = beta
    ax.set_xticks(range(len(betas))); ax.set_xticklabels(betas, rotation=45)
    ax.set_yticks(range(len(taus)));  ax.set_yticklabels(taus)
    ax.set_xlabel("beta"); ax.set_ylabel("tau"); ax.set_title(f"dataset F1 - {name}")
    plt.colorbar(im, ax=ax)                       # colour scale

    # ---- Print where the best point is (on the edge of the grid = grid may be too small) ----
    i, j = np.unravel_index(S.argmax(), S.shape)                     # best (tau, beta)
    print(f"\n{name}: F1 {S.min():.4f} to {S.max():.4f}  (spread {S.max()-S.min():.4f})")
    print(f"  best tau={taus[i]}, beta={betas[j]}")
    print(f"  on boundary: tau {'YES' if i in (0, len(taus)-1) else 'no'}, "
          f"beta {'YES' if j in (0, len(betas)-1) else 'no'}")
    print(f"  F1 at beta=0 (rung 1): {S[:, 0].max():.4f}")          # best F1 with no prior

plt.tight_layout(); plt.savefig("results/rung2_surface.png", dpi=110)   # save the figure
print("\nwrote results/rung2_surface.png")
