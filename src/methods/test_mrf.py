"""Unit tests for mrf.py. Run from the repo root: python src/methods/test_mrf.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mrf import solve_icm, solve_graphcut, energy

rng = np.random.default_rng(0)
s = rng.random((64, 64)) * 3                            # a random 64x64 score map

# 1. With beta = 0 both solvers must give exactly "score > tau" (rung 2 = rung 1).
for tau in (0.5, 1.0, 2.0):
    assert np.array_equal(solve_graphcut(s, tau, 0.0), s > tau), f"gc beta=0, tau={tau}"
    assert np.array_equal(solve_icm(s, tau, 0.0), s > tau), f"icm beta=0, tau={tau}"

# 2. The graph cut is exact, so its energy can never be higher than ICM's.
for beta in (0.1, 0.5, 2.0):
    xg, xi = solve_graphcut(s, 1.0, beta), solve_icm(s, 1.0, beta)
    eg, ei = energy(s, xg, 1.0, beta), energy(s, xi, 1.0, beta)
    assert eg <= ei + 1e-6, f"beta={beta}: graphcut {eg:.2f} > icm {ei:.2f}"

# 2b. More ICM iterations never raise the energy (border pixels included).
for beta in (0.5, 2.0, 8.0):
    e0 = energy(s, s > 1.0, 1.0, beta)
    prev = e0
    for it in (1, 2, 4, 8):
        e = energy(s, solve_icm(s, 1.0, beta, iters=it), 1.0, beta)
        assert e <= prev + 1e-6, f"icm energy rose: beta={beta}, iters={it}"
        prev = e

# 2c. Once ICM has converged, flipping any single pixel must not lower the energy
#     (it is a local minimum). Checked by brute force with energy().
small = rng.random((12, 12)) * 3
for beta in (0.3, 0.8):
    x = solve_icm(small, 1.0, beta, iters=50)
    e = energy(small, x, 1.0, beta)
    for i in range(12):
        for j in range(12):
            y = x.copy(); y[i, j] = ~y[i, j]            # flip one pixel
            assert energy(small, y, 1.0, beta) >= e - 1e-9, f"flip ({i},{j}) lowers E, beta={beta}"

# 3. A huge beta makes every pixel take the same label.
assert len(np.unique(solve_graphcut(s, 1.0, 1e4))) == 1

print("all MRF tests pass")
