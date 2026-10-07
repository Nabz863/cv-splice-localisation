import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mrf import solve_icm, solve_graphcut, energy

rng = np.random.default_rng(0)
s = rng.random((64, 64)) * 3

# 1. beta = 0 reduces exactly to thresholding (the paper's claim, checked)
for tau in (0.5, 1.0, 2.0):
    assert np.array_equal(solve_graphcut(s, tau, 0.0), s > tau), f"gc beta=0, tau={tau}"
    assert np.array_equal(solve_icm(s, tau, 0.0), s > tau), f"icm beta=0, tau={tau}"

# 2. graph cuts is exact, so it can never lose to ICM on energy
for beta in (0.1, 0.5, 2.0):
    xg, xi = solve_graphcut(s, 1.0, beta), solve_icm(s, 1.0, beta)
    eg, ei = energy(s, xg, 1.0, beta), energy(s, xi, 1.0, beta)
    assert eg <= ei + 1e-6, f"beta={beta}: graphcut {eg:.2f} > icm {ei:.2f}"

# 2b. ICM sweeps never raise the energy, including on the image border (an
#     earlier version wrapped neighbours around the edges with np.roll, so it
#     minimised a different, toroidal energy from the one scored here)
for beta in (0.5, 2.0, 8.0):
    e0 = energy(s, s > 1.0, 1.0, beta)
    prev = e0
    for it in (1, 2, 4, 8):
        e = energy(s, solve_icm(s, 1.0, beta, iters=it), 1.0, beta)
        assert e <= prev + 1e-6, f"icm energy rose: beta={beta}, iters={it}"
        prev = e

# 2c. at convergence ICM is a local minimum of THIS energy: no single-pixel flip
#     lowers it. Checked by brute force with energy(), which does not share code
#     with solve_icm. The toroidal version failed this on border pixels.
small = rng.random((12, 12)) * 3
for beta in (0.3, 0.8):
    x = solve_icm(small, 1.0, beta, iters=50)
    e = energy(small, x, 1.0, beta)
    for i in range(12):
        for j in range(12):
            y = x.copy(); y[i, j] = ~y[i, j]
            assert energy(small, y, 1.0, beta) >= e - 1e-9, f"flip ({i},{j}) lowers E, beta={beta}"

# 3. large beta collapses to a single label
assert len(np.unique(solve_graphcut(s, 1.0, 1e4))) == 1

print("all MRF tests pass")
