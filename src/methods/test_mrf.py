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

# 3. large beta collapses to a single label
assert len(np.unique(solve_graphcut(s, 1.0, 1e4))) == 1

print("all MRF tests pass")
