"""Rung 2: pairwise MRF over the rung-1 evidence map.

    E(x) = sum_i D_i(x_i) + beta * sum_{(i,j) in N} [x_i != x_j]

D encodes the evidence, the Potts term encodes spatial coherence. Two
parameters (tau, beta), calibrated by grid search -- not trained. At beta = 0
the model reduces exactly to thresholding the unary at tau, which is rung 1.
"""
import numpy as np


def unary(score, tau):
    """Non-negative costs with D1 - D0 = tau - score, so label 1 is preferred
    exactly where score > tau. Non-negativity is required by maxflow."""
    d = score - tau
    return np.maximum(d, 0.0), np.maximum(-d, 0.0)      # (D0, D1)


def solve_icm(score, tau, beta, iters=8):
    """Iterated Conditional Modes. Greedy coordinate descent on E, updating a
    checkerboard half at a time so each update sees fixed neighbours."""
    D0, D1 = unary(score, tau)
    x = (score > tau)
    ii, jj = np.indices(x.shape)
    parity = (ii + jj) % 2

    for _ in range(iters):
        for p in (0, 1):
            n1 = np.zeros(x.shape)          # count of neighbours labelled 1
            nb = np.zeros(x.shape)          # neighbour count (border-aware)
            for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                n1 += np.roll(x, sh, axis=ax)
                nb += np.roll(np.ones_like(x), sh, axis=ax)
            # cost of taking label 0 / 1 given current neighbours
            c0 = D0 + beta * n1
            c1 = D1 + beta * (nb - n1)
            m = parity == p
            x = np.where(m, c1 < c0, x)
    return x


def solve_graphcut(score, tau, beta):
    """Exact global minimum. The Potts energy is submodular for beta >= 0, so
    the s-t min cut is the MAP labelling."""
    import maxflow
    D0, D1 = unary(score, tau)
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(score.shape)
    if beta > 0:
        g.add_grid_edges(nodes, beta)
    # source edge cut => label 1, so its capacity is the cost of label 1
    g.add_grid_tedges(nodes, D1, D0)
    g.maxflow()
    return np.asarray(g.get_grid_segments(nodes))


def energy(score, x, tau, beta):
    D0, D1 = unary(score, tau)
    e = np.where(x, D1, D0).sum()
    e += beta * (x[:, 1:] != x[:, :-1]).sum()
    e += beta * (x[1:, :] != x[:-1, :]).sum()
    return float(e)
