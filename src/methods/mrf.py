"""Rung 2/5: Potts MRF over a score map. E(x) = sum D_i(x_i) + beta * #(neighbours that disagree)."""
import numpy as np


# ---- Energy terms ----

def unary(score, tau):
    """Cost of label 0 and label 1 per pixel; label 1 is cheaper where score > tau."""
    d = score - tau
    return np.maximum(d, 0.0), np.maximum(-d, 0.0)      # (D0, D1), kept >= 0 for max-flow


def _neighbour_counts(x):
    """Per pixel: n1 = 4-neighbours labelled 1, nb = number of 4-neighbours (no wrap-around)."""
    xi = x.astype(np.int16)
    n1 = np.zeros(x.shape, np.int16)
    nb = np.zeros(x.shape, np.int16)
    n1[1:, :] += xi[:-1, :]; nb[1:, :] += 1             # neighbour above
    n1[:-1, :] += xi[1:, :]; nb[:-1, :] += 1            # neighbour below
    n1[:, 1:] += xi[:, :-1]; nb[:, 1:] += 1             # neighbour to the left
    n1[:, :-1] += xi[:, 1:]; nb[:, :-1] += 1            # neighbour to the right
    return n1, nb


# ---- Greedy solver (ICM), only used for comparison ----

def solve_icm(score, tau, beta, iters=8):
    """Each pixel repeatedly takes its cheaper label given its neighbours (local minimum only)."""
    D0, D1 = unary(score, tau)
    x = (score > tau)                                   # start from plain thresholding
    ii, jj = np.indices(x.shape)
    parity = (ii + jj) % 2                              # checkerboard: neighbours never update together

    for _ in range(iters):
        for p in (0, 1):                                # update one checkerboard colour at a time
            n1, nb = _neighbour_counts(x)
            c0 = D0 + beta * n1                         # cost of 0 = unary + disagreeing with 1-neighbours
            c1 = D1 + beta * (nb - n1)                  # cost of 1 = unary + disagreeing with 0-neighbours
            m = parity == p
            x = np.where(m, c1 < c0, x)
    return x


# ---- Exact solver (graph cut), used for rungs 2 and 5 ----

def solve_graphcut(score, tau, beta):
    """Global minimum of E via s-t min cut (exact because the Potts energy is submodular)."""
    import maxflow
    D0, D1 = unary(score, tau)
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(score.shape)               # one node per pixel
    if beta > 0:
        g.add_grid_edges(nodes, beta)                   # 4-neighbour edges with weight beta
    g.add_grid_tedges(nodes, D1, D0)                    # source edge = cost of label 1, sink edge = cost of 0
    g.maxflow()                                         # solve the min cut
    return np.asarray(g.get_grid_segments(nodes))       # True = label 1 (spliced)


def energy(score, x, tau, beta):
    """E(x) computed directly, to check the solvers."""
    D0, D1 = unary(score, tau)
    e = np.where(x, D1, D0).sum()                       # unary part
    e += beta * (x[:, 1:] != x[:, :-1]).sum()           # horizontal disagreements
    e += beta * (x[1:, :] != x[:-1, :]).sum()           # vertical disagreements
    return float(e)
