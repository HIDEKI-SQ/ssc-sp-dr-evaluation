"""Determinism tests: identical seeds must give identical metric outputs.

These lock in the deterministic execution contract at the metric level, so a
change that introduces nondeterminism (e.g. dropping a fixed random_state) is
caught by CI.
"""

import numpy as np

from ssc_sp import metrics as M


def _pair(seed=3776, n=64):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 2))
    B = A + rng.normal(0, 0.3, size=(n, 2))
    return A, B


def test_sp_adj_deterministic():
    A, B = _pair()
    assert M.sp_adj(A, B, k=10) == M.sp_adj(A, B, k=10)


def test_sp_clu_deterministic_same_seed():
    A, B = _pair()
    assert M.sp_clu(A, B, n_clusters=4, random_state=7) == \
           M.sp_clu(A, B, n_clusters=4, random_state=7)


def test_sp_components_deterministic():
    A, B = _pair()
    c1 = M.sp_components(A, B, k=10, n_clusters=4, random_state=1)
    c2 = M.sp_components(A, B, k=10, n_clusters=4, random_state=1)
    assert (c1.sp_adj, c1.sp_ord, c1.sp_clu) == (c2.sp_adj, c2.sp_ord, c2.sp_clu)


def test_ssc_deterministic():
    from scipy.spatial.distance import pdist, squareform
    A, _ = _pair()
    d = squareform(pdist(A))
    assert M.ssc(d, A) == M.ssc(d, A)
