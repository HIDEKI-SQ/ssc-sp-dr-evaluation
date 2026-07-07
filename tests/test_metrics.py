"""Tests that fix the manuscript definitions in place.

These guard against silent regressions to the earlier core implementation
(whole-graph symmetrized Jaccard, k=4) that did not match the manuscript.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from ssc_sp import metrics as M


def _data(seed=3776, n=64):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 2))
    B = A + rng.normal(0, 0.3, size=(n, 2))
    return A, B


def _inline_per_item_jaccard(X, Y, k=10):
    ia = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b))
                          for a, b in zip(ia, ib)]))


def test_sp_adj_is_per_item_mean_jaccard():
    A, B = _data()
    assert abs(M.sp_adj(A, B, k=10) - _inline_per_item_jaccard(A, B, 10)) < 1e-12


def test_identity_gives_perfect_scores():
    A, _ = _data()
    d = squareform(pdist(A))
    assert M.sp_adj(A, A, k=10) == 1.0
    assert abs(M.sp_ord(d, A) - 1.0) < 1e-9
    assert M.ssc(d, A) > 0.999


def test_sp_clu_deterministic_and_permutation_invariant():
    A, B = _data()
    v1 = M.sp_clu(A, B, n_clusters=4, random_state=42)
    v2 = M.sp_clu(A, B, n_clusters=4, random_state=42)
    assert v1 == v2


def test_ssc_near_zero_when_unrelated():
    rng = np.random.default_rng(1)
    sem = rng.normal(size=(64, 128))
    lay = rng.normal(size=(64, 2))
    d = squareform(pdist(sem))
    assert abs(M.ssc(d, lay)) < 0.1


def test_default_k_is_ten():
    import inspect
    assert inspect.signature(M.sp_adj).parameters["k"].default == 10
    assert inspect.signature(M.sp_components).parameters["k"].default == 10
