"""Measurement library for SSC and decomposed SP.

Definitions follow the revised manuscript exactly:
  SSC     Spearman rank correlation between reference-space and 2D spatial
          pairwise-distance vectors.
  SP_adj  Mean per-item Jaccard overlap between the k-nearest-neighbor set in
          the reference space and the k-nearest-neighbor set in the 2D layout.
  SP_ord  Kendall's tau between the two pairwise-distance vectors, rescaled to
          [0, 1].
  SP_clu  Adjusted Rand Index between K-means labels of the reference and the
          2D layout.
  SP_total = (SP_adj + SP_ord + SP_clu) / 3.

Two reference modes are supported through the caller's choice of `ref`:
  layout      ref is the unperturbed baseline layout (synthetic perturbation
              experiments); SP = 1.0 at zero perturbation by construction.
  reference   ref is the high-dimensional reference representation (applied
              demonstrations, e.g. SPoSE for THINGS, pixels for MNIST).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


@dataclass
class SPComponents:
    sp_adj: float
    sp_ord: float
    sp_clu: float

    @property
    def total(self) -> float:
        return float((self.sp_adj + self.sp_ord + self.sp_clu) / 3.0)


def _condensed(mat: np.ndarray) -> np.ndarray:
    return squareform(np.asarray(mat, dtype=float), checks=False)


def knn_sets(coords: np.ndarray, k: int) -> list[set[int]]:
    """Per-item k-nearest-neighbor index sets, self excluded."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if k >= n:
        raise ValueError(f"k must be < N; got k={k}, N={n}")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    return [set(row) for row in idx]


def ssc(d_ref: np.ndarray, coords_2d: np.ndarray) -> float:
    """Semantic/reference-spatial correlation (Spearman rho)."""
    d_spa = pdist(np.asarray(coords_2d, dtype=float), metric="euclidean")
    rho, _ = spearmanr(_condensed(d_ref), d_spa)
    return float(rho)


def sp_adj(ref_coords: np.ndarray, coords_2d: np.ndarray, k: int = 10) -> float:
    """Mean per-item Jaccard overlap of k-NN sets (reference vs 2D layout)."""
    a = knn_sets(ref_coords, k)
    b = knn_sets(coords_2d, k)
    vals = []
    for sa, sb in zip(a, b):
        union = len(sa | sb)
        vals.append(len(sa & sb) / union if union else 1.0)
    return float(np.mean(vals))


def sp_ord(d_ref: np.ndarray, coords_2d: np.ndarray) -> float:
    """Kendall's tau of pairwise-distance ranks, mapped to [0, 1]."""
    d_spa = pdist(np.asarray(coords_2d, dtype=float), metric="euclidean")
    tau, _ = kendalltau(_condensed(d_ref), d_spa)
    if np.isnan(tau):
        return 0.5
    return float((tau + 1.0) / 2.0)


def sp_clu(ref_coords: np.ndarray, coords_2d: np.ndarray,
           n_clusters: int, random_state: int) -> float:
    """Adjusted Rand Index between K-means labels of reference and layout.

    ARI is invariant to label permutations; therefore no label alignment is
    applied.
    """
    ref = np.asarray(ref_coords, dtype=float)
    lay = np.asarray(coords_2d, dtype=float)
    lab_ref = KMeans(n_clusters=n_clusters, n_init=10,
                     random_state=random_state).fit_predict(ref)
    lab_lay = KMeans(n_clusters=n_clusters, n_init=10,
                     random_state=random_state).fit_predict(lay)
    # Manuscript definition clips negative ARI (worse than chance) to 0.
    return float(max(0.0, adjusted_rand_score(lab_ref, lab_lay)))


def sp_components(ref_coords: np.ndarray, coords_2d: np.ndarray, *,
                  d_ref: np.ndarray | None = None,
                  k: int = 10, n_clusters: int = 4,
                  random_state: int = 42) -> SPComponents:
    """All SP components for one (reference, 2D layout) pair.

    Parameters
    ----------
    ref_coords : (N, d) reference representation or baseline layout.
    coords_2d : (N, 2) projected/perturbed layout.
    d_ref : optional precomputed (N, N) reference distance matrix. If None it is
        computed as Euclidean distances on ref_coords. For applied datasets that
        define the reference distance directly (e.g. precomputed SPoSE
        distances), pass it here to keep SP_ord consistent with SSC input.
    """
    if d_ref is None:
        d_ref = squareform(pdist(np.asarray(ref_coords, dtype=float),
                                 metric="euclidean"))
    return SPComponents(
        sp_adj=sp_adj(ref_coords, coords_2d, k=k),
        sp_ord=sp_ord(d_ref, coords_2d),
        sp_clu=sp_clu(ref_coords, coords_2d, n_clusters=n_clusters,
                      random_state=random_state),
    )


def sp_total(ref_coords: np.ndarray, coords_2d: np.ndarray, **kwargs) -> float:
    return sp_components(ref_coords, coords_2d, **kwargs).total
