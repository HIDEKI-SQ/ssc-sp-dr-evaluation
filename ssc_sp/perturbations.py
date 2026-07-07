"""Perturbations and value coupling used by the synthetic experiments.

Metric distortions (rotation, scaling, shear) and coordinate noise act on
coordinates directly. Topology disruption (permutation of a proportion of
points) also acts on coordinates, so all of these are compatible with the
coordinate-based per-item SP_adj definition.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def add_coord_noise(coords: np.ndarray, rng: np.random.Generator,
                    sigma: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return coords + rng.normal(0.0, sigma, size=coords.shape)


def rotate(coords: np.ndarray, theta_deg: float,
           center=(0.0, 0.0)) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    t = np.deg2rad(theta_deg)
    cx, cy = center
    x = coords[:, 0] - cx
    y = coords[:, 1] - cy
    c, s = np.cos(t), np.sin(t)
    return np.stack([c * x - s * y + cx, s * x + c * y + cy], axis=1)


def scale(coords: np.ndarray, sx: float, sy: float,
          center=(0.0, 0.0)) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    cx, cy = center
    return np.stack([sx * (coords[:, 0] - cx) + cx,
                     sy * (coords[:, 1] - cy) + cy], axis=1)


def shear(coords: np.ndarray, k: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return np.stack([coords[:, 0] + k * coords[:, 1], coords[:, 1]], axis=1)


def permute_fraction(coords: np.ndarray, rng: np.random.Generator,
                     p: float) -> np.ndarray:
    """Permute the positions of a proportion p of points (topology disruption).

    A random subset of size round(p*N) has its coordinate rows shuffled among
    themselves; the rest stay fixed. This destroys local neighborhood structure
    while leaving the set of occupied positions unchanged.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    out = coords.copy()
    if p <= 0.0:
        return out
    m = int(round(p * n))
    if m < 2:
        return out
    idx = rng.choice(n, size=m, replace=False)
    perm = rng.permutation(idx)
    out[idx] = coords[perm]
    return out


def value_coupled_layout(baseline: np.ndarray, embeddings: np.ndarray,
                         lam: float) -> np.ndarray:
    """coordinates = (1 - lam) * baseline + lam * semantic_projection.

    semantic_projection is the 2D PCA of the embeddings, aligned in scale to the
    baseline so the mixture is meaningful at intermediate lambda.
    """
    baseline = np.asarray(baseline, dtype=float)
    if lam == 0.0:
        return baseline.copy()
    proj = PCA(n_components=2, random_state=0).fit_transform(
        np.asarray(embeddings, dtype=float))
    # Standardize projection to baseline's per-axis scale.
    proj = (proj - proj.mean(0)) / (proj.std(0) + 1e-12)
    proj = proj * (baseline.std(0) + 1e-12) + baseline.mean(0)
    if lam == 1.0:
        return proj
    return (1.0 - lam) * baseline + lam * proj
