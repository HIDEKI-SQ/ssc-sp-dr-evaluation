#!/usr/bin/env python3
"""Synthetic perturbation experiments (Figure 1) under the manuscript definition.

Reference mode: layout-preservation. SP is computed against the unperturbed
baseline layout, so SP = 1.0 at zero perturbation by construction. SSC is
computed against unrelated semantic embeddings and stays near zero.

Outputs CSV summaries to data/results/. Values differ from the legacy core
figures (per-item Jaccard and k=10 here vs whole-graph Jaccard and k=4 there);
qualitative patterns are checked against docs/legacy_mapping.md.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ssc_sp import deterministic  # noqa: F401  (enforces single-thread BLAS on import)
from ssc_sp import metrics as M
from ssc_sp import layouts, perturbations as P

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "synthetic.yaml").read_text())
OUT = ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = CFG["generation"]["n_items"]
DIM = CFG["generation"]["dim"]
SEED = CFG["generation"]["seed"]
K = CFG["metrics"]["k_nn"]
KC = CFG["metrics"]["k_clu"]
N_TRIALS = CFG["generation"]["n_trials"]


def _semantic(rng):
    x = rng.normal(size=(N, DIM))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def coordinate_noise():
    rng = np.random.default_rng(SEED)
    base = layouts.grid_layout(int(round(np.sqrt(N))))
    rows = []
    for sigma in CFG["coordinate_noise"]["sigma_values"]:
        sp, ssc = [], []
        for _ in range(N_TRIALS):
            sem = _semantic(rng)
            noisy = P.add_coord_noise(base, rng, sigma)
            sp.append(M.sp_total(base, noisy, k=K, n_clusters=KC, random_state=SEED))
            ssc.append(M.ssc(_dist(sem), noisy))
        rows.append(dict(sigma=sigma, sp_mean=np.mean(sp), sp_std=np.std(sp, ddof=1),
                         ssc_mean=np.mean(ssc), ssc_std=np.std(ssc, ddof=1)))
    return pd.DataFrame(rows)


def metric_vs_topology():
    rng = np.random.default_rng(SEED + 2)
    base = layouts.grid_layout(int(round(np.sqrt(N))))
    lvl = CFG["metric_vs_topology"]["matched_intensity"]
    rows = []
    # rotation (metric)
    for deg in CFG["metric_vs_topology"]["rotation_deg"]:
        v = M.sp_total(base, P.rotate(base, deg), k=K, n_clusters=KC, random_state=SEED)
        rows.append(dict(family="rotation", level=deg, sp=v))
    # shear (metric) at matched intensity
    sp_metric = M.sp_total(base, P.shear(base, lvl), k=K, n_clusters=KC, random_state=SEED)
    rows.append(dict(family="metric_shear", level=lvl, sp=sp_metric))
    # topology permutation at matched intensity (averaged over trials)
    vals = [M.sp_total(base, P.permute_fraction(base, rng, lvl),
                       k=K, n_clusters=KC, random_state=SEED) for _ in range(N_TRIALS)]
    rows.append(dict(family="topology_permute", level=lvl, sp=float(np.mean(vals))))
    return pd.DataFrame(rows)


def semantic_noise():
    """Negative control: perturbing the embedding leaves SP and SSC unaffected."""
    rng = np.random.default_rng(SEED + 1)
    base = layouts.grid_layout(int(round(np.sqrt(N))))
    # SP compares base to itself here, so it is constant across trials: compute once.
    sp_const = M.sp_total(base, base, k=K, n_clusters=KC, random_state=SEED)
    rows = []
    for sigma in CFG["semantic_noise"]["sigma_values"]:
        ssc = []
        for _ in range(N_TRIALS):
            sem = _semantic(rng)
            sem_noisy = sem + rng.normal(0, sigma, size=sem.shape)
            ssc.append(M.ssc(_dist(sem_noisy), base))
        rows.append(dict(sigma=sigma, sp_mean=sp_const,
                         ssc_mean=np.mean(ssc), ssc_std=np.std(ssc, ddof=1)))
    return pd.DataFrame(rows)


def _dist(x):
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(np.asarray(x, dtype=float)))


def main():
    cn = coordinate_noise()
    mt = metric_vs_topology()
    sn = semantic_noise()
    cn.to_csv(OUT / "fig1a_coordinate_noise.csv", index=False)
    mt.to_csv(OUT / "fig1b_metric_vs_topology.csv", index=False)
    sn.to_csv(OUT / "fig1_semantic_noise_control.csv", index=False)

    print("=== Fig 1a: coordinate noise (SP falls, SSC ~ 0) ===")
    print(cn.round(4).to_string(index=False))
    print("\n=== Fig 1b: metric vs topology at matched intensity ===")
    print(mt.round(4).to_string(index=False))
    print("\n=== Fig 1 negative control: semantic noise (SP=1, SSC ~ 0) ===")
    print(sn.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
