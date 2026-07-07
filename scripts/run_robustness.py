#!/usr/bin/env python3
"""Robustness checks (supplementary figures) under the unified library.

Covers three sweeps referenced by the supplementary material:
  - dimension / N robustness of the topology-vs-metric contrast,
  - layout x topology generalization,
  - k-NN neighborhood-size sensitivity of SP.

Reference mode: layout-preservation. Outputs CSV summaries to data/results/.
Values use the manuscript definition (per-item SP_adj, k=10 unless swept).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ssc_sp import deterministic  # noqa: F401
from ssc_sp import metrics as M
from ssc_sp import layouts, perturbations as P

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "synthetic.yaml").read_text())
OUT = ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = CFG["generation"]["seed"]
KC = CFG["metrics"]["k_clu"]
N_TRIALS = CFG["generation"]["n_trials"]


def dimension_n_robustness():
    """SP under topology vs metric perturbation across grid sizes (N)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n_side in (4, 6, 8, 10):
        n = n_side * n_side
        base = layouts.grid_layout(n_side)
        topo = np.mean([M.sp_total(base, P.permute_fraction(base, rng, 0.7),
                                   k=10, n_clusters=KC, random_state=SEED)
                        for _ in range(N_TRIALS)])
        metric = M.sp_total(base, P.shear(base, 0.7), k=10, n_clusters=KC, random_state=SEED)
        rows.append(dict(n_items=n, sp_topology=topo, sp_metric=metric))
    return pd.DataFrame(rows)


def layout_topology_robustness():
    """Topology vs metric at matched intensity across layout types."""
    rng = np.random.default_rng(SEED + 3)
    n = CFG["generation"]["n_items"]
    rows = []
    for kind in ("grid", "line", "circle", "random"):
        base = layouts.make_layout(kind, n, rng)
        topo = np.mean([M.sp_total(base, P.permute_fraction(base, rng, 0.7),
                                   k=10, n_clusters=KC, random_state=SEED)
                        for _ in range(N_TRIALS)])
        metric = M.sp_total(base, P.shear(base, 0.7), k=10, n_clusters=KC, random_state=SEED)
        rows.append(dict(layout=kind, sp_topology=topo, sp_metric=metric))
    return pd.DataFrame(rows)


def knn_k_robustness():
    """SP_total under a fixed moderate perturbation across k for SP_adj."""
    rng = np.random.default_rng(SEED + 4)
    n = CFG["generation"]["n_items"]
    base = layouts.grid_layout(int(round(np.sqrt(n))))
    rows = []
    for k in (5, 10, 20):
        vals = [M.sp_total(base, P.add_coord_noise(base, rng, 0.1),
                           k=k, n_clusters=KC, random_state=SEED)
                for _ in range(N_TRIALS)]
        rows.append(dict(k=k, sp_mean=float(np.mean(vals)), sp_std=float(np.std(vals, ddof=1))))
    return pd.DataFrame(rows)


def main():
    d = dimension_n_robustness()
    lt = layout_topology_robustness()
    kk = knn_k_robustness()
    d.to_csv(OUT / "supp_dimension_n_robustness.csv", index=False)
    lt.to_csv(OUT / "supp_layout_topology_robustness.csv", index=False)
    kk.to_csv(OUT / "supp_knn_k_robustness.csv", index=False)
    print("=== dimension/N robustness ===")
    print(d.round(4).to_string(index=False))
    print("\n=== layout x topology robustness ===")
    print(lt.round(4).to_string(index=False))
    print("\n=== k-NN size sensitivity ===")
    print(kk.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
