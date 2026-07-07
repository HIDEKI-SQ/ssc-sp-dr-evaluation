#!/usr/bin/env python3
"""THINGS demonstration (Table 1, Figure 4) under the unified library.

Reference mode: reference-space. SP and SSC are computed against the
high-dimensional SPoSE embedding (behavioral similarity). Each seed controls
both the 500-concept subsample and the DR initialization; reported values are
mean +/- SD across the three seeds.

Requires the SPoSE embedding (not redistributed). Download from OSF
(https://osf.io/z2784/) and pass its path via --embedding or the SPOSE_PATH
environment variable. Without it, this script exits with instructions; run it on
Colab (see colab_things_reproduction) where the file is available.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform

from ssc_sp import deterministic  # noqa: F401
from ssc_sp import metrics as M

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "things.yaml").read_text())
OUT = ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load_spose(path: str) -> np.ndarray:
    x = np.loadtxt(path)
    if x.shape[0] < x.shape[1]:  # stored as d x n_concepts
        x = x.T
    return x


def subsample(x: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]


def run_dr(d_sem: np.ndarray, seed: int) -> dict:
    from sklearn.manifold import MDS, TSNE
    import umap
    Y = {}
    Y["MDS"] = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
                   n_init=4, max_iter=300, normalized_stress="auto").fit_transform(d_sem)
    Y["tSNE"] = TSNE(n_components=2, metric="precomputed", random_state=seed,
                     init="random", perplexity=30,
                     learning_rate="auto").fit_transform(d_sem)
    Y["UMAP"] = umap.UMAP(n_components=2, metric="precomputed", random_state=seed,
                          n_neighbors=15, min_dist=0.1).fit_transform(d_sem)
    return Y


def evaluate(embedding_path: str) -> pd.DataFrame:
    X = load_spose(embedding_path)
    n = CFG["sampling"]["n_concepts"]
    seeds = CFG["sampling"]["seeds"]
    k_nn = CFG["metrics"]["k_nn"]["default"]
    k_clu = CFG["metrics"]["k_clu"]

    records = []
    for seed in seeds:
        X_sub = subsample(X, n, seed)
        d_sem = squareform(pdist(X_sub, metric="euclidean"))
        for method, Y in run_dr(d_sem, seed).items():
            comps = M.sp_components(X_sub, Y, d_ref=d_sem, k=k_nn,
                                    n_clusters=k_clu, random_state=seed)
            records.append(dict(seed=seed, method=method,
                                SSC=M.ssc(d_sem, Y),
                                SP_adj=comps.sp_adj, SP_ord=comps.sp_ord,
                                SP_clu=comps.sp_clu, SP_total=comps.total))
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby("method")[["SSC", "SP_adj", "SP_ord", "SP_clu", "SP_total"]]
             .agg(["mean", "std"]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default=os.environ.get("SPOSE_PATH"))
    args = ap.parse_args()
    if not args.embedding or not Path(args.embedding).exists():
        print("SPoSE embedding not found.")
        print("Download embedding00_epoch0500.txt from https://osf.io/z2784/ and")
        print("pass --embedding <path> or set SPOSE_PATH. On Colab, see the")
        print("THINGS reproduction notebook.")
        return 1

    raw = evaluate(args.embedding)
    raw.to_csv(OUT / "things_table1_raw.csv", index=False)
    summ = summarize(raw)
    summ.to_csv(OUT / "things_table1_summary.csv")
    print("=== Table 1 (mean +/- SD across seeds 20250124-26, N=500, k=10) ===")
    print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
