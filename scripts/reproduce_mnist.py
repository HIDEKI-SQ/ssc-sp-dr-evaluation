#!/usr/bin/env python3
"""MNIST demonstration under the unified library (manuscript conditions).

Reference mode: reference-space. The reference distance is pixel-space
input-feature distance (Euclidean on 784-d pixel vectors), NOT semantic
distance in the THINGS sense. 1,000 images are randomly sampled from the MNIST
test set with a fixed seed. SP_adj uses the per-item definition at k=10.

Requires fetch_openml (network) and umap-learn 0.5.12; run on Colab (see the
MNIST reproduction notebook). Values differ from the original submission, which
used the legacy core SP implementation (whole-graph Jaccard, k=4); qualitative
patterns are checked against docs/legacy_mapping.md.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform

from ssc_sp import deterministic  # noqa: F401
from ssc_sp import metrics as M

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "mnist.yaml").read_text())
OUT = ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load_mnist_test(n: int, seed: int):
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
    X_full = mnist.data.astype(float)        # pixels 0-255, unnormalized
    # Conventional split: last 10,000 are the test set.
    X_test = X_full[60000:]
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_test.shape[0], size=n, replace=False)
    return X_test[idx]


def run():
    n = CFG["sampling"]["n_samples"]
    seed = CFG["sampling"]["sampling_seed"]
    k_nn = CFG["metrics"]["k_nn"]
    k_clu = CFG["metrics"]["k_clu"]

    X = load_mnist_test(n, seed)
    d_ref = squareform(pdist(X, metric="euclidean"))

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap

    Y = {
        "tSNE": TSNE(n_components=2, metric="euclidean", random_state=seed,
                     init="random", perplexity=30, n_iter=1000,
                     learning_rate="auto").fit_transform(X),
        "UMAP": umap.UMAP(n_components=2, metric="euclidean", random_state=seed,
                          n_neighbors=15, min_dist=0.1).fit_transform(X),
        "PCA": PCA(n_components=2, random_state=seed).fit_transform(X),
    }

    rows = []
    for method, coords in Y.items():
        comps = M.sp_components(X, coords, d_ref=d_ref, k=k_nn,
                                n_clusters=k_clu, random_state=seed)
        rows.append(dict(method=method, SSC=M.ssc(d_ref, coords),
                         SP_adj=comps.sp_adj, SP_ord=comps.sp_ord,
                         SP_clu=comps.sp_clu, SP_total=comps.total))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mnist_results.csv", index=False)
    print("=== MNIST (N=%d, test set, seed=%d, per-item k=10) ===" % (n, seed))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
