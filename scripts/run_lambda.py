#!/usr/bin/env python3
"""Value-coupling sweep (Figure 3), synthetic side, under the unified library.

The BERT side requires the precomputed embedding at
data/intermediate/bert_embeddings.npz. If absent (e.g. torch not installed for
regeneration), only the synthetic side is produced and the BERT side is
reported as skipped.

Reference mode: layout-preservation.
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
from ssc_sp import layouts, perturbations as P

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "lambda.yaml").read_text())
OUT = ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = CFG["generation"]["n_items"]
SEED = CFG["generation"]["seed"]
LAMBDAS = CFG["generation"]["lambda_values"]
K = CFG["metrics"]["k_nn"]
KC = CFG["metrics"]["k_clu"]
DIM = CFG["embeddings"]["synthetic"]["dim"]
N_TRIALS = CFG["generation"]["n_trials"]


def _dist(x):
    return squareform(pdist(np.asarray(x, dtype=float)))


def sweep(get_embeddings, label, n, shuffle=False):
    rng = np.random.default_rng(SEED)
    base = layouts.circle_layout(n)
    rows = []
    for lam in LAMBDAS:
        sp, ssc = [], []
        for _ in range(N_TRIALS):
            emb = get_embeddings(rng)
            if shuffle:                      # word_shuffling: reassign concepts to positions
                emb = emb[rng.permutation(n)]
            coords = P.value_coupled_layout(base, emb, lam)
            sp.append(M.sp_total(base, coords, k=K, n_clusters=KC, random_state=SEED))
            ssc.append(M.ssc(_dist(emb), coords))
        rows.append(dict(lam=lam, sp_mean=np.mean(sp), sp_std=np.std(sp, ddof=1),
                         ssc_mean=np.mean(ssc), ssc_std=np.std(ssc, ddof=1)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"fig3_lambda_{label}.csv", index=False)
    return df


def main():
    def synth(rng):
        x = rng.normal(size=(N, DIM))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    print("=== Fig 3: lambda sweep (synthetic, N=%d) ===" % N)
    print(sweep(synth, "synth", N, shuffle=False).round(4).to_string(index=False))

    bert_path = ROOT / "data" / "intermediate" / "bert_embeddings.npz"
    if bert_path.exists():
        data = np.load(bert_path, allow_pickle=True)["embeddings"]
        n_bert = data.shape[0]
        print(f"\n=== Fig 3: lambda sweep (BERT, precomputed, N={n_bert}, word_shuffling) ===")
        print(sweep(lambda rng: data, "bert", n_bert, shuffle=True).round(4).to_string(index=False))
    else:
        print("\n[SKIPPED] BERT side: data/intermediate/bert_embeddings.npz not found.")
        print("          Provide the precomputed embedding, or regenerate with")
        print("          scripts/prepare_bert_embeddings.py (requires requirements-bert.txt).")


if __name__ == "__main__":
    main()
