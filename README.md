# SSC-SP: semantic-spatial alignment and decomposed structural preservation

Reproduction package for the manuscript
*"Semantic alignment can diverge from local neighborhood preservation in dimensionality reduction visualizations."*

This package measures two aspects of a dimensionality-reduction (DR) visualization
separately and reports them side by side.

## What this measures

- **SSC** — Spearman rank correlation between reference-space pairwise distances and
  2D spatial pairwise distances (global semantic-spatial alignment).
- **SP**, reported as decomposed components:
  - **SP_adj** — mean per-item Jaccard overlap of k-nearest-neighbor sets
    (local neighborhood preservation).
  - **SP_ord** — Kendall's tau of pairwise-distance ranks, rescaled to [0, 1]
    (global rank fidelity; near-redundant with SSC by construction).
  - **SP_clu** — Adjusted Rand Index of K-means labels (cluster agreement).
  - **SP_total** — mean of the three components.

Two reference modes:
- **layout-preservation** — reference is the unperturbed baseline layout
  (synthetic perturbation experiments); SP = 1.0 at zero perturbation by construction.
- **reference-space** — reference is the high-dimensional representation
  (applied demonstrations: SPoSE embedding for THINGS, pixel features for MNIST).

## Reproduce

```bash
pip install -r requirements.txt
pip install -e .

python scripts/run_synthetic.py       # perturbation experiments (Figure 1)
python scripts/run_baseline_null.py   # random-pairing null for SSC (Figure 2)
python scripts/run_lambda.py          # value-coupling sweep (Figure 3)
python scripts/reproduce_things.py    # THINGS demonstration (Table 1, Figure 4)
python scripts/reproduce_mnist.py     # MNIST demonstration
python scripts/run_robustness.py      # robustness checks (supplementary)
python scripts/make_figures.py        # render all figures from result CSVs
python tables/generate_hyperparameter_table.py   # hyperparameter table from config/

pytest -q                             # metric definitions, determinism, banned terms
```

All hyperparameters live in `config/*.yaml` and are the single source for the
hyperparameter table. Random seeds, reference-distance definitions, and DR
settings are recorded there.

## Data

The THINGS SPoSE concept embedding is not redistributed here. Download it from
OSF (https://osf.io/z2784/) and place the embedding file under `data/things/`.
See `data/things/README.md` for the exact filename and steps.

The BERT embedding used in the value-coupling sweep is shipped precomputed under
`data/intermediate/` so that base reproduction runs without `torch` or
`transformers`. To regenerate it, install `requirements-bert.txt` and run
`scripts/prepare_bert_embeddings.py`.

## Determinism

Computations run under a deterministic execution contract: fixed seeds and
single-threaded BLAS (`OPENBLAS/MKL/OMP_NUM_THREADS=1`). The recorded reference
environment is Python 3.10.19, numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.0.
Bitwise identity across all environments is not guaranteed.

## Citation and license

See `CITATION.cff`. Released under the MIT License (`LICENSE`).
Archived release DOI: *(to be added on Zenodo deposit)*.
