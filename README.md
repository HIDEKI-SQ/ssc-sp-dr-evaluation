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

Install, pinning the single-threaded BLAS settings the deterministic execution
contract expects:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
pip install -r requirements.txt
pip install -e .
```

### 1. Quick verification from shipped files (seconds)

No external data or network needed. Checks the library, the shipped Table 1
values, banned terms, and renders figures from the shipped result CSVs.

```bash
pytest -q
python tables/generate_hyperparameter_table.py
python scripts/make_figures.py
```

`tests/test_reproduce_table1.py` pins the values in the shipped THINGS summary
CSV; it is a value-fixing test, not a full reproduction from SPoSE.

### 2. Full synthetic reproduction (several minutes to tens of minutes)

Regenerates Figures 1-3 from scratch. Runtime scales with `n_trials` in the
configs (200 by default) and the K-means calls inside SP, and depends on CPU and
BLAS configuration; it may take several minutes to tens of minutes. For a quick
check instead, run the pytest suite and `make_figures.py` from the shipped CSVs
(section 1).

```bash
python scripts/run_baseline_null.py   # random-pairing null for SSC (Figure 2)
python scripts/run_synthetic.py       # perturbation experiments (Figure 1)
python scripts/run_lambda.py          # value-coupling sweep (Figure 3; needs BERT npz)
python scripts/run_robustness.py      # robustness checks (supplementary)
```

The BERT side of Figure 3 uses the shipped `data/intermediate/bert_embeddings.npz`.

### 3. Full applied-data reproduction (external data required)

- **THINGS** (Table 1, Figure 4): requires the SPoSE embedding file. Download
  `embedding00_epoch0500.txt` from https://osf.io/z2784/ into `data/things/`
  (see `data/things/README.md`), then:
  ```bash
  python scripts/reproduce_things.py --embedding data/things/embedding00_epoch0500.txt
  ```
- **MNIST**: requires network access (OpenML via `fetch_openml`):
  ```bash
  python scripts/reproduce_mnist.py
  ```

All hyperparameters live in `config/*.yaml` and are the single source for the
hyperparameter table and for the DR calls in the reproduction scripts. Random
seeds, reference-distance definitions, and DR settings are recorded there.

## Data

The THINGS SPoSE concept embedding is not redistributed here. Download it from
OSF (https://osf.io/z2784/) and place the embedding file under `data/things/`.
See `data/things/README.md` for the exact filename and steps.

The BERT embedding used in the value-coupling sweep is shipped precomputed under
`data/intermediate/` so that base reproduction runs without `torch` or
`transformers`, which is the intended path. Regenerating it is optional: install
`requirements-bert.txt` and run `scripts/prepare_bert_embeddings.py`. Bitwise
reproduction of the shipped embedding is not guaranteed across transformers/torch
versions; the recorded generation run used transformers 5.12.1 and torch
2.11.0+cpu (see `data/intermediate/bert_embeddings_provenance.json`).

## Determinism

Computations run under a deterministic execution contract: fixed seeds and
single-threaded BLAS (`OPENBLAS/MKL/OMP_NUM_THREADS=1`). The recorded reference
environment is Python 3.10.19, numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.0.
Bitwise identity across all environments is not guaranteed.

## Citation and license

See `CITATION.cff`. Released under the MIT License (`LICENSE`).
Archived release DOI: *(to be added on Zenodo deposit)*.
