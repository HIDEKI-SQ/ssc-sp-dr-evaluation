# Legacy mapping and pre-re-execution reference values

This file records (a) how the legacy development experiments map onto the
scripts in this package, and (b) the numeric values produced by the **legacy
core implementation** (whole-graph symmetrized Jaccard, k=4) that generated the
original manuscript figures. These legacy values are kept only as a reference
for checking that re-execution under the manuscript definition (per-item
Jaccard, k=10) preserves the qualitative patterns. The legacy experiments
themselves are not shipped in this package.

## Why values change on re-execution

- SP_adj definition: legacy core symmetrized the k-NN graph and took a single
  whole-graph Jaccard; the manuscript defines SP_adj as the mean per-item k-NN
  set Jaccard. On the same data at k=10 these differ substantially
  (e.g. 0.590 vs 0.469 on one synthetic pair).
- k: legacy synthetic runs used k=4; the manuscript default is k=10.

Absolute SP_adj values therefore move. The qualitative claims to preserve are:
noise increases -> SP falls while SSC stays near zero; rotation preserves SP,
neighborhood rewiring destroys it; lambda increases -> SSC rises and SP falls.

## Experiment -> script mapping

| legacy experiment            | manuscript target        | script                        | shipped |
|------------------------------|--------------------------|-------------------------------|---------|
| sp20_coord_noise_sp_ssc      | Figure 1a                | run_synthetic.py              | re-run  |
| sp10_metric_invariance       | Figure 1b (metric)       | run_synthetic.py              | re-run  |
| sp12_topology_vs_metric      | Figure 1b (topology)     | run_synthetic.py              | re-run  |
| sp21_semantic_noise_sp_ssc   | Figure 1 negative control| run_synthetic.py              | re-run  |
| sp22_mixed_noise_grid        | Supplementary            | run_synthetic.py              | re-run  |
| (SSC null, src/core)         | Figure 2                 | run_baseline_null.py          | re-run  |
| sp30_lambda_sweep_synth      | Figure 3 (synthetic)     | run_lambda.py                 | re-run  |
| sp31_lambda_sweep_bert       | Figure 3 (BERT)          | run_lambda.py                 | re-run  |
| THINGS notebook              | Table 1, Figure 4        | reproduce_things.py           | re-run  |
| MNIST demo                   | MNIST results            | reproduce_mnist.py            | new     |
| sp40_dimN_sp_robustness      | Supplementary S1         | run_robustness.py             | re-run  |
| sp41_layout_topology_robust  | Supplementary S2         | run_robustness.py             | re-run  |
| sp42_knn_k_robustness        | Supplementary S4         | run_robustness.py             | re-run  |
| sp13_layout_generalization   | "consistent across ..."  | run_synthetic.py              | re-run  |
| sp00/01/02/03/11/50/51       | not in manuscript        | -                             | omitted |

## Legacy reference values (pre-re-execution)

### Figure 1a - coordinate noise (sp20; SP against baseline layout, k=4 legacy)

| sigma | SP    | SSC     |
|-------|-------|---------|
| 0.0   | 1.000 | -0.000  |
| 0.1   | 0.609 | -0.001  |
| 0.3   | 0.334 |  0.001  |
| 0.5   | 0.251 |  0.001  |
| 0.7   | 0.219 |  0.000  |

### Figure 1b - rotation vs rewiring (sp10 / sp12, k=4 legacy)

| transform         | level | SP    |
|-------------------|-------|-------|
| rotation          | 30    | 0.943 |
| rotation          | 180   | 0.967 |
| topology rewiring | 0.7   | 0.343 |
| metric (shear)    | 0.7   | 0.719 |

Delta(rewiring vs metric at 0.7) approximately 0.38.

### Figure 3 - lambda sweep (sp30 synth / sp31 BERT, k=4 legacy)

| lambda | SP (synth) | SSC (synth) | SP (BERT) | SSC (BERT) |
|--------|------------|-------------|-----------|------------|
| 0.0    | 1.000      | -0.001      | 1.000     | -0.000     |
| 0.6    | 0.301      |  0.139      | 0.425     |  0.276     |
| 1.0    | 0.183      |  0.289      | 0.186     |  0.718     |

### Table 1 - THINGS (notebook, per-item, k=10; N=500, seeds 20250124-26)

Reproduced end-to-end from the real SPoSE embedding under the unified library.
MDS and t-SNE reproduce the manuscript values exactly (scikit-learn is
deterministic). UMAP is sensitive to umap-learn version (numba JIT); the values
below are those reproduced with umap-learn 0.5.12, which is pinned in
requirements. The revised manuscript Table 1 should use these UMAP values.

| method | SSC          | SP_adj       | SP_ord       | SP_clu       | SP_total     |
|--------|--------------|--------------|--------------|--------------|--------------|
| MDS    | 0.725+/-0.010| 0.127+/-0.005| 0.772+/-0.004| 0.367+/-0.027| 0.422+/-0.011|
| t-SNE  | 0.661+/-0.015| 0.350+/-0.006| 0.738+/-0.007| 0.507+/-0.057| 0.532+/-0.018|
| UMAP   | 0.633+/-0.006| 0.304+/-0.006| 0.726+/-0.003| 0.504+/-0.071| 0.511+/-0.026|

NOTES:
- Manuscript Table 1 prints the MDS SP_adj SD as 0.015; the correct value is
  0.005 (0.015 is the N=300 condition). Fix in the revised manuscript.
- The original submission's UMAP row (SP_clu 0.550, SP_total 0.525) was produced
  with a different umap-learn build. Under the pinned umap-learn 0.5.12 the UMAP
  SP_clu reproduces as 0.504. MDS and t-SNE are unaffected. The central
  dissociation (MDS highest SSC, lowest SP_adj) is unchanged; UMAP is the
  balanced-profile third method and does not carry the claim.

### MNIST (unified library, per-item, k=10, umap 0.5.12; N=1000, test set, seed 20250124)

Reproduced under the manuscript definition. Numbers differ from the original
submission (legacy core, k=4, balanced sampling over the full set, seed 42);
the qualitative pattern is preserved.

| method | SSC   | SP_adj | SP_ord | SP_clu | SP_total |
|--------|-------|--------|--------|--------|----------|
| t-SNE  | 0.474 | 0.320  | 0.664  | 0.519  | 0.501    |
| UMAP   | 0.412 | 0.259  | 0.642  | 0.530  | 0.477    |
| PCA    | 0.510 | 0.062  | 0.680  | 0.264  | 0.336    |

Preserved qualitative pattern: PCA highest SSC and lowest SP_adj; t-SNE and UMAP
preserve local neighborhoods (higher SP_adj) far better than PCA; t-SNE balanced
overall. Note the reference here is pixel-space input-feature distance, not
semantic distance.

Original-submission MNIST reference (legacy core, k=4; do not use in revision):
t-SNE SP_total 0.472 SSC 0.384; UMAP SP_clu 0.470 SSC 0.351; PCA SSC 0.488
SP_adj 0.042.

## Figure 1 revised values (for manuscript update)

Under the manuscript definition (per-item SP_adj, k=10), the Figure 1 numbers
change from the original submission. The manuscript text must be updated:

- Coordinate noise (Fig. 1a): SP falls 1.00 -> 0.24 (about 76% reduction), not
  the originally reported 1.00 -> 0.22 (78%). SSC stays near zero throughout.
- Metric vs topology at matched intensity 0.7 (Fig. 1b): shear SP = 0.718,
  topology-permutation SP = 0.258, difference about 0.459 (was Delta = 0.38).

If p-values are retained in the manuscript, the corresponding raw distributions
and the test script must be archived; otherwise report Figure 1b descriptively
and drop the p-value.
