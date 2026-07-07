# Metric definitions (authoritative)

These are the definitions implemented in `ssc_sp/metrics.py` and used for all
figures and tables. They match the revised manuscript.

## SSC (semantic/reference-spatial correlation)

Spearman rank correlation between the vectorized pairwise-distance matrices of
the reference representation and the 2D layout:

    SSC = spearman( vec(D_ref), vec(D_spatial) )

D_spatial is Euclidean distance in the 2D layout. D_ref is the reference-space
distance (Euclidean on the high-dimensional representation, or a precomputed
distance matrix for applied datasets).

## SP_adj (local neighborhood preservation)

Mean over items of the Jaccard overlap between the k-nearest-neighbor set in the
reference space and the k-nearest-neighbor set in the 2D layout (self excluded):

    SP_adj = mean_i |N_k^ref(i) ∩ N_k^lay(i)| / |N_k^ref(i) ∪ N_k^lay(i)|

Default k = 10. This is a per-item set-membership quantity, distinct from the
whole-graph edge-set Jaccard.

## SP_ord (global rank fidelity)

Kendall's tau between the two pairwise-distance vectors, rescaled to [0, 1]:

    SP_ord = (tau + 1) / 2

SP_ord and SSC both summarize global pairwise-distance rank association from the
same distance vectors, and are therefore near-redundant in this framework.

## SP_clu (cluster agreement), clipped at zero

Adjusted Rand Index between K-means labels of the reference representation and
the 2D layout, clipped at zero:

    SP_clu = max(0, ARI(labels_ref, labels_layout))

ARI can be negative (worse than chance). Clipping at zero keeps the preservation
component on the same non-negative reporting scale as SP_adj and SP_ord before
inclusion in SP_total. ARI is permutation-invariant, so no label alignment is
applied.

## SP_total

    SP_total = (SP_adj + SP_ord + SP_clu) / 3

## Reference modes

- layout-preservation: reference is the unperturbed baseline layout (synthetic
  perturbation experiments). SP = 1.0 at zero perturbation by construction.
- reference-space: reference is the high-dimensional representation (applied
  demonstrations: SPoSE for THINGS, pixels for MNIST).
