#!/usr/bin/env python3
"""Generate Supplementary Figure S4 (THINGS sensitivity) from s4_sensitivity.json.
Option A: uses the sensitivity-run values directly (same embedding swept over k).
Two panels matching the original: (a) SP_total vs k_nn (N=500); (b) SSC by N (k_nn=10).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
J = json.load(open(REPO / "data" / "results" / "s4_sensitivity.json"))
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight"})
MDS_C, TSNE_C, UMAP_C = "#e74c3c", "#3498db", "#2ecc71"
MK = {"MDS": ("o", MDS_C), "tSNE": ("s", TSNE_C), "UMAP": ("^", UMAP_C)}
LBL = {"MDS": "MDS", "tSNE": "t-SNE", "UMAP": "UMAP"}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))

# ----- Panel (a): SP_total vs k_nn (N=500) -----
ks = [5, 10, 20]
pa = J["panel_a_sp_vs_k"]
for m in ["MDS", "tSNE", "UMAP"]:
    mk, col = MK[m]
    means = [pa[str(k)][m][0] for k in ks]
    sds = [pa[str(k)][m][1] for k in ks]
    ax[0].errorbar(ks, means, yerr=sds, marker=mk, color=col, label=LBL[m],
                   lw=1.8, ms=8, capsize=3)
ax[0].set_xticks(ks)
ax[0].set_xlabel(r"$k_{\mathrm{nn}}$ (neighborhood size)")
ax[0].set_ylabel(r"SP$_{\mathrm{total}}$")
ax[0].set_title(r"(a) SP Sensitivity to $k_{\mathrm{nn}}$ (N=500)")
ax[0].set_ylim(0.35, 0.60)
ax[0].legend(loc="lower right")

# ----- Panel (b): SSC by sample size N (k_nn=10) -----
Ns = ["300", "500"]
pb = J["panel_b_ssc_vs_N"]
x = np.arange(len(Ns)); w = 0.26
for i, m in enumerate(["MDS", "tSNE", "UMAP"]):
    _, col = MK[m]
    means = [pb[n][m][0] for n in Ns]
    sds = [pb[n][m][1] for n in Ns]
    ax[1].bar(x + (i - 1) * w, means, w, yerr=sds, color=col, label=LBL[m],
              capsize=3, error_kw=dict(lw=1.2))
ax[1].set_xticks(x); ax[1].set_xticklabels([f"N={n}" for n in Ns])
ax[1].set_xlabel("Sample Size (N)")
ax[1].set_ylabel("SSC")
ax[1].set_title(r"(b) SSC by Sample Size ($k_{\mathrm{nn}}$=10)")
ax[1].set_ylim(0.55, 0.80)
ax[1].legend(loc="upper right")

fig.tight_layout()
fig.savefig(OUT / "figure_s4_things_sensitivity.pdf")
plt.close(fig)
print("S4 done ->", OUT / "figure_s4_things_sensitivity.pdf")
