#!/usr/bin/env python3
"""Regenerate supplementary figures S1, S2, S3 from confirmed data.
S4 (THINGS sensitivity) requires SPoSE recomputation -> separate Colab notebook.

Outputs: figure_s1_robustness.pdf, figure_s2_layout.pdf, figure_s3_independence.pdf
"""
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "data" / "results"
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from ssc_sp import deterministic  # noqa
from ssc_sp import metrics as M
from scipy.spatial.distance import pdist, squareform

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight"})
C_SP, C_SSC = "#1f77b4", "#ff7f0e"; RED = "#e74c3c"; NZ = 0.10

def load(name):
    with open(RES / name) as f:
        return list(csv.reader(f))

def _layout(kind, n, rng):
    if kind == "Spatial": return rng.uniform(size=(n, 2))
    if kind == "Grid":
        s = int(np.ceil(np.sqrt(n))); pts = [(i, j) for i in range(s) for j in range(s)]
        return np.array(pts[:n], float) / max(s - 1, 1)
    if kind == "Line":
        x = np.linspace(0, 1, n); return np.c_[x, np.zeros(n)]
    if kind == "Circle":
        t = np.linspace(0, 2*np.pi, n, endpoint=False); return np.c_[np.cos(t), np.sin(t)]
    if kind == "Random": return rng.normal(size=(n, 2))
    return rng.uniform(size=(n, 2))

def mean_ssc(n=64, dim=128, metric="euclidean", layout="Spatial", trials=200, seed=3776):
    rng = deterministic.set_seed(seed); vals = []
    for _ in range(trials):
        emb = rng.normal(size=(n, dim))
        d_ref = squareform(pdist(emb, metric=metric))
        vals.append(M.ssc(d_ref, _layout(layout, n, rng)))
    v = np.asarray(vals)
    return float(v.mean()), float(v.std())

# ---------------- S1: Dimension / Sample size / Metric robustness ----------------
def s1():
    Ds = [50, 100, 200, 500]; Ns = [10, 20, 40, 80]; Ms = ["correlation", "euclidean", "cosine"]
    dvals = [mean_ssc(dim=d) for d in Ds]
    nvals = [mean_ssc(n=n) for n in Ns]
    mvals = [mean_ssc(metric=m) for m in Ms]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a in ax:
        a.axhspan(-NZ, NZ, color="#e8e8e8"); a.axhline(0, color="k", lw=0.7)
        a.set_ylim(-0.15, 0.15); a.set_ylabel("Mean SSC")
    ax[0].errorbar(Ds, [v[0] for v in dvals], yerr=[v[1] for v in dvals], fmt="o-", color=RED, capsize=3)
    ax[0].set_title("(a) Dimension Robustness"); ax[0].set_xlabel("Embedding Dimension (D)")
    ax[1].errorbar(Ns, [v[0] for v in nvals], yerr=[v[1] for v in nvals], fmt="s-", color=RED, capsize=3)
    ax[1].set_title("(b) Sample Size Robustness"); ax[1].set_xlabel("Sample Size (N)")
    ax[2].errorbar(range(len(Ms)), [v[0] for v in mvals], yerr=[v[1] for v in mvals], fmt="o", color="k", capsize=3, ls="none")
    ax[2].set_xticks(range(len(Ms))); ax[2].set_xticklabels([m.capitalize() for m in Ms])
    ax[2].set_title("(c) Metric Robustness"); ax[2].set_xlabel("Distance Metric")
    fig.savefig(OUT / "figure_s1_robustness.pdf"); plt.close(fig); print("S1 done")

# ---------------- S2: Layout generalization (banned term fixed) ----------------
def s2():
    layouts = ["Spatial", "Grid", "Line", "Circle", "Random"]
    vals = [mean_ssc(layout=l) for l in layouts]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(-NZ, NZ, color="#e8e8e8")
    ax.axvline(-NZ, color="#bbb", ls="--"); ax.axvline(NZ, color="#bbb", ls="--")
    ax.axvline(0, color="k", lw=0.9)
    ys = list(range(len(layouts)))[::-1]
    ci = [1.645 * v[1] for v in vals]
    ax.errorbar([v[0] for v in vals], ys, xerr=ci, fmt="o", color=RED, ms=9, capsize=4, ls="none")
    ax.set_yticks(ys); ax.set_yticklabels(layouts)
    ax.set_xlim(-0.15, 0.15); ax.set_xlabel("Mean SSC (90% CI)")
    ax.set_title("Study 1: Layout Generalization\n(All layouts yield SSC $\\approx$ 0)")
    ax.text(-0.095, 1.3, "Near-zero region", fontsize=10, style="italic", color="#777")
    fig.savefig(OUT / "figure_s2_layout.pdf"); plt.close(fig); print("S2 done")

# ---------------- S3: Four-panel factorial (confirmed values, BERT for C/D) ----------------
def s3():
    a = load("fig1a_coordinate_noise.csv")[1:]
    sig = [float(r[0]) for r in a]; sp_a = [float(r[1]) for r in a]; ssc_a = [float(r[3]) for r in a]
    b = load("fig1_semantic_noise_control.csv")[1:]
    ssig = [float(r[0]) for r in b]; sp_b = [float(r[1]) for r in b]; ssc_b = [float(r[2]) for r in b]
    be = load("fig3_lambda_bert.csv")[1:]
    lam = [float(r[0]) for r in be]; sp_c = [float(r[1]) for r in be]; ssc_c = [float(r[3]) for r in be]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    p = ax[0, 0]
    p.plot(sig, sp_a, "o-", color=C_SP, label="SP"); p.plot(sig, ssc_a, "s-", color=C_SSC, label="SSC")
    p.axhline(0.5, color="#bbb", ls=":"); p.set_ylim(-0.1, 1.08)
    p.set_title("(A) Coordinate Noise: Affects SP only"); p.set_xlabel(r"Coordinate noise ($\sigma$)"); p.set_ylabel("Metric Value")
    p.text(0.45, 0.6, r"SP$\downarrow$", color=C_SP, fontsize=12, fontweight="bold")
    p.text(0.45, 0.06, r"SSC$\approx$0", color=C_SSC, fontsize=11, fontweight="bold"); p.legend()

    p = ax[0, 1]
    p.plot(ssig, sp_b, "o-", color=C_SP, label="SP"); p.plot(ssig, ssc_b, "s-", color=C_SSC, label="SSC")
    p.axhline(0.5, color="#bbb", ls=":"); p.set_ylim(-0.1, 1.08)
    p.set_title(r"(B) Semantic Noise: Affects neither (at $\lambda$=0)"); p.set_xlabel(r"Semantic noise ($\sigma$)"); p.set_ylabel("Metric Value")
    p.text(0.35, 0.92, "SP=1.0", color=C_SP, fontsize=12, fontweight="bold")
    p.text(0.35, 0.06, r"SSC$\approx$0", color=C_SSC, fontsize=11, fontweight="bold"); p.legend()

    p = ax[1, 0]
    p.plot(lam, sp_c, "o-", color=C_SP, label="SP"); p.plot(lam, ssc_c, "s-", color=C_SSC, label="SSC")
    p.axhline(0.5, color="#bbb", ls=":"); p.set_ylim(-0.1, 1.08)
    p.set_title(r"(C) Value Coupling: SP$\downarrow$ while SSC$\uparrow$ (BERT)"); p.set_xlabel(r"Coupling parameter ($\lambda$)"); p.set_ylabel("Metric Value")
    p.text(0.30, 0.62, r"Trade-off:" + "\n" + r"SP$\downarrow$ as SSC$\uparrow$", fontsize=9, style="italic", color="#777"); p.legend()

    p = ax[1, 1]
    p.axvline(0, color="#ccc", ls=":")
    p.plot(ssc_a, sp_a, "o-", color=C_SP, label=r"Coordinate noise ($\lambda$=0)")
    p.plot(ssc_c, sp_c, "o-", color=C_SSC, label=r"$\lambda$ sweep (no noise)")
    p.annotate("", xy=(ssc_a[-1], sp_a[-1]), xytext=(ssc_a[0], sp_a[0]), arrowprops=dict(arrowstyle="->", color=C_SP))
    p.set_title("(D) SSC-SP Space: Different axes of variation"); p.set_xlabel("SSC"); p.set_ylabel("SP")
    p.set_xlim(-0.2, 0.8); p.set_ylim(0.12, 1.05)
    p.text(0.0, 0.42, "Coord\nnoise", color=C_SP, fontsize=10, fontweight="bold", ha="center")
    p.text(0.45, 0.75, "Value\ncoupling", color=C_SSC, fontsize=10, fontweight="bold"); p.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "figure_s3_independence.pdf"); plt.close(fig); print("S3 done")

if __name__ == "__main__":
    s1(); s2(); s3()
    print("S1-S3 regenerated ->", OUT)
