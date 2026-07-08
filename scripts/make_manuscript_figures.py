#!/usr/bin/env python3
"""Regenerate the four multi-panel manuscript figures from the archived,
confirmed data. Panels and numbers match the revised manuscript captions.

Outputs (manuscript filenames):
  fig1_independence.pdf, fig2_baseline.pdf, fig3_lambda.pdf, fig4_things.pdf
"""
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "data" / "results"
OUT = REPO / "figures"
sys.path.insert(0, str(REPO))
from ssc_sp import deterministic  # noqa
from ssc_sp import metrics as M
from scipy.spatial.distance import pdist, squareform

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C_SP = "#1f6f8b"; C_SSC = "#c0392b"; C_COORD = "#2e86c1"; C_LAM = "#e67e22"
MDS_C, TSNE_C, UMAP_C = "#c0392b", "#2e86c1", "#27ae60"
NEARZERO = 0.10

def load(name):
    with open(RES / name) as f:
        return list(csv.reader(f))

def panel_label(ax, s):
    ax.text(-0.16, 1.02, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")

# ---------------- FIG 1 ----------------
def fig1():
    a = load("fig1a_coordinate_noise.csv")[1:]
    sig = [float(r[0]) for r in a]
    sp = [float(r[1]) for r in a]; sp_sd = [float(r[2]) for r in a]
    ssc = [float(r[3]) for r in a]; ssc_sd = [float(r[4]) for r in a]

    b = load("fig1b_metric_vs_topology.csv")[1:]
    rot = [float(r[2]) for r in b if r[0] == "rotation"]
    shear = [float(r[2]) for r in b if r[0] == "metric_shear"][0]
    topo = [float(r[2]) for r in b if r[0] == "topology_permute"][0]
    rot_mean = float(np.mean(rot))

    lam = load("fig3_lambda_synth.csv")[1:]
    lam_sp = [float(r[1]) for r in lam]; lam_ssc = [float(r[3]) for r in lam]

    fig = plt.figure(figsize=(11, 3.3))
    gs = gridspec.GridSpec(1, 3, wspace=0.42)

    ax = fig.add_subplot(gs[0]); panel_label(ax, "a")
    ax.axhspan(-NEARZERO, NEARZERO, color="#cfe8ef", alpha=0.5, zorder=0)
    ax.errorbar(sig, sp, yerr=sp_sd, marker="o", color=C_SP, label="SP", lw=1.6, capsize=2)
    ax.errorbar(sig, ssc, yerr=ssc_sd, marker="s", color=C_SSC, label="SSC", lw=1.6, capsize=2)
    ax.set_xlabel(r"Coordinate noise $\sigma$"); ax.set_ylabel("Score")
    ax.set_ylim(-0.2, 1.08); ax.annotate("1.00 \u2192 0.24\n(76% reduction)",
        xy=(0.7, 0.24), xytext=(0.33, 0.55), fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#555"))
    ax.legend(loc="center right", frameon=False)

    ax = fig.add_subplot(gs[1]); panel_label(ax, "b")
    bars = ax.bar([0, 1, 2], [rot_mean, shear, topo],
                  color=["#95a5a6", "#5dade2", "#e74c3c"], width=0.62)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["rotation\n(metric)", "shear\n(metric)", "topology\npermute"])
    ax.set_ylabel("SP"); ax.set_ylim(0, 1.05)
    for x, v in zip([0, 1, 2], [rot_mean, shear, topo]):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.annotate("", xy=(2, topo), xytext=(2, shear),
                arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text(2.12, (shear + topo) / 2, r"$\Delta = 0.459$", fontsize=8, va="center")

    ax = fig.add_subplot(gs[2]); panel_label(ax, "c")
    ax.axvspan(-NEARZERO, NEARZERO, color="#cfe8ef", alpha=0.5, zorder=0)
    ax.plot(ssc, sp, "-o", color=C_COORD, label="coordinate perturb.", lw=1.6, ms=4)
    ax.plot(lam_ssc, lam_sp, "-o", color=C_LAM, label=r"$\lambda$-coupling", lw=1.6, ms=4)
    ax.set_xlabel("SSC"); ax.set_ylabel("SP"); ax.set_xlim(-0.15, 0.4); ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", frameon=False)

    fig.savefig(OUT / "fig1_independence.pdf"); plt.close(fig)
    print("fig1 done")

# ---------------- FIG 2 ----------------
def _layout_coords(kind, n, rng):
    if kind == "uniform":
        return rng.uniform(size=(n, 2))
    if kind == "grid":
        s = int(np.ceil(np.sqrt(n))); pts = [(i, j) for i in range(s) for j in range(s)]
        return np.array(pts[:n], float) / max(s - 1, 1)
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[np.cos(t), np.sin(t)]
    if kind == "gaussian":
        return rng.normal(size=(n, 2))
    return rng.uniform(size=(n, 2))

def _mean_ssc(n, dim, metric, layout, trials=100, seed=3776):
    rng = deterministic.set_seed(seed)
    vals = []
    for _ in range(trials):
        emb = rng.normal(size=(n, dim))
        d_ref = squareform(pdist(emb, metric=metric))
        coords = _layout_coords(layout, n, rng)
        vals.append(M.ssc(d_ref, coords))
    return float(np.mean(vals)), float(np.std(vals))

def fig2():
    raw = [float(r[0]) for r in load("fig2_baseline_null_raw.csv")[1:]]
    s = load("fig2_baseline_null_summary.csv")[1:][0]
    mean_ssc = float(s[3]); sd_ssc = float(s[4])
    ci_lo = float(s[5]); ci_hi = float(s[6])

    # Fig 2b: condition-wise mean SSC (all near zero) -- computed deterministically
    conds = []
    for d in (50, 100, 200, 500):
        conds.append((f"D={d}", *_mean_ssc(64, d, "euclidean", "uniform")))
    for n in (10, 20, 40, 80):
        conds.append((f"N={n}", *_mean_ssc(n, 128, "euclidean", "uniform")))
    for m in ("euclidean", "cosine", "correlation"):
        conds.append((m[:4], *_mean_ssc(64, 128, m, "uniform")))
    for lay in ("uniform", "grid", "circle", "gaussian"):
        conds.append((lay[:4], *_mean_ssc(64, 128, "euclidean", lay)))
    # persist for repo consistency
    with open(RES / "fig2b_generalization.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition", "mean_ssc", "sd_ssc"])
        for c in conds: w.writerow(c)

    fig = plt.figure(figsize=(9.5, 3.4))
    gs = gridspec.GridSpec(1, 2, wspace=0.3, width_ratios=[1, 1.25])

    ax = fig.add_subplot(gs[0]); panel_label(ax, "a")
    ax.axvspan(-NEARZERO, NEARZERO, color="#cfe8ef", alpha=0.5, zorder=0)
    ax.hist(raw, bins=40, color="#5499c7", edgecolor="white", linewidth=0.3)
    ax.axvline(mean_ssc, color="#c0392b", lw=1.5)
    ax.set_xlabel("SSC (random pairings)"); ax.set_ylabel("Count")
    ax.set_xlim(-0.2, 0.2)
    ax.text(0.02, 0.95, f"mean = {mean_ssc:.4f}\nSD = {sd_ssc:.3f}\n90% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
            transform=ax.transAxes, va="top", fontsize=8)

    ax = fig.add_subplot(gs[1]); panel_label(ax, "b")
    ax.axhspan(-NEARZERO, NEARZERO, color="#cfe8ef", alpha=0.5, zorder=0)
    xs = list(range(len(conds)))
    means = [c[1] for c in conds]; sds = [c[2] for c in conds]
    groups = [4, 4, 3, 4]; bounds = np.cumsum(groups)
    ax.errorbar(xs, means, yerr=sds, fmt="o", color="#2e4053", ms=4, capsize=2, lw=1)
    ax.axhline(0, color="#888", lw=0.6, ls="--")
    for bnd in bounds[:-1]:
        ax.axvline(bnd - 0.5, color="#ddd", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in conds], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean SSC"); ax.set_ylim(-0.15, 0.15)
    gl = ["dimensionality", "sample size", "metric", "layout"]
    centers = [1.5, 5.5, 9, 12.5]
    for cx, lb in zip(centers, gl):
        ax.text(cx, 0.135, lb, ha="center", fontsize=7, color="#555")

    fig.savefig(OUT / "fig2_baseline.pdf"); plt.close(fig)
    print("fig2 done (fig2b_generalization.csv written)")

# ---------------- FIG 3 ----------------
def fig3():
    sy = load("fig3_lambda_synth.csv")[1:]; be = load("fig3_lambda_bert.csv")[1:]
    lam = [float(r[0]) for r in sy]
    sy_sp = [float(r[1]) for r in sy]; sy_ssc = [float(r[3]) for r in sy]
    be_sp = [float(r[1]) for r in be]; be_ssc = [float(r[3]) for r in be]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lam, sy_sp, "-o", color=C_SP, label="SP (synthetic)", lw=1.6, ms=4)
    ax.plot(lam, be_sp, "--o", color=C_SP, label="SP (BERT)", lw=1.6, ms=4, alpha=0.7)
    ax.plot(lam, sy_ssc, "-s", color=C_SSC, label="SSC (synthetic)", lw=1.6, ms=4)
    ax.plot(lam, be_ssc, "--s", color=C_SSC, label="SSC (BERT)", lw=1.6, ms=4, alpha=0.7)
    ax.set_xlabel(r"Coupling parameter $\lambda$"); ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05); ax.legend(frameon=False, loc="center left")
    ax.annotate(f"SSC \u2192 {be_ssc[-1]:.3f} (BERT)\nSSC \u2192 {sy_ssc[-1]:.3f} (synth)",
                xy=(1.0, be_ssc[-1]), xytext=(0.42, 0.62), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate(f"SP \u2192 {sy_sp[-1]:.2f}", xy=(1.0, sy_sp[-1]), xytext=(0.55, 0.12),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
    fig.savefig(OUT / "fig3_lambda.pdf"); plt.close(fig)
    print("fig3 done")

# ---------------- FIG 4 ----------------
def fig4():
    rows = load("things_table1_summary.csv")
    data = {}
    for r in rows[3:]:
        m = r[0]
        data[m] = dict(SSC=(float(r[1]), float(r[2])), SP_adj=(float(r[3]), float(r[4])),
                       SP_ord=(float(r[5]), float(r[6])), SP_clu=(float(r[7]), float(r[8])),
                       SP_total=(float(r[9]), float(r[10])))
    order = ["MDS", "tSNE", "UMAP"]; labels = ["MDS", "t-SNE", "UMAP"]
    cols = {"MDS": MDS_C, "tSNE": TSNE_C, "UMAP": UMAP_C}

    fig = plt.figure(figsize=(9.8, 3.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.32, width_ratios=[1, 1.2])

    ax = fig.add_subplot(gs[0]); panel_label(ax, "a")
    for m, lb in zip(order, labels):
        x, xe = data[m]["SSC"]; y, ye = data[m]["SP_adj"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=cols[m], ms=9, capsize=2, label=lb)
    ax.set_xlabel("SSC (semantic alignment)"); ax.set_ylabel(r"SP$_{\mathrm{adj}}$ (neighborhood preservation)")
    ax.annotate("SSC = 0.725\n" + r"SP$_{\mathrm{adj}}$ = 0.127", xy=(data["MDS"]["SSC"][0], data["MDS"]["SP_adj"][0]),
                xytext=(0.66, 0.20), fontsize=7.5, color=MDS_C,
                arrowprops=dict(arrowstyle="->", color=MDS_C))
    ax.set_xlim(0.60, 0.75); ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1]); panel_label(ax, "b")
    comps = ["SP_adj", "SP_ord", "SP_clu"]; comp_lbl = [r"SP$_{\mathrm{adj}}$", r"SP$_{\mathrm{ord}}$", r"SP$_{\mathrm{clu}}$"]
    x = np.arange(len(comps)); w = 0.26
    for i, (m, lb) in enumerate(zip(order, labels)):
        vals = [data[m][c][0] for c in comps]; err = [data[m][c][1] for c in comps]
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, color=cols[m], label=lb, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(comp_lbl); ax.set_ylabel("Score"); ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    # annotate MDS low SP_adj
    ax.text(0 - w, data["MDS"]["SP_adj"][0] + 0.02, "0.127", ha="center", fontsize=7, color=MDS_C)
    fig.savefig(OUT / "fig4_things.pdf"); plt.close(fig)
    print("fig4 done")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("ALL FIGURES REGENERATED ->", OUT)
