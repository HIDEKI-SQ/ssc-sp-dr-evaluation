#!/usr/bin/env python3
"""Render figures from the result CSVs in data/results/.

Reads the CSVs produced by the run/reproduce scripts and writes figure PDFs to
figures/. Run the experiment scripts first. Figures are intentionally simple;
the numbers are the deliverable, not the styling.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _exists(name: str) -> Path | None:
    p = RES / name
    return p if p.exists() else None


def fig1():
    p = _exists("fig1a_coordinate_noise.csv")
    if not p:
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df["sigma"], df["sp_mean"], yerr=df["sp_std"], marker="o", label="SP")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.plot(df["sigma"], df["ssc_mean"], marker="s", label="SSC")
    ax.set_xlabel("coordinate noise sigma")
    ax.set_ylabel("value")
    ax.set_title("Coordinate noise: SP falls, SSC near zero")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "figure1a_coordinate_noise.pdf")
    plt.close(fig)


def fig2():
    p = _exists("fig2_baseline_null_raw.csv")
    if not p:
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["ssc"], bins=40)
    ax.axvline(0, ls="--", c="red", lw=1)
    ax.set_xlabel("SSC under random pairing")
    ax.set_ylabel("count")
    ax.set_title("Random-pairing null baseline for SSC")
    fig.tight_layout()
    fig.savefig(FIG / "figure2_baseline_null.pdf")
    plt.close(fig)


def fig3():
    ps = _exists("fig3_lambda_synth.csv")
    pb = _exists("fig3_lambda_bert.csv")
    if not ps:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    s = pd.read_csv(ps)
    ax.plot(s["lam"], s["sp_mean"], marker="o", label="SP (synth)")
    ax.plot(s["lam"], s["ssc_mean"], marker="s", label="SSC (synth)")
    if pb:
        b = pd.read_csv(pb)
        ax.plot(b["lam"], b["sp_mean"], marker="o", ls="--", label="SP (BERT)")
        ax.plot(b["lam"], b["ssc_mean"], marker="s", ls="--", label="SSC (BERT)")
    ax.set_xlabel("lambda")
    ax.set_ylabel("value")
    ax.set_title("Value coupling: SSC up, SP down")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "figure3_lambda.pdf")
    plt.close(fig)


def fig4():
    p = _exists("things_table1_raw.csv")
    if not p:
        return
    df = pd.read_csv(p).groupby("method")[["SSC", "SP_adj"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["SSC"], df["SP_adj"])
    for m, r in df.iterrows():
        ax.annotate(m, (r["SSC"], r["SP_adj"]))
    ax.set_xlabel("SSC")
    ax.set_ylabel("SP_adj")
    ax.set_title("THINGS: SSC vs local neighborhood preservation")
    fig.tight_layout()
    fig.savefig(FIG / "figure4_things.pdf")
    plt.close(fig)


def main():
    fig1(); fig2(); fig3(); fig4()
    made = sorted(p.name for p in FIG.glob("*.pdf"))
    print("Rendered:", ", ".join(made) if made else "(no CSVs found; run experiments first)")


if __name__ == "__main__":
    main()
