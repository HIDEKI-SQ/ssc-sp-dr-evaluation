"""Guard the shipped THINGS Table 1 summary against silent changes.

The full THINGS reproduction needs the external SPoSE file, so CI cannot rerun
it. Instead we pin the expected values in the shipped summary CSV: MDS and
t-SNE are the manuscript-critical rows and must not drift.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "results" / "things_table1_summary.csv"


def _summary():
    return pd.read_csv(CSV, header=[0, 1], index_col=0)


def test_mds_central_values():
    s = _summary()
    assert abs(s.loc["MDS", ("SSC", "mean")] - 0.725) < 0.002
    assert abs(s.loc["MDS", ("SP_adj", "mean")] - 0.127) < 0.002
    assert abs(s.loc["MDS", ("SP_adj", "std")] - 0.005) < 0.002  # not 0.015


def test_tsne_central_values():
    s = _summary()
    assert abs(s.loc["tSNE", ("SSC", "mean")] - 0.661) < 0.002
    assert abs(s.loc["tSNE", ("SP_adj", "mean")] - 0.350) < 0.002


def test_dissociation_ranking():
    s = _summary()
    # MDS highest SSC but lowest SP_adj (the central dissociation)
    ssc = s[("SSC", "mean")]
    adj = s[("SP_adj", "mean")]
    assert ssc.idxmax() == "MDS"
    assert adj.idxmin() == "MDS"
