#!/usr/bin/env python3
"""random-pairing null baseline for SSC (Figure 2) under the unified library.

Random semantic embeddings paired with random 2D coordinates yield SSC centered
on zero. A two one-sided tests (TOST) procedure confirms equivalence to zero
within +/- delta. SP is not involved here; this characterizes SSC under a null
pairing.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ssc_sp import deterministic  # noqa: F401
from ssc_sp import metrics as M

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "synthetic.yaml").read_text())["baseline_null"]
OUT = ROOT / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def tost_equivalence(data: np.ndarray, delta: float) -> dict:
    n = len(data)
    m = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n))
    t_low = (m - (-delta)) / se
    t_high = (delta - m) / se
    p_low = stats.t.sf(t_low, n - 1)   # H0: mean <= -delta
    p_high = stats.t.sf(t_high, n - 1)  # H0: mean >= +delta
    return {"tost_p": float(max(p_low, p_high)), "mean": m, "se": se}


def run():
    n = CFG["n_items"]
    dim = CFG["dim"]
    seed = CFG["seed"]
    n_trials = CFG["n_trials"]
    delta = CFG["equivalence_delta"]

    rng = np.random.default_rng(seed)
    ssc_vals = []
    for _ in range(n_trials):
        emb = rng.normal(size=(n, dim))
        coords = rng.uniform(0, 1, size=(n, 2))
        d_ref = squareform(pdist(emb, metric=CFG["distance"]))
        ssc_vals.append(M.ssc(d_ref, coords))
    ssc_vals = np.asarray(ssc_vals)

    mean = float(np.mean(ssc_vals))
    sd = float(np.std(ssc_vals, ddof=1))
    # 90% CI for the mean
    se = sd / np.sqrt(n_trials)
    ci = (mean - 1.645 * se, mean + 1.645 * se)
    tost = tost_equivalence(ssc_vals, delta)

    pd.DataFrame({"ssc": ssc_vals}).to_csv(OUT / "fig2_baseline_null_raw.csv", index=False)
    summary = pd.DataFrame([{
        "n_trials": n_trials, "n_items": n, "dim": dim,
        "mean_ssc": mean, "sd_ssc": sd,
        "ci90_low": ci[0], "ci90_high": ci[1],
        "tost_delta": delta, "tost_p": tost["tost_p"],
    }])
    summary.to_csv(OUT / "fig2_baseline_null_summary.csv", index=False)

    print("=== Fig 2: random-pairing null baseline (SSC) (N=%d, D=%d, %d trials) ===" % (n, dim, n_trials))
    print(f"  mean SSC = {mean:.4f}  (SD = {sd:.4f})")
    print(f"  90% CI for the mean = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  TOST equivalence to 0 within +/-{delta}: p = {tost['tost_p']:.2e}")


if __name__ == "__main__":
    run()
