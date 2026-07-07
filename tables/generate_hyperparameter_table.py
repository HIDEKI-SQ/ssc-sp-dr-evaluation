#!/usr/bin/env python3
"""Generate the hyperparameter table from config/*.yaml.

Single source of truth: nothing here is hand-typed from the manuscript. Emits a
CSV (repo) and a LaTeX table (manuscript). This directly serves the reviewers'
request for a consolidated reproducibility table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
OUT = ROOT / "tables"


def _flatten(prefix: str, obj, rows: list[tuple[str, str, str]], dataset: str):
    if isinstance(obj, dict):
        for key, val in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else key, val, rows, dataset)
    elif isinstance(obj, list):
        rows.append((dataset, prefix, ", ".join(str(x) for x in obj)))
    else:
        rows.append((dataset, prefix, str(obj)))


def collect_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(CONFIG.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        dataset = cfg.get("dataset", path.stem)
        for section in ("sampling", "generation", "metrics", "dr_methods",
                        "coordinate_noise", "metric_vs_topology",
                        "semantic_noise", "embeddings"):
            if section in cfg:
                _flatten(section, cfg[section], rows, dataset)
    return rows


def write_csv(rows, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "parameter", "value"])
        w.writerows(rows)


def write_latex(rows, path: Path):
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Hyperparameters and reproducibility settings.}",
        r"\begin{tabular}{lll}", r"\toprule",
        r"Dataset & Parameter & Value \\", r"\midrule",
    ]
    for dataset, param, value in rows:
        esc = lambda s: s.replace("_", r"\_")
        lines.append(f"{esc(dataset)} & {esc(param)} & {esc(value)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


def main() -> int:
    rows = collect_rows()
    write_csv(rows, OUT / "hyperparameter_table.csv")
    write_latex(rows, OUT / "hyperparameter_table.tex")
    print(f"Wrote {len(rows)} parameter rows to tables/hyperparameter_table.(csv|tex)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
