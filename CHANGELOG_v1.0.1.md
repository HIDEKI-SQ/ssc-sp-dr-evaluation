# ssc-sp-dr-evaluation — v1.0.1 sync (figure reproducibility)

This delta adds the manuscript-figure generation scripts and their input/output,
and fills in the Zenodo DOIs in `README.md` / `CITATION.cff`. **It does not modify
any frozen v1.0.0 file** (the `ssc_sp/` library, configs, experiment scripts, tests,
and shipped result CSVs are untouched), so the numbers archived under
**version DOI 10.5281/zenodo.21244545 (v1.0.0)** — the manuscript's reproduction
target — remain exactly as published.

## What this delta contains

**New scripts** (portable; read `data/results/*`, write `figures/`):
- `scripts/make_manuscript_figures.py` — regenerates the multi-panel Fig. 1-4 and
  writes `data/results/fig2b_generalization.csv` (the baseline-null generalization sweep).
- `scripts/make_supplementary_figures.py` — regenerates Supp. Fig. S1-S3.
- `scripts/make_figure_s4.py` — regenerates Supp. Fig. S4 from `s4_sensitivity.json`.

**New data**:
- `data/results/fig2b_generalization.csv` — condition-wise mean SSC (dimensionality,
  sample size, metric, layout), all within the near-zero region.
- `data/results/s4_sensitivity.json` — THINGS neighborhood-size (k ∈ {5,10,20}) and
  sample-size (N ∈ {300,500}) sensitivity, computed under the pinned `umap-learn==0.5.12`.

**New figures** (the exact PDFs used in the manuscript):
- `figures/fig1_independence.pdf`, `fig2_baseline.pdf`, `fig3_lambda.pdf`, `fig4_things.pdf`
- `figures/figure_s1_robustness.pdf`, `figure_s2_layout.pdf`, `figure_s3_independence.pdf`,
  `figure_s4_things_sensitivity.pdf`

**Updated docs**:
- `CITATION.cff` — version 1.0.1, `doi: 10.5281/zenodo.21244544` (concept), date 2026-07-08.
- `README.md` — replaced the DOI placeholder with the concept + version DOIs, and added a
  "Regenerate the manuscript figures" subsection.

## How to apply (same workflow as v1.0.0)

1. Unzip this delta at the **root** of your local `ssc-sp-dr-evaluation` working copy
   (the folder that contains `.git`). It only adds new files and overwrites
   `CITATION.cff` and `README.md`. Nothing else is changed.
2. Sanity check locally (optional): `git status` should show only the new files plus
   `CITATION.cff` and `README.md`. The banned-terms checker still passes
   (`python scripts/check_banned_terms.py`).
3. Commit and push:
   ```
   git add -A
   git commit -m "v1.0.1: add manuscript/supplementary figure scripts + figures; fill in Zenodo DOIs"
   git push
   ```
4. Mint the new archived version on Zenodo:
   - GitHub → Releases → Draft a new release → tag `v1.0.1` → Publish.
   - Zenodo (already linked) will archive it automatically and issue a new **version DOI**.
   - The **concept DOI 10.5281/zenodo.21244544 continues to resolve to the latest version**,
     so nothing in the manuscript or the Response needs to change.

## Why no manuscript change is needed

The manuscript's Code availability cites the **version DOI 21244545 (v1.0.0)** as the exact
release whose numbers it reports, and the **concept DOI 21244544** for all versions. v1.0.1
only adds figure-generation code (the reported numbers are unchanged), and it is reachable
through the concept DOI. Both citations therefore remain correct after minting v1.0.1.
