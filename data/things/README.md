# THINGS SPoSE embedding

The THINGS demonstration uses the SPoSE concept embedding derived from human
odd-one-out similarity judgments. It is not redistributed here.

## How to obtain

1. Download the SPoSE embedding checkpoint `embedding00_epoch0500.txt` from the
   THINGS OSF repository (https://osf.io/z2784/).
2. Place the file in this directory:
   `data/things/embedding00_epoch0500.txt`

The file is a whitespace-delimited matrix. It may be stored as
`1854 x 120` (concepts x dimensions) or transposed; the loader in
`scripts/reproduce_things.py` transposes automatically so that rows are
concepts.

## Use

```bash
python scripts/reproduce_things.py --embedding data/things/embedding00_epoch0500.txt
# or set SPOSE_PATH=data/things/embedding00_epoch0500.txt
```

This reproduces Table 1 (mean +/- SD over seeds 20250124, 20250125, 20250126,
N = 500, k = 10) and the Figure 4 data. MDS and t-SNE reproduce the manuscript
values exactly; UMAP requires the pinned `umap-learn==0.5.12`.
