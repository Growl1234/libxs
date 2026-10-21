# Rosetta

Demonstrates hierarchical type discovery on opaque binary data.
A table of harmonic-series values is encoded as a flat byte blob
with no metadata. The analysis rediscovers the element type,
record stride, and per-field structure — recovering mathematical
content from anonymous bytes.

## What It Shows

The sample proceeds through these levels:

| Step           | Tool used                  | What it checks                       |
|----------------|----------------------------|--------------------------------------|
| Flat probe     | libxs_fprint (UNKNOWN)     | smoothness of the bytes as one array |
| Stride sweep   | libxs_fprint per-column    | element type and record stride       |
| Shuffle test   | libxs_shuffle + fprint     | whether the record order matters     |
| Field analysis | libxs_fprint per-field     | decay rate of each column            |
| Sort test      | libxs_sort_smooth (GREEDY) | whether a reordering is smoother     |
| Verification   | libxs_setdiff_min          | recovered records against the blob   |

## The Data

For n = 1, 2, ..., 64 the table stores six f64 fields per record:

    [0] n            integer index
    [1] 1/n          harmonic term
    [2] H_n          partial sum of the harmonic series
    [3] ln(n)        natural logarithm
    [4] H_n - ln(n)  difference (converges to gamma from above)
    [5] H_n - ln(n) - 1/(2n)   better gamma estimate

Total: 64 records x 6 fields = 384 doubles = 3072 bytes.

## Build

```bash
cd samples/rosetta
make
```

## Usage

```bash
./rosetta.x
```

No arguments. The sample is self-contained: it generates the data,
encodes it as a flat blob, runs the full analysis, and prints the
results.

To collect machine-readable artifacts, set `ROSETTA_OUTDIR` to an
existing directory:

```bash
mkdir -p results/rosetta
ROSETTA_OUTDIR=results/rosetta ./rosetta.x
```

This writes `summary.csv`, the record matrices `rosetta_original.mhd`
and `rosetta_shuffled.mhd`, and the images `rosetta_fields.png` and
`rosetta_fields_shuffled.png`. The `fields` images are normalized per
field, making the ordered and shuffled record traces easy to compare
visually. The artifact mode is optional and leaves the interactive
sample output unchanged.
