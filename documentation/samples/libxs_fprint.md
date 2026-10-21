# Fingerprint

This sample exercises the LIBXS fingerprint API (`libxs_fprint`) and reports
its properties as CSV tables. Rosetta covers the higher-level algebra and
type discovery.

## What It Produces

Set `FPRINT_OUTDIR` to collect CSV artifacts (without it, results go to stdout
only):

```bash
mkdir -p results/fprint
FPRINT_OUTDIR=results/fprint ./fprint.x
```

The sample writes:

| File | Purpose |
|------|---------|
| `convergence.csv` | Same functions sampled at increasing resolution and compared to a high-resolution reference. |
| `sensitivity.csv` | Perturbations with matched element RMS but different structural character. |
| `geometry.csv` | Bracket-polynomial boundary area, centroid, moments, and shape fingerprints. |
| `hierarchy.csv` | 2-D smooth, creased, diagonal-creased, and noisy fields in hierarchical and per-axis modes. |
| `compression.csv` | Newton truncation order versus reconstruction error. |
| `collision.csv` | Pseudometric collision test: pairs with shared L2 norms but different signed means, and a pair sharing the whole fingerprint; `bernstein_distance` is `libxs_fprint_bernstein_diff`. |
| `timing.csv` | Wall-clock cost per element at various input sizes. |
| `convergence_rate.csv` | Log-log convergence rate (slope and R-squared) for smooth functions. |

## Build

```bash
cd samples/fprint
make
```
