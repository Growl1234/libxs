# Setdiff

Runs small deterministic experiments for `libxs_setdiff` and
`libxs_setdiff_min`. The sample writes CSV files that are useful for
paper figures and for quick local proxy runs.

## Build

```bash
cd samples/setdiff
make GNU=1
```

## Usage

```bash
./setdiff.x [outdir [sizes [reps [land_n [land_steps]]]]]
```

| Positional | Default                 | Description |
|------------|-------------------------|-------------|
| outdir     | `results/setdiff`       | Directory for CSV output |
| sizes      | `128 1024 8192 65536`   | Vector sizes for tolerance and scaling runs |
| reps       | `10`                    | Repetitions per scaling size |
| land_n     | `32`                    | Tolerance-landscape vector length |
| land_steps | `40`                    | Number of tolerance grid steps |

## Output

- `summary.csv` -- order-independence and duplicate-consumption cases
- `landscape.csv` -- sampled tolerance landscape
- `tolerance.csv` -- `libxs_setdiff_min` compared with bisection
- `scaling.csv` -- fixed-tolerance and automatic-tolerance timings
- `complex.csv` -- complex-valued validation case

The sample uses a fixed pseudo-random seed and writes into a deterministic
output directory by default.
