# Bracket Units

This sample fits piecewise-constant targets (steps, a square wave, and a
checkerboard) with networks of truncated-power units ⟨w·x − b⟩ᵐ, and
measures how sharp the learned edges are. Training uses Adam with full
batches; the two-dimensional models use LIBXS GEMM.

| Variant         | Model                                                                 |
|-----------------|-----------------------------------------------------------------------|
| `direct`        | One hidden layer of ⟨w·x − b⟩¹ (ReLU) units fitted to the samples.    |
| `integrated`    | The same model fitted to the running integral, then differentiated.  |
| `*_wd`          | As above, with decoupled weight decay.                                |
| `mlp`           | Two hidden ReLU layers fitted to the checkerboard.                    |
| `tensor`        | Products ⟨x − a⟩¹⟨y − b⟩¹ fitted to the checkerboard.                 |
| `tensor_int`    | The same products fitted to the summed-area table, then ∂²/∂x∂y.      |

## Build and Run

```bash
cd samples/bracket
make
./bracket.x
```

Each row of the printed table reports:

| Column   | Meaning                                                                    |
|----------|----------------------------------------------------------------------------|
| `rms`    | Error against the noise-free target on the training grid.                  |
| `pos/h`  | Mean distance between true and learned edges, in training-grid spacings.   |
| `slope`  | Growth of the order-1 fingerprint norm under refinement: 0.5 is a jump, 0 a smooth edge. |
| `eps/h`  | Effective edge width implied by that norm, in training-grid spacings.      |
| `fpdist` | Fingerprint distance (`libxs_fprint_diff`) to the noise-free target.       |

## Environment Variables

| Variable          | Default | Meaning                                                    |
|-------------------|---------|------------------------------------------------------------|
| `BRACKET_OUTDIR`  | unset   | Directory for MHD output (none if unset).                  |
| `BRACKET_NSTEPS`  | 20000   | Adam steps in 1-D; the 2-D models run a quarter of them.   |
| `BRACKET_WD`      | 0.01    | Weight decay of the `*_wd` variants.                       |
| `BRACKET_SEED`    | 1       | Seed for initialization and noise.                         |

With `BRACKET_OUTDIR` set, `bracket_1d_<target>.mhd` stacks the target and each
variant's output as rows, and `bracket_2d_<variant>.mhd` holds each checkerboard
image. Set `LIBXS_MHD_PNG=1` to also write a PNG next to each MHD file:

```bash
mkdir -p out
BRACKET_OUTDIR=out LIBXS_MHD_PNG=1 ./bracket.x
```
