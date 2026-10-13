# Bracket Units

This sample fits piecewise-constant targets (steps, a square wave, and a
checkerboard, axis-aligned and turned by 30°, and a disc) with networks of truncated-power units ⟨w·x − b⟩ᵐ, and
measures how sharp the learned edges are. Training uses Adam with full
batches; the two-dimensional models use LIBXS GEMM. By default the training data
are pixels, i.e., averages over cells, as an image sensor records them.

| Variant         | Model                                                                  |
|-----------------|------------------------------------------------------------------------|
| `direct`        | One hidden layer of ⟨w·x − b⟩¹ (ReLU) units fitted to the samples.      |
| `integrated`    | The same model fitted to the running integral, then differentiated.    |
| `*_wd`          | As above, with decoupled weight decay.                                 |
| `*_merge`       | Integrated, then jumps within one sample spacing merged and refined.   |
| `*_greedy`      | As merged, then jumps inserted where the residual calls for one.       |
| `*_prune`       | As greedy, then jumps removed while the fit stays within the estimated noise. |
| `mlp`           | Two hidden ReLU layers fitted to the checkerboard.                     |
| `mlp_int`       | The same network fitted to the integral along (1,1), then ∂/∂x + ∂/∂y. |
| `tensor`        | Products ⟨x − a⟩¹⟨y − b⟩¹ fitted to the checkerboard.                    |
| `tensor_int`    | The same products fitted to the summed-area table, then ∂²/∂x∂y.       |
| `sep`           | Sums of products A(x)·B(y) of 1-D bracket fits, fitted to the pixels.   |
| `sep_int`       | The same fitted to the summed-area table (merged, greedy), then ∂²/∂x∂y. |
| `rot`           | One product A(u)·B(v) in a turned frame (u, v), fitted to the pixels.   |
| `rot_int`       | The same fitted to running sums in that frame, then ∂²/∂u∂v; the angle is searched. |
| `box`           | Products of clamped ramps (⟨x − a⟩¹ − ⟨x − b⟩¹)(⟨y − e⟩¹ − ⟨y − g⟩¹), fitted to the pixels. |
| `box_int`       | The same fitted to the summed-area table, then ∂²/∂x∂y: a sum of rectangles. |
| `box_greedy`    | Rectangles added one at a time where the residual gains most, over a background; refined on the pixels. |

## Build and Run

```bash
cd samples/bracket
make
./bracket.x
```

The two-dimensional networks dominate the runtime. The sample links BLAS by
default; `make BLAS=0` falls back to the built-in GEMM, which is about 20× slower.

Each row of the printed table reports:

| Column   | Meaning                                                                                  |
|----------|------------------------------------------------------------------------------------------|
| `err`    | Error against the noise-free target on the fine evaluation grid.                         |
| `pos/h`  | Mean distance between true and learned edges, in training-grid spacings.                 |
| `slope`  | Growth of the order-1 fingerprint norm under refinement: 0.5 is a jump, 0 a smooth edge. |
| `eps/h`  | Effective edge width implied by that norm, in training-grid spacings.                    |
| `fpdist` | Fingerprint distance (`libxs_fprint_diff`, orders 0 and 1) to the noise-free target.     |

For the checkerboard, the fingerprint norms are the root mean square over image rows.

## Environment Variables

| Variable          | Default | Meaning                                                    |
|-------------------|---------|------------------------------------------------------------|
| `BRACKET_OUTDIR`  | `.`     | Directory for MHD output (empty: no output).               |
| `BRACKET_NSTEPS`  | 20000   | Adam steps in 1-D; the 2-D models run a quarter of them.   |
| `BRACKET_WD`      | 0.01    | Weight decay of the `*_wd` variants.                       |
| `BRACKET_SEED`    | 1       | Seed for initialization and noise.                         |
| `BRACKET_UNITS`   | 32      | Hidden units of the 1-D models (at most 64).               |
| `BRACKET_SAMPLES` | 64      | Training samples of the 1-D targets (at most 256).         |
| `BRACKET_WIDTH`   | 64      | Hidden width of the 2-D networks.                          |
| `BRACKET_2D`      | 1       | Set to 0 to skip the checkerboard.                         |
| `BRACKET_PIXEL`   | 1       | Set to 0 for point samples instead of cell averages.       |
| `BRACKET_RANK`    | 4       | Largest number of products in `sep` and `sep_int`.         |
| `BRACKET_ANGLE`   | 30      | Angle of the turned checkerboard in degrees.               |

`bracket_1d_<target>.mhd` stacks the target and each variant's output as rows.
`bracket_2d_<target>.mhd` holds a target (`checker`, `turned`, or `disc`), and
`bracket_2d_<target>_<variant>.mhd` each model's image, clamped to the target's
range [−1, 1] after the metrics are taken. The directory must exist. Set `LIBXS_MHD_PNG=1` to also write a PNG next to each two-dimensional
MHD file:

```bash
LIBXS_MHD_PNG=1 ./bracket.x
```
