# Batched GEMM

This sample code exercises the LIBXS batched GEMM API (`libxs_gemm.h`) in three flavors: strided, pointer-array, and grouped. All three programs are multithreaded via OpenMP (using the `_task` variants where applicable), report wall-clock time and GFLOPS/s, and optionally validate results via `libxs_matdiff`.

## Building

LIBXS must be built first from the repository root. Then:

```bash
cd samples/gemm
make
```

For strict C89 pedantic checking:

```bash
make GNU=1 DBG=1 PEDANTIC=2
```

## Programs

### gemm_strided

Strided batch DGEMM: all matrices are packed contiguously in memory with constant stride between consecutive A, B, and C matrices. Uses `libxs_gemm_index` with `index_stride=0` (constant-stride mode). With OpenMP, work is distributed across threads via `libxs_gemm_index_task`.

```
./gemm_strided.x [M [N [K [batchsize [nrepeat]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3.

### gemm_batch

Pointer-array batch DGEMM: matrices are accessed through arrays of pointers. Supports a duplicate C-matrix mode to exercise the lock-forward synchronization path. With OpenMP, work is distributed via `libxs_gemm_batch_task`.

```
./gemm_batch.x [M [N [K [batchsize [nrepeat [dup]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, dup=0.

The `dup` argument controls duplicate C-matrix references:

| dup | Mode | Description |
|-----|------|-------------|
| 0 | None | Each C-matrix is unique (`LIBXS_GEMM_FLAG_NOLOCK`). |
| 1 | Sorted | Half-unique C-pointers, duplicates consecutive. |
| 2 | Shuffled | Half-unique C-pointers, randomly shuffled (stresses lock-forward). |

## MKL JIT Support

When built with Intel MKL (`__MKL` defined), `gemm_strided` and `gemm_batch` automatically create a JIT-compiled DGEMM kernel via `mkl_cblas_jit_create_dgemm` and wire it into `libxs_gemm_config_t`. The JIT path is selected at runtime only when the MKL headers provide the `mkl_jit_create_dgemm` macro; otherwise the built-in default kernel is used transparently.

## Validation

Set the `CHECK` environment variable to enable a post-run sanity check using `libxs_matdiff`:

```bash
CHECK=1 ./gemm_strided.x
CHECK=1 ./gemm_batch.x 23 23 23 1000 1 2
```

This prints the L1 norm (`l1_tst`) of the first C-matrix. The golden value is deterministic for a given shape and repeat count, so it can be used for regression testing across configurations and kernels.
