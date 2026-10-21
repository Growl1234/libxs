# Ozaki-Scheme Low-Precision GEMM

Intercepts DGEMM, SGEMM, ZGEMM, and CGEMM at link time, replacing
them with low-precision Ozaki schemes (mantissa slicing or CRT). Real
GEMM calls go through the selected scheme; complex GEMM (ZGEMM/CGEMM)
is mapped to real GEMM internally.

## Build

```bash
cd samples/ozaki
make -j $(nproc)
```

The sample builds header-only when LIBXS is not built yet. When a
sibling `libxstream` directory is detected, the optional OpenCL/GPU path
is compiled in (see OZAKI_OCL below); `make OCL=0` or `make LIBXSTREAM=0`
leaves it out (default: `OCL=2`). BLAS is linked multi-threaded
(`BLAS=2`) by default, or sequential (`BLAS=1`) with `make OMP=0`.

`make LIBXSMM=1` puts LIBXSMM's int8 kernels ahead of the built-in
VNNI/AMX kernels; `OZAKI_XSMM=0` returns to the built-in kernels at
runtime. It needs a built sibling `libxsmm` directory. `make DNNL=1`
(with `DNNLROOT` set) runs Scheme-1 int8 GEMMs through oneDNN, and
`make TESTROOT=<path>` compiles in external test code (see
`TESTROOT.md` in this directory).

## Link-Time Interception

One of two link-time variants is built for each of DGEMM, SGEMM, ZGEMM,
and CGEMM, depending on how BLAS is linked:

- `dgemm-wrap.x`, `sgemm-wrap.x`, `zgemm-wrap.x`, `cgemm-wrap.x` —
  static BLAS (default), linked with `--wrap=` (GNU ld)
- `dgemm-blas.x`, `sgemm-blas.x`, `zgemm-blas.x`, `cgemm-blas.x` —
  `make BLAS_STATIC=0`, dynamically linked against BLAS and intercepted
  with `LD_PRELOAD=libwrap.so`

Static (`libwrap.a`) and shared (`libwrap.so`) wrapper libraries are
built by default. The shared library can intercept any application:

```bash
LD_PRELOAD=/path/to/libwrap.so ./application
```

The static library requires explicit `--wrap` flags:

```bash
-Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_
```

Test scripts: `test-wrap.sh` exercises all variants, `test-check.sh`
validates both schemes, `test-mhd.sh` tests MHD file-based input.

## Test Driver

```bash
dgemm-wrap.x [A.mhd|M [B.mhd|N [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]]
```

Defaults: M=N=K=257, TA=TB=0, ALPHA=BETA=1.0. Passing MHD filenames
as the first two arguments loads A and B from files. The `zgemm-*.x`
and `cgemm-*.x` drivers call ZGEMM and CGEMM.

## Environment Variables

### Scheme Selection

| Variable        | Default   | Description                                                       |
|-----------------|-----------|-------------------------------------------------------------------|
| OZAKI           | 3         | 0=bypass (BLAS), 1=mantissa slicing, 2=CRT, 3=adaptive            |
| OZAKI_COMPLEX   | (auto)    | Complex dispatch: 0=BLAS, 1=CPU, 2=GPU+fallback, 3=BLAS, with real GEMMs issued inside it kept on BLAS. Auto: 0 if OZAKI=0, else 2 |
| OZAKI_N         | (auto)    | Slices (Sch.1: fp64=8, fp32=4) or moduli (Sch.2: fp64=16, fp32=9) |
| OZAKI_AMX       | (auto)    | CPU: 0=VNNI, 1=AMX. Auto: AMX for Sch.1, for Sch.2 above a size threshold |
| OZAKI_XSMM      | 1         | With `make LIBXSMM=1`: 0=built-in int8 kernels, 1=LIBXSMM         |

OZAKI=3 (the default) selects the scheme automatically. On the CPU, it
is Scheme 1 where AMX is available (unless OZAKI_AMX=0), and Scheme 2
otherwise. On the GPU, LIBXSTREAM selects the scheme per call. An
explicit OZAKI applies to both.

### Threading (CPU)

| Variable        | Scheme 1 (OZAKI=1) | Scheme 2 (OZAKI=2) |
|-----------------|---------------------|--------------------|
| OMP_PROC_BIND   | true                | true               |
| OMP_PLACES      | cores               | cores              |
| OMP_NUM_THREADS | number of cores     | unset              |

Keep threads bound as above; unbound threads are slower.

### Accuracy

| Variable          | Default   | Description                                                      |
|-------------------|-----------|------------------------------------------------------------------|
| OZAKI_FLAGS       | (auto)    | Sch.1: 3=triangular and symmetrized, 0=full S^2 square (other values are ignored). Auto: 3 below OZAKI_SYM_XOVER, else 0 |
| OZAKI_SYM_XOVER   | 2048      | Largest of M, N, K from which auto OZAKI_FLAGS selects 0         |
| OZAKI_TRIM        | 0         | Levels to trim (0=default); positive values lose accuracy, negative values add precision |
| OZAKI_MAXK        | 32768     | Max K per preprocessing pass (0=full K in one pass)              |
| OZAKI_THRESHOLD   | 12        | Intensity threshold (debug build: 0). Bypass when flops/(bytes\*thr)<1. 0=always |

`OZAKI_MAXK` also applies to the GPU path (see the
[LIBXSTREAM Ozaki README](https://github.com/hfp/libxstream/blob/main/samples/ozaki/README.md)).

### GPU Path (requires LIBXSTREAM)

| Variable    | Default   | Description                                |
|-------------|-----------|--------------------------------------------|
| OZAKI_OCL   | 0         | Enable OpenCL/GPU path (0=off, 1=on)       |
| OZAKI_TM    | (auto)    | GPU output tile height (multiple of 8)     |
| OZAKI_TN    | (auto)    | GPU output tile width (multiple of 16)     |

GPU-specific kernel tuning variables (OZAKI_RTM, OZAKI_RTN, OZAKI_WG,
OZAKI_SG, OZAKI_KU, OZAKI_RC, OZAKI_PB, OZAKI_HIER, OZAKI_GROUPS,
OZAKI_PREFETCH, OZAKI_SCALAR_ACC, OZAKI_CACHE, OZAKI_ARENA) are
documented in the
[LIBXSTREAM Ozaki README](https://github.com/hfp/libxstream/blob/main/samples/ozaki/README.md).
LIBXSTREAM reads them itself, except OZAKI_GROUPS, which this driver
reads and passes to the GPU context; all of them take effect only with
`OZAKI_OCL=1`.

### Monitoring and Diagnostics

| Variable        | Default   | Description                                                       |
|-----------------|-----------|-------------------------------------------------------------------|
| OZAKI_VERBOSE   | 0         | 0=silent, 1=stats at exit, N=print every Nth GEMM call            |
| OZAKI_STAT      | 0         | Track C-matrix (0), A-representation (1), or B-representation (2) |
| OZAKI_DUMP      | 0         | Dump A/B as MHD files at the given call-count (0=off)             |
| OZAKI_EPS       | inf       | Dump A/B when epsilon error exceeds threshold                     |
| OZAKI_IDX       | 0         | Additional OZAKI_EPS metric: 1=linf_abs, 2=linf_rel, 3=l2_rel (0=epsilon only) |
| OZAKI_RSQ       | 0         | Dump A/B when RSQ drops below threshold (updated after dump)      |
| OZAKI_EXIT      | 1         | Exit on accuracy violation after dump. 0=continue                 |

### Benchmark

| Variable     | Default   | Description                                                        |
|--------------|-----------|--------------------------------------------------------------------|
| CHECK        | 0         | Validate vs BLAS: 0=off, negative=auto-threshold, positive=custom  |
| GRADE        | 0         | Componentwise accuracy grade vs BLAS (see below)                   |
| XREF         | 0         | Compensated reference on n rows, negative=all rows (see below)     |
| NREPEAT      | 3         | Number of GEMM calls (first call is warmup when >1)                |
| EVIL         | 0         | Adversarial exponent-span test (see below)                         |
| TAME         | 0         | Keep n mantissa bits and clear the rest (see below)                |
| SCALE        | 0         | Multiply A and B by 2^n after generation                           |
| OZAKI_DECAY  | 0         | Decay diagnostic + K-permutation (0-4, see below)                  |
| GEMM_HOSTMEM | 0         | Operands from LIBXSTREAM (page-locked) instead of `malloc`         |

By default, operands are allocated with `malloc`, and no device is
opened unless the GPU path is enabled (`OZAKI_OCL=1`). A non-zero
`GEMM_HOSTMEM` allocates page-locked memory through LIBXSTREAM, which
opens a device.

### Adversarial Input (EVIL)

The EVIL variable initializes A and B with controlled exponent
structure for stress-testing accuracy.
The magnitude sets the exponent span in bits; the sign selects
the distribution:

- `EVIL=N` (N>0): per-column. Column j of A is scaled by
  2^(−N + round(2N·j/(ncols−1))), column j of B by the inverse, so the
  span is 2N binades centred on zero. The product A·B is
  well-conditioned, and exponents are uniform within each column.
- `EVIL=-N` (N>0): per-element. Each element gets a pseudorandom
  exponent in [0,N] via coprime shuffle, with opposite sign for B.
  Every row of A spans the full exponent range.
- `EVIL=0`: default shuffle mode (no exponent structure).

The per-column mode (EVIL>0) is a diagonal scaling with D and D^-1,
where EVIL is the exponent-range parameter.

`TAME=n` keeps n mantissa bits and clears the rest, e.g., `TAME=24` is
data promoted from single precision. TAME and EVIL can be combined.
Each run reports the generator version (`LIBXS_MATRNG_VERSION`) as
`DATA: matrng=...`.

### Accuracy Grade (GRADE)

`GRADE=1` applies the componentwise criterion of the graded BLAS
accuracy tests: `|C - C_ref| <= f(K) * u * (|alpha||A||B| +
|beta||C|)` with `f(K) = K + 1024`, `u` the unit roundoff, and K the
number of terms summed. It reports `GRADE: a=<max ratio> f(n)=<K>`
and fails the run when the ratio exceeds it. `GRADE=-1` reports
without failing (e.g., for runs with `OZAKI_TRIM`). By contrast,
`CHECK` compares one scalar against a fixed threshold (1e-10 for
double, 1e-3 for single). GRADE costs one extra reference GEMM and
one m-by-n buffer.

### Compensated Reference (XREF)

`XREF=n` adds a reference computed by `libxs_dot2_gemm`, as if in
twice the double precision, on n rows spread over C (`XREF=-1` takes
all rows). It reports

  XREF: rows=n ozaki_linf_rel=.. ozaki_l2_rel=.. ozaki_grade=..
        blas_linf_rel=.. blas_l2_rel=.. blas_grade=..

for the Ozaki and the BLAS result. The grades appear with `GRADE`
set, and `GRADE=1` then also fails the run on the compensated grade
(`GRADE XREF:`). With beta non-zero, the reference replays the warmup
and the timed calls. The cost is O(nNK) per replayed call on the
host, hence a few rows suit large sizes.

### Decay Diagnostic and K-Permutation (OZAKI_DECAY)

Controls K-dimension row reordering for smoothness and the
forward-difference decay diagnostic. Values map to libxs_sort_t:

| Value | Behavior                                                     |
|-------|--------------------------------------------------------------|
| 0     | Off (default). No permutation, no diagnostic.                |
| 1     | Identity permutation. Measure decay on original ordering.    |
| 2     | Sort B rows by L1-norm, apply same K-permutation to A.       |
| 3     | Sort B rows by mean value, apply same K-permutation to A.    |
| 4     | Greedy nearest-neighbor chain on B rows (O(K^2\*N)).         |

Values 2-4 compute a K-permutation from B (via libxs_sort_smooth)
before Ozaki-1 slicing; both A and B are sliced in the permuted
K-order, and the product is unchanged.

When nonzero, reports the forward-difference decay of int8 slice
buffers along K, M, and N axes (first K-group only, single-
threaded). Output goes to stderr:

  OZ1[MxNxK] Delta-K: d1=... d2=... ...
  OZ1[MxNxK] Delta-M: d1=... d2=... ...
  OZ1[MxNxK] Delta-N: d1=... d2=... ...

## Profiling

Set `LIBXSTREAM_PROFILE=1` for per-kernel device timings, reported at
program exit:

```text
PROF ACC/OpenCL: ID=<id> <kernel>=<time> ms <rate> GFLOPS/s
```

There is no separate host-side profile: on the CPU, the `OZAKI GEMM`
line reports the mean over the timed loop (see `NREPEAT`).

## Example

```bash
./dgemm-wrap.x 256                                      # adaptive scheme (default)
OZAKI=1 ./dgemm-wrap.x 256                              # mantissa slicing
OZAKI_TRIM=2 OZAKI=1 ./dgemm-wrap.x 256                 # trim 2 levels (less accurate)
OZAKI_OCL=1 OZAKI=2 OZAKI_GROUPS=4 ./dgemm-wrap.x 4096  # GPU CRT with K-grouping
OZAKI=3 ./dgemm-wrap.x 4096                             # adaptive scheme selection
EVIL=512 ./dgemm-wrap.x 1024                            # accuracy grading (wide exponent span)
EVIL=1 ./dgemm-wrap.x 1024                              # narrow exponent span
EVIL=-52 ./dgemm-wrap.x 1024                            # per-element exponent span
OZAKI_DECAY=1 OZAKI=1 ./dgemm-wrap.x 256                # decay diagnostic (no reorder)
OZAKI_DECAY=2 OZAKI=1 ./dgemm-wrap.x 256                # sort by norm + decay diagnostic
OZAKI_DECAY=4 OZAKI=1 ./dgemm-wrap.x 256                # greedy NN + decay diagnostic
```
