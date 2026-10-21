# Random Number Generator

Header: `libxs_rng.h`

Thread-safe pseudo-random number generation (SplitMix64 with
per-thread TLS state).

## Matrix Initialization Macros

```C
LIBXS_MATRNG(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE)
LIBXS_MATRNG_SEQ(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE)
LIBXS_MATRNG_OMP(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE)
```

Fill a column-major matrix (NROWS x NCOLS, leading dimension LD)
with deterministic values. The `_OMP` variant parallelizes with
OpenMP; `_SEQ` is serial.

`LIBXS_MATRNG_VERSION` (currently 2) changes whenever the generated values
change; record it alongside results so that data from different generators
is not mixed.

ESPAN (exponent span) selects the initialization mode:

- `ESPAN == 0`: shuffle mode. A co-prime permutation of the linear index
  (over LD·NCOLS) gives each element a distinct value in
  \[−|SCALE|, +|SCALE|\].
- `ESPAN != 0`: adversarial exponent-span mode for stress-testing
  floating-point emulation (Ozaki scheme, etc.). Base values are shuffled in
  \[1, 2), and column j is scaled by 2^e with

  ```text
  e = sign(ESPAN) · (round(2·|ESPAN|·j / (NCOLS−1)) − |ESPAN|)
  ```

  so column 0 has exponent −ESPAN and the last column +ESPAN, 2·|ESPAN|
  binades in total. SCALE is ignored in this mode. Use +ESPAN for matrix A
  and −ESPAN for matrix B so that A·B remains well-conditioned while each
  operand individually has adversarial exponent range.

In both modes, padding rows \[NROWS, LD) are zero-filled.

Example — adversarial DGEMM test with exponent span 512:

```C
int espan = 512;
LIBXS_MATRNG(int, double, +espan, a, m, k, lda, 1.0);
LIBXS_MATRNG(int, double, -espan, b, k, n, ldb, 1.0);
```

## Functions

```C
void libxs_rng_set_seed(unsigned int seed);
```

Set the seed of the calling thread's PRNG state. Each thread
maintains independent state via TLS. Unseeded threads start
with a deterministic default (seed = 1).

```C
unsigned int libxs_rng_u32(unsigned int n);
```

Return a pseudo-random value in \[0, n) with uniform distribution, or zero
for n <= 1. Thread-safe.

```C
double libxs_rng_f64(void);
```

Return a double-precision value in \[0, 1) with full 53-bit
mantissa resolution. Thread-safe.

```C
void libxs_rng_seq(void* data, size_t nbytes);
```

Fill a buffer with pseudo-random bytes. Thread-safe.
