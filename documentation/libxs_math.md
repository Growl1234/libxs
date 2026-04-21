# Math Utilities

Header: `libxs_math.h`

## Matrix Difference

```C
typedef struct libxs_matdiff_t {
  double norm1_abs, norm1_rel;  /* one-norm */
  double normi_abs, normi_rel;  /* infinity-norm */
  double normf_rel;             /* Frobenius-norm (relative) */
  double linf_abs, linf_rel;    /* max difference (abs/rel at same element) */
  double l2_abs, l2_rel, rsq;   /* L2-norm and R-squared */
  double l1_ref, min_ref, max_ref, avg_ref, var_ref;  /* reference stats */
  double l1_tst, min_tst, max_tst, avg_tst, var_tst;  /* test stats */
  double diag_min_ref, diag_max_ref;   /* diagonal min/max (reference) */
  double diag_min_tst, diag_max_tst;   /* diagonal min/max (test) */
  double v_ref, v_tst;          /* values at max-diff location */
  int m, n, i, r;               /* location and reduction count */
} libxs_matdiff_t;
```

The fields linf_abs, linf_rel, and v_ref/v_tst always refer to the same element. For complex types, element values and statistics use the modulus; differences use the complex absolute error.

```C
int libxs_matdiff(libxs_matdiff_t* info,
  libxs_data_t datatype, int m, int n,
  const void* ref, const void* tst,
  const int* ldref, const int* ldtst);
```

Compute scalar differences between two matrices. Supports all real and integer `libxs_data_t` types, plus `LIBXS_DATATYPE_C64` and `LIBXS_DATATYPE_C32` (interleaved complex; dimensions refer to complex elements).

```C
double libxs_matdiff_epsilon(const libxs_matdiff_t* input);
```

Combined error margin from absolute and relative norms. Optionally logs to file via `LIBXS_MATDIFF` env var.

```C
int libxs_matdiff_combine(libxs_matdiff_t* output,
  const libxs_matdiff_t* input);
```

Combine two single-matrix infos (`ref=NULL`) into a meta-diff. Per-side statistics are exact; `linf_abs` is the mean shift, `l2_abs` a statistical bound.

```C
void libxs_matdiff_reduce(libxs_matdiff_t* output,
  const libxs_matdiff_t* input);
void libxs_matdiff_clear(libxs_matdiff_t* info);
```

Worst-case reduction of matdiff results. Initialize with `libxs_matdiff_clear`.

```C
double libxs_matdiff_posdef(const libxs_matdiff_t* info);  /* inline */
```

Returns the smallest test-side diagonal element (positive = necessary
condition for positive definiteness met). Zero if no diagonal data.

## Multiset Distance

Order-independent distance between two vectors treated as multisets.
The metric counts how many elements have no counterpart (within
tolerance) in the other vector. The result is a non-negative integer
that satisfies the metric properties: symmetry, non-negativity, and
identity of indiscernibles.

For each element in b, the algorithm checks whether any element in a
matches (and vice versa). The maximum of both one-sided match counts
determines the distance. Elements are compared by absolute difference
for real types and by complex modulus for C64/C32. The tolerance
threshold is inclusive (less-than-or-equal).

```C
int libxs_setdiff(libxs_data_t datatype,
  const void* a, int na,
  const void* b, int nb, double tol);
```

Supports all libxs_data_t types. Returns the number of unmatched
elements, or -1 for unsupported types. The distance is at least
abs(na - nb) when the vector lengths differ.

```C
int libxs_setdiff_min(libxs_data_t datatype,
  const void* a, int na,
  const void* b, int nb, double* tol);
```

Minimizes the multiset distance over all tolerances using Golden
Section Search (libxs_gss_min). Returns the minimum unmatched count.
The pointer tol (may be NULL) receives the smallest tolerance that
achieves this minimum. Supported types: F64, F32, C64, C32 only
(integer types are not meaningful here; returns max(na, nb) with
tol set to zero).

The distance as a function of tolerance is monotonically
non-increasing (a step function with discrete drops), which makes
it unimodal -- the prerequisite for Golden Section Search.

## Golden Section Search

```C
double libxs_gss_min(
  double (*fn)(double x, const void* data),
  const void* data,
  double x0, double x1, double* xmin, int maxiter);
```

Minimizes a unimodal function fn on the interval [x0, x1]. The
callback receives x and an opaque context pointer. Returns f(x*)
where x* is the minimizer; xmin (may be NULL) receives x*.

The bracket shrinks by factor phi = (sqrt(5)-1)/2 per iteration,
reusing one evaluation from the previous step. Convergence is
reached when the bracket collapses to machine precision or maxiter
iterations are exhausted.

## Number Theory

```C
size_t libxs_gcd(size_t a, size_t b);
size_t libxs_lcm(size_t a, size_t b);
```

GCD (returns 1 for gcd(0,0)) and LCM.

```C
int libxs_primes_u32(unsigned int num,
  unsigned int num_factors_n32[], int num_factors_max);
```

Prime factors of `num` in ascending order. Returns factor count.

```C
size_t libxs_coprime(size_t n, size_t minco);
size_t libxs_coprime2(size_t n);
```

Co-prime of `n` not exceeding `minco` (or sqrt(n)).

```C
unsigned int libxs_remainder(unsigned int a, unsigned int b,
  const unsigned int* limit, const unsigned int* remainder);
```

Smallest multiple of `b` minimizing remainder mod `a`.

```C
unsigned int libxs_product_limit(unsigned int product,
  unsigned int limit, int is_lower);
```

Sub-product of prime factors within `limit` (0/1-Knapsack).

## Scalar Utilities

```C
double libxs_kahan_sum(double value, double* accumulator,
  double* compensation);
```

Kahan compensated summation.

```C
unsigned int libxs_isqrt_u64(unsigned long long x);
unsigned int libxs_isqrt_u32(unsigned int x);
unsigned int libxs_isqrt2_u32(unsigned int x);
```

Integer square root. The isqrt2 variant returns a factor of x.

```C
double libxs_pow2(int n);
```

2^n via IEEE-754 exponent. Valid for n in [-1022, 1023]. Returns 0.0 for underflow (subnormals flushed to zero) and +Inf for overflow.

## Modular Arithmetic

```C
unsigned int libxs_mod_inverse_u32(unsigned int a, unsigned int m);
```

Modular inverse (extended Euclidean). Requires gcd(a, m) = 1.

```C
unsigned int libxs_barrett_rcp(unsigned int p);
unsigned int libxs_barrett_pow18(unsigned int p);
unsigned int libxs_barrett_pow36(unsigned int p);
```

Barrett reduction constants for modulus `p`.

```C
unsigned int libxs_mod_u32(uint32_t x, unsigned int p,
  unsigned int rcp);                           /* inline */
unsigned int libxs_mod_u64(uint64_t x, unsigned int p,
  unsigned int rcp, unsigned int pow18,
  unsigned int pow36);                         /* inline */
```

Fast modular reduction via Barrett. The 64-bit variant uses a radix-2^18 split.

## BF16 Conversion

```C
typedef uint16_t libxs_bf16_t;
libxs_bf16_t libxs_round_bf16(double x);   /* inline */
double libxs_bf16_to_f64(libxs_bf16_t v);  /* inline */
```

Round to BF16 (round-to-nearest-even) and expand to double. Uses native `__bf16` when available (`LIBXS_BF16`), otherwise portable bit manipulation.
