# Histogram

Header: `libxs_hist.h`

Histogram with running statistics. Buckets by `vals[0]`; additional values per
entry track user-defined statistics. Functions taking a `lock` are thread-safe
when one is given; a NULL `lock` means no locking.

## Types

```C
typedef struct libxs_hist_t libxs_hist_t;  /* opaque */
typedef struct libxs_span_t libxs_span_t;  /* opaque */

typedef void (*libxs_hist_update_t)(double* dst, const double* src, int count);
typedef void (*libxs_hist_fold_t)(void* ctx, double* queue, int nvals, int nsamples);
```

An update function reduces one value into one bucket; the order of updates must
not matter. A fold receives a whole batch of samples at once and may keep state
of its own, e.g., to relate samples to one another.

```C
typedef struct libxs_hist_info_t {
  const int* buckets;   /* per-bucket counts [nbuckets] */
  const double* vals;   /* per-bucket values [nbuckets * nvals] */
  const double* sum;    /* running total per value [nvals] */
  double range[2];      /* [min, max] of vals[0] */
  int nbuckets, nvals, nsamples;
} libxs_hist_info_t;
```

`sum` is the total of the pushed values regardless of the update functions.
`nsamples` equals the sum of the bucket counts.

## Functions

```C
libxs_hist_t* libxs_hist_create(int nbuckets, int nvals,
  const libxs_hist_update_t update[], libxs_hist_fold_t fold, void* ctx);
void libxs_hist_destroy(libxs_hist_t* hist);
```

Create/destroy. `update[nvals]` specifies per-slot accumulation
(NULL defaults to avg). `fold` is optional (NULL for none) and receives `ctx`
unchanged; the histogram only borrows it, so it must outlive the histogram.
Destroy accepts NULL.

```C
void libxs_hist_push(libxs_lock_t* lock,
  libxs_hist_t* hist, const double vals[]);
```

Insert one sample. Samples are queued and folded into the buckets in batches,
so a histogram is exact while a batch fits and summarizes only once it is full.
Re-bins automatically if `vals[0]` exceeds the established range.

```C
void libxs_hist_query(libxs_lock_t* lock,
  const libxs_hist_t* hist, libxs_hist_info_t* info);
```

Query statistics, folding whatever is still queued. Repeated calls report the
same result rather than folding the same samples twice.

```C
void libxs_hist_query_percentile(libxs_lock_t* lock,
  const libxs_hist_t* hist, double vals[], double percentile);
void libxs_hist_query_median(libxs_lock_t* lock,
  const libxs_hist_t* hist, double vals[]);
void libxs_hist_query_mode(libxs_lock_t* lock,
  const libxs_hist_t* hist, double vals[]);
```

Values at a percentile [0..1], at the median (0.5), or of the most populated
bucket. All report per-bucket aggregates, blending only towards a populated
neighbour. Prefer the mode for a multi-modal distribution; its `vals[0]` is the
bucket's upper bound rather than an aggregate.

```C
void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist,
  const int prec[], const char fmt[], ...);
```

Print to stream: `fmt` (may be NULL) is a `printf` format for a header line,
followed by its arguments. Each non-empty bucket then prints one line with its
number, upper bound, count, and values. `prec[k]` (`nvals` entries, or NULL for
`%f`) sets the precision of value `k`, and `prec[0]` also that of the bound; a
negative entry omits its value, and a negative `prec[0]` omits the bucket lines.
A NULL `ostream` prints nothing. Printing takes no lock.

## Interval Union

```C
libxs_span_t* libxs_span_create(int nsegments);
void libxs_span_destroy(libxs_span_t* span);
void libxs_span_push(libxs_span_t* span, double begin, double end);
double libxs_span_total(const libxs_span_t* span, int* inexact);

void libxs_hist_fold_union(void* ctx, double* queue, int nvals, int nsamples);
```

The time a set of intervals covers, counting overlaps once. `nsegments` bounds
how far out of order intervals may arrive, not how many there may be: an
interval beginning before what has already been retired is added whole and
counted in `inexact`, and the total is then an upper bound.

`libxs_hist_fold_union` is a ready-made `libxs_hist_fold_t` that expects `ctx`
to be a `libxs_span_t*` and reads the last two values of a sample as
`(begin, end)`; pass it to `libxs_hist_create` to obtain both a distribution of
durations and the time they covered from one push.

Values must share an origin a double can hold exactly: subtract an epoch from
wall-clock nanoseconds first.

## Update Functions

```C
void libxs_hist_update_avg(double* dst, const double* src, int count);
void libxs_hist_update_add(double* dst, const double* src, int count);
void libxs_hist_update_min(double* dst, const double* src, int count);
void libxs_hist_update_max(double* dst, const double* src, int count);
```

- `avg` — Welford online mean: `*dst += (*src - *dst) / count`
- `add` — sum: `*dst += *src`
- `min` — `*dst = min(*dst, *src)`
- `max` — `*dst = max(*dst, *src)`

Custom callbacks use the same signature; `count` enables online
algorithms without external state.
