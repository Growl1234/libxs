# Histogram

Header: `libxs_hist.h`

Thread-safe histogram that accumulates values and provides running statistics. Values are bucketed by the first element of each inserted tuple; additional per-entry values track user-defined statistics (e.g., timing data).

## Types

```C
typedef struct libxs_hist_t libxs_hist_t;  /* opaque */
```

```C
typedef void (*libxs_hist_update_t)(double* dst, const double* src);
```

Per-value accumulation callback. Two built-in update functions are provided: `libxs_hist_update_avg` (sliding average) and `libxs_hist_update_add` (additive accumulation).

## Functions

```C
libxs_hist_t* libxs_hist_create(int nbuckets, int nvals,
  const libxs_hist_update_t update[]);
```

Create a histogram with `nbuckets` resolution. `nvals` is the number of values per entry (the first value determines the bucket). The `update` array (length `nvals`) specifies the accumulation function for each value slot; NULL entries default to `libxs_hist_update_avg`. Returns NULL on failure.

```C
void libxs_hist_destroy(libxs_hist_t* hist);
```

Destroy a histogram. Accepts NULL.

```C
void libxs_hist_push(libxs_lock_t* lock,
  libxs_hist_t* hist, const double vals[]);
```

Insert one sample. `vals[0]` selects the bucket; remaining entries are accumulated according to the update functions. The `lock` parameter can be NULL (internal locking is used).

```C
void libxs_hist_get(libxs_lock_t* lock,
  const libxs_hist_t* hist,
  const int** buckets, int* nbuckets,
  double range[2], const double** vals, int* nvals);
```

Query statistics. Output pointers can be NULL to skip fields.

```C
void libxs_hist_get_percentile(libxs_lock_t* lock,
  const libxs_hist_t* hist, double percentile,
  double vals[]);
```

Query interpolated values at the given percentile. `percentile` is in the range `[0..1]` (clamped). `vals[]` is filled with interpolated results: `vals[0]` is the interpolated primary value (position along the range), and subsequent entries are linearly interpolated between adjacent buckets. Commits queued items if pending.

```C
void libxs_hist_get_median(libxs_lock_t* lock,
  const libxs_hist_t* hist, double vals[]);
```

Convenience wrapper equivalent to `libxs_hist_get_percentile(lock, hist, 0.5, vals)`.

```C
void libxs_hist_print(FILE* ostream,
  const libxs_hist_t* hist, const int prec[],
  const char fmt[], ...);
```

Render the histogram to a stream. `prec` controls decimal precision per value (can be NULL); a negative `prec[k]` omits that value from output, and a negative `prec[0]` skips the bucket line entirely. `fmt` is a printf-style format string for the title line, followed by optional variadic arguments. NULL `ostream` or `fmt` is accepted.

```C
void libxs_hist_update_avg(double* dst, const double* src);
void libxs_hist_update_add(double* dst, const double* src);
```

Built-in update functions: `libxs_hist_update_avg` computes a sliding average (arithmetic mean), `libxs_hist_update_add` sums values.
