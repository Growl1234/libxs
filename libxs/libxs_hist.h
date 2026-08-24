/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_HIST_H
#define LIBXS_HIST_H

#include "libxs_sync.h"


/** Opaque histogram type. */
LIBXS_EXTERN_C typedef struct libxs_hist_t libxs_hist_t;

/** Per-value update function: accumulates src into dst (count includes current sample). */
LIBXS_EXTERN_C typedef void (*libxs_hist_update_t)(double* /*dst*/, const double* /*src*/, int /*count*/);

/**
 * Secondary fold, run over the raw batch each time the queue is folded, after
 * the per-bucket update has consumed it. It exists for what a bucket cannot
 * express: an update function reduces one value pairwise into one bucket, in an
 * order that must not matter, whereas this sees the whole batch at once and
 * keeps state of its own across folds.
 *
 * The batch rather than one sample at a time, deliberately: relating a sample to
 * its neighbours is the point, and having them together is what allows an exact
 * answer instead of an incremental approximation. The queue may be reordered in
 * place - it is discarded immediately after.
 *
 * ctx is whatever was handed to libxs_hist_create, opaque to the histogram.
 * Normally NULL, which is the identity.
 */
LIBXS_EXTERN_C typedef void (*libxs_hist_fold_t)(void* /*ctx*/,
  double* /*queue*/, int /*nvals*/, int /*nsamples*/);

/**
 * Union of intervals: the time they cover, as opposed to the sum of their
 * lengths, which counts overlapping intervals more than once. Opaque, because
 * only the total is meaningful outside - how it is retained is not.
 */
LIBXS_EXTERN_C typedef struct libxs_span_t libxs_span_t;

/** Histogram query result. */
LIBXS_EXTERN_C typedef struct libxs_hist_info_t {
  const int* buckets;
  const double* vals;
  /**
   * Running total per value over every pushed sample, accumulated on push and
   * hence independent of the update function: a min/max entry discards the
   * information its buckets would need to reconstruct a total, and an averaged
   * entry only yields one through count times mean. NULL if nothing was pushed.
   */
  const double* sum;
  double range[2];
  int nbuckets;
  int nvals;
  int nsamples;
} libxs_hist_info_t;


/**
 * Create histogram: nbuckets resolution, nvals per entry. Returns NULL on failure.
 * update may be NULL (all values use libxs_hist_update_avg), and individual
 * entries may be NULL to select that default per value.
 */
LIBXS_API libxs_hist_t* libxs_hist_create(int nbuckets, int nvals,
  const libxs_hist_update_t update[], libxs_hist_fold_t fold, void* ctx);

/**
 * Create an interval union retaining nsegments open segments. Only the segments
 * a later interval could still merge with are kept, so memory is fixed and the
 * result stays exact for anything inside that window; nsegments therefore bounds
 * how far out of order intervals may arrive, not how many there may be.
 */
LIBXS_API libxs_span_t* libxs_span_create(int nsegments);

/** Destroy an interval union (NULL is accepted). */
LIBXS_API void libxs_span_destroy(libxs_span_t* span);

/**
 * Merge one interval into the union. An interval beginning before what has
 * already been retired is added whole, because its overlap with that time can no
 * longer be subtracted; such intervals are counted (see libxs_span_total).
 *
 * The values must share an origin small enough for a double to hold exactly.
 * Wall-clock nanoseconds do not: they are far above 2^53, where the spacing
 * between representable values is hundreds of nanoseconds, enough to make
 * adjacent intervals appear to touch. Subtract an epoch before pushing.
 */
LIBXS_API void libxs_span_push(libxs_span_t* span, double begin, double end);

/**
 * Time covered by the union, including the segments still open. Optionally
 * reports how many intervals could not be merged exactly; because each of those
 * was added whole, a non-zero count makes the result an upper bound. It never
 * understates, which is what keeps a ratio of summed length over covered time a
 * lower bound on how many intervals overlapped.
 */
LIBXS_API double libxs_span_total(const libxs_span_t* span, int* inexact);

/**
 * Secondary fold (libxs_hist_fold_t) merging each sample's interval into a
 * libxs_span_t passed as ctx. The last two values of a sample are read as
 * (begin, end).
 */
LIBXS_API void libxs_hist_fold_union(void* ctx, double* queue, int nvals, int nsamples);

/** Destroy histogram (NULL is accepted). */
LIBXS_API void libxs_hist_destroy(libxs_hist_t* hist);

/** Insert a sample; vals[0] determines the bucket (lock can be NULL). */
LIBXS_API void libxs_hist_push(libxs_lock_t* lock, libxs_hist_t* hist, const double vals[]);

/**
 * Query statistics; commits queued items if pending (lock can be NULL).
 * Note: the histogram may be lazily committed (internal mutation) on the
 * first call; this is the C equivalent of C++ "mutable" and is safe
 * because the histogram is always heap-allocated.
 */
LIBXS_API void libxs_hist_query(libxs_lock_t* lock, const libxs_hist_t* hist,
  libxs_hist_info_t* info);

/** Query interpolated values at percentile (0..1); commits queued items if pending. */
LIBXS_API void libxs_hist_query_percentile(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[], double percentile);

/** Query interpolated values at median; commits queued items if pending. */
LIBXS_API void libxs_hist_query_median(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[]);

/**
 * Query the values of the most populated bucket ("modal"), without
 * interpolation; commits queued items if pending. Preferable to the median for a
 * multi-modal distribution: the median can fall between clusters and describe no
 * observation, whereas the mode always reports a bucket that samples landed in.
 * Ties resolve to the first such bucket. vals[0] receives the bucket's upper
 * bound, the remaining entries its aggregated values.
 */
LIBXS_API void libxs_hist_query_mode(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[]);

/** Print histogram to ostream (NULL ostream is accepted). */
LIBXS_API void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist, const int prec[],
  const char fmt[], ...);

/** Update function (libxs_hist_update_t): Welford's online mean. */
LIBXS_API void libxs_hist_update_avg(double* dst, const double* src, int count);
/** Update function (libxs_hist_update_t): accumulate. */
LIBXS_API void libxs_hist_update_add(double* dst, const double* src, int count);
/** Update function (libxs_hist_update_t): running minimum. */
LIBXS_API void libxs_hist_update_min(double* dst, const double* src, int count);
/** Update function (libxs_hist_update_t): running maximum. */
LIBXS_API void libxs_hist_update_max(double* dst, const double* src, int count);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_HIST_H*/
