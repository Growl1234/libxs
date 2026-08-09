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

/** Histogram query result. */
LIBXS_EXTERN_C typedef struct libxs_hist_info_t {
  const int* buckets;
  const double* vals;
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
  const libxs_hist_update_t update[]);

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
