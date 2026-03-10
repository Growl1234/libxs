/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
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

/** Per-value update function: accumulates src into dst. */
LIBXS_EXTERN_C typedef void (*libxs_hist_update_t)(double* /*dst*/, const double* /*src*/);


/** Create histogram: nbuckets resolution, nvals per entry. Returns NULL on failure. */
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
LIBXS_API void libxs_hist_get(libxs_lock_t* lock, const libxs_hist_t* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals);

/** Print histogram to ostream (NULL ostream is accepted). */
LIBXS_API void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist, const char title[],
  const int prec[]);

/** Update function (libxs_hist_update_t): sliding average, arithmetic at commit. */
LIBXS_API void libxs_hist_update_avg(double* dst, const double* src);
/** Update function (libxs_hist_update_t): accumulate. */
LIBXS_API void libxs_hist_update_add(double* dst, const double* src);

#endif /*LIBXS_HIST_H*/
