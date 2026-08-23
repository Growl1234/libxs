/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_hist.h>
#include <libxs/libxs_mem.h>
#include <stdarg.h>


/**
 * One allocation, three regions that never overlap: the per-bucket summary, the
 * queue of raw samples behind it, and the running totals. Sharing an allocation
 * is deliberate; sharing storage is not - a fold that aggregates into a slot the
 * queue still occupies has to displace whatever sits there, and a displaced
 * entry is one the scan can no longer reach.
 *
 * nbuckets may shrink at the first fold, so the region pointers are taken once
 * at creation and never recomputed from it.
 */
LIBXS_EXTERN_C struct libxs_hist_t {
  libxs_hist_update_t* update;
  libxs_hist_fold_t fold;
  double *vals, *queue, *sum, *state, min, max;
  int *buckets, nbuckets, nqueue, nq, nvals, n, nfold, nstate;
};


LIBXS_API libxs_hist_t* libxs_hist_create(int nbuckets, int nvals,
  const libxs_hist_update_t update[], libxs_hist_fold_t fold, int nstate)
{
  libxs_hist_t* h = (libxs_hist_t*)malloc(sizeof(libxs_hist_t));
  LIBXS_ASSERT(0 < nbuckets && 0 < nvals);
  if (NULL != h) {
    const int nqueue = 16 * nbuckets;
    if (NULL == fold || 0 > nstate) nstate = 0;
    h->vals = (double*)malloc(sizeof(double) * ((nbuckets + nqueue + 1) * nvals + nstate));
    h->update = (libxs_hist_update_t*)malloc(sizeof(libxs_hist_update_t) * nvals);
    h->buckets = (int*)calloc(nbuckets, sizeof(int));
    if (NULL != h->vals && NULL != h->buckets && NULL != h->update) {
      const union { uint32_t raw; float value; } inf = { 0x7F800000U };
      h->min = +inf.value;
      h->max = -inf.value;
      h->nbuckets = nbuckets;
      h->nqueue = nqueue;
      h->nvals = nvals;
      h->nq = 0;
      h->nfold = 0;
      h->fold = fold;
      h->nstate = nstate;
      h->queue = h->vals + (nbuckets * nvals);
      h->sum = h->queue + (nqueue * nvals);
      h->state = (0 < nstate ? (h->sum + nvals) : NULL);
      for (h->n = 0; h->n < nstate; ++h->n) h->state[h->n] = 0;
      for (h->n = 0; h->n < nvals; ++h->n) {
        h->update[h->n] = (NULL != update) ? update[h->n] : NULL;
        h->sum[h->n] = 0;
      }
      h->n = 0;
    }
    else {
      free(h->buckets);
      free(h->update);
      free(h->vals);
      free(h);
      h = NULL;
    }
  }
  return h;
}


/**
 * State layout: [0] union already retired, [1] the watermark it was retired up
 * to, [2] how many segments are still open, [3] intervals that could not be
 * merged exactly, then that many (begin, end) pairs kept in ascending order and
 * disjoint. Only the open segments are kept, so memory is fixed while the
 * result stays exact for anything still inside that window.
 */
LIBXS_API void libxs_hist_fold_union(double* state, int nstate,
  double* queue, int nvals, int nsamples)
{
  const int k = (6 <= nstate ? (nstate - 4) / 2 : 0);
  if (0 < k && 2 <= nvals && 0 < nsamples && NULL != state && NULL != queue) {
    const int ib = nvals - 2, ie = nvals - 1;
    double* const seg = state + 4;
    int nseg = (int)state[2], m, i, j, n;
    /* Sorted by begin so the sweep below is linear. In place, because the queue
       is discarded as soon as this returns. */
    for (m = 1; m < nsamples; ++m) {
      for (j = m; 0 < j && queue[j * nvals + ib] < queue[(j - 1) * nvals + ib]; --j) {
        for (n = 0; n < nvals; ++n) {
          const double t = queue[j * nvals + n];
          queue[j * nvals + n] = queue[(j - 1) * nvals + n];
          queue[(j - 1) * nvals + n] = t;
        }
      }
    }
    for (m = 0; m < nsamples; ++m) {
      double b = queue[m * nvals + ib], e = queue[m * nvals + ie];
      if (e < b) { /* tolerate a reversed pair rather than accumulating nonsense */
        const double t = b;
        b = e;
        e = t;
      }
      /* below the watermark it may overlap what was already retired, and that
         cannot be undone - counted, so the result is never quietly wrong */
      if (b < state[1]) state[3] += 1;
      for (i = 0; i < nseg && seg[2 * i + 1] < b; ++i);
      for (j = i; j < nseg && seg[2 * j] <= e; ++j) {
        if (seg[2 * j] < b) b = seg[2 * j];
        if (seg[2 * j + 1] > e) e = seg[2 * j + 1];
      }
      if (i == j) { /* disjoint from every open segment: insert it */
        if (k <= nseg) { /* no room, so the earliest segment is retired */
          state[0] += seg[1] - seg[0];
          if (seg[1] > state[1]) state[1] = seg[1];
          for (n = 0; n < 2 * (nseg - 1); ++n) seg[n] = seg[n + 2];
          --nseg;
          if (0 < i) --i;
        }
        for (n = 2 * nseg - 1; n >= 2 * i; --n) seg[n + 2] = seg[n];
        ++nseg;
      }
      else if (i + 1 < j) { /* it bridged several segments, now one */
        const int d = 2 * (j - i - 1);
        for (n = 2 * (i + 1); n < 2 * nseg - d; ++n) seg[n] = seg[n + d];
        nseg -= (j - i - 1);
      }
      seg[2 * i] = b;
      seg[2 * i + 1] = e;
    }
    state[2] = nseg;
  }
}


LIBXS_API double libxs_hist_union(const libxs_hist_info_t* info, int* inexact)
{
  double result = 0;
  int n = 0;
  if (NULL != info && NULL != info->state && 6 <= info->nstate) {
    const int k = (info->nstate - 4) / 2;
    int nseg = (int)info->state[2], i;
    if (nseg > k) nseg = k;
    result = info->state[0];
    for (i = 0; i < nseg; ++i) { /* the segments still open */
      result += info->state[5 + 2 * i] - info->state[4 + 2 * i];
    }
    n = (int)info->state[3];
  }
  if (NULL != inexact) *inexact = n;
  return result;
}


LIBXS_API void libxs_hist_destroy(libxs_hist_t* hist)
{
  if (NULL != hist) {
    free(hist->buckets);
    free(hist->update);
    free(hist->vals);
    free(hist);
  }
}


LIBXS_API_INTERN void internal_libxs_hist_rebin(libxs_hist_t* h, double new_min, double new_max);
LIBXS_API_INTERN void internal_libxs_hist_rebin(libxs_hist_t* h, double new_min, double new_max)
{
  const double old_w = h->max - h->min, new_w = new_max - new_min;
  const int nb = h->nbuckets, nv = h->nvals;
  int i, k, start, end, step;
  LIBXS_ASSERT(0 < new_w);
  if (new_min < h->min) {
    start = nb - 1; end = -1; step = -1;
  }
  else {
    start = 0; end = nb; step = 1;
  }
  for (i = start; i != end; i += step) {
    if (0 < h->buckets[i]) {
      const double mid = h->min + (i + 0.5) * old_w / nb;
      int ni = (int)((mid - new_min) * nb / new_w);
      if (ni < 0) ni = 0;
      if (ni >= nb) ni = nb - 1;
      if (ni != i) {
        const int nj = ni * nv, oj = i * nv;
        if (0 < h->buckets[ni]) {
          const double ca = h->buckets[i], cb = h->buckets[ni];
          for (k = 0; k < nv; ++k) {
            const libxs_hist_update_t update = h->update[k];
            if (NULL == update || libxs_hist_update_avg == update) {
              h->vals[nj + k] = (cb * h->vals[nj + k] + ca * h->vals[oj + k]) / (ca + cb);
            }
            else if (libxs_hist_update_min == update) {
              if (h->vals[oj + k] < h->vals[nj + k]) h->vals[nj + k] = h->vals[oj + k];
            }
            else if (libxs_hist_update_max == update) {
              if (h->vals[oj + k] > h->vals[nj + k]) h->vals[nj + k] = h->vals[oj + k];
            }
            else {
              h->vals[nj + k] += h->vals[oj + k];
            }
          }
          h->buckets[ni] += h->buckets[i];
        }
        else {
          for (k = 0; k < nv; ++k) h->vals[nj + k] = h->vals[oj + k];
          h->buckets[ni] = h->buckets[i];
        }
        h->buckets[i] = 0;
      }
    }
  }
  h->min = new_min;
  h->max = new_max;
}


/**
 * Fold the queued batch into the buckets and empty the queue. Called when the
 * queue wraps and whenever a query needs a summary of what is pending, so a
 * histogram is exact while the batch fits and approximates only once there is
 * no more space to stay accurate.
 *
 * The axis comes from the first batch; a later one that reaches outside it
 * rebins first, which is the only step that loses information - it relocates a
 * bucket's aggregate by the bucket's midpoint rather than by the samples that
 * formed it. Everything else here is exact.
 */
LIBXS_API_INTERN void internal_libxs_hist_fold(libxs_hist_t* h);
LIBXS_API_INTERN void internal_libxs_hist_fold(libxs_hist_t* h)
{
  if (0 < h->nq) {
    double lo = h->queue[0], hi = h->queue[0], w;
    int i, j, k, m;
    for (i = 1; i < h->nq; ++i) {
      const double v = h->queue[i * h->nvals];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (0 == h->nfold) { /* the first batch defines the axis */
      if (h->nq < h->nbuckets) h->nbuckets = h->nq;
      h->min = lo;
      h->max = hi;
    }
    else if (lo < h->min || hi > h->max) {
      internal_libxs_hist_rebin(h, lo < h->min ? lo : h->min, hi > h->max ? hi : h->max);
    }
    w = h->max - h->min;
    for (m = 0; m < h->nq; ++m) {
      const int mj = m * h->nvals;
      for (i = 1; i <= h->nbuckets; ++i) {
        const double q = h->min + i * w / h->nbuckets;
        if (h->queue[mj] <= q || h->nbuckets == i) {
          const int c = ++h->buckets[i - 1];
          j = (i - 1) * h->nvals;
          for (k = 0; k < h->nvals; ++k) {
            if (1 < c) {
              const libxs_hist_update_t update = h->update[k];
              if (NULL != update) update(h->vals + (j + k), h->queue + (mj + k), c);
              else libxs_hist_update_avg(h->vals + (j + k), h->queue + (mj + k), c);
            }
            else h->vals[j + k] = h->queue[mj + k];
          }
          break;
        }
      }
    }
    /* after the buckets have taken the batch, so the queue may be reordered */
    if (NULL != h->fold) h->fold(h->state, h->nstate, h->queue, h->nvals, h->nq);
    h->nq = 0;
    ++h->nfold;
  }
}


LIBXS_API void libxs_hist_push(libxs_lock_t* lock, libxs_hist_t* hist, const double vals[])
{
  if (NULL != hist) {
    int k;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    for (k = 0; k < hist->nvals; ++k) {
      hist->queue[hist->nq * hist->nvals + k] = vals[k];
      /* accumulated here rather than derived from the buckets: a running total
         is independent of the update function, and rebinning cannot touch it */
      hist->sum[k] += vals[k];
    }
    if (hist->n < INT_MAX) ++hist->n;
    if (hist->nqueue <= ++hist->nq) internal_libxs_hist_fold(hist);
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_hist_query(libxs_lock_t* lock, const libxs_hist_t* hist,
  libxs_hist_info_t* info)
{
  /* C "mutable": the pending batch is folded here (internal mutation) via cast,
     which is safe because a histogram is always heap-allocated */
  libxs_hist_t* const h = (libxs_hist_t*)(uintptr_t)hist;
  LIBXS_ASSERT(NULL != info);
  info->buckets = NULL;
  info->vals = NULL;
  info->sum = NULL;
  info->state = NULL;
  info->range[0] = 0;
  info->range[1] = 0;
  info->nbuckets = 0;
  info->nvals = 0;
  info->nsamples = 0;
  info->nstate = 0;
  if (NULL != h) {
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    /* folding empties the queue, so querying twice reports the same thing
       rather than folding the same samples again */
    internal_libxs_hist_fold(h);
    if (0 < h->n) {
      info->buckets = h->buckets;
      info->vals = h->vals;
      info->sum = h->sum;
      info->state = h->state;
      info->nstate = h->nstate;
      info->range[0] = h->min;
      info->range[1] = h->max;
      info->nbuckets = h->nbuckets;
      info->nvals = h->nvals;
      info->nsamples = h->n;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_hist_query_percentile(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[], double percentile)
{
  libxs_hist_info_t info;
  int i;
  LIBXS_ASSERT(NULL != vals);
  libxs_hist_query(lock, hist, &info);
  if (NULL != info.buckets && 0 < info.nbuckets) {
    int total = 0, cumulative = 0;
    if (0 > percentile) percentile = 0;
    if (1 < percentile) percentile = 1;
    for (i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
    if (0 < total) {
      const double target = percentile * total;
      for (i = 0; i < info.nbuckets; ++i) {
        cumulative += info.buckets[i];
        if (target <= cumulative) {
          const double fraction = (0 < info.buckets[i])
            ? (1.0 - (cumulative - target) / info.buckets[i]) : 0.5;
          /**
           * Interpolate only towards a *populated* neighbour. An empty bucket
           * carries no value at all - the storage is uninitialized until a
           * sample lands in it - so blending towards one produced a reading
           * that belonged to no sample, and with sparse data (clusters far
           * apart, as when transfer sizes are bimodal) it varied with the bucket
           * count alone. Values are per-bucket aggregates rather than samples on
           * a continuum, so smearing them across a gap is not meaningful even
           * when the neighbour happens to be populated by unrelated data.
           */
          const int ia = i * info.nvals;
          const int ib = (fraction < 0.5 && 0 < i && 0 < info.buckets[i - 1])
            ? (i - 1) * info.nvals : ((fraction >= 0.5 && i + 1 < info.nbuckets && 0 < info.buckets[i + 1])
            ? (i + 1) * info.nvals : ia);
          /**
           * From index 0 like every other value, rather than from the bucket's
           * position on the axis. The axis is where the bucket sits, not what
           * landed in it: a distribution with one far outlier - a first launch
           * paying a one-time cost, say - stretches the range so far that the
           * interpolated coordinate names a value no sample took, while the
           * bucket holding almost every sample carries their mean. This slot is
           * the binning key, but it is also aggregated on push like the rest,
           * so the information is there and only had to be used.
           */
          int k = 0;
          const double t = (ia != ib
            ? (fraction < 0.5 ? 0.5 + fraction : fraction - 0.5) : 0);
#if defined(__GNUC__) && !defined(__clang__)
          LIBXS_PRAGMA_DIAG_PUSH()
          LIBXS_PRAGMA_DIAG_OFF("-Warray-bounds")
          LIBXS_PRAGMA_DIAG_OFF("-Warray-bounds=")
#endif
          for (; k < info.nvals; ++k) {
            vals[k] = info.vals[ia + k] + t * (info.vals[ib + k] - info.vals[ia + k]);
          }
#if defined(__GNUC__) && !defined(__clang__)
          LIBXS_PRAGMA_DIAG_POP()
#endif
          break;
        }
      }
    }
  }
}


LIBXS_API void libxs_hist_query_median(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[])
{
  libxs_hist_query_percentile(lock, hist, vals, 0.5);
}


LIBXS_API void libxs_hist_query_mode(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[])
{
  libxs_hist_info_t info;
  LIBXS_ASSERT(NULL != vals);
  libxs_hist_query(lock, hist, &info);
  if (NULL != info.buckets && 0 < info.nbuckets && NULL != info.vals) {
    int best = -1, i;
    for (i = 0; i < info.nbuckets; ++i) {
      if (0 < info.buckets[i] && (0 > best || info.buckets[best] < info.buckets[i])) best = i;
    }
    if (0 <= best) {
      const double w = info.range[1] - info.range[0];
      const int j = best * info.nvals;
      int k = 1;
      vals[0] = info.range[0] + (best + 1) * w / info.nbuckets;
      for (; k < info.nvals; ++k) vals[k] = info.vals[j + k];
    }
  }
}


LIBXS_API void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist, const int prec[],
  const char fmt[], ...)
{
  libxs_hist_info_t info;
  int i = 1, j = 0, k;
  libxs_hist_query(NULL /*lock*/, hist, &info);
  if (NULL != ostream && NULL != info.buckets && 0 < info.nbuckets && NULL != info.vals && 0 < info.nvals) {
    const double w = info.range[1] - info.range[0];
    if (NULL != fmt) {
      va_list args;
      va_start(args, fmt);
#if defined(__clang__)
      LIBXS_PRAGMA_DIAG_PUSH()
      LIBXS_PRAGMA_DIAG_OFF("-Wformat-nonliteral")
#endif
      vfprintf(ostream, fmt, args);
#if defined(__clang__)
      LIBXS_PRAGMA_DIAG_POP()
#endif
      va_end(args);
    }
    for (; i <= info.nbuckets; j = info.nvals * i++) {
      const double q = info.range[0] + i * w / info.nbuckets;
      const int c = info.buckets[i - 1];
      /**
       * Skip empty buckets: a bucket holds no sample and hence no value to
       * report, so printing it states a range that nothing occupied. Bucket
       * numbers stay tied to the range (they are not renumbered), which is what
       * makes a gap in the sequence readable as "nothing fell here".
       */
      if (0 == c) continue;
      if (NULL != prec) {
        if (0 > prec[0]) continue;
        fprintf(ostream, "\t#%i <= %.*f: %i ->", i, prec[0], q, c);
      }
      else fprintf(ostream, "\t#%i <= %f: %i ->", i, q, c);
      for (k = 0; k < info.nvals; ++k) {
        const double value = info.vals[j + k];
        if (NULL != prec && 0 > prec[k]) continue;
        if (NULL != prec) fprintf(ostream, " %.*f", prec[k], value);
        else fprintf(ostream, " %f", value);
      }
      fprintf(ostream, "\n");
    }
  }
}


LIBXS_API void libxs_hist_update_avg(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src && 1 < count);
  *dst += (*src - *dst) / count;
}


LIBXS_API void libxs_hist_update_add(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  LIBXS_UNUSED(count);
  *dst += *src;
}


LIBXS_API void libxs_hist_update_min(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  LIBXS_UNUSED(count);
  if (*src < *dst) *dst = *src;
}


LIBXS_API void libxs_hist_update_max(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  LIBXS_UNUSED(count);
  if (*src > *dst) *dst = *src;
}
