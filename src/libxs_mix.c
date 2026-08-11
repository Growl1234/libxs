/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_mix.h>

#include <stdlib.h>
#include <math.h>


LIBXS_API int libxs_mix_create(libxs_mix_t* mix, int nslot,
  double rate, double share, double relmin)
{
  int result = EXIT_FAILURE;
  if (NULL != mix && 0 < nslot) {
    mix->weight = (double*)malloc((size_t)nslot * sizeof(double));
    if (NULL != mix->weight) {
      mix->nslot = nslot;
      mix->rate = rate;
      mix->share = share;
      mix->relmin = relmin;
      libxs_mix_reset(mix, NULL);
      result = EXIT_SUCCESS;
    }
    else mix->nslot = 0;
  }
  return result;
}


LIBXS_API void libxs_mix_destroy(libxs_mix_t* mix)
{
  if (NULL != mix) {
    free(mix->weight);
    mix->weight = NULL;
    mix->nslot = 0;
  }
}


LIBXS_API void libxs_mix_reset(libxs_mix_t* mix, const int active[])
{
  if (NULL != mix && NULL != mix->weight && 0 < mix->nslot) {
    int i, n = 0;
    for (i = 0; i < mix->nslot; ++i) {
      if (NULL == active || 0 != active[i]) ++n;
    }
    for (i = 0; i < mix->nslot; ++i) {
      mix->weight[i] = (0 < n && (NULL == active || 0 != active[i]))
        ? (1.0 / (double)n) : 0.0;
    }
  }
}


/**
 * The pool renormalizes over the slots that both hold weight and spoke. A slot
 * that abstains is excluded from the numerator AND the denominator, so it does
 * not dilute the mixture toward zero -- which it would if its zero contribution
 * were divided by the full weight mass.
 */
LIBXS_API double libxs_mix_pool(const libxs_mix_t* mix, const double prob[],
  const int active[])
{
  double pooled = 0.0;
  if (NULL != mix && NULL != mix->weight && NULL != prob) {
    double wtotal = 0.0;
    int i;
    for (i = 0; i < mix->nslot; ++i) {
      if (mix->weight[i] > 0.0 && (NULL == active || 0 != active[i])) {
        pooled += mix->weight[i] * prob[i];
        wtotal += mix->weight[i];
      }
    }
    if (wtotal > 0.0) pooled /= wtotal;
  }
  return pooled;
}


LIBXS_API void libxs_mix_update(libxs_mix_t* mix, const double prob[],
  const int active[], double mixture)
{
  if (NULL != mix && NULL != mix->weight && NULL != prob) {
    double total = 0.0;
    int i, nactive = 0;
    for (i = 0; i < mix->nslot; ++i) {
      if (mix->weight[i] > 0.0 && (NULL == active || 0 != active[i])) {
        double relative = (mixture > 0.0) ? (prob[i] / mixture) : 1.0;
        if (!(relative > 0.0)) relative = mix->relmin;
        mix->weight[i] *= pow(relative, mix->rate);
        total += mix->weight[i];
        ++nactive;
      }
    }
    /**
     * Renormalize and share over the slots that participated. Spending share
     * mass on a slot that never produces an opinion would take probability from
     * the experts that do; and a bank sized for the widest configuration holds
     * such slots at zero on purpose.
     */
    if (total > 0.0 && 0 < nactive) {
      const double uniform = 1.0 / (double)nactive;
      for (i = 0; i < mix->nslot; ++i) {
        if (mix->weight[i] > 0.0 && (NULL == active || 0 != active[i])) {
          mix->weight[i] = (1.0 - mix->share) * mix->weight[i] / total
            + mix->share * uniform;
        }
      }
    }
  }
}


LIBXS_API double libxs_mix_observe(libxs_mix_t* mix, const double prob[],
  const int active[])
{
  const double pooled = libxs_mix_pool(mix, prob, active);
  libxs_mix_update(mix, prob, active, pooled);
  return pooled;
}
