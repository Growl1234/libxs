/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_mem.h>

#include <stdio.h>

#define NENTRY 600
#define NINPUT 6


/**
 * Two inputs carry the label and four are noise, which is the situation the
 * feature-selecting modes exist for.  What is asserted is not which mode wins
 * that corpus - that is a measurement, not a contract - but that the sentinel
 * resolves to a real mode, that the resolved mode is reported, and that asking
 * for the choice is never worse than an arbitrary fixed answer would have been.
 */
static void fill(double inputs[], double* out, int i)
{
  const int a = (i / 20) % 5;
  const int b = i % 4;
  unsigned int s = (unsigned int)(i * 2654435761u);
  int j;
  inputs[0] = (double)a;
  inputs[1] = (double)b;
  for (j = 2; j < NINPUT; ++j) {
    s = s * 1103515245u + 12345u;
    inputs[j] = (double)((s >> 16) & 0xff);
  }
  *out = (double)(a * 4 + b);
}


static int build_model(libxs_predict_t* model, int ntrain)
{
  int i, result = EXIT_SUCCESS;
  for (i = 0; i < ntrain && EXIT_SUCCESS == result; ++i) {
    double inputs[NINPUT], out;
    fill(inputs, &out, i);
    result = libxs_predict_push(NULL, model, inputs, &out);
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_build(model, 0, 2, 0.0);
  }
  return result;
}


static double accuracy(const libxs_predict_t* model, int ntrain)
{
  int i, n = 0, hit = 0;
  for (i = ntrain; i < NENTRY; ++i) {
    double inputs[NINPUT], out, predicted = 0;
    fill(inputs, &out, i);
    libxs_predict_eval(NULL, model, inputs, &predicted, NULL, 1);
    if (LIBXS_ROUNDX(int, predicted) == LIBXS_ROUNDX(int, out)) ++hit;
    ++n;
  }
  return (0 < n) ? ((double)hit / n) : 0.0;
}


int main(void)
{
  const int ntrain = (NENTRY * 4) / 5;
  libxs_predict_t* chosen = libxs_predict_create(NINPUT, 1);
  libxs_predict_t* fixed = libxs_predict_create(NINPUT, 1);
  double acc_chosen = 0, acc_fixed = 0;
  int selected = -1;
  int result = (NULL != chosen && NULL != fixed) ? EXIT_SUCCESS : EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(chosen, LIBXS_PREDICT_AUTO_DECOMPOSE);
    result = build_model(chosen, ntrain);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_query_t info;
    LIBXS_MEMZERO(&info);
    libxs_predict_query(chosen, &info);
    selected = info.decompose;
    /* the sentinel is resolved by the build and never survives it */
    if (LIBXS_PREDICT_RAW > selected || LIBXS_PREDICT_HKNN < selected) {
      fprintf(stderr, "the sentinel resolved to %i\n", selected);
      result = EXIT_FAILURE;
    }
    acc_chosen = accuracy(chosen, ntrain);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(fixed, LIBXS_PREDICT_PCA);
    result = build_model(fixed, ntrain);
    if (EXIT_SUCCESS == result) acc_fixed = accuracy(fixed, ntrain);
  }
  /**
   * PCA rotates a space where four of six coordinates are noise, so it is the
   * arbitrary answer this corpus punishes.  A selector that cannot beat the mode
   * it was built to avoid is not doing anything.
   */
  if (EXIT_SUCCESS == result && acc_chosen < acc_fixed) {
    fprintf(stderr, "the selected mode (%i) scored %.3f against %.3f fixed\n",
      selected, acc_chosen, acc_fixed);
    result = EXIT_FAILURE;
  }
  /* a caller who never asks is left where they were */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* plain = libxs_predict_create(NINPUT, 1);
    if (NULL != plain) {
      if (EXIT_SUCCESS == build_model(plain, ntrain)) {
        libxs_predict_query_t info;
        LIBXS_MEMZERO(&info);
        libxs_predict_query(plain, &info);
        if (LIBXS_PREDICT_RAW != info.decompose) {
          fprintf(stderr, "an unasked model reported mode %i\n", info.decompose);
          result = EXIT_FAILURE;
        }
      }
      else result = EXIT_FAILURE;
      libxs_predict_destroy(plain);
    }
    else result = EXIT_FAILURE;
  }
  libxs_predict_destroy(fixed);
  libxs_predict_destroy(chosen);
  if (EXIT_SUCCESS == result) {
    fprintf(stdout, "OK (selected mode %i: %.3f against %.3f fixed)\n",
      selected, acc_chosen, acc_fixed);
  }
  return result;
}
