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
#include <math.h>
#include <string.h>

#define NENTRY 400
#define NINPUT 2
#define NOUTPUT 2
#define NQUERY 200
#define KNN 8


/**
 * Continuous inputs, so that no two neighbours tie, and two labels whose
 * boundaries cross any partition of the inputs: a query near a cluster's edge
 * then has neighbours beyond it that the cluster's own vote does not read.
 */
static void fill(double inputs[], double outputs[], int i)
{
  unsigned int s = (unsigned int)(i * 2654435761u) + 1u;
  int j;
  for (j = 0; j < NINPUT; ++j) {
    s = s * 1103515245u + 12345u;
    inputs[j] = (double)((s >> 8) & 0xffff) / 65536.0;
  }
  outputs[0] = (inputs[0] + 0.25 * inputs[1] > 0.6) ? 1.0 : 0.0;
  outputs[1] = (double)((int)(inputs[0] * 4.0 + inputs[1] * 3.0) % 3);
}


static libxs_predict_t* build_model(int nclusters, int reach)
{
  libxs_predict_t* result = libxs_predict_create(NINPUT, NOUTPUT);
  int i, ok = (NULL != result) ? EXIT_SUCCESS : EXIT_FAILURE;
  if (EXIT_SUCCESS == ok) {
    libxs_predict_set_decompose(result, LIBXS_PREDICT_RAW);
    libxs_predict_set_neighbors(result, KNN);
    libxs_predict_set_reach(result, reach);
  }
  for (i = 0; i < NENTRY && EXIT_SUCCESS == ok; ++i) {
    double inputs[NINPUT], outputs[NOUTPUT];
    fill(inputs, outputs, i);
    ok = libxs_predict_push(NULL, result, inputs, outputs);
  }
  if (EXIT_SUCCESS == ok) ok = libxs_predict_build(result, nclusters, 1, 0.0);
  if (EXIT_SUCCESS != ok) {
    libxs_predict_destroy(result);
    result = NULL;
  }
  return result;
}


/** How many off-corpus queries the two models answer differently. */
static int differ(const libxs_predict_t* a, const libxs_predict_t* b)
{
  int i, j, result = 0;
  for (i = 0; i < NQUERY; ++i) {
    double inputs[NINPUT], outputs[NOUTPUT];
    double pa[NOUTPUT], pb[NOUTPUT];
    int same = 1;
    fill(inputs, outputs, NENTRY + i);
    libxs_predict_eval(NULL, a, inputs, pa, NULL, 1);
    libxs_predict_eval(NULL, b, inputs, pb, NULL, 1);
    for (j = 0; j < NOUTPUT; ++j) {
      if (pa[j] != pb[j]) same = 0;
    }
    if (0 == same) ++result;
  }
  return result;
}


/**
 * How many off-corpus queries the two models vote on with different shares.
 * Without a confidence floor the share is the vote's alone, which reads the k
 * neighbours and nothing of the cluster that routed the query: two models that
 * gather the same neighbours agree on it whatever their partition.
 */
static int differ_share(const libxs_predict_t* a, const libxs_predict_t* b)
{
  int i, j, result = 0;
  for (i = 0; i < NQUERY; ++i) {
    double inputs[NINPUT], outputs[NOUTPUT], pa[NOUTPUT], pb[NOUTPUT];
    double ca[NOUTPUT] = { 0 };
    libxs_predict_info_t info;
    int same = 1;
    fill(inputs, outputs, NENTRY + i);
    memset(&info, 0, sizeof(info));
    libxs_predict_eval(NULL, a, inputs, pa, &info, 1);
    for (j = 0; j < NOUTPUT && NULL != info.confidence; ++j) ca[j] = info.confidence[j];
    memset(&info, 0, sizeof(info));
    libxs_predict_eval(NULL, b, inputs, pb, &info, 1);
    for (j = 0; j < NOUTPUT; ++j) {
      if (NULL == info.confidence || 1E-12 < fabs(ca[j] - info.confidence[j])) same = 0;
    }
    if (0 == same) ++result;
  }
  return result;
}


/** A saved model reloaded, or NULL. */
static libxs_predict_t* reload(const libxs_predict_t* model)
{
  libxs_predict_t* result = NULL;
  size_t size = 0;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && 0 < size) {
    unsigned char* buffer = (unsigned char*)malloc(size);
    if (NULL != buffer) {
      size_t written = size;
      if (EXIT_SUCCESS == libxs_predict_save(model, buffer, &written)) {
        result = libxs_predict_load(buffer, written);
      }
      free(buffer);
    }
  }
  return result;
}


int main(void)
{
  libxs_predict_t* one = build_model(1, LIBXS_PREDICT_REACH_NONE);
  libxs_predict_t* own = build_model(4, LIBXS_PREDICT_REACH_NONE);
  libxs_predict_t* all = build_model(4, 3);
  libxs_predict_t* tried = build_model(4, -3);
  libxs_predict_t* deflt = build_model(4, 0);
  int result = (NULL != one && NULL != own && NULL != all && NULL != tried
    && NULL != deflt) ? EXIT_SUCCESS : EXIT_FAILURE;
  /* the reach changes what the vote reads, or it never arrived */
  if (EXIT_SUCCESS == result && 0 == differ(own, all)) {
    fprintf(stderr, "a reach changed no answer\n");
    result = EXIT_FAILURE;
  }
  /**
   * Reaching every other cluster gathers the corpus's k nearest, however early
   * the bound stops the scan. The answers are not compared with one cluster's:
   * after the vote, a wide neighbourhood is pulled toward its routed cluster's
   * mean, which one cluster does not share.
   */
  if (EXIT_SUCCESS == result && 0 != differ_share(one, all)) {
    fprintf(stderr, "a full reach voted on other neighbours than one cluster (%i of %i)\n",
      differ_share(one, all), NQUERY);
    result = EXIT_FAILURE;
  }
  /* a trial, named or the default, settles on one of the two it compared */
  if (EXIT_SUCCESS == result && 0 != differ(own, tried) && 0 != differ(all, tried)) {
    fprintf(stderr, "a reach decided by trial answered as neither candidate\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 0 != differ(own, deflt) && 0 != differ(all, deflt)) {
    fprintf(stderr, "the default reach answered as neither candidate\n");
    result = EXIT_FAILURE;
  }
  /* the reach in effect is saved, and a loaded model takes another once set on it */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* const a = reload(all);
    libxs_predict_t* const t = reload(tried);
    libxs_predict_t* const o = reload(own);
    if (NULL == a || NULL == t || NULL == o) {
      fprintf(stderr, "a model failed to reload\n");
      result = EXIT_FAILURE;
    }
    else if (0 != differ(all, a) || 0 != differ(tried, t) || 0 != differ(own, o)) {
      fprintf(stderr, "a reloaded model gathers differently\n");
      result = EXIT_FAILURE;
    }
    else {
      libxs_predict_set_reach(o, 3);
      if (0 != differ(all, o)) {
        fprintf(stderr, "a reach set on a loaded model did not apply\n");
        result = EXIT_FAILURE;
      }
    }
    libxs_predict_destroy(o);
    libxs_predict_destroy(t);
    libxs_predict_destroy(a);
  }
  libxs_predict_destroy(deflt);
  libxs_predict_destroy(tried);
  libxs_predict_destroy(all);
  libxs_predict_destroy(own);
  libxs_predict_destroy(one);
  if (EXIT_SUCCESS == result) fprintf(stdout, "OK\n");
  return result;
}
