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
#include <libxs/libxs_malloc.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define NENTRY 800
#define NFEAT 4
#define NCLASS 3
#define DIRT 0xA5


/**
 * Same corpus every time, and one a forest has something to split on: the label
 * follows a threshold on two of the inputs, and the remaining inputs carry a
 * pattern that correlates with nothing.
 */
static void fill(double input[], double* out, int i)
{
  input[0] = (double)(i % 97);
  input[1] = (double)((i * 7) % 31);
  input[2] = (double)((i * 13) % 11);
  input[3] = (double)(i % 5);
  *out = (double)(((input[0] > 48.0) ? 1 : 0)
    + ((input[1] > 15.0) ? 1 : 0));
  if (*out >= NCLASS) *out = NCLASS - 1;
}


static int build_model(libxs_predict_t* model, int ntrees, int collective)
{
  int i, result = EXIT_SUCCESS;
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
    double input[NFEAT], out;
    fill(input, &out, i);
    result = libxs_predict_push(NULL, model, input, &out);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(model, LIBXS_PREDICT_RF);
    if (0 < ntrees) libxs_predict_set_forest(model, ntrees, 0);
#if defined(_OPENMP)
    if (0 != collective) {
#     pragma omp parallel num_threads(4)
      { const int build = libxs_predict_build_task(model, 0, 1, 0.0,
          omp_get_thread_num(), omp_get_num_threads());
        if (0 == omp_get_thread_num()) result = build;
      }
    }
    else
#else
    LIBXS_UNUSED(collective);
#endif
    result = libxs_predict_build(model, 0, 1, 0.0);
  }
  return result;
}


/** Serializes a freshly built model; the caller owns the buffer. */
static int save_model(void** buffer, size_t* size, double prediction[])
{
  libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
  double batch_input[NENTRY * NFEAT], batch_prediction[NENTRY];
  int result = EXIT_FAILURE;
  *buffer = NULL;
  *size = 0;
  if (NULL != model) {
    if (EXIT_SUCCESS == build_model(model, 0, 0)
      && EXIT_SUCCESS == libxs_predict_save(model, NULL, size) && 0 < *size)
    {
      int i;
      result = EXIT_SUCCESS;
      for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
        double input[NFEAT], out;
        fill(input, &out, i);
        memcpy(batch_input + (size_t)i * NFEAT,
          input, (size_t)NFEAT * sizeof(double));
        libxs_predict_eval(NULL, model, input, prediction + i, NULL, 0);
      }
      libxs_predict_eval_batch(
        model, batch_input, batch_prediction, NENTRY, 0);
      for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
        if (batch_prediction[i] != prediction[i]) {
          fprintf(stderr, "batch prediction %g differs from %g at entry %d\n",
            batch_prediction[i], prediction[i], i);
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        libxs_predict_eval_batch(
          model, batch_input, batch_prediction, NENTRY - 3, 0);
        for (i = 0; i < NENTRY - 3 && EXIT_SUCCESS == result; ++i) {
          if (batch_prediction[i] != prediction[i]) {
            fprintf(stderr, "batch tail %g differs from %g at entry %d\n",
              batch_prediction[i], prediction[i], i);
            result = EXIT_FAILURE;
          }
        }
      }
      if (EXIT_SUCCESS == result) {
        *buffer = malloc(*size);
        if (NULL != *buffer) {
          result = libxs_predict_save(model, *buffer, size);
        }
        else result = EXIT_FAILURE;
      }
    }
    libxs_predict_destroy(model);
  }
  return result;
}


static int check_team_build_case(int ntrees)
{
  libxs_predict_t* serial = libxs_predict_create(NFEAT, 1);
  libxs_predict_t* team = libxs_predict_create(NFEAT, 1);
  void* a = NULL;
  void* b = NULL;
  size_t na = 0, nb = 0;
  int result = (NULL != serial && NULL != team) ? EXIT_SUCCESS : EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = build_model(serial, ntrees, 0);
  if (EXIT_SUCCESS == result) result = build_model(team, ntrees, 1);
  if (EXIT_SUCCESS == result) result = libxs_predict_save(serial, NULL, &na);
  if (EXIT_SUCCESS == result) result = libxs_predict_save(team, NULL, &nb);
  if (EXIT_SUCCESS == result && na == nb) {
    a = malloc(na);
    b = malloc(nb);
    if (NULL != a && NULL != b) {
      result = libxs_predict_save(serial, a, &na);
      if (EXIT_SUCCESS == result) result = libxs_predict_save(team, b, &nb);
      if (EXIT_SUCCESS == result && 0 != memcmp(a, b, na)) {
        fprintf(stderr, "serial and team forests differ\n");
        result = EXIT_FAILURE;
      }
    }
    else result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == result) result = EXIT_FAILURE;
  free(b);
  free(a);
  libxs_predict_destroy(team);
  libxs_predict_destroy(serial);
  return result;
}


static int check_team_build(void)
{
  int result = check_team_build_case(2);
  if (EXIT_SUCCESS == result) result = check_team_build_case(8);
  return result;
}


static int check_roundtrip(const void* buffer, size_t size,
  const double prediction[])
{
  libxs_predict_t* model = libxs_predict_load(buffer, size);
  int result = (NULL != model) ? EXIT_SUCCESS : EXIT_FAILURE;
  int i;
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
    double input[NFEAT], out, loaded = 0;
    fill(input, &out, i);
    libxs_predict_eval(NULL, model, input, &loaded, NULL, 0);
    if (loaded != prediction[i]) {
      fprintf(stderr, "loaded prediction %g differs from %g at entry %d\n",
        loaded, prediction[i], i);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    size_t written = 0;
    void* saved;
    result = libxs_predict_save(model, NULL, &written);
    saved = (EXIT_SUCCESS == result && size == written)
      ? malloc(written) : NULL;
    if (NULL != saved) result = libxs_predict_save(model, saved, &written);
    else result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result && 0 != memcmp(buffer, saved, size)) {
      fprintf(stderr, "loaded model did not save byte-identically\n");
      result = EXIT_FAILURE;
    }
    free(saved);
  }
  libxs_predict_destroy(model);
  return result;
}


/**
 * Leaves the scratch pool holding recognizable bytes, so that the second build
 * meets a recycled buffer where the first met a fresh one. Without it the two
 * builds can see identical scratch and a field that is never initialized reads
 * the same both times, which is the case that hides the defect.
 */
static void dirty_scratch(void)
{
  static const size_t sizes[] = { 4096, 65536, 1048576 };
  void* held[3];
  int i;
  for (i = 0; i < 3; ++i) {
    held[i] = libxs_malloc(NULL, sizes[i], 0);
    if (NULL != held[i]) memset(held[i], DIRT, sizes[i]);
  }
  for (i = 0; i < 3; ++i) libxs_free(held[i]);
}


/**
 * A saved model must depend on the corpus and the settings, and on nothing else.
 * It is checked by BYTES rather than by a round trip: a field that is written
 * but never read survives a round trip unchanged and reports nothing, while an
 * uninitialized one differs here as soon as the allocator hands out a buffer
 * that has been used before.
 */
int main(void)
{
  void* first = NULL;
  void* second = NULL;
  size_t nfirst = 0, nsecond = 0;
  double prediction[NENTRY], repeated[NENTRY];
  int result = check_team_build();
  if (EXIT_SUCCESS != save_model(&first, &nfirst, prediction)) {
    fprintf(stderr, "the model could not be built or saved\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    dirty_scratch();
    if (EXIT_SUCCESS != save_model(&second, &nsecond, repeated)) {
      fprintf(stderr, "the model could not be rebuilt or saved\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && nfirst != nsecond) {
    fprintf(stderr, "the same corpus saved %llu bytes and then %llu\n",
      (unsigned long long)nfirst, (unsigned long long)nsecond);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 0 != memcmp(first, second, nfirst)) {
    const unsigned char* a = (const unsigned char*)first;
    const unsigned char* b = (const unsigned char*)second;
    size_t i, ndiff = 0, at = nfirst;
    for (i = 0; i < nfirst; ++i) {
      if (a[i] != b[i]) {
        if (0 == ndiff) at = i;
        ++ndiff;
      }
    }
    fprintf(stderr, "the same corpus saved differently: %llu of %llu bytes,"
      " first at %llu (%02x vs %02x)\n", (unsigned long long)ndiff,
      (unsigned long long)nfirst, (unsigned long long)at, a[at], b[at]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = check_roundtrip(first, nfirst, prediction);
  }
  free(second);
  free(first);
  return result;
}
