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
#include <string.h>

#define NENTRY 1200
#define NINPUT 16
#define NQUERY 400
#define NWIDE 8


/**
 * One input decides the class and the others are noise. A rank distance
 * weighs all of them alike and drowns the one that matters; a score gives the
 * noise an axis of zero length, which is the case the coordinate exists for.
 * Two classes and not more: with several equally frequent classes in pure bins
 * the leading eigenvalue is degenerate, and which line through that space the
 * solver returns, merging which classes, is not something to assert on.
 */
static void fill(double inputs[], double* output, int i)
{
  unsigned int s = (unsigned int)(i * 2654435761u);
  int j;
  for (j = 0; j < NINPUT; ++j) {
    s = s * 1103515245u + 12345u;
    inputs[j] = (double)((s >> 16) & 0x3f);
  }
  *output = (32 <= inputs[0]) ? 1.0 : 0.0;
}


static int build_model(libxs_predict_t* model, int coordinate)
{
  int i, result = EXIT_SUCCESS;
  libxs_predict_set_decompose(model, LIBXS_PREDICT_RAW);
  libxs_predict_set_coordinate(model, coordinate);
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
    double inputs[NINPUT], output;
    fill(inputs, &output, i);
    result = libxs_predict_push(NULL, model, inputs, &output);
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_build(model, 0, 2, 0.0);
  }
  return result;
}


/* Correct answers on queries drawn past the corpus. */
static int correct(const libxs_predict_t* model)
{
  int i, result = 0;
  for (i = 0; i < NQUERY; ++i) {
    double inputs[NINPUT], output, predicted;
    fill(inputs, &output, NENTRY + i);
    libxs_predict_eval(NULL, model, inputs, &predicted, NULL, 1);
    if (LIBXS_ROUNDX(int, predicted) == LIBXS_ROUNDX(int, output)) ++result;
  }
  return result;
}


static int coordinate(const libxs_predict_t* model)
{
  libxs_predict_query_t info;
  memset(&info, 0, sizeof(info));
  libxs_predict_query(model, &info);
  return info.coordinate;
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


/**
 * Four classes, either decided by the first input alone among noise (wide is
 * zero) or carried by each of NWIDE inputs with some of them blurred. One
 * coordinate per input cannot hold four classes apart, so the first case needs
 * several where the class is and none where it is not, and the second ends up
 * with more coordinates than inputs: the two ways a score reshapes the space.
 */
static void fill4(double inputs[], double* output, int i, int wide)
{
  unsigned int s = (unsigned int)(i * 2654435761u);
  const int y = (int)((s >> 7) & 3);
  int j;
  for (j = 0; j < (0 != wide ? NWIDE : NINPUT); ++j) {
    s = s * 1103515245u + 12345u;
    if (0 != wide) {
      const int blur = (0 == ((s >> 20) & 3)) ? (int)((s >> 22) & 3) : y;
      inputs[j] = (double)(16 * blur + ((s >> 16) & 0xf));
    }
    else inputs[j] = (double)((s >> 16) & 0x3f);
  }
  if (0 == wide) inputs[0] = (double)(16 * y + ((s >> 10) & 0xf));
  *output = (double)y;
}


/* Build, query and reload one four-class model; zero coordinates on failure. */
static int multiclass(int wide, int decompose, double quality, int* ncorrect)
{
  const int m = (0 != wide) ? NWIDE : NINPUT;
  libxs_predict_t* model = libxs_predict_create(m, 1);
  libxs_predict_query_t info;
  int result = 0, i;
  memset(&info, 0, sizeof(info));
  *ncorrect = 0;
  if (NULL != model) {
    int ok = EXIT_SUCCESS;
    libxs_predict_set_decompose(model, decompose);
    libxs_predict_set_coordinate(model, LIBXS_PREDICT_SCORE);
    for (i = 0; i < NENTRY && EXIT_SUCCESS == ok; ++i) {
      double inputs[NINPUT], output;
      fill4(inputs, &output, i, wide);
      ok = libxs_predict_push(NULL, model, inputs, &output);
    }
    if (EXIT_SUCCESS == ok) ok = libxs_predict_build(model, 0, 2, quality);
    if (EXIT_SUCCESS == ok) {
      libxs_predict_t* loaded = reload(model);
      libxs_predict_query(model, &info);
      for (i = 0; i < NQUERY; ++i) {
        double inputs[NINPUT], output, a, b;
        fill4(inputs, &output, NENTRY + i, wide);
        libxs_predict_eval(NULL, model, inputs, &a, NULL, 1);
        if (LIBXS_ROUNDX(int, a) == LIBXS_ROUNDX(int, output)) ++*ncorrect;
        if (NULL != loaded) {
          libxs_predict_eval(NULL, loaded, inputs, &b, NULL, 1);
          if (a != b) *ncorrect = -NQUERY;
        }
      }
      if (NULL != loaded && LIBXS_PREDICT_SCORE == info.coordinate) {
        result = info.ncoord;
      }
      libxs_predict_destroy(loaded);
    }
    libxs_predict_destroy(model);
  }
  return result;
}


int main(void)
{
  libxs_predict_t* rank = libxs_predict_create(NINPUT, 1);
  libxs_predict_t* score = libxs_predict_create(NINPUT, 1);
  libxs_predict_t* chosen = libxs_predict_create(NINPUT, 1);
  int result = (NULL != rank && NULL != score && NULL != chosen)
    ? EXIT_SUCCESS : EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = build_model(rank, LIBXS_PREDICT_RANK);
  if (EXIT_SUCCESS == result) result = build_model(score, LIBXS_PREDICT_SCORE);
  if (EXIT_SUCCESS == result) {
    result = build_model(chosen, LIBXS_PREDICT_AUTO_COORD);
  }
  if (EXIT_SUCCESS == result && (LIBXS_PREDICT_RANK != coordinate(rank)
    || LIBXS_PREDICT_SCORE != coordinate(score)))
  {
    fprintf(stderr, "a requested coordinate is not the one in force\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    const int nrank = correct(rank), nscore = correct(score);
    if (nscore <= nrank + NQUERY / 10) {
      fprintf(stderr, "the score did not beat the rank (%i vs %i of %i)\n",
        nscore, nrank, NQUERY);
      result = EXIT_FAILURE;
    }
  }
  /* noise that drowns the rank is the case the trial must not miss */
  if (EXIT_SUCCESS == result && LIBXS_PREDICT_SCORE != coordinate(chosen)) {
    fprintf(stderr, "the trial did not choose the score\n");
    result = EXIT_FAILURE;
  }
  /* the table survives a round trip, and the corpus is declined with it */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* loaded = reload(score);
    libxs_predict_query_t info;
    memset(&info, 0, sizeof(info));
    if (NULL != loaded) libxs_predict_query(loaded, &info);
    if (NULL == loaded || LIBXS_PREDICT_SCORE != info.coordinate
      || 0 != info.corpus || correct(loaded) != correct(score))
    {
      fprintf(stderr, "a model with a score did not reload as built\n");
      result = EXIT_FAILURE;
    }
    libxs_predict_destroy(loaded);
  }
  /* where a score cannot apply the rank is used, whatever was requested */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* two = libxs_predict_create(NINPUT, 2);
    int i;
    if (NULL != two) {
      libxs_predict_set_coordinate(two, LIBXS_PREDICT_SCORE);
      for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
        double inputs[NINPUT], outputs[2];
        fill(inputs, outputs, i);
        outputs[1] = outputs[0];
        result = libxs_predict_push(NULL, two, inputs, outputs);
      }
      if (EXIT_SUCCESS == result) result = libxs_predict_build(two, 0, 2, 0.0);
      if (EXIT_SUCCESS == result && LIBXS_PREDICT_RANK != coordinate(two)) {
        fprintf(stderr, "a score was fitted for two outputs\n");
        result = EXIT_FAILURE;
      }
      libxs_predict_destroy(two);
    }
    else result = EXIT_FAILURE;
  }
  /* four classes behind one input: noise drops out, the input spans several */
  if (EXIT_SUCCESS == result) {
    int ncorrect = 0;
    const int nc = multiclass(0, LIBXS_PREDICT_RAW, 0.0, &ncorrect);
    if (nc < 2 || NINPUT <= nc || ncorrect < NQUERY * 9 / 10) {
      fprintf(stderr, "four classes behind one input: %i coordinates, %i of %i"
        " correct\n", nc, ncorrect, NQUERY);
      result = EXIT_FAILURE;
    }
  }
  /* four classes in every input: more coordinates than inputs, every path */
  if (EXIT_SUCCESS == result) {
    const int modes[] = { LIBXS_PREDICT_RAW, LIBXS_PREDICT_HKNN };
    int mi;
    for (mi = 0; mi < 2 && EXIT_SUCCESS == result; ++mi) {
      int ncorrect = 0, ncompressed = 0;
      const int nc = multiclass(1, modes[mi], 0.0, &ncorrect);
      const int ncc = multiclass(1, modes[mi], 0.9, &ncompressed);
      if (nc <= NWIDE || ncc <= NWIDE || ncorrect < NQUERY / 2
        || ncompressed < NQUERY / 2)
      {
        fprintf(stderr, "four classes in every input (mode %i): %i/%i"
          " coordinates, %i/%i of %i correct\n", modes[mi], nc, ncc,
          ncorrect, ncompressed, NQUERY);
        result = EXIT_FAILURE;
      }
    }
  }
  libxs_predict_destroy(chosen);
  libxs_predict_destroy(score);
  libxs_predict_destroy(rank);
  if (EXIT_SUCCESS == result) fprintf(stdout, "OK\n");
  return result;
}
