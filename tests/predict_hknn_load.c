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

/* past 65536, where an entry's position no longer fits 16 bits */
#define NENTRY 66000
#define NINPUT 3
/* two, since only per-output partitions make the hKNN file format */
#define NOUTPUT 2
#define NQUERY 500


/**
 * Labels of the inputs, so a neighbourhood has something to agree on and the
 * consistency penalty something to discount where it does not.
 */
static void fill(double inputs[], double outputs[], int i)
{
  unsigned int s = (unsigned int)(i * 2654435761u);
  int j;
  for (j = 0; j < NINPUT; ++j) {
    s = s * 1103515245u + 12345u;
    inputs[j] = (double)((s >> 12) & 0x3ff);
  }
  outputs[0] = (double)((int)(inputs[0] + inputs[1]) / 256 % 4);
  outputs[1] = (double)((int)inputs[2] / 128 % 3);
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
  libxs_predict_t* built = libxs_predict_create(NINPUT, NOUTPUT);
  libxs_predict_t* loaded = NULL;
  int result = (NULL != built) ? EXIT_SUCCESS : EXIT_FAILURE, i;
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(built, LIBXS_PREDICT_HKNN);
    libxs_predict_set_consistency(built, 0.9);
    for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
      double inputs[NINPUT], outputs[NOUTPUT];
      fill(inputs, outputs, i);
      result = libxs_predict_push(NULL, built, inputs, outputs);
    }
  }
  if (EXIT_SUCCESS == result) result = libxs_predict_build(built, 0, 1, 0.0);
  if (EXIT_SUCCESS == result) {
    loaded = reload(built);
    if (NULL == loaded) {
      fprintf(stderr, "an hKNN model did not reload\n");
      result = EXIT_FAILURE;
    }
  }
  /* the corpus comes back, so the round trip has entries to invert through */
  if (EXIT_SUCCESS == result) {
    libxs_predict_query_t a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    libxs_predict_query(built, &a);
    libxs_predict_query(loaded, &b);
    if (a.nentries != b.nentries || 0 == b.corpus) {
      fprintf(stderr, "reloaded: %i entries (built %i), corpus %i\n",
        b.nentries, a.nentries, b.corpus);
      result = EXIT_FAILURE;
    }
  }
  /* an entry past 65535 is where a 16-bit position wrapped onto another one */
  if (EXIT_SUCCESS == result) {
    for (i = NENTRY - 400; i < NENTRY && EXIT_SUCCESS == result; ++i) {
      double want[NINPUT], got[NINPUT], outputs[NOUTPUT];
      int j;
      fill(want, outputs, i);
      libxs_predict_get(loaded, i, got, NULL);
      for (j = 0; j < NINPUT; ++j) {
        if (LIBXS_FABS(got[j] - want[j]) > 1e-6 * (1.0 + LIBXS_FABS(want[j]))) {
          fprintf(stderr, "entry %i input %i: %g, stored %g\n",
            i, j, got[j], want[j]);
          result = EXIT_FAILURE;
          break;
        }
      }
    }
  }
  /* the penalty reaches the reloaded model, so both answer the same */
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < NQUERY && EXIT_SUCCESS == result; ++i) {
      double inputs[NINPUT], outputs[NOUTPUT], pa[NOUTPUT], pb[NOUTPUT];
      libxs_predict_info_t ia, ib;
      int j;
      fill(inputs, outputs, NENTRY + i);
      libxs_predict_eval(NULL, built, inputs, pa, &ia, 1);
      libxs_predict_eval(NULL, loaded, inputs, pb, &ib, 1);
      for (j = 0; j < NOUTPUT && EXIT_SUCCESS == result; ++j) {
        if (pa[j] != pb[j] || NULL == ia.confidence || NULL == ib.confidence
          || ia.confidence[j] != ib.confidence[j])
        {
          fprintf(stderr, "query %i output %i: built %g, reloaded %g\n",
            i, j, pa[j], pb[j]);
          result = EXIT_FAILURE;
        }
      }
    }
  }
  libxs_predict_destroy(loaded);
  libxs_predict_destroy(built);
  if (EXIT_SUCCESS == result) fprintf(stdout, "OK\n");
  return result;
}
