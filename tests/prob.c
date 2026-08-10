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
#include <stdio.h>

#define NCLASS 8
#define NCTX 6
#define NREP 40


/**
 * A stream whose escape bank has something to learn: the class is a
 * deterministic function of the context, so local evidence beats the fallback
 * prior and the weights must move away from uniform to score well.
 */
static int stream_fill(double inputs[], double* target, int step)
{
  const int ctx = step % NCTX;
  inputs[0] = (double)ctx;
  inputs[1] = (double)(ctx % 3);
  *target = (double)(ctx % NCLASS);
  return ctx;
}


static libxs_predict_t* build_model(void)
{
  libxs_predict_t* model = libxs_predict_create(2, 1);
  if (NULL != model) {
    int step;
    libxs_predict_set_mode(model, LIBXS_PREDICT_CLASSIFY);
    for (step = 0; step < NCTX * NREP; ++step) {
      double in[2], out[1];
      stream_fill(in, out, step);
      if (EXIT_SUCCESS != libxs_predict_push(NULL, model, in, out)) {
        libxs_predict_destroy(model);
        model = NULL;
        break;
      }
    }
    if (NULL != model && EXIT_SUCCESS != libxs_predict_build(model, 1, 0, 0.0)) {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  return model;
}


/**
 * Score the stream frozen (context == NULL) and return the total code length.
 * Frozen scoring reads the model's weights and writes nothing, so this is a
 * pure function of the model -- which is the property under test.
 */
static double frozen_bits(const libxs_predict_t* model, int nstep)
{
  double bits = 0.0;
  int step;
  for (step = 0; step < nstep; ++step) {
    double in[2], target, p = 0.0;
    stream_fill(in, &target, step);
    libxs_predict_prob(NULL, model, NULL, in, &target, &p, NULL, NCLASS, 1);
    bits -= (p > 0.0) ? (log(p) / log(2.0)) : 0.0;
  }
  return bits;
}


/**
 * Frozen scoring must be independent of the order positions are visited in.
 * That is the whole reason the mode exists, so it is asserted rather than
 * assumed: an adaptive call reached by mistake would show up here as a
 * discrepancy between the forward and reverse sweeps.
 */
static int check_order_independent(const libxs_predict_t* model)
{
  int result = EXIT_SUCCESS;
  double fwd = 0.0, rev = 0.0;
  int step;
  for (step = 0; step < NCTX * 4; ++step) {
    double in[2], target, p = 0.0;
    stream_fill(in, &target, step);
    libxs_predict_prob(NULL, model, NULL, in, &target, &p, NULL, NCLASS, 1);
    fwd += p;
  }
  for (step = NCTX * 4 - 1; step >= 0; --step) {
    double in[2], target, p = 0.0;
    stream_fill(in, &target, step);
    libxs_predict_prob(NULL, model, NULL, in, &target, &p, NULL, NCLASS, 1);
    rev += p;
  }
  if (fabs(fwd - rev) > 1e-12) {
    fprintf(stderr, "frozen scoring depends on order: %.17g vs %.17g\n",
      fwd, rev);
    result = EXIT_FAILURE;
  }
  return result;
}


/**
 * The defect this test exists for: adaptation writes only into the context, so
 * without a commit the model keeps its uniform prior and frozen scoring is
 * frozen at that prior -- not at what the stream learned. The round-trip
 * through save/load passes either way, because uniform weights serialize
 * perfectly well, so only a BEHAVIOURAL check catches it.
 */
static int check_commit_moves_weights(libxs_predict_t* model)
{
  int result = EXIT_FAILURE;
  const double before = frozen_bits(model, NCTX * 4);
  void* context = libxs_predict_prob_create(model);
  if (NULL != context) {
    double after_nocommit, after_commit;
    int step;
    for (step = 0; step < NCTX * NREP; ++step) {
      double in[2], target;
      stream_fill(in, &target, step);
      libxs_predict_prob_observe(NULL, model, context, in, 0, &target,
        NULL, NULL, 0, NULL, NULL, NCLASS, 1);
    }
    /* adaptation must not have touched the model */
    after_nocommit = frozen_bits(model, NCTX * 4);
    if (fabs(after_nocommit - before) > 1e-12) {
      fprintf(stderr, "adaptive scoring wrote through to the model: "
        "%.17g -> %.17g\n", before, after_nocommit);
    }
    else if (EXIT_SUCCESS != libxs_predict_prob_commit(model, context)) {
      fprintf(stderr, "commit of a valid context failed\n");
    }
    else {
      after_commit = frozen_bits(model, NCTX * 4);
      /**
       * The converged bank must beat the uniform prior it started from on a
       * stream this predictable. Requiring strictly fewer bits (not merely a
       * different number) is what distinguishes a real commit from a commit
       * that wrote something arbitrary.
       */
      if (!(after_commit < before)) {
        fprintf(stderr, "commit did not improve frozen code length: "
          "%.17g -> %.17g\n", before, after_commit);
      }
      else result = check_order_independent(model);
    }
    libxs_predict_prob_destroy(context);
  }
  else fprintf(stderr, "model cannot be scored\n");
  return result;
}


/** A context from a different model must be refused, not adapted. */
static int check_commit_rejects_foreign(libxs_predict_t* model)
{
  int result = EXIT_SUCCESS;
  libxs_predict_t* other = build_model();
  if (NULL != other) {
    void* context = libxs_predict_prob_create(other);
    if (NULL != context
      && EXIT_FAILURE != libxs_predict_prob_commit(model, context))
    {
      fprintf(stderr, "commit accepted a context from another model\n");
      result = EXIT_FAILURE;
    }
    libxs_predict_prob_destroy(context);
    libxs_predict_destroy(other);
  }
  if (EXIT_SUCCESS == result
    && EXIT_FAILURE != libxs_predict_prob_commit(model, NULL))
  {
    fprintf(stderr, "commit accepted a NULL context\n");
    result = EXIT_FAILURE;
  }
  return result;
}


/**
 * Committed weights must survive save/load, since the point of committing is a
 * figure someone else can reproduce from the saved model.
 */
static int check_commit_survives_roundtrip(const libxs_predict_t* model)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && 0 < size) {
    void* buffer = malloc(size);
    if (NULL != buffer) {
      if (EXIT_SUCCESS == libxs_predict_save(model, buffer, &size)) {
        libxs_predict_t* loaded = libxs_predict_load(buffer, size);
        if (NULL != loaded) {
          const double a = frozen_bits(model, NCTX * 4);
          const double b = frozen_bits(loaded, NCTX * 4);
          if (fabs(a - b) > 1e-9) {
            fprintf(stderr, "committed weights did not round-trip: "
              "%.17g vs %.17g\n", a, b);
          }
          else result = EXIT_SUCCESS;
          libxs_predict_destroy(loaded);
        }
        else fprintf(stderr, "load failed\n");
      }
      free(buffer);
    }
  }
  return result;
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  libxs_predict_t* model = build_model();
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);
  if (NULL == model) {
    fprintf(stderr, "model could not be built\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = check_commit_moves_weights(model);
  }
  if (EXIT_SUCCESS == result) {
    result = check_commit_rejects_foreign(model);
  }
  if (EXIT_SUCCESS == result) {
    result = check_commit_survives_roundtrip(model);
  }
  libxs_predict_destroy(model);
  return result;
}
