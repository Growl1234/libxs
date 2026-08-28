/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PREDICT_XGB_H
#define PREDICT_XGB_H

#include <libxs/libxs_predict.h>
#include <xgboost/c_api.h>

/**
 * Largest attested value set an output may carry and still be posed to XGBoost
 * as a classification.  A wider output falls back to regression, which is
 * reported rather than silently substituted: the two tasks are not comparable.
 */
#define PREDICT_XGB_MAXCLASS 64

/** Prediction configuration: whole model, inference mode, plain 2D output. */
#define PREDICT_XGB_PREDCFG "{\"type\":0,\"training\":false," \
  "\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":false}"


static int predict_xgb_geti(const char* name, int fallback)
{
  const char* const env = getenv(name);
  int result = fallback;
  if (NULL != env && '\0' != *env) result = atoi(env);
  return result;
}


static double predict_xgb_getd(const char* name, double fallback)
{
  const char* const env = getenv(name);
  double result = fallback;
  if (NULL != env && '\0' != *env) result = atof(env);
  return result;
}


/**
 * Distinct values of one output over the training subset, ascending.  Returns
 * the count, or capacity+1 when the set is wider than capacity (the caller must
 * treat that as "too wide to classify" rather than as a truncated set).
 */
static int predict_xgb_support(const float labels[], int n,
  double values[], int capacity)
{
  int result = 0, i, ok = 1;
  for (i = 0; i < n && 0 != ok; ++i) {
    const double value = (double)labels[i];
    int j = 0;
    while (j < result && values[j] < value) ++j;
    if (j == result || values[j] != value) {
      if (result < capacity) {
        int k;
        for (k = result; k > j; --k) values[k] = values[k-1];
        values[j] = value;
        ++result;
      }
      else {
        result = capacity + 1;
        ok = 0;
      }
    }
  }
  return result;
}


/**
 * Regression objective: XGB_REGOBJ wins, else what the caller proposes, else
 * squared error.  A sample reporting mean absolute error should propose
 * reg:absoluteerror, because training on squared error and scoring on absolute
 * error is the same mismatch libxs_predict_set_central removes on the LIBXS side.
 */
static const char* predict_xgb_regobj(const char* proposed)
{
  const char* const env = getenv("XGB_REGOBJ");
  const char* result = "reg:squarederror";
  if (NULL != env && '\0' != *env) result = env;
  else if (NULL != proposed && '\0' != *proposed) result = proposed;
  return result;
}


static int predict_xgb_params(BoosterHandle booster, int nclass,
  const char* regobj)
{
  char buffer[64];
  int result = 0;
  if (0 < nclass) {
    result |= XGBoosterSetParam(booster, "objective", "multi:softprob");
    LIBXS_SNPRINTF(buffer, sizeof(buffer), "%i", nclass);
    result |= XGBoosterSetParam(booster, "num_class", buffer);
  }
  else {
    result |= XGBoosterSetParam(booster, "objective", regobj);
  }
  LIBXS_SNPRINTF(buffer, sizeof(buffer), "%i",
    predict_xgb_geti("XGB_DEPTH", 6));
  result |= XGBoosterSetParam(booster, "max_depth", buffer);
  LIBXS_SNPRINTF(buffer, sizeof(buffer), "%g",
    predict_xgb_getd("XGB_ETA", 0.1));
  result |= XGBoosterSetParam(booster, "eta", buffer);
  LIBXS_SNPRINTF(buffer, sizeof(buffer), "%i",
    predict_xgb_geti("XGB_NTHREAD", 0));
  result |= XGBoosterSetParam(booster, "nthread", buffer);
  result |= XGBoosterSetParam(booster, "seed", "0");
  result |= XGBoosterSetParam(booster, "verbosity", "0");
  return result;
}


/**
 * Train one booster over dtrain and write its prediction for every row of dall
 * into predicted[i*stride], confidence[i*stride] (confidence may be NULL, and
 * is the winning class probability for a classification, 0 for a regression).
 */
static int predict_xgb_output(DMatrixHandle dtrain, DMatrixHandle dall,
  int ntotal, int nclass, const double values[],
  double predicted[], double confidence[], int stride, const char* regobj)
{
  BoosterHandle booster = NULL;
  const int nrounds = predict_xgb_geti("XGB_ROUNDS", 200);
  int result = XGBoosterCreate(&dtrain, 1, &booster);
  if (0 == result) result = predict_xgb_params(booster, nclass, regobj);
  if (0 == result) {
    int i;
    for (i = 0; i < nrounds && 0 == result; ++i) {
      result = XGBoosterUpdateOneIter(booster, i, dtrain);
    }
  }
  if (0 == result) {
    const float* out = NULL;
    const bst_ulong* shape = NULL;
    bst_ulong ndim = 0;
    result = XGBoosterPredictFromDMatrix(booster, dall,
      PREDICT_XGB_PREDCFG, &shape, &ndim, &out);
    if (0 == result && NULL != out && 0 < ndim) {
      const int width = (1 < ndim) ? (int)shape[1] : 1;
      int i;
      for (i = 0; i < ntotal; ++i) {
        if (0 < nclass && 1 < width) {
          const float* const row = out + (size_t)i * width;
          int best = 0, c;
          for (c = 1; c < width; ++c) {
            if (row[c] > row[best]) best = c;
          }
          predicted[(size_t)i * stride] = values[best];
          if (NULL != confidence) {
            confidence[(size_t)i * stride] = (double)row[best];
          }
        }
        else {
          predicted[(size_t)i * stride] = (double)out[i];
          if (NULL != confidence) confidence[(size_t)i * stride] = 0.0;
        }
      }
    }
  }
  XGBoosterFree(booster);
  return result;
}


/**
 * Train XGBoost on exactly the entries LIBXS was built from and predict every
 * row, so both models are scored on one split.  Handing this a mask other than
 * the one the LIBXS model saw invalidates the comparison with no symptom to
 * notice, which is why the mask is an argument rather than recomputed here.
 *
 * source:    corpus the LIBXS model was pushed from (need not be built).
 * trained:   ntotal flags, non-zero where the entry was trained on.
 * classify:  noutputs flags requesting classification over the attested value
 *            set instead of regression (NULL requests regression throughout).
 * predicted: ntotal*noutputs values written, in user space.
 * confidence: ntotal*noutputs values written (may be NULL).
 * task:      noutputs values written (may be NULL) reporting what was actually
 *            posed per output: 0 regression, 1 constant, >1 classification over
 *            that many attested values.  An output asked to classify a set
 *            wider than PREDICT_XGB_MAXCLASS reports regression instead, which
 *            is a different task and must not be compared as if it were one.
 * regobj:    regression objective proposed for this corpus (NULL for the
 *            default); XGB_REGOBJ overrides it.
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int predict_xgb(const libxs_predict_t* source, int ntotal,
  int ninputs, int noutputs, const char trained[], const int classify[],
  double predicted[], double confidence[], int task[], const char* regobj)
{
  double* row = (double*)malloc((size_t)(ninputs + noutputs) * sizeof(double));
  float* x = (float*)malloc((size_t)ntotal * ninputs * sizeof(float));
  float* y = (float*)malloc((size_t)ntotal * noutputs * sizeof(float));
  float* xtrain = (float*)malloc((size_t)ntotal * ninputs * sizeof(float));
  float* ytrain = (float*)malloc((size_t)ntotal * sizeof(float));
  int result = EXIT_FAILURE;
  if (NULL != row && NULL != x && NULL != y
    && NULL != xtrain && NULL != ytrain)
  {
    DMatrixHandle dtrain = NULL, dall = NULL;
    int ntrain = 0, i, j, status;
    for (i = 0; i < ntotal; ++i) {
      libxs_predict_get(source, i, row, row + ninputs);
      for (j = 0; j < ninputs; ++j) {
        x[(size_t)i * ninputs + j] = (float)row[j];
      }
      for (j = 0; j < noutputs; ++j) {
        y[(size_t)i * noutputs + j] = (float)row[ninputs+j];
      }
      if (NULL == trained || 0 != trained[i]) {
        for (j = 0; j < ninputs; ++j) {
          xtrain[(size_t)ntrain * ninputs + j] = x[(size_t)i * ninputs + j];
        }
        ++ntrain;
      }
    }
    status = XGDMatrixCreateFromMat(xtrain, (bst_ulong)ntrain,
      (bst_ulong)ninputs, -1.0f, &dtrain);
    if (0 == status) {
      status = XGDMatrixCreateFromMat(x, (bst_ulong)ntotal,
        (bst_ulong)ninputs, -1.0f, &dall);
    }
    for (j = 0; j < noutputs && 0 == status; ++j) {
      double values[PREDICT_XGB_MAXCLASS];
      int nclass = 0, n = 0;
      for (i = 0; i < ntotal; ++i) {
        if (NULL == trained || 0 != trained[i]) {
          ytrain[n++] = y[(size_t)i * noutputs + j];
        }
      }
      if (NULL != classify && 0 != classify[j]) {
        nclass = predict_xgb_support(ytrain, n, values,
          PREDICT_XGB_MAXCLASS);
        if (PREDICT_XGB_MAXCLASS < nclass) nclass = 0;
      }
      if (1 < nclass) {
        for (i = 0; i < n; ++i) {
          int k = 0;
          while (k < nclass && values[k] != (double)ytrain[i]) ++k;
          ytrain[i] = (float)k;
        }
      }
      if (1 == nclass) {
        for (i = 0; i < ntotal; ++i) {
          predicted[(size_t)i * noutputs + j] = values[0];
          if (NULL != confidence) confidence[(size_t)i * noutputs + j] = 1.0;
        }
      }
      else {
        status = XGDMatrixSetFloatInfo(dtrain, "label", ytrain,
          (bst_ulong)n);
        if (0 == status) {
          status = predict_xgb_output(dtrain, dall, ntotal, nclass, values,
            predicted + j, (NULL != confidence) ? (confidence + j) : NULL,
            noutputs, predict_xgb_regobj(regobj));
        }
      }
      if (NULL != task) task[j] = nclass;
    }
    if (0 != status) {
      fprintf(stderr, "XGBoost error: %s\n", XGBGetLastError());
    }
    else {
      result = EXIT_SUCCESS;
    }
    XGDMatrixFree(dall);
    XGDMatrixFree(dtrain);
  }
  free(ytrain);
  free(xtrain);
  free(y);
  free(x);
  free(row);
  return result;
}

#endif /*PREDICT_XGB_H*/
