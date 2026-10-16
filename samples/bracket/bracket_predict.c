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
#include <libxs/libxs_timer.h>
#include "bracket.h"
#if defined(__XGBOOST)
# include <xgboost/c_api.h>
#endif

#define BRACKET_GATE 0.9
/* the defaults samples/predict compares against */
#define BRACKET_XGB_ROUNDS 200
#define BRACKET_XGB_PREDCFG "{\"type\":0,\"training\":false," \
  "\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":false}"


typedef struct predict_variant_t {
  const char* name;
  int decompose, coordinate, classify;
} predict_variant_t;

typedef struct predict_rank_t {
  double conf;
  size_t index;
  int right;
} predict_rank_t;

#if defined(__XGBOOST)
typedef struct predict_xgb_t {
  BoosterHandle booster;
} predict_xgb_t;
#endif


static const predict_variant_t predict_variants[] = {
  { "knn", LIBXS_PREDICT_RAW, LIBXS_PREDICT_RANK, 1 },
  { "knn_score", LIBXS_PREDICT_RAW, LIBXS_PREDICT_SCORE, 1 },
  { "fisher", LIBXS_PREDICT_FISHER, LIBXS_PREDICT_AUTO_COORD, 1 },
  { "hknn", LIBXS_PREDICT_HKNN, LIBXS_PREDICT_RANK, 1 },
  { "rf", LIBXS_PREDICT_RF, LIBXS_PREDICT_AUTO_COORD, 1 },
  { "knn_avg", LIBXS_PREDICT_RAW, LIBXS_PREDICT_RANK, 0 }
};


static void predict_grid(const libxs_predict_t* model, size_t n, double out[],
  double conf[]);
static int predict_eval(const void* model, size_t n, double out[]);
static libxs_predict_t* predict_build(const predict_variant_t* variant,
  const double data[], size_t n, double lo, double h);
static int predict_cmp(const void* a, const void* b);
static int predict_gate(const study_t* study, const char target[], const char variant[],
  int ncoord, const double image[], const double conf[]);
static int predict_write(const char target[], const char variant[], size_t ni,
  double image[], const double conf[]);
#if defined(__XGBOOST)
static int predict_xgb_grid(const predict_xgb_t* xgb, size_t n, double out[],
  double conf[]);
static int predict_xgb_eval(const void* model, size_t n, double out[]);
static int predict_xgb_build(predict_xgb_t* xgb, const double data[], size_t n,
  double lo, double h);
static int predict_xgb(const study_t* study, const char target[], const double data[],
  size_t n, double lo, double h, double image[], double conf[]);
#endif


/* the n x n image at x = i / (n - 1), and the confidence of each pixel if asked */
static void predict_grid(const libxs_predict_t* model, size_t n, double out[],
  double conf[])
{
  const double h = 1.0 / (double)(n - 1);
  size_t i, j;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      libxs_predict_info_t info;
      double inputs[2];
      inputs[0] = h * (double)i;
      inputs[1] = h * (double)j;
      libxs_predict_eval(NULL, model, inputs, out + i + n * j,
        (NULL != conf ? &info : NULL), 0);
      if (NULL != conf) {
        conf[i + n * j] = (NULL != info.confidence ? info.confidence[0] : 0);
      }
    }
  }
}


static int predict_eval(const void* model, size_t n, double out[])
{
  predict_grid((const libxs_predict_t*)model, n, out, NULL);
  return EXIT_SUCCESS;
}


static libxs_predict_t* predict_build(const predict_variant_t* variant,
  const double data[], size_t n, double lo, double h)
{
  libxs_predict_t* model = libxs_predict_create(2, 1);
  int result = (NULL != model ? EXIT_SUCCESS : EXIT_FAILURE);
  size_t i, j;
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(model, variant->decompose);
    libxs_predict_set_coordinate(model, variant->coordinate);
    if (0 != variant->classify) libxs_predict_set_mode(model, LIBXS_PREDICT_CLASSIFY);
  }
  for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
    for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
      const double value = data[i + n * j];
      double inputs[2], output;
      inputs[0] = lo + h * (double)i;
      inputs[1] = lo + h * (double)j;
      /* the sign of a cell average is the class covering most of the cell */
      output = (0 == variant->classify ? value : (0 <= value ? 1.0 : -1.0));
      result = libxs_predict_push(NULL, model, inputs, &output);
    }
  }
  if (EXIT_SUCCESS == result) result = libxs_predict_build(model, 0, 1, 0.0);
  if (EXIT_SUCCESS != result) {
    libxs_predict_destroy(model);
    model = NULL;
  }
  return model;
}


/* most confident first, ties in grid order so a plateau is cut the same way every run */
static int predict_cmp(const void* a, const void* b)
{
  const predict_rank_t *const ra = (const predict_rank_t*)a;
  const predict_rank_t *const rb = (const predict_rank_t*)b;
  int result = (ra->conf > rb->conf ? -1 : (ra->conf < rb->conf ? 1 : 0));
  if (0 == result) result = (ra->index < rb->index ? -1 : (ra->index > rb->index ? 1 : 0));
  return result;
}


/**
 * Precision at the gate, and at fixed coverage among the most confident pixels: a
 * confidence that ranks without being a probability never reaches some gates, and
 * only a matched coverage compares two models then.
 */
static int predict_gate(const study_t* study, const char target[], const char variant[],
  int ncoord, const double image[], const double conf[])
{
  static const double coverage[] = { 0.5, 0.8 };
  const size_t ni = study->nimage * study->nimage;
  predict_rank_t* const rank = (predict_rank_t*)malloc(sizeof(predict_rank_t) * ni);
  int result = (NULL != rank ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    size_t nacted = 0, nright = 0, i, c;
    for (i = 0; i < ni; ++i) {
      rank[i].conf = conf[i];
      rank[i].index = i;
      rank[i].right = (0 < image[i] * study->image[i] ? 1 : 0);
      if (BRACKET_GATE <= conf[i]) {
        ++nacted;
        nright += (size_t)rank[i].right;
      }
    }
    qsort(rank, ni, sizeof(predict_rank_t), predict_cmp);
    printf("  %s %s", target, variant);
    if (0 < ncoord) printf(" ncoord=%i", ncoord);
    printf(" conf>=%.2f: coverage %.1f%% precision %.1f%%; at coverage", BRACKET_GATE,
      100.0 * (double)nacted / (double)ni,
      0 < nacted ? 100.0 * (double)nright / (double)nacted : 0.0);
    for (c = 0; c < sizeof(coverage) / sizeof(*coverage); ++c) {
      const size_t ntop = (size_t)(coverage[c] * (double)ni + 0.5);
      size_t ntopright = 0;
      for (i = 0; i < ntop; ++i) ntopright += (size_t)rank[i].right;
      printf(" %.0f%%: %.1f%%", 100.0 * coverage[c],
        0 < ntop ? 100.0 * (double)ntopright / (double)ntop : 0.0);
    }
    printf("\n");
  }
  free(rank);
  return result;
}


/* metrics are taken; the image shows the target's range */
static int predict_write(const char target[], const char variant[], size_t ni,
  double image[], const double conf[])
{
  /* the targets' levels, and the confidence of a vote between two classes */
  static const double image_range[2] = { -1.0, 1.0 }, conf_range[2] = { 0.5, 1.0 };
  char name[64];
  size_t i;
  int result;
  for (i = 0; i < ni * ni; ++i) image[i] = LIBXS_CLMP(image[i], -1.0, 1.0);
  LIBXS_SNPRINTF(name, sizeof(name), "bracket_2d_%s_%s", target, variant);
  result = write_image(name, image, ni, ni, image_range);
  if (EXIT_SUCCESS == result) {
    LIBXS_SNPRINTF(name, sizeof(name), "bracket_2d_%s_%s_conf", target, variant);
    result = write_image(name, conf, ni, ni, conf_range);
  }
  return result;
}


#if defined(__XGBOOST)
/* as predict_grid: the class of the larger probability, and that probability */
static int predict_xgb_grid(const predict_xgb_t* xgb, size_t n, double out[],
  double conf[])
{
  const double h = 1.0 / (double)(n - 1);
  float* const x = (float*)malloc(sizeof(float) * 2 * n * n);
  DMatrixHandle grid = NULL;
  int result = (NULL != x ? 0 : -1);
  if (0 == result) {
    size_t i, j;
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        x[2 * (i + n * j) + 0] = (float)(h * (double)i);
        x[2 * (i + n * j) + 1] = (float)(h * (double)j);
      }
    }
    result = XGDMatrixCreateFromMat(x, (bst_ulong)(n * n), 2, -1.0f, &grid);
  }
  if (0 == result) {
    const bst_ulong* shape = NULL;
    const float* prob = NULL;
    bst_ulong ndim = 0;
    result = XGBoosterPredictFromDMatrix(xgb->booster, grid, BRACKET_XGB_PREDCFG,
      &shape, &ndim, &prob);
    if (0 == result) {
      size_t i;
      for (i = 0; i < n * n; ++i) {
        const double p0 = prob[2 * i + 0], p1 = prob[2 * i + 1];
        out[i] = (p0 <= p1 ? 1.0 : -1.0);
        if (NULL != conf) conf[i] = LIBXS_MAX(p0, p1);
      }
    }
  }
  if (NULL != grid) XGDMatrixFree(grid);
  free(x);
  return (0 == result ? EXIT_SUCCESS : EXIT_FAILURE);
}


static int predict_xgb_eval(const void* model, size_t n, double out[])
{
  return predict_xgb_grid((const predict_xgb_t*)model, n, out, NULL);
}


static int predict_xgb_build(predict_xgb_t* xgb, const double data[], size_t n,
  double lo, double h)
{
  float* const x = (float*)malloc(sizeof(float) * 3 * n * n);
  DMatrixHandle train = NULL;
  int result = (NULL != x ? 0 : -1), round;
  xgb->booster = NULL;
  if (0 == result) {
    float* const y = x + 2 * n * n;
    size_t i, j;
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        const size_t k = i + n * j;
        x[2 * k + 0] = (float)(lo + h * (double)i);
        x[2 * k + 1] = (float)(lo + h * (double)j);
        y[k] = (0 <= data[k] ? 1.0f : 0.0f);
      }
    }
    result = XGDMatrixCreateFromMat(x, (bst_ulong)(n * n), 2, -1.0f, &train);
    if (0 == result) result = XGDMatrixSetFloatInfo(train, "label", y, (bst_ulong)(n * n));
  }
  if (0 == result) result = XGBoosterCreate(&train, 1, &xgb->booster);
  if (0 == result) {
    result |= XGBoosterSetParam(xgb->booster, "objective", "multi:softprob");
    result |= XGBoosterSetParam(xgb->booster, "num_class", "2");
    result |= XGBoosterSetParam(xgb->booster, "max_depth", "6");
    result |= XGBoosterSetParam(xgb->booster, "eta", "0.1");
    result |= XGBoosterSetParam(xgb->booster, "nthread", "0");
    result |= XGBoosterSetParam(xgb->booster, "seed", "0");
    result |= XGBoosterSetParam(xgb->booster, "verbosity", "0");
  }
  for (round = 0; round < BRACKET_XGB_ROUNDS && 0 == result; ++round) {
    result = XGBoosterUpdateOneIter(xgb->booster, round, train);
  }
  if (NULL != train) XGDMatrixFree(train);
  free(x);
  return (0 == result ? EXIT_SUCCESS : EXIT_FAILURE);
}


static int predict_xgb(const study_t* study, const char target[], const double data[],
  size_t n, double lo, double h, double image[], double conf[])
{
  const libxs_timer_tick_t start = libxs_timer_tick();
  predict_xgb_t xgb;
  int result = predict_xgb_build(&xgb, data, n, lo, h);
  if (EXIT_SUCCESS == result) {
    result = report(study, target, "xgb", &xgb, predict_xgb_eval,
      libxs_timer_duration(start, libxs_timer_tick()), image);
  }
  if (EXIT_SUCCESS == result) {
    result = predict_xgb_grid(&xgb, study->nimage, image, conf);
  }
  if (EXIT_SUCCESS == result) result = predict_gate(study, target, "xgb", 0, image, conf);
  if (EXIT_SUCCESS == result) result = predict_write(target, "xgb", study->nimage, image, conf);
  if (NULL != xgb.booster) XGBoosterFree(xgb.booster);
  return result;
}
#endif


int bracket_predict(const study_t* study, const char target[], const double data[],
  size_t n, double lo, double h, int pixel, double image[])
{
  const size_t ni = study->nimage, nvariants = sizeof(predict_variants)
    / sizeof(*predict_variants);
  double* const conf = (double*)malloc(sizeof(double) * ni * ni);
  int result = (NULL != conf ? EXIT_SUCCESS : EXIT_FAILURE);
  size_t k;
  for (k = 0; k < nvariants && EXIT_SUCCESS == result; ++k) {
    const predict_variant_t* const variant = predict_variants + k;
    /* point samples are the classes already, and a regression on them is the vote */
    if (0 != variant->classify || 0 != pixel) {
      const libxs_timer_tick_t start = libxs_timer_tick();
      libxs_predict_t* const model = predict_build(variant, data, n, lo, h);
      result = (NULL != model ? EXIT_SUCCESS : EXIT_FAILURE);
      if (EXIT_SUCCESS == result) {
        result = report(study, target, variant->name, model, predict_eval,
          libxs_timer_duration(start, libxs_timer_tick()), image);
      }
      if (EXIT_SUCCESS == result) {
        predict_grid(model, ni, image, conf);
        if (0 != variant->classify) {
          libxs_predict_query_t query;
          memset(&query, 0, sizeof(query));
          libxs_predict_query(model, &query);
          result = predict_gate(study, target, variant->name, query.ncoord, image, conf);
        }
      }
      if (EXIT_SUCCESS == result) {
        result = predict_write(target, variant->name, ni, image, conf);
      }
      libxs_predict_destroy(model);
    }
  }
#if defined(__XGBOOST)
  if (EXIT_SUCCESS == result) {
    result = predict_xgb(study, target, data, n, lo, h, image, conf);
  }
#endif
  free(conf);
  return result;
}
