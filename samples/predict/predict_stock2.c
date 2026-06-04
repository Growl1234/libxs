/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_timer.h>
#include <libxs_math.h>
#include <libxs_mem.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

enum { WINDOW = 20, HORIZON = 5, NSERIES = 2, NINPUTS = WINDOW * NSERIES };

static void build_inputs(const libxs_predict_t* source,
  int t, int series, double* inputs);
static void evaluate_forecast(const libxs_predict_t* const split_models[],
  const libxs_predict_t* const full_models[],
  const libxs_predict_t* source, int total, int train_end,
  int joint, int col1, int col2);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const int col1 = (argc > 2) ? atoi(argv[2]) : 1;
  const int col2 = (argc > 3) ? atoi(argv[3]) : 2;
  const double split = (argc > 4) ? atof(argv[4]) : 0.8;
  int result = EXIT_FAILURE;
  if (NULL == filename) {
    fprintf(stderr,
      "Usage: %s <csv_file> [col1] [col2] [train_fraction]\n"
      "  Paired-stock timeseries prediction (SPREAD decomposition).\n"
      "  Input: CSV with header; columns selected by 0-based index.\n"
      "  Uses set_diff(0) for auto-differencing.\n"
      "  Default col1=1, col2=2, train_fraction=0.8\n", argv[0]);
  }
  else {
    char inputs_spec[32], target_spec[16];
    libxs_predict_t* source;
    int total;
    LIBXS_SNPRINTF(inputs_spec, sizeof(inputs_spec), "%d,%d", col1, col2);
    LIBXS_SNPRINTF(target_spec, sizeof(target_spec), "%d", col1);
    source = libxs_predict_create(NSERIES, 1);
    total = (NULL != source)
      ? libxs_predict_load_csv(source, filename, NULL, inputs_spec, target_spec)
      : 0;
    if (0 < total) {
      const int train_end = LIBXS_MAX((int)(total * split + 0.5), WINDOW + 1);
      int s, t;
      fprintf(stdout, "Loaded %d rows from %s (columns %d, %d)\n",
        total, filename, col1, col2);
      fprintf(stdout, "Window=%d, Horizon=%d, Train=%d, Test=%d\n",
        WINDOW, HORIZON, train_end - WINDOW, total - train_end);
      { libxs_predict_t* split_m[NSERIES], *full_m[NSERIES];
        int ok = 1;
        for (s = 0; s < NSERIES; ++s) {
          split_m[s] = libxs_predict_create(NINPUTS, HORIZON);
          full_m[s] = libxs_predict_create(NINPUTS, HORIZON);
          if (NULL != split_m[s] && NULL != full_m[s]) {
            libxs_predict_set_mode(split_m[s], LIBXS_PREDICT_TEMPORAL);
            libxs_predict_set_diff(split_m[s], 0);
            libxs_predict_set_series(split_m[s], NSERIES, WINDOW);
            libxs_predict_set_target(split_m[s], s);
            libxs_predict_set_decompose(split_m[s], LIBXS_PREDICT_SPREAD);
            libxs_predict_set_mode(full_m[s], LIBXS_PREDICT_TEMPORAL);
            libxs_predict_set_diff(full_m[s], 0);
            libxs_predict_set_series(full_m[s], NSERIES, WINDOW);
            libxs_predict_set_target(full_m[s], s);
            libxs_predict_set_decompose(full_m[s], LIBXS_PREDICT_SPREAD);
            for (t = 0; t < total; ++t) {
              double vals[NSERIES];
              libxs_predict_get(source, t, vals, NULL);
              if (t < train_end) {
                libxs_predict_push(NULL, split_m[s], vals, NULL);
              }
              libxs_predict_push(NULL, full_m[s], vals, NULL);
            }
            if (EXIT_SUCCESS != libxs_predict_build(split_m[s], 0, 2)) ok = 0;
            if (EXIT_SUCCESS != libxs_predict_build(full_m[s], 0, 2)) ok = 0;
          }
          else {
            ok = 0;
          }
        }
        if (0 != ok) {
          libxs_predict_query_t qi;
          LIBXS_MEMZERO(&qi);
          libxs_predict_query(split_m[0], &qi);
          fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d,"
            " diff=%d\n", qi.nclusters, qi.compression, qi.order,
            qi.diff_order);
          fprintf(stdout, "\n--- SPREAD decomposition (sum/diff modes) ---\n");
          evaluate_forecast((const libxs_predict_t* const*)split_m,
            (const libxs_predict_t* const*)full_m,
            source, total, train_end, 1, col1, col2);
          result = EXIT_SUCCESS;
        }
        for (s = 0; s < NSERIES; ++s) {
          if (NULL != split_m[s]) libxs_predict_destroy(split_m[s]);
          if (NULL != full_m[s]) libxs_predict_destroy(full_m[s]);
        }
      }
      { libxs_predict_t* split_m[NSERIES], *full_m[NSERIES];
        int ok = 1;
        for (s = 0; s < NSERIES; ++s) {
          split_m[s] = libxs_predict_create(NINPUTS, HORIZON);
          full_m[s] = libxs_predict_create(NINPUTS, HORIZON);
          if (NULL != split_m[s] && NULL != full_m[s]) {
            libxs_predict_set_mode(split_m[s], LIBXS_PREDICT_TEMPORAL);
            libxs_predict_set_diff(split_m[s], 0);
            libxs_predict_set_series(split_m[s], NSERIES, WINDOW);
            libxs_predict_set_target(split_m[s], s);
            libxs_predict_set_mode(full_m[s], LIBXS_PREDICT_TEMPORAL);
            libxs_predict_set_diff(full_m[s], 0);
            libxs_predict_set_series(full_m[s], NSERIES, WINDOW);
            libxs_predict_set_target(full_m[s], s);
            for (t = 0; t < total; ++t) {
              double vals[NSERIES];
              libxs_predict_get(source, t, vals, NULL);
              if (t < train_end) {
                libxs_predict_push(NULL, split_m[s], vals, NULL);
              }
              libxs_predict_push(NULL, full_m[s], vals, NULL);
            }
            if (EXIT_SUCCESS != libxs_predict_build(split_m[s], 0, 2)) ok = 0;
            if (EXIT_SUCCESS != libxs_predict_build(full_m[s], 0, 2)) ok = 0;
          }
          else {
            ok = 0;
          }
        }
        if (0 != ok) {
          fprintf(stdout, "\n--- RAW concatenation (baseline) ---\n");
          evaluate_forecast((const libxs_predict_t* const*)split_m,
            (const libxs_predict_t* const*)full_m,
            source, total, train_end, 1, col1, col2);
        }
        for (s = 0; s < NSERIES; ++s) {
          if (NULL != split_m[s]) libxs_predict_destroy(split_m[s]);
          if (NULL != full_m[s]) libxs_predict_destroy(full_m[s]);
        }
      }
      { libxs_predict_t* split_m[NSERIES], *full_m[NSERIES];
        int ok = 1;
        for (s = 0; s < NSERIES; ++s) {
          split_m[s] = libxs_predict_create(WINDOW, HORIZON);
          full_m[s] = libxs_predict_create(WINDOW, HORIZON);
          if (NULL != split_m[s] && NULL != full_m[s]) {
            libxs_predict_set_mode(split_m[s], LIBXS_PREDICT_TEMPORAL);
            libxs_predict_set_diff(split_m[s], 0);
            libxs_predict_set_series(split_m[s], 1, WINDOW);
            libxs_predict_set_mode(full_m[s], LIBXS_PREDICT_TEMPORAL);
            libxs_predict_set_diff(full_m[s], 0);
            libxs_predict_set_series(full_m[s], 1, WINDOW);
            for (t = 0; t < total; ++t) {
              double vals[NSERIES];
              libxs_predict_get(source, t, vals, NULL);
              if (t < train_end) {
                libxs_predict_push(NULL, split_m[s], &vals[s], NULL);
              }
              libxs_predict_push(NULL, full_m[s], &vals[s], NULL);
            }
            if (EXIT_SUCCESS != libxs_predict_build(split_m[s], 0, 2)) ok = 0;
            if (EXIT_SUCCESS != libxs_predict_build(full_m[s], 0, 2)) ok = 0;
          }
          else {
            ok = 0;
          }
        }
        if (0 != ok) {
          fprintf(stdout, "\n--- Single-series (independent) ---\n");
          evaluate_forecast((const libxs_predict_t* const*)split_m,
            (const libxs_predict_t* const*)full_m,
            source, total, train_end, 0, col1, col2);
        }
        for (s = 0; s < NSERIES; ++s) {
          if (NULL != split_m[s]) libxs_predict_destroy(split_m[s]);
          if (NULL != full_m[s]) libxs_predict_destroy(full_m[s]);
        }
      }
    }
    else {
      fprintf(stderr, "Failed to load data from %s\n", filename);
    }
    if (NULL != source) libxs_predict_destroy(source);
  }
  return result;
}


static void build_inputs(const libxs_predict_t* source,
  int t, int series, double* inputs)
{
  int i;
  if (series < 0) {
    for (i = 0; i < WINDOW; ++i) {
      double vals[NSERIES];
      libxs_predict_get(source, t - WINDOW + i, vals, NULL);
      inputs[i] = vals[0];
      inputs[WINDOW + i] = vals[1];
    }
  }
  else {
    for (i = 0; i < WINDOW; ++i) {
      double vals[NSERIES];
      libxs_predict_get(source, t - WINDOW + i, vals, NULL);
      inputs[i] = vals[series];
    }
  }
}


static void evaluate_forecast(const libxs_predict_t* const split_models[],
  const libxs_predict_t* const full_models[],
  const libxs_predict_t* source, int total, int train_end,
  int joint, int col1, int col2)
{
  double sum_err[NSERIES][HORIZON] = {{0}}, max_err[NSERIES][HORIZON] = {{0}};
  double forecast[NSERIES][HORIZON];
  double sum_conf = 0, fc_conf = 0;
  int neval = 0, h, s, t;
  for (t = train_end; t <= total - HORIZON; ++t) {
    double inputs[NINPUTS], outputs[HORIZON];
    libxs_predict_info_t info;
    for (s = 0; s < NSERIES; ++s) {
      build_inputs(source, t, (0 != joint) ? -1 : s, inputs);
      libxs_predict_eval(NULL, split_models[s], inputs, outputs, &info, 1);
      for (h = 0; h < HORIZON; ++h) {
        double vals[NSERIES], actual;
        libxs_predict_get(source, t + h, vals, NULL);
        actual = vals[s];
        { const double err = LIBXS_FABS(outputs[h] - actual);
          sum_err[s][h] += err;
          if (err > max_err[s][h]) max_err[s][h] = err;
        }
      }
      if (0 == s) sum_conf += info.confidence[0];
    }
    ++neval;
  }
  for (s = 0; s < NSERIES; ++s) {
    double inputs[NINPUTS];
    libxs_predict_info_t info;
    build_inputs(source, total, (0 != joint) ? -1 : s, inputs);
    libxs_predict_eval(NULL, full_models[s], inputs, forecast[s], &info, 1);
    fc_conf = info.confidence[0];
  }
  if (0 < neval) {
    fprintf(stdout, "Quality (%d windows) and forecast (confidence %.3f):\n",
      neval, fc_conf);
    fprintf(stdout, "  step   avg-err   max-err   col %-4d"
      "   avg-err   max-err   col %-4d\n", col1, col2);
    for (h = 0; h < HORIZON; ++h) {
      fprintf(stdout, "  t+%-2d  %8.4f  %8.4f  %8.4f"
        "  %8.4f  %8.4f  %8.4f\n", h + 1,
        sum_err[0][h] / neval, max_err[0][h], forecast[0][h],
        sum_err[1][h] / neval, max_err[1][h], forecast[1][h]);
    }
    fprintf(stdout, "  avg confidence: %.3f\n", sum_conf / neval);
  }
}
