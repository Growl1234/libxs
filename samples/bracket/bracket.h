/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef BRACKET_H
#define BRACKET_H

#include <libxs/libxs.h>

#define BRACKET_MAX_EDGES 8

/* piecewise-constant profile: level[i] holds on [edge[i-1], edge[i]) */
typedef struct target1_t {
  const char* name;
  double edge[BRACKET_MAX_EDGES];
  double level[BRACKET_MAX_EDGES + 1];
  double noise;
  int nedges;
} target1_t;

typedef int (*eval_fn_t)(const void* model, size_t n, double out[]);
typedef double (*value_fn_t)(double x, double y, double angle);

typedef struct study_t {
  const target1_t* row;
  const double* image;
  double* buffer;
  size_t nimage, nlo, nhi, rowindex;
  double htrain, sumjsq;
  int ndims;
} study_t;

/**
 * Writes an image as MHD file and as PNG file, the latter scaled over range, or
 * over the image's own extremes where range is NULL.
 */
int write_image(const char name[], const double data[], size_t ncols, size_t nrows,
  const double range[2]);
int report(const study_t* study, const char target[], const char variant[],
  const void* model, eval_fn_t eval, double seconds, double image[]);
/**
 * Reports the nearest-neighbour and forest predictors of libxs_predict on the
 * training data of a two-dimensional target: n x n values at lo + h * i. With
 * pixel set, they are cell averages and a regression on them is reported too.
 * Each model's image goes where the other variants' go, and its confidence
 * next to it.
 */
int bracket_predict(const study_t* study, const char target[], const double data[],
  size_t n, double lo, double h, int pixel, double image[]);

#endif /*BRACKET_H*/
