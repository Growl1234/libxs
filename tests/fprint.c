/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_math.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  /* 1-D: constant array, all derivatives beyond order 0 should be zero */
  { double a[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4)) {
      FPRINTF(stderr, "ERROR line #%i: constant fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (4 != fa.order || 5 != fa.n) {
      FPRINTF(stderr, "ERROR line #%i: order=%d n=%d\n", __LINE__, fa.order, fa.n);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.norms[0]) {
      FPRINTF(stderr, "ERROR line #%i: constant norm[0]=%.17g\n", __LINE__, fa.norms[0]);
      exit(EXIT_FAILURE);
    }
    { int k;
      for (k = 1; k <= fa.order; ++k) {
        if (1E-10 < fa.norms[k]) {
          FPRINTF(stderr, "ERROR line #%i: constant norm[%d]=%.17g\n", __LINE__, k, fa.norms[k]);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  /* 1-D: linear ramp, order-1 norm nonzero, order>=2 should be zero */
  { double a[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4)) {
      FPRINTF(stderr, "ERROR line #%i: linear fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.norms[0] || 1E-12 > fa.norms[1]) {
      FPRINTF(stderr, "ERROR line #%i: linear norm[0]=%.17g norm[1]=%.17g\n",
        __LINE__, fa.norms[0], fa.norms[1]);
      exit(EXIT_FAILURE);
    }
    { int k;
      for (k = 2; k <= fa.order; ++k) {
        if (1E-8 < fa.norms[k]) {
          FPRINTF(stderr, "ERROR line #%i: linear norm[%d]=%.17g\n", __LINE__, k, fa.norms[k]);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  /* 1-D: self-distance is zero */
  { double a[] = {1.0, 3.0, 2.0, 7.0, 5.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4);
    d = libxs_fprint_diff(&fa, &fa, NULL);
    if (1E-15 < d) {
      FPRINTF(stderr, "ERROR line #%i: self-distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: identical data, different lengths */
  { double a5[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    double a3[] = {0.0, 4.0, 16.0};
    const size_t s5[] = {5}, s3[] = {3};
    libxs_fprint_t f5, f3;
    libxs_fprint(&f5, LIBXS_DATATYPE_F64, a5, 1, s5, NULL, 2);
    libxs_fprint(&f3, LIBXS_DATATYPE_F64, a3, 1, s3, NULL, 2);
    FPRINTF(stderr, "INFO line #%i: quadratic 5vs3 dist=%.17g\n",
      __LINE__, libxs_fprint_diff(&f5, &f3, NULL));
  }

  /* 1-D: structurally different data yields nonzero distance */
  { double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {1.0, 4.0, 2.0, 8.0};
    const size_t shape[] = {4};
    libxs_fprint_t fa, fb;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 3);
    d = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-12 > d) {
      FPRINTF(stderr, "ERROR line #%i: different distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: symmetry d(a,b) == d(b,a) */
  { double a[] = {1.0, 4.0, 2.0, 8.0};
    double b[] = {2.0, 3.0, 5.0, 7.0};
    const size_t shape[] = {4};
    libxs_fprint_t fa, fb;
    double dab, dba;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 3);
    dab = libxs_fprint_diff(&fa, &fb, NULL);
    dba = libxs_fprint_diff(&fb, &fa, NULL);
    if (1E-15 < LIBXS_DELTA(dab, dba)) {
      FPRINTF(stderr, "ERROR line #%i: symmetry dab=%.17g dba=%.17g\n", __LINE__, dab, dba);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: custom weights */
  { double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {1.1, 2.1, 3.1, 4.1};
    double w[] = {1.0, 0.0, 0.0, 0.0};
    const size_t shape[] = {4};
    libxs_fprint_t fa, fb;
    double d_val, d_def;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 3);
    d_val = libxs_fprint_diff(&fa, &fb, w);
    d_def = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-15 < LIBXS_DELTA(d_val, d_def) && d_val > d_def) {
      FPRINTF(stderr, "ERROR line #%i: value-only > default\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: F32 type */
  { float a[] = {1.0f, 2.0f, 4.0f, 8.0f};
    const size_t shape[] = {4};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F32, a, 1, shape, NULL, 3)) {
      FPRINTF(stderr, "ERROR line #%i: F32 fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.norms[0]) {
      FPRINTF(stderr, "ERROR line #%i: F32 norm[0]=%.17g\n", __LINE__, fa.norms[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: I32 type */
  { int a[] = {10, 20, 30, 40};
    const size_t shape[] = {4};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_I32, a, 1, shape, NULL, 2)) {
      FPRINTF(stderr, "ERROR line #%i: I32 fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.norms[0]) {
      FPRINTF(stderr, "ERROR line #%i: I32 norm[0]=%.17g\n", __LINE__, fa.norms[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: order clamped to n-1 */
  { double a[] = {1.0, 2.0};
    const size_t shape[] = {2};
    libxs_fprint_t fa;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 10);
    if (1 != fa.order) {
      FPRINTF(stderr, "ERROR line #%i: clamp order=%d\n", __LINE__, fa.order);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: single element */
  { double a[] = {42.0};
    const size_t shape[] = {1};
    libxs_fprint_t fa;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 5);
    if (0 != fa.order || 1 != fa.n) {
      FPRINTF(stderr, "ERROR line #%i: single order=%d n=%d\n", __LINE__, fa.order, fa.n);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.norms[0]) {
      FPRINTF(stderr, "ERROR line #%i: single norm[0]=%.17g\n", __LINE__, fa.norms[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: identical matrices yield zero self-distance */
  { double m[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0,
                  7.0, 8.0, 9.0,
                  10.0, 11.0, 12.0};
    const size_t shape[] = {3, 4};
    libxs_fprint_t fa, fb;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, m, 2, shape, NULL, 2);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, m, 2, shape, NULL, 2);
    d = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-15 < d) {
      FPRINTF(stderr, "ERROR line #%i: 2D self-distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: different matrices yield nonzero distance */
  { double a[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    double b[] = {1.0, 9.0, 3.0,
                  4.0, 0.0, 6.0};
    const size_t shape[] = {3, 2};
    libxs_fprint_t fa, fb;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 2, shape, NULL, 2);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 2, shape, NULL, 2);
    d = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-12 > d) {
      FPRINTF(stderr, "ERROR line #%i: 2D different distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: symmetry */
  { double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double b[] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    const size_t shape[] = {3, 2};
    libxs_fprint_t fa, fb;
    double dab, dba;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 2, shape, NULL, 2);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 2, shape, NULL, 2);
    dab = libxs_fprint_diff(&fa, &fb, NULL);
    dba = libxs_fprint_diff(&fb, &fa, NULL);
    if (1E-14 < LIBXS_DELTA(dab, dba)) {
      FPRINTF(stderr, "ERROR line #%i: 2D symmetry dab=%.17g dba=%.17g\n", __LINE__, dab, dba);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D with stride: column-major 3x2 with leading dimension 4 */
  { double m[] = {1.0, 2.0, 3.0, 99.0,
                  4.0, 5.0, 6.0, 99.0};
    const size_t shape[] = {3, 2};
    const size_t stride[] = {1, 4};
    double ref[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const size_t rshape[] = {3, 2};
    libxs_fprint_t fm, fr;
    double d;
    libxs_fprint(&fm, LIBXS_DATATYPE_F64, m, 2, shape, stride, 2);
    libxs_fprint(&fr, LIBXS_DATATYPE_F64, ref, 2, rshape, NULL, 2);
    d = libxs_fprint_diff(&fm, &fr, NULL);
    if (1E-14 < d) {
      FPRINTF(stderr, "ERROR line #%i: 2D stride distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: different shapes are comparable */
  { double a[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    double b[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                  7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    const size_t sa[] = {3, 2}, sb[] = {6, 2};
    libxs_fprint_t fa, fb;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 2, sa, NULL, 2);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 2, sb, NULL, 2);
    FPRINTF(stderr, "INFO line #%i: 2D shape diff dist=%.17g\n",
      __LINE__, libxs_fprint_diff(&fa, &fb, NULL));
  }

  /* 3-D: smoke test */
  { double a[2*3*4];
    const size_t shape[] = {4, 3, 2};
    libxs_fprint_t fa;
    int i;
    for (i = 0; i < 24; ++i) a[i] = (double)(i * i);
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 3, shape, NULL, 2)) {
      FPRINTF(stderr, "ERROR line #%i: 3D fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.norms[0]) {
      FPRINTF(stderr, "ERROR line #%i: 3D norm[0]=%.17g\n", __LINE__, fa.norms[0]);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
