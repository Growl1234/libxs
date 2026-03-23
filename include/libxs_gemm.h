/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_GEMM_H
#define LIBXS_GEMM_H

#include "libxs.h"

/** Standard Fortran BLAS dgemm signature (e.g., dgemm_). */
typedef void (*libxs_dgemm_blas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc);

/** Standard Fortran BLAS sgemm signature (e.g., sgemm_). */
typedef void (*libxs_sgemm_blas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc);

/**
 * MKL JIT dgemm kernel signature (mkl_jit_get_dgemm_ptr).
 * Shape, alpha, beta, and transpose info are baked into the jitter handle.
 */
typedef void (*libxs_dgemm_jit_t)(void* jitter,
  const double* a, const double* b, double* c);

/**
 * MKL JIT sgemm kernel signature (mkl_jit_get_sgemm_ptr).
 * Shape, alpha, beta, and transpose info are baked into the jitter handle.
 */
typedef void (*libxs_sgemm_jit_t)(void* jitter,
  const float* a, const float* b, float* c);

/**
 * XGEMM parameter struct, layout-compatible with libxsmm_gemm_param.
 * op: 4 void pointers (op state), a/b/c: 6 void pointers each (matrix arg).
 * Only a.primary, b.primary, c.primary are used for plain GEMM.
 */
typedef struct libxs_xgemm_param_t {
  void* op[4]; void* a[6]; void* b[6]; void* c[6];
} libxs_xgemm_param_t;

/** Opaque XGEMM kernel: void(const libxs_xgemm_param_t*). */
typedef void (*libxs_xgemm_t)(const void*);

/** Flags controlling GEMM batch synchronization (bitfield). */
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1
} libxs_gemm_flags_t;

/**
 * Configuration supplying GEMM kernels. Pass NULL to batch functions
 * to use the built-in default kernel (auto-vectorized, no BLAS dependency).
 * Users can supply their own kernels. Kernel selection priority:
 *   1. JIT kernel (dgemm_jit/sgemm_jit + jitter) if non-NULL,
 *   2. XGEMM kernel (xgemm) if non-NULL,
 *   3. BLAS kernel (dgemm/sgemm) if non-NULL,
 *   4. built-in default kernel.
 * Only the function pointers matching the datatype need to be set.
 * By default (flags=0), _task variants synchronize C-matrix updates.
 * Set LIBXS_GEMM_FLAG_NOLOCK if no duplicate C pointers exist.
 */
typedef struct libxs_gemm_config_t {
  libxs_dgemm_blas_t dgemm_blas;
  libxs_sgemm_blas_t sgemm_blas;
  libxs_dgemm_jit_t dgemm_jit;
  libxs_sgemm_jit_t sgemm_jit;
  libxs_xgemm_t xgemm;
  void* jitter;
  libxs_gemm_flags_t flags;
} libxs_gemm_config_t;

/**
 * Process a batch of GEMMs with strided access (constant offsets between matrices).
 * C_i := alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchsize).
 * Matrices are at: A + i*stride_a*elemsize, B + i*stride_b*elemsize, C + i*stride_c*elemsize.
 * Pass config=NULL to use the built-in kernel.
 */
LIBXS_API void libxs_gemm_strided(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config);

/** Per-thread form of libxs_gemm_strided. */
LIBXS_API void libxs_gemm_strided_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);

/**
 * Process a batch of GEMMs given arrays of pointers to matrices.
 * C_i := alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchsize).
 * Pass config=NULL to use the built-in kernel.
 */
LIBXS_API void libxs_gemm_batch(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config);

/** Per-thread form of libxs_gemm_batch. */
LIBXS_API void libxs_gemm_batch_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);

/**
 * Process groups of batched GEMMs with varying parameters.
 * Each group i has its own transa, transb, m, n, k, lda, ldb, ldc, and batchsize.
 * The a/b/c pointer arrays are concatenated across groups.
 * alpha and beta are arrays of ngroups scalars (each LIBXS_TYPESIZE Bytes).
 * Pass config=NULL to use the built-in kernel.
 */
LIBXS_API void libxs_gemm_groups(
  libxs_data_t datatype, const char transa_array[], const char transb_array[],
  const int m_array[], const int n_array[], const int k_array[],
  const void* alpha_array, const void* a_array[], const int lda_array[],
                           const void* b_array[], const int ldb_array[],
  const void* beta_array,        void* c_array[], const int ldc_array[],
  int ngroups, const int batchsize[],
  const libxs_gemm_config_t* config);

/**
 * Dispatch a GEMM kernel and populate config accordingly.
 * With MKL JIT (highest priority): sets config->dgemm_jit or sgemm_jit + jitter.
 * With LIBXSMM (fallback): sets config->xgemm via libxsmm_dispatch_gemm.
 * Without either: config is unchanged (falls through to BLAS/default).
 * The caller should memset config to zero before the first call.
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE when no JIT backend is available.
 */
LIBXS_API_INLINE int libxs_dispatch_gemm(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta)
{
#if defined(mkl_jit_create_dgemm)
  { const int ta = ('N' == transa || 'n' == transa) ? 111 : 112;
    const int tb = ('N' == transb || 'n' == transb) ? 111 : 112;
    if (LIBXS_DATATYPE_F64 == datatype) {
      void* jitter = NULL;
      const int status = mkl_cblas_jit_create_dgemm(&jitter,
        MKL_COL_MAJOR, (CBLAS_TRANSPOSE)ta, (CBLAS_TRANSPOSE)tb,
        m, n, k,
        NULL != alpha ? *(const double*)alpha : 1.0, lda, ldb,
        NULL != beta ? *(const double*)beta : 0.0, ldc);
      if (MKL_JIT_SUCCESS == status) {
        config->dgemm_jit = (libxs_dgemm_jit_t)mkl_jit_get_dgemm_ptr(jitter);
        config->jitter = jitter;
        return EXIT_SUCCESS;
      }
    }
    else if (LIBXS_DATATYPE_F32 == datatype) {
      void* jitter = NULL;
      const int status = mkl_cblas_jit_create_sgemm(&jitter,
        MKL_COL_MAJOR, (CBLAS_TRANSPOSE)ta, (CBLAS_TRANSPOSE)tb,
        m, n, k,
        NULL != alpha ? *(const float*)alpha : 1.0f, lda, ldb,
        NULL != beta ? *(const float*)beta : 0.0f, ldc);
      if (MKL_JIT_SUCCESS == status) {
        config->sgemm_jit = (libxs_sgemm_jit_t)mkl_jit_get_sgemm_ptr(jitter);
        config->jitter = jitter;
        return EXIT_SUCCESS;
      }
    }
  }
#endif
#if defined(LIBXSMM_H)
  { libxsmm_gemm_shape gemm_shape;
    libxsmm_gemmfunction result;
    LIBXS_UNUSED(transa); LIBXS_UNUSED(transb);
    LIBXS_UNUSED(alpha); LIBXS_UNUSED(beta);
    gemm_shape.m = m; gemm_shape.n = n; gemm_shape.k = k;
    gemm_shape.lda = lda; gemm_shape.ldb = ldb; gemm_shape.ldc = ldc;
    switch (datatype) {
      case LIBXS_DATATYPE_F64:
        gemm_shape.a_in_type = LIBXSMM_DATATYPE_F64;
        gemm_shape.b_in_type = LIBXSMM_DATATYPE_F64;
        gemm_shape.comp_type = LIBXSMM_DATATYPE_F64;
        gemm_shape.out_type = LIBXSMM_DATATYPE_F64;
        break;
      case LIBXS_DATATYPE_F32:
        gemm_shape.a_in_type = LIBXSMM_DATATYPE_F32;
        gemm_shape.b_in_type = LIBXSMM_DATATYPE_F32;
        gemm_shape.comp_type = LIBXSMM_DATATYPE_F32;
        gemm_shape.out_type = LIBXSMM_DATATYPE_F32;
        break;
      default: return EXIT_FAILURE;
    }
    result = libxsmm_dispatch_gemm(gemm_shape, 0/*flags*/,
      LIBXSMM_GEMM_PREFETCH_NONE);
    if (NULL != result) {
      config->xgemm = (libxs_xgemm_t)result;
      return EXIT_SUCCESS;
    }
  }
#endif
#if !defined(mkl_jit_create_dgemm) && !defined(LIBXSMM_H)
  LIBXS_UNUSED(config);
  LIBXS_UNUSED(datatype); LIBXS_UNUSED(transa); LIBXS_UNUSED(transb);
  LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(k);
  LIBXS_UNUSED(lda); LIBXS_UNUSED(ldb); LIBXS_UNUSED(ldc);
  LIBXS_UNUSED(alpha); LIBXS_UNUSED(beta);
#endif
  return EXIT_FAILURE;
}

/**
 * Release resources acquired by libxs_dispatch_gemm (e.g., MKL jitter).
 * Safe to call even if dispatch was not used or returned zero.
 */
LIBXS_API_INLINE void libxs_release_gemm(libxs_gemm_config_t* config)
{
  if (NULL != config) {
#if defined(mkl_jit_create_dgemm)
    if (NULL != config->jitter) {
      mkl_jit_destroy(config->jitter);
      config->jitter = NULL;
    }
#endif
    config->xgemm = NULL;
    config->dgemm_jit = NULL;
    config->sgemm_jit = NULL;
  }
}

#endif /*LIBXS_GEMM_H*/
