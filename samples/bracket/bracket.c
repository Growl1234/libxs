/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs.h>
#include <libxs/libxs_gemm.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_rng.h>
#include <libxs/libxs_timer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BRACKET_PATH_MAX 1024
#define BRACKET_MAX_EDGES 8
#define BRACKET_PI 3.14159265358979323846
#define BRACKET_PHASE 0.37
#define BRACKET_NOISE 0.1
#define BRACKET_CELLS 4
#define BRACKET_N1 64
#define BRACKET_H1 32
#define BRACKET_NPARAM1 (4 * BRACKET_H1 + 2)
#define BRACKET_IMAGE1 1024
#define BRACKET_LO1 1024
#define BRACKET_HI1 4096
#define BRACKET_LR1 1e-2
#define BRACKET_N2 48
#define BRACKET_U2 64
#define BRACKET_H2 32
#define BRACKET_IMAGE2 256
#define BRACKET_LO2 128
#define BRACKET_HI2 512
#define BRACKET_LR2 1e-2
#define BRACKET_LRMLP 2e-3
#define BRACKET_CHUNK 8192


/* piecewise-constant profile: level[i] holds on [edge[i-1], edge[i]) */
typedef struct target1_t {
  const char* name;
  double edge[BRACKET_MAX_EDGES];
  double level[BRACKET_MAX_EDGES + 1];
  double noise;
  int nedges;
} target1_t;

typedef struct train_t {
  int nsteps;
  double lr, wd;
} train_t;

typedef struct model1_t {
  double param[BRACKET_NPARAM1];
  int integrated;
} model1_t;

typedef struct tensor_t {
  double* param;
  int nu, integrated;
} tensor_t;

typedef struct mlp_t {
  double* param;
  int nh;
} mlp_t;

typedef int (*eval_fn_t)(const void* model, size_t n, double out[]);

typedef struct study_t {
  const target1_t* row;
  const double* clean;
  const double* image;
  double* buffer;
  size_t ntrain, nimage, nlo, nhi, rowindex;
  double sumjsq;
  int ndims;
} study_t;


static char output_dir[BRACKET_PATH_MAX] = { 0 };

static int write_mhd(const char name[], const double data[], size_t ncols, size_t nrows);
static double gaussian(void);
static double cosine_lr(double lr, int step, int nsteps);
static void adam_update(double param[], const double grad[], double m[], double v[],
  int nparam, int ndecay, int step, double lr, double wd);
static int gemm_f64(char transa, char transb, int m, int n, int k,
  const double a[], int lda, const double b[], int ldb, double c[], int ldc);
static double target1_value(const target1_t* target, double x);
static void integrate1(const double f[], size_t n, double out[]);
static void model1_init(model1_t* model, int integrated);
static double model1_value(const model1_t* model, double x);
static void model1_train(model1_t* model, const double x[], const double y[], int n,
  const train_t* train);
static int model1_eval(const void* model, size_t n, double out[]);
static double checker_edge(int i);
static double checker_value(double x, double y);
static void checker_row(double y, target1_t* row);
static int tensor_init(tensor_t* model, int nu, int integrated);
static void tensor_factor(const double param[], int nu, int axis, size_t n,
  double relu[], double step[]);
static int tensor_train(tensor_t* model, const double y[], int n, const train_t* train);
static int tensor_eval(const void* model, size_t n, double out[]);
static int mlp_init(mlp_t* model, int nh);
static int mlp_forward(const double param[], int nh, const double xs[], const double ys[],
  int npts, double a1[], double a2[], double out[]);
static int mlp_train(mlp_t* model, const double y[], int n, const train_t* train);
static int mlp_eval(const void* model, size_t n, double out[]);
static double edge_error(const double y[], size_t n, const target1_t* target);
static int fprint_field(libxs_fprint_t* info, const double data[], int ndims, size_t n,
  int order);
static int report(const study_t* study, const char target[], const char variant[],
  const void* model, eval_fn_t eval, double seconds, double image[]);
static int run_1d(const train_t* train, unsigned int seed);
static int run_2d(const train_t* train, unsigned int seed);


int main(void)
{
  const char *const env_dir = getenv("BRACKET_OUTDIR");
  const char *const env_nsteps = getenv("BRACKET_NSTEPS");
  const char *const env_wd = getenv("BRACKET_WD");
  const char *const env_seed = getenv("BRACKET_SEED");
  int result = EXIT_SUCCESS;
  train_t train;
  unsigned int seed;
  train.nsteps = (NULL != env_nsteps ? atoi(env_nsteps) : 0);
  if (0 >= train.nsteps) train.nsteps = 20000;
  train.wd = (NULL != env_wd ? atof(env_wd) : 1e-2);
  train.lr = BRACKET_LR1;
  seed = (unsigned int)(NULL != env_seed ? atoi(env_seed) : 1);
  if (NULL != env_dir && 0 != *env_dir && strlen(env_dir) < sizeof(output_dir)) {
    strcpy(output_dir, env_dir);
  }
  libxs_init();
  printf("BRACKET: sharp boundaries from bracket units (steps=%i wd=%g seed=%u)\n",
    train.nsteps, train.wd, seed);
  printf("  output: %s\n", 0 != output_dir[0] ? output_dir : "stdout only");
  printf("%-8s %-14s %10s %8s %7s %9s %10s %8s\n",
    "target", "variant", "rms", "pos/h", "slope", "eps/h", "fpdist", "time");
  if (EXIT_SUCCESS == result) result = run_1d(&train, seed);
  if (EXIT_SUCCESS == result) result = run_2d(&train, seed);
  if (EXIT_SUCCESS == result) printf("BRACKET: done\n");
  libxs_finalize();
  return result;
}


static int write_mhd(const char name[], const double data[], size_t ncols, size_t nrows)
{
  int result = EXIT_SUCCESS;
  if (0 != output_dir[0]) {
    char path[BRACKET_PATH_MAX];
    const size_t dir_len = strlen(output_dir);
    if (dir_len + strlen(name) + 6 < sizeof(path)) {
      libxs_mhd_info_t info;
      size_t size[2];
      strcpy(path, output_dir);
      if ('/' != path[dir_len - 1]) strcat(path, "/");
      strcat(path, name);
      strcat(path, ".mhd");
      size[0] = ncols;
      size[1] = nrows;
      info.ndims = 2;
      info.ncomponents = 1;
      info.type = LIBXS_DATATYPE_F64;
      info.header_size = 0;
      result = libxs_mhd_write(path, NULL, size, NULL, &info, data, NULL);
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


static double gaussian(void)
{
  const double u1 = 1.0 - libxs_rng_f64();
  const double u2 = libxs_rng_f64();
  return sqrt(-2.0 * log(u1)) * cos(2.0 * BRACKET_PI * u2);
}


static double cosine_lr(double lr, int step, int nsteps)
{
  return 0.5 * lr * (1.0 + cos(BRACKET_PI * (double)step / (double)nsteps));
}


/* decoupled weight decay (AdamW) applies to the leading ndecay parameters only */
static void adam_update(double param[], const double grad[], double m[], double v[],
  int nparam, int ndecay, int step, double lr, double wd)
{
  const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  const double scale1 = 1.0 / (1.0 - pow(beta1, (double)step));
  const double scale2 = 1.0 / (1.0 - pow(beta2, (double)step));
  int i;
  for (i = 0; i < nparam; ++i) {
    m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
    if (i < ndecay) param[i] -= lr * wd * param[i];
    param[i] -= lr * (m[i] * scale1) / (sqrt(v[i] * scale2) + epsilon);
  }
}


static int gemm_f64(char transa, char transb, int m, int n, int k,
  const double a[], int lda, const double b[], int ldb, double c[], int ldc)
{
  const double alpha = 1.0, beta = 0.0;
  const libxs_gemm_config_t *const config = libxs_gemm_dispatch(LIBXS_DATATYPE_F64,
    transa, transb, m, n, k, lda, ldb, ldc, &alpha, &beta, NULL);
  int result = EXIT_SUCCESS;
  if (NULL != config) libxs_gemm_call(config, a, b, c);
  else result = EXIT_FAILURE;
  return result;
}


static double target1_value(const target1_t* target, double x)
{
  int i = 0;
  while (i < target->nedges && target->edge[i] <= x) ++i;
  return target->level[i];
}


/* trapezoid: a step between two samples integrates to a ramp through their midpoint */
static void integrate1(const double f[], size_t n, double out[])
{
  const double h = 1.0 / (double)(n - 1);
  size_t i;
  out[0] = 0;
  for (i = 1; i < n; ++i) out[i] = out[i - 1] + 0.5 * h * (f[i - 1] + f[i]);
}


static void model1_init(model1_t* model, int integrated)
{
  double *const w = model->param, *const c = w + BRACKET_H1;
  double *const b = c + BRACKET_H1, *const d = b + BRACKET_H1;
  int i;
  for (i = 0; i < BRACKET_H1; ++i) {
    w[i] = (0.5 > libxs_rng_f64() ? -1.0 : 1.0) * (0.5 + libxs_rng_f64());
    b[i] = w[i] * libxs_rng_f64();
    c[i] = 0.1 * (2.0 * libxs_rng_f64() - 1.0);
  }
  d[0] = d[1] = 0;
  model->integrated = integrated;
}


/* direct: d0 + d1 x + sum c<wx-b>^1; integrated: its derivative d1 + sum cw<wx-b>^0 */
static double model1_value(const model1_t* model, double x)
{
  const double *const w = model->param, *const c = w + BRACKET_H1;
  const double *const b = c + BRACKET_H1, *const d = b + BRACKET_H1;
  double result = (0 == model->integrated ? d[0] + d[1] * x : d[1]);
  int i;
  for (i = 0; i < BRACKET_H1; ++i) {
    const double u = w[i] * x - b[i];
    if (0 < u) result += (0 == model->integrated ? c[i] * u : c[i] * w[i]);
  }
  return result;
}


/* both variants fit the same function form; only the target (f or its integral) differs */
static void model1_train(model1_t* model, const double x[], const double y[], int n,
  const train_t* train)
{
  double grad[BRACKET_NPARAM1], m[BRACKET_NPARAM1], v[BRACKET_NPARAM1];
  double *const w = model->param, *const c = w + BRACKET_H1;
  double *const b = c + BRACKET_H1, *const d = b + BRACKET_H1;
  double *const gw = grad, *const gc = gw + BRACKET_H1;
  double *const gb = gc + BRACKET_H1, *const gd = gb + BRACKET_H1;
  int step, i, j;
  memset(m, 0, sizeof(m));
  memset(v, 0, sizeof(v));
  for (step = 1; step <= train->nsteps; ++step) {
    memset(grad, 0, sizeof(grad));
    for (j = 0; j < n; ++j) {
      double r = d[0] + d[1] * x[j] - y[j];
      for (i = 0; i < BRACKET_H1; ++i) {
        const double u = w[i] * x[j] - b[i];
        if (0 < u) r += c[i] * u;
      }
      r /= (double)n;
      for (i = 0; i < BRACKET_H1; ++i) {
        const double u = w[i] * x[j] - b[i];
        if (0 < u) {
          gc[i] += r * u;
          gw[i] += r * c[i] * x[j];
          gb[i] -= r * c[i];
        }
      }
      gd[0] += r;
      gd[1] += r * x[j];
    }
    adam_update(model->param, grad, m, v, BRACKET_NPARAM1, 2 * BRACKET_H1, step,
      cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
  }
}


static int model1_eval(const void* model, size_t n, double out[])
{
  const model1_t *const model1 = (const model1_t*)model;
  size_t j;
  for (j = 0; j < n; ++j) out[j] = model1_value(model1, (double)j / (double)(n - 1));
  return EXIT_SUCCESS;
}


static double checker_edge(int i)
{
  return ((double)i + BRACKET_PHASE) / BRACKET_CELLS;
}


static double checker_value(double x, double y)
{
  int ix = 0, iy = 0;
  while (ix < BRACKET_CELLS && checker_edge(ix) <= x) ++ix;
  while (iy < BRACKET_CELLS && checker_edge(iy) <= y) ++iy;
  return (0 == ((ix + iy) & 1) ? 1.0 : -1.0);
}


static void checker_row(double y, target1_t* row)
{
  int i;
  row->name = "checker";
  row->nedges = BRACKET_CELLS;
  row->noise = 0;
  for (i = 0; i < BRACKET_CELLS; ++i) row->edge[i] = checker_edge(i);
  for (i = 0; i <= BRACKET_CELLS; ++i) {
    const double x = (0 == i ? 0.0 : row->edge[i - 1]);
    row->level[i] = checker_value(x, y);
  }
}


/* parameters: w, v, c (decayed), then b, e */
static int tensor_init(tensor_t* model, int nu, int integrated)
{
  int result = EXIT_SUCCESS, u;
  model->param = (double*)malloc(sizeof(double) * 5 * (size_t)nu);
  model->nu = nu;
  model->integrated = integrated;
  if (NULL != model->param) {
    double *const w = model->param, *const v = w + nu, *const c = v + nu;
    double *const b = c + nu, *const e = b + nu;
    for (u = 0; u < nu; ++u) {
      w[u] = (0.5 > libxs_rng_f64() ? -1.0 : 1.0) * (0.5 + libxs_rng_f64());
      v[u] = (0.5 > libxs_rng_f64() ? -1.0 : 1.0) * (0.5 + libxs_rng_f64());
      b[u] = w[u] * libxs_rng_f64();
      e[u] = v[u] * libxs_rng_f64();
      c[u] = 0.1 * (2.0 * libxs_rng_f64() - 1.0);
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


/* column u of relu/step holds <s x - t>^1 and <s x - t>^0 on the grid (axis 0: w,b; 1: v,e) */
static void tensor_factor(const double param[], int nu, int axis, size_t n,
  double relu[], double step[])
{
  const double *const s = param + (0 == axis ? 0 : nu);
  const double *const t = param + 3 * nu + (0 == axis ? 0 : nu);
  size_t j;
  int u;
  for (u = 0; u < nu; ++u) {
    for (j = 0; j < n; ++j) {
      const double z = s[u] * (double)j / (double)(n - 1) - t[u];
      if (NULL != relu) relu[j + n * (size_t)u] = (0 < z ? z : 0);
      if (NULL != step) step[j + n * (size_t)u] = (0 < z ? 1 : 0);
    }
  }
}


/* G = sum_u c_u p_u q_u^T = (P diag(c)) Q^T, and the gradient needs R Q and R^T P */
static int tensor_train(tensor_t* model, const double y[], int n, const train_t* train)
{
  const int nu = model->nu, nparam = 5 * nu;
  const size_t nn = (size_t)n * (size_t)n, nf = (size_t)n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * (nn + 5 * nf + 3 * (size_t)nparam));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), step, u, j;
  if (EXIT_SUCCESS == result) {
    double *const r = work, *const p = r + nn, *const q = p + nf, *const pc = q + nf;
    double *const rq = pc + nf, *const rp = rq + nf, *const grad = rp + nf;
    double *const m = grad + nparam, *const v = m + nparam;
    const double *const w = model->param, *const vy = w + nu, *const c = vy + nu;
    double *const gw = grad, *const gv = gw + nu, *const gc = gv + nu;
    double *const gb = gc + nu, *const ge = gb + nu;
    memset(m, 0, sizeof(double) * 2 * (size_t)nparam);
    for (step = 1; step <= train->nsteps && EXIT_SUCCESS == result; ++step) {
      size_t i;
      tensor_factor(model->param, nu, 0, (size_t)n, p, NULL);
      tensor_factor(model->param, nu, 1, (size_t)n, q, NULL);
      for (u = 0; u < nu; ++u) {
        for (j = 0; j < n; ++j) pc[j + (size_t)n * u] = c[u] * p[j + (size_t)n * u];
      }
      result = gemm_f64('N', 'T', n, n, nu, pc, n, q, n, r, n);
      for (i = 0; i < nn; ++i) r[i] = (r[i] - y[i]) / (double)nn;
      if (EXIT_SUCCESS == result) result = gemm_f64('N', 'N', n, nu, n, r, n, q, n, rq, n);
      if (EXIT_SUCCESS == result) result = gemm_f64('T', 'N', n, nu, n, r, n, p, n, rp, n);
      if (EXIT_SUCCESS == result) {
        for (u = 0; u < nu; ++u) {
          double sc = 0, sw = 0, sb = 0, sv = 0, se = 0;
          for (j = 0; j < n; ++j) {
            const size_t k = j + (size_t)n * u;
            const double x = (double)j / (double)(n - 1);
            sc += p[k] * rq[k];
            if (0 < p[k]) {
              sw += x * rq[k];
              sb -= rq[k];
            }
            if (0 < q[k]) {
              sv += x * rp[k];
              se -= rp[k];
            }
          }
          gc[u] = sc;
          gw[u] = c[u] * sw;
          gb[u] = c[u] * sb;
          gv[u] = c[u] * sv;
          ge[u] = c[u] * se;
        }
        adam_update(model->param, grad, m, v, nparam, 3 * nu, step,
          cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
      }
    }
  }
  free(work);
  return result;
}


/* integrated: d^2/dxdy turns each (1,1) product into the (0,0) product of steps */
static int tensor_eval(const void* model, size_t n, double out[])
{
  const tensor_t *const tensor = (const tensor_t*)model;
  const int nu = tensor->nu;
  const size_t nf = n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * 2 * nf);
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), u;
  if (EXIT_SUCCESS == result) {
    double *const p = work, *const q = p + nf;
    const double *const w = tensor->param, *const v = w + nu, *const c = v + nu;
    size_t j;
    if (0 == tensor->integrated) {
      tensor_factor(tensor->param, nu, 0, n, p, NULL);
      tensor_factor(tensor->param, nu, 1, n, q, NULL);
    }
    else {
      tensor_factor(tensor->param, nu, 0, n, NULL, p);
      tensor_factor(tensor->param, nu, 1, n, NULL, q);
    }
    for (u = 0; u < nu; ++u) {
      const double s = (0 == tensor->integrated ? c[u] : c[u] * w[u] * v[u]);
      for (j = 0; j < n; ++j) p[j + n * (size_t)u] *= s;
    }
    result = gemm_f64('N', 'T', (int)n, (int)n, nu, p, (int)n, q, (int)n, out, (int)n);
  }
  free(work);
  return result;
}


/* parameters: W1 (x then y column), W2, w3 (decayed), then b1, b2, b3 */
static int mlp_init(mlp_t* model, int nh)
{
  const size_t nparam = (size_t)(nh * nh + 6 * nh + 1);
  int result = EXIT_SUCCESS, h;
  model->param = (double*)malloc(sizeof(double) * nparam);
  model->nh = nh;
  if (NULL != model->param) {
    double *const w1 = model->param, *const w2 = w1 + 2 * nh, *const w3 = w2 + nh * nh;
    double *const b1 = w3 + nh, *const b2 = b1 + nh, *const b3 = b2 + nh;
    const double s1 = sqrt(3.0), s2 = sqrt(6.0 / nh);
    for (h = 0; h < nh; ++h) {
      w1[h] = s1 * (2.0 * libxs_rng_f64() - 1.0);
      w1[nh + h] = s1 * (2.0 * libxs_rng_f64() - 1.0);
      b1[h] = w1[h] * libxs_rng_f64() + w1[nh + h] * libxs_rng_f64();
      b2[h] = 0;
      w3[h] = 0.1 * s2 * (2.0 * libxs_rng_f64() - 1.0);
    }
    for (h = 0; h < nh * nh; ++h) w2[h] = s2 * (2.0 * libxs_rng_f64() - 1.0);
    b3[0] = 0;
  }
  else result = EXIT_FAILURE;
  return result;
}


static int mlp_forward(const double param[], int nh, const double xs[], const double ys[],
  int npts, double a1[], double a2[], double out[])
{
  const double *const w1 = param, *const w2 = w1 + 2 * nh, *const w3 = w2 + nh * nh;
  const double *const b1 = w3 + nh, *const b2 = b1 + nh, *const b3 = b2 + nh;
  int result, i, h;
  for (i = 0; i < npts; ++i) {
    for (h = 0; h < nh; ++h) {
      const double z = w1[h] * xs[i] + w1[nh + h] * ys[i] - b1[h];
      a1[h + (size_t)nh * i] = (0 < z ? z : 0);
    }
  }
  result = gemm_f64('N', 'N', nh, npts, nh, w2, nh, a1, nh, a2, nh);
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < npts; ++i) {
      double s = -b3[0];
      for (h = 0; h < nh; ++h) {
        const size_t k = h + (size_t)nh * i;
        const double z = a2[k] - b2[h];
        a2[k] = (0 < z ? z : 0);
        s += w3[h] * a2[k];
      }
      out[i] = s;
    }
  }
  return result;
}


static int mlp_train(mlp_t* model, const double y[], int n, const train_t* train)
{
  const int nh = model->nh, npts = n * n, nparam = nh * nh + 6 * nh + 1;
  const size_t na = (size_t)nh * (size_t)npts;
  double *const work = (double*)malloc(sizeof(double)
    * (4 * na + 3 * (size_t)npts + 3 * (size_t)nparam));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), step, i, h;
  if (EXIT_SUCCESS == result) {
    double *const a1 = work, *const a2 = a1 + na, *const d1 = a2 + na, *const d2 = d1 + na;
    double *const xs = d2 + na, *const ys = xs + npts, *const out = ys + npts;
    double *const grad = out + npts, *const m = grad + nparam, *const v = m + nparam;
    const double *const w2 = model->param + 2 * nh, *const w3 = w2 + nh * nh;
    double *const gw1 = grad, *const gw2 = gw1 + 2 * nh, *const gw3 = gw2 + nh * nh;
    double *const gb1 = gw3 + nh, *const gb2 = gb1 + nh, *const gb3 = gb2 + nh;
    for (i = 0; i < npts; ++i) {
      xs[i] = (double)(i % n) / (double)(n - 1);
      ys[i] = (double)(i / n) / (double)(n - 1);
    }
    memset(m, 0, sizeof(double) * 2 * (size_t)nparam);
    for (step = 1; step <= train->nsteps && EXIT_SUCCESS == result; ++step) {
      result = mlp_forward(model->param, nh, xs, ys, npts, a1, a2, out);
      if (EXIT_SUCCESS == result) {
        gb3[0] = 0;
        for (h = 0; h < nh; ++h) gw3[h] = 0;
        for (i = 0; i < npts; ++i) {
          const double g = (out[i] - y[i]) / (double)npts;
          gb3[0] -= g;
          for (h = 0; h < nh; ++h) {
            const size_t k = h + (size_t)nh * i;
            gw3[h] += g * a2[k];
            d2[k] = (0 < a2[k] ? w3[h] * g : 0);
          }
        }
        result = gemm_f64('N', 'T', nh, nh, npts, d2, nh, a1, nh, gw2, nh);
      }
      if (EXIT_SUCCESS == result) {
        result = gemm_f64('T', 'N', nh, npts, nh, w2, nh, d2, nh, d1, nh);
      }
      if (EXIT_SUCCESS == result) {
        for (h = 0; h < nh; ++h) gw1[h] = gw1[nh + h] = gb1[h] = gb2[h] = 0;
        for (i = 0; i < npts; ++i) {
          for (h = 0; h < nh; ++h) {
            const size_t k = h + (size_t)nh * i;
            gb2[h] -= d2[k];
            if (0 < a1[k]) {
              gw1[h] += d1[k] * xs[i];
              gw1[nh + h] += d1[k] * ys[i];
              gb1[h] -= d1[k];
            }
          }
        }
        adam_update(model->param, grad, m, v, nparam, nh * nh + 3 * nh, step,
          cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
      }
    }
  }
  free(work);
  return result;
}


static int mlp_eval(const void* model, size_t n, double out[])
{
  const mlp_t *const mlp = (const mlp_t*)model;
  const int nh = mlp->nh;
  const size_t npts = n * n, na = (size_t)nh * BRACKET_CHUNK;
  double *const work = (double*)malloc(sizeof(double) * (2 * na + 2 * BRACKET_CHUNK));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    double *const a1 = work, *const a2 = a1 + na, *const xs = a2 + na;
    double *const ys = xs + BRACKET_CHUNK;
    size_t base, i;
    for (base = 0; base < npts && EXIT_SUCCESS == result; base += BRACKET_CHUNK) {
      const size_t size = LIBXS_MIN(npts - base, (size_t)BRACKET_CHUNK);
      for (i = 0; i < size; ++i) {
        xs[i] = (double)((base + i) % n) / (double)(n - 1);
        ys[i] = (double)((base + i) / n) / (double)(n - 1);
      }
      result = mlp_forward(mlp->param, nh, xs, ys, (int)size, a1, a2, out + base);
    }
  }
  free(work);
  return result;
}


/* mean distance between each edge and the nearest mid-level crossing of y */
static double edge_error(const double y[], size_t n, const target1_t* target)
{
  const double h = 1.0 / (double)(n - 1);
  double result = 0, window = 0.5;
  size_t j;
  int i;
  for (i = 0; i <= target->nedges; ++i) {
    const double lo = (0 == i ? 0.0 : target->edge[i - 1]);
    const double hi = (target->nedges == i ? 1.0 : target->edge[i]);
    window = LIBXS_MIN(window, 0.5 * (hi - lo));
  }
  for (i = 0; i < target->nedges; ++i) {
    const double edge = target->edge[i];
    const double mid = 0.5 * (target->level[i] + target->level[i + 1]);
    double best = window;
    for (j = 0; j + 1 < n; ++j) {
      const double x = (double)j * h;
      if (edge - window <= x && x + h <= edge + window) {
        const double ylo = y[j] - mid, yhi = y[j + 1] - mid;
        if (0 >= ylo * yhi && ylo != yhi) {
          const double cross = x - h * ylo / (yhi - ylo);
          if (fabs(cross - edge) < best) best = fabs(cross - edge);
        }
      }
    }
    result += best;
  }
  return (0 < target->nedges ? result / target->nedges : 0);
}


static int fprint_field(libxs_fprint_t* info, const double data[], int ndims, size_t n,
  int order)
{
  const unsigned int flags = (2 == ndims ? (unsigned int)LIBXS_FPRINT_PERAXIS : 0U);
  size_t shape[2];
  shape[0] = shape[1] = n;
  return libxs_fprint(info, LIBXS_DATATYPE_F64, data, ndims, shape, NULL, order, 0, 0, flags);
}


/**
 * Order-1 norm of a jump J grows as J/sqrt(h) but saturates at J/sqrt(eps) for an edge
 * of width eps: slope 0.5 is a true jump, slope 0 a smooth edge of width sum(J^2)/norm^2.
 */
static int report(const study_t* study, const char target[], const char variant[],
  const void* model, eval_fn_t eval, double seconds, double image[])
{
  const size_t ntrain = (2 == study->ndims ? study->ntrain * study->ntrain : study->ntrain);
  const double htrain = 1.0 / (double)(study->ntrain - 1);
  libxs_fprint_t lo, hi, fmodel, fclean;
  double rms = 0, pos = 0, slope = 0, eps = 0, dist = 0;
  size_t i;
  int result = eval(model, study->ntrain, study->buffer);
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < ntrain; ++i) {
      const double diff = study->buffer[i] - study->clean[i];
      rms += diff * diff;
    }
    rms = sqrt(rms / (double)ntrain);
    result = eval(model, study->nimage, image);
  }
  if (EXIT_SUCCESS == result) {
    pos = edge_error(image + study->nimage * study->rowindex, study->nimage, study->row);
    result = fprint_field(&fmodel, image, study->ndims, study->nimage, 2);
  }
  if (EXIT_SUCCESS == result) {
    result = fprint_field(&fclean, study->image, study->ndims, study->nimage, 2);
  }
  if (EXIT_SUCCESS == result) {
    dist = libxs_fprint_diff(&fmodel, &fclean, NULL);
    result = eval(model, study->nlo, study->buffer);
  }
  if (EXIT_SUCCESS == result) {
    result = fprint_field(&lo, study->buffer, study->ndims, study->nlo, 1);
  }
  if (EXIT_SUCCESS == result) result = eval(model, study->nhi, study->buffer);
  if (EXIT_SUCCESS == result) {
    result = fprint_field(&hi, study->buffer, study->ndims, study->nhi, 1);
  }
  if (EXIT_SUCCESS == result && 0 < lo.l2[1] && 0 < hi.l2[1]) {
    slope = log(hi.l2[1] / lo.l2[1])
      / log((double)(study->nhi - 1) / (double)(study->nlo - 1));
    eps = study->sumjsq / (hi.l2[1] * hi.l2[1]);
  }
  if (EXIT_SUCCESS == result) {
    printf("%-8s %-14s %10.3e %8.3f %7.3f %9.3e %10.3e %7.2fs\n", target, variant,
      rms, pos / htrain, slope, eps / htrain, dist, seconds);
  }
  return result;
}


static int run_1d(const train_t* train, unsigned int seed)
{
  static const char *const variants[] = {
    "direct", "direct_wd", "integrated", "integrated_wd"
  };
  const int nvariants = (int)(sizeof(variants) / sizeof(*variants));
  double x[BRACKET_N1], f[BRACKET_N1], clean[BRACKET_N1], area[BRACKET_N1];
  double stack[5 * BRACKET_IMAGE1], buffer[BRACKET_HI1];
  target1_t targets[3];
  study_t study;
  train_t train1 = *train;
  int result = EXIT_SUCCESS, t, k, i;
  targets[0].name = "step";
  targets[0].nedges = 1;
  targets[0].edge[0] = BRACKET_PHASE;
  targets[0].level[0] = 0;
  targets[0].level[1] = 1;
  targets[0].noise = 0;
  targets[1].name = "square";
  targets[1].nedges = BRACKET_CELLS;
  targets[1].noise = 0;
  for (i = 0; i <= BRACKET_CELLS; ++i) {
    if (i < BRACKET_CELLS) targets[1].edge[i] = checker_edge(i);
    targets[1].level[i] = (double)(i & 1);
  }
  targets[2] = targets[0];
  targets[2].name = "noisy";
  targets[2].noise = BRACKET_NOISE;
  train1.lr = BRACKET_LR1;
  memset(&study, 0, sizeof(study));
  study.ndims = 1;
  study.ntrain = BRACKET_N1;
  study.nimage = BRACKET_IMAGE1;
  study.nlo = BRACKET_LO1;
  study.nhi = BRACKET_HI1;
  study.clean = clean;
  study.image = stack;
  study.buffer = buffer;
  for (t = 0; t < 3 && EXIT_SUCCESS == result; ++t) {
    const target1_t *const target = targets + t;
    char name[64];
    study.row = target;
    study.sumjsq = 0;
    for (i = 0; i < target->nedges; ++i) {
      const double jump = target->level[i + 1] - target->level[i];
      study.sumjsq += jump * jump;
    }
    libxs_rng_set_seed(seed + 1000);
    for (i = 0; i < BRACKET_N1; ++i) {
      x[i] = (double)i / (double)(BRACKET_N1 - 1);
      clean[i] = target1_value(target, x[i]);
      f[i] = clean[i] + target->noise * gaussian();
    }
    integrate1(f, BRACKET_N1, area);
    for (i = 0; i < BRACKET_IMAGE1; ++i) {
      stack[i] = target1_value(target, (double)i / (double)(BRACKET_IMAGE1 - 1));
    }
    for (k = 0; k < nvariants && EXIT_SUCCESS == result; ++k) {
      const int integrated = (2 <= k ? 1 : 0);
      libxs_timer_tick_t start;
      model1_t model;
      libxs_rng_set_seed(seed + (unsigned int)t);
      model1_init(&model, integrated);
      train1.wd = (0 != (k & 1) ? train->wd : 0);
      start = libxs_timer_tick();
      model1_train(&model, x, 0 != integrated ? area : f, BRACKET_N1, &train1);
      result = report(&study, target->name, variants[k], &model, model1_eval,
        libxs_timer_duration(start, libxs_timer_tick()),
        stack + (size_t)(k + 1) * BRACKET_IMAGE1);
    }
    if (EXIT_SUCCESS == result) {
      LIBXS_SNPRINTF(name, sizeof(name), "bracket_1d_%s", target->name);
      result = write_mhd(name, stack, BRACKET_IMAGE1, (size_t)nvariants + 1);
    }
  }
  return result;
}


static int run_2d(const train_t* train, unsigned int seed)
{
  const size_t n = BRACKET_N2, nn = n * n, ni = BRACKET_IMAGE2;
  const size_t nbuffer = (size_t)BRACKET_HI2 * BRACKET_HI2;
  double *const work = (double*)malloc(sizeof(double) * (3 * nn + 2 * ni * ni + nbuffer));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    double *const clean = work, *const area = clean + nn, *const tmp = area + nn;
    double *const image = tmp + nn, *const model_image = image + ni * ni;
    double *const buffer = model_image + ni * ni;
    const double yrow = 0.5 * (checker_edge(1) + checker_edge(2));
    target1_t row;
    study_t study;
    train_t train2 = *train;
    tensor_t tensor;
    mlp_t mlp;
    libxs_timer_tick_t start;
    size_t i, j;
    int k;
    train2.nsteps = LIBXS_MAX(train->nsteps / 4, 1);
    train2.wd = 0;
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        clean[i + n * j] = checker_value((double)i / (double)(n - 1),
          (double)j / (double)(n - 1));
      }
    }
    for (j = 0; j < n; ++j) integrate1(clean + n * j, n, tmp + n * j);
    for (i = 0; i < n; ++i) {
      double s = 0;
      area[i] = 0;
      for (j = 1; j < n; ++j) {
        s += 0.5 * (tmp[i + n * (j - 1)] + tmp[i + n * j]) / (double)(n - 1);
        area[i + n * j] = s;
      }
    }
    for (j = 0; j < ni; ++j) {
      for (i = 0; i < ni; ++i) {
        image[i + ni * j] = checker_value((double)i / (double)(ni - 1),
          (double)j / (double)(ni - 1));
      }
    }
    checker_row(yrow, &row);
    memset(&study, 0, sizeof(study));
    study.ndims = 2;
    study.ntrain = n;
    study.nimage = ni;
    study.nlo = BRACKET_LO2;
    study.nhi = BRACKET_HI2;
    study.rowindex = (size_t)(yrow * (double)(ni - 1) + 0.5);
    study.clean = clean;
    study.image = image;
    study.buffer = buffer;
    study.row = &row;
    study.sumjsq = 0;
    for (k = 0; k < row.nedges; ++k) {
      const double jump = row.level[k + 1] - row.level[k];
      study.sumjsq += jump * jump;
    }
    result = write_mhd("bracket_2d_target", image, ni, ni);
    if (EXIT_SUCCESS == result) {
      libxs_rng_set_seed(seed);
      result = mlp_init(&mlp, BRACKET_H2);
      train2.lr = BRACKET_LRMLP;
      start = libxs_timer_tick();
      if (EXIT_SUCCESS == result) result = mlp_train(&mlp, clean, (int)n, &train2);
      if (EXIT_SUCCESS == result) {
        result = report(&study, "checker", "mlp", &mlp, mlp_eval,
          libxs_timer_duration(start, libxs_timer_tick()), model_image);
      }
      if (EXIT_SUCCESS == result) result = write_mhd("bracket_2d_mlp", model_image, ni, ni);
      free(mlp.param);
    }
    for (k = 0; k < 2 && EXIT_SUCCESS == result; ++k) {
      libxs_rng_set_seed(seed);
      result = tensor_init(&tensor, BRACKET_U2, k);
      train2.lr = BRACKET_LR2;
      start = libxs_timer_tick();
      if (EXIT_SUCCESS == result) {
        result = tensor_train(&tensor, 0 != k ? area : clean, (int)n, &train2);
      }
      if (EXIT_SUCCESS == result) {
        result = report(&study, "checker", 0 != k ? "tensor_int" : "tensor", &tensor,
          tensor_eval, libxs_timer_duration(start, libxs_timer_tick()), model_image);
      }
      if (EXIT_SUCCESS == result) {
        result = write_mhd(0 != k ? "bracket_2d_tensor_int" : "bracket_2d_tensor",
          model_image, ni, ni);
      }
      free(tensor.param);
    }
  }
  free(work);
  return result;
}
