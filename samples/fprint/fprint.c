#include <libxs_math.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FPRINT_PATH_MAX 1024
#define FPRINT_MAX_TERMS 8
#define FPRINT_MAX_DEGREE 8
#define FPRINT_ORDER 4
#define FPRINT_NREF 1024
#define FPRINT_PI 3.14159265358979323846

typedef enum sample_kind_t {
  SAMPLE_SIN = 0,
  SAMPLE_ARCH,
  SAMPLE_KINK,
  SAMPLE_OSC,
  SAMPLE_NOISE
} sample_kind_t;

typedef struct bracket_term_t {
  double coeff;
  double knot;
  int power;
} bracket_term_t;

typedef struct shape_t {
  const char* name;
  double poly[FPRINT_MAX_DEGREE + 1];
  int degree;
  bracket_term_t terms[FPRINT_MAX_TERMS];
  int nterms;
} shape_t;

static char output_dir[FPRINT_PATH_MAX] = { 0 };

static int make_path(char path[FPRINT_PATH_MAX], const char name[]);
static FILE* open_csv(const char name[]);
static double sample_value(sample_kind_t kind, double position);
static double noise_value(size_t index);
static int fill_sequence(double values[], size_t count, sample_kind_t kind);
static int fingerprint_sequence(libxs_fprint_t* info,
  const double values[], size_t count, int order);
static void write_fprint_values(FILE* file, const libxs_fprint_t* info);
static int run_convergence(void);
static int run_sensitivity(void);
static double shape_eval(const shape_t* shape, double position);
static void shape_piece_poly(const shape_t* shape, double position,
  double coeffs[FPRINT_MAX_DEGREE + 1]);
static double poly_integral(const double coeffs[], int degree,
  double left, double right, int xpower);
static void poly_square(const double coeffs[], int degree,
  double square[2 * FPRINT_MAX_DEGREE + 1]);
static void sort_bounds(double bounds[], int count);
static void shape_integrals(const shape_t* shape, double* area,
  double* moment_x, double* moment_y);
static int fill_shape(double values[], size_t count, const shape_t* shape);
static int run_geometry(void);
static double field_value(const char name[], size_t row, size_t col,
  size_t nrows, size_t ncols);
static int run_hierarchy(void);
static int run_compression(void);


int main(void)
{
  const char* const env_dir = getenv("FPRINT_OUTDIR");
  int result = EXIT_SUCCESS;
  if (NULL != env_dir && 0 != *env_dir && strlen(env_dir) < sizeof(output_dir)) {
    strcpy(output_dir, env_dir);
  }
  printf("FPRINT: Foeppl polynomial fingerprint experiments\n");
  printf("  output: %s\n", 0 != output_dir[0] ? output_dir : "stdout only");
  if (EXIT_SUCCESS == result) result = run_convergence();
  if (EXIT_SUCCESS == result) result = run_sensitivity();
  if (EXIT_SUCCESS == result) result = run_geometry();
  if (EXIT_SUCCESS == result) result = run_hierarchy();
  if (EXIT_SUCCESS == result) result = run_compression();
  if (EXIT_SUCCESS == result) printf("FPRINT: done\n");
  return result;
}


static int make_path(char path[FPRINT_PATH_MAX], const char name[])
{
  int result = EXIT_FAILURE;
  if (0 != output_dir[0] && NULL != name) {
    const size_t dir_len = strlen(output_dir);
    const size_t name_len = strlen(name);
    if (dir_len + name_len + 2 < FPRINT_PATH_MAX) {
      strcpy(path, output_dir);
      if ('/' != path[dir_len - 1]) strcat(path, "/");
      strcat(path, name);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static FILE* open_csv(const char name[])
{
  FILE* result = NULL;
  char path[FPRINT_PATH_MAX];
  if (EXIT_SUCCESS == make_path(path, name)) result = fopen(path, "w");
  return result;
}


static double sample_value(sample_kind_t kind, double position)
{
  double result = 0.0;
  if (SAMPLE_SIN == kind) {
    result = sin(2.0 * FPRINT_PI * position);
  }
  else if (SAMPLE_ARCH == kind) {
    result = 4.0 * position * (1.0 - position);
  }
  else if (SAMPLE_KINK == kind) {
    result = 4.0 * position * (1.0 - position)
      + (0.45 * LIBXS_MAX(0.0, position - 0.5));
  }
  else if (SAMPLE_OSC == kind) {
    result = sin(2.0 * FPRINT_PI * position)
      + 0.08 * sin(30.0 * FPRINT_PI * position);
  }
  return result;
}


static double noise_value(size_t index)
{
  unsigned int state = (unsigned int)(1103515245u * (unsigned int)(index + 1u)
    + 12345u);
  state = 1103515245u * state + 12345u;
  return ((double)(state & 65535u) / 32767.5) - 1.0;
}


static int fill_sequence(double values[], size_t count, sample_kind_t kind)
{
  int result = EXIT_FAILURE;
  if (NULL != values && 1 < count) {
    size_t index;
    for (index = 0; index < count; ++index) {
      const double position = (double)index / (double)(count - 1);
      values[index] = (SAMPLE_NOISE == kind)
        ? noise_value(index) : sample_value(kind, position);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


static int fingerprint_sequence(libxs_fprint_t* info,
  const double values[], size_t count, int order)
{
  const size_t stride = 1;
  int result = EXIT_FAILURE;
  if (NULL != info && NULL != values) {
    result = libxs_fprint(info, LIBXS_DATATYPE_F64, values,
      1, &count, &stride, order, -1);
  }
  return result;
}


static void write_fprint_values(FILE* file, const libxs_fprint_t* info)
{
  int order;
  for (order = 0; order <= FPRINT_ORDER; ++order) {
    const double value = (NULL != info && order <= info->order)
      ? info->l2[order] : 0.0;
    fprintf(file, ",%.17g", value);
  }
}


static int run_convergence(void)
{
  static const size_t sizes[] = { 16, 32, 64, 128, 256, 512 };
  static const sample_kind_t kinds[] = { SAMPLE_SIN, SAMPLE_ARCH, SAMPLE_KINK };
  static const char* const names[] = { "sin", "arch", "kink" };
  FILE* file = open_csv("convergence.csv");
  int result = (NULL != file) ? EXIT_SUCCESS : EXIT_FAILURE;
  double* reference = NULL;
  double* values = NULL;
  if (EXIT_SUCCESS == result) {
    fprintf(file, "case,n,distance,decay,l2_0,l2_1,l2_2,l2_3,l2_4\n");
    reference = (double*)malloc(FPRINT_NREF * sizeof(double));
  }
  if (EXIT_SUCCESS == result && NULL == reference) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    size_t kind_index;
    for (kind_index = 0; kind_index < sizeof(kinds) / sizeof(*kinds); ++kind_index) {
      libxs_fprint_t fp_ref;
      result = fill_sequence(reference, FPRINT_NREF, kinds[kind_index]);
      if (EXIT_SUCCESS == result) {
        result = fingerprint_sequence(&fp_ref, reference, FPRINT_NREF, FPRINT_ORDER);
      }
      if (EXIT_SUCCESS == result) {
        size_t size_index;
        for (size_index = 0; size_index < sizeof(sizes) / sizeof(*sizes)
          && EXIT_SUCCESS == result; ++size_index)
        {
          libxs_fprint_t fp;
          values = (double*)malloc(sizes[size_index] * sizeof(double));
          if (NULL == values) result = EXIT_FAILURE;
          if (EXIT_SUCCESS == result) {
            result = fill_sequence(values, sizes[size_index], kinds[kind_index]);
          }
          if (EXIT_SUCCESS == result) {
            result = fingerprint_sequence(&fp, values, sizes[size_index], FPRINT_ORDER);
          }
          if (EXIT_SUCCESS == result) {
            fprintf(file, "%s,%i,%.17g,%.17g", names[kind_index],
              (int)sizes[size_index], libxs_fprint_diff(&fp, &fp_ref, NULL),
              libxs_fprint_decay(&fp));
            write_fprint_values(file, &fp);
            fprintf(file, "\n");
          }
          free(values);
          values = NULL;
        }
      }
    }
  }
  free(values);
  free(reference);
  if (NULL != file) fclose(file);
  printf("  convergence.csv\n");
  return result;
}


static int run_sensitivity(void)
{
  static const char* const names[] = { "smooth_drift", "oscillation",
    "kink", "local_spike" };
  const size_t count = 256;
  const double target_rms = 0.04;
  FILE* file = open_csv("sensitivity.csv");
  int result = (NULL != file) ? EXIT_SUCCESS : EXIT_FAILURE;
  double* base = NULL;
  double* trial = NULL;
  if (EXIT_SUCCESS == result) {
    base = (double*)malloc(count * sizeof(double));
    trial = (double*)malloc(count * sizeof(double));
    fprintf(file, "case,element_rms,fprint_distance,decay,l2_0,l2_1,l2_2,l2_3,l2_4\n");
  }
  if (EXIT_SUCCESS == result && (NULL == base || NULL == trial)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = fill_sequence(base, count, SAMPLE_ARCH);
  if (EXIT_SUCCESS == result) {
    libxs_fprint_t fp_base;
    size_t variant;
    result = fingerprint_sequence(&fp_base, base, count, FPRINT_ORDER);
    for (variant = 0; variant < 4 && EXIT_SUCCESS == result; ++variant) {
      double norm = 0.0;
      double rms = 0.0;
      size_t index;
      for (index = 0; index < count; ++index) {
        const double position = (double)index / (double)(count - 1);
        double perturb = 0.0;
        if (0 == variant) perturb = position - 0.5;
        else if (1 == variant) perturb = sin(24.0 * FPRINT_PI * position);
        else if (2 == variant) perturb = LIBXS_MAX(0.0, position - 0.5);
        else if (position > 0.46 && position < 0.54) perturb = 1.0;
        norm += perturb * perturb;
        trial[index] = perturb;
      }
      norm = sqrt(norm / (double)count);
      for (index = 0; index < count; ++index) {
        trial[index] = base[index] + target_rms * trial[index] / norm;
        rms += (trial[index] - base[index]) * (trial[index] - base[index]);
      }
      if (EXIT_SUCCESS == result) {
        libxs_fprint_t fp_trial;
        result = fingerprint_sequence(&fp_trial, trial, count, FPRINT_ORDER);
        if (EXIT_SUCCESS == result) {
          fprintf(file, "%s,%.17g,%.17g,%.17g", names[variant],
            sqrt(rms / (double)count), libxs_fprint_diff(&fp_base, &fp_trial, NULL),
            libxs_fprint_decay(&fp_trial));
          write_fprint_values(file, &fp_trial);
          fprintf(file, "\n");
        }
      }
    }
  }
  free(trial);
  free(base);
  if (NULL != file) fclose(file);
  printf("  sensitivity.csv\n");
  return result;
}


static double shape_eval(const shape_t* shape, double position)
{
  double result = 0.0;
  if (NULL != shape) {
    double power_value = 1.0;
    int degree;
    int term;
    for (degree = 0; degree <= shape->degree; ++degree) {
      result += shape->poly[degree] * power_value;
      power_value *= position;
    }
    for (term = 0; term < shape->nterms; ++term) {
      if (position >= shape->terms[term].knot) {
        result += shape->terms[term].coeff
          * pow(position - shape->terms[term].knot, shape->terms[term].power);
      }
    }
  }
  return result;
}


static void shape_piece_poly(const shape_t* shape, double position,
  double coeffs[FPRINT_MAX_DEGREE + 1])
{
  int degree;
  int term;
  for (degree = 0; degree <= FPRINT_MAX_DEGREE; ++degree) coeffs[degree] = 0.0;
  if (NULL != shape) {
    for (degree = 0; degree <= shape->degree; ++degree) coeffs[degree] = shape->poly[degree];
    for (term = 0; term < shape->nterms; ++term) {
      const bracket_term_t* const current = shape->terms + term;
      if (position >= current->knot) {
        if (0 == current->power) coeffs[0] += current->coeff;
        else if (1 == current->power) {
          coeffs[0] -= current->coeff * current->knot;
          coeffs[1] += current->coeff;
        }
        else if (2 == current->power) {
          coeffs[0] += current->coeff * current->knot * current->knot;
          coeffs[1] -= 2.0 * current->coeff * current->knot;
          coeffs[2] += current->coeff;
        }
      }
    }
  }
}


static double poly_integral(const double coeffs[], int degree,
  double left, double right, int xpower)
{
  double result = 0.0;
  int poly_degree;
  for (poly_degree = 0; poly_degree <= degree; ++poly_degree) {
    const int exponent = poly_degree + xpower + 1;
    result += coeffs[poly_degree]
      * (pow(right, exponent) - pow(left, exponent)) / exponent;
  }
  return result;
}


static void poly_square(const double coeffs[], int degree,
  double square[2 * FPRINT_MAX_DEGREE + 1])
{
  int index;
  int left_degree;
  for (index = 0; index <= 2 * FPRINT_MAX_DEGREE; ++index) square[index] = 0.0;
  for (left_degree = 0; left_degree <= degree; ++left_degree) {
    int right_degree;
    for (right_degree = 0; right_degree <= degree; ++right_degree) {
      square[left_degree + right_degree] += coeffs[left_degree] * coeffs[right_degree];
    }
  }
}


static void sort_bounds(double bounds[], int count)
{
  int pass;
  for (pass = 0; pass + 1 < count; ++pass) {
    int index;
    for (index = 0; index + 1 < count - pass; ++index) {
      if (bounds[index + 1] < bounds[index]) {
        const double swap = bounds[index];
        bounds[index] = bounds[index + 1];
        bounds[index + 1] = swap;
      }
    }
  }
}


static void shape_integrals(const shape_t* shape, double* area,
  double* moment_x, double* moment_y)
{
  double bounds[FPRINT_MAX_TERMS + 2];
  int nbounds = 2;
  int term;
  int bound;
  *area = 0.0;
  *moment_x = 0.0;
  *moment_y = 0.0;
  bounds[0] = 0.0;
  bounds[1] = 1.0;
  for (term = 0; NULL != shape && term < shape->nterms; ++term) {
    if (0.0 < shape->terms[term].knot && shape->terms[term].knot < 1.0) {
      bounds[nbounds++] = shape->terms[term].knot;
    }
  }
  sort_bounds(bounds, nbounds);
  for (bound = 0; bound + 1 < nbounds; ++bound) {
    const double left = bounds[bound];
    const double right = bounds[bound + 1];
    const double mid = 0.5 * (left + right);
    double coeffs[FPRINT_MAX_DEGREE + 1];
    double square[2 * FPRINT_MAX_DEGREE + 1];
    shape_piece_poly(shape, mid, coeffs);
    poly_square(coeffs, FPRINT_MAX_DEGREE, square);
    *area += poly_integral(coeffs, FPRINT_MAX_DEGREE, left, right, 0);
    *moment_y += poly_integral(coeffs, FPRINT_MAX_DEGREE, left, right, 1);
    *moment_x += 0.5 * poly_integral(square, 2 * FPRINT_MAX_DEGREE, left, right, 0);
  }
}


static int fill_shape(double values[], size_t count, const shape_t* shape)
{
  int result = EXIT_FAILURE;
  if (NULL != values && NULL != shape && 1 < count) {
    size_t index;
    for (index = 0; index < count; ++index) {
      values[index] = shape_eval(shape, (double)index / (double)(count - 1));
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


static int run_geometry(void)
{
  shape_t shapes[4];
  const size_t count = 256;
  FILE* file = open_csv("geometry.csv");
  int result = (NULL != file) ? EXIT_SUCCESS : EXIT_FAILURE;
  double* values = NULL;
  memset(shapes, 0, sizeof(shapes));
  shapes[0].name = "smooth_arch";
  shapes[0].degree = 2;
  shapes[0].poly[1] = 4.0;
  shapes[0].poly[2] = -4.0;
  shapes[1] = shapes[0];
  shapes[1].name = "kinked_arch";
  shapes[1].terms[0].coeff = 0.45;
  shapes[1].terms[0].knot = 0.5;
  shapes[1].terms[0].power = 1;
  shapes[1].nterms = 1;
  shapes[2] = shapes[0];
  shapes[2].name = "notched_arch";
  shapes[2].terms[0].coeff = -1.1;
  shapes[2].terms[0].knot = 0.38;
  shapes[2].terms[0].power = 1;
  shapes[2].terms[1].coeff = 2.2;
  shapes[2].terms[1].knot = 0.50;
  shapes[2].terms[1].power = 1;
  shapes[2].terms[2].coeff = -1.1;
  shapes[2].terms[2].knot = 0.62;
  shapes[2].terms[2].power = 1;
  shapes[2].nterms = 3;
  shapes[3].name = "roof";
  shapes[3].degree = 1;
  shapes[3].poly[1] = 2.0;
  shapes[3].terms[0].coeff = -4.0;
  shapes[3].terms[0].knot = 0.5;
  shapes[3].terms[0].power = 1;
  shapes[3].nterms = 1;
  if (EXIT_SUCCESS == result) {
    values = (double*)malloc(count * sizeof(double));
    fprintf(file, "shape,area,centroid_x,centroid_y,decay,l2_0,l2_1,l2_2,l2_3,l2_4\n");
  }
  if (EXIT_SUCCESS == result && NULL == values) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    int shape_index;
    for (shape_index = 0; shape_index < 4 && EXIT_SUCCESS == result; ++shape_index) {
      double area;
      double moment_x;
      double moment_y;
      libxs_fprint_t fp;
      shape_integrals(shapes + shape_index, &area, &moment_x, &moment_y);
      result = fill_shape(values, count, shapes + shape_index);
      if (EXIT_SUCCESS == result) result = fingerprint_sequence(&fp, values, count, FPRINT_ORDER);
      if (EXIT_SUCCESS == result) {
        fprintf(file, "%s,%.17g,%.17g,%.17g,%.17g", shapes[shape_index].name,
          area, moment_y / area, moment_x / area, libxs_fprint_decay(&fp));
        write_fprint_values(file, &fp);
        fprintf(file, "\n");
      }
    }
  }
  free(values);
  if (NULL != file) fclose(file);
  printf("  geometry.csv\n");
  return result;
}


static double field_value(const char name[], size_t row, size_t col,
  size_t nrows, size_t ncols)
{
  const double xpos = (double)col / (double)(ncols - 1);
  const double ypos = (double)row / (double)(nrows - 1);
  double result = sin(FPRINT_PI * xpos) * cos(FPRINT_PI * ypos);
  if (0 == strcmp(name, "x_crease")) {
    result += 0.4 * LIBXS_MAX(0.0, xpos - 0.5);
  }
  else if (0 == strcmp(name, "diag_crease")) {
    result += 0.4 * LIBXS_MAX(0.0, xpos + ypos - 1.0);
  }
  else if (0 == strcmp(name, "noise")) {
    result = noise_value(row * ncols + col);
  }
  return result;
}


static int run_hierarchy(void)
{
  static const char* const names[] = { "smooth", "x_crease", "diag_crease", "noise" };
  const size_t shape[2] = { 64, 64 };
  FILE* file = open_csv("hierarchy.csv");
  int result = (NULL != file) ? EXIT_SUCCESS : EXIT_FAILURE;
  double* values = NULL;
  if (EXIT_SUCCESS == result) {
    values = (double*)malloc(shape[0] * shape[1] * sizeof(double));
    fprintf(file, "field,mode,decay,l2_0,l2_1,l2_2,l2_3,l2_4\n");
  }
  if (EXIT_SUCCESS == result && NULL == values) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    size_t field_index;
    for (field_index = 0; field_index < sizeof(names) / sizeof(*names)
      && EXIT_SUCCESS == result; ++field_index)
    {
      size_t row;
      for (row = 0; row < shape[1]; ++row) {
        size_t col;
        for (col = 0; col < shape[0]; ++col) {
          values[row * shape[0] + col] = field_value(names[field_index], row, col,
            shape[1], shape[0]);
        }
      }
      {
        static const int axes[] = { -1, 0, 1 };
        static const char* const modes[] = { "hierarchical", "axis_x", "axis_y" };
        size_t mode;
        for (mode = 0; mode < sizeof(axes) / sizeof(*axes) && EXIT_SUCCESS == result; ++mode) {
          libxs_fprint_t fp;
          result = libxs_fprint(&fp, LIBXS_DATATYPE_F64, values, 2, shape, NULL,
            FPRINT_ORDER, axes[mode]);
          if (EXIT_SUCCESS == result) {
            fprintf(file, "%s,%s,%.17g", names[field_index], modes[mode],
              libxs_fprint_decay(&fp));
            write_fprint_values(file, &fp);
            fprintf(file, "\n");
          }
        }
      }
    }
  }
  free(values);
  if (NULL != file) fclose(file);
  printf("  hierarchy.csv\n");
  return result;
}


static int run_compression(void)
{
  static const sample_kind_t kinds[] = { SAMPLE_ARCH, SAMPLE_SIN,
    SAMPLE_KINK, SAMPLE_NOISE };
  static const char* const names[] = { "arch", "sin", "kink", "noise" };
  const size_t count = 128;
  FILE* file = open_csv("compression.csv");
  int result = (NULL != file) ? EXIT_SUCCESS : EXIT_FAILURE;
  double* values = NULL;
  double* diffs = NULL;
  if (EXIT_SUCCESS == result) {
    values = (double*)malloc(count * sizeof(double));
    diffs = (double*)malloc(count * sizeof(double));
    fprintf(file, "case,order,max_error,rms_error,raw_coeff\n");
  }
  if (EXIT_SUCCESS == result && (NULL == values || NULL == diffs)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    size_t kind_index;
    for (kind_index = 0; kind_index < sizeof(kinds) / sizeof(*kinds)
      && EXIT_SUCCESS == result; ++kind_index)
    {
      size_t nactive;
      result = fill_sequence(values, count, kinds[kind_index]);
      if (EXIT_SUCCESS == result) memcpy(diffs, values, count * sizeof(double));
      for (nactive = count; 1 < nactive; --nactive) {
        size_t index;
        for (index = 0; index + 1 < nactive; ++index) {
          diffs[index] = diffs[index + 1] - diffs[index];
        }
      }
      if (EXIT_SUCCESS == result) {
        int order;
        double coeffs[FPRINT_ORDER + 1];
        memcpy(diffs, values, count * sizeof(double));
        coeffs[0] = diffs[0];
        for (order = 1; order <= FPRINT_ORDER; ++order) {
          size_t index;
          for (index = 0; index + (size_t)order < count; ++index) {
            diffs[index] = diffs[index + 1] - diffs[index];
          }
          coeffs[order] = diffs[0];
        }
        for (order = 0; order <= FPRINT_ORDER; ++order) {
          double max_error = 0.0;
          double rms_error = 0.0;
          size_t index;
          for (index = 0; index < count; ++index) {
            double approx = 0.0;
            int coeff_index;
            for (coeff_index = 0; coeff_index <= order; ++coeff_index) {
              approx += libxs_binom((double)index, coeff_index) * coeffs[coeff_index];
            }
            {
              const double error = fabs(approx - values[index]);
              if (max_error < error) max_error = error;
              rms_error += error * error;
            }
          }
          fprintf(file, "%s,%i,%.17g,%.17g,%.17g\n", names[kind_index], order,
            max_error, sqrt(rms_error / (double)count), coeffs[order]);
        }
      }
    }
  }
  free(diffs);
  free(values);
  if (NULL != file) fclose(file);
  printf("  compression.csv\n");
  return result;
}