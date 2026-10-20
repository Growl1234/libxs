/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_pack.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_timer.h>
#include <libxs/libxs_utils.h>
#include <float.h>
#include <math.h>

#define BLOCK 512
#define NGROUPS 4
#define MAXBITS 64
#define WIDTH_HEADER 6 /* bits per block that record its width (CP2K: maxval stream) */


typedef struct stream_t {
  int64_t* q;
  size_t n, cap;
} stream_t;

typedef struct stats_t {
  stream_t plain[MAXBITS + 1], delta[MAXBITS + 1], twice[MAXBITS + 1];
  size_t nblocks, nvalues, nzeros, nskipped;
  double bits_fixed, bits_twice, bits_needed, eps;
  int has_twice;
} stats_t;


typedef struct header_t {
  double payload, byte, seven, base3, base2, nwidths, hist[MAXBITS + 1];
} header_t;


static const int groups[NGROUPS] = { 4, 8, 16, 32 };
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;


static uint64_t rng_next(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}


static int stream_push(stream_t* s, int64_t v)
{
  int result = EXIT_SUCCESS;
  if (s->n == s->cap) {
    const size_t cap = (0 != s->cap ? 2 * s->cap : 1024);
    int64_t *const q = (int64_t*)realloc(s->q, cap * sizeof(int64_t));
    if (NULL != q) {
      s->q = q;
      s->cap = cap;
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) s->q[s->n++] = v;
  return result;
}


/* Width CP2K derives for a block (qs_tensors.F: compress_tensor) at the given step. */
static int cp2k_nbits(double maxabs, double eps)
{
  int e, r;
  frexp(0 < maxabs ? maxabs : DBL_MIN, &e);
  if (-15 > e) e = -15;
  frexp(round(ldexp(1.0, e) / eps), &r);
  return r + 1;
}


static int64_t cp2k_quantize(double x, double eps, double step)
{
  return (fabs(x) > eps) ? (int64_t)llround(x / step) : 0;
}


static int needed_bits(int64_t q)
{
  const uint64_t z = ((uint64_t)q << 1) ^ (0 - ((uint64_t)q >> 63));
  return (int)LIBXS_NBITS(z);
}


/** Quantizes a block as CP2K does and appends it to the streams of its width. */
static int add_block(stats_t* st, const double* x, size_t n, double eps)
{
  double maxabs = 0;
  int result = EXIT_SUCCESS, nbits, nbits2;
  int64_t prev = 0;
  size_t i;
  for (i = 0; i < n; ++i) maxabs = LIBXS_MAX(maxabs, fabs(x[i]));
  nbits = cp2k_nbits(maxabs, eps);
  nbits2 = cp2k_nbits(maxabs, 2 * eps);
  if (MAXBITS <= nbits) ++st->nskipped; /* CP2K aborts on this */
  else {
    ++st->nblocks;
    st->nvalues += n;
    st->bits_fixed += (double)n * nbits + WIDTH_HEADER;
    st->bits_twice += (double)n * nbits2 + WIDTH_HEADER;
    for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
      const int64_t q = cp2k_quantize(x[i], eps, eps);
      if (0 == q) ++st->nzeros;
      st->bits_needed += needed_bits(q);
      result = stream_push(&st->plain[nbits], q);
      if (EXIT_SUCCESS == result) result = stream_push(&st->delta[nbits + 1], q - prev);
      if (EXIT_SUCCESS == result) result = stream_push(&st->twice[nbits2], cp2k_quantize(x[i], eps, 2 * eps));
      prev = q;
    }
  }
  return result;
}


/** Integer values as they are: the fixed width is what the largest magnitude needs. */
static int add_block_int(stats_t* st, const int64_t* q, size_t n)
{
  int result = EXIT_SUCCESS, nbits = 1;
  int64_t prev = 0;
  size_t i;
  for (i = 0; i < n; ++i) nbits = LIBXS_MAX(nbits, needed_bits(q[i]));
  ++st->nblocks;
  st->nvalues += n;
  st->bits_fixed += (double)n * nbits + WIDTH_HEADER;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    if (0 == q[i]) ++st->nzeros;
    st->bits_needed += needed_bits(q[i]);
    result = stream_push(&st->plain[nbits], q[i]);
    if (EXIT_SUCCESS == result) result = stream_push(&st->delta[nbits + 1], q[i] - prev);
    prev = q[i];
  }
  return result;
}


/** Adds an MHD file: CP2K's block table and eps from the extension, otherwise blocks of BLOCK. */
static int add_file(stats_t* st, const char* filename, double eps)
{
  char datafile[4096];
  size_t size[8], extsize = 0, n = 1, i;
  libxs_mhd_info_t info;
  char* ext = NULL;
  void* data = NULL;
  int result;
  info.ndims = sizeof(size) / sizeof(*size);
  result = libxs_mhd_read_header(filename, sizeof(datafile), datafile, &info, size, NULL, &extsize);
  if (EXIT_SUCCESS == result && 0 != extsize) {
    ext = (char*)malloc(extsize);
    info.ndims = sizeof(size) / sizeof(*size);
    result = (NULL != ext ? libxs_mhd_read_header(filename, sizeof(datafile), datafile,
      &info, size, ext, &extsize) : EXIT_FAILURE);
  }
  for (i = 0; i < info.ndims; ++i) n *= size[i];
  n *= info.ncomponents;
  if (EXIT_SUCCESS == result) {
    data = malloc(n * LIBXS_TYPESIZE(info.type));
    result = (NULL != data ? libxs_mhd_read(datafile, NULL, size, NULL, &info, data,
      NULL, NULL, NULL) : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) {
    const int is_float = (LIBXS_DATATYPE_F64 == info.type || LIBXS_DATATYPE_F32 == info.type);
    int32_t nblk = 0;
    double* x = NULL;
    int64_t* q = NULL;
    if (12 <= extsize) { /* CP2K: REAL(8) eps, INT32 number of blocks, INT32 shape(3) per block */
      memcpy(&nblk, ext + 8, sizeof(nblk));
      if (0 < nblk && (size_t)extsize == 12 + 12 * (size_t)nblk) memcpy(&eps, ext, sizeof(eps));
      else nblk = 0;
    }
    if (0 != is_float) {
      x = (double*)malloc(n * sizeof(double));
      if (NULL != x) {
        for (i = 0; i < n; ++i) {
          x[i] = (LIBXS_DATATYPE_F64 == info.type ? ((const double*)data)[i] : ((const float*)data)[i]);
        }
      }
      if (0 >= eps) {
        fprintf(stderr, "ERROR: %s holds floating-point data, an eps is needed\n", filename);
        result = EXIT_FAILURE;
      }
      st->has_twice = 1;
      st->eps = eps;
    }
    else {
      q = (int64_t*)malloc(n * sizeof(int64_t));
      if (NULL != q) {
        for (i = 0; i < n; ++i) {
          switch (info.type) {
            case LIBXS_DATATYPE_I8: q[i] = ((const int8_t*)data)[i]; break;
            case LIBXS_DATATYPE_U8: q[i] = ((const uint8_t*)data)[i]; break;
            case LIBXS_DATATYPE_I16: q[i] = ((const int16_t*)data)[i]; break;
            case LIBXS_DATATYPE_U16: q[i] = ((const uint16_t*)data)[i]; break;
            case LIBXS_DATATYPE_I32: q[i] = ((const int32_t*)data)[i]; break;
            case LIBXS_DATATYPE_U32: q[i] = ((const uint32_t*)data)[i]; break;
            default: q[i] = ((const int64_t*)data)[i];
          }
        }
      }
    }
    if (EXIT_SUCCESS == result && (NULL != x || NULL != q)) {
      size_t offset = 0, iblk = 0;
      while (offset < n && EXIT_SUCCESS == result) {
        size_t m = LIBXS_MIN((size_t)BLOCK, n - offset);
        if (0 < nblk && iblk < (size_t)nblk) {
          int32_t shape[3];
          memcpy(shape, ext + 12 + 12 * iblk, sizeof(shape));
          m = LIBXS_MIN((size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2], n - offset);
        }
        result = (NULL != x ? add_block(st, x + offset, m, eps) : add_block_int(st, q + offset, m));
        offset += m;
        ++iblk;
      }
    }
    else if (EXIT_SUCCESS == result) result = EXIT_FAILURE;
    free(x);
    free(q);
  }
  else fprintf(stderr, "ERROR: cannot read %s as MHD file\n", filename);
  free(ext);
  free(data);
  return result;
}


/* Header costs of the widths: bytes, 7 bits, or a base per 8 groups with 3- or 2-bit offsets. */
static void header_costs(header_t* h, const unsigned char* widths, size_t ngroups)
{
  size_t i, j;
  for (i = 0; i < ngroups; i += 8) {
    const size_t m = LIBXS_MIN((size_t)8, ngroups - i);
    int wmin = MAXBITS, wmax = 0;
    for (j = 0; j < m; ++j) {
      wmin = LIBXS_MIN(wmin, (int)widths[i + j]);
      wmax = LIBXS_MAX(wmax, (int)widths[i + j]);
      h->hist[widths[i + j]] += 1;
    }
    h->base3 += 1.0 + (7 >= wmax - wmin ? 7.0 + 3.0 * m : 7.0 * m); /* flag selects the scheme */
    h->base2 += 1.0 + (3 >= wmax - wmin ? 7.0 + 2.0 * m : 7.0 * m);
  }
  h->byte += 8.0 * (1 + ngroups);
  h->seven += 7.0 * ngroups;
  h->nwidths += (double)ngroups;
}


/** Packs every stream at its width (one stream per width, as CP2K's containers). */
static double vpack_bits(const stream_t* s, int group, double* seconds, double* useconds, int* ok,
  header_t* h)
{
  double bits = 0;
  int nbits;
  *seconds = *useconds = 0;
  for (nbits = 1; nbits <= MAXBITS; ++nbits) {
    if (0 != s[nbits].n) {
      const size_t bound = libxs_vpack_bound(s[nbits].n, nbits, group);
      uint64_t *const packed = (uint64_t*)malloc(bound * sizeof(uint64_t));
      int64_t *const back = (int64_t*)malloc(s[nbits].n * sizeof(int64_t));
      if (NULL != packed && NULL != back) {
        const libxs_timer_tick_t t0 = libxs_timer_tick();
        const size_t nwords = libxs_vpack(packed, s[nbits].q, LIBXS_DATATYPE_I64, s[nbits].n, nbits, group);
        const libxs_timer_tick_t t1 = libxs_timer_tick();
        const size_t nread = libxs_vunpack(back, packed, nwords, LIBXS_DATATYPE_I64, s[nbits].n);
        *useconds += libxs_timer_duration(t1, libxs_timer_tick());
        *seconds += libxs_timer_duration(t0, t1);
        if (nread != nwords || 0 != memcmp(back, s[nbits].q, s[nbits].n * sizeof(int64_t))) *ok = 0;
        bits += 64.0 * (double)nwords;
        if (NULL != h) {
          const size_t ngroups = (s[nbits].n + group - 1) / group;
          h->payload += 64.0 * ((double)nwords - (double)((1 + ngroups + 7) / 8));
          header_costs(h, (const unsigned char*)packed + 1, ngroups);
        }
      }
      else *ok = 0;
      free(packed);
      free(back);
    }
  }
  return bits;
}


static int run_files(int nfiles, char* files[], double eps)
{
  stats_t st;
  int result = EXIT_SUCCESS, i;
  memset(&st, 0, sizeof(st));
  for (i = 0; i < nfiles && EXIT_SUCCESS == result; ++i) result = add_file(&st, files[i], eps);
  if (EXIT_SUCCESS == result && 0 != st.nvalues) {
    const double hdr = (double)st.nblocks * WIDTH_HEADER, n = (double)st.nvalues;
    header_t hd[NGROUPS];
    double seconds, useconds, bits;
    int ok = 1, ig;
    printf("files %i  blocks %lu  values %lu  zeros %.1f%%  eps %g  skipped %lu\n", nfiles,
      (unsigned long)st.nblocks, (unsigned long)st.nvalues, 100.0 * st.nzeros / n, st.eps,
      (unsigned long)st.nskipped);
    printf("%-26s %10s %8s %14s %14s\n", "variant", "bits/value", "ratio", "pack ns/value", "unpack ns/value");
    printf("%-26s %10.2f %8.2f\n", "fixed", st.bits_fixed / n, 1.0);
    for (ig = 0; ig < NGROUPS; ++ig) {
      char label[32];
      memset(&hd[ig], 0, sizeof(hd[ig]));
      bits = vpack_bits(st.plain, groups[ig], &seconds, &useconds, &ok, &hd[ig]) + hdr;
      LIBXS_SNPRINTF(label, sizeof(label), "vpack group %i", groups[ig]);
      printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", label, bits / n, st.bits_fixed / bits,
        1E9 * seconds / n, 1E9 * useconds / n);
    }
    bits = vpack_bits(st.delta, LIBXS_VPACK_GROUP, &seconds, &useconds, &ok, NULL) + hdr;
    printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", "vpack deltas", bits / n, st.bits_fixed / bits,
      1E9 * seconds / n, 1E9 * useconds / n);
    if (0 != st.has_twice) {
      printf("%-26s %10.2f %8.2f\n", "fixed, step 2*eps", st.bits_twice / n, st.bits_fixed / st.bits_twice);
      bits = vpack_bits(st.twice, LIBXS_VPACK_GROUP, &seconds, &useconds, &ok, NULL) + hdr;
      printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", "vpack, step 2*eps", bits / n, st.bits_fixed / bits,
        1E9 * seconds / n, 1E9 * useconds / n);
    }
    printf("%-26s %10.2f %8.2f\n", "needed per value (bound)", (st.bits_needed + hdr) / n,
      st.bits_fixed / (st.bits_needed + hdr));
    printf("\n%-8s %9s %26s\n", "group", "payload", "+ header: bytes 7-bit base+3 base+2 entropy");
    for (ig = 0; ig < NGROUPS; ++ig) {
      const double nw = LIBXS_MAX(1.0, hd[ig].nwidths), pay = hd[ig].payload + hdr;
      double entropy = 0;
      int w;
      for (w = 0; w <= MAXBITS; ++w) {
        if (0 < hd[ig].hist[w]) entropy -= hd[ig].hist[w] * log2(hd[ig].hist[w] / nw);
      }
      printf("%-8i %9.2f %10.2f %6.2f %6.2f %6.2f %7.2f   bits/value\n", groups[ig], pay / n,
        (pay + hd[ig].byte) / n, (pay + hd[ig].seven) / n, (pay + hd[ig].base3) / n,
        (pay + hd[ig].base2) / n, (pay + entropy) / n);
      printf("%-8s %9s %10.2f %6.2f %6.2f %6.2f %7.2f   ratio\n", "", "",
        st.bits_fixed / (pay + hd[ig].byte), st.bits_fixed / (pay + hd[ig].seven),
        st.bits_fixed / (pay + hd[ig].base3), st.bits_fixed / (pay + hd[ig].base2),
        st.bits_fixed / (pay + entropy));
    }
    if (0 == ok) {
      fprintf(stderr, "ERROR: a round trip failed\n");
      result = EXIT_FAILURE;
    }
  }
  for (i = 0; i <= MAXBITS; ++i) {
    free(st.plain[i].q);
    free(st.delta[i].q);
    free(st.twice[i].q);
  }
  return result;
}


/* Signed values of up to nbits bits: per block, magnitudes log-uniform over range bits. */
static void synthesize(int64_t* q, size_t n, int nbits, int range, int zeros)
{
  size_t i;
  for (i = 0; i < n; ++i) {
    const int top = nbits - 1 - (int)((i / BLOCK) % (uint64_t)LIBXS_MAX(1, nbits / 4));
    const int drop = (int)(rng_next() % (uint64_t)LIBXS_MAX(1, range)); /* LIBXS_MAX evaluates twice */
    const int e = LIBXS_MAX(0, top - drop);
    const int64_t v = (int64_t)(rng_next() & ((((uint64_t)1) << e) - 1));
    q[i] = ((int)(rng_next() % 100) < zeros) ? 0 : (0 != (1 & rng_next()) ? -v : v);
  }
}


static int run_synthetic(size_t n, int nbits, int range, int zeros, int nrepeat)
{
  const size_t nfixed = (n * (size_t)nbits + 63) / 64;
  const size_t bound = libxs_vpack_bound(n, nbits, 1);
  int64_t *const q = (int64_t*)malloc(n * sizeof(int64_t));
  int64_t *const back = (int64_t*)malloc(n * sizeof(int64_t));
  uint64_t *const packed = (uint64_t*)malloc(LIBXS_MAX(bound, nfixed + 1) * sizeof(uint64_t));
  int result = EXIT_FAILURE;
  if (NULL != q && NULL != back && NULL != packed) {
    const uint64_t offset = ((uint64_t)1) << (nbits - 1);
    double dpack = 0, dunpack = 0;
    size_t i, nwords = 0;
    int r, ig;
    result = EXIT_SUCCESS;
    synthesize(q, n, nbits, range, zeros);
    printf("values %lu  nbits %i  range %i  zeros %i%%\n", (unsigned long)n, nbits, range, zeros);
    printf("%-14s %10s %8s %14s %14s\n", "variant", "bits/value", "ratio", "pack ns/value", "unpack ns/value");
    for (i = 0; i < n; ++i) back[i] = (int64_t)((uint64_t)q[i] + offset); /* CP2K: offset binary */
    for (r = 0; r < nrepeat; ++r) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      libxs_pack(packed, back, LIBXS_DATATYPE_U64, n, nbits);
      dpack += libxs_timer_duration(t0, libxs_timer_tick());
    }
    for (r = 0; r < nrepeat; ++r) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      libxs_unpack(back, packed, LIBXS_DATATYPE_U64, n, nbits);
      dunpack += libxs_timer_duration(t0, libxs_timer_tick());
    }
    for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
      if ((uint64_t)back[i] != (uint64_t)q[i] + offset) result = EXIT_FAILURE;
    }
    printf("%-14s %10.2f %8.2f %14.2f %14.2f\n", "fixed", (double)nbits, 1.0,
      1E9 * dpack / nrepeat / n, 1E9 * dunpack / nrepeat / n);
    for (ig = 0; ig < NGROUPS && EXIT_SUCCESS == result; ++ig) {
      char label[32];
      dpack = dunpack = 0;
      for (r = 0; r < nrepeat; ++r) {
        const libxs_timer_tick_t t0 = libxs_timer_tick();
        nwords = libxs_vpack(packed, q, LIBXS_DATATYPE_I64, n, nbits, groups[ig]);
        dpack += libxs_timer_duration(t0, libxs_timer_tick());
      }
      for (r = 0; r < nrepeat; ++r) {
        const libxs_timer_tick_t t0 = libxs_timer_tick();
        if (nwords != libxs_vunpack(back, packed, nwords, LIBXS_DATATYPE_I64, n)) result = EXIT_FAILURE;
        dunpack += libxs_timer_duration(t0, libxs_timer_tick());
      }
      if (0 != memcmp(back, q, n * sizeof(int64_t))) result = EXIT_FAILURE;
      LIBXS_SNPRINTF(label, sizeof(label), "vpack %i", groups[ig]);
      printf("%-14s %10.2f %8.2f %14.2f %14.2f\n", label, 64.0 * nwords / n, (double)nfixed / nwords,
        1E9 * dpack / nrepeat / n, 1E9 * dunpack / nrepeat / n);
    }
    if (EXIT_SUCCESS != result) fprintf(stderr, "ERROR: a round trip failed\n");
  }
  free(q);
  free(back);
  free(packed);
  return result;
}


int main(int argc, char* argv[])
{
  FILE *const file = (1 < argc ? fopen(argv[1], "rb") : NULL);
  int result;
  if (NULL != file) { /* MHD files, optionally followed by eps */
    double eps = 0;
    int nfiles = argc - 1;
    fclose(file);
    if (2 < argc && 0 < atof(argv[argc - 1])) {
      FILE *const last = fopen(argv[argc - 1], "rb");
      if (NULL == last) {
        eps = atof(argv[argc - 1]);
        --nfiles;
      }
      else fclose(last);
    }
    result = run_files(nfiles, argv + 1, eps);
  }
  else {
    const size_t n = (size_t)LIBXS_MAX(1, 1 < argc ? atoi(argv[1]) : (1 << 20));
    const int nbits = LIBXS_CLMP(2 < argc ? atoi(argv[2]) : 24, 2, 63);
    const int range = LIBXS_CLMP(3 < argc ? atoi(argv[3]) : 12, 1, nbits);
    const int zeros = LIBXS_CLMP(4 < argc ? atoi(argv[4]) : 10, 0, 100);
    const int nrepeat = LIBXS_MAX(1, 5 < argc ? atoi(argv[5]) : 10);
    result = run_synthetic(n, nbits, range, zeros, nrepeat);
  }
  return result;
}
