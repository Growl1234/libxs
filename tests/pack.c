/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_source.h>

#define NMAX 1000
#define SENTINEL 0x5A5A5A5A5A5A5A5AULL


static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}


/* Random integers of nbits bits (signed: sign-extended), stored in the type. */
static void fill(void* values, libxs_data_t type, size_t n, int nbits, int zeros)
{
  const uint64_t mask = (64 > nbits ? ((((uint64_t)1) << nbits) - 1) : ~((uint64_t)0));
  const uint64_t sign = (0 == (1 & LIBXS_TYPEORDER(type)) ? (((uint64_t)1) << (nbits - 1)) : 0);
  size_t i;
  for (i = 0; i < n; ++i) {
    uint64_t v = rng_next() & mask;
    if (0 != zeros && 0 != ((i / 16) & 1)) v = 0; /* every other group of 16 is zero */
    else if (0 != zeros) v >>= (int)(rng_next() % (uint64_t)nbits); /* mixed magnitudes */
    if (0 != sign) v = (v ^ sign) - sign;
    switch (LIBXS_TYPESIZE(type)) {
      case 1: ((uint8_t*)values)[i] = (uint8_t)v; break;
      case 2: ((uint16_t*)values)[i] = (uint16_t)v; break;
      case 4: ((uint32_t*)values)[i] = (uint32_t)v; break;
      default: ((uint64_t*)values)[i] = v;
    }
  }
}


static int check_fixed(libxs_data_t type)
{
  static const size_t counts[] = { 0, 1, 63, 64, 65, 200, NMAX };
  const int typesize = (int)LIBXS_TYPESIZE(type);
  uint64_t values[NMAX], back[NMAX], packed[NMAX + 2];
  int result = EXIT_SUCCESS, nbits;
  size_t ic, i;
  for (nbits = 1; nbits <= 8 * typesize && EXIT_SUCCESS == result; ++nbits) {
    for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
      const size_t n = counts[ic], nwords = (n * (size_t)nbits + 63) / 64;
      fill(values, type, n, nbits, 0);
      for (i = 0; i < NMAX + 2; ++i) packed[i] = SENTINEL;
      memset(back, 0xFF, sizeof(back));
      if (EXIT_SUCCESS != libxs_pack(packed, values, type, n, nbits)
        || EXIT_SUCCESS != libxs_unpack(back, packed, type, n, nbits))
      {
        fprintf(stderr, "fixed: call failed (type %i, nbits %i, n %i)\n", (int)type, nbits, (int)n);
        result = EXIT_FAILURE;
      }
      else if (SENTINEL != packed[nwords]) {
        fprintf(stderr, "fixed: overrun (type %i, nbits %i, n %i)\n", (int)type, nbits, (int)n);
        result = EXIT_FAILURE;
      }
      else if (0 != memcmp(values, back, n * (size_t)typesize)) {
        fprintf(stderr, "fixed: mismatch (type %i, nbits %i, n %i)\n", (int)type, nbits, (int)n);
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


static int check_variable(libxs_data_t type)
{
  static const size_t counts[] = { 1, 15, 16, 17, 333, NMAX };
  static const int groups[] = { 0, 1, 7, 64, 255 };
  const int typesize = (int)LIBXS_TYPESIZE(type);
  uint64_t values[NMAX], back[NMAX], packed[2 * NMAX];
  int result = EXIT_SUCCESS, maxbits, ig, zeros;
  size_t ic, i;
  for (maxbits = 1; maxbits <= 8 * typesize && EXIT_SUCCESS == result; maxbits += 3) {
    for (ig = 0; ig < (int)(sizeof(groups) / sizeof(*groups)) && EXIT_SUCCESS == result; ++ig) {
      for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
        for (zeros = 0; zeros < 2 && EXIT_SUCCESS == result; ++zeros) {
          const size_t n = counts[ic];
          const size_t bound = libxs_vpack_bound(n, maxbits, groups[ig]);
          size_t nwords, nread;
          fill(values, type, n, maxbits, zeros);
          for (i = 0; i < 2 * NMAX; ++i) packed[i] = SENTINEL;
          memset(back, 0xFF, sizeof(back));
          nwords = libxs_vpack(packed, values, type, n, maxbits, groups[ig]);
          nread = libxs_vunpack(back, packed, type, n);
          if (0 == nwords || nwords > bound || nread != nwords) {
            fprintf(stderr, "variable: %i words (bound %i, read %i) for type %i, maxbits %i, group %i, n %i\n",
              (int)nwords, (int)bound, (int)nread, (int)type, maxbits, groups[ig], (int)n);
            result = EXIT_FAILURE;
          }
          else if (SENTINEL != packed[nwords]) {
            fprintf(stderr, "variable: overrun (type %i, maxbits %i, group %i, n %i)\n",
              (int)type, maxbits, groups[ig], (int)n);
            result = EXIT_FAILURE;
          }
          else if (0 != memcmp(values, back, n * (size_t)typesize)) {
            fprintf(stderr, "variable: mismatch (type %i, maxbits %i, group %i, n %i)\n",
              (int)type, maxbits, groups[ig], (int)n);
            result = EXIT_FAILURE;
          }
        }
      }
    }
  }
  return result;
}


/** All-zero values cost only the header, and invalid arguments are rejected. */
static int check_edges(void)
{
  uint64_t values[64] = { 0 }, packed[64];
  int result = EXIT_SUCCESS;
  if (1 != libxs_vpack(packed, values, LIBXS_DATATYPE_I64, 32, 40, 16)) {
    fprintf(stderr, "edges: zeros are not header-only\n");
    result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == libxs_pack(packed, values, LIBXS_DATATYPE_F64, 64, 8)
    || EXIT_SUCCESS == libxs_pack(packed, values, LIBXS_DATATYPE_I32, 64, 33)
    || EXIT_SUCCESS == libxs_pack(packed, values, LIBXS_DATATYPE_U8, 64, 0)
    || 0 != libxs_vpack(packed, values, LIBXS_DATATYPE_F32, 64, 8, 0)
    || 0 != libxs_vpack(packed, values, LIBXS_DATATYPE_I16, 64, 8, 256))
  {
    fprintf(stderr, "edges: invalid arguments accepted\n");
    result = EXIT_FAILURE;
  }
  return result;
}


int main(int argc, char* argv[])
{
  static const libxs_data_t types[] = {
    LIBXS_DATATYPE_I8, LIBXS_DATATYPE_U8, LIBXS_DATATYPE_I16, LIBXS_DATATYPE_U16,
    LIBXS_DATATYPE_I32, LIBXS_DATATYPE_U32, LIBXS_DATATYPE_I64, LIBXS_DATATYPE_U64
  };
  int result = check_edges(), i;
  LIBXS_UNUSED(argc);
  LIBXS_UNUSED(argv);
  for (i = 0; i < (int)(sizeof(types) / sizeof(*types)) && EXIT_SUCCESS == result; ++i) {
    result = check_fixed(types[i]);
    if (EXIT_SUCCESS == result) result = check_variable(types[i]);
  }
  return result;
}
