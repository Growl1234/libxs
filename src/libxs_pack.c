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
#include <libxs/libxs_utils.h>

#define LIBXS_PACK_MASK(NBITS) (64 > (NBITS) \
  ? ((((uint64_t)1) << (NBITS)) - 1) : ~((uint64_t)0))
/* expands M(1) to M(64): a literal width lets the compiler specialize per case */
#define LIBXS_PACK_CASE8(M, B) M(B + 1) M(B + 2) M(B + 3) M(B + 4) \
  M(B + 5) M(B + 6) M(B + 7) M(B + 8)
#define LIBXS_PACK_CASES(M) LIBXS_PACK_CASE8(M, 0) LIBXS_PACK_CASE8(M, 8) \
  LIBXS_PACK_CASE8(M, 16) LIBXS_PACK_CASE8(M, 24) LIBXS_PACK_CASE8(M, 32) \
  LIBXS_PACK_CASE8(M, 40) LIBXS_PACK_CASE8(M, 48) LIBXS_PACK_CASE8(M, 56)
#define LIBXS_PACK_CASE_PACK(N) case N: internal_libxs_pack64(out, in, N); break;
#define LIBXS_PACK_CASE_UNPACK(N) case N: internal_libxs_unpack64(out, in, N); break;
/* Decoding paths compiled (AVX-512, AVX2), and the one the static target covers already:
 * 3 = AVX-512, 2 = AVX2, 1 = scalar. A runtime check is needed only above the static one. */
#if defined(LIBXS_INTRINSICS_INCLUDE) && !defined(__NO_INTRINSICS)
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define LIBXS_PACK_AVX512
# endif
# if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define LIBXS_PACK_AVX2
# endif
#endif
#if defined(LIBXS_PACK_AVX512) && (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_PACK_STATIC 3
#elif defined(LIBXS_PACK_AVX2) && (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_PACK_STATIC 2
#else
# define LIBXS_PACK_STATIC 1
#endif
#if defined(LIBXS_PACK_AVX512) && (3 > LIBXS_PACK_STATIC)
# define LIBXS_PACK_RUNTIME
#elif defined(LIBXS_PACK_AVX2) && (2 > LIBXS_PACK_STATIC)
# define LIBXS_PACK_RUNTIME
#endif
#define LIBXS_PACK_CASE_VPACK(N) case N: internal_libxs_vpack8(out, in, N); break;
#define LIBXS_PACK_CASE_VUNPACK(N) case N: internal_libxs_vunpack8(out, in, N); break;


typedef struct internal_libxs_pack_writer_t {
  uint64_t* out;
  size_t nwords;
  uint64_t acc;
  int fill;
} internal_libxs_pack_writer_t;


#if defined(LIBXS_PACK_RUNTIME)
/* 0: not determined yet, else as LIBXS_PACK_STATIC */
LIBXS_APIVAR_DEFINE(int internal_libxs_pack_kind);
#endif


/* Size of the type in bytes if it is an integer type, zero otherwise. */
LIBXS_API_INLINE int internal_libxs_pack_typesize(libxs_data_t type)
{
  const int order = LIBXS_TYPEORDER(type);
  return (4 <= order && 11 >= order) ? (int)LIBXS_TYPESIZE(type) : 0;
}


LIBXS_API_INLINE int internal_libxs_pack_signed(libxs_data_t type)
{
  return 0 == (1 & LIBXS_TYPEORDER(type));
}


/* Widens n values starting at offset into dst (signed types sign-extended). */
LIBXS_API_INLINE void internal_libxs_pack_load(uint64_t* LIBXS_RESTRICT dst,
  const void* LIBXS_RESTRICT src, libxs_data_t type, size_t offset, size_t n)
{
  size_t i;
  switch (type) {
    case LIBXS_DATATYPE_I32: {
      for (i = 0; i < n; ++i) dst[i] = (uint64_t)(int64_t)((const int32_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_U32: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint32_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_I16: {
      for (i = 0; i < n; ++i) dst[i] = (uint64_t)(int64_t)((const int16_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_U16: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint16_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_I8: {
      for (i = 0; i < n; ++i) dst[i] = (uint64_t)(int64_t)((const int8_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_U8: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint8_t*)src)[offset + i];
    } break;
    default: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint64_t*)src)[offset + i];
    }
  }
}


/* Narrows n values into dst starting at offset (two's complement truncation). */
LIBXS_API_INLINE void internal_libxs_pack_store(void* LIBXS_RESTRICT dst,
  const uint64_t* LIBXS_RESTRICT src, libxs_data_t type, size_t offset, size_t n)
{
  size_t i;
  switch (LIBXS_TYPESIZE(type)) {
    case 4: {
      for (i = 0; i < n; ++i) ((uint32_t*)dst)[offset + i] = (uint32_t)src[i];
    } break;
    case 2: {
      for (i = 0; i < n; ++i) ((uint16_t*)dst)[offset + i] = (uint16_t)src[i];
    } break;
    case 1: {
      for (i = 0; i < n; ++i) ((uint8_t*)dst)[offset + i] = (uint8_t)src[i];
    } break;
    default: {
      for (i = 0; i < n; ++i) ((uint64_t*)dst)[offset + i] = src[i];
    }
  }
}


/* 64 values into nbits words: each word is assembled in a register and stored once. */
LIBXS_INLINE_ALWAYS void internal_libxs_pack64(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  const uint64_t mask = LIBXS_PACK_MASK(nbits);
  uint64_t acc = 0;
  int fill = 0, nwords = 0, i;
  LIBXS_PRAGMA_UNROLL_N(64)
  for (i = 0; i < 64; ++i) {
    const uint64_t v = in[i] & mask;
    acc |= v << fill;
    fill += nbits;
    if (64 <= fill) {
      out[nwords] = acc;
      ++nwords;
      fill -= 64;
      acc = (0 != fill ? (v >> (nbits - fill)) : 0);
    }
  }
}


LIBXS_INLINE_ALWAYS void internal_libxs_unpack64(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  const uint64_t mask = LIBXS_PACK_MASK(nbits);
  int i;
  LIBXS_PRAGMA_UNROLL_N(64)
  for (i = 0; i < 64; ++i) {
    const int bit = i * nbits, off = bit & 63;
    uint64_t v = in[bit >> 6] >> off;
    if (64 < off + nbits) v |= in[(bit >> 6) + 1] << (64 - off);
    out[i] = v & mask;
  }
}


LIBXS_API_INLINE void internal_libxs_pack_group(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  switch (nbits) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_PACK)
    default:;
  }
}


LIBXS_API_INLINE void internal_libxs_unpack_group(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  switch (nbits) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_UNPACK)
    default:;
  }
}


/* 8 values of w bits (no higher bits set) into exactly w bytes. */
LIBXS_INLINE_ALWAYS void internal_libxs_vpack8(unsigned char* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int w)
{
  uint64_t tmp[8], acc = 0;
  int fill = 0, nwords = 0, i;
  LIBXS_PRAGMA_UNROLL_N(8)
  for (i = 0; i < 8; ++i) {
    const uint64_t v = in[i];
    acc |= v << fill;
    fill += w;
    if (64 <= fill) {
      tmp[nwords] = acc;
      ++nwords;
      fill -= 64;
      acc = (0 != fill ? (v >> (w - fill)) : 0);
    }
  }
  if (0 != fill) tmp[nwords] = acc;
  memcpy(out, tmp, w);
}


LIBXS_INLINE_ALWAYS void internal_libxs_vunpack8(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT in, int w)
{
  const uint64_t mask = LIBXS_PACK_MASK(w);
  uint64_t tmp[8];
  int i;
  tmp[(w - 1) >> 3] = 0; /* bytes beyond w in the last word */
  memcpy(tmp, in, w);
  LIBXS_PRAGMA_UNROLL_N(8)
  for (i = 0; i < 8; ++i) {
    const int bit = i * w, off = bit & 63;
    uint64_t v = tmp[bit >> 6] >> off;
    if (64 < off + w) v |= tmp[(bit >> 6) + 1] << (64 - off);
    out[i] = v & mask;
  }
}


/* 8 values of w bits (w <= 56) by unaligned loads; reads up to 7 bytes past the w bytes. */
LIBXS_API_INLINE void internal_libxs_vunpack8_load(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT in, int w)
{
  const uint64_t mask = LIBXS_PACK_MASK(w);
  int i;
  LIBXS_PRAGMA_UNROLL_N(8)
  for (i = 0; i < 8; ++i) {
    const int bit = i * w;
    uint64_t x;
    memcpy(&x, in + (bit >> 3), sizeof(x));
    out[i] = (x >> (bit & 7)) & mask;
  }
}


LIBXS_API_INLINE void internal_libxs_vpack8_group(unsigned char* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int w)
{
  switch (w) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_VPACK)
    default:;
  }
}


LIBXS_API_INLINE void internal_libxs_vunpack8_group(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT in, int w)
{
  switch (w) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_VUNPACK)
    default: {
      int i;
      for (i = 0; i < 8; ++i) out[i] = 0;
    }
  }
}


/* Appends the w low bits of v (1 <= w <= 64, no higher bits set). */
LIBXS_API_INLINE void internal_libxs_pack_put(internal_libxs_pack_writer_t* wr,
  uint64_t v, int w)
{
  wr->acc |= v << wr->fill;
  wr->fill += w;
  if (64 <= wr->fill) {
    wr->out[wr->nwords] = wr->acc;
    ++wr->nwords;
    wr->fill -= 64;
    wr->acc = (0 != wr->fill ? (v >> (w - wr->fill)) : 0);
  }
}


LIBXS_API_INLINE void internal_libxs_pack_flush(internal_libxs_pack_writer_t* wr)
{
  if (0 != wr->fill) {
    wr->out[wr->nwords] = wr->acc;
    ++wr->nwords;
    wr->acc = 0;
    wr->fill = 0;
  }
}


/* Reads w bits (1 <= w <= 64) at the given bit position. */
LIBXS_API_INLINE uint64_t internal_libxs_pack_get(const uint64_t* in,
  size_t bit, int w)
{
  const int off = (int)(bit & 63);
  uint64_t v = in[bit >> 6] >> off;
  if (64 < off + w) v |= in[(bit >> 6) + 1] << (64 - off);
  return v & LIBXS_PACK_MASK(w);
}


/* Payload bytes of a byte-aligned stream, rounded to words: unaligned loads stay within. */
LIBXS_API_INLINE size_t internal_libxs_vpack_nbytes(const unsigned char* widths, size_t n, int g)
{
  const size_t ngroups = (n + g - 1) / g;
  size_t i, nbytes = 0;
  for (i = 0; i < ngroups; ++i) nbytes += widths[i] * ((LIBXS_MIN((size_t)g, n - i * g) + 7) / 8);
  return (nbytes + 7) & ~(size_t)7;
}


/* Decodes 8 values (sign unfolded) from a byte-aligned group; fast reads where safe. */
LIBXS_API_INLINE void internal_libxs_vunpack8_any(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT data, size_t pos, size_t nbytes, int w, int is_signed)
{
  int i;
  if (0 < w && 56 >= w && pos + (size_t)w + 7 <= nbytes) internal_libxs_vunpack8_load(out, data + pos, w);
  else internal_libxs_vunpack8_group(out, data + pos, w);
  if (0 != is_signed) {
    for (i = 0; i < 8; ++i) out[i] = (out[i] >> 1) ^ (0 - (out[i] & 1));
  }
}


/* Decodes a byte-aligned stream into 64-bit integers; returns the bits consumed. */
LIBXS_API_INLINE size_t internal_libxs_vunpack64_sw(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g, nbytes = internal_libxs_vpack_nbytes(widths, n, g);
  uint64_t tmp[8];
  size_t i, k, pos = 0;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i];
    for (k = 0; k < m; k += 8) {
      if (k + 8 <= m) internal_libxs_vunpack8_any(out + base + k, data, pos, nbytes, w, is_signed);
      else {
        internal_libxs_vunpack8_any(tmp, data, pos, nbytes, w, is_signed);
        memcpy(out + base + k, tmp, (m - k) * sizeof(*tmp));
      }
      pos += (size_t)w;
    }
  }
  return 8 * pos;
}


#if defined(LIBXS_PACK_AVX512)
/* As internal_libxs_vunpack64_sw, a group of 8 per vector: the w bytes are loaded (masked,
 * hence no read past them), and each lane funnel-shifts its two words into place. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
size_t internal_libxs_vunpack64_avx512(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g;
  const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i word = _mm512_set1_epi64(63), one = _mm512_set1_epi64(1);
  const __m512i sixtyfour = _mm512_set1_epi64(64);
  size_t i, k, pos = 0;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i];
    const __m512i bits = _mm512_mullo_epi64(lane, _mm512_set1_epi64(w));
    const __m512i lo = _mm512_srli_epi64(bits, 6), hi = _mm512_add_epi64(lo, one);
    const __m512i shift = _mm512_and_si512(bits, word), back = _mm512_sub_epi64(sixtyfour, shift);
    const __m512i mask = _mm512_set1_epi64((long long)LIBXS_PACK_MASK(w));
    const __mmask64 bytes = (64 > w ? ((((__mmask64)1) << w) - 1) : ~((__mmask64)0));
    for (k = 0; k < m; k += 8) {
      const __m512i src = _mm512_maskz_loadu_epi8(bytes, data + pos);
      __m512i x = _mm512_or_si512( /* a shift by 64 yields zero */
        _mm512_srlv_epi64(_mm512_permutexvar_epi64(lo, src), shift),
        _mm512_sllv_epi64(_mm512_permutexvar_epi64(hi, src), back));
      x = _mm512_and_si512(x, mask);
      if (0 != is_signed) {
        x = _mm512_xor_si512(_mm512_srli_epi64(x, 1),
          _mm512_sub_epi64(_mm512_setzero_si512(), _mm512_and_si512(x, one)));
      }
      if (k + 8 <= m) _mm512_storeu_si512((void*)(out + base + k), x);
      else _mm512_mask_storeu_epi64(out + base + k, (__mmask8)((1u << (m - k)) - 1), x);
      pos += (size_t)w;
    }
  }
  return 8 * pos;
}
#endif


#if defined(LIBXS_PACK_AVX2)
/* As internal_libxs_vunpack64_sw, a group of 8 as two halves: qword loads (masked to the
 * w bytes, rounded up to whole qwords), and a permute across two registers per word. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
size_t internal_libxs_vunpack64_avx2(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g, nbytes = internal_libxs_vpack_nbytes(widths, n, g);
  const __m256i qlane = _mm256_set_epi64x(3, 2, 1, 0), one = _mm256_set1_epi64x(1);
  const __m256i three = _mm256_set1_epi64x(3), six = _mm256_set1_epi64x(6);
  const __m256i odd = _mm256_set1_epi64x(((long long)1) << 32);
  const __m256i word = _mm256_set1_epi64x(63), sixtyfour = _mm256_set1_epi64x(64);
  __m256i plo[2], phi[2], mlo[2], mhi[2], shift[2], back[2];
  uint64_t tmp[8];
  size_t i, k, pos = 0;
  int h;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i], nq = (w + 7) >> 3;
    const __m256i wide = _mm256_set1_epi64x(w);
    const __m256i mask = _mm256_set1_epi64x((long long)LIBXS_PACK_MASK(w));
    const __m256i load0 = _mm256_cmpgt_epi64(_mm256_set1_epi64x(nq), qlane);
    const __m256i load1 = _mm256_cmpgt_epi64(_mm256_set1_epi64x(nq - 4), qlane);
    for (h = 0; h < 2; ++h) { /* per half: word indexes as dword pairs, and which register */
      const __m256i b = _mm256_mul_epu32(_mm256_add_epi64(qlane, _mm256_set1_epi64x(4 * h)), wide);
      const __m256i lo = _mm256_srli_epi64(b, 6), hi = _mm256_add_epi64(lo, one);
      const __m256i tlo = _mm256_and_si256(_mm256_slli_epi64(lo, 1), six); /* even dword */
      const __m256i thi = _mm256_and_si256(_mm256_slli_epi64(hi, 1), six);
      plo[h] = _mm256_or_si256(_mm256_or_si256(tlo, _mm256_slli_epi64(tlo, 32)), odd);
      phi[h] = _mm256_or_si256(_mm256_or_si256(thi, _mm256_slli_epi64(thi, 32)), odd);
      mlo[h] = _mm256_cmpgt_epi64(lo, three);
      mhi[h] = _mm256_cmpgt_epi64(hi, three);
      shift[h] = _mm256_and_si256(b, word);
      back[h] = _mm256_sub_epi64(sixtyfour, shift[h]);
    }
    for (k = 0; k < m; k += 8) {
      if (k + 8 <= m && 0 < w && pos + (size_t)w + 7 <= nbytes) {
        const __m256i q0 = _mm256_maskload_epi64((const long long*)(data + pos), load0);
        const __m256i q1 = _mm256_maskload_epi64((const long long*)(data + pos + 32), load1);
        for (h = 0; h < 2; ++h) {
          const __m256i a = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(q0, plo[h]),
            _mm256_permutevar8x32_epi32(q1, plo[h]), mlo[h]);
          const __m256i c = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(q0, phi[h]),
            _mm256_permutevar8x32_epi32(q1, phi[h]), mhi[h]);
          __m256i x = _mm256_or_si256( /* a shift by 64 yields zero */
            _mm256_srlv_epi64(a, shift[h]), _mm256_sllv_epi64(c, back[h]));
          x = _mm256_and_si256(x, mask);
          if (0 != is_signed) {
            x = _mm256_xor_si256(_mm256_srli_epi64(x, 1),
              _mm256_sub_epi64(_mm256_setzero_si256(), _mm256_and_si256(x, one)));
          }
          _mm256_storeu_si256((__m256i*)(out + base + k + 4 * h), x);
        }
      }
      else if (k + 8 <= m) internal_libxs_vunpack8_any(out + base + k, data, pos, nbytes, w, is_signed);
      else {
        internal_libxs_vunpack8_any(tmp, data, pos, nbytes, w, is_signed);
        memcpy(out + base + k, tmp, (m - k) * sizeof(*tmp));
      }
      pos += (size_t)w;
    }
  }
  return 8 * pos;
}
#endif


/* Decodes a byte-aligned stream into 64-bit integers with the best path available. */
LIBXS_API_INLINE size_t internal_libxs_vunpack64(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  size_t result;
#if defined(LIBXS_PACK_RUNTIME)
  int kind = internal_libxs_pack_kind;
  if (0 == kind) {
    const int target_arch = libxs_cpuid(NULL);
    kind = LIBXS_PACK_STATIC;
# if defined(LIBXS_PACK_AVX2)
    if (LIBXS_X86_AVX2 <= target_arch) kind = 2;
# endif
# if defined(LIBXS_PACK_AVX512)
    if (LIBXS_X86_AVX512 <= target_arch) kind = 3;
# endif
    internal_libxs_pack_kind = kind;
  }
#else
  const int kind = LIBXS_PACK_STATIC;
#endif
#if defined(LIBXS_PACK_AVX512)
  if (3 == kind) result = internal_libxs_vunpack64_avx512(out, widths, data, n, g, is_signed);
  else
#endif
#if defined(LIBXS_PACK_AVX2)
  /* 8 permutes per group (no qword permute across registers) amortize from 16 values on */
  if (2 <= kind && 16 <= g) result = internal_libxs_vunpack64_avx2(out, widths, data, n, g, is_signed);
  else
#endif
  result = internal_libxs_vunpack64_sw(out, widths, data, n, g, is_signed);
  LIBXS_UNUSED(kind);
  return result;
}


LIBXS_API int libxs_pack(void* packed, const void* values,
  libxs_data_t type, size_t n, int nbits)
{
  const int typesize = internal_libxs_pack_typesize(type);
  int result = EXIT_FAILURE;
  if (0 < typesize && 0 < nbits && nbits <= 8 * typesize
    && ((NULL != packed && NULL != values) || 0 == n))
  {
    uint64_t *const out = (uint64_t*)packed;
    const size_t ngroups = n / 64, rest = n - 64 * ngroups;
    uint64_t buf[64];
    size_t g;
    for (g = 0; g < ngroups; ++g) {
      const uint64_t* in = buf;
      if (8 == typesize) in = (const uint64_t*)values + 64 * g;
      else internal_libxs_pack_load(buf, values, type, 64 * g, 64);
      internal_libxs_pack_group(out + (size_t)nbits * g, in, nbits);
    }
    if (0 != rest) {
      internal_libxs_pack_writer_t wr;
      const uint64_t mask = LIBXS_PACK_MASK(nbits);
      size_t i;
      internal_libxs_pack_load(buf, values, type, 64 * ngroups, rest);
      wr.out = out + (size_t)nbits * ngroups;
      wr.nwords = 0;
      wr.acc = 0;
      wr.fill = 0;
      for (i = 0; i < rest; ++i) internal_libxs_pack_put(&wr, buf[i] & mask, nbits);
      internal_libxs_pack_flush(&wr);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API int libxs_unpack(void* values, const void* packed,
  libxs_data_t type, size_t n, int nbits)
{
  const int typesize = internal_libxs_pack_typesize(type);
  int result = EXIT_FAILURE;
  if (0 < typesize && 0 < nbits && nbits <= 8 * typesize
    && ((NULL != packed && NULL != values) || 0 == n))
  {
    const uint64_t *const in = (const uint64_t*)packed;
    const uint64_t sign = (0 != internal_libxs_pack_signed(type) && 64 > nbits)
      ? (((uint64_t)1) << (nbits - 1)) : 0;
    const size_t ngroups = n / 64, rest = n - 64 * ngroups;
    uint64_t buf[64];
    size_t g, i;
    for (g = 0; g < ngroups; ++g) {
      uint64_t *const out = (8 == typesize ? (uint64_t*)values + 64 * g : buf);
      internal_libxs_unpack_group(out, in + (size_t)nbits * g, nbits);
      if (0 != sign) {
        for (i = 0; i < 64; ++i) out[i] = (out[i] ^ sign) - sign;
      }
      if (8 != typesize) internal_libxs_pack_store(values, buf, type, 64 * g, 64);
    }
    if (0 != rest) {
      const uint64_t *const tail = in + (size_t)nbits * ngroups;
      for (i = 0; i < rest; ++i) {
        const uint64_t v = internal_libxs_pack_get(tail, i * (size_t)nbits, nbits);
        buf[i] = (v ^ sign) - sign;
      }
      internal_libxs_pack_store(values, buf, type, 64 * ngroups, rest);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API size_t libxs_vpack_bound(size_t n, int maxbits, int group)
{
  const int g = (0 == group ? LIBXS_VPACK_GROUP : group);
  size_t result = 0;
  if (0 < n && 0 < maxbits && 64 >= maxbits && 0 < g && 255 >= g) {
    const size_t ngroups = (n + g - 1) / g;
    result = (1 + ngroups + 7) / 8 + (((n + 7) / 8) * (size_t)maxbits + 7) / 8;
  }
  return result;
}


LIBXS_API size_t libxs_vpack(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int group)
{
  const int typesize = internal_libxs_pack_typesize(type);
  const int g = (0 == group ? LIBXS_VPACK_GROUP : group);
  size_t result = 0;
  if (0 < typesize && 0 < maxbits && maxbits <= 8 * typesize && 0 < g && 255 >= g
    && 0 < n && NULL != packed && NULL != values)
  {
    const size_t ngroups = (n + g - 1) / g, nhead = (1 + ngroups + 7) / 8;
    const uint64_t mask = LIBXS_PACK_MASK(maxbits);
    const uint64_t sign = (0 != internal_libxs_pack_signed(type)
      ? (((uint64_t)1) << (maxbits - 1)) : 0);
    unsigned char *const head = (unsigned char*)packed;
    unsigned char *const data = (unsigned char*)((uint64_t*)packed + nhead);
    internal_libxs_pack_writer_t wr;
    uint64_t buf[255 + 7];
    size_t i, k, nbytes = 0;
    head[0] = (unsigned char)g;
    for (i = 1 + ngroups; i < 8 * nhead; ++i) head[i] = 0;
    wr.out = (uint64_t*)packed + nhead;
    wr.nwords = 0;
    wr.acc = 0;
    wr.fill = 0;
    for (i = 0; i < ngroups; ++i) {
      const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
      uint64_t any = 0;
      int w;
      const uint64_t* src = buf;
      if (8 == typesize) src = (const uint64_t*)values + base;
      else internal_libxs_pack_load(buf, values, type, base, m);
      for (k = 0; k < m; ++k) {
        uint64_t v = src[k] & mask;
        if (0 != sign) { /* sign-extend from maxbits, then fold the sign (zig-zag) */
          v = (v ^ sign) - sign;
          v = (v << 1) ^ (0 - (v >> 63));
        }
        buf[k] = v;
        any |= v;
      }
      w = (int)LIBXS_NBITS(any);
      head[1 + i] = (unsigned char)w;
      if (0 < w && 0 == (g & 7)) { /* byte-aligned: 8 values take w bytes */
        for (k = m; k < ((m + 7) & ~(size_t)7); ++k) buf[k] = 0;
        for (k = 0; k < m; k += 8) {
          internal_libxs_vpack8_group(data + nbytes, buf + k, w);
          nbytes += (size_t)w;
        }
      }
      else if (0 < w) {
        for (k = 0; k < m; ++k) internal_libxs_pack_put(&wr, buf[k], w);
      }
    }
    if (0 == (g & 7)) {
      for (; 0 != (nbytes & 7); ++nbytes) data[nbytes] = 0;
      result = nhead + nbytes / 8;
    }
    else {
      internal_libxs_pack_flush(&wr);
      result = nhead + wr.nwords;
    }
  }
  return result;
}


LIBXS_API size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n)
{
  const int typesize = internal_libxs_pack_typesize(type);
  size_t result = 0;
  if (0 < typesize && 0 < n && 0 < npacked && NULL != packed && NULL != values) {
    const unsigned char *const head = (const unsigned char*)packed;
    const int g = head[0], is_signed = internal_libxs_pack_signed(type);
    if (0 < g) {
      const size_t ngroups = (n + g - 1) / g, nhead = (1 + ngroups + 7) / 8;
      const uint64_t *const in = (const uint64_t*)packed + nhead;
      const unsigned char *const data = (const unsigned char*)in;
      const union { uint32_t u; unsigned char c[4]; } probe = { 1 };
      uint64_t buf[255 + 7];
      size_t i, k, bit = 0;
#if !defined(NDEBUG)
      { /* the stream fits: the widths are read only once the header fits */
        size_t nbits = 0, j;
        LIBXS_ASSERT(nhead <= npacked);
        for (j = 0; j < ngroups && nhead <= npacked; ++j) {
          const size_t m = LIBXS_MIN((size_t)g, n - j * g);
          nbits += (size_t)head[1 + j] * (0 == (g & 7) ? ((m + 7) & ~(size_t)7) : m);
        }
        LIBXS_ASSERT(nhead + (nbits + 63) / 64 <= npacked);
      }
#endif
      if (0 == (g & 7) && 8 == typesize && 0 != probe.c[0]) { /* byte-aligned into 64-bit */
        bit = internal_libxs_vunpack64((uint64_t*)values, head + 1, data, n, g, is_signed);
      }
      else {
        const size_t nbytes = (0 == (g & 7) ? internal_libxs_vpack_nbytes(head + 1, n, g) : 0);
        for (i = 0; i < ngroups; ++i) {
          const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
          const int w = head[1 + i];
          if (0 == (g & 7)) { /* byte-aligned: 8 values take w bytes */
            for (k = 0; k < m; k += 8) {
              internal_libxs_vunpack8_any(buf + k, data, bit / 8, (0 != probe.c[0] ? nbytes : 0), w, 0);
              bit += 8 * (size_t)w;
            }
          }
          else {
            for (k = 0; k < m; ++k) {
              buf[k] = 0;
              if (0 < w) {
                buf[k] = internal_libxs_pack_get(in, bit, w);
                bit += w;
              }
            }
          }
          if (0 != is_signed) {
            for (k = 0; k < m; ++k) buf[k] = (buf[k] >> 1) ^ (0 - (buf[k] & 1));
          }
          internal_libxs_pack_store(values, buf, type, base, m);
        }
      }
      result = nhead + (bit + 63) / 64;
    }
  }
  return result;
}
