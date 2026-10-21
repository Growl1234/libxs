/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_HASH_H
#define LIBXS_HASH_H

#include "libxs.h"

/** Calculate a hash value for the given buffer and seed; NULL-buffer only if size is zero. */
LIBXS_API unsigned int libxs_hash(const void* data, size_t size, unsigned int seed);
LIBXS_API unsigned int libxs_hash8(unsigned int data);
LIBXS_API unsigned int libxs_hash16(unsigned int data);
LIBXS_API unsigned int libxs_hash32(uint64_t data);

/**
 * Calculate a 64-bit hash for the given buffer and seed; accepts NULL-buffer
 * (returns seed). Unlike libxs_hash, it is not a CRC: the mixing is non-linear,
 * so two inputs collide by chance rather than by a difference the CRC cancels.
 */
LIBXS_API uint64_t libxs_hash64(const void* data, size_t size, uint64_t seed);

/** Calculate a CRC32 value (ISO 3309 polynomial) for the given buffer and seed. */
LIBXS_API unsigned int libxs_hash_iso3309(const void* data, size_t size, unsigned int seed);

/** Calculate an Adler-32 checksum for the given buffer and seed. */
LIBXS_API unsigned int libxs_adler32(const void* data, size_t size, unsigned int seed);

/**
 * Calculate a 64-bit hash for the given character string; accepts NULL-string.
 * A string of at most eight bytes is its own value (host byte order), which cannot collide.
 */
LIBXS_API uint64_t libxs_hash_string(const char string[]);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_HASH_H*/
