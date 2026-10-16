# Hashing and Checksums

Header: `libxs_hash.h`

Hash functions for buffers, fixed-width keys, and strings.
The default hash uses hardware-accelerated CRC32-C (SSE4.2)
with a software fallback.

## Buffer Hashing

```C
unsigned int libxs_hash(const void* data, size_t size,
  unsigned int seed);
```

Produce a 32-bit hash from a byte buffer of the given size.
NULL buffer is accepted (returns seed). Uses CRC32-C when
SSE4.2 is available, otherwise a software table-based CRC32.

```C
uint64_t libxs_hash64(const void* data, size_t size, uint64_t seed);
```

Produce a 64-bit hash from a byte buffer of the given size. NULL
buffer or zero size is accepted (returns seed). It is not a CRC:
each 8-byte word is mixed non-linearly (splitmix64 finalizer), so
two inputs collide by chance and not by a difference a CRC cancels.
The value does not depend on the byte order of the host.

## Fixed-Width Hashing

```C
unsigned int libxs_hash8(unsigned int data);
unsigned int libxs_hash16(unsigned int data);
unsigned int libxs_hash32(uint64_t data);
```

Hash an 8-, 16-, or 32-bit value. The input is split into a
data part and an implicit seed derived from the remaining bits.
The result is masked to the corresponding width (8, 16, or 32
bits).

## String Hashing

```C
uint64_t libxs_hash_string(const char string[]);
```

64-bit hash of a character string. Short strings (up to 8
bytes) are stored directly, which cannot collide; longer strings
use `libxs_hash64`. NULL-string is accepted (returns zero).

From Fortran (`USE libxs`), the generic `libxs_hash64` takes a
contiguous array of `CHARACTER(C_CHAR)`, `INTEGER(C_INT8_T)`,
`INTEGER(C_INT)`, or `INTEGER(C_LONG_LONG)` and an optional seed,
and returns `INTEGER(C_INT64_T)`, as `libxs_hash` does for 32 bits.

## CRC32 (ISO 3309)

```C
unsigned int libxs_hash_iso3309(const void* data,
  size_t size, unsigned int seed);
```

CRC-32 using the ISO 3309 polynomial (used by PNG, gzip, and
similar formats). Software-only, no hardware acceleration.
Pre-condition and post-XOR with 0xFFFFFFFF are the caller's
responsibility. NULL buffer is accepted.

## Adler-32

```C
unsigned int libxs_adler32(const void* data,
  size_t size, unsigned int seed);
```

Adler-32 checksum (the variant used by zlib). The standard
initial seed is 1. NULL buffer is accepted.
