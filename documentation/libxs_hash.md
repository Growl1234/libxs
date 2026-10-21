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

Produce a 32-bit hash from a byte buffer of the given size. A NULL
buffer is accepted only with a size of zero (returns seed).

```C
uint64_t libxs_hash64(const void* data, size_t size, uint64_t seed);
```

Produce a 64-bit hash from a byte buffer of the given size. NULL
buffer or zero size is accepted (returns seed). Unlike `libxs_hash`,
it is not a CRC, so inputs collide only by chance. The value does not
depend on the byte order of the host.

## Fixed-Width Hashing

```C
unsigned int libxs_hash8(unsigned int data);
unsigned int libxs_hash16(unsigned int data);
unsigned int libxs_hash32(uint64_t data);
```

Hash an 8-, 16-, or 32-bit value. The result has the same width
(8, 16, or 32 bits).

## String Hashing

```C
uint64_t libxs_hash_string(const char string[]);
```

64-bit hash of a character string. Short strings (up to 8
bytes) are stored directly, which cannot collide; longer strings
use `libxs_hash64`. NULL-string is accepted (returns zero). A short
string is stored in the host's byte order, hence its value differs
between little- and big-endian hosts.

From Fortran (`USE libxs`), the generic `libxs_hash64` takes a
contiguous array of `CHARACTER(C_CHAR)`, `INTEGER(C_INT8_T)`,
`INTEGER(C_INT)`, or `INTEGER(C_LONG_LONG)` and an optional seed,
and returns `INTEGER(C_INT64_T)`, as `libxs_hash` does for 32 bits.
`libxs_hash_string(string)` takes a NUL-terminated `CHARACTER(C_CHAR)`
array and returns `INTEGER(C_INT64_T)`. `libxs_hash_iso3309(data, size,
seed)` and `libxs_adler32(data, size, seed)` take a `TYPE(C_PTR)`, an
`INTEGER(C_SIZE_T)` size, and an `INTEGER(C_INT)` seed (all by value), and
return `INTEGER(C_INT)`.

## CRC32 (ISO 3309)

```C
unsigned int libxs_hash_iso3309(const void* data,
  size_t size, unsigned int seed);
```

CRC-32 using the ISO 3309 polynomial (used by PNG, gzip, and
similar formats). Pre-condition and post-XOR with 0xFFFFFFFF are the caller's
responsibility. A NULL buffer is accepted only with a size of zero.

## Adler-32

```C
unsigned int libxs_adler32(const void* data,
  size_t size, unsigned int seed);
```

Adler-32 checksum (the variant used by zlib). The standard
initial seed is 1. A NULL buffer is accepted only with a size of zero.
