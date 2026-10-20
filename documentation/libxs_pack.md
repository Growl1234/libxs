# Pack

Header: `libxs_pack.h`

Bit packing of integer arrays, at a fixed width or at a width chosen per group
of values. Packed data is a stream of 64-bit words.

## Fixed Width

```C
int libxs_pack(void* packed, const void* values,
  libxs_data_t type, size_t n, int nbits);
int libxs_unpack(void* values, const void* packed,
  libxs_data_t type, size_t n, int nbits);
```

Packs `n` integers of `type` (`LIBXS_DATATYPE_I8` to `I64`, `U8` to `U64`) at
`nbits` each (1 to eight times the type size) into `CEILING(n*nbits/64)` words.
Bits beyond `nbits` are ignored. Unpacking sign-extends signed types from
`nbits` and zero-extends unsigned types. Both return `EXIT_SUCCESS`, or
`EXIT_FAILURE` for a type or width out of range.

## Variable Width

```C
size_t libxs_vpack(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int group);
size_t libxs_vpack_bound(size_t n, int maxbits, int group);
size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n);
```

Every group of `group` values (1 to 255, 0 selects `LIBXS_VPACK_GROUP`) is
stored at the width its largest magnitude needs, so groups of small values or
zeros cost little. Values are taken as integers of `maxbits` bits; signed types
fold the sign so that small magnitudes of either sign stay narrow. The stream
records the group size and the widths, so unpacking needs only `n` and the
number of words available (`npacked`, checked against the stream in debug
builds only).

`libxs_vpack` returns the number of words written, at most
`libxs_vpack_bound(n, maxbits, group)`, and `libxs_vunpack` the number of words
consumed. Both return zero for invalid arguments; `libxs_vunpack` also returns
zero for an empty stream (`npacked` of zero) and leaves the values untouched.

The [pack sample](samples/libxs_pack.md) measures both formats on synthetic
data or on MHD files.
