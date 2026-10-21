# Pack

Measures bit packing (`libxs_pack.h`) at a fixed width against a width per
group of values (`libxs_vpack`): bits per value, ratio, and time per value.
Every round trip is verified.

## Build

```bash
cd samples/pack
make
```

## Usage

```bash
./pack.x [nvalues [nbits [range [zeros [nrepeat]]]]]
./pack.x FILE.mhd [FILE.mhd ...] [eps]
```

Without a file, the values are synthetic: signed integers of up to `nbits`
bits whose magnitudes are log-uniform over `range` bits per block of 512
values, with `zeros` percent set to zero.

| Positional | Default | Description                                  |
|------------|---------|----------------------------------------------|
| nvalues    | 1048576 | Number of values                             |
| nbits      | 24      | Width of the fixed format (2 to 63)          |
| range      | 12      | Spread of magnitudes in bits                 |
| zeros      | 10      | Percentage of zeros                          |
| nrepeat    | 10      | Repetitions per measurement                  |

With MHD files (any element type), the values are measured in blocks of 512
and summed over all files. Integer data is packed as it is. Floating-point
data is quantized to multiples of `eps` (values not above `eps` become zero),
and each block gets the width its largest value needs. A file whose MHD
extension holds a REAL(8) eps, an INT32 number of blocks, and an INT32 shape(3)
per block provides eps and the blocks itself; CP2K writes such files
(`CP2K_COMPRESS_DUMP=prefix`, from its tensor compression).

The output compares the fixed width with a width per group of 4 to 32 values,
with deltas between neighbors, and for floating-point data with a quantization
step of twice `eps`, plus the bits that a width per value would need.
With MHD files, a second table splits each group size into the payload and
the cost of storing the widths: one byte per width, 7 bits per width, a base
per 8 groups with 3- or 2-bit offsets (`base+3`, `base+2`), and the entropy
of the widths as a bound. Each row gives bits per value and the ratio.
Synthetic data shows speed; decisions about the format rest on real data.
