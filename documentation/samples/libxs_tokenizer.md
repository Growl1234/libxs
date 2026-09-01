# Metatoken Sample

The sample encodes the same input with each `libxs_tokenizer_t` granularity,
prints the resulting logical metatokens, and verifies exact decoding.

## Build

```bash
cd ../..
make PEDANTIC=2
make -C samples/tokenizer PEDANTIC=2
```

## Usage

```bash
./samples/tokenizer/tokenizer.x
./samples/tokenizer/tokenizer.x "token tokenization tokenizer token"
printf '%s\n' 'UTF-8 text here' | ./samples/tokenizer/tokenizer.x -
```

Output reports the input byte count followed by each token's kind, logical byte
length, physical cell count, and sentence-boundary flag. A nonzero exit status
means encoding or round-trip verification failed.
