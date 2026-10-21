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

Without an argument the input is `Who is Alice? Alice saw 123.`; several arguments
are joined by single spaces, and `-` reads stdin.

The input is encoded at the native, word and syllable granularity in turn. Each
prints a line `NAME: input-bytes=N cells=N`, followed by one line per token with its
cell offset, kind, logical byte length, physical cell count, and sentence-boundary
flag. A nonzero exit status means encoding or round-trip verification failed.
