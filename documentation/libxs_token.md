# Metatokens and Lexemes

Header: `libxs_token.h`

The token module provides two representations:

- reversible metatokens for prediction and exact text reconstruction;
- vocabulary-backed lexemes for interpretation, matching, and rules.

## Metatoken Layout

`libxs_token_t` is one fixed 8-byte physical cell:

```text
bit 7     sentence boundary on the final cell
bit 6..4  generic payload kind
bit 3     another cell continues this logical metatoken
bit 2..0  payload bytes in this cell, 1..7
bytes 1..7 payload
```

A logical metatoken may occupy several cells. Long text units therefore do not
require a larger struct or an external vocabulary. `libxs_token_span` validates
the continuation chain, while `libxs_token_read` reconstructs its payload and
metadata.

Payload kinds are independent of segmentation policy: word and syllable
settings both emit `LIBXS_TOKEN_TEXT`. Their boundaries differ, but consumers
use the same read and decode APIs.

## Tokenizer Configuration

Granularity belongs to the tokenizer, not to each encode call:

```c
libxs_tokenizer_t* tokenizer =
  libxs_tokenizer_create(LIBXS_TOKEN_GRANULARITY_WORD);
libxs_token_stream_t stream;

libxs_token_stream_init(&stream);
libxs_token_stream_encode(tokenizer, &stream, text, size);
libxs_tokenizer_set_granularity(tokenizer,
  LIBXS_TOKEN_GRANULARITY_SYLLABLE);
```

Available policies are native chunks, words, and syllables. Encoding is
otherwise policy-neutral, and every policy preserves all input bytes.

The stream is self-describing. Decoding does not require the tokenizer or a
lexicon:

```c
libxs_token_stream_decode(&stream, &decoded, &decoded_size);
```

For every successful encoding:

```text
decode(encode(text)) == text
```

including whitespace, spelling, case, punctuation, numbers, and UTF-8 bytes.

## Logical Iteration

Physical cells are traversed by logical span:

```c
size_t pos = 0;
while (pos < stream.size) {
  libxs_token_info_t info;
  unsigned char payload[64];
  if (EXIT_SUCCESS != libxs_token_read(stream.data, stream.size, pos,
    payload, sizeof(payload), &info)) break;
  pos += info.cells;
}
```

`info.kind` describes text, number, whitespace, punctuation, markup, literal,
reference, or control payload. `info.is_sentence` is carried by the final cell
of a logical metatoken.

## Lexical API

`libxs_lexeme_t` remains an independent 8-byte lexical occurrence:

```c
typedef struct libxs_lexeme_t {
  unsigned int id;
  unsigned short length;
  unsigned short flags;
} libxs_lexeme_t;
```

`libxs_lexeme_stream_encode` normalizes words, maps them through a persistent
lexicon, applies caller-owned normalization and classification rules, and
emits stable IDs. This representation is intentionally interpretive rather
than reversible. It is suitable for grounded QA, lexical matching, and rule
evaluation, but raw-text likelihood should use metatokens.
