# Tokenizer

Header: `libxs_token.h`

The tokenizer provides reversible metatokens and vocabulary-backed lexemes.
These representations serve separate APIs:

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

| Macro | Value | Meaning |
|:------|:------|:--------|
| `LIBXS_TOKEN_BYTES` | 8 | Size of one cell |
| `LIBXS_TOKEN_PAYLOAD_BYTES` | 7 | Payload bytes per cell |
| `LIBXS_TOKEN_LENGTH_MASK` | `0x07` | Payload length in the control byte |
| `LIBXS_TOKEN_CONTINUED` | `0x08` | Continuation bit |
| `LIBXS_TOKEN_KIND_MASK`, `LIBXS_TOKEN_KIND_SHIFT` | `0x70`, 4 | Payload kind |
| `LIBXS_TOKEN_SENTENCE` | `0x80` | Sentence boundary |

Payload kinds (`enum libxs_token_kind_t`, 0 to 7): `LIBXS_TOKEN_LITERAL`,
`LIBXS_TOKEN_TEXT`, `LIBXS_TOKEN_NUMBER`, `LIBXS_TOKEN_SPACE`,
`LIBXS_TOKEN_PUNCT`, `LIBXS_TOKEN_MARKUP`, `LIBXS_TOKEN_REFERENCE`, and
`LIBXS_TOKEN_CONTROL`.

Encoder-produced cells are zero-initialized, including unused payload bytes.
Binary comparison therefore remains deterministic when control metadata matters.
For content matching, use `libxs_token_payload_equal` on physical cells or
`libxs_token_payload_match` on complete logical metatokens. These functions
compare declared payload length and bytes while ignoring kind, continuation,
sentence flags, unused trailing bytes, and (for logical matching) cell layout.
Do not infer payload length from trailing zeros; zero is a valid payload byte.

Payload kinds are independent of segmentation policy: word and syllable
settings both emit `LIBXS_TOKEN_TEXT`. Their boundaries differ, but consumers
use the same read and decode APIs.

## Tokenizer Configuration

Granularity belongs to the tokenizer, not to each encode call:

```C
libxs_tokenizer_t* tokenizer =
  libxs_tokenizer_create(LIBXS_TOKEN_GRANULARITY_WORD);
libxs_token_stream_t stream;

libxs_token_stream_init(&stream);
libxs_token_stream_encode(tokenizer, &stream, text, size);
libxs_tokenizer_set_granularity(tokenizer,
  LIBXS_TOKEN_GRANULARITY_SYLLABLE);
```

Available policies (`enum libxs_token_granularity_t`) are
`LIBXS_TOKEN_GRANULARITY_NATIVE` (0), `_WORD` (1), and `_SYLLABLE` (2). Encoding
is otherwise policy-neutral, and every policy preserves all input bytes.

```C
libxs_tokenizer_t* libxs_tokenizer_create(int granularity);
void libxs_tokenizer_destroy(libxs_tokenizer_t* tokenizer);
int libxs_tokenizer_set_granularity(libxs_tokenizer_t* tokenizer,
  int granularity);
int libxs_tokenizer_granularity(const libxs_tokenizer_t* tokenizer);
```

`destroy` accepts NULL. `set_granularity` returns `EXIT_SUCCESS` or
`EXIT_FAILURE`; `granularity` returns the policy, or -1 for NULL.

```C
typedef struct libxs_token_stream_t {
  libxs_token_t* data;
  size_t size;
  size_t capacity;
} libxs_token_stream_t;

void libxs_token_stream_init(libxs_token_stream_t* stream);
int libxs_token_stream_reserve(libxs_token_stream_t* stream,
  size_t capacity);
int libxs_token_stream_push(libxs_token_stream_t* stream,
  const libxs_token_t* token);
void libxs_token_stream_release(libxs_token_stream_t* stream);
int libxs_token_stream_encode(const libxs_tokenizer_t* tokenizer,
  libxs_token_stream_t* stream, const unsigned char* text, size_t size);
int libxs_token_stream_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size);
```

A growable array of cells. `release` frees the memory, not the struct.
`encode` appends to the stream. All `int` functions return `EXIT_SUCCESS` or
`EXIT_FAILURE`.

The stream is self-describing. Decoding does not require the tokenizer or a
lexicon:

```C
libxs_token_stream_decode(&stream, &decoded, &decoded_size);
```

The decoded bytes are allocated with `malloc` and zero-terminated; the caller
releases them with `free`.

For every successful encoding:

```text
decode(encode(text)) == text
```

including whitespace, spelling, case, punctuation, numbers, and UTF-8 bytes.

## Logical Iteration

Physical cells are traversed by logical span:

```C
size_t pos = 0;
while (pos < stream.size) {
  libxs_token_info_t info;
  if (EXIT_SUCCESS != libxs_token_read(stream.data, stream.size, pos,
    NULL, 0, &info)) break;
  pos += info.cells;
}
```

A NULL payload reads metadata only. Otherwise, the capacity must cover
`info.length` bytes, or the read fails.

```C
typedef struct libxs_token_info_t {
  size_t length;    /* logical payload bytes */
  size_t cells;     /* physical cells */
  int kind;
  int continued;
  int is_sentence;
} libxs_token_info_t;

int libxs_token_read(const libxs_token_t* tokens,
  size_t ntokens, size_t pos, unsigned char* payload, size_t capacity,
  libxs_token_info_t* info);
size_t libxs_token_span(const libxs_token_t* tokens,
  size_t ntokens, size_t pos, size_t* payload_size);
void libxs_token_info(const libxs_token_t* token,
  libxs_token_info_t* info);
int libxs_token_payload_equal(const libxs_token_t* lhs,
  const libxs_token_t* rhs);
int libxs_token_payload_match(const libxs_token_t* lhs,
  size_t nlhs, size_t lhs_pos, const libxs_token_t* rhs,
  size_t nrhs, size_t rhs_pos);
```

- `read` returns `EXIT_SUCCESS` only for a complete, valid chain.
- `span` returns the number of cells of the metatoken at `pos` (0 for an invalid
  position or chain); `payload_size` (may be NULL) receives its payload size.
- `info` decodes the control byte of a single cell.
- `payload_equal` and `payload_match` return non-zero if the payloads match.

Inline helpers on a single cell (NULL-safe):

```C
size_t libxs_token_len(const libxs_token_t* token);           /* payload bytes */
int libxs_token_kind(const libxs_token_t* token);
int libxs_token_is_continued(const libxs_token_t* token);
int libxs_token_is_sentence_end(const libxs_token_t* token);
```

`info.kind` describes text, number, whitespace, punctuation, markup, literal,
reference, or control payload. `info.is_sentence` is carried by the final cell
of a logical metatoken.

## Lexical API

`libxs_lexeme_t` is an 8-byte (`LIBXS_LEXEME_BYTES`) lexical occurrence:

```C
typedef struct libxs_lexeme_t {
  unsigned int id;
  unsigned short length;
  unsigned short flags;
} libxs_lexeme_t;
```

`libxs_lexeme_stream_encode` normalizes words, maps them through a persistent
lexicon, applies caller-owned normalization and classification rules, and
emits stable IDs. The result is not reversible: use metatokens for exact text
reconstruction or raw-text likelihood.

```C
int libxs_lexeme_stream_encode(libxs_lexicon_t* lexicon,
  libxs_lexeme_stream_t* stream, const unsigned char* text, size_t size,
  const libxs_lexrule_t* rules, int nrules,
  const libxs_lexnorm_t* norms, int nnorms, int create);
```

Words are lowercased, then rewritten by `norms` (may be NULL), numbers become
`<num>`, and punctuation is kept. Class flags come from built-in detection and
then `rules` (may be NULL). A non-zero `create` interns unknown words into the
lexicon; otherwise they receive id 0. The lexemes are appended to `stream`.
Returns `EXIT_SUCCESS` or `EXIT_FAILURE`.

```C
typedef struct libxs_lexnorm_t {
  char from[LIBXS_LEXEME_MAXBYTES + 1];
  char to[LIBXS_LEXEME_MAXBYTES + 1];
} libxs_lexnorm_t;
```

Maps normalized `from` text to `to`. `to` may hold several words separated by
`LIBXS_LEXNORM_SEPARATOR` (space), e.g., "is not": only the first emitted
lexeme carries the source length, the others carry length zero. Normalized text
is at most `LIBXS_LEXEME_MAXBYTES` (63) bytes.

Lexeme flags (`libxs_lexeme_t::flags`):

| Flag | Meaning |
|:-----|:--------|
| `LIBXS_LEXEME_WORD` | Word |
| `LIBXS_LEXEME_NUMBER` | Number |
| `LIBXS_LEXEME_PUNCT` | Punctuation |
| `LIBXS_LEXEME_MARKUP` | Markup |
| `LIBXS_LEXEME_SENTENCE` | Sentence-ending punctuation |
| `LIBXS_LEXEME_QUESTION` | Question mark or question word (default rules) |
| `LIBXS_LEXEME_STOP` | Stop word (default rules) |
| `LIBXS_LEXEME_ENTITY` | Uppercase-initial word (default rules) |
| `LIBXS_LEXEME_BREAK` | Preceded by whitespace (starts a source word) |

```C
typedef struct libxs_lexeme_info_t {
  unsigned int id;
  size_t length;
  unsigned int flags;
  int is_word, is_number, is_punct, has_break, is_sentence;
  int is_question, is_stop, is_entity, is_markup;
} libxs_lexeme_info_t;

void libxs_lexeme_info(const libxs_lexeme_t* lexeme,
  libxs_lexeme_info_t* info);
size_t libxs_lexeme_word_next(const libxs_lexeme_t* lexemes,
  size_t nlexemes, size_t pos);
int libxs_lexeme_is_word_char(const unsigned char* text, size_t size,
  int* length);

size_t libxs_lexeme_len(const libxs_lexeme_t* lexeme);           /* inline */
int libxs_lexeme_has_break(const libxs_lexeme_t* lexeme);        /* inline */
int libxs_lexeme_is_sentence_end(const libxs_lexeme_t* lexeme);  /* inline */
```

- `info` decodes all flags of a lexeme.
- `word_next` returns the number of lexemes of the source word at `pos` (up to
  the next lexeme with `LIBXS_LEXEME_BREAK`).
- `is_word_char` returns non-zero if `text[0]` starts a word character (UTF-8
  letters included, encoded punctuation excluded); `length` (may be NULL)
  receives its byte count. Use it instead of `ctype` when scanning words.

Lexeme streams mirror metatoken streams (`libxs_lexeme_stream_t` with `data`,
`size`, `capacity`):

```C
void libxs_lexeme_stream_init(libxs_lexeme_stream_t* stream);
int libxs_lexeme_stream_reserve(libxs_lexeme_stream_t* stream,
  size_t capacity);
int libxs_lexeme_stream_push(libxs_lexeme_stream_t* stream,
  const libxs_lexeme_t* lexeme);
void libxs_lexeme_stream_release(libxs_lexeme_stream_t* stream);
```

## Lexicon

```C
libxs_lexicon_t* libxs_lexicon_create(void);
void libxs_lexicon_destroy(libxs_lexicon_t* lexicon);
unsigned int libxs_lexicon_size(const libxs_lexicon_t* lexicon);
unsigned int libxs_lexicon_id(libxs_lexicon_t* lexicon,
  const char* text, int length, unsigned int flags, int create);
const char* libxs_lexicon_text(const libxs_lexicon_t* lexicon,
  unsigned int id, int* length, unsigned int* flags);
unsigned int libxs_lexicon_count(const libxs_lexicon_t* lexicon,
  unsigned int id);
int libxs_lexicon_save(const libxs_lexicon_t* lexicon,
  void* buffer, size_t* size);
libxs_lexicon_t* libxs_lexicon_load(const void* buffer, size_t size);
```

Persistent vocabulary of normalized text; id 0 is reserved for unknown text.

- `destroy` accepts NULL. `size` excludes id 0.
- `id` maps text to its id; a non-zero `create` interns missing text, otherwise
  missing text yields 0.
- `text` returns the text of an id (not terminated, length via `length`) and its
  flags, or NULL.
- `count` returns how often the text was interned (0 for an unknown id).
- `save` follows `libxs_registry_save`: pass `buffer = NULL` to query the size,
  `*size` is the capacity on input and the bytes written on output.
- `load` returns NULL on failure, including a vocabulary written by a different
  library version.

## Rules

Rules are fixed-size records that select a built-in template, supply an
argument, and name an action. They can be stored in a registry and thereby
serialized.

```C
typedef struct libxs_textrule_t {
  unsigned char tmpl;
  unsigned char action;
  unsigned char reserved[2];
  unsigned int argument;
  unsigned char pad[8];
} libxs_textrule_t;

typedef struct libxs_textrule_ctx_t {
  const unsigned char* text;
  size_t text_size;
  int byte_pos;
  const libxs_lexeme_t* token;
  const libxs_lexeme_t* prev_token;
} libxs_textrule_ctx_t;

int libxs_textrule_eval(const libxs_textrule_ctx_t* ctx,
  const libxs_textrule_t* rules, int nrules);
unsigned int libxs_textrule_wordhash(
  const unsigned char* word, int len);
int libxs_textrule_load(const libxs_registry_t* registry,
  libxs_textrule_t* rules, int max_rules);
int libxs_textrule_save(libxs_registry_t* registry,
  const libxs_textrule_t* rules, int nrules);
int libxs_textrule_defaults(libxs_textrule_t* rules, int max_rules);
```

Text rules decide sentence boundaries. `eval` returns 1 (boundary confirmed) or
0 (suppressed); a token without the sentence flag returns 0, and the last
matching rule wins. Templates (`tmpl`):

| Template | Matches if |
|:---------|:-----------|
| `LIBXS_TRULE_PREV_WORD_SHORT_UPPER` | Word before the position is short (<= 4) and uppercase-initial |
| `LIBXS_TRULE_NEXT_CHAR_UPPER` | Character after the following whitespace is uppercase |
| `LIBXS_TRULE_NEXT_CHAR_LOWER` | Character after the following whitespace is lowercase |
| `LIBXS_TRULE_PREV_TOKEN_BREAK` | Previous token has `LIBXS_LEXEME_BREAK` |
| `LIBXS_TRULE_INSIDE_DELIMITERS` | Position is inside paired delimiters (`argument` is the opening character) |

Actions are `LIBXS_TRULE_SUPPRESS` and `LIBXS_TRULE_CONFIRM`. `wordhash`
produces the `argument` for word matching.

```C
typedef struct libxs_lexrule_t {
  unsigned char tmpl;
  unsigned char action;
  unsigned short flags;
  unsigned int argument;
  unsigned char pad[8];
} libxs_lexrule_t;

typedef struct libxs_lexrule_ctx_t {
  const unsigned char* text;
  int length;
  unsigned int flags;
  unsigned int hash;
} libxs_lexrule_ctx_t;

unsigned int libxs_lexrule_eval(const libxs_lexrule_ctx_t* ctx,
  const libxs_lexrule_t* rules, int nrules);
int libxs_lexrule_load(const libxs_registry_t* registry,
  libxs_lexrule_t* rules, int max_rules);
int libxs_lexrule_save(libxs_registry_t* registry,
  const libxs_lexrule_t* rules, int nrules);
int libxs_lexrule_defaults(libxs_lexrule_t* rules, int max_rules);
```

Lexical rules set (`LIBXS_LRULE_SET`) or clear (`LIBXS_LRULE_CLEAR`) the rule's
`flags` on a matching lexeme; `eval` returns the resulting flags. Templates are
`LIBXS_LRULE_WORD_HASH` (normalized word by `libxs_textrule_wordhash`),
`LIBXS_LRULE_WORD_INITIAL_UPPER` (original spelling starts uppercase), and
`LIBXS_LRULE_PUNCT_CHAR` (punctuation by character).

For both rule kinds, `load` returns the number of rules read from the registry
(keys prefixed `TRULE:` or `LRULE:`), `save` returns `EXIT_SUCCESS` or
`EXIT_FAILURE`, and `defaults` fills in the built-in rules and returns their
count.

## Text Reflow

```C
int libxs_text_reflow(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size);
int libxs_text_reflow_map(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size, size_t** line_offsets, size_t* nlines);
```

Replace cosmetic newlines (column wrapping) with spaces while keeping
structural ones (enumerations, headings, blank lines, verse). `*out` is
allocated and must be released with `free`. `reflow_map` also allocates
`*line_offsets` with `*nlines` entries, where entry k is the offset in `*out`
at which input line k+1 begins; both may be NULL to skip the map. Returns
`EXIT_SUCCESS` or `EXIT_FAILURE`.
