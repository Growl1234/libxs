#include <libxs/libxs.h>

#include "converse.h"
#include "converse_hier.h"

#include <math.h>

#define HIER_KEY_MAX (COMPOSE_MAXTEXT + 1)
#define HIER_ESCAPE_TEXT 1u
#define HIER_ESCAPE_NATIVE 2u
#define HIER_SYMBOL_FIRST 3u
#define HIER_SYLLABLE_ESCAPE 1u
#define HIER_SYLLABLE_END 2u
#define HIER_SYLLABLE_FIRST 3u
#define HIER_BYTE_END 257u
#define HIER_BYTE_VOCAB 257u
#define HIER_HISTORY_START (~0u)
#define HIER_CLOCK_WORD_BASE 0x10000000u
#define HIER_CLOCK_SYLLABLE_BASE 0x20000000u
#define HIER_CLOCK_BYTE_START 0x30000000u
#define HIER_CLOCK_RAW_MAX 4
#define HIER_RECURRENT_BASE 0x40000000u
#define HIER_RECURRENT_DIM 8
#define HIER_RECURRENT_RAW 2
#define HIER_RECURRENT_ORDER 3


typedef struct hier_symbol_t {
  long count;
  unsigned int id;
} hier_symbol_t;

typedef struct hier_eval_t {
  long ntokens;
  long ntop1;
  long ntext_escape;
  long nnative_escape;
  long nsyllable_escape;
  long ndeep;
  long nshallow;
  double bits;
  double bytes;
  double top_bits;
  double syllable_bits;
  double byte_bits;
  double deep_bits;
  double deep_bytes;
  double shallow_bits;
  double shallow_bytes;
} hier_eval_t;

typedef struct hier_clock_eval_t {
  long nbytes;
  long nppm;
  long nraw_top1;
  long ncontext_top1;
  long nppm_top1;
  long nppm_top3;
  long nadaptive_top1;
  long nadaptive_top3;
  long ndeep;
  long nshallow;
  double raw_bits;
  double context_bits;
  double mix_bits;
  double deep_bits;
  double shallow_bits;
  double raw_ppm_bits;
  double context_ppm_bits;
  double ppm_mix_bits;
  double recurrent_bits;
  double recurrent_mix_bits;
  double frozen_interp_bits;
  double adaptive_bits;
} hier_clock_eval_t;

typedef struct hier_ppm_key_t {
  int order;
  unsigned int context[LIBXS_NGRAM_ORDER_MAX];
} hier_ppm_key_t;

typedef struct hier_ppm_pair_t {
  hier_ppm_key_t key;
  unsigned int next;
} hier_ppm_pair_t;

typedef struct hier_ppm_stats_t {
  unsigned int total;
  unsigned int distinct;
  unsigned int seen[8];
  double backoff_norm;
} hier_ppm_stats_t;

typedef struct hier_ppm_t {
  libxs_registry_t* contexts;
  libxs_registry_t* pairs;
  unsigned int unigram[256];
  double total;
  int maxorder;
} hier_ppm_t;

struct converse_hier_t {
  libxs_registry_t* symbols;
  libxs_registry_t* syllables;
  libxs_tokenizer_t* word_tokenizer;
  libxs_tokenizer_t* syllable_tokenizer;
  libxs_ngram_t word_model;
  libxs_ngram_t syllable_model;
  libxs_ngram_t byte_model;
  libxs_ngram_t stream_byte_model;
  libxs_ngram_t clock_byte_model;
  hier_ppm_t stream_ppm;
  hier_ppm_t clock_ppm;
  hier_ppm_t recurrent_ppm;
  hier_ppm_t adaptive_ppm;
  unsigned int word_vocab;
  unsigned int syllable_vocab;
  int maxorder;
  int clock_order;
  int mincount;
  int top_stride;
  double recurrent_decay;
  int ready;
};


static unsigned int hier_recurrent_code(const double state[])
{
  unsigned int result = 0;
  int dim;
  for (dim = 0; dim < HIER_RECURRENT_DIM; ++dim) {
    if (state[dim] >= 0.0) result |= 1u << dim;
  }
  return HIER_RECURRENT_BASE + result;
}


static void hier_recurrent_update(double state[], unsigned int byte,
  double decay)
{
  double next[HIER_RECURRENT_DIM];
  int dim;
  for (dim = 0; dim < HIER_RECURRENT_DIM; ++dim) {
    const unsigned int hash = (byte + 1u) * 2654435761u
      + (unsigned int)(dim + 1) * 2246822519u;
    const double feature = (0 != (hash & 0x80000000u)) ? 1.0 : -1.0;
    next[dim] = decay * state[(dim + 1) % HIER_RECURRENT_DIM] + feature;
  }
  for (dim = 0; dim < HIER_RECURRENT_DIM; ++dim) state[dim] = next[dim];
}


static int hier_recurrent_history(const double state[],
  const unsigned int raw_history[], int raw_length, unsigned int history[])
{
  int keep = raw_length;
  int result = 1;
  int pos;
  if (keep > HIER_RECURRENT_RAW) keep = HIER_RECURRENT_RAW;
  history[0] = hier_recurrent_code(state);
  for (pos = raw_length - keep; pos < raw_length; ++pos) {
    history[result++] = raw_history[pos];
  }
  return result;
}


static int hier_ppm_create(hier_ppm_t* model, int maxorder)
{
  int result = EXIT_FAILURE;
  if (NULL != model && maxorder >= 1
    && maxorder <= LIBXS_NGRAM_ORDER_MAX)
  {
    memset(model, 0, sizeof(*model));
    model->contexts = libxs_registry_create();
    model->pairs = libxs_registry_create();
    model->maxorder = maxorder;
    if (NULL != model->contexts && NULL != model->pairs) result = EXIT_SUCCESS;
  }
  return result;
}


static void hier_ppm_destroy(hier_ppm_t* model)
{
  if (NULL != model) {
    libxs_registry_destroy(model->contexts);
    libxs_registry_destroy(model->pairs);
    memset(model, 0, sizeof(*model));
  }
}


static void hier_ppm_key(hier_ppm_key_t* key,
  const unsigned int history[], int history_length, int order)
{
  int pos;
  memset(key, 0, sizeof(*key));
  key->order = order;
  for (pos = 0; pos < order; ++pos) {
    key->context[pos] = history[history_length - order + pos];
  }
}


static void hier_ppm_observe(hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next)
{
  int order;
  if (NULL == model || NULL == history || next < 1 || next > 256) return;
  ++model->unigram[next - 1];
  model->total += 1.0;
  for (order = 1; order <= model->maxorder && order <= history_length;
    ++order)
  {
    hier_ppm_key_t key;
    hier_ppm_pair_t pair;
    hier_ppm_stats_t* stats;
    unsigned int* count;
    hier_ppm_key(&key, history, history_length, order);
    pair.key = key;
    pair.next = next;
    stats = (hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    count = (unsigned int*)libxs_registry_get(model->pairs,
      &pair, sizeof(pair), NULL);
    if (NULL == stats) {
      hier_ppm_stats_t fresh;
      memset(&fresh, 0, sizeof(fresh));
      fresh.total = 1;
      fresh.distinct = 1;
      fresh.seen[(next - 1) / 32] |= 1u << ((next - 1) % 32);
      libxs_registry_set(model->contexts, &key, sizeof(key), &fresh,
        sizeof(fresh), NULL);
    }
    else {
      ++stats->total;
      if (NULL == count) {
        ++stats->distinct;
        stats->seen[(next - 1) / 32] |= 1u << ((next - 1) % 32);
      }
    }
    if (NULL != count) ++*count;
    else {
      const unsigned int one = 1;
      libxs_registry_set(model->pairs, &pair, sizeof(pair), &one,
        sizeof(one), NULL);
    }
  }
}


static double hier_ppm_unigram(const hier_ppm_t* model, unsigned int next)
{
  const double alpha = 0.5;
  double result = 1.0 / 256.0;
  if (NULL != model && next >= 1 && next <= 256 && model->total > 0.0) {
    result = ((double)model->unigram[next - 1] + alpha)
      / (model->total + alpha * 256.0);
  }
  return result;
}


static double hier_ppm_prob_order(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next,
  int order)
{
  double result;
  if (order <= 0) return hier_ppm_unigram(model, next);
  { hier_ppm_key_t key;
    hier_ppm_pair_t pair;
    const hier_ppm_stats_t* stats;
    const unsigned int* count;
    const double lower = hier_ppm_prob_order(model, history, history_length,
      next, order - 1);
    hier_ppm_key(&key, history, history_length, order);
    pair.key = key;
    pair.next = next;
    stats = (const hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    count = (const unsigned int*)libxs_registry_get(model->pairs,
      &pair, sizeof(pair), NULL);
    if (NULL != stats && stats->total > 0) {
      const double denom = (double)stats->total + (double)stats->distinct;
      if (NULL != count) result = (double)*count / denom;
      else if (stats->backoff_norm > 0.0) {
        result = ((double)stats->distinct / denom)
          * lower / stats->backoff_norm;
      }
      else result = lower;
    }
    else result = lower;
  }
  return result;
}


static void hier_ppm_finalize(hier_ppm_t* model)
{
  int order;
  if (NULL == model) return;
  for (order = 1; order <= model->maxorder; ++order) {
    const void* registry_key = NULL;
    size_t cursor = 0;
    void* value = libxs_registry_begin(model->contexts, &registry_key,
      &cursor);
    while (NULL != value) {
      const hier_ppm_key_t* key = (const hier_ppm_key_t*)registry_key;
      hier_ppm_stats_t* stats = (hier_ppm_stats_t*)value;
      if (key->order == order) {
        double seen_mass = 0.0;
        unsigned int id;
        for (id = 1; id <= 256; ++id) {
          if (0 != (stats->seen[(id - 1) / 32] & (1u << ((id - 1) % 32)))) {
            seen_mass += hier_ppm_prob_order(model, key->context, order, id,
              order - 1);
          }
        }
        stats->backoff_norm = 1.0 - seen_mass;
        if (stats->backoff_norm < 1e-15) stats->backoff_norm = 1e-15;
      }
      value = libxs_registry_next(model->contexts, &registry_key, &cursor);
    }
  }
}


static double hier_ppm_prob(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next)
{
  int order;
  if (NULL == model || NULL == history || next < 1 || next > 256) return 0.0;
  order = model->maxorder;
  if (order > history_length) order = history_length;
  return hier_ppm_prob_order(model, history, history_length, next, order);
}


static double hier_ppm_interp_prob(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next)
{
  double result;
  int order;
  if (NULL == model || NULL == history || next < 1 || next > 256) return 0.0;
  result = hier_ppm_unigram(model, next);
  for (order = 1; order <= model->maxorder && order <= history_length;
    ++order)
  {
    hier_ppm_key_t key;
    hier_ppm_pair_t pair;
    const hier_ppm_stats_t* stats;
    const unsigned int* count;
    hier_ppm_key(&key, history, history_length, order);
    pair.key = key;
    pair.next = next;
    stats = (const hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    count = (const unsigned int*)libxs_registry_get(model->pairs,
      &pair, sizeof(pair), NULL);
    if (NULL != stats && stats->total > 0) {
      const double denom = (double)stats->total + (double)stats->distinct;
      const double observed = (NULL != count) ? (double)*count : 0.0;
      result = observed / denom + (double)stats->distinct / denom * result;
    }
  }
  return result;
}


static int hier_ppm_rank(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int target,
  int interpolate)
{
  unsigned int best_id[3] = { 0, 0, 0 };
  double best_probability[3] = { -1.0, -1.0, -1.0 };
  unsigned int id;
  int result = 0;
  for (id = 1; id <= 256; ++id) {
    const double probability = (0 != interpolate)
      ? hier_ppm_interp_prob(model, history, history_length, id)
      : hier_ppm_prob(model, history, history_length, id);
    int slot;
    for (slot = 0; slot < 3; ++slot) {
      if (probability > best_probability[slot]) break;
    }
    if (slot < 3) {
      int move;
      for (move = 2; move > slot; --move) {
        best_probability[move] = best_probability[move - 1];
        best_id[move] = best_id[move - 1];
      }
      best_probability[slot] = probability;
      best_id[slot] = id;
    }
  }
  if (best_id[0] == target) result = 1;
  else if (best_id[1] == target) result = 2;
  else if (best_id[2] == target) result = 3;
  return result;
}


static int hier_ppm_check(const hier_ppm_t* model)
{
  int result = EXIT_SUCCESS;
  const void* registry_key = NULL;
  size_t cursor = 0;
  int checked = 0;
  void* value;
  double mass = 0.0;
  unsigned int id;
  for (id = 1; id <= 256; ++id) mass += hier_ppm_unigram(model, id);
  if (fabs(mass - 1.0) > 1e-10) result = EXIT_FAILURE;
  value = libxs_registry_begin(model->contexts, &registry_key, &cursor);
  while (EXIT_SUCCESS == result && NULL != value && checked < 128) {
    const hier_ppm_key_t* key = (const hier_ppm_key_t*)registry_key;
    mass = 0.0;
    for (id = 1; id <= 256; ++id) {
      mass += hier_ppm_prob_order(model, key->context, key->order, id,
        key->order);
    }
    if (fabs(mass - 1.0) > 1e-10) result = EXIT_FAILURE;
    ++checked;
    value = libxs_registry_next(model->contexts, &registry_key, &cursor);
  }
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "PPM probability mass check failed: %.17g\n", mass);
  }
  return result;
}


static int hier_is_test(long index, int holdout, long corpus_size)
{
  int result = 0;
  const char* tail = getenv("CONVERSE_HOLDOUT_TAIL");
  if (holdout > 0) {
    if (NULL != tail && '0' != tail[0] && corpus_size > 0) {
      const long split = corpus_size - corpus_size / (long)holdout;
      result = (index >= split) ? 1 : 0;
    }
    else result = (0 == (index % (long)holdout)) ? 1 : 0;
  }
  return result;
}


static int hier_key(int kind, const unsigned char* payload, size_t length,
  unsigned char key[], size_t* key_size)
{
  int result = EXIT_FAILURE;
  if (NULL != payload && length > 0 && length + 1 <= HIER_KEY_MAX
    && NULL != key && NULL != key_size)
  {
    key[0] = (unsigned char)kind;
    memcpy(key + 1, payload, length);
    *key_size = length + 1;
    result = EXIT_SUCCESS;
  }
  return result;
}


static void hier_symbol_observe(libxs_registry_t* symbols, int kind,
  const unsigned char* payload, size_t length)
{
  unsigned char key[HIER_KEY_MAX];
  size_t key_size = 0;
  if (NULL != symbols
    && EXIT_SUCCESS == hier_key(kind, payload, length, key, &key_size))
  {
    hier_symbol_t* symbol = (hier_symbol_t*)libxs_registry_get(symbols,
      key, key_size, NULL);
    if (NULL != symbol) ++symbol->count;
    else {
      hier_symbol_t fresh;
      fresh.count = 1;
      fresh.id = 0;
      libxs_registry_set(symbols, key, key_size, &fresh, sizeof(fresh), NULL);
    }
  }
}


static unsigned int hier_symbol_find(const libxs_registry_t* symbols,
  int kind, const unsigned char* payload, size_t length)
{
  unsigned int result = 0;
  unsigned char key[HIER_KEY_MAX];
  size_t key_size = 0;
  if (NULL != symbols
    && EXIT_SUCCESS == hier_key(kind, payload, length, key, &key_size))
  {
    const hier_symbol_t* symbol = (const hier_symbol_t*)libxs_registry_get(
      symbols, key, key_size, NULL);
    if (NULL != symbol) result = symbol->id;
  }
  return result;
}


static unsigned int hier_symbol_assign(libxs_registry_t* symbols,
  int mincount, unsigned int first)
{
  unsigned int result = first - 1;
  const void* key = NULL;
  size_t cursor = 0;
  void* value = libxs_registry_begin(symbols, &key, &cursor);
  while (NULL != value) {
    hier_symbol_t* symbol = (hier_symbol_t*)value;
    if (symbol->count >= mincount) {
      ++result;
      symbol->id = result;
    }
    value = libxs_registry_next(symbols, &key, &cursor);
  }
  return result;
}


static void hier_history_push(unsigned int history[], int* length,
  int capacity, unsigned int id)
{
  if (*length < capacity) history[(*length)++] = id;
  else {
    int pos;
    for (pos = 1; pos < capacity; ++pos) history[pos - 1] = history[pos];
    history[capacity - 1] = id;
  }
}


static void hier_ngram_observe(libxs_ngram_t* model,
  unsigned int history[], int* history_length, unsigned int id)
{
  libxs_ngram_observe(model, history, *history_length, id);
  hier_history_push(history, history_length, LIBXS_NGRAM_ORDER_MAX, id);
}


static int hier_read(const libxs_token_stream_t* stream, size_t token_pos,
  unsigned char payload[], libxs_token_info_t* info)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != payload && NULL != info
    && EXIT_SUCCESS == libxs_token_read(stream->data, stream->size,
      token_pos, payload, COMPOSE_MAXTEXT, info))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}


static void hier_count_word(converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  libxs_token_stream_t stream;
  size_t token_pos = 0;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->syllable_tokenizer,
    &stream, payload, length))
  {
    while (token_pos < stream.size) {
      unsigned char syllable[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, syllable, &info)) break;
      hier_symbol_observe(model->syllables, info.kind, syllable, info.length);
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static void hier_count_text(converse_hier_t* model,
  const char* text, int text_length)
{
  libxs_token_stream_t stream;
  size_t token_pos = 0;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->word_tokenizer,
    &stream, (const unsigned char*)text, (size_t)text_length))
  {
    while (token_pos < stream.size) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) break;
      hier_symbol_observe(model->symbols, info.kind, payload, info.length);
      if (LIBXS_TOKEN_TEXT == info.kind) {
        hier_count_word(model, payload, info.length);
      }
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static void hier_train_bytes(converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t pos;
  history[0] = HIER_HISTORY_START;
  for (pos = 0; pos < length; ++pos) {
    const unsigned int id = (unsigned int)payload[pos] + 1u;
    hier_ngram_observe(&model->byte_model, history, &history_length, id);
  }
  hier_ngram_observe(&model->byte_model, history, &history_length,
    HIER_BYTE_END);
}


static void hier_train_word(converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->syllable_tokenizer,
    &stream, payload, length))
  {
    while (token_pos < stream.size) {
      unsigned char syllable[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, syllable, &info)) break;
      id = hier_symbol_find(model->syllables, info.kind, syllable, info.length);
      if (0 == id) id = HIER_SYLLABLE_ESCAPE;
      hier_ngram_observe(&model->syllable_model, history, &history_length, id);
      hier_train_bytes(model, syllable, info.length);
      token_pos += info.cells;
    }
    hier_ngram_observe(&model->syllable_model, history, &history_length,
      HIER_SYLLABLE_END);
  }
  libxs_token_stream_release(&stream);
}


static void hier_train_text(converse_hier_t* model,
  const char* text, int text_length)
{
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->word_tokenizer,
    &stream, (const unsigned char*)text, (size_t)text_length))
  {
    while (token_pos < stream.size) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) break;
      id = hier_symbol_find(model->symbols, info.kind, payload, info.length);
      if (0 == id) {
        id = (LIBXS_TOKEN_TEXT == info.kind)
          ? HIER_ESCAPE_TEXT : HIER_ESCAPE_NATIVE;
      }
      hier_ngram_observe(&model->word_model, history, &history_length, id);
      if (LIBXS_TOKEN_TEXT == info.kind) {
        hier_train_word(model, payload, info.length);
      }
      else hier_train_bytes(model, payload, info.length);
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static unsigned int hier_clock_symbol(const converse_hier_t* model,
  const libxs_registry_t* symbols, int kind, const unsigned char* payload,
  size_t length, unsigned int base)
{
  unsigned int id = hier_symbol_find(symbols, kind, payload, length);
  if (0 == id) {
    if (symbols == model->symbols) {
      id = (LIBXS_TOKEN_TEXT == kind)
        ? HIER_ESCAPE_TEXT : HIER_ESCAPE_NATIVE;
    }
    else id = HIER_SYLLABLE_ESCAPE;
  }
  return base + id;
}


static int hier_clock_states(const converse_hier_t* model,
  const char* text, int text_length, const libxs_tokenizer_t* tokenizer,
  const libxs_registry_t* symbols, unsigned int base,
  unsigned int states[])
{
  int result = EXIT_FAILURE;
  libxs_token_stream_t stream;
  size_t token_pos = 0, byte_pos = 0;
  unsigned int previous = base;
  libxs_token_stream_init(&stream);
  if (NULL != model && NULL != text && text_length >= 0
    && NULL != tokenizer && NULL != symbols && NULL != states
    && EXIT_SUCCESS == libxs_token_stream_encode(tokenizer, &stream,
      (const unsigned char*)text, (size_t)text_length))
  {
    result = EXIT_SUCCESS;
    while (token_pos < stream.size && byte_pos < (size_t)text_length) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      size_t offset;
      unsigned int current;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) {
        result = EXIT_FAILURE;
        break;
      }
      current = hier_clock_symbol(model, symbols, info.kind, payload,
        info.length, base);
      for (offset = 0; offset < info.length
        && byte_pos < (size_t)text_length; ++offset)
      {
        states[byte_pos++] = previous;
      }
      previous = current;
      token_pos += info.cells;
    }
    if (byte_pos != (size_t)text_length) result = EXIT_FAILURE;
  }
  libxs_token_stream_release(&stream);
  return result;
}


static int hier_clock_history(const unsigned int raw_history[],
  int raw_length, unsigned int word_state, unsigned int syllable_state,
  unsigned int history[])
{
  int keep = raw_length;
  int result = 2;
  int pos;
  if (keep > HIER_CLOCK_RAW_MAX) keep = HIER_CLOCK_RAW_MAX;
  history[0] = word_state;
  history[1] = syllable_state;
  for (pos = raw_length - keep; pos < raw_length; ++pos) {
    history[result++] = raw_history[pos];
  }
  return result;
}


static void hier_train_clock_text(converse_hier_t* model,
  const char* text, int text_length)
{
  unsigned int word_state[COMPOSE_MAXTEXT];
  unsigned int syllable_state[COMPOSE_MAXTEXT];
  unsigned int raw_history[LIBXS_NGRAM_ORDER_MAX];
  int raw_length = 1;
  int pos;
  double recurrent[HIER_RECURRENT_DIM];
  memset(recurrent, 0, sizeof(recurrent));
  raw_history[0] = HIER_CLOCK_BYTE_START;
  if (text_length > 0 && text_length <= COMPOSE_MAXTEXT
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->word_tokenizer, model->symbols, HIER_CLOCK_WORD_BASE,
      word_state)
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->syllable_tokenizer, model->syllables,
      HIER_CLOCK_SYLLABLE_BASE, syllable_state))
  {
    for (pos = 0; pos < text_length; ++pos) {
      unsigned int history[LIBXS_NGRAM_ORDER_MAX];
      const unsigned int id = (unsigned int)(unsigned char)text[pos] + 1u;
      const int history_length = hier_clock_history(raw_history, raw_length,
        word_state[pos], syllable_state[pos], history);
      unsigned int recurrent_history[HIER_RECURRENT_ORDER];
      const int recurrent_length = hier_recurrent_history(recurrent,
        raw_history, raw_length, recurrent_history);
      libxs_ngram_observe(&model->clock_byte_model, history, history_length,
        id);
      libxs_ngram_observe(&model->stream_byte_model, raw_history, raw_length,
        id);
      hier_ppm_observe(&model->clock_ppm, history, history_length, id);
      hier_ppm_observe(&model->stream_ppm, raw_history, raw_length, id);
      hier_ppm_observe(&model->recurrent_ppm, recurrent_history,
        recurrent_length, id);
      hier_ppm_observe(&model->adaptive_ppm, raw_history, raw_length, id);
      hier_history_push(raw_history, &raw_length, LIBXS_NGRAM_ORDER_MAX, id);
      hier_recurrent_update(recurrent, (unsigned int)(unsigned char)text[pos],
        model->recurrent_decay);
    }
  }
}


static double hier_bits(const libxs_ngram_t* model,
  const unsigned int history[], int history_length, unsigned int id)
{
  double result = HUGE_VAL;
  const double probability = libxs_ngram_prob(model, history,
    history_length, id);
  if (probability > 0.0) result = -log(probability) / log(2.0);
  return result;
}


static double hier_score_bytes(const converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  double result = 0.0;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t pos;
  history[0] = HIER_HISTORY_START;
  for (pos = 0; pos < length; ++pos) {
    const unsigned int id = (unsigned int)payload[pos] + 1u;
    result += hier_bits(&model->byte_model, history, history_length, id);
    hier_history_push(history, &history_length, LIBXS_NGRAM_ORDER_MAX, id);
  }
  result += hier_bits(&model->byte_model, history, history_length,
    HIER_BYTE_END);
  return result;
}


static double hier_score_word(const converse_hier_t* model,
  const unsigned char* payload, size_t length, hier_eval_t* evaluation)
{
  double result = 0.0;
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->syllable_tokenizer,
    &stream, payload, length))
  {
    while (token_pos < stream.size) {
      unsigned char syllable[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      double bits;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, syllable, &info)) break;
      id = hier_symbol_find(model->syllables, info.kind, syllable, info.length);
      if (0 == id) id = HIER_SYLLABLE_ESCAPE;
      bits = hier_bits(&model->syllable_model, history, history_length, id);
      result += bits;
      evaluation->syllable_bits += bits;
      if (HIER_SYLLABLE_ESCAPE == id) {
        const double byte_bits = hier_score_bytes(model, syllable, info.length);
        result += byte_bits;
        evaluation->byte_bits += byte_bits;
        ++evaluation->nsyllable_escape;
      }
      hier_history_push(history, &history_length, LIBXS_NGRAM_ORDER_MAX, id);
      token_pos += info.cells;
    }
    { const double bits = hier_bits(&model->syllable_model, history,
        history_length, HIER_SYLLABLE_END);
      result += bits;
      evaluation->syllable_bits += bits;
    }
  }
  libxs_token_stream_release(&stream);
  return result;
}


static void hier_score_text(const converse_hier_t* model,
  const char* text, int text_length, hier_eval_t* evaluation)
{
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->word_tokenizer,
    &stream, (const unsigned char*)text, (size_t)text_length))
  {
    while (token_pos < stream.size) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      unsigned int top_ids[1];
      double bits;
      int deep;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) break;
      id = hier_symbol_find(model->symbols, info.kind, payload, info.length);
      if (0 == id) {
        id = (LIBXS_TOKEN_TEXT == info.kind)
          ? HIER_ESCAPE_TEXT : HIER_ESCAPE_NATIVE;
      }
      bits = hier_bits(&model->word_model, history, history_length, id);
      evaluation->top_bits += bits;
      if (HIER_ESCAPE_TEXT == id) {
        bits += hier_score_word(model, payload, info.length, evaluation);
        ++evaluation->ntext_escape;
      }
      else if (HIER_ESCAPE_NATIVE == id) {
        const double byte_bits = hier_score_bytes(model, payload, info.length);
        bits += byte_bits;
        evaluation->byte_bits += byte_bits;
        ++evaluation->nnative_escape;
      }
      deep = (history_length >= model->maxorder
        && NULL != libxs_ngram_lookup(&model->word_model, history,
          history_length, model->maxorder)) ? 1 : 0;
      if (0 < libxs_ngram_predict(&model->word_model, history,
        history_length, top_ids, 1, NULL) && top_ids[0] == id)
      {
        ++evaluation->ntop1;
      }
      ++evaluation->ntokens;
      evaluation->bits += bits;
      evaluation->bytes += (double)info.length;
      if (0 != deep) {
        ++evaluation->ndeep;
        evaluation->deep_bits += bits;
        evaluation->deep_bytes += (double)info.length;
      }
      else {
        ++evaluation->nshallow;
        evaluation->shallow_bits += bits;
        evaluation->shallow_bytes += (double)info.length;
      }
      hier_history_push(history, &history_length, LIBXS_NGRAM_ORDER_MAX, id);
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static void hier_score_clock_text(converse_hier_t* model,
  const char* text, int text_length, double mix,
  hier_clock_eval_t* evaluation)
{
  unsigned int word_state[COMPOSE_MAXTEXT];
  unsigned int syllable_state[COMPOSE_MAXTEXT];
  unsigned int raw_history[LIBXS_NGRAM_ORDER_MAX];
  int raw_length = 1;
  int pos;
  double recurrent[HIER_RECURRENT_DIM];
  memset(recurrent, 0, sizeof(recurrent));
  raw_history[0] = HIER_CLOCK_BYTE_START;
  if (text_length > 0 && text_length <= COMPOSE_MAXTEXT
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->word_tokenizer, model->symbols, HIER_CLOCK_WORD_BASE,
      word_state)
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->syllable_tokenizer, model->syllables,
      HIER_CLOCK_SYLLABLE_BASE, syllable_state))
  {
    for (pos = 0; pos < text_length; ++pos) {
      unsigned int history[LIBXS_NGRAM_ORDER_MAX];
      unsigned int raw_ids[1], context_ids[1];
      const unsigned int id = (unsigned int)(unsigned char)text[pos] + 1u;
      const int history_length = hier_clock_history(raw_history, raw_length,
        word_state[pos], syllable_state[pos], history);
      unsigned int recurrent_history[HIER_RECURRENT_ORDER];
      const int recurrent_length = hier_recurrent_history(recurrent,
        raw_history, raw_length, recurrent_history);
      const double raw_probability = libxs_ngram_prob(
        &model->stream_byte_model, raw_history, raw_length, id);
      const double context_probability = libxs_ngram_prob(
        &model->clock_byte_model, history, history_length, id);
      const double probability = mix * context_probability
        + (1.0 - mix) * raw_probability;
      const double raw_ppm_probability = hier_ppm_prob(&model->stream_ppm,
        raw_history, raw_length, id);
      const double context_ppm_probability = hier_ppm_prob(&model->clock_ppm,
        history, history_length, id);
      const double ppm_probability = mix * context_ppm_probability
        + (1.0 - mix) * raw_ppm_probability;
      const double recurrent_probability = hier_ppm_prob(
        &model->recurrent_ppm, recurrent_history, recurrent_length, id);
      const double recurrent_mix_probability = mix * recurrent_probability
        + (1.0 - mix) * raw_ppm_probability;
      const double frozen_interp_probability = hier_ppm_interp_prob(
        &model->stream_ppm, raw_history, raw_length, id);
      const double adaptive_probability = hier_ppm_interp_prob(
        &model->adaptive_ppm, raw_history, raw_length, id);
      const double raw_bits = -log(raw_probability) / log(2.0);
      const double context_bits = -log(context_probability) / log(2.0);
      const double bits = -log(probability) / log(2.0);
      const int deep = (history_length >= model->clock_order
        && NULL != libxs_ngram_lookup(&model->clock_byte_model, history,
          history_length, model->clock_order)) ? 1 : 0;
      if (0 < libxs_ngram_predict(&model->stream_byte_model, raw_history,
        raw_length, raw_ids, 1, NULL) && raw_ids[0] == id)
      {
        ++evaluation->nraw_top1;
      }
      if (0 < libxs_ngram_predict(&model->clock_byte_model, history,
        history_length, context_ids, 1, NULL) && context_ids[0] == id)
      {
        ++evaluation->ncontext_top1;
      }
      if (0 == (evaluation->nbytes % model->top_stride)) {
        const int ppm_rank = hier_ppm_rank(&model->stream_ppm, raw_history,
          raw_length, id, 0);
        const int adaptive_rank = hier_ppm_rank(&model->adaptive_ppm,
          raw_history, raw_length, id, 1);
        ++evaluation->nppm;
        if (1 == ppm_rank) ++evaluation->nppm_top1;
        if (ppm_rank >= 1 && ppm_rank <= 3) ++evaluation->nppm_top3;
        if (1 == adaptive_rank) ++evaluation->nadaptive_top1;
        if (adaptive_rank >= 1 && adaptive_rank <= 3) {
          ++evaluation->nadaptive_top3;
        }
      }
      ++evaluation->nbytes;
      evaluation->raw_bits += raw_bits;
      evaluation->context_bits += context_bits;
      evaluation->mix_bits += bits;
      evaluation->raw_ppm_bits += -log(raw_ppm_probability) / log(2.0);
      evaluation->context_ppm_bits += -log(context_ppm_probability) / log(2.0);
      evaluation->ppm_mix_bits += -log(ppm_probability) / log(2.0);
      evaluation->recurrent_bits += -log(recurrent_probability) / log(2.0);
      evaluation->recurrent_mix_bits += -log(recurrent_mix_probability)
        / log(2.0);
      evaluation->frozen_interp_bits += -log(frozen_interp_probability)
        / log(2.0);
      evaluation->adaptive_bits += -log(adaptive_probability) / log(2.0);
      if (0 != deep) {
        ++evaluation->ndeep;
        evaluation->deep_bits += bits;
      }
      else {
        ++evaluation->nshallow;
        evaluation->shallow_bits += bits;
      }
      hier_ppm_observe(&model->adaptive_ppm, raw_history, raw_length, id);
      hier_history_push(raw_history, &raw_length, LIBXS_NGRAM_ORDER_MAX, id);
      hier_recurrent_update(recurrent, (unsigned int)(unsigned char)text[pos],
        model->recurrent_decay);
    }
  }
}


converse_hier_t* converse_hier_build(const libxs_registry_t* corpus,
  int holdout, long corpus_size, int maxorder)
{
  converse_hier_t* result = NULL;
  converse_hier_t* model = (converse_hier_t*)calloc(1, sizeof(*model));
  LIBXS_UNUSED(corpus_key_from_fprint);
  if (NULL != model) {
    const char* env = getenv("CONVERSE_HIER_MINCOUNT");
    const char* clock_env = getenv("CONVERSE_HIER_CLOCK_ORDER");
    const char* decay_env = getenv("CONVERSE_HIER_STATE_DECAY");
    const char* stride_env = getenv("CONVERSE_HIER_TOP_STRIDE");
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value;
    model->mincount = 2;
    if (NULL != env && '\0' != *env) {
      const int parsed = atoi(env);
      if (parsed > 0) model->mincount = parsed;
    }
    model->maxorder = maxorder;
    model->clock_order = 2;
    model->recurrent_decay = 0.875;
    model->top_stride = 40;
    if (NULL != clock_env && '\0' != *clock_env) {
      const int parsed = atoi(clock_env);
      if (parsed >= 1 && parsed <= LIBXS_NGRAM_ORDER_MAX) {
        model->clock_order = parsed;
      }
    }
    if (NULL != decay_env && '\0' != *decay_env) {
      const double parsed = atof(decay_env);
      if (parsed >= 0.0 && parsed < 1.0) model->recurrent_decay = parsed;
    }
    if (NULL != stride_env && '\0' != *stride_env) {
      const int parsed = atoi(stride_env);
      if (parsed > 0) model->top_stride = parsed;
    }
    model->symbols = libxs_registry_create();
    model->syllables = libxs_registry_create();
    model->word_tokenizer = libxs_tokenizer_create(
      LIBXS_TOKEN_GRANULARITY_WORD);
    model->syllable_tokenizer = libxs_tokenizer_create(
      LIBXS_TOKEN_GRANULARITY_SYLLABLE);
    if (NULL != corpus && NULL != model->symbols && NULL != model->syllables
      && NULL != model->word_tokenizer && NULL != model->syllable_tokenizer)
    {
      value = libxs_registry_begin(corpus, &key, &cursor);
      while (NULL != value) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        if (0 == hier_is_test(index, holdout, corpus_size)) {
          hier_count_text(model, entry->text, entry->text_len);
        }
        ++index;
        value = libxs_registry_next(corpus, &key, &cursor);
      }
      model->word_vocab = hier_symbol_assign(model->symbols,
        model->mincount, HIER_SYMBOL_FIRST);
      model->syllable_vocab = hier_symbol_assign(model->syllables,
        model->mincount, HIER_SYLLABLE_FIRST);
      if (EXIT_SUCCESS == libxs_ngram_create(&model->word_model, maxorder)
        && EXIT_SUCCESS == libxs_ngram_create(&model->syllable_model, maxorder)
        && EXIT_SUCCESS == libxs_ngram_create(&model->byte_model, maxorder)
        && EXIT_SUCCESS == libxs_ngram_create(&model->stream_byte_model,
          model->clock_order)
        && EXIT_SUCCESS == libxs_ngram_create(&model->clock_byte_model,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->stream_ppm,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->clock_ppm,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->recurrent_ppm,
          HIER_RECURRENT_ORDER)
        && EXIT_SUCCESS == hier_ppm_create(&model->adaptive_ppm,
          model->clock_order))
      {
        key = NULL;
        cursor = 0;
        index = 0;
        value = libxs_registry_begin(corpus, &key, &cursor);
        while (NULL != value) {
          const corpus_entry_t* entry = (const corpus_entry_t*)value;
          if (0 == hier_is_test(index, holdout, corpus_size)) {
            hier_train_text(model, entry->text, entry->text_len);
            hier_train_clock_text(model, entry->text, entry->text_len);
          }
          ++index;
          value = libxs_registry_next(corpus, &key, &cursor);
        }
        libxs_ngram_finalize(&model->word_model, model->word_vocab);
        libxs_ngram_finalize(&model->syllable_model, model->syllable_vocab);
        libxs_ngram_finalize(&model->byte_model, HIER_BYTE_VOCAB);
        libxs_ngram_finalize(&model->stream_byte_model, 256);
        libxs_ngram_finalize(&model->clock_byte_model, 256);
        hier_ppm_finalize(&model->stream_ppm);
        hier_ppm_finalize(&model->clock_ppm);
        hier_ppm_finalize(&model->recurrent_ppm);
        if (EXIT_SUCCESS == hier_ppm_check(&model->stream_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->clock_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->recurrent_ppm))
        {
          model->ready = 1;
          fprintf(stderr, "hierarchy: word-vocab=%u syllable-vocab=%u"
            " mincount=%d order=%d clock-order=%d state-decay=%.3f\n",
            model->word_vocab,
            model->syllable_vocab, model->mincount, model->maxorder,
            model->clock_order, model->recurrent_decay);
          result = model;
        }
      }
    }
  }
  if (NULL == result) converse_hier_destroy(model);
  return result;
}


void converse_hier_destroy(converse_hier_t* model)
{
  if (NULL != model) {
    libxs_ngram_destroy(&model->word_model);
    libxs_ngram_destroy(&model->syllable_model);
    libxs_ngram_destroy(&model->byte_model);
    libxs_ngram_destroy(&model->stream_byte_model);
    libxs_ngram_destroy(&model->clock_byte_model);
    hier_ppm_destroy(&model->stream_ppm);
    hier_ppm_destroy(&model->clock_ppm);
    hier_ppm_destroy(&model->recurrent_ppm);
    hier_ppm_destroy(&model->adaptive_ppm);
    libxs_registry_destroy(model->symbols);
    libxs_registry_destroy(model->syllables);
    libxs_tokenizer_destroy(model->word_tokenizer);
    libxs_tokenizer_destroy(model->syllable_tokenizer);
    free(model);
  }
}


int converse_hier_eval(converse_hier_t* model,
  const libxs_registry_t* corpus, int holdout, long corpus_size,
  const char* label)
{
  int result = EXIT_FAILURE;
  if (NULL != model && 0 != model->ready && NULL != corpus) {
    hier_eval_t evaluation;
    hier_clock_eval_t clock_evaluation;
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value;
    memset(&evaluation, 0, sizeof(evaluation));
    memset(&clock_evaluation, 0, sizeof(clock_evaluation));
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      if (0 == holdout || 0 != hier_is_test(index, holdout, corpus_size)) {
        hier_score_text(model, entry->text, entry->text_len, &evaluation);
        hier_score_clock_text(model, entry->text, entry->text_len, 0.5,
          &clock_evaluation);
      }
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
    if (evaluation.bytes > 0.0 && evaluation.ntokens > 0) {
      fprintf(stdout, "predict-hier[%s%s]: top1=%.1f%% n=%ld bpc=%.3f\n",
        (NULL != label) ? label : "hier", (holdout > 0) ? ":heldout" : "",
        100.0 * (double)evaluation.ntop1 / (double)evaluation.ntokens,
        evaluation.ntokens, evaluation.bits / evaluation.bytes);
      fprintf(stderr, "  hierarchy bits: top=%.3f syllable=%.3f byte=%.3f"
        " | escapes: text=%.1f%% native=%.1f%% syllable=%ld\n",
        evaluation.top_bits / evaluation.bytes,
        evaluation.syllable_bits / evaluation.bytes,
        evaluation.byte_bits / evaluation.bytes,
        100.0 * (double)evaluation.ntext_escape / (double)evaluation.ntokens,
        100.0 * (double)evaluation.nnative_escape / (double)evaluation.ntokens,
        evaluation.nsyllable_escape);
      fprintf(stderr, "  attested-context split: verbatim %.1f%% of positions"
        " (bpc=%.3f) | novel %.1f%% (bpc=%.3f)\n",
        100.0 * (double)evaluation.ndeep / (double)evaluation.ntokens,
        (evaluation.deep_bytes > 0.0)
          ? evaluation.deep_bits / evaluation.deep_bytes : 0.0,
        100.0 * (double)evaluation.nshallow / (double)evaluation.ntokens,
        (evaluation.shallow_bytes > 0.0)
          ? evaluation.shallow_bits / evaluation.shallow_bytes : 0.0);
      if (clock_evaluation.nbytes > 0) {
        fprintf(stdout, "predict-clock[%s]: raw-top1=%.1f%%"
          " context-top1=%.1f%% n=%ld raw-bpc=%.3f context-bpc=%.3f"
          " mix-bpc=%.3f\n", (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nraw_top1
            / (double)clock_evaluation.nbytes,
          100.0 * (double)clock_evaluation.ncontext_top1
            / (double)clock_evaluation.nbytes,
          clock_evaluation.nbytes,
          clock_evaluation.raw_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.context_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.mix_bits / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-ppm[%s]: top1=%.1f%% top3=%.1f%% n=%ld"
          " (stride=%d) raw-bpc=%.3f context-bpc=%.3f mix-bpc=%.3f\n",
          (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nppm_top1
            / (double)clock_evaluation.nppm,
          100.0 * (double)clock_evaluation.nppm_top3
            / (double)clock_evaluation.nppm,
          clock_evaluation.nppm, model->top_stride,
          clock_evaluation.raw_ppm_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.context_ppm_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.ppm_mix_bits / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-recurrent[%s]: n=%ld state-bpc=%.3f"
          " mix-bpc=%.3f\n", (NULL != label) ? label : "metatoken",
          clock_evaluation.nbytes,
          clock_evaluation.recurrent_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.recurrent_mix_bits
            / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-adaptive[%s]: top1=%.1f%% top3=%.1f%% n=%ld"
          " (stride=%d) frozen-interp-bpc=%.3f adaptive-bpc=%.3f\n",
          (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nadaptive_top1
            / (double)clock_evaluation.nppm,
          100.0 * (double)clock_evaluation.nadaptive_top3
            / (double)clock_evaluation.nppm,
          clock_evaluation.nppm, model->top_stride,
          clock_evaluation.frozen_interp_bits
            / (double)clock_evaluation.nbytes,
          clock_evaluation.adaptive_bits / (double)clock_evaluation.nbytes);
        fprintf(stderr, "  clock attested split: verbatim %.1f%%"
          " (bpc=%.3f) | novel %.1f%% (bpc=%.3f)\n",
          100.0 * (double)clock_evaluation.ndeep
            / (double)clock_evaluation.nbytes,
          (clock_evaluation.ndeep > 0)
            ? clock_evaluation.deep_bits / (double)clock_evaluation.ndeep
            : 0.0,
          100.0 * (double)clock_evaluation.nshallow
            / (double)clock_evaluation.nbytes,
          (clock_evaluation.nshallow > 0)
            ? clock_evaluation.shallow_bits / (double)clock_evaluation.nshallow
            : 0.0);
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}