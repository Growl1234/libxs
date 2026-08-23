# N-gram

Header: `libxs_ngram.h`

Count-based variable-order n-gram model over unsigned token ids, backed by a
registry keyed on the context. Stores every context length up to a configured
maximum from one pass of observations, then answers interpolated-backoff
probabilities and top-k predictions.

Token id 0 is reserved and treated as absent context: it is never stored and
never predicted.

## Constants

```C
#define LIBXS_NGRAM_ORDER_MAX 6  /* longest context that can be stored */
#define LIBXS_NGRAM_SUCC_MAX  8  /* successors retained per context */
```

## Types

```C
typedef struct libxs_ngram_succ_t {
  unsigned int id, count;
} libxs_ngram_succ_t;

typedef struct libxs_ngram_entry_t {
  unsigned int total;   /* observations of this context */
  unsigned int nsucc;   /* successors retained */
  libxs_ngram_succ_t succ[LIBXS_NGRAM_SUCC_MAX];
} libxs_ngram_entry_t;
```

The successor list is bounded by a Space-Saving policy that evicts the least
frequent entry, so `total` may exceed the sum of the retained counts. The
difference is real uncertainty, and the probability model charges it to backoff
rather than hiding it.

```C
typedef struct libxs_ngram_t {
  libxs_registry_t* store;
  unsigned int* unifreq;
  double unifreq_total;
  unsigned int unifreq_size;
  unsigned int backoff_ids[LIBXS_NGRAM_SUCC_MAX];
  int backoff_count, maxorder;
} libxs_ngram_t;
```

Callers own the struct, so stack allocation is fine; the store and backoff
tables are heap-managed by create and destroy.

```C
typedef struct libxs_ngram_stats_t {
  long keys[LIBXS_NGRAM_ORDER_MAX + 1];       /* per order, index 0 unused */
  double obs[LIBXS_NGRAM_ORDER_MAX + 1];
  long saturated[LIBXS_NGRAM_ORDER_MAX + 1];
  size_t entries, nbytes;
} libxs_ngram_stats_t;
```

## Functions

```C
int libxs_ngram_create(libxs_ngram_t* model, int maxorder);
void libxs_ngram_destroy(libxs_ngram_t* model);
```

`maxorder` is clamped to `1..LIBXS_NGRAM_ORDER_MAX`. `create` returns
`EXIT_SUCCESS`, or `EXIT_FAILURE` if the backing store cannot be created.

```C
void libxs_ngram_observe(libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int succ);
void libxs_ngram_finalize(libxs_ngram_t* model, unsigned int vocab);
```

`observe` increments every context length `1..min(maxorder,hlen)` ending at
`hist[hlen-1]`, so one call per position populates all orders. `finalize` builds
the global unigram distribution and the most-frequent-token backoff list from the
order-1 counts; call it once after all observations and before any query, with
`vocab` the largest token id in use.

```C
const libxs_ngram_entry_t* libxs_ngram_lookup(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, int n);
double libxs_ngram_prob(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int next);
int libxs_ngram_predict(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int out_ids[], int k,
  int* order);
int libxs_ngram_stats(const libxs_ngram_t* model,
  libxs_ngram_stats_t* stats);
```

`prob` is interpolated backoff: it blends orders `1..min(maxorder,hlen)` with
weight `total/(total+1)` per order over an additive-smoothed unigram prior. After
`finalize`, probabilities over ids `1..vocab` sum to one for every history —
which makes the model usable as a code-length model and not only as a ranker.

`predict` uses hard backoff instead: the top-k successors of the longest context
that has a record, falling back to the global most-frequent list when none
matches. A non-NULL `order` receives the context length that produced the result
(0 for the global fallback), so a caller can tell a deeply grounded prediction
from a guess.

`stats` reports per-order key and observation counts plus the live footprint.
Report `entries` as the honest size: `nbytes` follows the hash table's
power-of-two capacity and therefore steps rather than growing smoothly.

## Example

```C
libxs_ngram_t model;
unsigned int hist[LIBXS_NGRAM_ORDER_MAX];
int hlen = 0;
if (EXIT_SUCCESS == libxs_ngram_create(&model, 3)) {
  unsigned int id;
  while (next_token(&id)) {
    libxs_ngram_observe(&model, hist, hlen, id);
    if (LIBXS_NGRAM_ORDER_MAX > hlen) hist[hlen++] = id;
    else {
      memmove(hist, hist + 1, (LIBXS_NGRAM_ORDER_MAX - 1) * sizeof(*hist));
      hist[LIBXS_NGRAM_ORDER_MAX - 1] = id;
    }
  }
  libxs_ngram_finalize(&model, vocab);
  /* ... libxs_ngram_prob / libxs_ngram_predict ... */
  libxs_ngram_destroy(&model);
}
```
