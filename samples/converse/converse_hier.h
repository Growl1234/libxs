#ifndef CONVERSE_HIER_H
#define CONVERSE_HIER_H

#include <libxs/libxs_reg.h>


typedef struct converse_hier_t converse_hier_t;


converse_hier_t* converse_hier_build(const libxs_registry_t* corpus,
  int holdout, long corpus_size, int maxorder);

void converse_hier_destroy(converse_hier_t* model);

int converse_hier_eval(converse_hier_t* model,
  const libxs_registry_t* corpus, int holdout, long corpus_size,
  const char* label);

/**
 * Bits the query saves on each candidate under the same expert mixture the BPC
 * evaluation measures: bits[i] is log2 of P(candidates[i] | query) /
 * P(candidates[i]), so a LARGER value means the query was more informative about
 * that candidate. Ranking by the conditional length alone would instead rank by
 * how ordinary each candidate is. Returns EXIT_SUCCESS only when every candidate
 * scored, since the values are signed and comparable only against each other.
 * The model is not updated, so repeated calls are order-independent.
 */
int converse_hier_rescore(const converse_hier_t* model,
  const char* query, int query_length, const char* const candidates[],
  const int candidate_lengths[], int ncandidates, double bits[]);

/**
 * Index of the candidate continuation with the smallest -log2 P(candidate |
 * context), or -1 if none could be scored. Unlike converse_hier_rescore this
 * ranks by conditional length: the candidates share one context, so the model's
 * own preference is exactly the quantity wanted.
 */
int converse_hier_choose(const converse_hier_t* model,
  const char* context, int context_length, const char* const candidates[],
  const int candidate_lengths[], int ncandidates);

#endif /*CONVERSE_HIER_H*/