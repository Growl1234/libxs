#ifndef CONVERSE_RECOMB_H
#define CONVERSE_RECOMB_H

#include <libxs/libxs_reg.h>
#include <libxs/libxs_token.h>

#include "converse.h"
#include "converse_hier.h"


/**
 * Word-scale probability of a successor given a context, as the recombination
 * syntax gate needs it. The gate compares the seam-crossing span against the
 * span it displaced, so it needs a backoff model but not any particular one: it
 * is passed as a function rather than linked, which keeps the caller's model
 * bridge (and its skip-gram interpolation) on the caller's side of this header.
 */
typedef double (*converse_recomb_prob_t)(const unsigned int hist[], int hlen,
  unsigned int next);


/**
 * Text predicates and the corpus encoder the recombination gates rely on. These
 * are owned by converse.c, which is the corpus and text layer; they are declared
 * here rather than duplicated so both translation units share one definition.
 */
typedef struct converse_recomb_host_t {
  int (*ends_sentence)(const char* text, int text_len);
  /**
   * Word-character test over a byte SEQUENCE, not a single byte: an encoded letter
   * spans several bytes and must be accepted whole, while encoded punctuation must
   * not be accepted at all. `length` receives the span so a scan advances by
   * characters rather than bytes.
   */
  int (*is_wordchar)(const unsigned char* text, size_t size, int* length);
  int (*entry_build)(corpus_entry_t* entry, const unsigned char* text, int len,
    unsigned char scale, libxs_lexicon_t* lexicon,
    const libxs_lexrule_t* rules, int nrules);
  converse_recomb_prob_t word_prob;
  int maxorder;
} converse_recomb_host_t;


/**
 * Run the recombination probe over the first `limit` sentence-scale entries.
 *
 * `judge` is the byte model that scores a seam, and it is required rather than
 * optional: fluency at the junction is the measurement the whole probe is built
 * around, so a NULL judge is a caller error rather than a degraded mode.
 */
void converse_recomb_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int limit, const converse_hier_t* judge,
  const converse_recomb_host_t* host);

#endif /*CONVERSE_RECOMB_H*/
