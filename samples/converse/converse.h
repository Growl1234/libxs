#ifndef CONVERSE_H
#define CONVERSE_H

#include <libxs/libxs_predict.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_reg.h>

#include <string.h>

#define FPRINT_ORDER 4
#define CORPUS_FILE "converse.dat"
#define COMPOSE_NDIMS 10
#define COMPOSE_BITS 6
#define COMPOSE_MAXTEXT 512
#define ENTRY_TOKEN_MAX 48
#define ENTRY_SECTION_MAX 64

#define ENTRY_LEX_ENTITY 0x0001u
#define ENTRY_LEX_NUMBER 0x0002u
#define ENTRY_LEX_QUESTION 0x0004u
#define ENTRY_LEX_PLACE 0x0008u
#define ENTRY_LEX_CAUSE 0x0010u
#define ENTRY_LEX_METHOD 0x0020u
/**
 * The entry is a clause FRAGMENT cut from a larger sentence, not a sentence.
 * Ingest stores both, so a byte of source text belongs to several entries at
 * the same scale; anything that must count each source byte once (above all the
 * BPC denominator) has to exclude these.
 */
#define ENTRY_LEX_FRAGMENT 0x0040u

/**
 * Which half of the system a binary exposes.
 *
 * The two halves answer different questions and are being separated into
 * different papers and different translation units: QA is grounded answering,
 * attribution and grounded recombination, LM is next-token prediction and its
 * quantification. The role gates the MODES an entry point accepts, so each
 * binary has a coherent command surface and nothing links a mode it cannot serve.
 * CONVERSE_ROLE_ALL keeps the historical single-binary behaviour so every
 * documented reproduction command still runs.
 */
enum {
  CONVERSE_ROLE_ALL = 0,
  CONVERSE_ROLE_QA = 1,
  CONVERSE_ROLE_LM = 2
};

#define GEN_CAND_MAX 8
#define EVAL_LINE_MAX 2048
#define NGRAM_ORDER_MAX LIBXS_NGRAM_ORDER_MAX
#if !defined(TOKEN_EMB_DIM)
# define TOKEN_EMB_DIM 16
#endif
#define TOKEN_CTX_MAX 8
#define ANSWER_PREDICT_INPUTS 10

enum { CONN_SPACE = 0, CONN_COMMA = 1, CONN_PERIOD = 2, CONN_NEWLINE = 3 };
enum { SCALE_PHRASE = 0, SCALE_SENTENCE = 1, SCALE_PARAGRAPH = 2 };

/**
 * text is LAST so an entry can be stored at its actual length
 * (corpus_entry_size) instead of the full COMPOSE_MAXTEXT. The corpus dominates
 * memory -- 1512 B per entry for a mean 34 B of enwik8 sentence text, which is
 * what made 90 MB exhaust RAM -- and every field before text keeps a fixed
 * offset, so readers are unaffected. The registry stores variable-size values
 * and readers already consult libxs_registry_value_size, which is why the
 * section helpers take an entry_size.
 */
/**
 * The stored projection of a fingerprint. libxs_fprint_t is 624 B because its
 * eight arrays are sized to LIBXS_FPRINT_MAXORDER (8) and three of them are
 * streaming accumulators used only while building. Converse needs exactly four
 * arrays at FPRINT_ORDER: l2 and mean for the Hilbert key, acc_sq/acc_sum/nk for
 * the similarity score. At 120k entries per 4 MB of text the difference is the
 * single largest term in corpus memory.
 */
typedef struct corpus_fprint_t {
  double l2[FPRINT_ORDER + 1];
  double mean[FPRINT_ORDER + 1];
  double acc_sq[FPRINT_ORDER + 1];
  double acc_sum[FPRINT_ORDER + 1];
  int nk[FPRINT_ORDER + 1];
  int order;
} corpus_fprint_t;


LIBXS_INLINE void corpus_fprint_pack(corpus_fprint_t* dst,
  const libxs_fprint_t* src)
{
  int k;
  for (k = 0; k <= FPRINT_ORDER; ++k) {
    const int use = (k <= src->order) ? 1 : 0;
    dst->l2[k] = (0 != use) ? src->l2[k] : 0.0;
    dst->mean[k] = (0 != use) ? src->mean[k] : 0.0;
    dst->acc_sq[k] = (0 != use) ? src->acc_sq[k] : 0.0;
    dst->acc_sum[k] = (0 != use) ? src->acc_sum[k] : 0.0;
    dst->nk[k] = (0 != use) ? src->nk[k] : 0;
  }
  dst->order = (src->order < FPRINT_ORDER) ? src->order : FPRINT_ORDER;
}


/** Widen the stored projection back to the library form. */
LIBXS_INLINE void corpus_fprint_unpack(libxs_fprint_t* dst,
  const corpus_fprint_t* src)
{
  int k;
  memset(dst, 0, sizeof(*dst));
  for (k = 0; k <= FPRINT_ORDER; ++k) {
    dst->l2[k] = src->l2[k];
    dst->mean[k] = src->mean[k];
    dst->acc_sq[k] = src->acc_sq[k];
    dst->acc_sum[k] = src->acc_sum[k];
    dst->nk[k] = src->nk[k];
  }
  dst->order = src->order;
}


typedef struct corpus_entry_t {
  corpus_fprint_t fprint;
  int text_len;
  unsigned char connector;
  unsigned char scale;
  unsigned short ntokens;
  unsigned short ncontent;
  unsigned short nentities;
  unsigned short nnumbers;
  unsigned short lexical_flags;
  unsigned short source;
  unsigned int token_ids[ENTRY_TOKEN_MAX];
  unsigned short token_flags[ENTRY_TOKEN_MAX];
  unsigned short section_len;
  char section[ENTRY_SECTION_MAX];
  char text[COMPOSE_MAXTEXT];
} corpus_entry_t;


/**
 * Bytes an entry occupies through its section field, i.e. every fixed-offset
 * field. A stored value at least this large carries complete metadata; the old
 * test was "entry_size >= sizeof(*entry)", which variable-length text makes
 * false for every entry.
 */
#define CORPUS_ENTRY_META_SIZE \
  (sizeof(corpus_entry_t) - COMPOSE_MAXTEXT)

/** Bytes actually occupied by an entry: everything up to its text length. */
LIBXS_INLINE size_t corpus_entry_size(const corpus_entry_t* entry)
{
  const size_t used = (0 < entry->text_len) ? (size_t)entry->text_len : 0;
  return sizeof(*entry) - COMPOSE_MAXTEXT + used + 1;
}


LIBXS_INLINE void corpus_key_from_fprint(const corpus_fprint_t* fp,
  unsigned char key[], size_t* key_size)
{
  unsigned int coords[COMPOSE_NDIMS];
  uint64_t hcode;
  int k;
  for (k = 0; k <= FPRINT_ORDER && k <= fp->order; ++k) {
    double v = fp->l2[k];
    double m = fp->mean[k];
    unsigned int qv, qm;
    if (v < 0) v = 0;
    if (v > 1.0) v = 1.0;
    if (m < -1.0) m = -1.0;
    if (m > 1.0) m = 1.0;
    qv = (unsigned int)(v * ((1 << COMPOSE_BITS) - 1));
    qm = (unsigned int)((m + 1.0) * 0.5 * ((1 << COMPOSE_BITS) - 1));
    coords[k] = qv;
    coords[FPRINT_ORDER + 1 + k] = qm;
  }
  for (k = fp->order + 1; k <= FPRINT_ORDER; ++k) {
    coords[k] = 0;
    coords[FPRINT_ORDER + 1 + k] = 0;
  }
  hcode = libxs_hilbert_bits(coords, COMPOSE_NDIMS, COMPOSE_BITS);
  memcpy(key, &hcode, 8);
  *key_size = 8;
}


enum { QUERY_GENERIC = 0, QUERY_WHO, QUERY_WHAT, QUERY_WHERE,
  QUERY_WHEN, QUERY_WHY, QUERY_HOW, QUERY_YESNO };

typedef struct answer_predict_profile_t {
  const char* name;
  int mode;
  int decompose;
  int clusters;
  int order;
  double quality;
  double smooth;
  int nseries;
  int window;
  int target;
  int diff_order;
} answer_predict_profile_t;

enum { RELATION_RULE_ALIAS = 1, RELATION_RULE_PERSON, RELATION_RULE_SKIP,
  RELATION_RULE_NEGATE, RELATION_RULE_NORM };

/**
 * Where a rule came from. ASSERTED means someone wrote it in the rule file.
 * The other two come from rule learning, above and below its acceptance bar.
 *
 * Both learned levels are labelled in replies, not just the margin. The margin
 * cannot be promoted by moving the threshold -- wrong terms score between right
 * ones -- and the ACCEPTED band is not trustworthy either: at the default bar
 * adjectives and interjections enter the person class, and one adjective is
 * enough to turn a correct reply into a confident false assertion. Labelling
 * only the margin would say the accepted band is safe, which it is not.
 */
enum { RELATION_RULE_ASSERTED = 0, RELATION_RULE_LEARNED,
  RELATION_RULE_PROPOSED };

typedef struct answer_relation_rule_t {
  int kind;
  int provenance;
  char relation[64];
  char term[64];
} answer_relation_rule_t;

typedef struct answer_relation_match_t {
  char answer[128];
  char relation[64];
  char actor[64];
  int answer_len;
  int relation_len;
  int actor_len;
  int plural;
  int made;
  double score;
} answer_relation_match_t;


/**
 * What one invocation has to work with once the shared state is loaded: the
 * corpus, the lexicon and the rules every model is built from, plus the modes
 * parsed from the command line. Passed to whichever half serves the invocation,
 * so the setup that both halves need exists once and neither half parses
 * arguments.
 */
typedef struct converse_run_t {
  libxs_registry_t* corpus;
  libxs_lexicon_t* lexicon;
  libxs_predict_t* answer_model;
  libxs_lexrule_t* rules;
  const answer_predict_profile_t* profile;
  const char* ngram_kind;
  const char* test_prefix;
  long nsentences;
  int nrules;
  int budget;
  int ngram_order;
  int ngram_holdout;
  int eval_mode;
  int predict_eval_mode;
  int complete_mode;
  int learn_mode;
  int role;
  /** A half must run: 0 after -L, which is complete once setup returns. */
  int pending;
} converse_run_t;


/**
 * The byte model as an INSTRUMENT the entry point installs, not a dependency the
 * halves name.
 *
 * Every judge in this system is a callback for the same reason converse_recomb's
 * word_prob and seam_bits are: a seam or candidate score has never separated a
 * true continuation from a fluent false one, so all of them are diagnostics.
 * Making them a hook is what lets the grounded half compile and link without the
 * byte model, while the binary that measures with it installs one. Every entry
 * point below degrades to "no instrument" rather than failing: rescore and choose
 * leave the caller's order untouched, seam_bits reports no bits.
 */
typedef struct converse_judge_t {
  void* (*open)(const libxs_registry_t* corpus, int maxorder);
  void (*close)(void* model);
  int (*rescore)(const void* model, const char* query, int query_length,
    const char* const candidates[], const int candidate_lengths[],
    int ncandidates, double bits[]);
  int (*choose)(const void* model, const char* context, int context_length,
    const char* const candidates[], const int candidate_lengths[],
    int ncandidates);
  int (*seam_bits)(const void* model, const char* prefix, int prefix_length,
    const char* suffix, int suffix_length, int score_length, double* bits);
} converse_judge_t;

void converse_judge_install(const converse_judge_t* judge);
/** Open the installed judge when CONVERSE_HIER_RESCORE asks for one. */
void converse_judge_open(const libxs_registry_t* corpus);
void converse_judge_close(void);
int converse_judge_active(void);
/** CONVERSE_HIER_RESCORE, so a caller can report what the judge changed. */
int converse_judge_verbose(void);
int converse_judge_rescore(const char* query, int query_length,
  const char* const candidates[], const int candidate_lengths[],
  int ncandidates, double bits[]);
int converse_judge_choose(const char* context, int context_length,
  const char* const candidates[], const int candidate_lengths[],
  int ncandidates);
/** Matches converse_recomb_seam_t, so it binds straight into the recomb host. */
int converse_judge_seam_bits(const char* prefix, int prefix_length,
  const char* suffix, int suffix_length, int score_length, double* bits);


/**
 * Parse the command line, reject a mode this role does not serve, then load or
 * ingest the corpus, the lexicon, the rules and the answer ranker: everything
 * both halves read and everything -L writes. Fills `run` on success.
 */
int converse_setup(int argc, char* argv[], int role, converse_run_t* run);

/** Release what converse_setup acquired, including the core model caches. */
void converse_release(converse_run_t* run);

/**
 * Which half an invocation needs, from its flags alone. The role gate and the
 * ALL binary's dispatch read this same answer, so "which binary serves this
 * command" is decided in one place.
 */
int converse_role_of(int argc, char* argv[]);

const libxs_lexnorm_t* converse_lexnorms(void);
int converse_lexnorms_size(void);
const answer_relation_rule_t* converse_rules(void);
size_t converse_rules_size(void);
libxs_ngram_t* converse_ngram_handle(void);
const char* converse_bridge_path(void);
const char* converse_eval_path(void);
const char* converse_predict_eval_path(void);

#endif /*CONVERSE_H*/