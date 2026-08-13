#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mix.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_hash.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_timer.h>

#include "converse.h"
#include "converse_hier.h"
#include "converse_recomb.h"

#include <unistd.h>

#define RESPONSE_BUDGET 1
#define CONVERSE_STAGE_MAX 12
#define ANSWER_MAX 4
#define GEN_CAND_MAX 8
#define ANSWER_MIN_SCORE 0.35
#define LEXICON_FILE "converse.lex"
#define PREDICT_FILE "converse.prd"
#define BRIDGE_FILE "converse.bridges"
#define BRIDGE_LINE_MAX 2048
#define RELATION_FILE "converse.relations"
/**
 * Language rules, shared by every corpus of that language and committed to the
 * repository (unlike the per-corpus .relations file). Kept language-neutral in
 * name so it can be a symlink to the actual language, e.g. english.rules.
 */
#define LANGUAGE_FILE "converse.rules"
/**
 * Normalization rules, kept apart from the interpretive rules because they are
 * a different kind: they rewrite what the tokenizer interns and therefore alter
 * the token stream every model is built from. Separating them keeps prediction
 * results independent of a change motivated by question answering, and keeps
 * the effect measurable by loading or omitting one file.
 */
#define NORM_FILE "converse.norms"
#define RELATION_LINE_MAX 1024
#define EVAL_FILE "converse.eval"
#define EVAL_LINE_MAX 2048
/** Fact-field marker: this expectation depends on loaded rules. */
#define EVAL_RULE_GOVERNED "-"
#define NGRAM_FILE "converse.ngram"
#define NGRAM_SUCC_MAX LIBXS_NGRAM_SUCC_MAX
#define NGRAM_TOPK 3
#define NGRAM_NATIVE_WIDTH 4
#define NGRAM_ORDER_MAX LIBXS_NGRAM_ORDER_MAX
/**
 * Expert-bank slots. Slots 1..NGRAM_ORDER_MAX are the fixed-order experts and
 * keep their order as their index, so the array stays order-indexed where that
 * is the natural key; slots beyond that are experts of a different KIND. An
 * extra slot has a FIXED index rather than one relative to the running
 * maxorder: a slot whose meaning shifted with the order would carry a weight
 * that was learned for a different expert.
 */
#define NGRAM_BANK_SKIP (NGRAM_ORDER_MAX + 1)
#define NGRAM_BANK_PRIOR (NGRAM_ORDER_MAX + 2)
#define NGRAM_BANK_PREDICT (NGRAM_ORDER_MAX + 3)
#define NGRAM_BANK_EMB (NGRAM_ORDER_MAX + 4)
#define NGRAM_BANK_LAST NGRAM_BANK_EMB
#define NGRAM_BANK_MAX (NGRAM_BANK_LAST + 1)
/** Ratio substituted for an expert that gave the target no mass at all. */
#define NGRAM_BANK_RELMIN 1e-4
/**
 * Escape-bank entropy above which the predict slot's estimate is still settling.
 * log2(13) == 3.700 is the uninformative prior; a figure taken while the bank is
 * near it is not trustworthy, so the run reports the mean rather than assuming
 * convergence.
 */
#define NGRAM_PREDICT_ENTROPY_MAX 3.5
#define BPE_SYMBOL_MAX 32
#define BPE_WORD_MAX 128
#define BPE_MERGES_DEFAULT 750
#define BPE_WORD_CAP 80000
#define PREDICT_EVAL_FILE "converse.predict"
#define CONVERSE_PATH_MAX 512
#define CONV_TOPIC_MAX 64
#if !defined(TOKEN_PREDICT_TRAIN_MAX)
# define TOKEN_PREDICT_TRAIN_MAX 40000
#endif
#define TOKEN_PREDICT_EVAL_STRIDE 40
#if !defined(TOKEN_EMB_DIM)
# define TOKEN_EMB_DIM 16
#endif
#if !defined(TOKEN_EMB_WINDOW)
# define TOKEN_EMB_WINDOW 4
#endif
#if !defined(TOKEN_EMB_ITER)
# define TOKEN_EMB_ITER 24
#endif
#define TOKEN_EMB_BACKFILL_MIN 3
#define TOKEN_EMB_BACKFILL_REF 5
#define KNNLM_K 24
#define KNNLM_VOTE_MAX 4
#define KNNLM_ANN_DIMS 8
#define KNNLM_ANN_BITS 8
#define KNNLM_ANN_WINDOW 512
#define TOKEN_CTX_MAX 8
#define RERANK_INPUTS 9
#define RERANK_RELIABILITY 32.0
#define RERANK_LIFT_MAX 8.0
#define ANSWER_PREDICT_INPUTS 10
#define CORPUS_BASENAME_PART_MAX 999


enum { PROFILE_PROSE = 0, PROFILE_MARKDOWN = 1 };

enum { GRAN_WORD = 0, GRAN_NATIVE = 1, GRAN_SYLLABLE = 2, GRAN_BPE = 3,
  GRAN_META_NATIVE = 4, GRAN_META_WORD = 5, GRAN_META_SYLLABLE = 6 };

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

typedef struct answer_bridge_t {
  const char* name;
  const char* query;
  const char* evidence;
  const char* reply;
  double score;
} answer_bridge_t;

enum { RELATION_RULE_ALIAS = 1, RELATION_RULE_PERSON, RELATION_RULE_SKIP,
  RELATION_RULE_NEGATE, RELATION_RULE_NORM };

typedef struct answer_relation_rule_t {
  int kind;
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

typedef struct answer_relation_fact_t {
  char answer[128];
  char relation[64];
  char actor[64];
  char section[ENTRY_SECTION_MAX];
  int answer_len;
  int relation_len;
  int actor_len;
  int section_len;
  int plural;
  int made;
  double score;
} answer_relation_fact_t;

typedef struct answer_identity_fact_t {
  char name[64];
  char role[64];
  char section[ENTRY_SECTION_MAX];
  int name_len;
  int role_len;
  int section_len;
  double score;
} answer_identity_fact_t;

typedef struct answer_describe_fact_t {
  char role[64];
  char text[192];
  char section[ENTRY_SECTION_MAX];
  int role_len;
  int text_len;
  int section_len;
  double score;
} answer_describe_fact_t;

typedef struct answer_docdef_fact_t {
  char title[ENTRY_SECTION_MAX];
  char header[64];
  char text[COMPOSE_MAXTEXT];
  int title_len;
  int header_len;
  int text_len;
} answer_docdef_fact_t;

typedef struct ngram_key_t {
  unsigned int a;
  unsigned int b;
} ngram_key_t;

typedef libxs_ngram_succ_t ngram_succ_t;
typedef libxs_ngram_entry_t ngram_entry_t;

/**
 * One expert-bank slot at one position. ACTIVE is not "probability zero":
 * an expert that cannot speak here (the skip tier without a seen pair, a
 * retrieval store without neighbours) must be left out of the pool and out of
 * the weight update entirely. Folding abstention in as zero drives the slot's
 * weight to exactly zero through the multiplicative update, and the uniform
 * recovery share only reaches slots that still hold mass -- so one abstention
 * would retire the expert permanently.
 */
typedef struct ngram_expert_t {
  double probability;
  int active;
} ngram_expert_t;

typedef struct token_emb_pair_t {
  unsigned int center;
  unsigned int context;
} token_emb_pair_t;

typedef struct bpe_symbol_t {
  int len;
  char bytes[BPE_SYMBOL_MAX];
} bpe_symbol_t;

typedef struct bpe_pair_t {
  int a;
  int b;
} bpe_pair_t;

typedef struct bpe_rank_t {
  int rank;
  int merged;
} bpe_rank_t;

typedef struct bpe_word_t {
  long count;
  int nsyms;
  int syms[BPE_WORD_MAX];
} bpe_word_t;


static const answer_predict_profile_t answer_predict_profiles[] = {
  { "raw", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "poly2", LIBXS_PREDICT_INTERPOLATE, LIBXS_PREDICT_RAW, 0, 2, 0.0, 0.0, 0, 0, 0, 0 },
  { "smooth", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 0, 0, 0, 0 },
  { "temporal", LIBXS_PREDICT_TEMPORAL, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 1, ANSWER_PREDICT_INPUTS, 0, 1 },
  { "rf", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_RF, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "fisher", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_FISHER, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "hknn", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_HKNN, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  /**
   * The two cells that complete the mode x decompose square. "raw" is
   * (AUTO,RAW) and "hknn" is (CLASSIFY,HKNN), so those two differ in BOTH
   * variables and a difference between them cannot be attributed. These fill in
   * (CLASSIFY,RAW) and (AUTO,HKNN) so the partition can be varied with the mode
   * held fixed, and the other way round.
   */
  { "classify", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_RAW, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "autohknn", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_HKNN, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  /**
   * k-means with the cluster count raised rather than auto-selected, kept as the
   * record of a REFUTED hypothesis: raising it does not buy the scan back for
   * free. Measured at 2 MB, 40000 entries -- auto (200 clusters) scans 3819 on
   * average for bpc 2.176; 1000 clusters scans 1594 for 2.184; the kd-tree
   * partition scans 155 for 2.180. So more clusters is dominated on BOTH axes,
   * and across all three the bits track the scan: a smaller cluster is less
   * local evidence, so the cost and the estimate are the same quantity. Only an
   * INDEX over an unchanged cluster can be free, which is why that is the
   * remaining option and a partition tweak is not.
   */
  { "raw1k", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 1000, 1, 0.0, 0.0, 0, 0, 0, 0 }
};

static char converse_path_corpus[CONVERSE_PATH_MAX] = CORPUS_FILE;
static char converse_path_lexicon[CONVERSE_PATH_MAX] = LEXICON_FILE;
static char converse_path_predict[CONVERSE_PATH_MAX] = PREDICT_FILE;
static char converse_path_bridge[CONVERSE_PATH_MAX] = BRIDGE_FILE;
static char converse_path_relation[CONVERSE_PATH_MAX] = RELATION_FILE;
static char converse_path_language[CONVERSE_PATH_MAX] = LANGUAGE_FILE;
static char converse_path_norm[CONVERSE_PATH_MAX] = NORM_FILE;
static char converse_path_eval[CONVERSE_PATH_MAX] = EVAL_FILE;
static char converse_path_predict_eval[CONVERSE_PATH_MAX] = PREDICT_EVAL_FILE;
static answer_bridge_t* answer_bridge_loaded = NULL;
static size_t answer_bridge_loaded_size = 0;
static answer_relation_rule_t* answer_relation_rules = NULL;
static size_t answer_relation_rules_size = 0;
static libxs_lexnorm_t* answer_lexnorms = NULL;
static int answer_lexnorms_size = 0;
/**
 * Source file an entry came from, assigned in ingest order (1-based; 0 means
 * unknown). A section alone cannot identify a source in the documentation
 * corpus: 14 pages carry a "Usage" heading and 12 carry "Example", so two
 * entries with equal section text may well be different modules. Coherence
 * measurements that compare sections therefore need this to tell a same-page
 * join from a cross-page one, and only the latter combines two sources.
 */
static unsigned short corpus_source_id = 0;
/**
 * Section of the most recent fact reply, so a citation can be printed without
 * threading an out-parameter through five resolver signatures. Cleared by
 * answer_fact_reply before dispatch and set by whichever resolver answered;
 * empty means the fact carried no section and no citation is printed.
 */
static char answer_fact_section[ENTRY_SECTION_MAX];
static int answer_fact_section_len = 0;
static libxs_lexicon_t* answer_negate_lexicon = NULL;
static const libxs_lexrule_t* answer_negate_rules = NULL;
static int answer_negate_nrules = 0;
static answer_relation_fact_t* answer_relation_facts = NULL;
static size_t answer_relation_facts_size = 0;
static answer_identity_fact_t* answer_identity_facts = NULL;
static size_t answer_identity_facts_size = 0;
static answer_describe_fact_t* answer_describe_facts = NULL;
static size_t answer_describe_facts_size = 0;
static answer_docdef_fact_t* answer_docdef_facts = NULL;
static size_t answer_docdef_facts_size = 0;
static char conv_topic[CONV_TOPIC_MAX] = "";
static int conv_topic_len = 0;
static long predict_ntotal = 0;


static void converse_namespace_init(const char* prefix);
static libxs_registry_t* corpus_load(void);
static int corpus_save(const libxs_registry_t* corpus);
static libxs_lexicon_t* converse_lexicon_load(void);
static int converse_lexicon_save(const libxs_lexicon_t* lexicon);
static libxs_predict_t* converse_predict_load(void);
static int converse_predict_save(const libxs_predict_t* model);
static libxs_predict_t* converse_predict_train(const libxs_registry_t* corpus,
  const answer_predict_profile_t* profile);
static const answer_predict_profile_t* answer_predict_profile_default(void);
static const answer_predict_profile_t* answer_predict_profile_find(
  const char* name);
static void answer_predict_profile_list(FILE* stream);
static void answer_bridge_free_loaded(void);
static size_t answer_bridge_load_file(const char* path);
static void answer_bridge_report(FILE* stream);
static void answer_relation_rules_free(void);
static void answer_lexnorms_build(void);
static int answer_query_is_negated(const char* query_text, size_t query_len);
static void answer_fact_section_set(const char* section, int section_len);
static void answer_fact_section_set(const char* section, int section_len);
static size_t answer_relation_rules_load_file(const char* path);
static void answer_relation_rules_report(FILE* stream);
static size_t answer_relation_rules_learn(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static void token_emb_pair_probe(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static double token_emb_succ_prob(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab, double temp);
static int token_emb_succ_rank(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab);
static int token_emb_directed(void);
static const double* token_emb_get(unsigned int id);
static int token_emb_isnull(unsigned int id);
static int token_emb_ready(void);
static const double* token_semb_get(unsigned int id);
static int answer_relation_rule_has_term(int kind, const char* term,
  int term_len);
static int answer_relation_rule_has_term(int kind, const char* text,
  int text_len);
static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len);
static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int corpus_ingest_basename(libxs_registry_t* corpus,
  const char* basename, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules);
static int corpus_profile_for_path(const char* path);
static int corpus_md_store(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like);
static int corpus_md_emit_block(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like);
static int corpus_ingest_markdown(libxs_registry_t* corpus,
  const unsigned char* text, size_t text_size, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int corpus_md_sentences(void);
static int count_words(const unsigned char* text, int length);
static size_t text_closer_size(const unsigned char* text, size_t size,
  size_t pos);
static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos);
static int text_starts_sentence(const char* text, int text_len);
static int text_ends_sentence(const char* text, int text_len);
static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id);
static int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id);
static int lexeme_text_is(const libxs_lexicon_t* lexicon,
  const libxs_lexeme_t* lexeme, const char* text);
static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text);
static int lexeme_stream_has_similar_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  int tolerance);
static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon);
static int query_type_prefers_sentence(int query_type);
static int corpus_entry_build(corpus_entry_t* entry,
  const unsigned char* text, int len, unsigned char scale,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static void corpus_entry_set_section(corpus_entry_t* entry,
  const char* section, int section_len);
static int corpus_title_prefix(const unsigned char* text, int len,
  char* title, int title_size);
static int corpus_entry_same_section(const corpus_entry_t* lhs,
  size_t lhs_size, const corpus_entry_t* rhs);
static int corpus_store_entry(libxs_registry_t* corpus,
  const corpus_entry_t* entry);
static int corpus_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* corpus);
static int answer_query_section(const char* query_text, size_t query_len,
  char* title, int title_size);
static int corpus_entry_section_match(const corpus_entry_t* entry,
  size_t entry_size, const char* title, int title_len);
static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size, int* upper_initial);
static int answer_query_relation_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size);
static int answer_relation_copy_antecedent(char* output, int output_size,
  const char* text, int text_len, int cue_pos);
static int answer_relation_find_person_before(const char* text, int text_len,
  int limit, int* term_pos, int* term_len);
static int answer_relation_copy_person_before(char* output, int output_size,
  const char* text, int text_len, int limit);
static int answer_relation_match_query(const char* query_text,
  size_t query_len, int query_type, const corpus_entry_t* entry,
  answer_relation_match_t* match);
static int answer_relation_reply(const answer_relation_match_t* match,
  char* output, size_t output_size);
static void answer_relation_facts_free(void);
static size_t answer_relation_facts_build(const libxs_registry_t* corpus);
static void answer_relation_facts_report(FILE* stream);
static int answer_relation_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_identity_facts_free(void);
static size_t answer_identity_facts_build(const libxs_registry_t* corpus);
static void answer_identity_facts_report(FILE* stream);
static void answer_describe_facts_free(void);
static size_t answer_describe_facts_build(const libxs_registry_t* corpus);
static void answer_describe_facts_report(FILE* stream);
static int answer_describe_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_docdef_facts_free(void);
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus);
static void answer_docdef_facts_report(FILE* stream);
static int answer_docdef_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void conv_reset(void);
static void conv_remember(const char* query_text, size_t query_len);
static int conv_rewrite(const char* query_text, size_t query_len,
  char* out, size_t out_size);
static int answer_identity_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static int answer_reply_role(char* output, size_t output_size,
  const char* name, int name_len, const char* role);
static int answer_relation_aggregate_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size);
static double answer_identity_score(const char* query_text, size_t query_len,
  int query_type, const corpus_entry_t* entry);
static int answer_features_fill(const corpus_entry_t* entry,
  size_t entry_size, double overlap, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS]);
static int answer_features(const libxs_lexeme_stream_t* query,
  const corpus_entry_t* entry, size_t entry_size, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS]);
static double answer_weak_label(const corpus_entry_t* entry, int query_type);
static libxs_predict_t* answer_predict_create(
  const answer_predict_profile_t* profile);
static int answer_predict_build_model(libxs_predict_t* model,
  const answer_predict_profile_t* profile);
static void answer_predict_report(const char* label,
  const libxs_predict_t* model, int ntrain,
  const answer_predict_profile_t* profile);
static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type);
static const answer_bridge_t* answer_bridge_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const corpus_entry_t* entry);
static double answer_semantic_bridge_score(const answer_bridge_t* bridge);
static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile);
static double answer_predict_score(const libxs_predict_t* model,
  const double inputs[ANSWER_PREDICT_INPUTS], double base_score);
static int answer_select(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX]);
static int answer_reply(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_hier_reorder(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX],
  int answer_count);
static converse_hier_t* answer_hier_build(const libxs_registry_t* corpus);
static void slot_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
static int answer_evidence_sentence(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size);
static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  char* out_reply, size_t out_size);
static int text_find_ci(const char* text, int text_len, const char* term);
static int text_find_word_ci(const char* text, int text_len,
  const char* term);
static int text_contains_ci(const char* text, int text_len, const char* term);
static int text_contains_word_ci(const char* text, int text_len,
  const char* term);
static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
static libxs_registry_t* ngram_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
static void ngram_backoff_build(libxs_registry_t* model,
  const libxs_lexicon_t* lexicon);
static const ngram_entry_t* ngram_lookup(libxs_registry_t* model,
  unsigned int ctx_a, unsigned int ctx_b);
static double ngram_unigram_prior(unsigned int id);
static int ngram_maxorder(void);
static int ngram_dedup_scale(void);
static int ngram_oracle_probe(void);
static unsigned int corpus_chain_max(void);
static int converse_predict_on(void);
static void converse_stage_begin(void);
static void converse_stage_end(const char* name);
static void converse_stage_report(void);
static int ngram_select_order(void);
static int ngram_bank_probe(void);
static double ngram_bank_rate(void);
static double ngram_bank_share(void);
static void ngram_bank_update(double weight[], const ngram_expert_t expert[],
  double mixture, double rate, double share);
static unsigned int ngram_bank_slots(void);
static int ngram_bank_enabled(unsigned int slots, int slot);
static const char* ngram_bank_slotname(int slot);
static void ngram_bank_experts(const unsigned int hist[], int hlen,
  int maxorder, unsigned int cur, unsigned int slots,
  const libxs_predict_t* store, void* context, int use_emb, int vocabulary,
  ngram_expert_t expert[]);
static int token_input_vector(unsigned int prev2, unsigned int prev1,
  int use_emb, double inputs[]);
static double ngram_bank_pool(const double weight[],
  const ngram_expert_t expert[]);
static int ngram_bank_geometric(void);
static int ngram_bank_frozen(void);
static int ngram_gen_embrank(void);
static int ngram_gen_embcand(void);
static double ngram_emb_temp(void);
static int ngram_emb_ctx(void);
static double ngram_emb_decay(void);
static double token_emb_succ_prob(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab, double temp);
static int ngram_bank_warmup(libxs_predict_t* store, int vocabulary);
static void ngram_syllable_probe(void);
static void ngram_hist_push(unsigned int hist[], int* hlen, int cap,
  unsigned int id);
static void ngram_syllable_oracle(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_registry_t* corpus, int holdout, int maxorder);
static int ngram_bank_support(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int ids[], int max);
static double ngram_bank_pool_geo(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int cur,
  const double weight[], const ngram_expert_t expert[]);
static int ngram_is_wordchar(unsigned char c);
static int recomb_compose_on(void);
static double recomb_word_prob(const unsigned int hist[], int hlen,
  unsigned int next);
static double ngramk_prob_exact(const unsigned int hist[], int hlen,
  unsigned int next);
static double ngram_skip_prob(const unsigned int hist[], int hlen,
  unsigned int succ_id);
static int ngram_skip_ready(const unsigned int hist[], int hlen);
static double ngram_skip_mu(void);
static void ngram_train_text(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len);
static libxs_ngram_t converse_ngram;
static libxs_ngram_t converse_skip;
static int converse_skip_on = 0;
static const converse_hier_t* answer_hier_model = NULL;
static long answer_hier_nreorder = 0;
static long answer_hier_nchanged = 0;
static int converse_profile_override = -1;
static int converse_order_max = 0;
static const char* converse_stage_name[CONVERSE_STAGE_MAX];
static double converse_stage_time[CONVERSE_STAGE_MAX];
static int converse_stage_count = 0;
static long corpus_chain_dropped = 0;
/**
 * Predict-slot diagnostics. The escape bank's entropy says whether its estimate
 * has settled; a figure taken near the uninformative prior is not trustworthy,
 * so the mean is reported rather than convergence being assumed.
 */
static double predict_entropy_sum = 0.0;
static long predict_nscored = 0;
static int predict_support_last = 0;
/**
 * Does the slot's kNN vote contribute anything, or is its value entirely the
 * learned escape over the frequency prior? The inputs are RAW TOKEN IDS, so
 * Euclidean distance between them is meaningless and the vote may be reading
 * noise. info->attested says whether the observed token received any local
 * evidence at all, which splits the slot's own bits into the part the vote
 * touched and the part it did not.
 */
static long predict_att_n = 0, predict_unatt_n = 0;
static double predict_att_bits = 0.0, predict_unatt_bits = 0.0;
static libxs_timer_tick_t converse_stage_tick = 0;
static bpe_symbol_t* bpe_symbols = NULL;
static int bpe_nsymbols = 0;
static int bpe_cap_symbols = 0;
static libxs_registry_t* bpe_merges = NULL;
static void ngram_complete(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static void complete_respond(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile, int budget,
  const char* text, int text_len);
static int ngram_generate(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  char* out, size_t out_size, double* order_mean, int* order_min_out);
static int ngram_eval(libxs_registry_t* model, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind, const libxs_predict_t* store,
  int use_emb);
static int ngram_gen_eval(libxs_registry_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int holdout, const char* kind,
  const libxs_predict_t* store, int use_emb);
static int ngram_gen_bank_rank(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int ids[], int n,
  unsigned int slots, const libxs_predict_t* store, void* context,
  int use_emb, int vocabulary, double weight[]);
static void ngram_stats(const libxs_registry_t* model);
static int ngram_gran_mode(void);
static int predict_is_test(long index, int holdout);
static int bpe_add_symbol(const char* bytes, int len);
static void bpe_free(void);
static void bpe_build(const libxs_registry_t* corpus, int holdout);
static int bpe_encode_run(const char* text, int len, libxs_lexeme_t tokens[],
  int max, int start, libxs_lexicon_t* lexicon, int create);
static void token_emb_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
static void token_emb_free(void);
static int knnlm_topk(libxs_registry_t* ngram, const libxs_predict_t* store,
  const unsigned int hist[], int hlen, int ctxlen, int order,
  unsigned int out_ids[], int k, unsigned int target, double* target_prob);
static int knnlm_ctxlen(void);
static void knnlm_cache_free(void);
static int knnlm_eval(libxs_registry_t* ngram, const libxs_predict_t* store,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, int holdout,
  const char* kind);
static void knnlm_complete(libxs_registry_t* ngram,
  const libxs_predict_t* store, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static libxs_predict_t* token_predict_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const answer_predict_profile_t* profile, int use_emb, int holdout,
  int ctxlen);
static int token_predict_eval(const libxs_predict_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int use_emb, int holdout,
  const char* kind);
static void token_complete(const libxs_predict_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int use_emb, const char* text, int text_len);
static libxs_predict_t* rerank_build(const libxs_registry_t* corpus,
  libxs_registry_t* ngram, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order,
  const answer_predict_profile_t* profile, int holdout);
static int rerank_eval(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind);
static void rerank_complete(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static double converse_score(const corpus_fprint_t* candidate,
  const libxs_fprint_t* query, const corpus_fprint_t* prev);
static int entry_used(const corpus_entry_t* e,
  const corpus_entry_t* const used[], int nused);
static const corpus_entry_t* select_best(void* const candidates[],
  int ncandidates, const corpus_entry_t* const used[], int nused,
  const libxs_fprint_t* query, const corpus_fprint_t* prev,
  int require_scale, unsigned char preferred_scale);
static uint64_t query_hilbert_code(const libxs_fprint_t* fp);
static int respond(const libxs_spatial_t* spatial,
  const libxs_registry_t* corpus, const char* query_text,
  size_t query_len, const libxs_fprint_t* query, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);


int main(int argc, char* argv[])
{
  libxs_registry_t* corpus;
  libxs_lexicon_t* lexicon;
  libxs_predict_t* answer_model;
  libxs_lexrule_t rules[96];
  int i, budget = RESPONSE_BUDGET, eval_mode = 0;
  int complete_mode = 0, predict_eval_mode = 0, learn_mode = 0;
  int warm_start = 0;
  int ngram_order = 2;
  int ngram_holdout = 0;
  const char* ngram_kind = "trigram";
  libxs_registry_t* ngram_model = NULL;
  const char** basenames;
  int nbasenames = 0;
  const char* test_prefix = NULL;
  int nrules;
  libxs_registry_info_t rinfo;
  char line[4096];
  const answer_predict_profile_t* predict_profile =
    answer_predict_profile_default();

  if (argc < 2) {
    fprintf(stderr,
      "Usage: %s [-e] [-n N] [-P PROFILE] [-b PREFIX] corpus1.txt [corpus2.txt ...]\n"
      "  Interactive conversational agent.\n"
      "  -e: run converse.eval evaluation and exit.\n"
      "  -E: run next-token prediction evaluation and exit.\n"
      "  -L: learn from the corpus, save state, and exit.\n"
      "  -c: read prompts and print next-token continuations.\n"
      "  -K KIND: next-token model "
      "(bigram|trigram|predict|embed|rerank|knnlm|hier).\n"
      "  -H N: held-out split; train on non-test, eval 1-in-N test.\n"
      "  -T PREFIX: fixed held-out corpus for -E (train all of -b, eval on -T).\n"
      "  -n N: response sentence budget (default %d).\n"
      "  -P PROFILE: answer predictor profile.\n"
      "  -p STRUCTURE: corpus structure (prose|markdown; default by ext).\n"
      "  -x: extreme mode; deepest n-gram context (affects -c and -E).\n"
      "  -b PREFIX: ingest matching files next to PREFIX.\n"
      "Environment variables (override defaults):\n"
      "  CONVERSE_NGRAM_ORDER=N   n-gram context order 1..%d (default 2; -x=%d).\n"
      "  CONVERSE_GRAN=UNIT       token unit: word|native|syllable|bpe|"
      "meta-native|meta-word|meta-syllable.\n"
      "  CONVERSE_GEN_MINORDER=N  generation grounding floor (default 2).\n"
      "  CONVERSE_GEN_CONTORDER=F -c continuation mean-order floor (default 3).\n"
      "  CONVERSE_BPE_MERGES=N    BPE merge budget (default 750).\n"
      "  CONVERSE_HOLDOUT_TAIL=1  held-out split is a contiguous tail.\n"
      "  CONVERSE_SHUFFLE=1       shuffle words in each sentence (control).\n"
      "  CONVERSE_EVAL_STRIDE=N   evaluate 1-in-N test items (default 40).\n"
      "  CONVERSE_GEN_EVAL=1      -E reports generation reproduction.\n"
      "  CONVERSE_GEN_NCAND=N     successors offered per step 1..%d (default 1).\n"
      "  CONVERSE_GEN_FULL=1      gen-eval scores every lookahead position\n"
      "                           (teacher-forced) instead of stopping at the\n"
      "                           first miss; adds per-position top1/mrr.\n"
      "  CONVERSE_GEN_EMBRANK=1   rank the truth by the successor-side embedding\n"
      "                           over the whole vocabulary (needs -K embed).\n"
      "  CONVERSE_EMB_DIRECTED=N  factorize PPMI(next|cur) at forward distance\n"
      "                           1..N instead of the symmetric window (0=off).\n"
      "  CONVERSE_NGRAM_BANK_EMB=1 carry the successor-side embedding as a bank\n"
      "                           slot: total, and scores unattested successors.\n"
      "  CONVERSE_EMB_TEMP=F      temperature of that slot (default 1 = plain\n"
      "                           exp(PMI) rescaling of the unigram prior).\n"
      "  CONVERSE_EMB_CTX=N       history tokens the slot conditions on 1..%d\n"
      "                           (default 1); summing them combines per-token\n"
      "                           PMI evidence multiplicatively.\n"
      "  CONVERSE_EMB_DECAY=F     weight decay per extra history token (0<F<=1).\n"
      "  CONVERSE_GEN_EMBCAND=N   let the embedding PROPOSE N successors per step\n"
      "                           (default 0). Off by default because a proposed\n"
      "                           successor was never attested here, so the path\n"
      "                           synthesizes instead of selecting.\n"
      "  CONVERSE_NGRAM_STATS=1   print per-order n-gram footprint.\n"
      "  CONVERSE_KNNLM_LAMBDA=F  fixed kNN-LM interpolation weight.\n"
      "  CONVERSE_KNNLM_DYN=1     dynamic (test-time) kNN-LM datastore.\n"
      "  CONVERSE_KNNLM_ANN=1     Hilbert-indexed kNN-LM retrieval (faster).\n"
      "  CONVERSE_KNNLM_CTX=N     kNN-LM context tokens 2..8 (default 2).\n"
      "  CONVERSE_KNNLM_DECAY=F   kNN-LM context decay 0..1 (default 0.5).\n"
      "  CONVERSE_KNNLM_WEIGHTS=L per-position context weights (overrides decay).\n"
      "  CONVERSE_KNNLM_ORDERPROBE=1 report context order sensitivity.\n"
      "  CONVERSE_KNNLM_CONTROL=N control vote: 1=unigram, 2=arbitrary nbrs.\n"
      "  CONVERSE_KNNLM_TEMP=F    softmax temperature (0=inverse distance).\n"
      "  CONVERSE_KNNLM_HEADS=N   retrieval heads over embedding subspaces.\n"
      "  CONVERSE_HIER_MINCOUNT=N hierarchy known-unit count (default 2).\n"
      "  CONVERSE_HIER_CLOCK_ORDER=N byte/state context order 1..6 (default 2).\n"
      "  CONVERSE_HIER_STATE_DECAY=F recurrent-state decay 0..<1.\n"
      "  CONVERSE_HIER_TOP_STRIDE=N PPM top-k evaluation stride (default 40).\n"
      "  CONVERSE_HIER_EXPERT_ORDER=N highest mixed PPM order 0..6.\n"
      "  CONVERSE_HIER_EXPERT_RATE=F expert update rate 0..1 (default 0.15).\n"
      "  CONVERSE_HIER_EXPERT_SHARE=F fixed share 0..<1 (default 0.005).\n"
      "  CONVERSE_SKIP=1          add skip-gram (w _ w) generalization tier.\n"
      "  CONVERSE_SKIP_MU=F       skip-gram interpolation weight (default 0.3).\n"
      "  CONVERSE_NGRAM_BANK_SKIP=1 carry the skip tier as its own bank slot\n"
      "                           (learned weight instead of fixed SKIP_MU).\n"
      "  CONVERSE_NGRAM_BANK_PRIOR=1 carry the unigram prior as a bank slot\n"
      "                           (a total expert where counts are partial).\n"
      "  CONVERSE_NGRAM_BANK_PREDICT=1 carry libxs_predict as a bank slot\n"
      "                           (distribution via prob_observe).\n"
      "  CONVERSE_NGRAM_BANK_FROZEN=1 warm up the predict slot on the training\n"
      "                           entries, commit, then score frozen: the\n"
      "                           figure stops depending on iteration order.\n"
      "  CONVERSE_NGRAM_BANK_WARMUP=N observe only the first N entries during\n"
      "                           warm-up (0=all); the bank settles long before\n"
      "                           the store is exhausted.\n"
      "  CONVERSE_GEN_RELEVANCE=F require F content-word overlap between an\n"
      "                           answer and its continuation (default 0=off).\n"
      "  CONVERSE_NGRAM_BANK_GEO=1 geometric (log-linear) pool instead of the\n"
      "                           linear one; exactly normalized.\n"
      "  CONVERSE_SYLLABLE_PROBE=w1,w2 print the syllable split of each word.\n"
      "  CONVERSE_SYLLABLE_ORACLE=1 bound any cut rule: cheapest split per word\n"
      "                           vs the heuristic's, over held-out words.\n"
      "  CONVERSE_EMB_PROBE=w1,w2 print nearest embedding neighbors.\n"
      "  CONVERSE_EMB_BACKFILL=0  disable rare-token vector backfill.\n"
      "  CONVERSE_RECOMB=N        recombination probe over N sentences.\n"
      "  CONVERSE_RECOMB_REFERENT=1 test whether a pivot denotes one thing.\n"
      "  CONVERSE_RECOMB_COMPOSE=1 -c may synthesize when replay fails.\n"
      "  CONVERSE_PREDICT=1       train the answer ranker (off by default:\n"
      "                           40%% of wall, no measured BPC or QA change).\n",
      argv[0], RESPONSE_BUDGET, NGRAM_ORDER_MAX, NGRAM_ORDER_MAX,
      GEN_CAND_MAX, TOKEN_CTX_MAX);
    answer_predict_profile_list(stderr);
    return EXIT_FAILURE;
  }

  basenames = (const char**)malloc((size_t)argc * sizeof(*basenames));
  if (NULL == basenames) return EXIT_FAILURE;

  i = 1;
  while (i < argc && '-' == argv[i][0] && '\0' != argv[i][1]) {
    if (0 == strcmp(argv[i], "-e")) {
      eval_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-E")) {
      predict_eval_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-L")) {
      learn_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-c")) {
      complete_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-x")) {
      converse_order_max = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-H") && i + 1 < argc) {
      ngram_holdout = atoi(argv[i + 1]);
      if (ngram_holdout < 0) ngram_holdout = 0;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-K") && i + 1 < argc) {
      if (0 == strcmp(argv[i + 1], "bigram")) {
        ngram_kind = "bigram";
        ngram_order = 1;
      }
      else if (0 == strcmp(argv[i + 1], "trigram")) {
        ngram_kind = "trigram";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "predict")) {
        ngram_kind = "predict";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "embed")) {
        ngram_kind = "embed";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "rerank")) {
        ngram_kind = "rerank";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "knnlm")) {
        ngram_kind = "knnlm";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "hier")) {
        ngram_kind = "hier";
        ngram_order = 2;
      }
      else {
        fprintf(stderr,
          "unknown prediction kind: %s "
          "(use bigram|trigram|predict|embed|rerank|knnlm|hier)\n",
          argv[i + 1]);
        free(basenames);
        return EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-n") && i + 1 < argc) {
      budget = atoi(argv[i + 1]);
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-P") && i + 1 < argc) {
      predict_profile = answer_predict_profile_find(argv[i + 1]);
      if (NULL == predict_profile) {
        fprintf(stderr, "unknown predictor profile: %s\n", argv[i + 1]);
        answer_predict_profile_list(stderr);
        return EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-p") && i + 1 < argc) {
      if (0 == strcmp(argv[i + 1], "prose")) {
        converse_profile_override = PROFILE_PROSE;
      }
      else if (0 == strcmp(argv[i + 1], "markdown")) {
        converse_profile_override = PROFILE_MARKDOWN;
      }
      else {
        fprintf(stderr, "unknown structure profile: %s (use prose|markdown)\n",
          argv[i + 1]);
        free(basenames);
        return EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-b") && i + 1 < argc) {
      basenames[nbasenames] = argv[i + 1];
      ++nbasenames;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-T") && i + 1 < argc) {
      test_prefix = argv[i + 1];
      i += 2;
    }
    else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      answer_predict_profile_list(stderr);
      free(basenames);
      return EXIT_FAILURE;
    }
  }

  if (i == argc && 0 == nbasenames) {
    fprintf(stderr, "no corpus source given\n");
    free(basenames);
    return EXIT_FAILURE;
  }

  converse_namespace_init((nbasenames > 0) ? basenames[0] : NULL);

  if (0 != eval_mode && nbasenames + argc - i < 1) {
    fprintf(stderr,
      "eval mode expects at least one corpus source and reads %s\n",
      converse_path_eval);
    free(basenames);
    return EXIT_FAILURE;
  }

  corpus = corpus_load();
  if (NULL == corpus) corpus = libxs_registry_create();
  lexicon = converse_lexicon_load();
  if (NULL == lexicon) lexicon = libxs_lexicon_create();
  answer_model = converse_predict_load();
  nrules = libxs_lexrule_defaults(rules, 96);
  answer_negate_lexicon = lexicon;
  answer_negate_rules = rules;
  answer_negate_nrules = nrules;
  if (NULL == corpus || NULL == lexicon || nrules <= 0) {
    libxs_registry_destroy(corpus);
    libxs_lexicon_destroy(lexicon);
    libxs_predict_destroy(answer_model);
    free(basenames);
    return EXIT_FAILURE;
  }

  answer_bridge_load_file(converse_path_bridge);
  answer_bridge_report(stderr);
  /**
   * Rules layer: the language file (committed, shared by every corpus of that
   * language) is loaded first, then the optional per-corpus file extends it.
   * Loading appends, so a corpus adds its own vocabulary without restating the
   * language's function words, and neither file contains anything the source
   * needs to know about.
   */
  answer_relation_rules_load_file(converse_path_language);
  answer_relation_rules_load_file(converse_path_norm);
  answer_relation_rules_load_file(converse_path_relation);
  answer_relation_rules_report(stderr);
  /* Needs no corpus and no model, so it runs before any of them is built. */
  ngram_syllable_probe();

  /**
   * A warm start reuses the persisted corpus, lexicon, and predictor instead
   * of re-ingesting and re-training: the state is complete when the corpus
   * loaded non-empty, the lexicon is populated, and the predictor loaded.
   * Learn mode (-L) always rebuilds. Only the cheap fact index is rebuilt each
   * run below.
   */
  { libxs_registry_info_t warm;
    warm.size = 0;
    warm_start = (0 == learn_mode && NULL != answer_model
      && libxs_lexicon_size(lexicon) > 0
      && EXIT_SUCCESS == libxs_registry_info(corpus, &warm)
      && warm.size > 0) ? 1 : 0;
  }

  if (0 == warm_start) {
    int basename_index;
    int have_positional = (i < argc) ? 1 : 0;
    converse_stage_begin();
    for (basename_index = 0; basename_index < nbasenames; ++basename_index) {
      if (EXIT_SUCCESS != corpus_ingest_basename(corpus,
        basenames[basename_index], lexicon, rules, nrules)
        && 0 == have_positional)
      {
        libxs_registry_destroy(corpus);
        libxs_lexicon_destroy(lexicon);
        libxs_predict_destroy(answer_model);
        answer_bridge_free_loaded();
        answer_relation_rules_free();
        answer_relation_facts_free();
        answer_identity_facts_free();
        answer_describe_facts_free();
        answer_docdef_facts_free();
        free(basenames);
        return EXIT_FAILURE;
      }
    }
    for (; i < argc; ++i) {
      corpus_ingest_file(corpus, argv[i], lexicon, rules, nrules);
    }
    converse_stage_end("ingest");
    corpus_save(corpus);
    converse_lexicon_save(lexicon);
    converse_stage_end("save");
    /**
     * The answer ranker is skippable, and on a large corpus it has to be. It is
     * the only consumer of this model, it degrades to the unmodified base score
     * when the model is absent (see answer_predict_score), and training it is the
     * stage that stops a big corpus being usable at all: pushing every entry x
     * query-type superlinearly outgrows ingest, so 88k Fortran sentences never
     * reach the model that the recombination and byte-model tracks actually need.
     * CONVERSE_NO_PREDICT=1 trades a ranking refinement for reaching those tracks.
     */
    if (0 != converse_predict_on()) {
      libxs_predict_t* trained = converse_predict_train(corpus,
        predict_profile);
      if (NULL != trained) {
        libxs_predict_destroy(answer_model);
        answer_model = trained;
        converse_predict_save(answer_model);
      }
      converse_stage_end("predict_train");
    }
    else {
      fprintf(stderr, "predict: training skipped"
        " (set CONVERSE_PREDICT=1 to enable)\n");
    }
  }
  else fprintf(stderr, "warm start: reusing %s\n", converse_path_corpus);
  free(basenames);
  /* Before the facts, because a widened person class is what makes new facts
     extractable at all: run it after and the learned terms would sit unused. */
  answer_relation_rules_learn(corpus, lexicon, rules, nrules);
  token_emb_pair_probe(corpus, lexicon, rules, nrules);
  answer_relation_facts_build(corpus);
  answer_relation_facts_report(stderr);
  converse_stage_end("f_relation");
  answer_identity_facts_build(corpus);
  answer_identity_facts_report(stderr);
  converse_stage_end("f_identity");
  answer_describe_facts_build(corpus);
  answer_describe_facts_report(stderr);
  converse_stage_end("f_describe");
  answer_docdef_facts_build(corpus);
  answer_docdef_facts_report(stderr);
  converse_stage_end("f_docdef");

  libxs_registry_info(corpus, &rinfo);
  fprintf(stderr, "corpus: %lu sentences\n", (unsigned long)rinfo.size);
  if (corpus_chain_dropped > 0) {
    fprintf(stderr, "  fingerprint chain cap reached: %ld texts dropped"
      " (CONVERSE_CHAIN_MAX=%u)\n", corpus_chain_dropped, corpus_chain_max());
  }
  predict_ntotal = (long)rinfo.size;

  if (0 != learn_mode) {
    fprintf(stderr, "learned: %s, %s, %s\n", converse_path_corpus,
      converse_path_lexicon, converse_path_predict);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    answer_bridge_free_loaded();
    answer_relation_rules_free();
    answer_relation_facts_free();
    answer_identity_facts_free();
    answer_describe_facts_free();
    answer_docdef_facts_free();
    return EXIT_SUCCESS;
  }

  if (0 != predict_eval_mode || 0 != complete_mode) {
    int mode_result = EXIT_FAILURE;
    int use_predict = (0 == strcmp(ngram_kind, "predict")) ? 1 : 0;
    int use_embed = (0 == strcmp(ngram_kind, "embed")) ? 1 : 0;
    int use_rerank = (0 == strcmp(ngram_kind, "rerank")) ? 1 : 0;
    int use_knnlm = (0 == strcmp(ngram_kind, "knnlm")) ? 1 : 0;
    int use_hier = (0 == strcmp(ngram_kind, "hier")) ? 1 : 0;
    libxs_predict_t* token_model = NULL;
    converse_hier_t* hier_model = NULL;
    converse_hier_t* rescore_model = NULL;
    libxs_registry_t* test_corpus = NULL;
    const libxs_registry_t* eval_corpus = corpus;
    /**
     * A -T prefix supplies a fixed held-out corpus: train on all of the main
     * corpus (holdout forced 0) and evaluate on the separate test set. This
     * keeps the test set identical across training-corpus sizes, so BPC is
     * comparable along a scaling curve.
     */
    if (NULL != test_prefix) {
      test_corpus = libxs_registry_create();
      converse_stage_begin();
      if (NULL != test_corpus && EXIT_SUCCESS == corpus_ingest_basename(
        test_corpus, test_prefix, lexicon, rules, nrules))
      {
        libxs_registry_info_t tinfo;
        libxs_registry_info(test_corpus, &tinfo);
        fprintf(stderr, "test corpus: %lu sentences (%s)\n",
          (unsigned long)tinfo.size, test_prefix);
        eval_corpus = test_corpus;
        ngram_holdout = 0;
      }
      else {
        fprintf(stderr, "failed to load test corpus: %s\n", test_prefix);
        libxs_registry_destroy(test_corpus);
        test_corpus = NULL;
      }
    }
    rescore_model = answer_hier_build(corpus);
    if (0 != use_hier) {
      hier_model = converse_hier_build(corpus, ngram_holdout,
        predict_ntotal, ngram_maxorder());
    }
    else if (0 != use_predict) {
      token_model = token_predict_build(corpus, lexicon, rules, nrules,
        predict_profile, 0, ngram_holdout, 2);
    }
    else {
      if (GRAN_BPE == ngram_gran_mode()) {
        bpe_build(corpus, ngram_holdout);
      }
      converse_stage_end("test_ingest");
      ngram_model = ngram_build(corpus, lexicon, rules, nrules, ngram_holdout);
      ngram_backoff_build(ngram_model, lexicon);
      converse_stage_end("ngram_build");
      /* The embedding-rank probe reads the embedding without being a prediction
         KIND, so it declares the dependency itself rather than forcing -K embed,
         which the dispatch below would route away from gen-eval. Split out of
         the kind chain so the probe does not also build a predict store: that
         store is what gen-eval scores as the bank's kind-different slot, and
         handing it one it would not otherwise have would change the run being
         measured. */
      if (0 != use_embed || 0 != use_knnlm || 0 != ngram_gen_embrank()
        || 0 != ngram_gen_embcand()
        || 0 != (ngram_bank_slots() & (1u << NGRAM_BANK_EMB)))
      {
        token_emb_build(corpus, lexicon, rules, nrules, ngram_holdout);
        converse_stage_end("emb_build");
      }
      if (0 != use_embed || 0 != use_knnlm) {
        token_model = token_predict_build(corpus, lexicon, rules, nrules,
          predict_profile, 1, ngram_holdout,
          (0 != use_knnlm) ? knnlm_ctxlen() : 2);
        converse_stage_end("token_predict");
      }
      else if (0 != use_rerank) {
        token_model = rerank_build(corpus, ngram_model, lexicon, rules,
          nrules, ngram_order, predict_profile, ngram_holdout);
      }
      /* The expert bank can carry the predict store as a slot, which needs the
         store in the plain n-gram path too. Gated on the slot being asked for,
         so every other configuration stays byte-identical. */
      else if (0 != (ngram_bank_slots() & (1u << NGRAM_BANK_PREDICT))) {
        token_model = token_predict_build(corpus, lexicon, rules, nrules,
          predict_profile, 0, ngram_holdout, 2);
        /* Separate timers: one stage covering both made a slow build
           indistinguishable from slow scoring, and attributing the cost to the
           wrong one of those is how a fix gets aimed at the wrong code. */
        converse_stage_end("predict_build");
        /* Converge and publish the escape weights BEFORE scoring, so frozen
           mode freezes something converged rather than the uniform prior. The
           model is mutable only here, which is also the only place it may be
           written: scoring requires it read-only. */
        if (0 != ngram_bank_frozen() && NULL != token_model) {
          ngram_bank_warmup(token_model, (int)libxs_lexicon_size(lexicon));
          converse_stage_end("predict_warmup");
        }
      }
    }
    converse_stage_begin();
    if (0 != predict_eval_mode) {
      if (0 != use_hier) {
        mode_result = converse_hier_eval(hier_model, eval_corpus,
          ngram_holdout, (eval_corpus == corpus) ? predict_ntotal : 0,
          "metatoken");
      }
      else if (0 != use_predict || 0 != use_embed) {
        char label[64];
        sprintf(label, "%s:%s", use_embed ? "embed" : predict_profile->name,
          predict_profile->name);
        mode_result = token_predict_eval(token_model, eval_corpus, lexicon,
          rules, nrules, use_embed, ngram_holdout,
          use_embed ? label : predict_profile->name);
      }
      else if (0 != use_rerank) {
        char label[64];
        sprintf(label, "rerank:%s", predict_profile->name);
        mode_result = rerank_eval(ngram_model, token_model, eval_corpus,
          lexicon, rules, nrules, ngram_order, ngram_holdout, label);
      }
      else if (0 != use_knnlm) {
        char label[64];
        sprintf(label, "knnlm:%s", predict_profile->name);
        mode_result = knnlm_eval(ngram_model, token_model, eval_corpus,
          lexicon, rules, nrules, ngram_order, ngram_holdout, label);
      }
      else if (NULL != getenv("CONVERSE_SLOT_PROBE")) {
        slot_probe_run(eval_corpus, lexicon, rules, nrules, ngram_holdout);
        mode_result = EXIT_SUCCESS;
      }
      else if (NULL != getenv("CONVERSE_RECOMB")) {
        if (NULL != answer_hier_model) {
          const char* env = getenv("CONVERSE_RECOMB");
          const int limit = ('\0' != *env && 0 < atoi(env)) ? atoi(env) : 50;
          converse_recomb_host_t host;
          host.ends_sentence = text_ends_sentence;
          host.is_wordchar = libxs_lexeme_is_word_char;
          host.entry_build = corpus_entry_build;
          host.word_prob = recomb_word_prob;
          host.maxorder = ngram_maxorder();
          converse_recomb_probe_run(corpus, lexicon, rules, nrules, limit,
            answer_hier_model, &host);
          mode_result = EXIT_SUCCESS;
        }
        else {
          fprintf(stderr, "recomb needs CONVERSE_HIER_RESCORE=1"
            " (it judges seams with the byte model)\n");
          mode_result = EXIT_FAILURE;
        }
      }
      else if (NULL != getenv("CONVERSE_GEN_EVAL")) {
        mode_result = ngram_gen_eval(ngram_model, eval_corpus, lexicon, rules,
          nrules, ngram_holdout, ngram_kind, token_model, use_embed);
      }
      else {
        mode_result = ngram_eval(ngram_model, eval_corpus, lexicon, rules,
          nrules, ngram_order, ngram_holdout, ngram_kind, token_model,
          use_embed);
      }
      if (0 == use_hier) ngram_stats(ngram_model);
      /* Bounds any cut rule against the model just trained, so it runs after
         the store exists and reads only held-out entries. */
      if (NULL != getenv("CONVERSE_SYLLABLE_ORACLE") && NULL != ngram_model) {
        ngram_syllable_oracle(ngram_model, lexicon, rules, nrules, eval_corpus,
          ngram_holdout, ngram_maxorder());
      }
      converse_stage_end("eval");
      converse_stage_report();
    }
    else if (0 != use_hier) {
      fprintf(stderr, "hier is currently an evaluation-only model (-E)\n");
      mode_result = EXIT_FAILURE;
    }
    else {
      converse_recomb_host_t recomb_hostcb;
      int recomb_ready = 0;
      if (0 != recomb_compose_on()) {
        if (NULL == answer_hier_model) {
          fprintf(stderr, "compose needs CONVERSE_HIER_RESCORE=1"
            " (it judges seams with the byte model)\n");
        }
        else {
          recomb_hostcb.ends_sentence = text_ends_sentence;
          recomb_hostcb.is_wordchar = libxs_lexeme_is_word_char;
          recomb_hostcb.entry_build = corpus_entry_build;
          recomb_hostcb.word_prob = recomb_word_prob;
          recomb_hostcb.maxorder = ngram_maxorder();
          /* One corpus pass, kept for the session: per-query would dominate. */
          if (EXIT_SUCCESS == converse_recomb_open(corpus, lexicon,
            answer_hier_model, &recomb_hostcb))
          {
            recomb_ready = 1;
          }
          else fprintf(stderr, "compose: pivot index could not be built\n");
        }
      }
      printf("> ");
      fflush(stdout);
      while (NULL != fgets(line, (int)sizeof(line), stdin)) {
        size_t len = strlen(line);
        int is_q;
        while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
        if (0 == len) { printf("> "); fflush(stdout); continue; }
        is_q = is_question_query(line, len, lexicon, rules, nrules);
        /**
         * Questions answer from the corpus first (with a continuation on top),
         * independent of the -K generator; only non-questions use the per-kind
         * token generators. The predict/embed kinds lack an n-gram model, so
         * they keep their own completion path.
         */
        if (0 != is_q && 0 == use_predict && 0 == use_embed) {
          complete_respond(corpus, lexicon, rules, nrules, answer_model,
            predict_profile, budget, line, (int)len);
        }
        else if (0 != use_predict || 0 != use_embed) {
          token_complete(token_model, lexicon, rules, nrules, use_embed, line,
            (int)len);
        }
        else if (0 != use_rerank) {
          rerank_complete(ngram_model, token_model, lexicon, rules, nrules,
            ngram_order, line, (int)len);
        }
        else if (0 != use_knnlm) {
          knnlm_complete(ngram_model, token_model, lexicon, rules, nrules,
            ngram_order, line, (int)len);
        }
        else ngram_complete(ngram_model, lexicon, rules, nrules,
          ngram_order, line, (int)len);
        printf("> ");
        fflush(stdout);
      }
      if (0 != recomb_ready) converse_recomb_close();
      mode_result = EXIT_SUCCESS;
    }
    answer_hier_model = NULL;
    converse_hier_destroy(rescore_model);
    libxs_predict_destroy(token_model);
    converse_hier_destroy(hier_model);
    libxs_registry_destroy(test_corpus);
    knnlm_cache_free();
    token_emb_free();
    bpe_free();
    libxs_ngram_destroy(&converse_ngram);
    if (0 != converse_skip_on) libxs_ngram_destroy(&converse_skip);
    converse_skip_on = 0;
    ngram_model = NULL;
    converse_lexicon_save(lexicon);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    answer_bridge_free_loaded();
    answer_relation_rules_free();
    answer_relation_facts_free();
    answer_identity_facts_free();
    answer_describe_facts_free();
    answer_docdef_facts_free();
    return mode_result;
  }

  if (0 != eval_mode) {
    int eval_result;
    converse_hier_t* rescore_model = answer_hier_build(corpus);
    eval_result = eval_converse(corpus, lexicon, rules, nrules,
      answer_model, predict_profile);
    answer_hier_model = NULL;
    converse_hier_destroy(rescore_model);
    converse_lexicon_save(lexicon);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    answer_bridge_free_loaded();
    answer_relation_rules_free();
    answer_relation_facts_free();
    answer_identity_facts_free();
    answer_describe_facts_free();
    answer_docdef_facts_free();
    return eval_result;
  }

  { libxs_spatial_t spatial;
    converse_hier_t* rescore_model = NULL;
    if (EXIT_SUCCESS != corpus_spatial_build(&spatial, corpus)) {
      libxs_predict_destroy(answer_model);
      libxs_lexicon_destroy(lexicon);
      libxs_registry_destroy(corpus);
      answer_bridge_free_loaded();
      answer_relation_rules_free();
      answer_relation_facts_free();
      answer_identity_facts_free();
      answer_describe_facts_free();
      answer_docdef_facts_free();
      return EXIT_FAILURE;
    }
    ngram_model = ngram_build(corpus, lexicon, rules, nrules, 0);
    ngram_backoff_build(ngram_model, lexicon);
    rescore_model = answer_hier_build(corpus);
    conv_reset();
    printf("> ");
    fflush(stdout);
    while (NULL != fgets(line, (int)sizeof(line), stdin)) {
      size_t len = strlen(line);
      libxs_fprint_t query;
      size_t shape;
      while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
      if (0 == len) { printf("> "); fflush(stdout); continue; }
      shape = len;
      libxs_fprint(&query, LIBXS_DATATYPE_U8, line, 1,
        &shape, NULL, FPRINT_ORDER, 0, 0, 0);
      respond(&spatial, corpus, line, len, &query, budget,
        lexicon, rules, nrules, answer_model, predict_profile);
      printf("> ");
      fflush(stdout);
    }
    answer_hier_model = NULL;
    converse_hier_destroy(rescore_model);
    libxs_spatial_destroy(&spatial);
  }

  libxs_ngram_destroy(&converse_ngram);
  if (0 != converse_skip_on) libxs_ngram_destroy(&converse_skip);
  converse_skip_on = 0;
  ngram_model = NULL;
  converse_lexicon_save(lexicon);
  libxs_predict_destroy(answer_model);
  libxs_lexicon_destroy(lexicon);
  libxs_registry_destroy(corpus);
  answer_bridge_free_loaded();
  answer_relation_rules_free();
  answer_relation_facts_free();
  answer_identity_facts_free();
  answer_describe_facts_free();
  answer_docdef_facts_free();
  return EXIT_SUCCESS;
}


static void corpus_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata)
{
  LIBXS_UNUSED(value); LIBXS_UNUSED(key);
  LIBXS_UNUSED(key_size); LIBXS_UNUSED(value_size);
  LIBXS_UNUSED(udata);
}


static const answer_predict_profile_t* answer_predict_profile_default(void)
{
  return answer_predict_profiles;
}


static const answer_predict_profile_t* answer_predict_profile_find(
  const char* name)
{
  const answer_predict_profile_t* result = NULL;
  size_t nprofiles = sizeof(answer_predict_profiles)
    / sizeof(answer_predict_profiles[0]);
  size_t profile_pos;
  if (NULL != name) {
    for (profile_pos = 0; profile_pos < nprofiles && NULL == result;
      ++profile_pos)
    {
      if (0 == strcmp(name, answer_predict_profiles[profile_pos].name)) {
        result = answer_predict_profiles + profile_pos;
      }
    }
  }
  return result;
}


static void answer_predict_profile_list(FILE* stream)
{
  size_t nprofiles = sizeof(answer_predict_profiles)
    / sizeof(answer_predict_profiles[0]);
  size_t profile_pos;
  if (NULL != stream) {
    fprintf(stream, "  profiles:");
    for (profile_pos = 0; profile_pos < nprofiles; ++profile_pos) {
      fprintf(stream, " %s", answer_predict_profiles[profile_pos].name);
    }
    fprintf(stream, "\n");
  }
}


static void answer_bridge_free_const(const char* ptr)
{
  union { const char* cptr; void* ptr; } cvt;
  if (NULL != ptr) {
    cvt.cptr = ptr;
    free(cvt.ptr);
  }
}


static char* answer_bridge_copy_trim(const char* text)
{
  char* result = NULL;
  const char* begin;
  const char* end;
  size_t size;
  if (NULL != text) {
    begin = text;
    while ('\0' != *begin && 0 != isspace((unsigned char)*begin)) ++begin;
    end = begin + strlen(begin);
    while (end > begin && 0 != isspace((unsigned char)end[-1])) --end;
    size = (size_t)(end - begin);
    result = (char*)malloc(size + 1);
    if (NULL != result) {
      memcpy(result, begin, size);
      result[size] = '\0';
    }
  }
  return result;
}


static void answer_bridge_free_loaded(void)
{
  size_t bridge_pos;
  for (bridge_pos = 0; bridge_pos < answer_bridge_loaded_size; ++bridge_pos) {
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].name);
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].query);
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].evidence);
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].reply);
  }
  free(answer_bridge_loaded);
  answer_bridge_loaded = NULL;
  answer_bridge_loaded_size = 0;
}


static int answer_bridge_append_loaded(const char* name, const char* query,
  const char* evidence, const char* score, const char* reply)
{
  int result = EXIT_FAILURE;
  answer_bridge_t bridge;
  answer_bridge_t* bridges;
  LIBXS_MEMZERO(&bridge);
  bridge.name = answer_bridge_copy_trim(name);
  bridge.query = answer_bridge_copy_trim(query);
  bridge.evidence = answer_bridge_copy_trim(evidence);
  bridge.reply = answer_bridge_copy_trim(reply);
  bridge.score = (NULL != score) ? atof(score) : 0.0;
  if (bridge.score <= 0.0) bridge.score = 0.90;
  bridges = (answer_bridge_t*)realloc(answer_bridge_loaded,
    (answer_bridge_loaded_size + 1) * sizeof(*bridges));
  if (NULL != bridge.name && '\0' != bridge.name[0]
    && NULL != bridge.query && '\0' != bridge.query[0]
    && NULL != bridge.evidence && '\0' != bridge.evidence[0]
    && NULL != bridge.reply && '\0' != bridge.reply[0]
    && NULL != bridges)
  {
    answer_bridge_loaded = bridges;
    answer_bridge_loaded[answer_bridge_loaded_size] = bridge;
    ++answer_bridge_loaded_size;
    result = EXIT_SUCCESS;
  }
  else {
    answer_bridge_free_const(bridge.name);
    answer_bridge_free_const(bridge.query);
    answer_bridge_free_const(bridge.evidence);
    answer_bridge_free_const(bridge.reply);
  }
  return result;
}


static int answer_bridge_parse_line(char* line)
{
  int result = EXIT_FAILURE;
  char* fields[5];
  char* cursor;
  int field_pos;
  if (NULL == line) return EXIT_FAILURE;
  cursor = line;
  for (field_pos = 0; field_pos < 5; ++field_pos) fields[field_pos] = NULL;
  for (field_pos = 0; field_pos < 4 && NULL != cursor; ++field_pos) {
    char* sep = strchr(cursor, '|');
    if (NULL != sep) {
      *sep = '\0';
      fields[field_pos] = cursor;
      cursor = sep + 1;
    }
    else {
      cursor = NULL;
    }
  }
  if (NULL != cursor) {
    fields[4] = cursor;
    result = answer_bridge_append_loaded(fields[0], fields[1], fields[2],
      fields[3], fields[4]);
  }
  return result;
}


static size_t answer_bridge_load_file(const char* path)
{
  size_t result = 0;
  FILE* file;
  if (NULL == path) return 0;
  answer_bridge_free_loaded();
  file = fopen(path, "r");
  if (NULL != file) {
    char line[BRIDGE_LINE_MAX];
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len = strlen(line);
      char* begin;
      while (len > 0 && ('\n' == line[len - 1] || '\r' == line[len - 1])) {
        line[--len] = '\0';
      }
      begin = line;
      while ('\0' != *begin && 0 != isspace((unsigned char)*begin)) ++begin;
      if ('\0' != *begin && '#' != *begin
        && EXIT_SUCCESS == answer_bridge_parse_line(begin))
      {
        ++result;
      }
    }
    fclose(file);
  }
  return result;
}


static void answer_bridge_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "bridges: %lu loaded\n",
      (unsigned long)answer_bridge_loaded_size);
  }
}


static void answer_relation_rules_free(void)
{
  free(answer_relation_rules);
  answer_relation_rules = NULL;
  answer_relation_rules_size = 0;
  free(answer_lexnorms);
  answer_lexnorms = NULL;
  answer_lexnorms_size = 0;
}


static char* answer_relation_rule_trim(char* text)
{
  char* result = text;
  char* end;
  if (NULL != result) {
    while ('\0' != *result && 0 != isspace((unsigned char)*result)) ++result;
    end = result + strlen(result);
    while (end > result && 0 != isspace((unsigned char)end[-1])) --end;
    *end = '\0';
  }
  return result;
}


static int answer_relation_rule_kind(const char* text)
{
  int result = 0;
  if (NULL != text) {
    if (0 == strcmp(text, "alias")) result = RELATION_RULE_ALIAS;
    else if (0 == strcmp(text, "person")) result = RELATION_RULE_PERSON;
    else if (0 == strcmp(text, "skip")) result = RELATION_RULE_SKIP;
    else if (0 == strcmp(text, "negate")) result = RELATION_RULE_NEGATE;
    else if (0 == strcmp(text, "norm")) result = RELATION_RULE_NORM;
  }
  return result;
}


static int answer_relation_rule_append(int kind, const char* relation,
  const char* term)
{
  int result = EXIT_FAILURE;
  answer_relation_rule_t* rules;
  if (kind > 0 && NULL != term && '\0' != term[0]
    && strlen(term) < sizeof(answer_relation_rules[0].term)
    && ((RELATION_RULE_ALIAS != kind && RELATION_RULE_NORM != kind)
      || (NULL != relation && '\0' != relation[0]
        && strlen(relation) < sizeof(answer_relation_rules[0].relation))))
  {
    rules = (answer_relation_rule_t*)realloc(answer_relation_rules,
      (answer_relation_rules_size + 1) * sizeof(*rules));
    if (NULL != rules) {
      answer_relation_rules = rules;
      LIBXS_MEMZERO(answer_relation_rules + answer_relation_rules_size);
      answer_relation_rules[answer_relation_rules_size].kind = kind;
      if (NULL != relation) {
        strcpy(answer_relation_rules[answer_relation_rules_size].relation,
          relation);
      }
      strcpy(answer_relation_rules[answer_relation_rules_size].term, term);
      ++answer_relation_rules_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_relation_rule_parse_line(char* line)
{
  int result = EXIT_FAILURE;
  char* fields[3];
  char* cursor;
  int field_pos;
  if (NULL == line) return EXIT_FAILURE;
  cursor = answer_relation_rule_trim(line);
  for (field_pos = 0; field_pos < 3; ++field_pos) fields[field_pos] = NULL;
  for (field_pos = 0; field_pos < 3 && NULL != cursor; ++field_pos) {
    char* sep = strchr(cursor, '|');
    if (NULL != sep) {
      *sep = '\0';
      fields[field_pos] = answer_relation_rule_trim(cursor);
      cursor = sep + 1;
    }
    else {
      fields[field_pos] = answer_relation_rule_trim(cursor);
      cursor = NULL;
    }
  }
  if (NULL != fields[0] && NULL != fields[1]) {
    int kind = answer_relation_rule_kind(fields[0]);
    if ((RELATION_RULE_ALIAS == kind || RELATION_RULE_NORM == kind)
      && NULL != fields[2])
    {
      result = answer_relation_rule_append(kind, fields[1], fields[2]);
    }
    else if (RELATION_RULE_ALIAS != kind && RELATION_RULE_NORM != kind) {
      result = answer_relation_rule_append(kind, NULL, fields[1]);
    }
  }
  return result;
}


static size_t answer_relation_rules_load_file(const char* path)
{
  size_t result = 0;
  FILE* file;
  if (NULL == path) return 0;
  /* Appends: successive files layer (language rules, then per-corpus rules). */
  file = fopen(path, "r");
  if (NULL != file) {
    char line[RELATION_LINE_MAX];
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len = strlen(line);
      char* begin;
      while (len > 0 && ('\n' == line[len - 1] || '\r' == line[len - 1])) {
        line[--len] = '\0';
      }
      begin = answer_relation_rule_trim(line);
      if ('\0' != *begin && '#' != *begin
        && EXIT_SUCCESS == answer_relation_rule_parse_line(begin))
      {
        ++result;
      }
    }
    fclose(file);
  }
  answer_lexnorms_build();
  return result;
}


static void answer_relation_rules_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "relation rules: %lu loaded\n",
      (unsigned long)answer_relation_rules_size);
  }
}


/**
 * Score named successions with the directed embedding: CONVERSE_EMB_PAIRS="a>b,c>d".
 *
 * Aimed at the one failure the paper calls unsolvable with counts. `be made fat`
 * (true) and `be made in` (false) each occur EXACTLY ONCE, so relative
 * attestation has nothing to compare and the seam gate cannot separate them.
 * Sparse PMI was already retired here for a related reason: with zero observed
 * co-occurrence both sides collapse onto the smoothing floor. A LOW-RANK
 * COMPLETION is a different instrument -- it interpolates from the whole matrix,
 * so it assigns a graded score to a pair it never saw, which is the only way a
 * continuous signal can exist where the counts are flat. Whether that score
 * ORDERS good before bad is the question; this prints it rather than assuming it.
 */
static void token_emb_pair_probe(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  const char* spec = getenv("CONVERSE_EMB_PAIRS");
  if (NULL != spec && '\0' != *spec && NULL != lexicon) {
    const unsigned int vocab = libxs_lexicon_size(lexicon);
    const char* at = spec;
    if (0 == token_emb_ready()) {
      token_emb_build(corpus, lexicon, rules, nrules, 0);
    }
    fprintf(stderr, "successor pairs (directed=%d temp=%.2f):\n",
      token_emb_directed(), ngram_emb_temp());
    while ('\0' != *at) {
      char lhs[64], rhs[64];
      int ln = 0, rn = 0;
      while ('\0' != *at && '>' != *at && ',' != *at) {
        if (ln + 1 < (int)sizeof(lhs)) lhs[ln++] = *at;
        ++at;
      }
      if ('>' == *at) ++at;
      while ('\0' != *at && ',' != *at) {
        if (rn + 1 < (int)sizeof(rhs)) rhs[rn++] = *at;
        ++at;
      }
      if (',' == *at) ++at;
      lhs[ln] = '\0';
      rhs[rn] = '\0';
      if (0 < ln && 0 < rn) {
        const unsigned int a = libxs_lexicon_id(lexicon, lhs, ln, 0, 0);
        const unsigned int b = libxs_lexicon_id(lexicon, rhs, rn, 0, 0);
        if (0 != a && 0 != b) {
          const double p = token_emb_succ_prob(&a, 1, b, vocab,
            ngram_emb_temp());
          const int rank = token_emb_succ_rank(&a, 1, b, vocab);
          fprintf(stderr, "  %-10s > %-12s p=%.3e rank=%d of %u\n",
            lhs, rhs, p, rank, vocab);
        }
        else {
          fprintf(stderr, "  %-10s > %-12s (not in vocabulary)\n", lhs, rhs);
        }
      }
    }
  }
}


static int answer_rules_learn_count(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RULES_LEARN");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


static double answer_rules_learn_env(const char* name, double fallback)
{
  const char* env = getenv(name);
  double result = fallback;
  if (NULL != env && '\0' != *env) result = atof(env);
  return result;
}


/**
 * Propose person-class terms from the corpus instead of requiring every one to
 * be configured by hand.
 *
 * WHY THIS IS THE PIECE THAT WAS MISSING. The fact layer is the one path in this
 * system that produces unseen, grammatical, attributed sentences without
 * diverging, because what it renders is a PROPOSITION rather than a token path --
 * there is nowhere for it to divert to. But its reach is set by `person|...`
 * rules written by hand, so the capability was a demonstration and not learning.
 * Every consumer already goes through answer_relation_rule_has_term, so widening
 * the class here widens facts, replies and citations at once.
 *
 * The geometry is the directed successor embedding: two words sit together when
 * they are FOLLOWED by similar things, and a class like "person" is exactly a set
 * of words that take the same continuations ("the girl said", "the boy said").
 * That is also why this is not circular with the seed list -- the similarity is
 * computed from corpus succession, never from the rule file.
 *
 * SIDE-SIGNALS decide acceptance, not similarity alone:
 *  - nsrc, the number of distinct sections the word occurs in. This separates a
 *    CLASS member from a CHARACTER: "girl" recurs across tales, "Rapunzel" lives
 *    in one. A term with a single source is a name wearing a class's clothes.
 *  - freq, so a hapax cannot enter the class on one lucky neighbourhood.
 *  - two thresholds. At or above accept the term becomes a rule; between
 *    speculate and accept it is REPORTED AND NOT USED, so a reply never rests on
 *    a guess without that having been a decision. Unlocking those is a separate
 *    act, which is what makes "speculation" a mode rather than an accident.
 */
static size_t answer_relation_rules_learn(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  enum { LEARN_SECT_MAX = 64 };
  size_t result = 0;
  const int want = answer_rules_learn_count();
  const unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  if (0 < want && 0 < vocab && NULL != corpus) {
    const double accept = answer_rules_learn_env("CONVERSE_RULES_ACCEPT", 0.55);
    const double specul = answer_rules_learn_env("CONVERSE_RULES_SPECULATE",
      0.40);
    const int minsrc = (int)answer_rules_learn_env("CONVERSE_RULES_MINSRC", 3.0);
    const int minfreq = (int)answer_rules_learn_env("CONVERSE_RULES_MINFREQ",
      5.0);
    double* centroid = (double*)calloc(TOKEN_EMB_DIM, sizeof(double));
    double* scentroid = (double*)calloc(TOKEN_EMB_DIM, sizeof(double));
    long* freq = (long*)calloc((size_t)vocab + 1, sizeof(long));
    unsigned char* seen = (unsigned char*)calloc(((size_t)vocab + 1)
      * LEARN_SECT_MAX, sizeof(unsigned char));
    char (*sect)[ENTRY_SECTION_MAX] = (char (*)[ENTRY_SECTION_MAX])calloc(
      LEARN_SECT_MAX, ENTRY_SECTION_MAX);
    if (NULL != centroid && NULL != scentroid && NULL != freq && NULL != seen
      && NULL != sect)
    {
      int nsect = 0, nseed = 0, d, shown = 0;
      const void* key = NULL;
      size_t cursor = 0;
      unsigned int id;
      void* value;
      /* The learner needs the embedding, so it says so rather than depending on
         a prediction kind having been asked for. */
      if (0 == token_emb_ready()) {
        token_emb_build(corpus, lexicon, rules, nrules, 0);
      }
      value = libxs_registry_begin(corpus, &key, &cursor);
      while (NULL != value) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        libxs_lexeme_stream_t stream;
        libxs_lexeme_stream_init(&stream);
        /* Sentence scale only: the corpus holds each text at sentence AND
           paragraph scale, so counting both would double every frequency and
           make one source look like two. */
        if (SCALE_SENTENCE == entry->scale && entry->text_len > 0
          && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
            (const unsigned char*)entry->text, (size_t)entry->text_len,
            rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
        {
          int at = -1, s;
          size_t pos;
          for (s = 0; s < nsect && at < 0; ++s) {
            if (0 == strncmp(sect[s], entry->section, ENTRY_SECTION_MAX - 1)) {
              at = s;
            }
          }
          if (at < 0 && nsect < LEARN_SECT_MAX && 0 < entry->section_len) {
            int copy = entry->section_len;
            if (copy > ENTRY_SECTION_MAX - 1) copy = ENTRY_SECTION_MAX - 1;
            memcpy(sect[nsect], entry->section, (size_t)copy);
            sect[nsect][copy] = '\0';
            at = nsect++;
          }
          for (pos = 0; pos < stream.size; ++pos) {
            const libxs_lexeme_t* lex = stream.data + pos;
            if (0 != (lex->flags & LIBXS_LEXEME_WORD) && 0 != lex->id
              && lex->id <= vocab)
            {
              ++freq[lex->id];
              if (0 <= at) seen[(size_t)lex->id * LEARN_SECT_MAX + at] = 1;
            }
          }
        }
        libxs_lexeme_stream_release(&stream);
        value = libxs_registry_next(corpus, &key, &cursor);
      }
      { size_t rule_pos;
        for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
          const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
          if (RELATION_RULE_PERSON == rule->kind) {
            const unsigned int sid = libxs_lexicon_id(lexicon, rule->term,
              (int)strlen(rule->term), 0, 0);
            if (0 != sid && sid <= vocab && 0 == token_emb_isnull(sid)) {
              const double* e = token_emb_get(sid);
              const double* v = token_semb_get(sid);
              for (d = 0; d < TOKEN_EMB_DIM; ++d) {
                centroid[d] += e[d];
                scentroid[d] += v[d];
              }
              ++nseed;
            }
          }
        }
      }
      if (0 < nseed) {
        double norm = 0.0, snorm = 0.0;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) {
          norm += centroid[d] * centroid[d];
          snorm += scentroid[d] * scentroid[d];
        }
        norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
        snorm = (snorm > 0.0) ? (1.0 / sqrt(snorm)) : 0.0;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) {
          centroid[d] *= norm;
          scentroid[d] *= snorm;
        }
        fprintf(stderr, "rule learning: %d seeds, %d sections,"
          " accept>=%.2f speculate>=%.2f minsrc=%d minfreq=%d\n",
          nseed, nsect, accept, specul, minsrc, minfreq);
        while (shown < want) {
          unsigned int best = 0;
          double bestcos = 0.0;
          int bestsrc = 0;
          for (id = 1; id <= vocab; ++id) {
            if (freq[id] >= minfreq && 0 == token_emb_isnull(id)) {
              int textlen = 0;
              const char* text = libxs_lexicon_text(lexicon, id, &textlen, NULL);
              if (NULL != text && 0 < textlen
                && 0 == answer_relation_rule_has_term(RELATION_RULE_PERSON,
                  text, textlen))
              {
                double cf = 0.0, cp = 0.0, cos;
                double vnorm = 0.0;
                int nsrc = 0, s;
                for (d = 0; d < TOKEN_EMB_DIM; ++d) {
                  const double vd = token_semb_get(id)[d];
                  cf += centroid[d] * token_emb_get(id)[d];
                  cp += scentroid[d] * vd;
                  vnorm += vd * vd;
                }
                /* token_semb is unnormalized (it is raw V), so the successor side
                   needs its own normalization to be a cosine at all. */
                vnorm = (vnorm > 0.0) ? (1.0 / sqrt(vnorm)) : 0.0;
                cp *= vnorm;
                /**
                 * A class member must look like one in BOTH directions: followed
                 * by what persons are followed by, AND preceded by what persons
                 * are preceded by. Scoring the weaker side is what excludes a
                 * function word -- "there" takes person-like continuations but
                 * nothing puts a determiner in front of it, and nsrc cannot see
                 * that because a function word occurs in EVERY source (it had the
                 * highest nsrc of any candidate, 58).
                 */
                cos = (cf < cp) ? cf : cp;
                for (s = 0; s < nsect; ++s) {
                  nsrc += seen[(size_t)id * LEARN_SECT_MAX + s];
                }
                if (nsrc >= minsrc && (0 == best || cos > bestcos)) {
                  best = id;
                  bestcos = cos;
                  bestsrc = nsrc;
                }
              }
            }
          }
          if (0 == best || bestcos < specul) shown = want;
          else {
            int textlen = 0;
            const char* text = libxs_lexicon_text(lexicon, best, &textlen, NULL);
            const int ok = (bestcos >= accept) ? 1 : 0;
            fprintf(stderr, "  %-14s min-cos=%.3f freq=%ld nsrc=%d -> %s\n",
              (NULL != text) ? text : "?", bestcos, freq[best], bestsrc,
              (0 != ok) ? "ACCEPTED" : "speculative (not used)");
            if (0 != ok && NULL != text
              && EXIT_SUCCESS == answer_relation_rule_append(
                RELATION_RULE_PERSON, NULL, text))
            {
              ++result;
            }
            else if (0 == ok) {
              /* Keep it out of the class so no reply can rest on it, but stop it
                 being re-proposed every round. */
              freq[best] = 0;
            }
            ++shown;
          }
        }
        fprintf(stderr, "rule learning: %lu term%s accepted into person class\n",
          (unsigned long)result, (1 == result) ? "" : "s");
      }
    }
    free(centroid);
    free(scentroid);
    free(freq);
    free(seen);
    free(sect);
  }
  return result;
}


/**
 * Materialize the `norm|from|to` rules as a libxs_lexnorm_t table, which the
 * tokenizer applies during encoding. This is the library's data-only
 * normalization facility: the mechanism is language-agnostic, the vocabulary
 * lives in the caller-owned rule file. Rebuilt whenever rules are (re)loaded;
 * with no norm rules the table is empty and encoding is bit-identical.
 */
static void answer_lexnorms_build(void)
{
  size_t rule_pos;
  free(answer_lexnorms);
  answer_lexnorms = NULL;
  answer_lexnorms_size = 0;
  if (0 == answer_relation_rules_size) return;
  answer_lexnorms = (libxs_lexnorm_t*)calloc(answer_relation_rules_size,
    sizeof(*answer_lexnorms));
  if (NULL == answer_lexnorms) return;
  for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
    const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
    if (RELATION_RULE_NORM == rule->kind
      && strlen(rule->relation) <= LIBXS_LEXEME_MAXBYTES
      && strlen(rule->term) <= LIBXS_LEXEME_MAXBYTES)
    {
      strcpy(answer_lexnorms[answer_lexnorms_size].from, rule->relation);
      strcpy(answer_lexnorms[answer_lexnorms_size].to, rule->term);
      ++answer_lexnorms_size;
    }
  }
}


static int answer_relation_rule_has_term(int kind, const char* text,
  int text_len)
{
  int result = 0;
  size_t rule_pos;
  if (NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size && 0 == result;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (rule->kind == kind
        && 0 != text_contains_word_ci(text, text_len, rule->term))
      {
        result = 1;
      }
    }
  }
  return result;
}


static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len)
{
  int result = -1;
  size_t rule_pos;
  if (NULL != relation && NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size && result < 0;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (RELATION_RULE_ALIAS == rule->kind
        && 0 != text_contains_word_ci(rule->relation,
          (int)strlen(rule->relation), relation))
      {
        result = text_find_word_ci(text, text_len, rule->term);
        if (result >= 0 && NULL != alias_len) {
          *alias_len = (int)strlen(rule->term);
        }
      }
    }
  }
  return result;
}


/**
 * Resolve the per-corpus namespace from the prefix: state and companion files
 * live in the current directory keyed by the prefix basename, so each corpus
 * owns its own converse.dat/eval/relations/... A prefix of "." (the default)
 * takes the working directory's own name, which reproduces the plain
 * "converse.*" layout when run from the sample directory.
 */
static void converse_namespace_init(const char* prefix)
{
  char base[CONVERSE_PATH_MAX - 16];
  char cwd[CONVERSE_PATH_MAX];
  const char* name = NULL;
  int base_len = 0;
  if (NULL == prefix || '\0' == prefix[0] || 0 == strcmp(prefix, ".")) {
    if (NULL != getcwd(cwd, sizeof(cwd))) {
      const char* slash = strrchr(cwd, '/');
      name = (NULL != slash && '\0' != slash[1]) ? slash + 1 : cwd;
    }
  }
  else {
    const char* slash = strrchr(prefix, '/');
    name = (NULL != slash) ? slash + 1 : prefix;
  }
  if (NULL != name && '\0' != name[0]) {
    base_len = (int)strlen(name);
    if (base_len >= (int)sizeof(base)) base_len = (int)sizeof(base) - 1;
    memcpy(base, name, (size_t)base_len);
    base[base_len] = '\0';
  }
  if (base_len > 0 && 0 != strcmp(base, "converse")) {
    sprintf(converse_path_corpus, "%s.dat", base);
    sprintf(converse_path_lexicon, "%s.lex", base);
    sprintf(converse_path_predict, "%s.prd", base);
    sprintf(converse_path_bridge, "%s.bridges", base);
    sprintf(converse_path_relation, "%s.relations", base);
    sprintf(converse_path_eval, "%s.eval", base);
    sprintf(converse_path_predict_eval, "%s.predict", base);
  }
}


static libxs_registry_t* corpus_load(void)
{
  libxs_registry_t* result = NULL;
  FILE* f = fopen(converse_path_corpus, "rb");
  if (NULL != f) {
    long len;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, f) == len) {
          result = libxs_registry_load(buf, (size_t)len, corpus_fixup, NULL);
        }
        free(buf);
      }
    }
    fclose(f);
  }
  return result;
}


static int corpus_save(const libxs_registry_t* corpus)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == corpus) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_registry_save(corpus, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_registry_save(corpus, buf, &size)) {
        FILE* f = fopen(converse_path_corpus, "wb");
        if (NULL != f) {
          if (fwrite(buf, 1, size, f) == size) result = EXIT_SUCCESS;
          fclose(f);
        }
      }
      free(buf);
    }
  }
  return result;
}


static libxs_lexicon_t* converse_lexicon_load(void)
{
  libxs_lexicon_t* result = NULL;
  FILE* file = fopen(converse_path_lexicon, "rb");
  if (NULL != file) {
    long len;
    fseek(file, 0, SEEK_END);
    len = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, file) == len) {
          result = libxs_lexicon_load(buf, (size_t)len);
        }
        free(buf);
      }
    }
    fclose(file);
  }
  return result;
}


static int converse_lexicon_save(const libxs_lexicon_t* lexicon)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == lexicon) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_lexicon_save(lexicon, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_lexicon_save(lexicon, buf, &size)) {
        FILE* file = fopen(converse_path_lexicon, "wb");
        if (NULL != file) {
          if (fwrite(buf, 1, size, file) == size) result = EXIT_SUCCESS;
          fclose(file);
        }
      }
      free(buf);
    }
  }
  return result;
}


static libxs_predict_t* converse_predict_load(void)
{
  libxs_predict_t* result = NULL;
  FILE* file = fopen(converse_path_predict, "rb");
  if (NULL != file) {
    long len;
    fseek(file, 0, SEEK_END);
    len = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, file) == len) {
          result = libxs_predict_load(buf, (size_t)len);
        }
        free(buf);
      }
    }
    fclose(file);
  }
  return result;
}


static int converse_predict_save(const libxs_predict_t* model)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == model) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_predict_save(model, buf, &size)) {
        FILE* file = fopen(converse_path_predict, "wb");
        if (NULL != file) {
          if (fwrite(buf, 1, size, file) == size) result = EXIT_SUCCESS;
          fclose(file);
        }
      }
      free(buf);
    }
  }
  return result;
}


static int count_words(const unsigned char* text, int length)
{
  int n = 0, i = 0;
  while (i < length) {
    while (i < length && (0 != isspace(text[i]) || 0 != ispunct(text[i]))) ++i;
    if (i < length) {
      ++n;
      while (i < length && 0 == isspace(text[i]) && 0 == ispunct(text[i])) ++i;
    }
  }
  return n;
}


static size_t text_closer_size(const unsigned char* text, size_t size,
  size_t pos)
{
  size_t result = 0;
  if (NULL != text && pos < size) {
    unsigned char ch = text[pos];
    if ('"' == ch || '\'' == ch || ')' == ch || ']' == ch) result = 1;
    else if (pos + 2 < size && 0xe2 == text[pos]
      && 0x80 == text[pos + 1]
      && (0x99 == text[pos + 2] || 0x9d == text[pos + 2]))
    {
      result = 3;
    }
  }
  return result;
}


static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos)
{
  int result = 0;
  if (NULL != text && pos < size
    && ('.' == text[pos] || '?' == text[pos] || '!' == text[pos]))
  {
    size_t next = pos + 1;
    size_t close_size = text_closer_size(text, size, next);
    while (0 != close_size) {
      next += close_size;
      close_size = text_closer_size(text, size, next);
    }
    if (next >= size || 0 != isspace(text[next])) result = 1;
  }
  return result;
}


static int text_starts_sentence(const char* text, int text_len)
{
  int result = 0;
  int pos = 0;
  if (NULL == text || text_len <= 0) return 0;
  while (pos < text_len && 0 != isspace((unsigned char)text[pos])) ++pos;
  if (pos < text_len) {
    const unsigned char* utext = (const unsigned char*)text;
    unsigned char ch = utext[pos];
    if (',' != ch && ';' != ch && ':' != ch && ')' != ch && ']' != ch
      && 0 == (pos + 2 < text_len && 0xe2 == utext[pos]
        && 0x80 == utext[pos + 1]
        && (0x99 == utext[pos + 2] || 0x9d == utext[pos + 2])))
    {
      result = 1;
    }
  }
  return result;
}


static int text_ends_sentence(const char* text, int text_len)
{
  int result = 0;
  int end = text_len;
  if (NULL == text || text_len <= 0) return 0;
  while (end > 0 && 0 != isspace((unsigned char)text[end - 1])) --end;
  while (end > 0) {
    const unsigned char* utext = (const unsigned char*)text;
    unsigned char ch = utext[end - 1];
    if ('"' == ch || '\'' == ch || ')' == ch || ']' == ch) --end;
    else if (end >= 3 && 0xe2 == utext[end - 3]
      && 0x80 == utext[end - 2]
      && (0x99 == utext[end - 1] || 0x9d == utext[end - 1]))
    {
      end -= 3;
    }
    else break;
  }
  if (end > 0) {
    unsigned char ch = (unsigned char)text[end - 1];
    if ('.' == ch || '?' == ch || '!' == ch) result = 1;
  }
  return result;
}


static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = 0;
  libxs_lexeme_stream_t stream;
  size_t lexeme_pos;
  libxs_lexeme_stream_init(&stream);
  if (NULL != lexicon && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, length, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 1))
  {
    for (lexeme_pos = 0; lexeme_pos < stream.size && 0 == result;
      ++lexeme_pos)
    {
      if (0 != (stream.data[lexeme_pos].flags & LIBXS_LEXEME_QUESTION)) {
        result = 1;
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return result;
}


static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id)
{
  int result = 0;
  size_t lexeme_pos;
  if (NULL != stream && 0 != id) {
    for (lexeme_pos = 0; lexeme_pos < stream->size && 0 == result;
      ++lexeme_pos)
    {
      if (stream->data[lexeme_pos].id == id) result = 1;
    }
  }
  return result;
}


static int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id)
{
  int result = 0;
  unsigned short token_pos;
  if (NULL != entry && 0 != id) {
    for (token_pos = 0; token_pos < entry->ntokens && 0 == result;
      ++token_pos)
    {
      if (entry->token_ids[token_pos] == id) result = 1;
    }
  }
  return result;
}


static int lexeme_text_is(const libxs_lexicon_t* lexicon,
  const libxs_lexeme_t* lexeme, const char* text)
{
  int result = 0;
  int length = 0;
  const char* stored;
  if (NULL != lexicon && NULL != lexeme && NULL != text) {
    stored = libxs_lexicon_text(lexicon, lexeme->id, &length, NULL);
    if (NULL != stored && (int)strlen(text) == length
      && 0 == memcmp(stored, text, (size_t)length)) result = 1;
  }
  return result;
}


static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text)
{
  int result = 0;
  size_t lexeme_pos;
  if (NULL != stream && NULL != lexicon && NULL != text) {
    for (lexeme_pos = 0; lexeme_pos < stream->size && 0 == result;
      ++lexeme_pos)
    {
      result = lexeme_text_is(lexicon, stream->data + lexeme_pos, text);
    }
  }
  return result;
}


static int lexeme_stream_has_similar_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  int tolerance)
{
  int result = 0;
  size_t lexeme_pos;
  char lhs[64];
  if (NULL != stream && NULL != lexicon && NULL != text && text_len > 0
    && text_len < (int)sizeof(lhs) && tolerance >= 0)
  {
    memcpy(lhs, text, (size_t)text_len);
    lhs[text_len] = '\0';
    for (lexeme_pos = 0; lexeme_pos < stream->size && 0 == result;
      ++lexeme_pos)
    {
      const libxs_lexeme_t* lexeme = stream->data + lexeme_pos;
      if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
      {
        int rhs_len = 0;
        const char* rhs = libxs_lexicon_text(lexicon, lexeme->id,
          &rhs_len, NULL);
        if (NULL != rhs && rhs_len > 0 && rhs_len < 64) {
          char rhs_buf[64];
          memcpy(rhs_buf, rhs, (size_t)rhs_len);
          rhs_buf[rhs_len] = '\0';
          if (libxs_stridist(lhs, rhs_buf) <= tolerance) result = 1;
        }
      }
    }
  }
  return result;
}


static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon)
{
  int result = QUERY_GENERIC;
  size_t lexeme_pos;
  if (NULL != query && NULL != lexicon) {
    for (lexeme_pos = 0; lexeme_pos < query->size
      && QUERY_GENERIC == result; ++lexeme_pos)
    {
      const libxs_lexeme_t* lexeme = query->data + lexeme_pos;
      if (0 != (lexeme->flags & LIBXS_LEXEME_WORD)) {
        if (0 != lexeme_text_is(lexicon, lexeme, "who")) result = QUERY_WHO;
        else if (0 != lexeme_text_is(lexicon, lexeme, "what")) result = QUERY_WHAT;
        else if (0 != lexeme_text_is(lexicon, lexeme, "where")) result = QUERY_WHERE;
        else if (0 != lexeme_text_is(lexicon, lexeme, "when")) result = QUERY_WHEN;
        else if (0 != lexeme_text_is(lexicon, lexeme, "why")) result = QUERY_WHY;
        else if (0 != lexeme_text_is(lexicon, lexeme, "how")) result = QUERY_HOW;
        else if (0 != lexeme_text_is(lexicon, lexeme, "is")
          || 0 != lexeme_text_is(lexicon, lexeme, "are")
          || 0 != lexeme_text_is(lexicon, lexeme, "was")
          || 0 != lexeme_text_is(lexicon, lexeme, "were")
          || 0 != lexeme_text_is(lexicon, lexeme, "do")
          || 0 != lexeme_text_is(lexicon, lexeme, "does")
          || 0 != lexeme_text_is(lexicon, lexeme, "did")
          || 0 != lexeme_text_is(lexicon, lexeme, "can")
          || 0 != lexeme_text_is(lexicon, lexeme, "could")
          || 0 != lexeme_text_is(lexicon, lexeme, "has")
          || 0 != lexeme_text_is(lexicon, lexeme, "have"))
        {
          result = QUERY_YESNO;
        }
      }
    }
  }
  return result;
}


static int query_type_prefers_sentence(int query_type)
{
  int result = 0;
  if (QUERY_WHO == query_type || QUERY_WHAT == query_type
    || QUERY_WHERE == query_type || QUERY_WHEN == query_type
    || QUERY_YESNO == query_type)
  {
    result = 1;
  }
  return result;
}


/**
 * Destroy word order inside each sentence while keeping the multiset of words
 * (CONVERSE_SHUFFLE=1). A language model must degrade sharply; a model that is
 * really counting unigrams and local collocations will barely notice. Applied
 * at ingest so training and evaluation see the identical transformed text.
 * The permutation is a coprime affine map of the word index -- reproducible,
 * and derived from the sentence length so different sentences permute
 * differently.
 */
static int corpus_shuffle_words(unsigned char* text, int len)
{
  int begin[COMPOSE_MAXTEXT / 2], length[COMPOSE_MAXTEXT / 2];
  unsigned char copy[COMPOSE_MAXTEXT];
  int nwords = 0, pos = 0, out = 0, i;
  size_t stride;
  if (len <= 0 || len > COMPOSE_MAXTEXT) return EXIT_FAILURE;
  while (pos < len && nwords < (int)(sizeof(begin) / sizeof(*begin))) {
    int wlen = 0;
    while (pos + wlen < len && 0 == isspace(text[pos + wlen])) ++wlen;
    if (wlen > 0) {
      begin[nwords] = pos;
      length[nwords] = wlen;
      ++nwords;
    }
    pos += (wlen > 0) ? wlen : 1;
    while (pos < len && 0 != isspace(text[pos])) ++pos;
  }
  if (nwords < 2) return EXIT_FAILURE;
  stride = libxs_coprime_bias((size_t)nwords, -1.0);
  for (i = 0; i < nwords && out < len; ++i) {
    const size_t pick = LIBXS_SHUFFLE_INDEX(i, (size_t)nwords, stride, 1);
    const int w = (int)pick;
    if (out > 0 && out < len) copy[out++] = ' ';
    if (out + length[w] > len) break;
    memcpy(copy + out, text + begin[w], (size_t)length[w]);
    out += length[w];
  }
  if (out > 0) {
    memcpy(text, copy, (size_t)out);
    while (out < len) text[out++] = ' ';
  }
  return (out > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}


static int corpus_shuffle_mode(void)
{
  const char* env = getenv("CONVERSE_SHUFFLE");
  return (NULL != env && '0' != *env) ? 1 : 0;
}


static int corpus_entry_build(corpus_entry_t* entry,
  const unsigned char* text, int len, unsigned char scale,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = EXIT_FAILURE;
  const size_t shape = (size_t)len;
  libxs_lexeme_stream_t stream;
  libxs_fprint_t full;
  unsigned char shuffled[COMPOSE_MAXTEXT];
  size_t lexeme_pos;
  libxs_lexeme_stream_init(&stream);
  if (NULL == entry || NULL == text || len <= 0) return EXIT_FAILURE;
  if (0 != corpus_shuffle_mode() && len <= COMPOSE_MAXTEXT) {
    memcpy(shuffled, text, (size_t)len);
    if (EXIT_SUCCESS == corpus_shuffle_words(shuffled, len)) text = shuffled;
  }
  memset(entry, 0, sizeof(*entry));
  if (EXIT_SUCCESS == libxs_fprint(&full, LIBXS_DATATYPE_U8,
    text, 1, &shape, NULL, FPRINT_ORDER, 0, 0, 0))
  {
    corpus_fprint_pack(&entry->fprint, &full);
    entry->text_len = len;
    memcpy(entry->text, text, (size_t)len);
    entry->connector = CONN_NEWLINE;
    entry->scale = scale;
    entry->source = corpus_source_id;
    result = EXIT_SUCCESS;
  }
  if (EXIT_SUCCESS == result && NULL != lexicon && NULL != rules
    && nrules > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, text, (size_t)len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 1))
  {
    for (lexeme_pos = 0; lexeme_pos < stream.size; ++lexeme_pos) {
      const libxs_lexeme_t* lexeme = stream.data + lexeme_pos;
      if (0 != (lexeme->flags & LIBXS_LEXEME_ENTITY)) {
        entry->lexical_flags |= ENTRY_LEX_ENTITY;
        ++entry->nentities;
      }
      if (0 != (lexeme->flags & LIBXS_LEXEME_NUMBER)) {
        entry->lexical_flags |= ENTRY_LEX_NUMBER;
        ++entry->nnumbers;
      }
      if (0 != (lexeme->flags & LIBXS_LEXEME_QUESTION)) {
        entry->lexical_flags |= ENTRY_LEX_QUESTION;
      }
      if (0 != lexeme_text_is(lexicon, lexeme, "in")
        || 0 != lexeme_text_is(lexicon, lexeme, "at")
        || 0 != lexeme_text_is(lexicon, lexeme, "near")
        || 0 != lexeme_text_is(lexicon, lexeme, "from")
        || 0 != lexeme_text_is(lexicon, lexeme, "inside")
        || 0 != lexeme_text_is(lexicon, lexeme, "outside"))
      {
        entry->lexical_flags |= ENTRY_LEX_PLACE;
      }
      if (0 != lexeme_text_is(lexicon, lexeme, "because")
        || 0 != lexeme_text_is(lexicon, lexeme, "therefore")
        || 0 != lexeme_text_is(lexicon, lexeme, "since")
        || 0 != lexeme_text_is(lexicon, lexeme, "hence")
        || 0 != lexeme_text_is(lexicon, lexeme, "thus")
        || 0 != lexeme_text_is(lexicon, lexeme, "reason")
        || 0 != lexeme_text_is(lexicon, lexeme, "result"))
      {
        entry->lexical_flags |= ENTRY_LEX_CAUSE;
      }
      if (0 != lexeme_text_is(lexicon, lexeme, "by")
        || 0 != lexeme_text_is(lexicon, lexeme, "through")
        || 0 != lexeme_text_is(lexicon, lexeme, "using")
        || 0 != lexeme_text_is(lexicon, lexeme, "with")
        || 0 != lexeme_text_is(lexicon, lexeme, "via")
        || 0 != lexeme_text_is(lexicon, lexeme, "method")
        || 0 != lexeme_text_is(lexicon, lexeme, "process"))
      {
        entry->lexical_flags |= ENTRY_LEX_METHOD;
      }
      if (entry->ntokens < ENTRY_TOKEN_MAX
        && 0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 == (lexeme->flags & LIBXS_LEXEME_STOP)
        && 0 == entry_sketch_has_id(entry, lexeme->id))
      {
        entry->token_ids[entry->ntokens] = lexeme->id;
        entry->token_flags[entry->ntokens] = lexeme->flags;
        ++entry->ntokens;
        ++entry->ncontent;
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return result;
}


static void corpus_entry_set_section(corpus_entry_t* entry,
  const char* section, int section_len)
{
  int copy_len;
  if (NULL == entry || NULL == section || section_len <= 0) return;
  copy_len = section_len;
  if (copy_len >= ENTRY_SECTION_MAX) copy_len = ENTRY_SECTION_MAX - 1;
  memcpy(entry->section, section, (size_t)copy_len);
  entry->section[copy_len] = '\0';
  entry->section_len = (unsigned short)copy_len;
}


static int corpus_title_prefix(const unsigned char* text, int len,
  char* title, int title_size)
{
  int result = 0;
  int pos, prefix_words = 0, in_word = 0;
  int title_begin = 0, title_end;
  if (NULL == text || NULL == title || title_size <= 0 || len <= 0) return 0;
  title[0] = '\0';
  /**
   * A heading starts with a letter. Requiring that rejects all-caps text that
   * merely opens with punctuation -- a quoted letter signature in dialogue, for
   * instance -- which otherwise looks exactly like a title to this prober.
   */
  if (0 == isalpha(text[0])) return 0;
  for (pos = 0; pos < len; ++pos) {
    unsigned char ch = text[pos];
    if (0 != islower(ch) || '.' == ch || ',' == ch || ';' == ch
      || ':' == ch || '!' == ch || '?' == ch)
    {
      break;
    }
    if (0 != isupper(ch) && pos + 1 < len && 0 != islower(text[pos + 1])) {
      break;
    }
    if (0 != isupper(ch)) {
      if (0 == in_word) {
        ++prefix_words;
        in_word = 1;
      }
    }
    else if (0 != isspace(ch)) {
      in_word = 0;
    }
  }
  title_end = pos;
  if (prefix_words >= 2 && pos < len && pos > 0) {
    int prev_end = pos;
    int prev_start;
    while (prev_end > 0 && 0 != isspace(text[prev_end - 1])) --prev_end;
    prev_start = prev_end;
    while (prev_start > 0 && 0 == isspace(text[prev_start - 1])) {
      --prev_start;
    }
    if ((1 == prev_end - prev_start && 'A' == text[prev_start])
      || (2 == prev_end - prev_start && 'A' == text[prev_start]
        && 'N' == text[prev_start + 1]))
    {
      title_end = prev_start;
    }
    while (title_begin < title_end && 0 != isspace(text[title_begin])) {
      ++title_begin;
    }
    while (title_end > title_begin && 0 != isspace(text[title_end - 1])) {
      --title_end;
    }
    result = title_end - title_begin;
    if (result >= title_size) result = title_size - 1;
    if (result > 0) {
      memcpy(title, text + title_begin, (size_t)result);
      title[result] = '\0';
    }
  }
  return result;
}


static int corpus_entry_same_section(const corpus_entry_t* lhs,
  size_t lhs_size, const corpus_entry_t* rhs)
{
  int result = 0;
  if (NULL != lhs && NULL != rhs) {
    if (lhs_size < sizeof(*lhs)) result = (0 == rhs->section_len) ? 1 : 0;
    else if (lhs->section_len == rhs->section_len
      && 0 == libxs_memcmp(lhs->section, rhs->section, lhs->section_len))
    {
      result = 1;
    }
  }
  return result;
}


/**
 * Upper bound on the fingerprint-collision probe chain. 65536 was effectively
 * unbounded and made ingest quadratic on collision-heavy corpora.
 */
static unsigned int corpus_chain_max(void)
{
  unsigned int result = 65536;
  const char* env = getenv("CONVERSE_CHAIN_MAX");
  if (NULL != env && '\0' != *env) {
    const long v = atol(env);
    if (v >= 1 && v <= 65536) result = (unsigned int)v;
  }
  return result;
}


/**
 * Registry key = IDENTITY of the entry's content, not its fingerprint. The two
 * jobs were conflated: a Hilbert code of the fingerprint is locality-preserving
 * ON PURPOSE, so similar texts share a cell, and using it as a unique key forced
 * a linear probe over every colliding entry with a full text compare per step.
 * On collision-heavy corpora (wiki markup, code) that made ingest quadratic --
 * 2x the enwik8 text cost 10.8x the ingest CPU.
 *
 * A content hash collides only by accident, so insertion and duplicate detection
 * are one lookup. The similarity index gets its codes from the entries instead
 * (libxs_spatial_build_codes), which is where locality belongs.
 *
 * The 128-bit key (two independent hashes plus the length) makes an accidental
 * collision between DIFFERENT texts negligible. One is still resolved rather
 * than dropped: the walk appends a seq only from the second probe on, and the
 * registry supports mixed key sizes, so the common case stays 16 bytes and no
 * entry is lost to a genuine hash collision.
 */
static void corpus_key_from_text(const corpus_entry_t* entry,
  unsigned char key[], size_t* key_size)
{
  const unsigned int len = (unsigned int)entry->text_len;
  const unsigned int h1 = libxs_hash(entry->text, len, 0x9e3779b9u);
  const unsigned int h2 = libxs_hash(entry->text, len, 0x85ebca6bu);
  const unsigned int hs = (0 < entry->section_len)
    ? libxs_hash(entry->section, (unsigned int)entry->section_len, 0xc2b2ae35u)
    : 0u;
  memcpy(key, &h1, 4);
  memcpy(key + 4, &h2, 4);
  memcpy(key + 8, &hs, 4);
  memcpy(key + 12, &len, 4);
  *key_size = 16;
}


/**
 * Build the similarity index from the entries' own fingerprints. The registry key
 * is now a content hash, so libxs_spatial_build -- which reads the first 8 Bytes
 * of each key as the code -- would index hash values and destroy locality. The
 * code is recomputed here from the fingerprint, which is where it belongs.
 *
 * Iteration uses the _length flavor because this registry deliberately holds keys
 * of two sizes (16 Bytes, or 18 when a hash collision needed resolving); the
 * plain iterator cannot report which, and the Bytes beyond key_size are
 * undefined.
 */
static int corpus_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* corpus)
{
  int result = EXIT_FAILURE;
  libxs_registry_info_t info;
  if (NULL == sp || NULL == corpus) return EXIT_FAILURE;
  libxs_registry_info(corpus, &info);
  if (0 < info.size) {
    uint64_t* codes = (uint64_t*)libxs_malloc(NULL,
      info.size * sizeof(uint64_t), LIBXS_MALLOC_AUTO);
    void** values = (void**)libxs_malloc(NULL,
      info.size * sizeof(void*), LIBXS_MALLOC_AUTO);
    if (NULL != codes && NULL != values) {
      const void* key = NULL;
      size_t key_size = 0, cursor = 0;
      int count = 0;
      void* value = libxs_registry_begin_length(corpus, &key, &key_size,
        &cursor);
      while (NULL != value && count < (int)info.size) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        unsigned char code_key[16];
        size_t code_size = 0;
        uint64_t code = 0;
        corpus_key_from_fprint(&entry->fprint, code_key, &code_size);
        memcpy(&code, code_key, 8);
        codes[count] = code;
        values[count] = value;
        ++count;
        value = libxs_registry_next_length(corpus, &key, &key_size, &cursor);
      }
      result = libxs_spatial_build_codes(sp, codes, values, count);
    }
    libxs_free(codes);
    libxs_free(values);
  }
  return result;
}


static int corpus_store_entry(libxs_registry_t* corpus,
  const corpus_entry_t* entry)
{
  int result = 0;
  unsigned char key[20];
  size_t key_size = 0;
  unsigned int seq;
  int matched = 0;
  if (NULL == corpus || NULL == entry) return 0;
  corpus_key_from_text(entry, key, &key_size);
  /**
   * The sequence walk is a linear probe over every entry sharing a quantized
   * fingerprint, and it re-reads each one to compare text. On prose the chains
   * are short, but the key quantizes to COMPOSE_NDIMS x COMPOSE_BITS, so a
   * corpus of many near-identical short texts (wiki markup, code) collides
   * heavily and ingest becomes quadratic: 2x the enwik8 text cost 10.8x the
   * ingest CPU. The cap bounds the walk so ingest stays linear; a text beyond it
   * is dropped rather than stored, which is reported so a silent truncation
   * cannot be mistaken for a rare fingerprint (the same discipline as the
   * recomb postings cap).
   */
  for (seq = 0; seq < corpus_chain_max(); ++seq) {
    void* existing;
    if (0 < seq) {
      const unsigned short seq_key = (unsigned short)seq;
      memcpy(key + 16, &seq_key, 2);
      key_size = 18;
    }
    existing = libxs_registry_get(corpus, key, key_size, NULL);
    if (NULL == existing) {
      if (0 == matched) {
        libxs_registry_set(corpus, key, key_size,
          entry, corpus_entry_size(entry), NULL);
        result = 1;
      }
      break;
    }
    else {
      const corpus_entry_t* old_entry = (const corpus_entry_t*)existing;
      size_t old_size = libxs_registry_value_size(corpus, key,
        key_size, NULL);
      if (old_entry->text_len == entry->text_len
        && 0 == libxs_memcmp(old_entry->text, entry->text,
          (size_t)entry->text_len)
        && 0 != corpus_entry_same_section(old_entry, old_size, entry))
      {
        matched = 1;
        /* Replace only to gain token metadata; sizes now vary by text
           length, so an unequal size is no longer evidence of anything. */
        if (0 == old_entry->ntokens && entry->ntokens > 0) {
          libxs_registry_set(corpus, key, key_size,
            entry, corpus_entry_size(entry), NULL);
        }
        if (old_entry->ntokens > 0) break;
      }
    }
  }
  /* Chain exhausted without placing or matching: the text is dropped. */
  if (0 == result && 0 == matched) ++corpus_chain_dropped;
  return result;
}


static int answer_query_section(const char* query_text, size_t query_len,
  char* title, int title_size)
{
  int result = 0;
  size_t pos = 0, begin, end, comma_pos;
  int marker_pos;
  int end_pos;
  if (NULL == query_text || NULL == title || title_size <= 0) return 0;
  title[0] = '\0';
  while (pos < query_len && 0 != isspace((unsigned char)query_text[pos])) {
    ++pos;
  }
  if (pos + 3 < query_len
    && 'i' == tolower((unsigned char)query_text[pos])
    && 'n' == tolower((unsigned char)query_text[pos + 1])
    && 0 != isspace((unsigned char)query_text[pos + 2]))
  {
    begin = pos + 3;
    while (begin < query_len
      && 0 != isspace((unsigned char)query_text[begin])) ++begin;
    end = begin;
    while (end < query_len && ',' != query_text[end]
      && '?' != query_text[end] && '!' != query_text[end]) ++end;
    comma_pos = end;
    while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) {
      --end;
    }
    result = (int)(end - begin);
    if (result >= title_size) result = title_size - 1;
    if (result > 0 && comma_pos < query_len && ',' == query_text[comma_pos]) {
      memcpy(title, query_text + begin, (size_t)result);
      title[result] = '\0';
    }
    else result = 0;
  }
  if (0 == result) {
    marker_pos = text_find_ci(query_text, (int)query_len, " in ");
    if (marker_pos >= 0) {
      begin = (size_t)marker_pos + 4;
      while (begin < query_len
        && 0 != isspace((unsigned char)query_text[begin])) ++begin;
      end = query_len;
      end_pos = text_find_ci(query_text + begin, (int)(query_len - begin),
        " is ");
      if (end_pos < 0) end_pos = text_find_ci(query_text + begin,
        (int)(query_len - begin), " are ");
      if (end_pos < 0) end_pos = text_find_ci(query_text + begin,
        (int)(query_len - begin), " was ");
      if (end_pos < 0) end_pos = text_find_ci(query_text + begin,
        (int)(query_len - begin), " were ");
      if (end_pos >= 0) end = begin + (size_t)end_pos;
      else {
        while (end > begin && ('?' == query_text[end - 1]
          || '!' == query_text[end - 1] || '.' == query_text[end - 1]
          || ',' == query_text[end - 1]
          || 0 != isspace((unsigned char)query_text[end - 1]))) --end;
      }
      while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) {
        --end;
      }
      result = (int)(end - begin);
      if (result >= title_size) result = title_size - 1;
      if (result > 0) {
        memcpy(title, query_text + begin, (size_t)result);
        title[result] = '\0';
      }
    }
  }
  return result;
}


static int corpus_entry_section_match(const corpus_entry_t* entry,
  size_t entry_size, const char* title, int title_len)
{
  int result = 0;
  int entry_pos = 0, title_pos = 0, entry_len = 0;
  if (NULL == title || title_len <= 0) return 1;
  if (NULL != entry && entry_size >= CORPUS_ENTRY_META_SIZE
    && '\0' != entry->section[0])
  {
    while (entry_len < ENTRY_SECTION_MAX && '\0' != entry->section[entry_len]) {
      ++entry_len;
    }
    while (entry_pos < entry_len && title_pos < title_len) {
      while (entry_pos < entry_len
        && 0 == isalnum((unsigned char)entry->section[entry_pos]))
      {
        ++entry_pos;
      }
      while (title_pos < title_len
        && 0 == isalnum((unsigned char)title[title_pos]))
      {
        ++title_pos;
      }
      if (entry_pos < entry_len && title_pos < title_len) {
        if (tolower((unsigned char)entry->section[entry_pos])
          != tolower((unsigned char)title[title_pos]))
        {
          break;
        }
        ++entry_pos;
        ++title_pos;
      }
    }
    while (entry_pos < entry_len
      && 0 == isalnum((unsigned char)entry->section[entry_pos])) ++entry_pos;
    while (title_pos < title_len
      && 0 == isalnum((unsigned char)title[title_pos])) ++title_pos;
    if (title_pos == title_len) {
      result = 1;
    }
    else if (title_len < ENTRY_SECTION_MAX) {
      char title_buf[ENTRY_SECTION_MAX];
      memcpy(title_buf, title, (size_t)title_len);
      title_buf[title_len] = '\0';
      if (0 == libxs_stridiff(entry->section, title_buf, NULL, 1, NULL)) {
        result = 1;
      }
    }
  }
  return result;
}


static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size, int* upper_initial)
{
  static const char* const markers[] = {
    " is ", " are ", " was ", " were "
  };
  int result = 0;
  int marker_pos = -1;
  int marker_len = 0;
  int marker_index;
  size_t begin;
  size_t end;
  if (NULL == query_text || NULL == word || word_size <= 0) return 0;
  word[0] = '\0';
  if (NULL != upper_initial) *upper_initial = 0;
  for (marker_index = 0; marker_index < 4 && marker_pos < 0;
    ++marker_index)
  {
    marker_pos = text_find_ci(query_text, (int)query_len,
      markers[marker_index]);
    if (marker_pos >= 0) marker_len = (int)strlen(markers[marker_index]);
  }
  if (marker_pos >= 0) {
    begin = (size_t)marker_pos + (size_t)marker_len;
    do {
      while (begin < query_len
        && 0 != isspace((unsigned char)query_text[begin])) ++begin;
      end = begin;
      while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
        ++end;
      }
      result = (int)(end - begin);
      if (result > 0 && 0 != answer_relation_rule_has_term(RELATION_RULE_SKIP,
        query_text + begin, result))
      {
        begin = end;
        result = 0;
      }
    } while (0 == result && end < query_len);
    if (result >= word_size) result = word_size - 1;
    if (result > 0) {
      memcpy(word, query_text + begin, (size_t)result);
      word[result] = '\0';
      if (NULL != upper_initial) {
        *upper_initial = isupper((unsigned char)query_text[begin]) ? 1 : 0;
      }
    }
  }
  return result;
}


static int answer_query_relation_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size)
{
  int result = 0;
  int marker_pos;
  size_t begin, end;
  if (NULL == query_text || NULL == actor || actor_size <= 0) return 0;
  actor[0] = '\0';
  marker_pos = text_find_ci(query_text, (int)query_len, " by ");
  if (marker_pos >= 0) {
    begin = (size_t)marker_pos + 4;
    while (begin < query_len
      && 0 != isspace((unsigned char)query_text[begin])) ++begin;
    end = begin;
    while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
      ++end;
    }
    while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
      ++end;
    }
    if ((3 == end - begin && 0 == strncmp(query_text + begin, "the", 3))
      || (1 == end - begin && 'a' == tolower((unsigned char)query_text[begin]))
      || (2 == end - begin && 0 == strncmp(query_text + begin, "an", 2)))
    {
      while (end < query_len
        && 0 != isspace((unsigned char)query_text[end])) ++end;
      while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
        ++end;
      }
    }
    result = (int)(end - begin);
    if (result >= actor_size) result = actor_size - 1;
    if (result > 0) {
      memcpy(actor, query_text + begin, (size_t)result);
      actor[result] = '\0';
    }
  }
  return result;
}


static int answer_relation_copy_name(char* output, int output_size,
  const char* text, int text_len, int begin, int end)
{
  int result;
  while (begin < end && 0 == isalnum((unsigned char)text[begin])) ++begin;
  while (end > begin && 0 == isalnum((unsigned char)text[end - 1])) --end;
  result = end - begin;
  if (result >= output_size) result = output_size - 1;
  if (result > 0) {
    memcpy(output, text + begin, (size_t)result);
    output[result] = '\0';
    output[0] = (char)toupper((unsigned char)output[0]);
  }
  return result;
}


static int answer_relation_actor_has_token(const char* actor, int actor_len,
  const char* token, int token_len)
{
  int result = 0;
  char token_buf[64];
  if (NULL != actor && actor_len > 0 && NULL != token && token_len > 0
    && token_len < (int)sizeof(token_buf))
  {
    memcpy(token_buf, token, (size_t)token_len);
    token_buf[token_len] = '\0';
    result = text_contains_ci(actor, actor_len, token_buf);
  }
  return result;
}


static int answer_relation_copy_section_head(char* output, int output_size,
  const corpus_entry_t* entry)
{
  int result = 0;
  int end = 0;
  if (NULL == output || output_size <= 0 || NULL == entry
    || entry->section_len <= 0) return 0;
  while (end < entry->section_len
    && 0 != isalnum((unsigned char)entry->section[end])) ++end;
  result = answer_relation_copy_name(output, output_size, entry->section,
    entry->section_len, 0, end);
  if (3 == result && 0 != text_contains_ci(output, result, "the")) result = 0;
  if (1 == result && 0 != text_contains_ci(output, result, "a")) result = 0;
  if (2 == result && 0 != text_contains_ci(output, result, "an")) result = 0;
  if (result > 1) {
    int pos;
    for (pos = 1; pos < result; ++pos) {
      output[pos] = (char)tolower((unsigned char)output[pos]);
    }
  }
  return result;
}


static int answer_relation_copy_antecedent(char* output, int output_size,
  const char* text, int text_len, int cue_pos)
{
  int result = 0;
  int scan = 0;
  int first_word = 1;
  if (NULL != output && output_size > 0 && NULL != text && text_len > 0
    && cue_pos > 0)
  {
    output[0] = '\0';
    while (scan < cue_pos && scan < text_len) {
      int begin;
      int end;
      while (scan < cue_pos && scan < text_len
        && 0 == isalnum((unsigned char)text[scan])) ++scan;
      begin = scan;
      while (scan < cue_pos && scan < text_len
        && ('-' == text[scan] || 0 != isalnum((unsigned char)text[scan]))) {
        ++scan;
      }
      end = scan;
      if (end > begin && 0 != isupper((unsigned char)text[begin])
        && 0 == first_word
        && 0 == text_contains_ci(text + begin, end - begin, "Then")
        && 0 == text_contains_ci(text + begin, end - begin, "When")
        && 0 == text_contains_ci(text + begin, end - begin, "The")
        && 0 == text_contains_ci(text + begin, end - begin, "She")
        && 0 == text_contains_ci(text + begin, end - begin, "He"))
      {
        result = answer_relation_copy_name(output, output_size, text,
          text_len, begin, end);
        break;
      }
      if (end > begin) first_word = 0;
    }
  }
  return result;
}


static int answer_relation_find_person_before(const char* text, int text_len,
  int limit, int* term_pos, int* term_len)
{
  int result = 0;
  int best_pos = -1;
  int best_len = 0;
  size_t rule_pos;
  if (NULL != text && text_len > 0 && limit > 0)
  {
    if (limit > text_len) limit = text_len;
    for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      int rule_len = (int)strlen(rule->term);
      int pos = 0;
      if (RELATION_RULE_PERSON == rule->kind && rule_len > 0) {
        while (pos < limit) {
          int found = text_find_word_ci(text + pos, limit - pos, rule->term);
          if (found < 0) break;
          found += pos;
          if (found > best_pos) {
            best_pos = found;
            best_len = rule_len;
          }
          pos = found + rule_len;
        }
      }
    }
    if (best_pos >= 0 && best_len > 0) {
      if (NULL != term_pos) *term_pos = best_pos;
      if (NULL != term_len) *term_len = best_len;
      result = 1;
    }
  }
  return result;
}


static int answer_relation_copy_person_before(char* output, int output_size,
  const char* text, int text_len, int limit)
{
  int result = 0;
  int term_pos = -1;
  int term_len = 0;
  if (NULL != output && output_size > 0
    && 0 != answer_relation_find_person_before(text, text_len, limit,
      &term_pos, &term_len))
  {
      result = answer_relation_copy_name(output, output_size, text,
        text_len, term_pos, term_pos + term_len);
  }
  return result;
}


static int answer_relation_match_query(const char* query_text,
  size_t query_len, int query_type, const corpus_entry_t* entry,
  answer_relation_match_t* match)
{
  int result = 0;
  char relation[64];
  char actor[64];
  int relation_upper = 0;
  int relation_len;
  int actor_len;
  if (NULL == query_text || NULL == entry || NULL == match
    || QUERY_WHO != query_type) return 0;
  memset(match, 0, sizeof(*match));
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation), &relation_upper);
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  if (relation_len <= 0 || 0 != relation_upper) return 0;
  memcpy(match->relation, relation, (size_t)relation_len + 1);
  match->relation_len = relation_len;
  if (actor_len > 0) {
    memcpy(match->actor, actor, (size_t)actor_len + 1);
    match->actor_len = actor_len;
  }
  if (actor_len > 0)
  {
    int rel_pos = text_find_word_ci(entry->text, entry->text_len, relation);
    int alt_pos = -1;
    int alt_len = 0;
    int verb_pos;
    int verb_len;
    int actor_seen = text_contains_ci(entry->text, entry->text_len, actor);
    if (rel_pos < 0) alt_pos = answer_relation_rule_alias_pos(relation,
      entry->text, entry->text_len, &alt_len);
    verb_pos = (rel_pos >= 0) ? rel_pos : alt_pos;
    verb_len = (rel_pos >= 0) ? relation_len : alt_len;
    if (0 == actor_seen && verb_pos > 0
      && 0 != text_contains_word_ci(entry->text, verb_pos, "he"))
    {
      actor_seen = 1;
    }
    if (verb_pos >= 0 && 0 != actor_seen) {
      int obj_begin = verb_pos + ((verb_len > 0) ? verb_len : 1);
      int obj_end;
      if (verb_len <= 0) {
        while (obj_begin < entry->text_len
          && 0 != isalnum((unsigned char)entry->text[obj_begin])) ++obj_begin;
      }
      while (obj_begin < entry->text_len
        && 0 == isalnum((unsigned char)entry->text[obj_begin])) ++obj_begin;
      if (obj_begin + 2 < entry->text_len
        && 0 == strncmp(entry->text + obj_begin, "up", 2)
        && 0 == isalnum((unsigned char)entry->text[obj_begin + 2]))
      {
        obj_begin += 2;
        while (obj_begin < entry->text_len
          && 0 == isalnum((unsigned char)entry->text[obj_begin])) ++obj_begin;
      }
      obj_end = obj_begin;
      while (obj_end < entry->text_len
        && ('-' == entry->text[obj_end]
          || 0 != isalnum((unsigned char)entry->text[obj_end]))) ++obj_end;
      if ((3 == obj_end - obj_begin
          && 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "the"))
        || (1 == obj_end - obj_begin
          && 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "a"))
        || (2 == obj_end - obj_begin
          && 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "an")))
      {
        while (obj_end < entry->text_len
          && 0 == isalnum((unsigned char)entry->text[obj_end])) ++obj_end;
        obj_begin = obj_end;
        while (obj_end < entry->text_len
          && ('-' == entry->text[obj_end]
            || 0 != isalnum((unsigned char)entry->text[obj_end]))) ++obj_end;
      }
      if (obj_end > obj_begin
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "he")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "she")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "her")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "him")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "them"))
      {
        if (0 == answer_relation_actor_has_token(actor, actor_len,
          entry->text + obj_begin, obj_end - obj_begin))
        {
          match->answer_len = answer_relation_copy_name(match->answer,
            (int)sizeof(match->answer), entry->text, entry->text_len,
            obj_begin, obj_end);
        }
      }
      else if (obj_end > obj_begin
        && (0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "her")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "him")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "he")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "she")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "them")))
      {
        int scan = verb_pos - 1;
        while (scan > 0 && match->answer_len <= 0) {
          int end = scan + 1;
          int begin = scan;
          while (begin > 0
            && 0 != isalnum((unsigned char)entry->text[begin - 1])) --begin;
          if (end > begin
            && end - begin > 1
            && 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
              entry->text + begin, end - begin)
            && 0 == text_contains_ci(entry->text + begin, end - begin,
              actor)
            && 0 == answer_relation_actor_has_token(actor, actor_len,
              entry->text + begin, end - begin)
            && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
              entry->text + begin, end - begin))
          {
            match->answer_len = answer_relation_copy_name(match->answer,
              (int)sizeof(match->answer), entry->text, entry->text_len,
              begin, end);
          }
          scan = begin - 1;
        }
        if (match->answer_len <= 0) {
          match->answer_len = answer_relation_copy_person_before(
            match->answer, (int)sizeof(match->answer), entry->text,
            entry->text_len, verb_pos);
        }
      }
      if (match->answer_len > 0) {
        if (0 != answer_relation_actor_has_token(actor, actor_len,
          match->answer, match->answer_len))
        {
          match->answer_len = 0;
        }
      }
      if (match->answer_len > 0) {
        match->score = 1.65;
        result = 1;
      }
    }
  }
  else if (actor_len <= 0 && 0 != text_contains_word_ci(entry->text,
    entry->text_len, relation))
  {
    int rel_pos = text_find_word_ci(entry->text, entry->text_len, relation);
    int made_pos = -1;
    int made_scan = 0;
    int before = rel_pos;
    int answer_begin, answer_end;
    while (rel_pos > made_scan) {
      int next_pos = text_find_word_ci(entry->text + made_scan,
        rel_pos - made_scan, "made");
      if (next_pos < 0) break;
      made_pos = made_scan + next_pos;
      made_scan = made_pos + 4;
    }
    if (rel_pos >= 0 && made_pos >= 0 && made_pos < rel_pos
      && rel_pos - made_pos <= 16)
    {
      if (made_pos >= 0 && made_pos < rel_pos) {
        match->made = 1;
        {
          int cue_pos = -1;
          int cue_len = 0;
          int possessive = 0;
          match->answer_len = answer_relation_copy_section_head(match->answer,
            (int)sizeof(match->answer), entry);
          if (0 != answer_relation_find_person_before(entry->text,
            entry->text_len, made_pos, &cue_pos, &cue_len))
          {
            int prev_end = cue_pos;
            int prev_begin;
            while (prev_end > 0
              && 0 == isalnum((unsigned char)entry->text[prev_end - 1])) {
              --prev_end;
            }
            prev_begin = prev_end;
            while (prev_begin > 0
              && 0 != isalnum((unsigned char)entry->text[prev_begin - 1])) {
              --prev_begin;
            }
            if (prev_end > prev_begin
              && 0 != answer_relation_rule_has_term(RELATION_RULE_SKIP,
                entry->text + prev_begin, prev_end - prev_begin))
            {
              possessive = 1;
            }
          }
          if (match->answer_len <= 0 && 0 == possessive && cue_pos >= 0) {
            match->answer_len = answer_relation_copy_antecedent(match->answer,
              (int)sizeof(match->answer), entry->text, entry->text_len,
              cue_pos);
          }
          if (match->answer_len <= 0 && 0 == possessive && cue_pos >= 0) {
            match->answer_len = answer_relation_copy_name(match->answer,
              (int)sizeof(match->answer), entry->text, entry->text_len,
              cue_pos, cue_pos + cue_len);
          }
        }
      }
      if (match->answer_len <= 0 && 0 == match->made) {
        while (before > 0
          && 0 == isalnum((unsigned char)entry->text[before - 1])) --before;
        answer_end = before;
        while (before > 0
          && 0 != isalnum((unsigned char)entry->text[before - 1])) --before;
        answer_begin = before;
        if (answer_end - answer_begin > 1
          && 0 == text_contains_ci(entry->text + answer_begin,
            answer_end - answer_begin, "made")
          && 0 == text_contains_ci(entry->text + answer_begin,
            answer_end - answer_begin, "too")
          && 0 == text_contains_ci(entry->text + answer_begin,
            answer_end - answer_begin, "the"))
        {
          match->answer_len = answer_relation_copy_name(match->answer,
            (int)sizeof(match->answer), entry->text, entry->text_len,
            answer_begin, answer_end);
        }
      }
      match->score = (match->answer_len > 0)
        ? ((0 != match->made) ? 1.85 : 1.25) : 0.0;
      result = (match->answer_len > 0) ? 1 : 0;
    }
  }
  return result;
}


static int answer_relation_reply(const answer_relation_match_t* match,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  size_t pos = 0;
  const char* copula;
  if (NULL == match || NULL == output || 0 == output_size
    || match->answer_len <= 0 || match->relation_len <= 0) return EXIT_FAILURE;
  copula = (0 != match->plural) ? " were " : " was ";
  if ((size_t)match->answer_len + strlen(copula)
    + (size_t)match->relation_len + (size_t)match->actor_len + 8
    < output_size)
  {
    memcpy(output + pos, match->answer, (size_t)match->answer_len);
    pos += (size_t)match->answer_len;
    if (0 != match->made) {
      static const char made_text[] = " is to be made ";
      memcpy(output + pos, made_text, sizeof(made_text) - 1);
      pos += sizeof(made_text) - 1;
    }
    else {
      memcpy(output + pos, copula, strlen(copula));
      pos += strlen(copula);
    }
    memcpy(output + pos, match->relation, (size_t)match->relation_len);
    pos += (size_t)match->relation_len;
    if (match->actor_len > 0) {
      static const char by_text[] = " by ";
      memcpy(output + pos, by_text, sizeof(by_text) - 1);
      pos += sizeof(by_text) - 1;
      memcpy(output + pos, match->actor, (size_t)match->actor_len);
      pos += (size_t)match->actor_len;
    }
    output[pos++] = '.';
    output[pos] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static int answer_relation_section_title(char* output, int output_size,
  const corpus_entry_t* entry, const answer_relation_match_t* match)
{
  int result = 0;
  if (NULL != output && output_size > 0 && NULL != entry && NULL != match
    && entry->section_len > match->answer_len && match->answer_len > 0
    && entry->section_len < output_size
    && 0 != text_contains_ci(entry->section, entry->section_len,
      match->answer))
  {
    int pos;
    int new_word = 1;
    int title_len = entry->section_len;
    for (pos = 0; pos < title_len; ++pos) {
      if ('[' == entry->section[pos] || '(' == entry->section[pos]) {
        title_len = pos;
      }
    }
    while (title_len > 0
      && 0 != isspace((unsigned char)entry->section[title_len - 1])) {
      --title_len;
    }
    memcpy(output, entry->section, (size_t)title_len);
    output[title_len] = '\0';
    for (pos = 0; pos < title_len; ++pos) {
      if (0 != isalnum((unsigned char)output[pos])) {
        output[pos] = (char)(0 != new_word
          ? toupper((unsigned char)output[pos])
          : tolower((unsigned char)output[pos]));
        new_word = 0;
      }
      else new_word = 1;
    }
    result = title_len;
  }
  return result;
}


static int answer_relation_same_answer(const char* lhs, int lhs_len,
  const char* rhs, int rhs_len)
{
  int result = 0;
  if (NULL != lhs && NULL != rhs && lhs_len > 0 && rhs_len > 0) {
    if (lhs_len == rhs_len && 0 != text_contains_ci(lhs, lhs_len, rhs)) {
      result = 1;
    }
    else if (lhs_len > rhs_len && 0 != text_contains_ci(lhs, lhs_len, rhs)) {
      result = 1;
    }
    else if (rhs_len > lhs_len && 0 != text_contains_ci(rhs, rhs_len, lhs)) {
      result = 1;
    }
  }
  return result;
}


static void answer_relation_facts_free(void)
{
  free(answer_relation_facts);
  answer_relation_facts = NULL;
  answer_relation_facts_size = 0;
}


static int answer_relation_fact_append(const corpus_entry_t* entry,
  const answer_relation_match_t* match)
{
  int result = EXIT_FAILURE;
  answer_relation_fact_t fact;
  answer_relation_fact_t* facts;
  size_t fact_pos;
  if (NULL == entry || NULL == match || match->answer_len <= 0
    || match->relation_len <= 0) return EXIT_FAILURE;
  LIBXS_MEMZERO(&fact);
  fact.answer_len = 0;
  if (match->actor_len > 0) {
    fact.answer_len = answer_relation_section_title(fact.answer,
      (int)sizeof(fact.answer), entry, match);
  }
  if (fact.answer_len <= 0) {
    fact.answer_len = match->answer_len;
    memcpy(fact.answer, match->answer, (size_t)fact.answer_len + 1);
  }
  fact.relation_len = match->relation_len;
  memcpy(fact.relation, match->relation, (size_t)fact.relation_len + 1);
  fact.actor_len = match->actor_len;
  if (fact.actor_len > 0) {
    memcpy(fact.actor, match->actor, (size_t)fact.actor_len + 1);
  }
  fact.section_len = entry->section_len;
  if (fact.section_len > 0) {
    memcpy(fact.section, entry->section, (size_t)fact.section_len);
    fact.section[fact.section_len] = '\0';
  }
  fact.plural = match->plural;
  fact.made = match->made;
  fact.score = match->score;
  for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
    answer_relation_fact_t* old_fact = answer_relation_facts + fact_pos;
    if (old_fact->relation_len == fact.relation_len
      && old_fact->actor_len == fact.actor_len
      && old_fact->section_len == fact.section_len
      && 0 != answer_relation_same_answer(old_fact->answer,
        old_fact->answer_len, fact.answer, fact.answer_len)
      && 0 != text_contains_ci(old_fact->relation, old_fact->relation_len,
        fact.relation)
      && (0 == fact.actor_len
        || 0 != text_contains_ci(old_fact->actor, old_fact->actor_len,
          fact.actor))
      && (0 == fact.section_len
        || 0 != text_contains_ci(old_fact->section, old_fact->section_len,
          fact.section)))
    {
      if (fact.score > old_fact->score) *old_fact = fact;
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_relation_fact_t*)realloc(answer_relation_facts,
      (answer_relation_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_relation_facts = facts;
      answer_relation_facts[answer_relation_facts_size] = fact;
      ++answer_relation_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_relation_fact_extract_actor(const char* text, int text_len,
  int verb_pos, int* scan_pos, char* actor, int actor_size)
{
  int result = 0;
  int begin, end;
  if (NULL == text || NULL == scan_pos || NULL == actor || actor_size <= 0) {
    return 0;
  }
  actor[0] = '\0';
  while (*scan_pos < verb_pos) {
    while (*scan_pos < verb_pos
      && 0 == isalnum((unsigned char)text[*scan_pos])) ++*scan_pos;
    begin = *scan_pos;
    while (*scan_pos < verb_pos
      && ('-' == text[*scan_pos]
        || 0 != isalnum((unsigned char)text[*scan_pos]))) ++*scan_pos;
    end = *scan_pos;
    if (end > begin && end - begin < actor_size
      && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
        text + begin, end - begin))
    {
      memcpy(actor, text + begin, (size_t)(end - begin));
      actor[end - begin] = '\0';
      result = end - begin;
      break;
    }
  }
  return result;
}


static int answer_relation_fact_extract_made(const corpus_entry_t* entry,
  int made_pos)
{
  int result = 0;
  int rel_begin, rel_end;
  char relation[64];
  char query[96];
  answer_relation_match_t match;
  if (NULL == entry || made_pos < 0 || made_pos + 4 >= entry->text_len) {
    return 0;
  }
  rel_begin = made_pos + 4;
  while (rel_begin < entry->text_len
    && 0 == isalnum((unsigned char)entry->text[rel_begin])) ++rel_begin;
  rel_end = rel_begin;
  while (rel_end < entry->text_len
    && ('-' == entry->text[rel_end]
      || 0 != isalnum((unsigned char)entry->text[rel_end]))) ++rel_end;
  if (rel_end > rel_begin && rel_end - rel_begin > 2
    && rel_end - rel_begin < (int)sizeof(relation)
    && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
      entry->text + rel_begin, rel_end - rel_begin)
    && 0 == answer_relation_rule_has_term(RELATION_RULE_PERSON,
      entry->text + rel_begin, rel_end - rel_begin))
  {
    memcpy(relation, entry->text + rel_begin, (size_t)(rel_end - rel_begin));
    relation[rel_end - rel_begin] = '\0';
    sprintf(query, "who is %s?", relation);
    if (0 != answer_relation_match_query(query, strlen(query), QUERY_WHO,
      entry, &match)
      && match.made != 0
      && EXIT_SUCCESS == answer_relation_fact_append(entry, &match))
    {
      result = 1;
    }
  }
  return result;
}


static size_t answer_relation_facts_build(const libxs_registry_t* corpus)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_relation_facts_free();
  if (NULL == corpus || 0 == answer_relation_rules_size) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    int made_scan = 0;
    size_t rule_pos;
    while (made_scan < entry->text_len) {
      int made_pos = text_find_word_ci(entry->text + made_scan,
        entry->text_len - made_scan, "made");
      if (made_pos < 0) break;
      made_pos += made_scan;
      if (0 != answer_relation_fact_extract_made(entry, made_pos)) ++result;
      made_scan = made_pos + 4;
    }
    for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      int alias_len = 0;
      int verb_pos;
      if (RELATION_RULE_ALIAS != rule->kind) continue;
      verb_pos = answer_relation_rule_alias_pos(rule->relation, entry->text,
        entry->text_len, &alias_len);
      if (verb_pos >= 0 && alias_len > 0) {
        int scan_pos = 0;
        char actor[64];
        while (0 != answer_relation_fact_extract_actor(entry->text,
          entry->text_len, verb_pos, &scan_pos, actor, (int)sizeof(actor)))
        {
          char query[192];
          answer_relation_match_t match;
          sprintf(query, "who was %s by %s?", rule->relation, actor);
          if (0 != answer_relation_match_query(query, strlen(query),
            QUERY_WHO, entry, &match)
            && EXIT_SUCCESS == answer_relation_fact_append(entry, &match))
          {
            ++result;
          }
        }
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_relation_facts_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "relation facts: %lu learned\n",
      (unsigned long)answer_relation_facts_size);
  }
}


static void answer_identity_facts_free(void)
{
  free(answer_identity_facts);
  answer_identity_facts = NULL;
  answer_identity_facts_size = 0;
}


static int answer_identity_word_is_name(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word && word_len > 1
    && 0 != isupper((unsigned char)word[0])
    && 0 == answer_relation_rule_has_term(RELATION_RULE_PERSON, word, word_len)
    && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP, word, word_len))
  {
    int clean = 1;
    int pos;
    for (pos = 0; pos < word_len && 0 != clean; ++pos) {
      unsigned char c = (unsigned char)word[pos];
      if (0 == isalpha(c) && '-' != c) clean = 0;
    }
    result = clean;
  }
  return result;
}


static int answer_identity_fact_append(const char* name, int name_len,
  const char* role, int role_len, const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_identity_fact_t fact;
  answer_identity_fact_t* facts;
  size_t fact_pos;
  if (NULL == name || name_len <= 0 || name_len >= (int)sizeof(fact.name)
    || NULL == role || role_len <= 0 || role_len >= (int)sizeof(fact.role))
  {
    return EXIT_FAILURE;
  }
  LIBXS_MEMZERO(&fact);
  memcpy(fact.name, name, (size_t)name_len);
  fact.name[name_len] = '\0';
  fact.name_len = name_len;
  memcpy(fact.role, role, (size_t)role_len);
  fact.role[role_len] = '\0';
  fact.role_len = role_len;
  if (NULL != entry && entry->section_len > 0
    && entry->section_len < (int)sizeof(fact.section))
  {
    memcpy(fact.section, entry->section, (size_t)entry->section_len);
    fact.section[entry->section_len] = '\0';
    fact.section_len = entry->section_len;
  }
  fact.score = score;
  for (fact_pos = 0; fact_pos < answer_identity_facts_size; ++fact_pos) {
    answer_identity_fact_t* old_fact = answer_identity_facts + fact_pos;
    if (old_fact->name_len == fact.name_len
      && 0 != text_contains_word_ci(old_fact->name, old_fact->name_len,
        fact.name))
    {
      if (fact.score > old_fact->score) {
        memcpy(old_fact->role, fact.role, (size_t)fact.role_len + 1);
        old_fact->role_len = fact.role_len;
        memcpy(old_fact->section, fact.section, (size_t)fact.section_len + 1);
        old_fact->section_len = fact.section_len;
        old_fact->score = fact.score;
      }
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_identity_fact_t*)realloc(answer_identity_facts,
      (answer_identity_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_identity_facts = facts;
      answer_identity_facts[answer_identity_facts_size] = fact;
      ++answer_identity_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_identity_is_connective(const char* word, int word_len)
{
  static const char* const connectives[] = {
    "is", "was", "are", "were", "be", "called", "named", "known",
    "a", "an", "the", "that", "who", "as"
  };
  int result = 0;
  size_t idx;
  for (idx = 0; idx < sizeof(connectives) / sizeof(*connectives)
    && 0 == result; ++idx)
  {
    int clen = (int)strlen(connectives[idx]);
    if (clen == word_len
      && 0 != text_contains_word_ci(word, word_len, connectives[idx]))
    {
      result = 1;
    }
  }
  return result;
}


static size_t answer_identity_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  enum { IDENTITY_GAP_MAX = 3 };
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_identity_facts_free();
  if (NULL == corpus || 0 == answer_relation_rules_size) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    char role[64];
    int role_len = 0;
    int have_role = 0;
    int gap = 0;
    int token_index = 0;
    const char* token;
    const char* prev_end = entry->text;
    int token_len = 0;
    /**
     * Sentence scale only, fragments excluded. A role->name binding is defined
     * within ONE sentence, so the paragraph copy and every comma fragment of a
     * sentence re-derive the same facts from the same bytes: pure duplicated
     * work, and on a large corpus it dominates (13.4 s of 54.9 s at 16 MB of
     * enwik8). The facts themselves are unchanged because a fragment cannot
     * contain a binding its parent sentence does not.
     */
    if (SCALE_SENTENCE != entry->scale
      || 0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      value = libxs_registry_next(corpus, &key, &cursor);
      continue;
    }
    while (NULL != (token = libxs_strtoken(entry->text, delims,
      token_index, &token_len)))
    {
      const char* scan;
      for (scan = prev_end; scan < token; ++scan) {
        if ('.' == *scan || '!' == *scan || '?' == *scan
          || ';' == *scan || ':' == *scan) have_role = 0;
      }
      prev_end = token + token_len;
      if (token_len > 0) {
        int is_name = answer_identity_word_is_name(token, token_len);
        int is_role = (token_len < (int)sizeof(role)
          && 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
            token, token_len)) ? 1 : 0;
        if (0 != have_role && 0 != is_name) {
          if (EXIT_SUCCESS == answer_identity_fact_append(token, token_len,
            role, role_len, entry, (double)(IDENTITY_GAP_MAX - gap)))
          {
            ++result;
          }
          have_role = 0;
        }
        else if (0 != is_role) {
          memcpy(role, token, (size_t)token_len);
          role[token_len] = '\0';
          role_len = token_len;
          have_role = 1;
          gap = 0;
        }
        else if (0 != have_role
          && 0 != answer_identity_is_connective(token, token_len)
          && gap + 1 < IDENTITY_GAP_MAX)
        {
          ++gap;
        }
        else have_role = 0;
      }
      ++token_index;
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_identity_facts_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "identity facts: %lu learned\n",
      (unsigned long)answer_identity_facts_size);
  }
}


static int answer_identity_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char name[64];
  char query_section[ENTRY_SECTION_MAX];
  int name_len;
  int name_upper = 0;
  int query_section_len;
  const answer_identity_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_identity_facts_size) return EXIT_FAILURE;
  name_len = answer_query_be_word(query_text, query_len, name,
    (int)sizeof(name), &name_upper);
  if (name_len <= 0 || 0 == name_upper) return EXIT_FAILURE;
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  for (fact_pos = 0; fact_pos < answer_identity_facts_size; ++fact_pos) {
    const answer_identity_fact_t* fact = answer_identity_facts + fact_pos;
    if (fact->name_len == name_len
      && 0 != text_contains_word_ci(fact->name, fact->name_len, name)
      && (query_section_len <= 0 || fact->section_len <= 0
        || 0 != text_contains_ci(fact->section, fact->section_len,
          query_section))
      && (NULL == best || fact->score > best->score))
    {
      best = fact;
    }
  }
  if (NULL != best) {
    answer_fact_section_set(best->section, best->section_len);
    result = answer_reply_role(output, output_size, best->name,
      best->name_len, best->role);
  }
  return result;
}


static void answer_describe_facts_free(void)
{
  free(answer_describe_facts);
  answer_describe_facts = NULL;
  answer_describe_facts_size = 0;
}


static int answer_describe_fact_append(const char* role, int role_len,
  const char* text, int text_len, const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_describe_fact_t fact;
  answer_describe_fact_t* facts;
  size_t fact_pos;
  if (NULL == role || role_len <= 0 || role_len >= (int)sizeof(fact.role)
    || NULL == text || text_len <= 0 || NULL == entry) return EXIT_FAILURE;
  LIBXS_MEMZERO(&fact);
  if (text_len >= (int)sizeof(fact.text)) text_len = (int)sizeof(fact.text) - 1;
  memcpy(fact.role, role, (size_t)role_len);
  fact.role[role_len] = '\0';
  fact.role_len = role_len;
  memcpy(fact.text, text, (size_t)text_len);
  fact.text[text_len] = '\0';
  fact.text_len = text_len;
  fact.section_len = entry->section_len;
  if (fact.section_len > 0) {
    memcpy(fact.section, entry->section, (size_t)fact.section_len);
    fact.section[fact.section_len] = '\0';
  }
  fact.score = score;
  for (fact_pos = 0; fact_pos < answer_describe_facts_size; ++fact_pos) {
    answer_describe_fact_t* old_fact = answer_describe_facts + fact_pos;
    if (old_fact->role_len == fact.role_len
      && 0 != text_contains_word_ci(old_fact->role, old_fact->role_len,
        fact.role)
      && old_fact->section_len == fact.section_len
      && (0 == fact.section_len
        || 0 != text_contains_ci(old_fact->section, old_fact->section_len,
          fact.section)))
    {
      if (fact.score > old_fact->score) *old_fact = fact;
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_describe_fact_t*)realloc(answer_describe_facts,
      (answer_describe_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_describe_facts = facts;
      answer_describe_facts[answer_describe_facts_size] = fact;
      ++answer_describe_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_describe_word_is_article(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word) {
    if (1 == word_len && ('a' == (word[0] | 32))) result = 1;
    else if (2 == word_len && ('a' == (word[0] | 32))
      && ('n' == (word[1] | 32)))
    {
      result = 1;
    }
  }
  return result;
}


static size_t answer_describe_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n.,;:!?()[]{}\"";
  enum { DESCRIBE_GAP_MAX = 3 };
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_describe_facts_free();
  if (NULL == corpus || 0 == answer_relation_rules_size) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    /* Fragments re-derive their parent sentence's facts; see the identity
       builder. Sentence scale alone was not enough. */
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      const char* article = NULL;
      int gap = 0;
      int token_index = 0;
      const char* token;
      const char* prev_end = entry->text;
      int token_len = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const char* scan;
        for (scan = prev_end; scan < token; ++scan) {
          if (0 == isspace((unsigned char)*scan)) article = NULL;
        }
        prev_end = token + token_len;
        if (token_len > 0) {
          int is_role = (token_len < 64
            && 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
              token, token_len)) ? 1 : 0;
          if (0 != is_role && NULL != article) {
            const char* text_end = entry->text + entry->text_len;
            const char* clause = token + token_len;
            const char* end = clause;
            double score = 1.0;
            while (clause < text_end
              && 0 != isspace((unsigned char)*clause)) ++clause;
            if (clause < text_end && ',' == *clause) {
              int rel_len = 0;
              const char* rel = clause + 1;
              while (rel < text_end
                && 0 != isspace((unsigned char)*rel)) ++rel;
              while (rel + rel_len < text_end
                && 0 != isalnum((unsigned char)rel[rel_len])) ++rel_len;
              if ((3 == rel_len && 0 == strncmp(rel, "who", 3))
                || (5 == rel_len && 0 == strncmp(rel, "which", 5)))
              {
                end = rel + rel_len;
                while (end < text_end && ',' != *end && '.' != *end
                  && ';' != *end && '!' != *end && '?' != *end) ++end;
                score = 2.0;
              }
            }
            if (EXIT_SUCCESS == answer_describe_fact_append(token, token_len,
              article, (int)(end - article), entry, score))
            {
              ++result;
            }
            article = NULL;
          }
          else if (0 != answer_describe_word_is_article(token, token_len)) {
            article = token;
            gap = 0;
          }
          else if (NULL != article && gap + 1 < DESCRIBE_GAP_MAX) ++gap;
          else article = NULL;
        }
        ++token_index;
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_describe_facts_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "describe facts: %lu learned\n",
      (unsigned long)answer_describe_facts_size);
  }
}


static int answer_describe_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char role[64];
  char query_section[ENTRY_SECTION_MAX];
  int role_len;
  int query_section_len;
  const answer_describe_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_describe_facts_size) return EXIT_FAILURE;
  role_len = answer_query_be_word(query_text, query_len, role,
    (int)sizeof(role), NULL);
  if (role_len <= 0 || 0 == answer_relation_rule_has_term(
    RELATION_RULE_PERSON, role, role_len)) return EXIT_FAILURE;
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  for (fact_pos = 0; fact_pos < answer_describe_facts_size; ++fact_pos) {
    const answer_describe_fact_t* fact = answer_describe_facts + fact_pos;
    if (fact->role_len == role_len
      && 0 != text_contains_word_ci(fact->role, fact->role_len, role)
      && (query_section_len <= 0 || fact->section_len <= 0
        || 0 != text_contains_ci(fact->section, fact->section_len,
          query_section))
      && (NULL == best || fact->score > best->score))
    {
      best = fact;
    }
  }
  if (NULL != best && (size_t)best->text_len + 2 <= output_size) {
    answer_fact_section_set(best->section, best->section_len);
    memcpy(output, best->text, (size_t)best->text_len);
    output[0] = (char)toupper((unsigned char)output[0]);
    output[best->text_len] = '.';
    output[best->text_len + 1] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static void answer_docdef_facts_free(void)
{
  free(answer_docdef_facts);
  answer_docdef_facts = NULL;
  answer_docdef_facts_size = 0;
}


/**
 * Parse an optional leading "Header: `name`" line and return the byte offset
 * of the definition prose that follows it (0 when absent). Fills header with
 * the base module name (backticks and trailing extension stripped).
 */
static int answer_docdef_header(const char* text, int text_len,
  char* header, int header_size, int* header_len)
{
  int result = 0;
  *header_len = 0;
  if (text_len > 7 && 0 == strncmp(text, "Header:", 7)) {
    int i = 7;
    int begin, end;
    while (i < text_len && (0 != isspace((unsigned char)text[i])
      || '`' == text[i])) ++i;
    begin = i;
    while (i < text_len && '`' != text[i]
      && 0 == isspace((unsigned char)text[i])) ++i;
    end = i;
    while (end > begin && '.' != text[end - 1]) {
      if ('.' == text[end - 1]) break;
      --end;
    }
    if (end <= begin) end = i;
    else --end;
    if (end - begin > 0 && end - begin < header_size) {
      memcpy(header, text + begin, (size_t)(end - begin));
      header[end - begin] = '\0';
      *header_len = end - begin;
    }
    while (i < text_len && '\n' != text[i]) ++i;
    while (i < text_len && 0 != isspace((unsigned char)text[i])) ++i;
    result = i;
    for (;;) {
      int j = result;
      while (j < text_len && (0 != isalpha((unsigned char)text[j])
        || '-' == text[j] || '_' == text[j])) ++j;
      if (j > result && j < text_len && ':' == text[j]) {
        while (j < text_len && '\n' != text[j]) ++j;
        while (j < text_len && 0 != isspace((unsigned char)text[j])) ++j;
        result = j;
      }
      else break;
    }
  }
  return result;
}


/**
 * Learn module definitions from Markdown ingest: the first paragraph under a
 * heading becomes that title's definition, with the "Header:" filename kept as
 * an alias. Structural only (no corpus vocabulary), so prose corpora without
 * headings contribute nothing.
 */
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_docdef_facts_free();
  if (NULL == corpus) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_PARAGRAPH == entry->scale && entry->section_len > 0) {
      size_t fact_pos;
      int seen = 0;
      for (fact_pos = 0; fact_pos < answer_docdef_facts_size; ++fact_pos) {
        const answer_docdef_fact_t* old = answer_docdef_facts + fact_pos;
        if (old->title_len == entry->section_len
          && 0 == libxs_memcmp(old->title, entry->section,
            (size_t)entry->section_len))
        {
          seen = 1;
          break;
        }
      }
      if (0 == seen) {
        char header[64];
        int header_len = 0;
        int offset = answer_docdef_header(entry->text, entry->text_len,
          header, (int)sizeof(header), &header_len);
        int text_len = entry->text_len - offset;
        if (text_len > 0 && header_len > 0) {
          answer_docdef_fact_t* facts = (answer_docdef_fact_t*)realloc(
            answer_docdef_facts,
            (answer_docdef_facts_size + 1) * sizeof(*facts));
          if (NULL != facts) {
            answer_docdef_fact_t* fact = facts + answer_docdef_facts_size;
            answer_docdef_facts = facts;
            LIBXS_MEMZERO(fact);
            if (text_len >= (int)sizeof(fact->text)) {
              text_len = (int)sizeof(fact->text) - 1;
            }
            memcpy(fact->title, entry->section, (size_t)entry->section_len);
            fact->title[entry->section_len] = '\0';
            fact->title_len = entry->section_len;
            fact->header_len = header_len;
            if (header_len > 0) memcpy(fact->header, header,
              (size_t)header_len + 1);
            memcpy(fact->text, entry->text + offset, (size_t)text_len);
            fact->text[text_len] = '\0';
            fact->text_len = text_len;
            ++answer_docdef_facts_size;
            ++result;
          }
        }
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_docdef_facts_report(FILE* stream)
{
  if (NULL != stream && answer_docdef_facts_size > 0) {
    fprintf(stream, "docdef facts: %lu learned\n",
      (unsigned long)answer_docdef_facts_size);
  }
}


/**
 * Extract the subject term from "what is [the] X" or "what does X do",
 * skipping a leading article. Language-generic, no corpus vocabulary.
 */
static int answer_docdef_term(const char* query_text, size_t query_len,
  char* term, int term_size)
{
  static const char* const markers[] = { " is ", " does ", " are " };
  int result = 0;
  int marker_pos = -1;
  int marker_len = 0;
  int m;
  size_t begin, end;
  if (NULL == query_text || NULL == term || term_size <= 0) return 0;
  term[0] = '\0';
  for (m = 0; m < 3 && marker_pos < 0; ++m) {
    marker_pos = text_find_ci(query_text, (int)query_len, markers[m]);
    if (marker_pos >= 0) marker_len = (int)strlen(markers[m]);
  }
  if (marker_pos < 0) return 0;
  begin = (size_t)marker_pos + (size_t)marker_len;
  while (begin < query_len && 0 != isspace((unsigned char)query_text[begin])) {
    ++begin;
  }
  end = begin;
  while (end < query_len && '?' != query_text[end] && '.' != query_text[end]
    && '!' != query_text[end]) ++end;
  while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) --end;
  if (end - begin >= 3 && 0 != isspace((unsigned char)query_text[end - 3])
    && ('d' == (query_text[end - 2] | 32)) && ('o' == (query_text[end - 1] | 32)))
  {
    end -= 3;
  }
  while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) --end;
  if (end - begin >= 2 && 0 == strncmp(query_text + begin, "a ", 2)) begin += 2;
  else if (end - begin >= 3
    && 0 == strncmp(query_text + begin, "an ", 3)) begin += 3;
  else if (end - begin >= 4
    && 0 == strncmp(query_text + begin, "the ", 4)) begin += 4;
  while (begin < end && 0 != isspace((unsigned char)query_text[begin])) {
    ++begin;
  }
  result = (int)(end - begin);
  if (result >= term_size) result = term_size - 1;
  if (result > 0) {
    memcpy(term, query_text + begin, (size_t)result);
    term[result] = '\0';
  }
  return result;
}


/**
 * Answer "What is X?" / "What does X do?" from a learned module definition,
 * matching X against either the heading text or the Header: filename alias.
 */
static int answer_docdef_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char term[ENTRY_SECTION_MAX];
  int term_len;
  const answer_docdef_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_docdef_facts_size) return EXIT_FAILURE;
  term_len = answer_docdef_term(query_text, query_len, term,
    (int)sizeof(term));
  if (term_len <= 0) return EXIT_FAILURE;
  for (fact_pos = 0; fact_pos < answer_docdef_facts_size && NULL == best;
    ++fact_pos)
  {
    const answer_docdef_fact_t* fact = answer_docdef_facts + fact_pos;
    if ((fact->title_len == term_len
        && 0 != text_contains_word_ci(fact->title, fact->title_len, term))
      || (fact->header_len == term_len
        && 0 != text_contains_word_ci(fact->header, fact->header_len, term)))
    {
      best = fact;
    }
  }
  if (NULL != best && (size_t)best->text_len + 1 <= output_size) {
    answer_fact_section_set(best->title, best->title_len);
    memcpy(output, best->text, (size_t)best->text_len);
    output[best->text_len] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static void conv_reset(void)
{
  conv_topic[0] = '\0';
  conv_topic_len = 0;
}


/**
 * Remember the subject of a successfully answered question so that a later
 * follow-up can refer back to it. Uses the same subject extraction as the
 * definition path; only a real subject term updates the topic.
 */
static void conv_remember(const char* query_text, size_t query_len)
{
  char term[CONV_TOPIC_MAX];
  int term_len = answer_docdef_term(query_text, query_len, term,
    (int)sizeof(term));
  if (term_len > 0) {
    memcpy(conv_topic, term, (size_t)term_len);
    conv_topic[term_len] = '\0';
    conv_topic_len = term_len;
  }
}


static int conv_word_is_pronoun(const char* word, int len)
{
  static const char* const pronouns[] = { "it", "its", "it's", "that",
    "this", "they", "them", "their" };
  int result = 0;
  int p;
  for (p = 0; p < (int)(sizeof(pronouns) / sizeof(*pronouns)) && 0 == result;
    ++p)
  {
    if ((int)strlen(pronouns[p]) == len) {
      int i, same = 1;
      for (i = 0; i < len && 0 != same; ++i) {
        if ((word[i] | 32) != pronouns[p][i]) same = 0;
      }
      result = same;
    }
  }
  return result;
}


/**
 * Rewrite a follow-up against the remembered topic. Two grounded moves:
 * substitute a back-reference pronoun (it/its/that/...) with the topic, and,
 * when a question carries no subject of its own, append the topic so the
 * answer path is scoped to it. Returns EXIT_SUCCESS only when a rewrite was
 * made, leaving the original query untouched otherwise.
 */
static int conv_rewrite(const char* query_text, size_t query_len,
  char* out, size_t out_size)
{
  int result = EXIT_FAILURE;
  char own[CONV_TOPIC_MAX];
  int own_len;
  size_t pos = 0, w = 0;
  int replaced = 0;
  if (NULL == query_text || NULL == out || 0 == out_size
    || 0 == conv_topic_len) return EXIT_FAILURE;
  while (w < query_len && w + 1 < out_size) {
    size_t begin = w;
    while (w < query_len && 0 != isalpha((unsigned char)query_text[w])) ++w;
    if (w > begin && 0 != conv_word_is_pronoun(query_text + begin,
      (int)(w - begin)))
    {
      if (pos + (size_t)conv_topic_len < out_size) {
        memcpy(out + pos, conv_topic, (size_t)conv_topic_len);
        pos += (size_t)conv_topic_len;
        replaced = 1;
      }
    }
    else if (w > begin) {
      if (pos + (w - begin) < out_size) {
        memcpy(out + pos, query_text + begin, w - begin);
        pos += w - begin;
      }
    }
    if (w < query_len && 0 == isalpha((unsigned char)query_text[w])
      && pos + 1 < out_size)
    {
      out[pos++] = query_text[w++];
    }
  }
  out[pos] = '\0';
  if (0 != replaced) {
    result = EXIT_SUCCESS;
  }
  else {
    own_len = answer_docdef_term(query_text, query_len, own,
      (int)sizeof(own));
    if (own_len <= 0 && pos > 0) {
      static const char suffix[] = " of the ";
      size_t suffix_len = sizeof(suffix) - 1;
      if (pos + suffix_len + (size_t)conv_topic_len < out_size) {
        memcpy(out + pos, suffix, suffix_len);
        pos += suffix_len;
        memcpy(out + pos, conv_topic, (size_t)conv_topic_len);
        pos += (size_t)conv_topic_len;
        out[pos] = '\0';
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static int answer_relation_fact_relation_match(const char* query_relation,
  const answer_relation_fact_t* fact)
{
  int result = 0;
  size_t rule_pos;
  if (NULL == query_relation || NULL == fact) return 0;
  if (0 != text_contains_ci(query_relation, (int)strlen(query_relation),
    fact->relation))
  {
    result = 1;
  }
  for (rule_pos = 0; rule_pos < answer_relation_rules_size && 0 == result;
    ++rule_pos)
  {
    const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
    if (RELATION_RULE_ALIAS == rule->kind
      && 0 != text_contains_ci(rule->relation, (int)strlen(rule->relation),
        fact->relation)
      && 0 != text_contains_ci(query_relation, (int)strlen(query_relation),
        rule->term))
    {
      result = 1;
    }
  }
  return result;
}


static int answer_relation_fact_actor_match(const char* query_actor,
  int query_actor_len, const answer_relation_fact_t* fact)
{
  int result = 0;
  if (NULL == fact) return 0;
  if (query_actor_len <= 0) result = (0 == fact->actor_len) ? 1 : 0;
  else if (fact->actor_len > 0) {
    if (0 != answer_relation_actor_has_token(query_actor, query_actor_len,
        fact->actor, fact->actor_len)
      || 0 != answer_relation_actor_has_token(fact->actor, fact->actor_len,
        query_actor, query_actor_len))
    {
      result = 1;
    }
  }
  return result;
}


static int answer_relation_fact_section_match(const char* query_section,
  int query_section_len, const answer_relation_fact_t* fact)
{
  int result = 1;
  if (query_section_len > 0) {
    result = 0;
    if (NULL != fact && fact->section_len > 0
      && query_section_len < ENTRY_SECTION_MAX)
    {
      char section[ENTRY_SECTION_MAX];
      memcpy(section, query_section, (size_t)query_section_len);
      section[query_section_len] = '\0';
      if (0 != text_contains_ci(fact->section, fact->section_len, section)
        || 0 == libxs_stridiff(fact->section, section, NULL, 1, NULL))
      {
        result = 1;
      }
    }
  }
  return result;
}


static int answer_relation_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  enum { RELATION_FACT_MAX = 4 };
  int result = EXIT_FAILURE;
  char relation[64];
  char actor[64];
  char query_section[ENTRY_SECTION_MAX];
  char answers[RELATION_FACT_MAX][64];
  int answer_lens[RELATION_FACT_MAX];
  int answer_made[RELATION_FACT_MAX];
  double answer_scores[RELATION_FACT_MAX];
  int relation_upper = 0;
  int relation_len;
  int actor_len;
  int query_section_len;
  int count = 0;
  int slot;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_relation_facts_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation), &relation_upper);
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (relation_len <= 0 || 0 != relation_upper) {
    return EXIT_FAILURE;
  }
  for (slot = 0; slot < RELATION_FACT_MAX; ++slot) {
    answers[slot][0] = '\0';
    answer_lens[slot] = 0;
    answer_made[slot] = 0;
    answer_scores[slot] = 0.0;
  }
  for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
    const answer_relation_fact_t* fact = answer_relation_facts + fact_pos;
    if (0 != answer_relation_fact_relation_match(relation, fact)
      && 0 != answer_relation_fact_actor_match(actor, actor_len, fact)
      && 0 != answer_relation_fact_section_match(query_section,
        query_section_len, fact))
    {
      int duplicate = 0;
      for (slot = 0; slot < count; ++slot) {
        if (0 != answer_relation_same_answer(answers[slot],
          answer_lens[slot], fact->answer, fact->answer_len))
        {
          duplicate = 1;
          if (fact->score > answer_scores[slot]) {
            memcpy(answers[slot], fact->answer,
              (size_t)fact->answer_len + 1);
            answer_lens[slot] = fact->answer_len;
            answer_made[slot] = fact->made;
            answer_scores[slot] = fact->score;
          }
        }
      }
      if (0 == duplicate && count < RELATION_FACT_MAX) {
        int insert = count;
        while (insert > 0 && fact->score > answer_scores[insert - 1]) {
          memcpy(answers[insert], answers[insert - 1],
            (size_t)answer_lens[insert - 1] + 1);
          answer_lens[insert] = answer_lens[insert - 1];
          answer_made[insert] = answer_made[insert - 1];
          answer_scores[insert] = answer_scores[insert - 1];
          --insert;
        }
        memcpy(answers[insert], fact->answer,
          (size_t)fact->answer_len + 1);
        answer_lens[insert] = fact->answer_len;
        answer_made[insert] = fact->made;
        answer_scores[insert] = fact->score;
        ++count;
      }
    }
  }
  if (count > 0) {
    size_t pos = 0;
    int item;
    output[0] = '\0';
    for (item = 0; item < count && pos + 1 < output_size; ++item) {
      if (item > 0) {
        const char* joiner = (item + 1 == count) ? " and " : ", ";
        size_t joiner_len = strlen(joiner);
        if (pos + joiner_len + 1 >= output_size) break;
        memcpy(output + pos, joiner, joiner_len);
        pos += joiner_len;
      }
      if (pos + (size_t)answer_lens[item] + 1 >= output_size) break;
      memcpy(output + pos, answers[item], (size_t)answer_lens[item]);
      pos += (size_t)answer_lens[item];
    }
    if (pos + (size_t)relation_len + (size_t)actor_len + 24 < output_size) {
      const char* copula = (count > 1) ? " were " : " was ";
      if (actor_len <= 0) copula = (count > 1) ? " are " : " is ";
      if (actor_len <= 0 && 0 != answer_made[0]) {
        static const char made_one[] = " is to be made ";
        static const char made_many[] = " are to be made ";
        const char* made_text = (count > 1) ? made_many : made_one;
        memcpy(output + pos, made_text, strlen(made_text));
        pos += strlen(made_text);
      }
      else {
        memcpy(output + pos, copula, strlen(copula));
        pos += strlen(copula);
      }
      memcpy(output + pos, relation, (size_t)relation_len);
      pos += (size_t)relation_len;
      if (actor_len > 0) {
        static const char by_text[] = " by ";
        memcpy(output + pos, by_text, sizeof(by_text) - 1);
        pos += sizeof(by_text) - 1;
        memcpy(output + pos, actor, (size_t)actor_len);
        pos += (size_t)actor_len;
      }
      output[pos++] = '.';
      output[pos] = '\0';
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_relation_aggregate_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size)
{
  enum { RELATION_AGG_MAX = 4 };
  int result = EXIT_FAILURE;
  const void* key = NULL;
  size_t cursor = 0, key_size = 0;
  void* value;
  char query_section[ENTRY_SECTION_MAX];
  int query_section_len;
  char answers[RELATION_AGG_MAX][64];
  int answer_lens[RELATION_AGG_MAX];
  double answer_scores[RELATION_AGG_MAX];
  int count = 0;
  char relation[64];
  char actor[64];
  int relation_upper = 0;
  int relation_len;
  int actor_len;
  if (NULL == corpus || NULL == query_text || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation), &relation_upper);
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (query_section_len > 0 && relation_len > 0 && 0 == relation_upper
    && actor_len > 0)
  {
    int slot;
    for (slot = 0; slot < RELATION_AGG_MAX; ++slot) {
      answers[slot][0] = '\0';
      answer_lens[slot] = 0;
      answer_scores[slot] = 0.0;
    }
    value = libxs_registry_begin_length(corpus, &key, &key_size, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      /* The corpus deliberately holds keys of two sizes, so the size must come
         from the iterator; a hardcoded one silently misses every entry. */
      size_t entry_size = (NULL != key)
        ? libxs_registry_value_size(corpus, key, key_size, NULL)
        : sizeof(*entry);
      answer_relation_match_t match;
      if ((query_section_len <= 0
          || 0 != corpus_entry_section_match(entry, entry_size,
            query_section, query_section_len))
        && 0 != answer_relation_match_query(query_text, query_len,
          QUERY_WHO, entry, &match)
        && 0 == match.made && match.actor_len > 0 && match.answer_len > 0)
      {
        char candidate[64];
        int candidate_len = answer_relation_section_title(candidate,
          (int)sizeof(candidate), entry, &match);
        double candidate_score = match.score;
        int duplicate = 0;
        if (candidate_len > 0) candidate_score += 0.10;
        else {
          candidate_len = match.answer_len;
          memcpy(candidate, match.answer, (size_t)candidate_len);
          candidate[candidate_len] = '\0';
        }
        for (slot = 0; slot < count; ++slot) {
          if (0 != answer_relation_same_answer(answers[slot],
            answer_lens[slot], candidate, candidate_len))
          {
            duplicate = 1;
            if (candidate_score > answer_scores[slot]) {
              memcpy(answers[slot], candidate, (size_t)candidate_len + 1);
              answer_lens[slot] = candidate_len;
              answer_scores[slot] = candidate_score;
            }
          }
        }
        if (0 == duplicate && count < RELATION_AGG_MAX) {
          int insert = count;
          while (insert > 0 && candidate_score > answer_scores[insert - 1]) {
            memcpy(answers[insert], answers[insert - 1],
              (size_t)answer_lens[insert - 1] + 1);
            answer_lens[insert] = answer_lens[insert - 1];
            answer_scores[insert] = answer_scores[insert - 1];
            --insert;
          }
          memcpy(answers[insert], candidate, (size_t)candidate_len + 1);
          answer_lens[insert] = candidate_len;
          answer_scores[insert] = candidate_score;
          ++count;
        }
      }
      value = libxs_registry_next_length(corpus, &key, &key_size, &cursor);
    }
    if (count > 1) {
      size_t pos = 0;
      int item;
      output[0] = '\0';
      for (item = 0; item < count && pos + 1 < output_size; ++item) {
        if (item > 0) {
          const char* joiner = (item + 1 == count) ? " and " : ", ";
          size_t joiner_len = strlen(joiner);
          if (pos + joiner_len + 1 >= output_size) break;
          memcpy(output + pos, joiner, joiner_len);
          pos += joiner_len;
        }
        if (pos + (size_t)answer_lens[item] + 1 >= output_size) break;
        memcpy(output + pos, answers[item], (size_t)answer_lens[item]);
        pos += (size_t)answer_lens[item];
      }
      if (pos + (size_t)relation_len + (size_t)actor_len + 11 < output_size) {
        static const char were_text[] = " were ";
        static const char by_text[] = " by ";
        memcpy(output + pos, were_text, sizeof(were_text) - 1);
        pos += sizeof(were_text) - 1;
        memcpy(output + pos, relation, (size_t)relation_len);
        pos += (size_t)relation_len;
        memcpy(output + pos, by_text, sizeof(by_text) - 1);
        pos += sizeof(by_text) - 1;
        memcpy(output + pos, actor, (size_t)actor_len);
        pos += (size_t)actor_len;
        output[pos++] = '.';
        output[pos] = '\0';
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static double answer_identity_score(const char* query_text, size_t query_len,
  int query_type, const corpus_entry_t* entry)
{
  double result = 0.0;
  char word[64];
  int upper_initial = 0;
  int word_len;
  answer_relation_match_t relation_match;
  if (QUERY_WHO != query_type || NULL == entry) return 0.0;
  word_len = answer_query_be_word(query_text, query_len, word,
    (int)sizeof(word), &upper_initial);
  if (word_len <= 0) return 0.0;
  if (0 == upper_initial && 0 != answer_relation_match_query(query_text,
    query_len, query_type, entry, &relation_match))
  {
    result = relation_match.score;
  }
  if (0 == text_contains_word_ci(entry->text, entry->text_len, word)) {
    return result;
  }
  if (0 != upper_initial) {
    result = 0.55;
    if (0 != text_contains_ci(entry->text, entry->text_len, "called")) {
      result += 0.55;
    }
    if (0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
      entry->text, entry->text_len))
    {
      result += 0.35;
    }
  }
  else {
    result = 0.35;
    if (0 != text_contains_ci(entry->text, entry->text_len, " is ")
      || 0 != text_contains_ci(entry->text, entry->text_len, " was ")
      || 0 != text_contains_ci(entry->text, entry->text_len, " are ")
      || 0 != text_contains_ci(entry->text, entry->text_len, " were "))
    {
      result += 0.25;
    }
    if (0 != text_contains_ci(entry->text, entry->text_len, "made")) {
      result += 0.35;
    }
    if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) result += 0.15;
  }
  return result;
}


static int answer_features_fill(const corpus_entry_t* entry,
  size_t entry_size, double overlap, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS])
{
  int result = EXIT_FAILURE;
  int use_sketch = 0;
  int entry_words = 0;
  int input_pos;
  for (input_pos = 0; input_pos < ANSWER_PREDICT_INPUTS; ++input_pos) {
    inputs[input_pos] = 0.0;
  }
  if (NULL == entry || NULL == inputs) return EXIT_FAILURE;
  if (entry_size >= CORPUS_ENTRY_META_SIZE && entry->ntokens > 0) use_sketch = 1;
  if (0 == use_sketch) return EXIT_FAILURE;
  if (overlap < 0.0) overlap = 0.0;
  if (overlap > 1.0) overlap = 1.0;
  entry_words = count_words((const unsigned char*)entry->text, entry->text_len);
  inputs[0] = overlap;
  inputs[1] = (double)query_type / (double)QUERY_YESNO;
  inputs[2] = (SCALE_SENTENCE == entry->scale) ? 1.0 : 0.0;
  inputs[3] = (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) ? 1.0 : 0.0;
  inputs[4] = (0 != (entry->lexical_flags & ENTRY_LEX_NUMBER)) ? 1.0 : 0.0;
  inputs[5] = (0 != (entry->lexical_flags & ENTRY_LEX_PLACE)) ? 1.0 : 0.0;
  inputs[6] = (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) ? 1.0 : 0.0;
  inputs[7] = (0 != (entry->lexical_flags & ENTRY_LEX_METHOD)) ? 1.0 : 0.0;
  inputs[8] = (entry_words > 0) ? (double)entry->ncontent / entry_words : 0.0;
  inputs[9] = (entry->text_len > COMPOSE_MAXTEXT)
    ? 1.0 : (double)entry->text_len / (double)COMPOSE_MAXTEXT;
  result = EXIT_SUCCESS;
  return result;
}


static int answer_features(const libxs_lexeme_stream_t* query,
  const corpus_entry_t* entry, size_t entry_size, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS])
{
  double total = 0.0, matched = 0.0;
  size_t query_pos;
  if (NULL == query || NULL == entry || NULL == inputs) return EXIT_FAILURE;
  for (query_pos = 0; query_pos < query->size; ++query_pos) {
    const libxs_lexeme_t* lexeme = query->data + query_pos;
    if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
    {
      double weight = (lexeme->length >= 6) ? 1.5 : 1.0;
      total += weight;
      if (0 != entry_sketch_has_id(entry, lexeme->id)) matched += weight;
    }
  }
  return answer_features_fill(entry, entry_size,
    (total > 0.0) ? matched / total : 0.0, query_type, inputs);
}


static double answer_weak_label(const corpus_entry_t* entry, int query_type)
{
  double result = 0.0;
  if (NULL != entry) {
    switch (query_type) {
      case QUERY_WHO:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) ? 1.0 : 0.0;
        break;
      case QUERY_WHAT:
        result = (SCALE_SENTENCE == entry->scale) ? 0.60 : 0.35;
        break;
      case QUERY_WHERE:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_PLACE)) ? 1.0 : 0.0;
        break;
      case QUERY_WHEN:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_NUMBER)) ? 0.95 : 0.0;
        break;
      case QUERY_WHY:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) ? 1.0 : 0.0;
        break;
      case QUERY_HOW:
        if (0 != (entry->lexical_flags & ENTRY_LEX_METHOD)) result = 0.95;
        else if (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) result = 0.35;
        break;
      default:
        break;
    }
  }
  return result;
}


static libxs_predict_t* answer_predict_create(
  const answer_predict_profile_t* profile)
{
  static const double weights[ANSWER_PREDICT_INPUTS] = {
    3.0, 1.0, 0.5, 1.5, 1.2, 1.4, 1.4, 1.2, 0.4, 0.3
  };
  libxs_predict_t* result = libxs_predict_create(ANSWER_PREDICT_INPUTS, 1);
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL != result) {
    libxs_predict_set_mode(result, profile->mode);
    libxs_predict_set_decompose(result, profile->decompose);
    libxs_predict_set_weights(result, weights);
    if (0.0 != profile->smooth) libxs_predict_set_smooth(result,
      profile->smooth);
    if (0 < profile->nseries && 0 < profile->window) {
      libxs_predict_set_series(result, profile->nseries, profile->window);
      libxs_predict_set_target(result, profile->target);
    }
    if (0 != profile->diff_order) libxs_predict_set_diff(result,
      profile->diff_order);
  }
  return result;
}


static int answer_predict_build_model(libxs_predict_t* model,
  const answer_predict_profile_t* profile)
{
  int result = EXIT_FAILURE;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL != model) {
    result = libxs_predict_build(model, profile->clusters,
      profile->order, profile->quality);
  }
  return result;
}


static void answer_predict_report(const char* label,
  const libxs_predict_t* model, int ntrain,
  const answer_predict_profile_t* profile)
{
  libxs_predict_query_t info;
  LIBXS_MEMZERO(&info);
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL != label && NULL != model) {
    libxs_predict_query(model, &info);
    fprintf(stderr,
      "predict[%s:%s]: inputs=%d pushed=%d entries=%d clusters=%d order=%d diff=%d compression=%.2fx\n",
      label, profile->name, ANSWER_PREDICT_INPUTS, ntrain,
      info.nentries, info.nclusters, info.order, info.diff_order,
      info.compression);
  }
}


static libxs_predict_t* converse_predict_train(const libxs_registry_t* corpus,
  const answer_predict_profile_t* profile)
{
  libxs_predict_t* result = NULL;
  libxs_predict_t* model = NULL;
  const void* key = NULL;
  size_t cursor = 0, key_size = 0;
  void* value;
  int ntrain = 0;
  if (NULL == corpus) return NULL;
  model = answer_predict_create(profile);
  if (NULL == model) return NULL;
  value = libxs_registry_begin_length(corpus, &key, &key_size, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    /* Key size from the iterator: see the note in answer_relation_reply. */
    size_t entry_size = (NULL != key)
      ? libxs_registry_value_size(corpus, key, key_size, NULL)
      : sizeof(*entry);
    if (entry_size >= CORPUS_ENTRY_META_SIZE && entry->ntokens > 0
      && entry->text_len >= 16)
    {
      int query_type;
      for (query_type = QUERY_WHO; query_type <= QUERY_HOW; ++query_type) {
        double label = answer_weak_label(entry, query_type);
        double overlap = (label >= 0.5) ? 1.0 : 0.0;
        double inputs[ANSWER_PREDICT_INPUTS];
        if (QUERY_WHAT == query_type && label > 0.0) overlap = 0.65;
        if (EXIT_SUCCESS == answer_features_fill(entry, entry_size,
          overlap, query_type, inputs)
          && EXIT_SUCCESS == libxs_predict_push(NULL, model, inputs, &label))
        {
          ++ntrain;
        }
      }
    }
    value = libxs_registry_next_length(corpus, &key, &key_size, &cursor);
  }
  if (ntrain >= 8 && EXIT_SUCCESS == answer_predict_build_model(model,
    profile))
  {
    answer_predict_report("persistent", model, ntrain, profile);
    result = model;
  }
  else {
    libxs_predict_destroy(model);
  }
  return result;
}


static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile)
{
  libxs_predict_t* result = NULL;
  libxs_predict_t* model = NULL;
  const void* key = NULL;
  size_t cursor = 0, key_size = 0;
  void* value;
  int ntrain = 0;
  if (NULL == corpus || NULL == query || NULL == lexicon) return NULL;
  model = answer_predict_create(profile);
  if (NULL == model) return NULL;
  value = libxs_registry_begin_length(corpus, &key, &key_size, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    size_t entry_size = (NULL != key)
      ? libxs_registry_value_size(corpus, key, key_size, NULL) : sizeof(*entry);
    double inputs[ANSWER_PREDICT_INPUTS];
    double output;
    if (entry->text_len >= 16
      && EXIT_SUCCESS == answer_features(query, entry, entry_size,
        query_type, inputs))
    {
      output = lexical_score(query, lexicon, rules, nrules, entry,
        entry_size, query_type);
      if (EXIT_SUCCESS == libxs_predict_push(NULL, model, inputs, &output)) {
        ++ntrain;
      }
    }
    value = libxs_registry_next_length(corpus, &key, &key_size, &cursor);
  }
  if (ntrain >= 4 && EXIT_SUCCESS == answer_predict_build_model(model,
    profile))
  {
    result = model;
  }
  else {
    libxs_predict_destroy(model);
  }
  return result;
}


static double answer_predict_score(const libxs_predict_t* model,
  const double inputs[ANSWER_PREDICT_INPUTS], double base_score)
{
  double result = base_score;
  if (NULL != model && NULL != inputs) {
    double output = 0.0;
    libxs_predict_info_t info;
    libxs_predict_eval(NULL, model, inputs, &output, &info, 1);
    if (NULL != info.confidence && info.confidence[0] > 0.0) {
      double confidence = info.confidence[0];
      if (confidence > 1.0) confidence = 1.0;
      if (output < 0.0) output = 0.0;
      result = 0.75 * base_score + 0.25 * confidence * output;
    }
  }
  return result;
}


static int answer_bridge_query_group_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const char* group, int group_len)
{
  int result = 0;
  int start = 0;
  while (start < group_len && 0 == result) {
    int end = start;
    char term[64];
    int term_len;
    while (end < group_len && '/' != group[end]) ++end;
    term_len = end - start;
    if (term_len > 0 && term_len < (int)sizeof(term)) {
      memcpy(term, group + start, (size_t)term_len);
      term[term_len] = '\0';
      result = lexeme_stream_has_text(query, lexicon, term);
      if (0 == result && term_len >= 5) {
        result = lexeme_stream_has_similar_text(query, lexicon, term,
          term_len, 1);
      }
    }
    start = end + 1;
  }
  return result;
}


static int answer_bridge_query_match(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* spec)
{
  int result = 1;
  const char* group = spec;
  if (NULL == query || NULL == lexicon || NULL == spec) return 0;
  while ('\0' != *group && 0 != result) {
    const char* end;
    while ('\0' != *group && 0 != isspace((unsigned char)*group)) ++group;
    end = group;
    while ('\0' != *end && 0 == isspace((unsigned char)*end)) ++end;
    if (end > group) {
      result = answer_bridge_query_group_match(query, lexicon, group,
        (int)(end - group));
    }
    group = end;
  }
  return result;
}


static int answer_bridge_evidence_group_match(const char* text, int text_len,
  const char* group, int group_len)
{
  int result = 0;
  int start = 0;
  while (start < group_len && 0 == result) {
    int end = start;
    char term[64];
    int term_pos, term_len;
    while (end < group_len && '/' != group[end]) ++end;
    term_len = end - start;
    if (term_len > 0 && term_len < (int)sizeof(term)) {
      char text_buf[COMPOSE_MAXTEXT];
      memcpy(term, group + start, (size_t)term_len);
      term[term_len] = '\0';
      for (term_pos = 0; term_pos < term_len; ++term_pos) {
        if ('_' == term[term_pos]) term[term_pos] = ' ';
      }
      if (text_len > 0 && text_len < (int)sizeof(text_buf)) {
        int count = 0;
        int matches;
        memcpy(text_buf, text, (size_t)text_len);
        text_buf[text_len] = '\0';
        matches = libxs_strimatch(text_buf, term, NULL, &count);
        if (count > 0 && matches >= count) result = 1;
      }
      if (0 == result) result = text_contains_ci(text, text_len, term);
    }
    start = end + 1;
  }
  return result;
}


static int answer_bridge_evidence_match(const corpus_entry_t* entry,
  const char* spec)
{
  int result = 1;
  const char* group = spec;
  if (NULL == entry || NULL == spec) return 0;
  while ('\0' != *group && 0 != result) {
    const char* end;
    while ('\0' != *group && 0 != isspace((unsigned char)*group)) ++group;
    end = group;
    while ('\0' != *end && 0 == isspace((unsigned char)*end)) ++end;
    if (end > group) {
      result = answer_bridge_evidence_group_match(entry->text,
        entry->text_len, group, (int)(end - group));
    }
    group = end;
  }
  return result;
}


static const answer_bridge_t* answer_bridge_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const corpus_entry_t* entry)
{
  const answer_bridge_t* result = NULL;
  size_t bridge_pos;
  if (NULL != query && NULL != lexicon && NULL != entry) {
    for (bridge_pos = 0; bridge_pos < answer_bridge_loaded_size
      && NULL == result; ++bridge_pos)
    {
      const answer_bridge_t* bridge = answer_bridge_loaded + bridge_pos;
      if (0 != answer_bridge_query_match(query, lexicon, bridge->query)
        && 0 != answer_bridge_evidence_match(entry, bridge->evidence))
      {
        result = bridge;
      }
    }
  }
  return result;
}


static double answer_semantic_bridge_score(const answer_bridge_t* bridge)
{
  return (NULL != bridge) ? bridge->score : 0.0;
}


static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type)
{
  double score = 0.0, total = 0.0;
  double min_overlap = 0.75;
  int matches = 0, use_sketch = 0;
  int entry_stream_ready = 0;
  int long_terms = 0, long_matches = 0;
  int query_terms = 0, term_matches = 0, exact_matches = 0;
  int entry_words;
  double bridge_score;
  const answer_bridge_t* bridge;
  double overlap;
  libxs_lexeme_stream_t entry_stream;
  size_t query_pos;
  libxs_lexeme_stream_init(&entry_stream);
  if (NULL == query || NULL == lexicon || NULL == entry) return 0.0;
  if (entry_size >= CORPUS_ENTRY_META_SIZE && entry->ntokens > 0) {
    use_sketch = 1;
  }
  else if (EXIT_SUCCESS != libxs_lexeme_stream_encode(lexicon, &entry_stream,
      (const unsigned char*)entry->text, (size_t)entry->text_len,
      rules, nrules, answer_lexnorms, answer_lexnorms_size, 1))
  {
    libxs_lexeme_stream_release(&entry_stream);
    return 0.0;
  }
  else entry_stream_ready = 1;
  for (query_pos = 0; query_pos < query->size; ++query_pos) {
    const libxs_lexeme_t* lexeme = query->data + query_pos;
    if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
    {
      double weight = (lexeme->length >= 6) ? 1.5 : 1.0;
      int long_term = (lexeme->length >= 6) ? 1 : 0;
      int matched, exact;
      total += weight;
      ++query_terms;
      if (0 != long_term) ++long_terms;
      matched = ((0 != use_sketch && 0 != entry_sketch_has_id(entry,
          lexeme->id))
        || (0 == use_sketch && 0 != lexeme_stream_has_id(&entry_stream,
          lexeme->id))) ? 1 : 0;
      exact = matched;
      if (0 == matched && lexeme->length >= 5) {
        int term_len = 0;
        const char* term = libxs_lexicon_text(lexicon, lexeme->id,
          &term_len, NULL);
        if (0 == entry_stream_ready
          && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
            &entry_stream, (const unsigned char*)entry->text,
            (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 1))
        {
          entry_stream_ready = 1;
        }
        if (0 != entry_stream_ready
          && 0 != lexeme_stream_has_similar_text(&entry_stream, lexicon,
            term, term_len, 1))
        {
          matched = 1;
        }
      }
      if (0 != matched) {
        score += weight;
        ++matches;
        ++term_matches;
        if (0 != exact) ++exact_matches;
        if (0 != long_term) ++long_matches;
      }
    }
  }
  bridge = answer_bridge_match(query, lexicon, entry);
  bridge_score = answer_semantic_bridge_score(bridge);
  if (bridge_score > 0.0) {
    score += bridge_score;
    total += 1.0;
    ++matches;
    if (long_terms > 0) ++long_matches;
  }
  libxs_lexeme_stream_release(&entry_stream);
  if (total <= 0.0 || 0 == matches) return 0.0;
  if (QUERY_GENERIC != query_type && long_terms > 0 && 0 == long_matches) {
    return 0.0;
  }
  if (QUERY_GENERIC != query_type && term_matches > 0 && 0 == exact_matches) {
    return 0.0;
  }
  if (QUERY_GENERIC != query_type) min_overlap = 0.40;
  overlap = score / total;
  if (bridge_score > 0.0 && overlap < bridge_score) overlap = bridge_score;
  /**
   * A question with a single content term ("what is the X") scores a perfect
   * overlap as soon as that one term matches, so the ratio cannot distinguish
   * a real hit from a passage that merely mentions something spelled like X.
   * Require the lone term to match EXACTLY: an approximate match on the only
   * term the query carries is how an entity absent from the corpus gets a
   * confident reply.
   */
  if (query_terms <= 1 && term_matches > 0 && 0 == exact_matches) {
    return 0.0;
  }
  if (overlap < min_overlap && !(total <= 1.5 && matches >= 1)) {
    return 0.0;
  }
  score = overlap;
  if (0 != use_sketch) {
    switch (query_type) {
      case QUERY_WHO:
        if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) score += 0.16;
        else score -= 0.06;
        break;
      case QUERY_WHERE:
        if (0 != (entry->lexical_flags & ENTRY_LEX_PLACE)) score += 0.14;
        if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) score += 0.04;
        break;
      case QUERY_WHEN:
        if (0 != (entry->lexical_flags & ENTRY_LEX_NUMBER)) score += 0.14;
        break;
      case QUERY_WHY:
        if (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) score += 0.18;
        break;
      case QUERY_HOW:
        if (0 != (entry->lexical_flags & ENTRY_LEX_METHOD)) score += 0.14;
        if (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) score += 0.06;
        break;
      case QUERY_YESNO:
        score += 0.04;
        break;
      default:
        break;
    }
  }
  entry_words = count_words((const unsigned char*)entry->text, entry->text_len);
  if (SCALE_SENTENCE == entry->scale) score += 0.10;
  score += 0.02 * matches;
  if (entry_words > 80) score -= 0.08;
  if (entry_words > 160) score -= 0.12;
  return score;
}


static int answer_select(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX])
{
  libxs_lexeme_stream_t query;
  const void* key = NULL;
  size_t cursor = 0, key_size = 0;
  void* value;
  int answer_count = 0;
  int slot;
  int query_type = QUERY_GENERIC;
  int result = 0;
  int limit = budget;
  /**
   * Evidence selection scores term overlap, which is blind to polarity: a
   * negated question overlaps the affirmative sentence on every content word
   * and would rank it first. Abstain instead -- the corpus states positives,
   * so a complement is not answerable from it by selection either.
   */
  int negated = answer_query_is_negated(query_text, query_len);
  const libxs_predict_t* predictor = answer_model;
  libxs_predict_t* query_predictor = NULL;
  char query_section[ENTRY_SECTION_MAX];
  int query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  const char* rank_query_text = query_text;
  size_t rank_query_len = query_len;
  char query_be_word[64];
  int query_be_upper = 0;
  int query_be_len = 0;
  if (query_section_len > 0) {
    size_t body_pos = 0;
    while (body_pos < query_len && ',' != query_text[body_pos]) ++body_pos;
    if (body_pos < query_len) {
      ++body_pos;
      while (body_pos < query_len
        && 0 != isspace((unsigned char)query_text[body_pos])) ++body_pos;
      if (body_pos < query_len) {
        rank_query_text = query_text + body_pos;
        rank_query_len = query_len - body_pos;
      }
    }
  }
  libxs_lexeme_stream_init(&query);
  if (limit < 1) limit = 1;
  if (limit > ANSWER_MAX) limit = ANSWER_MAX;
  for (slot = 0; slot < ANSWER_MAX; ++slot) {
    entries[slot] = NULL;
    scores[slot] = 0.0;
  }
  if (0 == negated && NULL != lexicon && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)rank_query_text, rank_query_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 1))
  {
    query_type = query_type_of(&query, lexicon);
    query_be_len = answer_query_be_word(rank_query_text, rank_query_len,
      query_be_word, (int)sizeof(query_be_word), &query_be_upper);
    if (NULL == predictor) {
      query_predictor = answer_predict_build(corpus, &query, lexicon,
        rules, nrules, query_type, profile);
      predictor = query_predictor;
    }
    value = libxs_registry_begin_length(corpus, &key, &key_size, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      size_t entry_size = (NULL != key)
        ? libxs_registry_value_size(corpus, key, key_size, NULL) : sizeof(*entry);
      double base_score = 0.0;
      double score = base_score;
      double identity_score = 0.0;
      int relation_ranked = 0;
      answer_relation_match_t relation_match;
      double inputs[ANSWER_PREDICT_INPUTS];
      if (query_section_len > 0
        && 0 == corpus_entry_section_match(entry, entry_size,
          query_section, query_section_len))
      {
        value = libxs_registry_next_length(corpus, &key, &key_size, &cursor);
        continue;
      }
      if (0 != query_type_prefers_sentence(query_type)
        && SCALE_SENTENCE == entry->scale
        && (0 == text_starts_sentence(entry->text, entry->text_len)
          || 0 == text_ends_sentence(entry->text, entry->text_len)))
      {
        value = libxs_registry_next(corpus, &key, &cursor);
        continue;
      }
      if (QUERY_WHO == query_type && query_be_len > 0
        && 0 == query_be_upper
        && 0 == answer_relation_match_query(rank_query_text, rank_query_len,
          query_type, entry, &relation_match))
      {
        value = libxs_registry_next(corpus, &key, &cursor);
        continue;
      }
      base_score = lexical_score(&query, lexicon, rules, nrules, entry,
        entry_size, query_type);
      score = base_score;
      identity_score = answer_identity_score(rank_query_text, rank_query_len,
        query_type, entry);
      if (identity_score > base_score) {
        base_score = identity_score;
        score = base_score;
        if (identity_score > 1.0) relation_ranked = 1;
      }
      if (base_score >= ANSWER_MIN_SCORE && entry->text_len >= 16) {
        if (EXIT_SUCCESS == answer_features(&query, entry, entry_size,
          query_type, inputs))
        {
          score = answer_predict_score(predictor, inputs, base_score);
          if (base_score > 1.0 && score < base_score) score = base_score;
        }
        if (0 != relation_ranked) score += base_score;
        if (0 != query_type_prefers_sentence(query_type)) {
          score += (SCALE_SENTENCE == entry->scale) ? 0.12 : -0.18;
        }
        for (slot = 0; slot < limit; ++slot) {
          if (NULL == entries[slot] || score > scores[slot]) {
            int move_slot;
            for (move_slot = limit - 1; move_slot > slot; --move_slot) {
              entries[move_slot] = entries[move_slot - 1];
              scores[move_slot] = scores[move_slot - 1];
            }
            entries[slot] = entry;
            scores[slot] = score;
            break;
          }
        }
      }
      value = libxs_registry_next(corpus, &key, &cursor);
    }
  }
  libxs_predict_destroy(query_predictor);
  for (slot = 0; slot < limit && NULL != entries[slot]; ++slot) {
    ++answer_count;
  }
  if (answer_count > 0) result = answer_count;
  libxs_lexeme_stream_release(&query);
  return result;
}


static int answer_reply_role(char* output, size_t output_size,
  const char* name, int name_len, const char* role)
{
  static const char middle[] = " is the ";
  int result = EXIT_FAILURE;
  size_t role_len;
  size_t pos = 0;
  if (NULL == output || 0 == output_size || NULL == name || name_len <= 0
    || NULL == role) return EXIT_FAILURE;
  role_len = strlen(role);
  if ((size_t)name_len + sizeof(middle) - 1 + role_len + 2 < output_size) {
    memcpy(output + pos, name, (size_t)name_len);
    pos += (size_t)name_len;
    memcpy(output + pos, middle, sizeof(middle) - 1);
    pos += sizeof(middle) - 1;
    memcpy(output + pos, role, role_len);
    pos += role_len;
    output[pos++] = '.';
    output[pos] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static void answer_strip_heading_prefix(const char** text, int* text_len)
{
  int pos;
  int prefix_words = 0;
  int in_word = 0;
  if (NULL == text || NULL == *text || NULL == text_len || *text_len <= 0) {
    return;
  }
  for (pos = 0; pos < *text_len; ++pos) {
    unsigned char ch = (unsigned char)(*text)[pos];
    if (0 != islower(ch) || '.' == ch || ',' == ch || ';' == ch
      || ':' == ch || '!' == ch || '?' == ch)
    {
      break;
    }
    if (0 != isupper(ch)) {
      if (0 == in_word) {
        ++prefix_words;
        in_word = 1;
      }
    }
    else if (0 != isspace(ch)) {
      in_word = 0;
    }
  }
  if (prefix_words >= 2 && pos < *text_len && pos > 0) {
    const char* next = *text + pos;
    int remaining = *text_len - pos;
    int prev_end = pos;
    int prev_start;
    while (prev_end > 0
      && 0 != isspace((unsigned char)(*text)[prev_end - 1]))
    {
      --prev_end;
    }
    prev_start = prev_end;
    while (prev_start > 0
      && 0 == isspace((unsigned char)(*text)[prev_start - 1]))
    {
      --prev_start;
    }
    if ((1 == prev_end - prev_start
        && 'A' == (*text)[prev_start])
      || (2 == prev_end - prev_start
        && 'A' == (*text)[prev_start] && 'N' == (*text)[prev_start + 1]))
    {
      next = *text + prev_start;
      remaining = *text_len - prev_start;
    }
    while (remaining > 0 && 0 != isspace((unsigned char)*next)) {
      ++next;
      --remaining;
    }
    if (remaining > 0) {
      *text = next;
      *text_len = remaining;
    }
  }
}


static size_t answer_append_clean(char* output, size_t output_size,
  size_t output_pos, const char* text, int text_len)
{
  int text_pos;
  int last_space = 1;
  if (NULL == output || 0 == output_size || NULL == text) return output_pos;
  if (text_len < 0) text_len = (int)strlen(text);
  for (text_pos = 0; text_pos < text_len && output_pos + 1 < output_size;
    ++text_pos)
  {
    unsigned char ch = (unsigned char)text[text_pos];
    if (0 != isspace(ch)) {
      if (0 == last_space) output[output_pos++] = ' ';
      last_space = 1;
    }
    else {
      output[output_pos++] = (char)ch;
      last_space = 0;
    }
  }
  output[output_pos] = '\0';
  return output_pos;
}


static int answer_frame_after(char* output, size_t output_size,
  size_t* output_pos, const char* text, int text_len, const char* marker)
{
  int result = EXIT_FAILURE;
  int start, end;
  if (NULL == output_pos || NULL == marker) return EXIT_FAILURE;
  start = text_find_ci(text, text_len, marker);
  if (start >= 0) {
    start += (int)strlen(marker);
    while (start < text_len && 0 != isspace((unsigned char)text[start])) {
      ++start;
    }
    end = start;
    while (end < text_len && '.' != text[end] && '!' != text[end]
      && '?' != text[end]) ++end;
    if (end > start) {
      *output_pos = answer_append_clean(output, output_size, *output_pos,
        text + start, end - start);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_frame_keywords_after(char* output, size_t output_size,
  size_t* output_pos, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  const char* marker)
{
  int result = EXIT_FAILURE;
  int start, end;
  libxs_lexeme_stream_t stream;
  unsigned int used[32];
  size_t used_size = 0;
  size_t lexeme_pos;
  int wrote = 0;
  libxs_lexeme_stream_init(&stream);
  if (NULL == output_pos || NULL == lexicon || NULL == rules
    || nrules <= 0 || NULL == marker) return EXIT_FAILURE;
  start = text_find_ci(text, text_len, marker);
  if (start >= 0) {
    start += (int)strlen(marker);
    while (start < text_len && 0 != isspace((unsigned char)text[start])) {
      ++start;
    }
    end = start;
    while (end < text_len && '.' != text[end] && '!' != text[end]
      && '?' != text[end]) ++end;
    if (end > start && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, (const unsigned char*)text + start, (size_t)(end - start),
      rules, nrules, answer_lexnorms, answer_lexnorms_size, 1))
    {
      for (lexeme_pos = 0; lexeme_pos < stream.size && used_size < 32;
        ++lexeme_pos)
      {
        const libxs_lexeme_t* lexeme = stream.data + lexeme_pos;
        int duplicate = 0;
        size_t used_pos;
        if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
        {
          for (used_pos = 0; used_pos < used_size && 0 == duplicate;
            ++used_pos)
          {
            if (used[used_pos] == lexeme->id) duplicate = 1;
          }
          if (0 == duplicate) {
            int token_len = 0;
            const char* token = libxs_lexicon_text(lexicon, lexeme->id,
              &token_len, NULL);
            if (NULL != token && token_len > 0) {
              if (0 != wrote) {
                *output_pos = answer_append_clean(output, output_size,
                  *output_pos, ", ", -1);
              }
              *output_pos = answer_append_clean(output, output_size,
                *output_pos, token, token_len);
              used[used_size++] = lexeme->id;
              wrote = 1;
            }
          }
        }
      }
      if (0 != wrote) result = EXIT_SUCCESS;
    }
  }
  libxs_lexeme_stream_release(&stream);
  return result;
}


static int answer_bridge_expand_reply(const answer_bridge_t* bridge,
  const char* text, int text_len, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, char* output, size_t output_size)
{
  int result = EXIT_SUCCESS;
  size_t output_pos = 0;
  const char* cursor;
  if (NULL == bridge || NULL == bridge->reply || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  cursor = bridge->reply;
  output[0] = '\0';
  while ('\0' != *cursor && output_pos + 1 < output_size
    && EXIT_SUCCESS == result)
  {
    const char* open = strchr(cursor, '{');
    if (NULL == open) {
      output_pos = answer_append_clean(output, output_size, output_pos,
        cursor, -1);
      cursor += strlen(cursor);
    }
    else {
      const char* close = strchr(open + 1, '}');
      output_pos = answer_append_clean(output, output_size, output_pos,
        cursor, (int)(open - cursor));
      if (NULL == close) {
        result = EXIT_FAILURE;
      }
      else if (0 == strncmp(open + 1, "after:", 6)) {
        char marker[128];
        int marker_len = (int)(close - (open + 7));
        if (marker_len > 0 && marker_len < (int)sizeof(marker)) {
          memcpy(marker, open + 7, (size_t)marker_len);
          marker[marker_len] = '\0';
          result = answer_frame_after(output, output_size, &output_pos,
            text, text_len, marker);
        }
        else result = EXIT_FAILURE;
      }
      else if (0 == strncmp(open + 1, "keywords-after:", 15)) {
        char marker[128];
        int marker_len = (int)(close - (open + 16));
        if (marker_len > 0 && marker_len < (int)sizeof(marker)) {
          memcpy(marker, open + 16, (size_t)marker_len);
          marker[marker_len] = '\0';
          result = answer_frame_keywords_after(output, output_size,
            &output_pos, lexicon, rules, nrules, text, text_len, marker);
        }
        else result = EXIT_FAILURE;
      }
      else {
        result = EXIT_FAILURE;
      }
      cursor = close + 1;
    }
  }
  if (EXIT_SUCCESS == result) {
    while (output_pos > 0 && 0 != isspace((unsigned char)output[output_pos - 1])) {
      --output_pos;
    }
    if (output_pos > 0 && '.' != output[output_pos - 1]
      && '?' != output[output_pos - 1] && '!' != output[output_pos - 1]
      && output_pos + 1 < output_size)
    {
      output[output_pos++] = '.';
    }
    output[output_pos] = '\0';
  }
  return result;
}


static int answer_reply_what_is(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  const char* target = NULL;
  int target_len = 0;
  size_t lexeme_pos;
  if (NULL == query || NULL == lexicon || NULL == text || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  if (0 == lexeme_stream_has_text(query, lexicon, "what")
    || 0 == lexeme_stream_has_text(query, lexicon, "is"))
  {
    return EXIT_FAILURE;
  }
  for (lexeme_pos = 0; lexeme_pos < query->size; ++lexeme_pos) {
    const libxs_lexeme_t* lexeme = query->data + lexeme_pos;
    if (0 != (lexeme->flags & LIBXS_LEXEME_WORD)
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP)
      && 0 == lexeme_text_is(lexicon, lexeme, "what")
      && 0 == lexeme_text_is(lexicon, lexeme, "is"))
    {
      target = libxs_lexicon_text(lexicon, lexeme->id, &target_len, NULL);
    }
  }
  if (NULL != target && target_len > 0) {
    char target_buf[64];
    int pos;
    if (target_len < (int)sizeof(target_buf)) {
      memcpy(target_buf, target, (size_t)target_len);
      target_buf[target_len] = '\0';
      pos = text_find_ci(text, text_len, target_buf);
      if (pos >= 0) {
        int end = pos + target_len;
        while (end < text_len && ',' != text[end] && '.' != text[end]
          && '!' != text[end] && '?' != text[end]) ++end;
        if (end > pos) {
          size_t output_pos = answer_append_clean(output, output_size, 0,
            text + pos, end - pos);
          if (output_pos > 0) {
            output[0] = (char)toupper((unsigned char)output[0]);
            if ('.' != output[output_pos - 1] && '?' != output[output_pos - 1]
              && '!' != output[output_pos - 1] && output_pos + 1 < output_size)
            {
              output[output_pos++] = '.';
              output[output_pos] = '\0';
            }
            result = EXIT_SUCCESS;
          }
        }
      }
    }
  }
  return result;
}


static int answer_reply(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  int query_type = QUERY_GENERIC;
  libxs_lexeme_stream_t query;
  const answer_bridge_t* bridge = NULL;
  const char* text;
  int text_len;
  char be_word[64];
  int be_upper = 0;
  int be_len = 0;
  answer_relation_match_t relation_match;
  libxs_lexeme_stream_init(&query);
  if (NULL == query_text || NULL == entry || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  text = entry->text;
  text_len = entry->text_len;
  while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
    ++text;
    --text_len;
  }
  while (text_len > 0 && 0 != isspace((unsigned char)text[text_len - 1])) {
    --text_len;
  }
  answer_strip_heading_prefix(&text, &text_len);
  if (NULL != lexicon && NULL != rules && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)query_text, query_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 1))
  {
    query_type = query_type_of(&query, lexicon);
    bridge = answer_bridge_match(&query, lexicon, entry);
    be_len = answer_query_be_word(query_text, query_len, be_word,
      (int)sizeof(be_word), &be_upper);
  }
  if (NULL != bridge && NULL != bridge->reply) {
    result = answer_bridge_expand_reply(bridge, text, text_len, lexicon,
      rules, nrules, output, output_size);
  }
  else if (QUERY_WHO == query_type && be_len > 0 && 0 == be_upper
    && 0 != answer_relation_match_query(query_text, query_len, query_type,
      entry, &relation_match))
  {
    result = answer_relation_reply(&relation_match, output, output_size);
  }
  else if (QUERY_WHAT == query_type) {
    result = answer_reply_what_is(&query, lexicon, text, text_len,
      output, output_size);
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


static int answer_evidence_sentence(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  libxs_lexeme_stream_t query;
  int best_start = -1, best_end = -1, best_score = 0;
  int sent_start = 0;
  int text_len;
  const char* text;
  libxs_lexeme_stream_init(&query);
  if (NULL == query_text || NULL == entry || NULL == lexicon || NULL == rules
    || nrules <= 0 || NULL == output || 0 == output_size) return EXIT_FAILURE;
  text = entry->text;
  text_len = entry->text_len;
  while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
    ++text;
    --text_len;
  }
  while (text_len > 0 && 0 != isspace((unsigned char)text[text_len - 1])) {
    --text_len;
  }
  answer_strip_heading_prefix(&text, &text_len);
  if (EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
    (const unsigned char*)query_text, query_len, rules, nrules,
    answer_lexnorms, answer_lexnorms_size, 1))
  {
    int pos;
    for (pos = 0; pos <= text_len; ++pos) {
      if ((pos == text_len
          && 0 != text_ends_sentence(text + sent_start,
            text_len - sent_start))
        || '.' == text[pos] || '!' == text[pos]
        || '?' == text[pos])
      {
        int sent_end = (pos < text_len) ? pos + 1 : pos;
        int score = 0;
        size_t lexeme_pos;
        while (sent_end < text_len) {
          size_t close_size = text_closer_size((const unsigned char*)text,
            (size_t)text_len, (size_t)sent_end);
          if (0 == close_size) break;
          sent_end += (int)close_size;
        }
        for (lexeme_pos = 0; lexeme_pos < query.size; ++lexeme_pos) {
          const libxs_lexeme_t* lexeme = query.data + lexeme_pos;
          if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
          {
            int term_len = 0;
            const char* term = libxs_lexicon_text(lexicon, lexeme->id,
              &term_len, NULL);
            if (NULL != term && term_len > 0 && term_len < 64) {
              char term_buf[64];
              memcpy(term_buf, term, (size_t)term_len);
              term_buf[term_len] = '\0';
              if (0 != text_contains_ci(text + sent_start,
                sent_end - sent_start, term_buf)) ++score;
            }
          }
        }
        if (score > best_score) {
          best_score = score;
          best_start = sent_start;
          best_end = sent_end;
        }
        sent_start = sent_end;
        while (sent_start < text_len
          && 0 != isspace((unsigned char)text[sent_start])) ++sent_start;
      }
    }
  }
  if (best_score > 0 && best_end > best_start) {
    answer_append_clean(output, output_size, 0, text + best_start,
      best_end - best_start);
    result = EXIT_SUCCESS;
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


/**
 * Non-zero if the query carries a negator, per the caller-owned `negate|`
 * rules. The extractors answer affirmative questions only: they find what the
 * corpus asserts, never what it denies, and a complement ("who is NOT the
 * witch") is not groundable from evidence that states only positives. The
 * vocabulary stays in the rule file -- no negation words in the source -- and
 * with no rules loaded the check is inert, exactly like the other rule kinds.
 */
static int answer_query_is_negated(const char* query_text, size_t query_len)
{
  int result = 0;
  size_t rule_pos;
  libxs_lexeme_stream_t query;
  libxs_lexeme_stream_init(&query);
  /**
   * RULE ORDER: normalization first, interpretation second. The negators are
   * matched against the ENCODED query, so any `norm|` rewriting has already
   * been applied and the two rule kinds compose in a defined order instead of
   * inspecting different representations of the same text. A consequence worth
   * stating: a `norm|` rule must not map a negator onto its affirmative form,
   * or it would erase the polarity this check depends on.
   */
  if (NULL != answer_negate_lexicon && answer_negate_nrules > 0
    && NULL != query_text && query_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(answer_negate_lexicon,
      &query, (const unsigned char*)query_text, query_len,
      answer_negate_rules, answer_negate_nrules,
      answer_lexnorms, answer_lexnorms_size, 0))
  {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size && 0 == result;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (RELATION_RULE_NEGATE == rule->kind
        && 0 != lexeme_stream_has_text(&query, answer_negate_lexicon,
          rule->term))
      {
        result = 1;
      }
    }
  }
  else { /* no lexicon available yet: fall back to raw-text matching */
    result = answer_relation_rule_has_term(RELATION_RULE_NEGATE, query_text,
      (int)query_len);
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  answer_fact_section_set(NULL, 0);
  if (0 == answer_query_is_negated(query_text, query_len)
    && (EXIT_SUCCESS == answer_relation_fact_reply(query_text, query_len,
      output, output_size)
    || EXIT_SUCCESS == answer_relation_aggregate_reply(corpus, query_text,
      query_len, output, output_size)
    || EXIT_SUCCESS == answer_identity_fact_reply(query_text, query_len,
      output, output_size)
    || EXIT_SUCCESS == answer_describe_fact_reply(query_text, query_len,
      output, output_size)
    || EXIT_SUCCESS == answer_docdef_fact_reply(query_text, query_len,
      output, output_size)))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}


/**
 * Print the section an answer came from, when the corpus supplies one. Sections
 * are the story titles or Markdown headings recorded per entry at ingest, so a
 * citation is only emitted when the corpus actually carries that structure --
 * never invented, and silently omitted for flat text.
 */
static void answer_print_citation(const char* section, int section_len)
{
  if (NULL != section && section_len > 0 && '\0' != section[0]) {
    printf("citation: %.*s\n", section_len, section);
  }
}


static void answer_fact_section_set(const char* section, int section_len)
{
  answer_fact_section_len = 0;
  answer_fact_section[0] = '\0';
  if (NULL != section && section_len > 0
    && section_len < (int)sizeof(answer_fact_section))
  {
    memcpy(answer_fact_section, section, (size_t)section_len);
    answer_fact_section[section_len] = '\0';
    answer_fact_section_len = section_len;
  }
}


static int answer_hier_rescore_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_RESCORE");
    cached = (NULL != env && '0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Build the expert bank used to rescore answers, and publish it for
 * answer_hier_reorder. Returns NULL (and leaves the reorder inactive) unless the
 * rescorer is enabled, so the default path pays neither the training time nor
 * the memory.
 */
static converse_hier_t* answer_hier_build(const libxs_registry_t* corpus)
{
  converse_hier_t* result = NULL;
  if (0 != answer_hier_rescore_on()) {
    result = converse_hier_build(corpus, 0, 0, ngram_maxorder());
    answer_hier_model = result;
  }
  return result;
}


/**
 * Reorder the selected candidates by the hierarchical model's conditional code
 * length, -log2 P(candidate | query), normalized per byte so a long sentence is
 * not penalized for its length. Selection stays in charge of WHICH sentences are
 * admitted -- this only reorders them -- so the abstention discipline is
 * untouched: a candidate promoted here was already attested and already cleared
 * the selection threshold. Returns the (unchanged) candidate count.
 *
 * The reorder is opt-in and off by default: it is a measurement of whether a
 * better-calibrated likelihood carries downstream, and the QA harness is the
 * gate. Without a built model, or when a candidate cannot be scored, the
 * selection order is preserved exactly.
 */
static int answer_hier_reorder(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX],
  int answer_count)
{
  if (NULL != answer_hier_model && 1 < answer_count) {
    const char* candidates[ANSWER_MAX];
    int lengths[ANSWER_MAX];
    double bits[ANSWER_MAX];
    int slot, nvalid = 0;
    for (slot = 0; slot < answer_count; ++slot) {
      candidates[slot] = (NULL != entries[slot]) ? entries[slot]->text : NULL;
      lengths[slot] = (NULL != entries[slot]) ? entries[slot]->text_len : 0;
    }
    if (EXIT_SUCCESS == converse_hier_rescore(answer_hier_model, query_text,
      (int)query_len, candidates, lengths, answer_count, bits))
    {
      for (slot = 0; slot < answer_count; ++slot) {
        if (0 < lengths[slot]) ++nvalid;
      }
    }
    /**
     * Only reorder when EVERY candidate scored: a partial ranking would mix two
     * incomparable orderings, and preferring whichever subset the model happened
     * to handle is not a measurement of the model.
     */
    if (nvalid == answer_count) {
      const corpus_entry_t* was_first = entries[0];
      ++answer_hier_nreorder;
      for (slot = 1; slot < answer_count; ++slot) {
        const corpus_entry_t* entry = entries[slot];
        const double bpc = bits[slot];
        const double score = scores[slot];
        int probe = slot;
        while (0 < probe && bpc > bits[probe - 1]) {
          entries[probe] = entries[probe - 1];
          scores[probe] = scores[probe - 1];
          bits[probe] = bits[probe - 1];
          --probe;
        }
        entries[probe] = entry;
        scores[probe] = score;
        bits[probe] = bpc;
      }
      if (was_first != entries[0]) {
        ++answer_hier_nchanged;
        if (1 < answer_hier_rescore_on()) {
          fprintf(stderr, "  rescore[%.*s]\n    was: [%d B] %.*s\n"
            "    now: [%d B] %.*s (%.3f bpc)\n", (int)query_len, query_text,
            was_first->text_len,
            (was_first->text_len < 90) ? was_first->text_len : 90,
            was_first->text, entries[0]->text_len,
            (entries[0]->text_len < 90) ? entries[0]->text_len : 90,
            entries[0]->text, bits[0]);
        }
      }
    }
  }
  return answer_count;
}


static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  char* out_reply, size_t out_size)
{
  const corpus_entry_t* entries[ANSWER_MAX];
  double scores[ANSWER_MAX];
  int answer_count;
  int slot;
  char reply[COMPOSE_MAXTEXT];
  if (NULL != out_reply && out_size > 0) out_reply[0] = '\0';
  if (EXIT_SUCCESS == answer_fact_reply(corpus, query_text, query_len,
    reply, sizeof(reply)))
  {
    printf("%s\n", reply);
    answer_print_citation(answer_fact_section, answer_fact_section_len);
    if (NULL != out_reply && out_size > 0) {
      size_t rn = strlen(reply);
      if (rn >= out_size) rn = out_size - 1;
      memcpy(out_reply, reply, rn);
      out_reply[rn] = '\0';
    }
    return 1;
  }
  answer_count = answer_select(corpus, query_text, query_len, budget,
    lexicon, rules, nrules, answer_model, profile, entries, scores);
  answer_count = answer_hier_reorder(query_text, query_len, entries, scores,
    answer_count);
  if (answer_count > 0 && NULL != entries[0]
    && EXIT_SUCCESS == answer_reply(query_text, query_len, entries[0],
      lexicon, rules, nrules, reply, sizeof(reply)))
  {
    printf("%s\n", reply);
    answer_print_citation(entries[0]->section, entries[0]->section_len);
    if (NULL != out_reply && out_size > 0) {
      size_t rn = strlen(reply);
      if (rn >= out_size) rn = out_size - 1;
      memcpy(out_reply, reply, rn);
      out_reply[rn] = '\0';
    }
    LIBXS_UNUSED(scores);
    return 1;
  }
  for (slot = 0; slot < answer_count && NULL != entries[slot]; ++slot) {
    const char* text = entries[slot]->text;
    int text_len = entries[slot]->text_len;
    if (EXIT_SUCCESS == answer_evidence_sentence(query_text, query_len,
      entries[slot], lexicon, rules, nrules, reply, sizeof(reply)))
    {
      if (slot > 0) printf("\n");
      printf("%s\n", reply);
      answer_print_citation(entries[slot]->section,
        entries[slot]->section_len);
      continue;
    }
    while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
      ++text;
      --text_len;
    }
    while (text_len > 0
      && 0 != isspace((unsigned char)text[text_len - 1])) --text_len;
    answer_strip_heading_prefix(&text, &text_len);
    if (text_len > 0 && (SCALE_SENTENCE != entries[slot]->scale
        || (0 != text_starts_sentence(text, text_len)
          && 0 != text_ends_sentence(text, text_len))))
    {
      if (slot > 0) printf("\n");
      printf("%.*s\n", text_len, text);
    }
  }
  LIBXS_UNUSED(scores);
  return (answer_count > 0) ? 1 : 0;
}


static int text_find_ci(const char* text, int text_len, const char* term)
{
  int result = -1;
  int term_len, text_pos;
  if (NULL == text || NULL == term || text_len <= 0) return -1;
  term_len = (int)strlen(term);
  if (term_len <= 0 || term_len > text_len) return -1;
  for (text_pos = 0; text_pos <= text_len - term_len && result < 0;
    ++text_pos)
  {
    int term_pos, match = 1;
    for (term_pos = 0; term_pos < term_len && 0 != match; ++term_pos) {
      unsigned char a = (unsigned char)text[text_pos + term_pos];
      unsigned char b = (unsigned char)term[term_pos];
      if (tolower(a) != tolower(b)) match = 0;
    }
    if (0 != match) result = text_pos;
  }
  return result;
}


static int text_contains_ci(const char* text, int text_len, const char* term)
{
  return (text_find_ci(text, text_len, term) >= 0) ? 1 : 0;
}


static int text_find_word_ci(const char* text, int text_len, const char* term)
{
  int result = -1;
  int term_len;
  int pos = 0;
  if (NULL == text || NULL == term || text_len <= 0) return -1;
  term_len = (int)strlen(term);
  if (term_len <= 0 || term_len > text_len) return -1;
  while (pos <= text_len - term_len && result < 0) {
    int found = text_find_ci(text + pos, text_len - pos, term);
    if (found < 0) break;
    pos += found;
    if ((0 == pos || 0 == isalnum((unsigned char)text[pos - 1]))
      && (pos + term_len >= text_len
        || 0 == isalnum((unsigned char)text[pos + term_len])))
    {
      result = pos;
    }
    pos += term_len;
  }
  return result;
}


static int text_contains_word_ci(const char* text, int text_len,
  const char* term)
{
  int result = 0;
  int term_len;
  int pos = 0;
  if (NULL == text || NULL == term || text_len <= 0) return 0;
  term_len = (int)strlen(term);
  if (term_len <= 0 || term_len > text_len) return 0;
  while (pos <= text_len - term_len && 0 == result) {
    int found = text_find_ci(text + pos, text_len - pos, term);
    if (found < 0) break;
    pos += found;
    if ((0 == pos || 0 == isalnum((unsigned char)text[pos - 1]))
      && (pos + term_len >= text_len
        || 0 == isalnum((unsigned char)text[pos + term_len])))
    {
      result = 1;
    }
    pos += term_len;
  }
  return result;
}


static char* eval_trim(char* text)
{
  char* result = text;
  char* end;
  if (NULL != result) {
    while ('\0' != *result && 0 != isspace((unsigned char)*result)) ++result;
    end = result + strlen(result);
    while (end > result && 0 != isspace((unsigned char)end[-1])) --end;
    *end = '\0';
  }
  return result;
}


static int eval_parse_line(char* line, char* fields[4])
{
  int result = EXIT_FAILURE;
  char* cursor;
  int field_pos;
  if (NULL != fields) {
    for (field_pos = 0; field_pos < 4; ++field_pos) fields[field_pos] = NULL;
  }
  if (NULL != line && NULL != fields) {
    cursor = eval_trim(line);
    if ('\0' != *cursor && '#' != *cursor) {
      for (field_pos = 0; field_pos < 3 && NULL != cursor; ++field_pos) {
        char* sep = strchr(cursor, '|');
        if (NULL != sep) {
          *sep = '\0';
          fields[field_pos] = eval_trim(cursor);
          cursor = sep + 1;
        }
        else cursor = NULL;
      }
      if (NULL != cursor) fields[3] = eval_trim(cursor);
      if (NULL != fields[0] && '\0' != fields[0][0]
        && NULL != fields[1] && NULL != fields[2])
      {
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static int eval_terms_empty(const char* spec)
{
  int result = 1;
  if (NULL != spec) {
    while ('\0' != *spec && 0 != isspace((unsigned char)*spec)) ++spec;
    result = ('\0' == *spec) ? 1 : 0;
  }
  return result;
}


static int eval_terms_match_text(const char* text, int text_len,
  const char* spec)
{
  int result = 1;
  int count = 0;
  int term_pos = 0;
  if (NULL == text || text_len <= 0) result = 0;
  while (0 != result && NULL != spec && NULL != text && text_len > 0) {
    int term_len = 0;
    const char* token = libxs_strtoken(spec, ",", term_pos, &term_len);
    char term[128];
    if (NULL == token) break;
    if (term_len > 0) {
      if (term_len >= (int)sizeof(term)) result = 0;
      else {
        memcpy(term, token, (size_t)term_len);
        term[term_len] = '\0';
        if (0 == text_contains_ci(text, text_len, term)) result = 0;
      }
      ++count;
    }
    ++term_pos;
  }
  if (0 == count && 0 == eval_terms_empty(spec)) result = 0;
  return result;
}


static int eval_terms_match_answers(const corpus_entry_t* entries[],
  int nanswers, const char* spec, int top_only)
{
  int result = 1;
  int count = 0;
  int term_pos = 0;
  if (NULL == entries || nanswers <= 0) result = 0;
  while (0 != result && NULL != spec && NULL != entries && nanswers > 0) {
    int term_len = 0;
    const char* token = libxs_strtoken(spec, ",", term_pos, &term_len);
    char term[128];
    if (NULL == token) break;
    if (term_len > 0) {
      int found = 0;
      int answer_pos;
      if (term_len >= (int)sizeof(term)) result = 0;
      else {
        memcpy(term, token, (size_t)term_len);
        term[term_len] = '\0';
        for (answer_pos = 0; answer_pos < nanswers && 0 == found;
          ++answer_pos)
        {
          if ((0 == top_only || 0 == answer_pos)
            && NULL != entries[answer_pos]
            && 0 != text_contains_ci(entries[answer_pos]->text,
              entries[answer_pos]->text_len, term))
          {
            found = 1;
          }
        }
        if (0 == found) result = 0;
      }
      ++count;
    }
    ++term_pos;
  }
  if (0 == count && 0 == eval_terms_empty(spec)) result = 0;
  return result;
}


static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile)
{
  int result = EXIT_FAILURE;
  int npass = 0, ntop = 0, nany = 0, nreply = 0, nfact = 0;
  int ncases = 0;
  int have_facts = (0 != answer_relation_facts_size
    || 0 != answer_docdef_facts_size) ? 1 : 0;
  FILE* file;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL == corpus || NULL == lexicon || NULL == rules) return EXIT_FAILURE;
  file = fopen(converse_path_eval, "r");
  if (NULL == file) {
    fprintf(stderr, "eval: no %s file found\n", converse_path_eval);
  }
  while (NULL != file) {
    char line[EVAL_LINE_MAX];
    char* fields[4];
    const corpus_entry_t* entries[ANSWER_MAX];
    double scores[ANSWER_MAX];
    int nanswers;
    int top_pass;
    int any_pass;
    int reply_pass;
    int fact_pass;
    int fact_checked;
    int pass;
    char reply[COMPOSE_MAXTEXT];
    char rewritten[COMPOSE_MAXTEXT];
    const char* qtext;
    size_t qlen;
    if (NULL == fgets(line, (int)sizeof(line), file)) break;
    if (EXIT_SUCCESS != eval_parse_line(line, fields)) continue;
    ++ncases;
    if ('>' == fields[0][0]) {
      char* cont = eval_trim(fields[0] + 1);
      if (EXIT_SUCCESS == conv_rewrite(cont, strlen(cont), rewritten,
        sizeof(rewritten)))
      {
        qtext = rewritten;
      }
      else qtext = cont;
    }
    else {
      conv_reset();
      qtext = fields[0];
    }
    qlen = strlen(qtext);
    fact_pass = 1;
    fact_checked = (have_facts && NULL != fields[3]
      && 0 == eval_terms_empty(fields[3])
      && 0 != strcmp(fields[3], EVAL_RULE_GOVERNED)) ? 1 : 0;
    if (0 != fact_checked) {
      if (EXIT_SUCCESS == answer_fact_reply(corpus, qtext, qlen,
        reply, sizeof(reply)))
      {
        fact_pass = eval_terms_match_text(reply, (int)strlen(reply),
          fields[3]);
      }
      else fact_pass = 0;
    }
    conv_remember(qtext, qlen);
    nanswers = answer_select(corpus, qtext, qlen,
      ANSWER_MAX, lexicon, rules, nrules,
      answer_model, profile, entries, scores);
    nanswers = answer_hier_reorder(qtext, qlen, entries, scores, nanswers);
    top_pass = eval_terms_match_answers(entries, nanswers, fields[1], 1);
    any_pass = eval_terms_match_answers(entries, nanswers, fields[1], 0);
    reply_pass = 1;
    LIBXS_UNUSED(scores);
    if (0 != eval_terms_empty(fields[1])) {
      if (0 != fact_checked) {
        pass = fact_pass;
        fprintf(stdout, "%s fact %s\n", (0 != fact_pass) ? "PASS" : "FAIL",
          fields[0]);
        if (0 != fact_pass) ++nfact;
      }
      else if (NULL != fields[3] && 0 == eval_terms_empty(fields[3])
        && 0 != strcmp(fields[3], EVAL_RULE_GOVERNED))
      {
        fprintf(stdout, "SKIP fact %s\n", fields[0]);
        --ncases;
        continue;
      }
      /**
       * A lone EVAL_RULE_GOVERNED marker in the fact field means the expected
       * abstention depends on loaded rules (negation, for instance, is a rule
       * kind): check it when rules are present, skip it when they are not.
       * This keeps one fixture valid in both configurations -- the property
       * that proves no language vocabulary is compiled into the source --
       * without pretending a rule-driven capability works without rules.
       */
      else if (NULL != fields[3]
        && 0 == strcmp(fields[3], EVAL_RULE_GOVERNED)
        && 0 == answer_relation_rules_size)
      {
        fprintf(stdout, "SKIP rules %s\n", fields[0]);
        --ncases;
        continue;
      }
      else {
        /**
         * An abstention case must abstain on EVERY path an interactive query
         * would take, not merely on evidence selection: the fact and
         * definition resolvers answer first and are the ones that can
         * fabricate, so a check that only looked at answer_select would pass
         * a question the engine actually answers.
         */
        char probe[COMPOSE_MAXTEXT];
        int answered = (0 != nanswers) ? 1 : 0;
        if (EXIT_SUCCESS == answer_fact_reply(corpus, qtext, qlen,
          probe, sizeof(probe)))
        {
          answered = 1;
        }
        pass = (0 == answered) ? 1 : 0;
        fprintf(stdout, "%s abstain %s\n", (0 != pass) ? "PASS" : "FAIL",
          fields[0]);
      }
      if (0 != pass) ++npass;
      continue;
    }
    if (0 == eval_terms_empty(fields[2])) {
      if (nanswers <= 0 || EXIT_SUCCESS != answer_reply(qtext,
        qlen, entries[0], lexicon, rules, nrules, reply,
        sizeof(reply)))
      {
        reply_pass = 0;
        reply[0] = '\0';
      }
      else {
        reply_pass = eval_terms_match_text(reply, (int)strlen(reply),
          fields[2]);
      }
    }
    pass = (0 != any_pass && 0 != reply_pass && 0 != fact_pass) ? 1 : 0;
    fprintf(stdout, "%s top %s\n", (0 != top_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s any %s\n", (0 != any_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s reply %s\n", (0 != reply_pass) ? "PASS" : "FAIL",
      fields[0]);
    if (0 != fact_checked) {
      fprintf(stdout, "%s fact %s\n", (0 != fact_pass) ? "PASS" : "FAIL",
        fields[0]);
      if (0 != fact_pass) ++nfact;
    }
    if (0 != top_pass) ++ntop;
    if (0 != any_pass) ++nany;
    if (0 != reply_pass) ++nreply;
    if (0 != pass) ++npass;
  }
  fprintf(stdout,
    "eval[%s]: %d/%d passed (top=%d, any=%d, reply=%d, fact=%d)\n",
    profile->name, npass, ncases, ntop, nany, nreply, nfact);
  if (NULL != answer_hier_model) {
    fprintf(stderr, "  hier rescore: %ld rankings, top-1 changed on %ld\n",
      answer_hier_nreorder, answer_hier_nchanged);
  }
  if (NULL != file) fclose(file);
  if (ncases > 0 && npass == ncases) result = EXIT_SUCCESS;
  return result;
}


/**
 * The variable-order n-gram engine now lives in libxs_ngram; the wrappers
 * below adapt converse's call sites (which thread the model's registry and an
 * explicit maxorder) to the single shared model. model == converse_ngram.store
 * and maxorder == converse_ngram.maxorder by construction.
 */
static void ngramk_observe(libxs_registry_t* model, const unsigned int hist[],
  int hlen, unsigned int succ_id, int maxorder)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  libxs_ngram_observe(&converse_ngram, hist, hlen, succ_id);
}


static const ngram_entry_t* ngramk_lookup(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int n)
{
  LIBXS_UNUSED(model);
  return libxs_ngram_lookup(&converse_ngram, hist, hlen, n);
}


/**
 * The pure count-based estimate: interpolated backoff with NO skip tier folded
 * in. The bank needs this because it carries skip as its own slot, and an order
 * expert that already contained skip at a fixed weight would double-count it --
 * making the learned skip weight meaningless.
 */
static double ngramk_prob_exact(const unsigned int hist[], int hlen,
  unsigned int next)
{
  return libxs_ngram_prob(&converse_ngram, hist, hlen, next);
}


static double ngramk_prob(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int next)
{
  double p = ngramk_prob_exact(hist, hlen, next);
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  if (0 != converse_skip_on) {
    double p_skip = ngram_skip_prob(hist, hlen, next);
    if (p_skip > 0.0) {
      double mu = ngram_skip_mu();
      p = (1.0 - mu) * p + mu * p_skip;
    }
  }
  return p;
}


/**
 * Adapter matching converse_recomb_prob_t. The recombination syntax gate needs a
 * word-scale backoff probability and nothing more, so the model handle, the
 * maximum order and the skip-gram interpolation all stay on this side of the
 * boundary rather than being threaded through it.
 */
static double recomb_word_prob(const unsigned int hist[], int hlen,
  unsigned int next)
{
  return ngramk_prob(NULL, hist, hlen, ngram_maxorder(), next);
}


static int ngramk_predict_order(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int out_ids[],
  int k, int* order)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  return libxs_ngram_predict(&converse_ngram, hist, hlen, out_ids, k, order);
}


static int ngram_gran_mode(void)
{
  const char* env = getenv("CONVERSE_GRAN");
  if (NULL != env) {
    if (0 == strcmp(env, "native")) return GRAN_NATIVE;
    if (0 == strcmp(env, "syllable")) return GRAN_SYLLABLE;
    if (0 == strcmp(env, "bpe")) return GRAN_BPE;
    if (0 == strcmp(env, "meta-native")) return GRAN_META_NATIVE;
    if (0 == strcmp(env, "meta-word")) return GRAN_META_WORD;
    if (0 == strcmp(env, "meta-syllable")) return GRAN_META_SYLLABLE;
  }
  return GRAN_WORD;
}


static int ngram_native_mode(void)
{
  return (GRAN_WORD != ngram_gran_mode()) ? 1 : 0;
}


/**
 * Number of prior whole words to carry as sub-word prediction context, or 0
 * (off, byte-identical to piece-only context). Ignored at word/native mode.
 */
static int ngram_wordctx(void)
{
  int result = 0;
  const char* env = getenv("CONVERSE_WORDCTX");
  int mode = ngram_gran_mode();
  if (NULL != env && '\0' != *env
    && (GRAN_SYLLABLE == mode || GRAN_BPE == mode))
  {
    int v = atoi(env);
    if (v < 0) v = 0;
    if (v > NGRAM_ORDER_MAX) v = NGRAM_ORDER_MAX;
    result = v;
  }
  return result;
}


static int ngram_is_vowel(unsigned char c)
{
  c = (unsigned char)tolower(c);
  return ('a' == c || 'e' == c || 'i' == c || 'o' == c || 'u' == c
    || 'y' == c) ? 1 : 0;
}


/**
 * Signed-length adapter over libxs_utf8_decode, which is the strict form: an
 * invalid or truncated sequence reports width 1, so scanning always advances and
 * a malformed byte is simply not a vowel.
 */
static unsigned long ngram_utf8_decode(const char* text, int len, int* width)
{
  return libxs_utf8_decode((const unsigned char*)text, (size_t)len, width);
}


/**
 * Vowel test over CODE POINTS rather than bytes.  The byte test cannot see an
 * encoded letter at all, so every accented vowel read as a consonant and the
 * splitter cut German and French words at the wrong places -- a defect, not a
 * tuning choice.  Latin-1 Supplement and Latin Extended-A cover the vowels of
 * the languages this corpus set contains; anything outside is not claimed as a
 * vowel rather than guessed at.
 */
static int ngram_is_vowel_cp(unsigned long cp)
{
  int result = 0;
  if (cp < 128) result = ngram_is_vowel((unsigned char)cp);
  else if (0xC0 <= cp && 0x24F >= cp) {
    /* fold to the base letter: the accented ranges run in blocks whose
       residues mod the block size follow the base vowel order */
    static const char* const vowels = "aeiouy";
    unsigned long base = 0;
    if (0xC0 <= cp && 0xFF >= cp) {
      static const char latin1[64] = {
        'a','a','a','a','a','a','a','c','e','e','e','e','i','i','i','i',
        'd','n','o','o','o','o','o','x','o','u','u','u','u','y','t','s',
        'a','a','a','a','a','a','a','c','e','e','e','e','i','i','i','i',
        'd','n','o','o','o','o','o','/','o','u','u','u','u','y','t','y'
      };
      base = (unsigned long)latin1[cp - 0xC0];
    }
    if (0 != base && NULL != strchr(vowels, (int)base)) result = 1;
  }
  return result;
}


/**
 * A UTF-8 continuation or lead byte counts as a word character so that a pivot
 * span stays whole on a non-ASCII corpus: ctype rejects every byte of an encoded
 * letter, which would cut a German word mid-letter and offer the fragments as
 * separate pivots.
 *
 * Unlike the tokenizer's predicate this does NOT exclude encoded punctuation, and
 * it does not need to: this only extends a span whose FIRST byte the tokenizer
 * already classified as a word, so a leading typographic quote cannot start one.
 * The span is re-tokenized before its id is taken, so the authoritative decision
 * still belongs to the tokenizer.
 */
static int ngram_is_wordchar(unsigned char c)
{
  return (0 != isalnum(c)) ? 1 : 0;
}


static int bpe_add_symbol(const char* bytes, int len)
{
  int result = -1;
  if (len > 0 && len <= BPE_SYMBOL_MAX) {
    if (bpe_nsymbols >= bpe_cap_symbols) {
      int grown = (bpe_cap_symbols > 0) ? bpe_cap_symbols * 2 : 512;
      bpe_symbol_t* next = (bpe_symbol_t*)realloc(bpe_symbols,
        (size_t)grown * sizeof(*next));
      if (NULL != next) {
        bpe_symbols = next;
        bpe_cap_symbols = grown;
      }
    }
    if (bpe_nsymbols < bpe_cap_symbols) {
      bpe_symbols[bpe_nsymbols].len = len;
      memcpy(bpe_symbols[bpe_nsymbols].bytes, bytes, (size_t)len);
      result = bpe_nsymbols;
      ++bpe_nsymbols;
    }
  }
  return result;
}


static void bpe_free(void)
{
  free(bpe_symbols);
  bpe_symbols = NULL;
  bpe_nsymbols = 0;
  bpe_cap_symbols = 0;
  libxs_registry_destroy(bpe_merges);
  bpe_merges = NULL;
}


/**
 * Learn byte-pair merges from the training split only. Words are byte runs
 * split on whitespace (a leading-space marker keeps word starts distinct);
 * each iteration adds the most frequent adjacent symbol pair as a new symbol
 * and records its rank so encoding can replay the merges.
 */
static void bpe_build(const libxs_registry_t* corpus, int holdout)
{
  int nmerges = BPE_MERGES_DEFAULT;
  const char* env = getenv("CONVERSE_BPE_MERGES");
  bpe_word_t* words = NULL;
  long nwords = 0, cap_words = 0;
  int merge;
  bpe_free();
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 0) nmerges = v;
  }
  bpe_merges = libxs_registry_create();
  if (NULL == corpus || NULL == bpe_merges) return;
  { int b;
    char one[1];
    for (b = 0; b < 256; ++b) {
      one[0] = (char)(unsigned char)b;
      bpe_add_symbol(one, 1);
    }
  }
  { const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value && nwords < BPE_WORD_CAP) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      if (0 == predict_is_test(index, holdout)) {
        int pos = 0;
        while (pos < entry->text_len && nwords < BPE_WORD_CAP) {
          int wlen = 0, marker;
          bpe_word_t w;
          while (pos + wlen < entry->text_len
            && 0 == isspace((unsigned char)entry->text[pos + wlen])) ++wlen;
          if (wlen > 0) {
            int i;
            w.count = 1;
            w.nsyms = 0;
            marker = (pos > 0) ? 1 : 0;
            if (0 != marker && w.nsyms < BPE_WORD_MAX) {
              w.syms[w.nsyms++] = (int)(unsigned char)' ';
            }
            for (i = 0; i < wlen && w.nsyms < BPE_WORD_MAX; ++i) {
              w.syms[w.nsyms++] =
                (int)(unsigned char)entry->text[pos + i];
            }
            if (nwords >= cap_words) {
              long grown = (cap_words > 0) ? cap_words * 2 : 4096;
              bpe_word_t* next = (bpe_word_t*)realloc(words,
                (size_t)grown * sizeof(*next));
              if (NULL == next) break;
              words = next;
              cap_words = grown;
            }
            words[nwords++] = w;
            pos += wlen;
          }
          while (pos < entry->text_len
            && 0 != isspace((unsigned char)entry->text[pos])) ++pos;
        }
      }
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
  }
  for (merge = 0; merge < nmerges; ++merge) {
    libxs_registry_t* counts = libxs_registry_create();
    bpe_pair_t best;
    long best_count = 0;
    long w;
    int new_sym;
    char merged[BPE_SYMBOL_MAX];
    int merged_len;
    if (NULL == counts) break;
    best.a = best.b = -1;
    for (w = 0; w < nwords; ++w) {
      int i;
      for (i = 0; i + 1 < words[w].nsyms; ++i) {
        bpe_pair_t pair;
        long* slot;
        long acc;
        pair.a = words[w].syms[i];
        pair.b = words[w].syms[i + 1];
        slot = (long*)libxs_registry_get(counts, &pair, sizeof(pair), NULL);
        acc = (NULL != slot) ? *slot + words[w].count : words[w].count;
        if (NULL != slot) *slot = acc;
        else libxs_registry_set(counts, &pair, sizeof(pair), &acc,
          sizeof(acc), NULL);
        if (acc > best_count) {
          best_count = acc;
          best = pair;
        }
      }
    }
    libxs_registry_destroy(counts);
    if (best.a < 0 || best_count < 2) break;
    merged_len = bpe_symbols[best.a].len + bpe_symbols[best.b].len;
    if (merged_len > BPE_SYMBOL_MAX) break;
    memcpy(merged, bpe_symbols[best.a].bytes, (size_t)bpe_symbols[best.a].len);
    memcpy(merged + bpe_symbols[best.a].len, bpe_symbols[best.b].bytes,
      (size_t)bpe_symbols[best.b].len);
    new_sym = bpe_add_symbol(merged, merged_len);
    if (new_sym < 0) break;
    { bpe_rank_t rec;
      rec.rank = merge;
      rec.merged = new_sym;
      libxs_registry_set(bpe_merges, &best, sizeof(best), &rec, sizeof(rec),
        NULL);
    }
    for (w = 0; w < nwords; ++w) {
      int i = 0, out = 0;
      while (i < words[w].nsyms) {
        if (i + 1 < words[w].nsyms && words[w].syms[i] == best.a
          && words[w].syms[i + 1] == best.b)
        {
          words[w].syms[out++] = new_sym;
          i += 2;
        }
        else words[w].syms[out++] = words[w].syms[i++];
      }
      words[w].nsyms = out;
    }
  }
  free(words);
  fprintf(stderr, "  bpe: %d symbols (%d merges) from %ld words\n",
    bpe_nsymbols, bpe_nsymbols - 256, nwords);
}


/**
 * Encode one whitespace-delimited byte run [text,len) into BPE pieces by
 * greedily applying the lowest-rank applicable merge, then intern each piece.
 * Falls back to single bytes for anything the merges do not cover, so the
 * encoder never fails on unseen input.
 */
static int bpe_encode_run(const char* text, int len, libxs_lexeme_t tokens[],
  int max, int start, libxs_lexicon_t* lexicon, int create)
{
  int result = start;
  int marker = (len > 0 && ' ' == text[0]) ? 1 : 0;
  int syms[BPE_WORD_MAX];
  int nsyms = 0, i;
  for (i = 0; i < len && nsyms < BPE_WORD_MAX; ++i) {
    syms[nsyms++] = (int)(unsigned char)text[i];
  }
  for (;;) {
    int best_rank = -1, best_pos = -1, best_sym = -1;
    for (i = 0; i + 1 < nsyms; ++i) {
      bpe_pair_t pair;
      const bpe_rank_t* rec;
      pair.a = syms[i];
      pair.b = syms[i + 1];
      rec = (const bpe_rank_t*)libxs_registry_get(bpe_merges, &pair,
        sizeof(pair), NULL);
      if (NULL != rec && (best_rank < 0 || rec->rank < best_rank)) {
        best_rank = rec->rank;
        best_pos = i;
        best_sym = rec->merged;
      }
    }
    if (best_pos < 0) break;
    syms[best_pos] = best_sym;
    for (i = best_pos + 1; i + 1 < nsyms; ++i) syms[i] = syms[i + 1];
    --nsyms;
  }
  for (i = 0; i < nsyms && result < max; ++i) {
    const bpe_symbol_t* sym = bpe_symbols + syms[i];
    int nbytes = sym->len;
    unsigned int id = libxs_lexicon_id(lexicon, sym->bytes, sym->len,
      LIBXS_LEXEME_WORD, create);
    if (0 == id) break;
    if (0 == i && 0 != marker && nbytes > 0) --nbytes;
    tokens[result].id = id;
    tokens[result].length = (unsigned short)nbytes;
    tokens[result].flags = (unsigned short)(LIBXS_LEXEME_WORD
      | ((0 == i && 0 != marker) ? LIBXS_LEXEME_BREAK : 0));
    ++result;
  }
  return result;
}


/**
 * Split a word [text,wlen) into syllable pieces by a simple VC|CV heuristic:
 * cut before a consonant that is followed by a vowel, once the current piece
 * already contains a vowel. Caps piece length; always ends at word end.
 */
/**
 * Whether the consonant run text[a,b) is a legal syllable ONSET, i.e. may begin
 * a syllable in this orthography.
 *
 * This is the notion the previous rule lacked entirely: it cut before the last
 * consonant preceding a vowel, which splits `strength` inside `ngth` and
 * `rhythm` inside `thm` because it cannot tell a cluster from a coda. Maximal
 * onset says the onset takes as many consonants as may legally start a syllable
 * and the rest stay as the coda of the previous one.
 *
 * The digraph list is English-leaning and that is a KNOWN limit, recorded rather
 * than hidden: syllabification is strongly language-dependent (German compounds,
 * sch/tsch; Italian near-perfect CV), which is exactly the per-language cost
 * this project has refused to pay by hand. It is here to make the unit
 * measurable at all -- the current output is wrong on words a speaker of any of
 * these languages reads correctly -- not as the final rule.
 */
static int ngram_onset_legal(const char* text, int a, int b)
{
  const int n = b - a;
  int result = 0;
  if (0 >= n) result = 1; /* empty onset is always legal */
  else if (1 == n) result = 1;
  else if (2 == n) {
    static const char* const two[] = {
      "bl","br","ch","cl","cr","dr","fl","fr","gl","gn","gr","kl","kn","kr",
      "ph","pl","pr","qu","sc","sh","sk","sl","sm","sn","sp","st","sw","th",
      "tr","tw","wh","wr","ts","pf","sz","gh","dw","vr","zw",
      NULL
    };
    int k;
    char c0 = (char)tolower((unsigned char)text[a]);
    char c1 = (char)tolower((unsigned char)text[a + 1]);
    for (k = 0; NULL != two[k] && 0 == result; ++k) {
      if (two[k][0] == c0 && two[k][1] == c1) result = 1;
    }
  }
  else if (3 == n) {
    static const char* const three[] = {
      "chr","phr","sch","scr","shr","spl","spr","str","thr","tsch",
      NULL
    };
    int k;
    char c0 = (char)tolower((unsigned char)text[a]);
    char c1 = (char)tolower((unsigned char)text[a + 1]);
    char c2 = (char)tolower((unsigned char)text[a + 2]);
    for (k = 0; NULL != three[k] && 0 == result; ++k) {
      if (three[k][0] == c0 && three[k][1] == c1 && three[k][2] == c2) {
        result = 1;
      }
    }
  }
  return result;
}


/**
 * Split a word into syllable-like pieces by maximal onset.
 *
 * Walk the letters (code points, so an encoded vowel is seen); after a vowel has
 * been seen, a run of consonants followed by another vowel is a boundary, and
 * the cut goes as late as legality allows: try the longest legal onset first and
 * shorten it until the remainder before it is non-empty. A word with no vowel at
 * all stays one piece rather than being chopped at a fixed width, which is what
 * produced `stre|ngth`.
 */
static int ngram_syllable_split(const char* text, int wlen, int piece_begin[],
  int piece_len[], int max)
{
  int result = 0;
  int start = 0, i = 0, seen_vowel = 0;
  while (i < wlen && result < max) {
    int w = 1;
    const unsigned long cp = ngram_utf8_decode(text + i, wlen - i, &w);
    if (0 == ngram_is_vowel_cp(cp)) {
      if (0 != seen_vowel) {
        /* scan the whole consonant run, then decide where it divides */
        int run = i, j = i, cut;
        while (j < wlen) {
          int wj = 1;
          const unsigned long cj = ngram_utf8_decode(text + j, wlen - j, &wj);
          if (0 != ngram_is_vowel_cp(cj)) break;
          j += wj;
        }
        if (j < wlen) { /* a vowel follows, so this run spans a boundary */
          cut = run;
          while (cut < j && 0 == ngram_onset_legal(text, cut, j)) ++cut;
          /* never cut at the piece start: that would emit an empty piece */
          if (cut <= start) cut = (start < j) ? j : wlen;
          if (cut > start && cut < wlen) {
            piece_begin[result] = start;
            piece_len[result] = cut - start;
            ++result;
            start = cut;
            seen_vowel = 0;
          }
          i = (cut > i) ? cut : j;
          continue;
        }
        i = j; /* trailing consonants: they belong to the final piece */
        continue;
      }
    }
    else seen_vowel = 1;
    i += w;
  }
  if (start < wlen && result < max) {
    piece_begin[result] = start;
    piece_len[result] = wlen - start;
    ++result;
  }
  return result;
}


static int ngram_metatoken_granularity(int mode)
{
  int result = -1;
  if (GRAN_META_NATIVE == mode) result = LIBXS_TOKEN_GRANULARITY_NATIVE;
  else if (GRAN_META_WORD == mode) result = LIBXS_TOKEN_GRANULARITY_WORD;
  else if (GRAN_META_SYLLABLE == mode) {
    result = LIBXS_TOKEN_GRANULARITY_SYLLABLE;
  }
  return result;
}


static unsigned int ngram_metatoken_flags(int kind, int sentence,
  int have_break)
{
  unsigned int result = 0;
  if (LIBXS_TOKEN_TEXT == kind) {
    result = LIBXS_LEXEME_WORD;
  }
  else if (LIBXS_TOKEN_NUMBER == kind) result = LIBXS_LEXEME_NUMBER;
  else result = LIBXS_LEXEME_PUNCT;
  if (LIBXS_TOKEN_MARKUP == kind || LIBXS_TOKEN_SPACE == kind) {
    result |= LIBXS_LEXEME_MARKUP;
  }
  if (0 != sentence) result |= LIBXS_LEXEME_SENTENCE;
  if (0 != have_break) result |= LIBXS_LEXEME_BREAK;
  return result;
}


static int ngram_metatoken_tokens(libxs_lexicon_t* lexicon,
  const char* text, int text_len, libxs_lexeme_t tokens[],
  unsigned int word_ids[], int max, int create, int granularity)
{
  int result = 0;
  libxs_token_stream_t stream;
  libxs_tokenizer_t* tokenizer = libxs_tokenizer_create(granularity);
  size_t token_pos = 0;
  int have_break = 0;
  libxs_token_stream_init(&stream);
  if (NULL != tokenizer && NULL != lexicon && NULL != text && text_len > 0
    && EXIT_SUCCESS == libxs_token_stream_encode(tokenizer, &stream,
      (const unsigned char*)text, (size_t)text_len))
  {
    const char bos = '\1';
    unsigned int bos_id = libxs_lexicon_id(lexicon, &bos, 1,
      LIBXS_LEXEME_MARKUP, create);
    if (0 == bos_id && 0 == create) {
      bos_id = libxs_lexicon_id(lexicon, &bos, 1,
        LIBXS_LEXEME_MARKUP, 1);
    }
    if (0 != bos_id && result < max) {
      tokens[result].id = bos_id;
      tokens[result].length = 0;
      tokens[result].flags = LIBXS_LEXEME_MARKUP;
      if (NULL != word_ids) word_ids[result] = bos_id;
      ++result;
    }
    while (token_pos < stream.size && result < max) {
      size_t payload_size = 0;
      size_t cells = libxs_token_span(stream.data, stream.size, token_pos,
        &payload_size);
      unsigned char payload[LIBXS_LEXEME_MAXBYTES];
      libxs_token_info_t info;
      if (0 == cells || 0 == payload_size
        || payload_size > sizeof(payload)
        || EXIT_SUCCESS != libxs_token_read(stream.data, stream.size,
          token_pos, payload, sizeof(payload), &info))
      {
        break;
      }
      else {
        unsigned int flags = ngram_metatoken_flags(info.kind,
          info.is_sentence, have_break);
        unsigned int id = libxs_lexicon_id(lexicon, (const char*)payload,
          (int)payload_size, flags, create);
        if (0 == id && 0 == create) {
          id = libxs_lexicon_id(lexicon, (const char*)payload,
            (int)payload_size, flags, 1);
        }
        if (0 == id) break;
        tokens[result].id = id;
        tokens[result].length = (unsigned short)payload_size;
        tokens[result].flags = (unsigned short)flags;
        if (NULL != word_ids) word_ids[result] = id;
        ++result;
        have_break = (LIBXS_TOKEN_SPACE == info.kind) ? 1 : 0;
        token_pos += cells;
      }
    }
  }
  libxs_token_stream_release(&stream);
  libxs_tokenizer_destroy(tokenizer);
  return result;
}


/**
 * Cost in bits of emitting one word as the pieces implied by a cut mask, scored
 * against the trained store. Bit i of mask means "cut before letter i+1".
 *
 * THE TRAP THIS AVOIDS: a candidate split invents pieces the training text never
 * contained, and an unknown piece has NO id. Scoring only the pieces that happen
 * to be known would make exotic splits look free -- fewer scored positions, less
 * accumulated cost -- and the oracle would "win" by producing garbage. So a
 * piece with no id is charged the unigram floor rather than skipped, which is the
 * honest price of a unit the model cannot represent.
 *
 * Returns bits, and writes the piece count. Scoring is done with create=0: the
 * oracle must not grow the lexicon it is measuring against.
 */
static double ngram_syllable_cost(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const char* word, int wlen, unsigned long mask,
  const unsigned int hist_in[], int hlen_in, int maxorder, int* out_npiece)
{
  const double inv_log2 = 1.0 / log(2.0);
  unsigned int hist[NGRAM_ORDER_MAX];
  double bits = 0.0;
  int hlen = hlen_in, npiece = 0, at = 0, i;
  for (i = 0; i < hlen && i < NGRAM_ORDER_MAX; ++i) hist[i] = hist_in[i];
  for (i = 1; i <= wlen; ++i) {
    /* a cut before letter i, or the end of the word, closes a piece */
    if (wlen == i || 0 != (mask & (1UL << (i - 1)))) {
      char buf[LIBXS_LEXEME_MAXBYTES + 1];
      int plen = i - at;
      if (0 < plen && plen <= (int)sizeof(buf) - 1) {
        unsigned int id;
        memcpy(buf, word + at, (size_t)plen);
        id = libxs_lexicon_id(lexicon, buf, plen, LIBXS_LEXEME_WORD, 0);
        if (0 != id) {
          const double p = ngramk_prob(model, hist, hlen, maxorder, id);
          bits -= log((p > 0.0) ? p : 1e-12) * inv_log2;
          ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, id);
        }
        else {
          /* unrepresentable piece: charge the floor, do not extend history */
          bits -= log(1e-12) * inv_log2;
        }
        ++npiece;
      }
      at = i;
    }
  }
  if (NULL != out_npiece) *out_npiece = npiece;
  return bits;
}


/**
 * Per-word split ORACLE: the cheapest cut set over all candidates, which bounds
 * what ANY cut rule -- hand-written or learned -- can achieve on this corpus.
 *
 * The decision rule was pre-committed before this was written (and is the reason
 * to write it): if the oracle is not materially better than the maximal-onset
 * repair, a learned splitter CANNOT PAY and the line is dropped. Exactly how
 * CONVERSE_NGRAM_ORACLE bounded order selection at 2.018 and closed that route.
 *
 * Words longer than ORACLE_MAXLEN are left to the heuristic: the candidate space
 * is 2^(n-1), so an exhaustive oracle is only honest where it is affordable, and
 * silently sampling a subset would report a bound that is not one.
 */
static void ngram_syllable_oracle(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_registry_t* corpus, int holdout, int maxorder)
{
  enum { ORACLE_MAXLEN = 12 };
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  double heur_bits = 0.0, oracle_bits = 0.0, single_bits = 0.0;
  long nword = 0, nskipped = 0, nheur_optimal = 0;
  long index = 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    const int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    if (0 != is_test && entry->text_len > 0) {
      int pos = 0;
      while (pos < entry->text_len) {
        const char* text = entry->text;
        if (0 == ngram_is_wordchar((unsigned char)text[pos])) { ++pos; continue; }
        { int wlen = 0;
          while (pos + wlen < entry->text_len
            && 0 != ngram_is_wordchar((unsigned char)text[pos + wlen])) ++wlen;
          if (1 < wlen && ORACLE_MAXLEN >= wlen) {
            int begin[LIBXS_LEXEME_MAXBYTES], plen[LIBXS_LEXEME_MAXBYTES];
            const int np = ngram_syllable_split(text + pos, wlen, begin, plen,
              LIBXS_LEXEME_MAXBYTES);
            const unsigned long ncand = 1UL << (wlen - 1);
            unsigned long mask, hmask = 0, best_mask = 0;
            double best = -1.0, hcost;
            int k;
            /* the heuristic's own cut set, as a mask */
            for (k = 1; k < np; ++k) hmask |= 1UL << (begin[k] - 1);
            hcost = ngram_syllable_cost(model, lexicon, text + pos, wlen,
              hmask, NULL, 0, maxorder, NULL);
            for (mask = 0; mask < ncand; ++mask) {
              const double c = ngram_syllable_cost(model, lexicon, text + pos,
                wlen, mask, NULL, 0, maxorder, NULL);
              if (best < 0.0 || c < best) { best = c; best_mask = mask; }
            }
            heur_bits += hcost;
            oracle_bits += best;
            single_bits += ngram_syllable_cost(model, lexicon, text + pos,
              wlen, 0, NULL, 0, maxorder, NULL);
            if (best_mask == hmask) ++nheur_optimal;
            ++nword;
          }
          else if (1 < wlen) ++nskipped;
          pos += wlen;
        }
      }
    }
    LIBXS_UNUSED(rules); LIBXS_UNUSED(nrules);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (0 < nword) {
    fprintf(stderr, "syllable oracle (words 2..%d letters, n=%ld, skipped %ld"
      " longer):\n  heuristic=%.3f bits/word | ORACLE=%.3f | whole-word=%.3f"
      " | heuristic optimal for %.1f%% of words\n", (int)ORACLE_MAXLEN, nword,
      nskipped, heur_bits / (double)nword, oracle_bits / (double)nword,
      single_bits / (double)nword,
      100.0 * (double)nheur_optimal / (double)nword);
  }
}


/**
 * Emits LIBXS_LEXEME_BREAK on a token preceded by whitespace in the source, so
 * libxs_lexeme_word_next groups the pieces of one word. Native granularity cuts
 * fixed-width chunks across word boundaries and hence marks none. When word_ids
 * is non-NULL, each piece receives the whole-word lexicon id of the word it
 * belongs to (its own id for native chunks and standalone punctuation), which
 * lets a caller build word-span context over sub-word emission.
 */
static int ngram_native_tokens(libxs_lexicon_t* lexicon, const char* text,
  int text_len, libxs_lexeme_t tokens[], unsigned int word_ids[], int max,
  int create)
{
  int result = 0;
  int pos = 0;
  int have_break = 0;
  int mode = ngram_gran_mode();
  int meta_granularity = ngram_metatoken_granularity(mode);
  if (meta_granularity >= 0) {
    return ngram_metatoken_tokens(lexicon, text, text_len, tokens, word_ids,
      max, create, meta_granularity);
  }
  if (GRAN_BPE == mode) {
    while (pos < text_len && result < max) {
      int wlen = 0;
      int run_start = pos;
      int marker = (pos > 0) ? 1 : 0;
      char run[BPE_WORD_MAX + 1];
      int run_len = 0;
      int prev = result;
      while (pos + wlen < text_len
        && 0 == isspace((unsigned char)text[pos + wlen])) ++wlen;
      if (0 == wlen) {
        while (pos < text_len
          && 0 != isspace((unsigned char)text[pos])) ++pos;
        continue;
      }
      if (0 != marker && run_len < BPE_WORD_MAX) run[run_len++] = ' ';
      { int i;
        for (i = 0; i < wlen && run_len < BPE_WORD_MAX; ++i) {
          run[run_len++] = text[run_start + i];
        }
      }
      result = bpe_encode_run(run, run_len, tokens, max, result,
        lexicon, create);
      if (NULL != word_ids && result > prev) {
        unsigned int wid = libxs_lexicon_id(lexicon, text + run_start,
          wlen, LIBXS_LEXEME_WORD, create);
        int j;
        for (j = prev; j < result; ++j) word_ids[j] = wid;
      }
      pos = run_start + wlen;
      while (pos < text_len && 0 != isspace((unsigned char)text[pos])) ++pos;
    }
    return result;
  }
  if (GRAN_SYLLABLE != mode) {
    while (pos < text_len && result < max) {
      int len = text_len - pos;
      unsigned int id;
      if (len > NGRAM_NATIVE_WIDTH) len = NGRAM_NATIVE_WIDTH;
      id = libxs_lexicon_id(lexicon, text + pos, len,
        LIBXS_LEXEME_WORD, create);
      /**
       * An unknown chunk must be EMITTED with id 0, not abort the entry. When
       * scoring held-out text (create=0) a chunk absent from training is the
       * normal case, and breaking here silently truncated the entry at its first
       * novel chunk -- dropping ~95% of the held-out bytes and making the BPC
       * denominator, hence BPC itself, incomparable with the other units. Id 0
       * is the reserved unknown, which the scoring loop skips as non-content,
       * so the bytes are still counted where the caller counts source bytes.
       */
      tokens[result].id = id;
      tokens[result].length = (unsigned short)len;
      tokens[result].flags = LIBXS_LEXEME_WORD;
      if (NULL != word_ids) word_ids[result] = id;
      ++result;
      pos += len;
    }
    return result;
  }
  while (pos < text_len && result < max) {
    unsigned char c = (unsigned char)text[pos];
    if (0 == ngram_is_wordchar(c)) {
      char buf[3];
      int off = 0;
      unsigned int id;
      /**
       * Whitespace is NOT emitted as a piece: the boundary it marks is already
       * carried by the leading space baked into the next piece's text (below,
       * and the same convention the byte-pair encoder uses). Emitting it here
       * too would represent one space twice, so concatenating the pieces would
       * render "to  the" and no such text occurs in any corpus.
       */
      if (0 != isspace(c)) {
        have_break = 1;
        ++pos;
        continue;
      }
      if (0 != have_break) buf[off++] = ' ';
      buf[off] = (char)c;
      id = libxs_lexicon_id(lexicon, buf, off + 1, LIBXS_LEXEME_PUNCT, create);
      if (0 == id) break;
      tokens[result].id = id;
      tokens[result].length = 1;
      tokens[result].flags = (unsigned short)(LIBXS_LEXEME_PUNCT
        | ((0 != have_break) ? LIBXS_LEXEME_BREAK : 0));
      if (NULL != word_ids) word_ids[result] = id;
      ++result;
      have_break = 0;
      ++pos;
    }
    else {
      int wlen = 0;
      int piece_begin[COMPOSE_MAXTEXT / 2];
      int piece_len[COMPOSE_MAXTEXT / 2];
      int np, pi;
      unsigned int wid = 0;
      while (pos + wlen < text_len
        && 0 != ngram_is_wordchar((unsigned char)text[pos + wlen])) ++wlen;
      if (NULL != word_ids) {
        wid = libxs_lexicon_id(lexicon, text + pos, wlen,
          LIBXS_LEXEME_WORD, create);
      }
      np = ngram_syllable_split(text + pos, wlen, piece_begin, piece_len,
        (int)(sizeof(piece_len) / sizeof(*piece_len)));
      for (pi = 0; pi < np && result < max; ++pi) {
        char buf[LIBXS_LEXEME_MAXBYTES + 1];
        int plen = piece_len[pi];
        int off = 0;
        unsigned int id;
        if (0 == pi && 0 != have_break) buf[off++] = ' ';
        if (plen > (int)sizeof(buf) - 1 - off) plen = (int)sizeof(buf) - 1 - off;
        memcpy(buf + off, text + pos + piece_begin[pi], (size_t)plen);
        id = libxs_lexicon_id(lexicon, buf, off + plen,
          LIBXS_LEXEME_WORD, create);
        if (0 == id) break;
        tokens[result].id = id;
        tokens[result].length = (unsigned short)piece_len[pi];
        tokens[result].flags = (unsigned short)(LIBXS_LEXEME_WORD
          | ((0 == pi && 0 != have_break) ? LIBXS_LEXEME_BREAK : 0));
        if (NULL != word_ids) word_ids[result] = wid;
        ++result;
      }
      have_break = 0;
      pos += wlen;
    }
  }
  return result;
}


static int ngram_maxorder(void)
{
  int result = (0 != converse_order_max) ? NGRAM_ORDER_MAX : 2;
  const char* env = getenv("CONVERSE_NGRAM_ORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}


/**
 * Score (and train on) each text at ONE scale only. The corpus holds every text
 * at both sentence and paragraph scale, so the default loops see each sentence
 * TWICE: once standalone and once inside its paragraph. That is multiplicity,
 * not two observations -- the second copy is the same source bytes. It inflates
 * training counts and, worse, lets a paragraph copy make a sentence's own
 * contexts look attested, which is exactly the confound the slot probe already
 * filters against (it takes SCALE_SENTENCE only).
 *
 * Off by default so published figures stay reproducible; the point of the knob
 * is to measure how much the duplication was worth.
 */
static int ngram_dedup_scale(void)
{
  int result = 0;
  const char* env = getenv("CONVERSE_NGRAM_ONESCALE");
  if (NULL != env && '\0' != *env) {
    const int v = atoi(env);
    if (v >= 1 && v <= 2) result = v;
    else if ('0' != env[0]) result = 1;
  }
  return result;
}


/**
 * Wall-clock per startup stage, so a slow run is attributed rather than guessed
 * at. Sec 23 blamed the hierarchy for a stall that was actually
 * converse_predict_train and lost the measurement; the fix is to make the
 * attribution a printed number instead of an inference.
 */
/**
 * The answer ranker trains OFF by default. Measured: it is 40% of wall time on a
 * 2 MB corpus, leaves BPC identical to three decimals, and leaves QA at 9/9 and
 * 14/14 -- its only consumer (answer_predict_score) degrades to the unmodified
 * base score when the model is absent. A stage that costs 40% and moves nothing
 * measurable should be opt-in. CONVERSE_NO_PREDICT is still honoured so existing
 * scripts keep working.
 */
static int converse_predict_on(void)
{
  const char* env = getenv("CONVERSE_PREDICT");
  if (NULL != env) return ('0' != env[0] && '\0' != env[0]) ? 1 : 0;
  return 0;
}


static int converse_stage_on(void)
{
  const char* env = getenv("CONVERSE_STAGES");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


static void converse_stage_begin(void)
{
  if (0 != converse_stage_on()) converse_stage_tick = libxs_timer_tick();
}


static void converse_stage_end(const char* name)
{
  if (0 != converse_stage_on() && converse_stage_count < CONVERSE_STAGE_MAX) {
    const libxs_timer_tick_t now = libxs_timer_tick();
    converse_stage_name[converse_stage_count] = name;
    converse_stage_time[converse_stage_count]
      = libxs_timer_duration(converse_stage_tick, now);
    ++converse_stage_count;
    converse_stage_tick = now;
  }
}


static void converse_stage_report(void)
{
  if (0 != converse_stage_on() && converse_stage_count > 0) {
    double total = 0.0;
    int i;
    for (i = 0; i < converse_stage_count; ++i) {
      total += converse_stage_time[i];
    }
    fprintf(stderr, "stages (s):");
    for (i = 0; i < converse_stage_count; ++i) {
      fprintf(stderr, " %s=%.1f", converse_stage_name[i],
        converse_stage_time[i]);
    }
    fprintf(stderr, " | total=%.1f\n", total);
  }
}


/**
 * Report, per position, the cost of every fixed-order expert and of an oracle
 * that picks the cheapest. A measurement, not a mechanism: the oracle reads the
 * truth, so it is an upper bound on what any per-position order selection can
 * win over the single interpolated model.
 */
static int ngram_oracle_probe(void)
{
  const char* env = getenv("CONVERSE_NGRAM_ORACLE");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


/**
 * Which fixed-order expert the achievable selector falls back to. The best
 * single order is unit-dependent (order 1 for whole words, order 2 for the
 * sub-word units), so a hardcoded 1 measures the rule on the wrong expert and
 * reports a loss where there is a gain.
 */
static int ngram_select_order(void)
{
  int result = 1;
  const char* env = getenv("CONVERSE_NGRAM_SELORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}


/**
 * Word-level expert bank: mix the fixed-order experts with causally updated
 * fixed-share weights instead of the count-derived backoff weights. The byte
 * side earned 0.5-0.8 bits cross-document this way, and the order-selection
 * probe showed the same headroom exists at word and sub-word units, so this is
 * the same mechanism at a coarser unit rather than a new one.
 *
 * The threshold rule proved the headroom is reachable from an observable
 * feature; this replaces the hard threshold with a learned per-position
 * weighting, which is what the byte side actually does.
 */
static int ngram_bank_probe(void)
{
  const char* env = getenv("CONVERSE_NGRAM_BANK");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


static double ngram_bank_rate(void)
{
  double result = 0.15;
  const char* env = getenv("CONVERSE_NGRAM_BANK_RATE");
  if (NULL != env && '\0' != *env) {
    const double v = atof(env);
    /**
     * Zero is ACCEPTED: it means the weights never move from their uniform
     * prior, which is the control the learned weighting has to beat and the
     * only way to read the rate sweep's endpoint. Rejecting it silently
     * substituted the default, so a run labelled rate 0 was measured at 0.15
     * and looked like evidence that the rate does not matter.
     */
    if (v >= 0.0 && v <= 4.0) result = v;
  }
  return result;
}


static double ngram_bank_share(void)
{
  double result = 0.005;
  const char* env = getenv("CONVERSE_NGRAM_BANK_SHARE");
  if (NULL != env && '\0' != *env) {
    const double v = atof(env);
    if (v >= 0.0 && v < 1.0) result = v;
  }
  return result;
}


/**
 * One causal fixed-share step over the expert slots. Multiplicative log-loss
 * update toward the experts that beat the mixture, then a share of the mass
 * redistributed uniformly so an expert that was wrong for a while can recover.
 * Only slots that are ENABLED (weight > 0) and ACTIVE (spoke at this position)
 * participate: an abstaining slot is neither updated nor renormalized, so it
 * carries its weight to the next position where it can speak. Restricting the
 * update to the slots that spoke is also what keeps the order experts bit-exact
 * to the pre-slot bank, where the loop simply stopped at min(maxorder, hlen).
 *
 * The ratio is floored only where it is EXACTLY zero, which happens when an
 * active expert assigns no mass at all to the observed target (the skip tier
 * matches a pair whose successor list omits it). pow(0, rate) is exactly zero,
 * so such a slot would lose all its mass in one step and the uniform recovery
 * term, which only reaches slots that still hold mass, could never revive it.
 * A merely tiny ratio needs no floor -- it decays steeply but stays positive,
 * and clamping it would perturb the order-only weights that are otherwise
 * bit-exact to the pre-slot bank.
 */
/**
 * Spread the slot-indexed experts into the parallel arrays libxs_mix takes. Slot
 * 0 is unused here and holds weight zero, so the shared primitive skips it on
 * the same weight>0 test the local loop used.
 */
static void ngram_bank_view(libxs_mix_t* mix, double weight[],
  const ngram_expert_t expert[], double prob[], int active[],
  double rate, double share)
{
  int slot;
  for (slot = 0; slot < NGRAM_BANK_MAX; ++slot) {
    prob[slot] = expert[slot].probability;
    active[slot] = (slot >= 1 && slot <= NGRAM_BANK_LAST)
      ? expert[slot].active : 0;
  }
  mix->weight = weight;
  mix->nslot = NGRAM_BANK_MAX;
  mix->rate = rate;
  mix->share = share;
  mix->relmin = NGRAM_BANK_RELMIN;
}


static void ngram_bank_update(double weight[], const ngram_expert_t expert[],
  double mixture, double rate, double share)
{
  libxs_mix_t mix;
  double prob[NGRAM_BANK_MAX];
  int active[NGRAM_BANK_MAX];
  ngram_bank_view(&mix, weight, expert, prob, active, rate, share);
  /* the caller's mixture, which carries the log-loss floor ngram_bank_pool
     applies; recomputing it here would drop that floor */
  libxs_mix_update(&mix, prob, active, mixture);
}


/**
 * Which extra slots the bank carries beyond the fixed orders, as a mask over
 * slot indices. Empty by default: with no extra slot the bank is the order-only
 * pool, which is the control the kind-different experts have to beat. The knobs
 * are independent so one candidate can be measured without the others.
 */
static unsigned int ngram_bank_slots(void)
{
  unsigned int result = 0;
  const char* skip = getenv("CONVERSE_NGRAM_BANK_SKIP");
  const char* prior = getenv("CONVERSE_NGRAM_BANK_PRIOR");
  if (NULL != skip && '0' != skip[0] && '\0' != skip[0]) {
    result |= 1u << NGRAM_BANK_SKIP;
  }
  if (NULL != prior && '0' != prior[0] && '\0' != prior[0]) {
    result |= 1u << NGRAM_BANK_PRIOR;
  }
  { const char* prd = getenv("CONVERSE_NGRAM_BANK_PREDICT");
    if (NULL != prd && '0' != prd[0] && '\0' != prd[0]) {
      result |= 1u << NGRAM_BANK_PREDICT;
    }
  }
  { const char* emb = getenv("CONVERSE_NGRAM_BANK_EMB");
    if (NULL != emb && '0' != emb[0] && '\0' != emb[0]) {
      result |= 1u << NGRAM_BANK_EMB;
    }
  }
  return result;
}


static int ngram_bank_enabled(unsigned int slots, int slot)
{
  return (slot <= NGRAM_ORDER_MAX || 0 != (slots & (1u << slot))) ? 1 : 0;
}


static const char* ngram_bank_slotname(int slot)
{
  static char name[8];
  const char* result = name;
  if (NGRAM_BANK_SKIP == slot) result = "skip";
  else if (NGRAM_BANK_PRIOR == slot) result = "uni";
  else if (NGRAM_BANK_PREDICT == slot) result = "prd";
  else if (NGRAM_BANK_EMB == slot) result = "emb";
  else sprintf(name, "o%d", slot);
  return result;
}


/**
 * Fill the per-position expert set. The fixed-order experts are the SAME
 * interpolated model given only their last k words, so expert k is what a model
 * capped at order k would say.
 *
 * When the skip slot is enabled the order experts are asked for the SKIP-FREE
 * estimate: ngramk_prob folds the skip tier into every order at a fixed weight,
 * so leaving that in would double-count it and make the learned skip weight
 * meaningless. Without the slot they keep the folded estimate, which is what
 * holds every existing configuration bit-exact.
 *
 * An expert that cannot speak is left inactive rather than given probability
 * zero -- see ngram_expert_t. Order expert k needs k words of history; the skip
 * slot needs three and a pair that was actually observed. The unigram slot is
 * TOTAL: it always speaks, which is the point of carrying it -- the pool then
 * has a floor at every position, including those where counts have nothing.
 */
static void ngram_bank_experts(const unsigned int hist[], int hlen,
  int maxorder, unsigned int cur, unsigned int slots,
  const libxs_predict_t* store, void* context, int use_emb, int vocabulary,
  ngram_expert_t expert[])
{
  const int skip_slot = (0 != (slots & (1u << NGRAM_BANK_SKIP))) ? 1 : 0;
  int slot;
  for (slot = 0; slot < NGRAM_BANK_MAX; ++slot) {
    expert[slot].probability = 0.0;
    expert[slot].active = 0;
  }
  for (slot = 1; slot <= maxorder && slot <= hlen; ++slot) {
    const unsigned int* sub = hist + (hlen - slot);
    expert[slot].probability = (0 != skip_slot)
      ? ngramk_prob_exact(sub, slot, cur)
      : ngramk_prob(NULL, sub, slot, slot, cur);
    expert[slot].active = 1;
  }
  if (0 != skip_slot) {
    const double ps = ngram_skip_prob(hist, hlen, cur);
    /**
     * Abstention is a property of the CONTEXT, not of the target: the tier
     * speaks when its pair was observed, whatever probability it then assigns
     * to this particular successor. Testing ps > 0.0 instead would silently
     * make the slot active only where it happens to be right, which reads the
     * target and would flatter the expert.
     */
    if (0 != ngram_skip_ready(hist, hlen)) {
      expert[NGRAM_BANK_SKIP].probability = ps;
      expert[NGRAM_BANK_SKIP].active = 1;
    }
  }
  if (0 != (slots & (1u << NGRAM_BANK_PRIOR))) {
    /* Zero history reaches the smoothed unigram prior with no backoff term. */
    expert[NGRAM_BANK_PRIOR].probability = ngramk_prob_exact(hist, 0, cur);
    expert[NGRAM_BANK_PRIOR].active = 1;
  }
  if (0 != (slots & (1u << NGRAM_BANK_PREDICT)) && NULL != store
    && hlen >= 1)
  {
    /**
     * The store is keyed on the last two tokens, so it speaks wherever one token
     * of history exists -- including positions no count context attested, which
     * is the reason to carry it. The escape mass covers the unattested
     * remainder, so it is TOTAL: active whenever the output is scoreable.
     *
     * One call per position. prob_observe reports the distribution in effect
     * BEFORE the observation and advances the escape bank only afterwards, so
     * the ordering that keeps a stream figure honest is enforced by the library
     * rather than by call sequence here.
     */
    const unsigned int prev1 = hist[hlen - 1];
    const unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
    libxs_predict_prob_info_t pinfo;
    double in[2 * TOKEN_EMB_DIM];
    const double candidate = (double)cur;
    int nsupport;
    token_input_vector(prev2, prev1, use_emb, in);
    if (NULL != context) {
      nsupport = libxs_predict_prob_observe(NULL, store, context, in, 0,
        &candidate, NULL, NULL, 0, NULL, &pinfo, vocabulary, 1);
      /**
       * Mass mode only: a token id is discrete, so PDENSITY cannot arise, and an
       * unexpected PNONE must surface as abstention rather than a silent zero.
       */
      if (0 < nsupport && NULL != pinfo.kind
        && LIBXS_PREDICT_PMASS == pinfo.kind[0] && NULL != pinfo.prob)
      {
        expert[NGRAM_BANK_PREDICT].probability = pinfo.prob[0];
        expert[NGRAM_BANK_PREDICT].active = 1;
        predict_entropy_sum += pinfo.entropy;
        predict_support_last = nsupport;
        ++predict_nscored;
        { const double bits = -log((pinfo.prob[0] > 0.0)
            ? pinfo.prob[0] : 1e-12) / log(2.0);
          if (NULL != pinfo.attested && 0 != pinfo.attested[0]) {
            ++predict_att_n;
            predict_att_bits += bits;
          }
          else {
            ++predict_unatt_n;
            predict_unatt_bits += bits;
          }
        }
      }
    }
    else {
      /**
       * Frozen: the weights the warm-up committed are read and never written, so
       * the score does not depend on how many positions preceded this one. info
       * aliases the context, so the distribution cannot be reported here and the
       * point query is what is available -- which is also cheaper, since it
       * answers P(y|x) without enumerating the support. The mass-mode check
       * moves to the reported probability being a usable number.
       */
      double p = 0.0;
      libxs_predict_prob(NULL, store, NULL, in, &candidate, &p, NULL,
        vocabulary, 1);
      nsupport = 0;
      if (p > 0.0 && p <= 1.0) {
        expert[NGRAM_BANK_PREDICT].probability = p;
        expert[NGRAM_BANK_PREDICT].active = 1;
        ++predict_nscored;
      }
    }
  }
  if (0 != (slots & (1u << NGRAM_BANK_EMB)) && hlen >= 1 && vocabulary > 0) {
    /**
     * TOTAL by construction and for a different reason than the predict slot:
     * that one is total because its escape mass covers an unattested remainder,
     * this one because a low-rank completion of PPMI assigns a score to pairs
     * that were never observed at all. It reads ONE token of history, so it is
     * the weakest possible context -- carried anyway because the measured wall is
     * that 78% of novel-context positions have no attested successor to rank, and
     * coverage there is worth more than context depth here.
     */
    const int nctx = (ngram_emb_ctx() < hlen) ? ngram_emb_ctx() : hlen;
    const double p = token_emb_succ_prob(hist + (hlen - nctx), nctx, cur,
      (unsigned int)vocabulary, ngram_emb_temp());
    if (p > 0.0 && p <= 1.0) {
      expert[NGRAM_BANK_EMB].probability = p;
      expert[NGRAM_BANK_EMB].active = 1;
    }
  }
}


/**
 * Which pooling rule the bank uses: 0 = linear (default, bit-exact), 1 =
 * geometric (log-linear).
 *
 * The linear pool cannot beat its best member by much: it is a convex
 * combination, so a confident expert can be outvoted but never veto. Three
 * added slots were each capped near 0.008 bits, which makes the RULE a suspect
 * independent of the expert supply. A geometric pool multiplies instead of
 * averaging, so agreement sharpens and any expert can suppress a candidate.
 */
static int ngram_bank_geometric(void)
{
  const char* env = getenv("CONVERSE_NGRAM_BANK_GEO");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


/**
 * Collect the union of successor ids reachable from this history, which is the
 * only part of the vocabulary where the experts differ from a scaled prior.
 * Every expert is the same interpolated backoff over suffixes of one history,
 * so the union over all of them is the union of the successor lists at suffix
 * lengths 1..maxorder -- at most maxorder * SUCC_MAX ids.
 */
static int ngram_bank_support(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int ids[], int max)
{
  int result = 0, n;
  for (n = 1; n <= maxorder && n <= hlen; ++n) {
    const ngram_entry_t* entry = ngramk_lookup(model, hist, hlen, n);
    if (NULL != entry) {
      unsigned int s;
      for (s = 0; s < entry->nsucc && result < max; ++s) {
        const unsigned int id = entry->succ[s].id;
        int seen = 0, i;
        for (i = 0; i < result; ++i) {
          if (ids[i] == id) {
            seen = 1;
            break;
          }
        }
        if (0 == seen && 0 != id) ids[result++] = id;
      }
    }
  }
  return result;
}


/**
 * Geometric (log-linear) pool, normalized EXACTLY rather than approximately.
 * An unnormalized log-linear score is not a code length, so a BPC taken from
 * one would be meaningless -- the whole point of the measurement.
 *
 * The normalizer is exact at bounded cost because of the backoff structure: for
 * an id outside every successor list along the chain, each expert reduces to
 * c_k * prior(id) with c_k its total backoff mass. With the weights normalized
 * to sum to one, the product over such ids is (prod c_k^w_k) * prior(id), so
 * that entire tail of the vocabulary sums in closed form and only the support
 * needs explicit terms. c_k itself comes from each expert summing to one over
 * the vocabulary, so no probe id and no vocabulary scan is needed.
 *
 * Partial experts are excluded: an expert assigning exactly zero would drive
 * the product to zero everywhere outside its own successor list, which is the
 * known pathology of log-linear pools rather than a property of the data. Only
 * the total experts (the orders and the prior) participate.
 */
static double ngram_bank_pool_geo(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int cur,
  const double weight[], const ngram_expert_t expert[])
{
  unsigned int support[NGRAM_ORDER_MAX * NGRAM_SUCC_MAX];
  double logprod[NGRAM_ORDER_MAX * NGRAM_SUCC_MAX];
  double prior_sum = 0.0, tail_log = 0.0, wtotal = 0.0;
  double zsum = 0.0, numerator;
  const int nsupport = ngram_bank_support(model, hist, hlen, maxorder, support,
    (int)(sizeof(support) / sizeof(*support)));
  int slot, i, cur_index = -1;
  for (slot = 1; slot <= NGRAM_BANK_LAST; ++slot) {
    if (weight[slot] > 0.0 && 0 != expert[slot].active
      && expert[slot].probability > 0.0)
    {
      wtotal += weight[slot];
    }
  }
  if (!(wtotal > 0.0)) return 1e-12;
  for (i = 0; i < nsupport; ++i) {
    logprod[i] = 0.0;
    prior_sum += ngramk_prob_exact(hist, 0, support[i]);
    if (support[i] == cur) cur_index = i;
  }
  if (prior_sum >= 1.0) prior_sum = 1.0;
  for (slot = 1; slot <= NGRAM_BANK_LAST; ++slot) {
    if (weight[slot] > 0.0 && 0 != expert[slot].active
      && expert[slot].probability > 0.0)
    {
      const double w = weight[slot] / wtotal;
      const int k = (slot <= NGRAM_ORDER_MAX) ? slot : 0;
      double mass = 0.0, backoff;
      for (i = 0; i < nsupport; ++i) {
        const double pk = (k > 0)
          ? ngramk_prob_exact(hist + (hlen - k), k, support[i])
          : ngramk_prob_exact(hist, 0, support[i]);
        logprod[i] += w * log((pk > 0.0) ? pk : 1e-300);
        mass += pk;
      }
      /* Backoff mass left for the tail, from this expert summing to one. */
      backoff = (prior_sum < 1.0) ? (1.0 - mass) / (1.0 - prior_sum) : 0.0;
      tail_log += w * log((backoff > 0.0) ? backoff : 1e-300);
    }
  }
  for (i = 0; i < nsupport; ++i) zsum += exp(logprod[i]);
  zsum += exp(tail_log) * (1.0 - prior_sum);
  /**
   * The target is scored through the same partition, not added to the support:
   * the tail form is exact for any id outside it, so the normalizer stays
   * independent of which target is being scored.
   */
  numerator = (cur_index >= 0) ? exp(logprod[cur_index])
    : exp(tail_log) * ngramk_prob_exact(hist, 0, cur);
  if (!(zsum > 0.0)) return 1e-12;
  numerator /= zsum;
  return (numerator > 0.0) ? numerator : 1e-12;
}


/**
 * Score one position under the bank: a linear pool over the active slots,
 * renormalized by their weight mass so an abstaining expert costs no
 * probability mass rather than draining it.
 */
static double ngram_bank_pool(const double weight[],
  const ngram_expert_t expert[])
{
  libxs_mix_t mix;
  double copy[NGRAM_BANK_MAX];
  double prob[NGRAM_BANK_MAX];
  int active[NGRAM_BANK_MAX];
  double pooled;
  int slot;
  /* pool does not write the weights, but the view type is mutable; copying the
     few slots keeps the const contract instead of casting it away */
  for (slot = 0; slot < NGRAM_BANK_MAX; ++slot) copy[slot] = weight[slot];
  ngram_bank_view(&mix, copy, expert, prob, active, 0.0, 0.0);
  pooled = libxs_mix_pool(&mix, prob, active);
  /* the log-loss floor stays here: the primitive reports what it computed and
     leaves the choice of floor to the caller that takes the logarithm */
  return (pooled > 0.0) ? pooled : 1e-12;
}


/**
 * Skip-gram tier: an auxiliary store keyed by the pair (w[-3], w[-1]) that
 * abstracts over the varying middle slot, so an unseen exact context can still
 * match a seen "w ___ w" pattern (analogic generalization, no parameters).
 * Off by default -> bit-exact to the exact-only model.
 */
static int ngram_skip(void)
{
  const char* env = getenv("CONVERSE_SKIP");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


/* Interpolation weight for the skip tier (0..1). */
static double ngram_skip_mu(void)
{
  double result = 0.3;
  const char* env = getenv("CONVERSE_SKIP_MU");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v >= 0.0 && v <= 1.0) result = v;
  }
  return result;
}


/**
 * Record a skip-gram observation from a rolling history: keys the pair
 * (hist[hlen-3], hist[hlen-1]) to the successor, requiring hlen >= 3.
 */
static void ngram_skip_observe(const unsigned int hist[], int hlen,
  unsigned int succ_id)
{
  if (0 != converse_skip_on && hlen >= 3) {
    unsigned int pair[2];
    pair[0] = hist[hlen - 3];
    pair[1] = hist[hlen - 1];
    if (0 != pair[0] && 0 != pair[1]) {
      libxs_ngram_observe(&converse_skip, pair, 2, succ_id);
    }
  }
}


/**
 * Skip-tier probability of succ given the rolling history, or 0 when the tier
 * is off, the context is too short, or the pattern was never seen.
 */
static double ngram_skip_prob(const unsigned int hist[], int hlen,
  unsigned int succ_id)
{
  double result = 0.0;
  if (0 != converse_skip_on && hlen >= 3) {
    unsigned int pair[2];
    pair[0] = hist[hlen - 3];
    pair[1] = hist[hlen - 1];
    if (0 != pair[0] && 0 != pair[1]) {
      const libxs_ngram_entry_t* entry =
        libxs_ngram_lookup(&converse_skip, pair, 2, 2);
      if (NULL != entry && entry->total > 0) {
        unsigned int slot;
        for (slot = 0; slot < entry->nsucc; ++slot) {
          if (entry->succ[slot].id == succ_id) {
            result = (double)entry->succ[slot].count / (double)entry->total;
            break;
          }
        }
      }
    }
  }
  return result;
}


/**
 * Whether the skip tier can speak for this context at all: the pair exists and
 * was observed. Target-independent by construction, so the bank can mark the
 * slot active without reading the answer.
 */
static int ngram_skip_ready(const unsigned int hist[], int hlen)
{
  int result = 0;
  if (0 != converse_skip_on && hlen >= 3) {
    unsigned int pair[2];
    pair[0] = hist[hlen - 3];
    pair[1] = hist[hlen - 1];
    if (0 != pair[0] && 0 != pair[1]) {
      const libxs_ngram_entry_t* entry =
        libxs_ngram_lookup(&converse_skip, pair, 2, 2);
      if (NULL != entry && entry->total > 0) result = 1;
    }
  }
  return result;
}


static void ngram_hist_push(unsigned int hist[], int* hlen, int cap,
  unsigned int id)
{
  if (*hlen < cap) hist[(*hlen)++] = id;
  else {
    int s;
    for (s = 1; s < cap; ++s) hist[s - 1] = hist[s];
    hist[cap - 1] = id;
  }
}


/**
 * Word-span context for predicting sub-word token i: the whole-word ids of the
 * preceding wctx words followed by the pieces of the current word emitted so
 * far, kept as the most-recent cap entries. Rebuilt per position so that train
 * and eval derive identical keys; groups are delimited by LIBXS_LEXEME_BREAK.
 */
static int ngram_wordctx_hist(const libxs_lexeme_t nat[],
  const unsigned int word_ids[], int i, int wctx, unsigned int hist[], int cap)
{
  int hlen = 0;
  int wstart = i;
  unsigned int words[NGRAM_ORDER_MAX];
  int nw = 0, p, k;
  while (wstart > 0 && 0 == (nat[wstart].flags & LIBXS_LEXEME_BREAK)) --wstart;
  p = wstart;
  while (p > 0 && nw < wctx && nw < (int)(sizeof(words) / sizeof(*words))) {
    int ws = p - 1;
    while (ws > 0 && 0 == (nat[ws].flags & LIBXS_LEXEME_BREAK)) --ws;
    words[nw++] = word_ids[ws];
    p = ws;
  }
  for (k = nw - 1; k >= 0; --k) ngram_hist_push(hist, &hlen, cap, words[k]);
  for (k = wstart; k < i; ++k) ngram_hist_push(hist, &hlen, cap, nat[k].id);
  return hlen;
}


/* Per-order footprint of the n-gram store (gated by CONVERSE_NGRAM_STATS). */
static void ngram_stats(const libxs_registry_t* model)
{
  const char* env = getenv("CONVERSE_NGRAM_STATS");
  libxs_ngram_stats_t st;
  st.entries = 0;
  st.nbytes = 0;
  LIBXS_UNUSED(model);
  if (NULL != env && '\0' != *env && '0' != *env
    && EXIT_SUCCESS == libxs_ngram_stats(&converse_ngram, &st))
  {
    int n;
    for (n = 1; n <= LIBXS_NGRAM_ORDER_MAX; ++n) {
      if (st.keys[n] > 0) {
        fprintf(stdout,
          "ngram-stats[n=%d]: keys=%ld obs=%.0f obs/key=%.2f full=%.1f%%\n",
          n, st.keys[n], st.obs[n], st.obs[n] / (double)st.keys[n],
          100.0 * (double)st.saturated[n] / (double)st.keys[n]);
      }
    }
    if (st.entries > 0) {
      fprintf(stdout,
        "ngram-stats[all]: entries=%lu bytes=%lu bytes/entry=%.1f\n",
        (unsigned long)st.entries, (unsigned long)st.nbytes,
        (double)st.nbytes / (double)st.entries);
    }
  }
}


static void ngram_train_text(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len)
{
  int maxorder = ngram_maxorder();
  if (0 != ngram_native_mode()) {
    int wctx = ngram_wordctx();
    libxs_lexeme_t nat[COMPOSE_MAXTEXT];
    unsigned int word_ids[COMPOSE_MAXTEXT];
    int ntok = ngram_native_tokens(lexicon, text, text_len, nat,
      (0 != wctx) ? word_ids : NULL, COMPOSE_MAXTEXT, 1);
    unsigned int hist[NGRAM_ORDER_MAX];
    int hlen = 0, i;
    if (NULL == model) return;
    for (i = 0; i < ntok; ++i) {
      if (0 != wctx) {
        hlen = ngram_wordctx_hist(nat, word_ids, i, wctx, hist,
          NGRAM_ORDER_MAX);
        if (hlen > 0) ngramk_observe(model, hist, hlen, nat[i].id, maxorder);
      }
      else {
        if (hlen > 0) ngramk_observe(model, hist, hlen, nat[i].id, maxorder);
        ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, nat[i].id);
      }
    }
    return;
  }
  { libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    if (NULL != model && NULL != lexicon && NULL != rules && nrules > 0
      && text_len > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
        &stream, (const unsigned char*)text, (size_t)text_len, rules, nrules,
        answer_lexnorms, answer_lexnorms_size, 1))
    {
      size_t pos;
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (hlen > 0) ngramk_observe(model, hist, hlen, lex->id, maxorder);
          ngram_skip_observe(hist, hlen, lex->id);
          if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = lex->id;
          else {
            int s;
            for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
            hist[NGRAM_ORDER_MAX - 1] = lex->id;
          }
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
      }
    }
    libxs_lexeme_stream_release(&stream);
  }
}


static int predict_is_test(long index, int holdout)
{
  const char* tail;
  if (holdout <= 0) return 0;
  tail = getenv("CONVERSE_HOLDOUT_TAIL");
  if (NULL != tail && '0' != tail[0] && predict_ntotal > 0) {
    long split = predict_ntotal - predict_ntotal / (long)holdout;
    return (index >= split) ? 1 : 0;
  }
  return (0 == (index % (long)holdout)) ? 1 : 0;
}


static int predict_eval_stride(void)
{
  int result = TOKEN_PREDICT_EVAL_STRIDE;
  const char* env = getenv("CONVERSE_EVAL_STRIDE");
  if (NULL != env && '\0' != *env) {
    int value = atoi(env);
    if (value > 0) result = value;
  }
  return result;
}


static libxs_registry_t* ngram_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  libxs_ngram_destroy(&converse_ngram);
  if (0 != converse_skip_on) libxs_ngram_destroy(&converse_skip);
  converse_skip_on = ngram_skip();
  if (EXIT_SUCCESS != libxs_ngram_create(&converse_ngram, ngram_maxorder())) {
    return NULL;
  }
  if (0 != converse_skip_on
    && EXIT_SUCCESS != libxs_ngram_create(&converse_skip, 2))
  {
    converse_skip_on = 0;
  }
  if (NULL != corpus) {
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    const int onescale = ngram_dedup_scale();
    void* value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      if (0 == predict_is_test(index, holdout)
        && (0 == onescale
          || (((2 == onescale) ? SCALE_PARAGRAPH : SCALE_SENTENCE)
              == entry->scale
            && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))))
      {
        ngram_train_text(converse_ngram.store, lexicon, rules, nrules,
          entry->text, entry->text_len);
      }
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
  }
  return converse_ngram.store;
}


/**
 * Rank the successors of a single record by count (converse's legacy order-2
 * adapters expose a bare entry; the library ranks internally via predict).
 */
static int ngram_topk(const ngram_entry_t* entry, unsigned int out_ids[],
  int k)
{
  int result = 0;
  if (NULL != entry && NULL != out_ids && k > 0) {
    unsigned int taken[NGRAM_SUCC_MAX];
    unsigned int nsucc = entry->nsucc;
    unsigned int slot;
    for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) taken[slot] = 0;
    while (result < k) {
      int best = -1;
      for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) {
        if (0 == taken[slot]
          && (best < 0 || entry->succ[slot].count > entry->succ[best].count))
        {
          best = (int)slot;
        }
      }
      if (best < 0) break;
      taken[best] = 1;
      out_ids[result] = entry->succ[best].id;
      ++result;
    }
  }
  return result;
}


static void ngram_backoff_build(libxs_registry_t* model,
  const libxs_lexicon_t* lexicon)
{
  unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  LIBXS_UNUSED(model);
  libxs_ngram_finalize(&converse_ngram, vocab);
  if (0 != converse_skip_on) libxs_ngram_finalize(&converse_skip, vocab);
}


static double ngram_unigram_prior(unsigned int id)
{
  double result = 0.0;
  if (NULL != converse_ngram.unifreq && 0 != id
    && id <= converse_ngram.unifreq_size)
  {
    result = (double)converse_ngram.unifreq[id] / converse_ngram.unifreq_total;
  }
  return result;
}


static double ngram_pair_relfreq(libxs_registry_t* model, unsigned int ctx_a,
  unsigned int ctx_b, unsigned int succ_id)
{
  double result = 0.0;
  const ngram_entry_t* entry = ngram_lookup(model, ctx_a, ctx_b);
  if (NULL != entry && entry->total > 0) {
    unsigned int slot;
    for (slot = 0; slot < entry->nsucc; ++slot) {
      if (entry->succ[slot].id == succ_id) {
        result = (double)entry->succ[slot].count / (double)entry->total;
        break;
      }
    }
  }
  return result;
}


static double* token_emb = NULL;
static double* token_semb = NULL;
static double* token_emb_prior = NULL;
static double* token_emb_work = NULL;
static unsigned int token_emb_size = 0;
/**
 * Memo for the successor normalizer. Z depends on the CONTEXT only, so scoring k
 * candidates at one position recomputed the same vocabulary pass k times -- which
 * is invisible in the BPC path (one candidate per position) and made generation,
 * where the bank ranks up to GEN_CAND_MAX candidates, cost k vocabulary scans per
 * token. A pure memo: identical values, not an approximation.
 */
static unsigned int token_emb_zids[TOKEN_CTX_MAX];
static int token_emb_znctx = 0;
static unsigned int token_emb_zvocab = 0;
static double token_emb_ztemp = 0.0;
static double token_emb_zmax = 0.0;
static double token_emb_zsum = 0.0;
static int token_emb_zvalid = 0;


static void token_emb_free(void)
{
  free(token_emb);
  free(token_semb);
  free(token_emb_prior);
  free(token_emb_work);
  token_emb = NULL;
  token_semb = NULL;
  token_emb_prior = NULL;
  token_emb_work = NULL;
  token_emb_size = 0;
  token_emb_zvalid = 0;
}


/**
 * Maximum FORWARD distance at which a co-occurrence is counted, or 0 for the
 * historical symmetric window.
 *
 * The nine flat axes all varied how the SYMMETRIC PPMI factorization is built or
 * searched -- radius, rank, iterations, hashing, weighting, temperature, heads,
 * projections -- and never which matrix is factorized. So the objective itself
 * was never a variable: the representation is fitted to "what appears near
 * what" and then asked "what comes next here". At 1 the matrix becomes
 * PPMI(next=j | cur=i), whose row space places two tokens together when they are
 * FOLLOWED by similar things, which is the predictive geometry rather than the
 * topical one. It discards distances 2..radius, which the flat radius axis
 * (1..5 all 48.7% top-1) measured at zero.
 */
static int token_emb_directed(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_EMB_DIRECTED");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
    if (cached > TOKEN_EMB_WINDOW) cached = TOKEN_EMB_WINDOW;
  }
  return cached;
}


/**
 * Count ONLY forward distance exactly CONVERSE_EMB_DIRECTED, instead of pooling
 * 1..N.
 *
 * This exists to answer one question before a much larger build. Extending the
 * slot's context by summing vectors is refuted: the optimal weight on a token
 * further back is zero, because token_emb[p] answers "what follows p" and asking
 * that of a token k positions back is a question about the wrong relation. The
 * principled repair is a SEPARATE factorization per distance, so distance k's
 * evidence is about position k. That costs k SVDs and k vocabulary-sized tables,
 * so it should only be built if a distance-k-only model carries real information
 * about the successor at all -- which is what this measures.
 */
static int token_emb_distonly(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_EMB_DISTONLY");
    cached = (NULL != env && '\0' != *env && 0 != atoi(env)) ? 1 : 0;
  }
  return cached;
}


static int token_emb_ready(void)
{
  return (NULL != token_emb && 0 != token_emb_size) ? 1 : 0;
}


/** Successor-side row of V: the geometry of what PRECEDES this token. */
static const double* token_semb_get(unsigned int id)
{
  static const double zero[TOKEN_EMB_DIM] = { 0 };
  if (NULL != token_semb && 0 != id && id <= token_emb_size) {
    return token_semb + (size_t)id * TOKEN_EMB_DIM;
  }
  return zero;
}


static const double* token_emb_get(unsigned int id)
{
  static const double zero[TOKEN_EMB_DIM] = { 0 };
  if (NULL != token_emb && 0 != id && id <= token_emb_size) {
    return token_emb + (size_t)id * TOKEN_EMB_DIM;
  }
  return zero;
}


/**
 * Non-zero if the id has no usable embedding: out of range, or an all-zero row
 * (a word that never occurred in the training split has an empty PPMI row, so
 * it contributes nothing to a retrieval query). This is the headroom available
 * to subword composition.
 */
static int token_emb_isnull(unsigned int id)
{
  const double* emb = token_emb_get(id);
  int d, result = 1;
  for (d = 0; d < TOKEN_EMB_DIM && 0 != result; ++d) {
    if (0.0 != emb[d]) result = 0;
  }
  return result;
}


static double ngram_emb_temp(void)
{
  static int done = 0;
  static double cached = 1.0;
  if (0 == done) {
    const char* env = getenv("CONVERSE_EMB_TEMP");
    const double v = (NULL != env && '\0' != *env) ? atof(env) : 1.0;
    cached = (v > 0.0) ? v : 1.0;
    done = 1;
  }
  return cached;
}


/**
 * How many tokens of history the successor distribution conditions on (1 = the
 * immediate predecessor only, which is where this expert started).
 *
 * Summing the context vectors is not the retired "decayed bag": under this link
 * it is the naive-Bayes combination of independent evidence. Because
 * <sum_i u_i, v_c> = sum_i PMI(p_i, c), adding context in embedding space
 * multiplies the per-position likelihood ratios, so each history token
 * contributes its own testimony about the successor. What the flat position-
 * weighting axis retired was averaging TOPICAL vectors to form a retrieval query;
 * these vectors are successor-predictive and the sum has a defined meaning.
 *
 * The magnitude grows with the number of terms, which is correct for evidence
 * accumulation (more testimony, sharper posterior) but does interact with the
 * temperature, so the two want sweeping together rather than in isolation.
 */
static int ngram_emb_ctx(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_EMB_CTX");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 1;
    if (cached < 1) cached = 1;
    if (cached > TOKEN_CTX_MAX) cached = TOKEN_CTX_MAX;
  }
  return cached;
}


/** Per-position weight decay, oldest token weighted decay^(distance-1). */
static double ngram_emb_decay(void)
{
  static int done = 0;
  static double cached = 1.0;
  if (0 == done) {
    const char* env = getenv("CONVERSE_EMB_DECAY");
    const double v = (NULL != env && '\0' != *env) ? atof(env) : 1.0;
    cached = (v > 0.0 && v <= 1.0) ? v : 1.0;
    done = 1;
  }
  return cached;
}


/**
 * P(cand | ctx) from the successor score, as a proper distribution over the
 * whole vocabulary.
 *
 * The link is not an arbitrary softmax. PMI is a log ratio,
 * PMI(p,c) = log[P(c|p)/P(c)], so exponentiating it RESCALES THE UNIGRAM PRIOR:
 * P(c|p) = P(c) * exp(PMI(p,c)) / Z. Two things follow that a bare softmax over
 * the scores would not give. Where the reconstruction says nothing the expert
 * degrades exactly to the unigram prior rather than to uniform, which is what
 * lets it be TOTAL without being absurd on common words; and the temperature has
 * a defined neutral value of 1 instead of needing to be fitted before the
 * mechanism can be judged at all.
 *
 * Cost is one vocabulary pass with an exp per entry, per position. That is the
 * price of a total scorer and the reason this is behind a slot knob.
 */
static int token_emb_succ_prepare(const unsigned int ctx[], int nctx,
  unsigned int vocab, double temp)
{
  int result = 0;
  if (NULL != token_emb && NULL != token_semb && NULL != token_emb_prior
    && NULL != token_emb_work && vocab <= token_emb_size && 0 < nctx
    && nctx <= TOKEN_CTX_MAX)
  {
    int same = (0 != token_emb_zvalid && vocab == token_emb_zvocab
      && temp == token_emb_ztemp && nctx == token_emb_znctx) ? 1 : 0;
    int i;
    for (i = 0; i < nctx && 0 != same; ++i) {
      if (ctx[i] != token_emb_zids[i]) same = 0;
    }
    if (0 == same) {
      const double scale = (temp > 0.0) ? (1.0 / temp) : 1.0;
      const double decay = ngram_emb_decay();
      double u[TOKEN_EMB_DIM];
      double max = 0.0, sum = 0.0, w = 1.0;
      unsigned int id, nvalid = 0;
      int d;
      for (d = 0; d < TOKEN_EMB_DIM; ++d) u[d] = 0.0;
      /* Walk from the immediate predecessor backwards so weight 1 is the nearest
         token and the decay applies to older testimony. */
      for (i = nctx - 1; i >= 0; --i) {
        const double* e = token_emb_get(ctx[i]);
        for (d = 0; d < TOKEN_EMB_DIM; ++d) u[d] += w * e[d];
        w *= decay;
      }
      for (id = 1; id <= vocab; ++id) {
        if (token_emb_prior[id] > 0.0) {
          const double* v = token_semb + (size_t)id * TOKEN_EMB_DIM;
          double x = 0.0;
          for (d = 0; d < TOKEN_EMB_DIM; ++d) x += u[d] * v[d];
          token_emb_work[id] = x * scale + log(token_emb_prior[id]);
          if (0 == nvalid || token_emb_work[id] > max) max = token_emb_work[id];
          ++nvalid;
        }
      }
      for (id = 1; id <= vocab; ++id) {
        if (token_emb_prior[id] > 0.0) sum += exp(token_emb_work[id] - max);
      }
      for (i = 0; i < nctx; ++i) token_emb_zids[i] = ctx[i];
      token_emb_znctx = nctx;
      token_emb_zvocab = vocab;
      token_emb_ztemp = temp;
      token_emb_zmax = max;
      token_emb_zsum = sum;
      token_emb_zvalid = 1;
    }
    result = (token_emb_zsum > 0.0) ? 1 : 0;
  }
  return result;
}


static double token_emb_succ_prob(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab, double temp)
{
  double result = 0.0;
  if (0 != cand && cand <= vocab
    && 0 != token_emb_succ_prepare(ctx, nctx, vocab, temp)
    && token_emb_prior[cand] > 0.0)
  {
    result = exp(token_emb_work[cand] - token_emb_zmax) / token_emb_zsum;
  }
  return result;
}


/**
 * Append up to want embedding-proposed successors to the candidate list, best
 * first, skipping ids already offered by the counts.
 *
 * This is what makes the successor embedding reach generation at all. The bank
 * can only REORDER what it is given, so with an attested-only candidate list its
 * measured ceiling is the inlist share (22.23% on novel context) no matter how
 * good the expert is. Proposing is a different act from ranking, and only this
 * one can put a successor in front of the caller that the counts never saw here.
 *
 * The selection reads token_emb_work, which the memo has already filled with the
 * whole log-distribution for this context, so no dot product is repeated: it is
 * want passes of a comparison over the vocabulary.
 */
static int token_emb_succ_append(const unsigned int ctx[], int nctx,
  unsigned int vocab, double temp, unsigned int ids[], int n, int max, int want)
{
  int result = 0;
  if (0 != token_emb_succ_prepare(ctx, nctx, vocab, temp)) {
    int more = 1;
    while (0 != more && result < want && (n + result) < max) {
      unsigned int best = 0;
      double bestval = 0.0;
      unsigned int id;
      for (id = 1; id <= vocab; ++id) {
        if (token_emb_prior[id] > 0.0
          && (0 == best || token_emb_work[id] > bestval))
        {
          int taken = 0, k;
          for (k = 0; k < n + result && 0 == taken; ++k) {
            if (ids[k] == id) taken = 1;
          }
          if (0 == taken) {
            best = id;
            bestval = token_emb_work[id];
          }
        }
      }
      if (0 != best) {
        ids[n + result] = best;
        ++result;
      }
      else more = 0;
    }
  }
  return result;
}


/**
 * Rank of cand among all vocabulary ids by the successor score, 0 = best. No
 * sort: the rank is the number of ids that strictly outscore it, which is one
 * pass and needs no candidate list at all -- the point being that this scorer is
 * TOTAL where the count model is partial.
 */
static int token_emb_succ_rank(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab)
{
  int result = -1;
  /* Shares the prepared distribution so the probe scores exactly what the slot
     scores: a probe that built its own context vector could report a mechanism
     nobody is running. */
  if (0 != cand && cand <= vocab
    && 0 != token_emb_succ_prepare(ctx, nctx, vocab, ngram_emb_temp())
    && token_emb_prior[cand] > 0.0)
  {
    const double best = token_emb_work[cand];
    unsigned int id;
    int rank = 0;
    for (id = 1; id <= vocab; ++id) {
      if (id != cand && token_emb_prior[id] > 0.0
        && token_emb_work[id] > best)
      {
        ++rank;
      }
    }
    result = rank;
  }
  return result;
}


static void token_emb_pair_observe(libxs_registry_t* pairs,
  unsigned int center, unsigned int context)
{
  token_emb_pair_t key;
  double* count;
  key.center = center;
  key.context = context;
  count = (double*)libxs_registry_get(pairs, &key, sizeof(key), NULL);
  if (NULL != count) *count += 1.0;
  else {
    double one = 1.0;
    libxs_registry_set(pairs, &key, sizeof(key), &one, sizeof(one), NULL);
  }
}


static void token_emb_cooc_text(libxs_registry_t* pairs, double* rowcnt,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len)
{
  libxs_lexeme_stream_t stream;
  libxs_lexeme_stream_init(&stream);
  if (NULL != pairs && NULL != lexicon && NULL != rules && nrules > 0
    && text_len > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, (const unsigned char*)text, (size_t)text_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 0))
  {
    /* Trailing buffer of the last TOKEN_EMB_WINDOW ids: each token pairs with
       every buffered predecessor and both directions are counted, so the union
       over positions is a symmetric window of that same radius (no 2*W+1). */
    unsigned int window[TOKEN_EMB_WINDOW];
    int fill = 0;
    size_t pos;
    for (pos = 0; pos <= stream.size; ++pos) {
      const libxs_lexeme_t* lex = (pos < stream.size)
        ? (stream.data + pos) : NULL;
      int boundary = (NULL == lex
        || 0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) ? 1 : 0;
      if (NULL != lex && 0 == boundary
        && 0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        const int dir = token_emb_directed();
        const int distonly = token_emb_distonly();
        int i;
        for (i = 0; i < fill; ++i) {
          if (0 == dir) {
            token_emb_pair_observe(pairs, lex->id, window[i]);
            token_emb_pair_observe(pairs, window[i], lex->id);
          }
          else if ((0 == distonly && (fill - i) <= dir)
            || (0 != distonly && (fill - i) == dir))
          {
            /* Row = predecessor, column = successor, so the row of the matrix
               is the successor profile of that token and nothing symmetrizes
               it. window[fill-1] is the immediate predecessor. */
            token_emb_pair_observe(pairs, window[i], lex->id);
          }
        }
        if (fill < TOKEN_EMB_WINDOW) {
          window[fill] = lex->id;
          ++fill;
        }
        else {
          for (i = 1; i < fill; ++i) window[i - 1] = window[i];
          window[fill - 1] = lex->id;
        }
        if (NULL != rowcnt) rowcnt[lex->id] += 1.0;
      }
      if (0 != boundary) fill = 0;
    }
  }
  libxs_lexeme_stream_release(&stream);
}


/* Multiply the sparse PPMI matrix (CSR) by a dense (vocab+1) x DIM block:
   out = A * in when transpose is zero, out = A^T * in otherwise. */
static void token_emb_spmm(const size_t* rowptr, const unsigned int* colidx,
  const double* val, unsigned int vocab, const double* in, int transpose,
  double* out)
{
  size_t n = ((size_t)vocab + 1) * TOKEN_EMB_DIM, k;
  unsigned int id;
  for (k = 0; k < n; ++k) out[k] = 0.0;
  for (id = 1; id <= vocab; ++id) {
    size_t nz;
    for (nz = rowptr[id]; nz < rowptr[id + 1]; ++nz) {
      const double v = val[nz];
      if (0.0 != v) {
        const size_t src = (size_t)(0 != transpose ? id : colidx[nz])
          * TOKEN_EMB_DIM;
        const size_t dst = (size_t)(0 != transpose ? colidx[nz] : id)
          * TOKEN_EMB_DIM;
        int d;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) out[dst + d] += v * in[src + d];
      }
    }
  }
}


/* Gram-Schmidt orthonormalization of the DIM columns of a (vocab+1) x DIM
   block, in place. */
static void token_emb_orthonormalize(double* block, unsigned int vocab)
{
  const size_t rows = (size_t)vocab + 1;
  int d, e;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) {
    double norm = 0.0;
    size_t i;
    for (e = 0; e < d; ++e) {
      double dot = 0.0;
      for (i = 0; i < rows; ++i) {
        dot += block[i * TOKEN_EMB_DIM + d] * block[i * TOKEN_EMB_DIM + e];
      }
      for (i = 0; i < rows; ++i) {
        block[i * TOKEN_EMB_DIM + d] -= dot * block[i * TOKEN_EMB_DIM + e];
      }
    }
    for (i = 0; i < rows; ++i) {
      norm += block[i * TOKEN_EMB_DIM + d] * block[i * TOKEN_EMB_DIM + d];
    }
    norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
    for (i = 0; i < rows; ++i) block[i * TOKEN_EMB_DIM + d] *= norm;
  }
}


/**
 * Truncated SVD of the sparse PPMI matrix by subspace iteration on A^T A,
 * which is never materialized: each iteration applies A and then A^T, so the
 * cost is O(nnz * DIM) and the full context vocabulary is carried exactly.
 * Embeddings are the projected rows A*V (that is U*Sigma), row-normalized.
 */
static int token_emb_reduce(const size_t* rowptr, const unsigned int* colidx,
  const double* val, unsigned int vocab)
{
  int result = EXIT_FAILURE;
  const size_t block = ((size_t)vocab + 1) * TOKEN_EMB_DIM;
  double* basis = (double*)malloc(block * sizeof(double));
  double* work = (double*)malloc(block * sizeof(double));
  if (NULL != basis && NULL != work) {
    /* Symmetry-breaking start: LIBXS_SHUFFLE_INDEX is a coprime affine map,
       hence a bijection onto [0, block), so the values are an evenly spread
       (not clustered) permutation and reproducible across runs. */
    const size_t stride = libxs_coprime_bias(block, -1.0);
    unsigned int id, iter;
    size_t i;
    int d;
    for (i = 0; i < block; ++i) {
      const size_t k = LIBXS_SHUFFLE_INDEX(i, block, stride, 1);
      basis[i] = (2.0 * (double)k / (double)block) - 1.0;
    }
    token_emb_orthonormalize(basis, vocab);
    for (iter = 0; iter < TOKEN_EMB_ITER; ++iter) {
      token_emb_spmm(rowptr, colidx, val, vocab, basis, 0, work);
      token_emb_spmm(rowptr, colidx, val, vocab, work, 1, basis);
      token_emb_orthonormalize(basis, vocab);
    }
    token_emb_spmm(rowptr, colidx, val, vocab, basis, 0, work);
    for (id = 1; id <= vocab; ++id) {
      const double* row = work + (size_t)id * TOKEN_EMB_DIM;
      double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
      double norm = 0.0;
      for (d = 0; d < TOKEN_EMB_DIM; ++d) norm += row[d] * row[d];
      norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
      for (d = 0; d < TOKEN_EMB_DIM; ++d) emb[d] = row[d] * norm;
    }
    /**
     * basis still holds V after the final projection, so the successor side is
     * free: it was already computed and then thrown away with the scratch. With
     * the directed matrix <token_emb[p], token_semb[c]> is the rank-DIM
     * reconstruction of PPMI(c follows p), and a low-rank completion is nonzero
     * on pairs NEVER OBSERVED -- the one property a count model cannot have.
     *
     * Kept UNNORMALIZED while token_emb rows are unit-normalized, so the product
     * is not calibrated as a probability: within one context the row scale is a
     * positive constant and drops out of any ranking, but the magnitude that was
     * normalized away was confidence, and a fixed temperature does not recover
     * it.
     */
    if (NULL != token_semb) {
      for (id = 1; id <= vocab; ++id) {
        const double* row = basis + (size_t)id * TOKEN_EMB_DIM;
        double* semb = token_semb + (size_t)id * TOKEN_EMB_DIM;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) semb[d] = row[d];
      }
    }
    result = EXIT_SUCCESS;
  }
  free(basis);
  free(work);
  return result;
}


static void token_emb_backfill(libxs_lexicon_t* lexicon,
  const double* rowcnt, unsigned int vocab)
{
  unsigned int id;
  for (id = 1; id <= vocab; ++id) {
    unsigned int flags = 0;
    int len = 0;
    const char* text = libxs_lexicon_text(lexicon, id, &len, &flags);
    if (NULL != text && len > 0 && len <= LIBXS_LEXEME_MAXBYTES
      && 0 != (flags & LIBXS_LEXEME_WORD)
      && rowcnt[id] < (double)TOKEN_EMB_BACKFILL_MIN)
    {
      char word[LIBXS_LEXEME_MAXBYTES + 1];
      char cand[LIBXS_LEXEME_MAXBYTES + 1];
      unsigned int other, best_id = 0;
      int best_dist = 1 + len / 4;
      memcpy(word, text, (size_t)len);
      word[len] = '\0';
      for (other = 1; other <= vocab; ++other) {
        unsigned int cand_flags = 0;
        int cand_len = 0;
        const char* cand_text = libxs_lexicon_text(lexicon, other, &cand_len,
          &cand_flags);
        if (other != id && NULL != cand_text && cand_len > 0
          && cand_len <= LIBXS_LEXEME_MAXBYTES
          && 0 != (cand_flags & LIBXS_LEXEME_WORD)
          && rowcnt[other] >= (double)TOKEN_EMB_BACKFILL_REF
          && (word[0] | 32) == (cand_text[0] | 32)
          && cand_len > len - best_dist && cand_len < len + best_dist)
        {
          int dist;
          memcpy(cand, cand_text, (size_t)cand_len);
          cand[cand_len] = '\0';
          dist = libxs_stridist(word, cand);
          if (dist >= 0 && dist < best_dist) {
            best_dist = dist;
            best_id = other;
          }
        }
      }
      if (0 != best_id) {
        double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
        const double* ref = token_emb + (size_t)best_id * TOKEN_EMB_DIM;
        double w = rowcnt[id] / (rowcnt[id] + 2.0);
        double norm = 0.0;
        int d;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) {
          emb[d] = w * emb[d] + (1.0 - w) * ref[d];
          norm += emb[d] * emb[d];
        }
        norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) emb[d] *= norm;
      }
    }
  }
}


/**
 * Print the split of comma-separated words, so the splitter can be inspected on
 * the words it is known to get wrong without running a corpus. The unit was
 * never fairly evaluated because the output was broken; a probe makes the repair
 * checkable in one command instead of inferred from a BPC delta.
 */
static void ngram_syllable_probe(void)
{
  const char* probe = getenv("CONVERSE_SYLLABLE_PROBE");
  while (NULL != probe && '\0' != *probe) {
    const char* end = strchr(probe, ',');
    const int len = (NULL != end) ? (int)(end - probe) : (int)strlen(probe);
    if (0 < len && LIBXS_LEXEME_MAXBYTES >= len) {
      int begin[LIBXS_LEXEME_MAXBYTES], plen[LIBXS_LEXEME_MAXBYTES];
      const int np = ngram_syllable_split(probe, len, begin, plen,
        LIBXS_LEXEME_MAXBYTES);
      int k;
      fprintf(stderr, "syllable[%.*s]:", len, probe);
      for (k = 0; k < np; ++k) {
        fprintf(stderr, "%s%.*s", (0 < k) ? "|" : " ", plen[k],
          probe + begin[k]);
      }
      fprintf(stderr, " (%d pieces)\n", np);
    }
    probe = (NULL != end) ? (end + 1) : NULL;
  }
}


static void token_emb_probe(libxs_lexicon_t* lexicon, unsigned int vocab)
{
  const char* probe = getenv("CONVERSE_EMB_PROBE");
  char word[LIBXS_LEXEME_MAXBYTES + 1];
  int len;
  while (NULL != probe && '\0' != *probe) {
    const char* end = strchr(probe, ',');
    unsigned int id;
    len = (NULL != end) ? (int)(end - probe) : (int)strlen(probe);
    if (len > 0 && len <= LIBXS_LEXEME_MAXBYTES) {
      memcpy(word, probe, (size_t)len);
      word[len] = '\0';
      id = libxs_lexicon_id(lexicon, word, len, 0, 0);
      if (0 != id && id <= vocab) {
        const double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
        unsigned int best[5] = { 0 }, other, slot, nbest = 0;
        double best_sim[5] = { 0 };
        fprintf(stderr, "emb[%s]:", word);
        for (other = 1; other <= vocab; ++other) {
          if (other != id) {
            const double* cand = token_emb + (size_t)other * TOKEN_EMB_DIM;
            double sim = 0.0;
            int d;
            for (d = 0; d < TOKEN_EMB_DIM; ++d) sim += emb[d] * cand[d];
            for (slot = 0; slot < nbest; ++slot) {
              if (sim > best_sim[slot]) break;
            }
            if (slot < 5) {
              unsigned int move;
              if (nbest < 5) ++nbest;
              for (move = nbest - 1; move > slot; --move) {
                best[move] = best[move - 1];
                best_sim[move] = best_sim[move - 1];
              }
              best[slot] = other;
              best_sim[slot] = sim;
            }
          }
        }
        for (slot = 0; slot < nbest; ++slot) {
          int blen = 0;
          const char* btext = libxs_lexicon_text(lexicon, best[slot], &blen,
            NULL);
          if (NULL != btext && blen > 0) {
            fprintf(stderr, " %.*s(%.2f)", blen, btext, best_sim[slot]);
          }
        }
        fprintf(stderr, "\n");
      }
      else fprintf(stderr, "emb[%s]: not in lexicon\n", word);
    }
    probe = (NULL != end) ? (end + 1) : (probe + len);
  }
}


static void token_emb_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  token_emb_free();
  if (NULL != corpus && vocab > 0) {
    libxs_registry_t* pairs = libxs_registry_create();
    double* rowcnt = (double*)calloc((size_t)vocab + 1, sizeof(double));
    double* rowsum = (double*)calloc((size_t)vocab + 1, sizeof(double));
    double* colsum = (double*)calloc((size_t)vocab + 1, sizeof(double));
    size_t* rowptr = (size_t*)calloc((size_t)vocab + 2, sizeof(size_t));
    token_emb = (double*)calloc((size_t)(vocab + 1) * TOKEN_EMB_DIM,
      sizeof(double));
    token_semb = (double*)calloc((size_t)(vocab + 1) * TOKEN_EMB_DIM,
      sizeof(double));
    token_emb_prior = (double*)calloc((size_t)vocab + 1, sizeof(double));
    token_emb_work = (double*)calloc((size_t)vocab + 1, sizeof(double));
    if (NULL != pairs && NULL != rowcnt && NULL != rowsum && NULL != colsum
      && NULL != rowptr && NULL != token_emb && NULL != token_semb
      && NULL != token_emb_prior && NULL != token_emb_work)
    {
      double total = 0.0;
      const void* key = NULL;
      size_t cursor = 0, nnz = 0;
      long index = 0;
      unsigned int id;
      void* value = libxs_registry_begin(corpus, &key, &cursor);
      token_emb_size = vocab;
      while (NULL != value) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        if (0 == predict_is_test(index, holdout)) {
          token_emb_cooc_text(pairs, rowcnt, lexicon, rules, nrules,
            entry->text, entry->text_len);
        }
        ++index;
        value = libxs_registry_next(corpus, &key, &cursor);
      }
      /* rowcnt was already accumulated for the backfill, so the unigram prior
         the successor distribution rescales costs one normalization. */
      { double ntok = 0.0;
        for (id = 1; id <= vocab; ++id) ntok += rowcnt[id];
        if (ntok > 0.0) {
          for (id = 1; id <= vocab; ++id) {
            token_emb_prior[id] = rowcnt[id] / ntok;
          }
        }
      }
      key = NULL;
      cursor = 0;
      value = libxs_registry_begin(pairs, &key, &cursor);
      while (NULL != value) {
        const token_emb_pair_t* pair = (const token_emb_pair_t*)key;
        const double count = *(const double*)value;
        if (pair->center <= vocab && pair->context <= vocab) {
          rowsum[pair->center] += count;
          colsum[pair->context] += count;
          total += count;
          ++rowptr[pair->center + 1];
          ++nnz;
        }
        value = libxs_registry_next(pairs, &key, &cursor);
      }
      for (id = 1; id <= vocab + 1; ++id) rowptr[id] += rowptr[id - 1];
      if (total > 0.0 && nnz > 0) {
        unsigned int* colidx = (unsigned int*)malloc(nnz * sizeof(*colidx));
        double* val = (double*)malloc(nnz * sizeof(*val));
        size_t* fill = (size_t*)malloc(((size_t)vocab + 1) * sizeof(*fill));
        if (NULL != colidx && NULL != val && NULL != fill) {
          const char* env;
          for (id = 0; id <= vocab; ++id) fill[id] = rowptr[id];
          key = NULL;
          cursor = 0;
          value = libxs_registry_begin(pairs, &key, &cursor);
          while (NULL != value) {
            const token_emb_pair_t* pair = (const token_emb_pair_t*)key;
            const double count = *(const double*)value;
            if (pair->center <= vocab && pair->context <= vocab
              && rowsum[pair->center] > 0.0 && colsum[pair->context] > 0.0)
            {
              const double pmi = log((count * total)
                / (rowsum[pair->center] * colsum[pair->context]));
              const size_t at = fill[pair->center]++;
              colidx[at] = pair->context;
              val[at] = (pmi > 0.0) ? pmi : 0.0;
            }
            value = libxs_registry_next(pairs, &key, &cursor);
          }
          fprintf(stderr, "embedding: vocab=%u nnz=%lu density=%.4f%%"
            " dim=%d window=%d directed=%d%s\n", vocab, (unsigned long)nnz,
            100.0 * (double)nnz / ((double)vocab * (double)vocab),
            TOKEN_EMB_DIM, TOKEN_EMB_WINDOW, token_emb_directed(),
            (0 != token_emb_distonly()) ? " (exact distance)" : "");
          if (EXIT_SUCCESS == token_emb_reduce(rowptr, colidx, val, vocab)) {
            env = getenv("CONVERSE_EMB_BACKFILL");
            if (NULL == env || '0' != *env) {
              token_emb_backfill(lexicon, rowcnt, vocab);
            }
            token_emb_probe(lexicon, vocab);
          }
        }
        free(colidx);
        free(val);
        free(fill);
      }
    }
    libxs_registry_destroy(pairs);
    free(rowcnt);
    free(rowsum);
    free(colsum);
    free(rowptr);
  }
}


static int token_input_vector(unsigned int prev2, unsigned int prev1,
  int use_emb, double inputs[])
{
  int result;
  if (0 != use_emb) {
    const double* emb2 = token_emb_get(prev2);
    const double* emb1 = token_emb_get(prev1);
    int dim;
    for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
      inputs[dim] = emb2[dim];
      inputs[TOKEN_EMB_DIM + dim] = emb1[dim];
    }
    result = 2 * TOKEN_EMB_DIM;
  }
  else {
    inputs[0] = (double)prev2;
    inputs[1] = (double)prev1;
    result = 2;
  }
  return result;
}


/**
 * Number of context tokens summarized into a kNN-LM query vector (>=2; 2 is
 * the historical prev2/prev1 pair, bit-exact). Wider context = longer reach.
 */
static int knnlm_ctxlen(void)
{
  int result = 2;
  const char* env = getenv("CONVERSE_KNNLM_CTX");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 2 && v <= TOKEN_CTX_MAX) result = v;
  }
  return result;
}


/* Geometric decay applied to older context tokens in the summarized half. */
static double knnlm_decay(void)
{
  double result = 0.5;
  const char* env = getenv("CONVERSE_KNNLM_DECAY");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v >= 0.0 && v <= 1.0) result = v;
  }
  return result;
}


/**
 * Build a kNN-LM query vector from a rolling history (most-recent last). The
 * prev1 half is the embedding of the most recent token; the prev2 half is a
 * decayed, L2-normalized sum of the preceding ctxlen-1 tokens. With ctxlen==2
 * this reproduces token_input_vector(prev2, prev1, 1, .) exactly, so the
 * default path is byte-identical to the two-token model.
 */
/**
 * Per-position weights for the summarized half, replacing the geometric decay
 * when CONVERSE_KNNLM_WEIGHTS is set (comma-separated, nearest position first).
 * Returns the number parsed, or 0 to keep the geometric profile. A weight
 * profile is the scalar case of a position-specific projection: if position
 * identity carries usable signal, an explicit profile must beat the geometric
 * one, which is a far cheaper test than fitting matrices.
 */
static int knnlm_weights(double out[], int max)
{
  const char* env = getenv("CONVERSE_KNNLM_WEIGHTS");
  int result = 0;
  if (NULL != env && '\0' != *env) {
    const char* p = env;
    while (result < max && '\0' != *p) {
      out[result++] = atof(p);
      while ('\0' != *p && ',' != *p) ++p;
      if (',' == *p) ++p;
    }
  }
  return result;
}


static void knnlm_ctx_vector(const unsigned int hist[], int hlen, int ctxlen,
  double decay, double inputs[])
{
  const double* emb1 = token_emb_get((hlen > 0) ? hist[hlen - 1] : 0);
  double profile[TOKEN_CTX_MAX];
  int nprofile = knnlm_weights(profile, TOKEN_CTX_MAX);
  int dim, back;
  double weight = 1.0, norm = 0.0;
  for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
    inputs[TOKEN_EMB_DIM + dim] = emb1[dim];
    inputs[dim] = 0.0;
  }
  for (back = 2; back <= ctxlen; ++back) {
    int idx = hlen - back;
    if (idx >= 0) {
      const double* emb = token_emb_get(hist[idx]);
      const double w = (back - 2) < nprofile ? profile[back - 2] : weight;
      for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) inputs[dim] += w * emb[dim];
    }
    weight *= decay;
  }
  for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) norm += inputs[dim] * inputs[dim];
  if (norm > 0.0) {
    double inv = 1.0 / sqrt(norm);
    for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) inputs[dim] *= inv;
  }
}


static double knnlm_cosine(const double* a, const double* b, int n)
{
  double dot = 0.0, na = 0.0, nb = 0.0;
  int i;
  for (i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na > 0.0 && nb > 0.0) ? (dot / (sqrt(na) * sqrt(nb))) : 0.0;
}


/**
 * Quantify how much word order survives in a query vector: for each evaluated
 * context, compare the vector against the one built from the SAME tokens in a
 * swapped order. Cosine 1.0 means order was discarded entirely; lower means the
 * representation distinguishes the permutation. Reports the mean over contexts,
 * separately for a swap inside the summarized half (positions 2 and 3) and for
 * a swap that crosses into the most-recent slot (positions 1 and 2).
 */
static void knnlm_order_probe(const unsigned int hist[], int hlen, int ctxlen,
  double decay, double* sum_inner, double* sum_cross, long* n_inner,
  long* n_cross)
{
  double base[2 * TOKEN_EMB_DIM], perm[2 * TOKEN_EMB_DIM];
  unsigned int swapped[TOKEN_CTX_MAX];
  int i;
  if (hlen < 2) return;
  for (i = 0; i < hlen && i < TOKEN_CTX_MAX; ++i) swapped[i] = hist[i];
  knnlm_ctx_vector(hist, hlen, ctxlen, decay, base);
  if (hlen >= 3 && ctxlen >= 3 && hist[hlen - 2] != hist[hlen - 3]) {
    swapped[hlen - 2] = hist[hlen - 3];
    swapped[hlen - 3] = hist[hlen - 2];
    knnlm_ctx_vector(swapped, hlen, ctxlen, decay, perm);
    *sum_inner += knnlm_cosine(base, perm, 2 * TOKEN_EMB_DIM);
    ++*n_inner;
    swapped[hlen - 2] = hist[hlen - 2];
    swapped[hlen - 3] = hist[hlen - 3];
  }
  if (hist[hlen - 1] != hist[hlen - 2]) {
    swapped[hlen - 1] = hist[hlen - 2];
    swapped[hlen - 2] = hist[hlen - 1];
    knnlm_ctx_vector(swapped, hlen, ctxlen, decay, perm);
    *sum_cross += knnlm_cosine(base, perm, 2 * TOKEN_EMB_DIM);
    ++*n_cross;
  }
}


static const ngram_entry_t* ngram_lookup(libxs_registry_t* model,
  unsigned int ctx_a, unsigned int ctx_b)
{
  const ngram_entry_t* result = NULL;
  if (NULL != model && 0 != ctx_b) {
    unsigned int hist[2];
    if (0 == ctx_a) {
      hist[0] = ctx_b;
      result = ngramk_lookup(model, hist, 1, 1);
    }
    else {
      hist[0] = ctx_a;
      hist[1] = ctx_b;
      result = ngramk_lookup(model, hist, 2, 2);
    }
  }
  return result;
}


static int ngram_predict(libxs_registry_t* model, unsigned int prev2,
  unsigned int prev1, int order, unsigned int out_ids[], int k)
{
  int result = 0;
  if (NULL != model && NULL != out_ids && k > 0) {
    const ngram_entry_t* entry = NULL;
    if (order >= 2 && 0 != prev2) entry = ngram_lookup(model, prev2, prev1);
    if (NULL == entry) entry = ngram_lookup(model, 0, prev1);
    if (NULL != entry) result = ngram_topk(entry, out_ids, k);
    if (0 == result) {
      int slot;
      for (slot = 0; slot < k && slot < converse_ngram.backoff_count; ++slot) {
        out_ids[slot] = converse_ngram.backoff_ids[slot];
        ++result;
      }
    }
  }
  return result;
}


static void ngram_last_context(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  unsigned int* prev2, unsigned int* prev1)
{
  libxs_lexeme_stream_t stream;
  unsigned int p2 = 0, p1 = 0;
  libxs_lexeme_stream_init(&stream);
  if (NULL != lexicon && NULL != rules && nrules > 0 && text_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, (size_t)text_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 0))
  {
    size_t pos;
    for (pos = 0; pos < stream.size; ++pos) {
      const libxs_lexeme_t* lex = stream.data + pos;
      if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        p2 = p1;
        p1 = lex->id;
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  if (NULL != prev2) *prev2 = p2;
  if (NULL != prev1) *prev1 = p1;
}


/**
 * Fill hist[] with the trailing content-token ids of the prompt (at most
 * NGRAM_ORDER_MAX), returning the count. Mirrors ngram_last_context but keeps
 * the whole window the deep store can use, not just the final two ids.
 */
static int ngram_history(libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, const char* text, int text_len, unsigned int hist[])
{
  libxs_lexeme_stream_t stream;
  const int native = ngram_native_mode();
  int hlen = 0;
  libxs_lexeme_stream_init(&stream);
  /**
   * The prompt has to be read in the unit the model was trained on, or the
   * seeded history holds ids from a different vocabulary and nothing matches.
   */
  if (0 != native) {
    libxs_lexeme_t nat[COMPOSE_MAXTEXT];
    const int ntok = (NULL != lexicon && 0 < text_len)
      ? ngram_native_tokens(lexicon, text, text_len, nat, NULL,
        COMPOSE_MAXTEXT, 0) : 0;
    int ti;
    for (ti = 0; ti < ntok; ++ti) {
      if (0 != nat[ti].id) {
        ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, nat[ti].id);
      }
    }
  }
  else if (NULL != lexicon && NULL != rules && nrules > 0 && text_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, (size_t)text_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 0))
  {
    size_t pos;
    for (pos = 0; pos < stream.size; ++pos) {
      const libxs_lexeme_t* lex = stream.data + pos;
      if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, lex->id);
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return hlen;
}


/**
 * Successors offered to the hierarchical model at each generation step (1 = the
 * count-ranked winner, i.e. plain greedy decoding). Widening this is what gives
 * the byte model anything to say about generation: it never proposes a
 * continuation, it only chooses among ones the n-gram already attested, so the
 * grounding gate below still decides what may be shown.
 */
static int ngram_gen_ncand(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_NCAND");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 1;
    if (cached < 1) cached = 1;
    if (cached > GEN_CAND_MAX) cached = GEN_CAND_MAX;
  }
  return cached;
}


/**
 * Whether gen-eval keeps scoring after the first wrong token instead of ending
 * the sentence there.
 *
 * Off is the historical definition and stays bit-exact: mean-reproduced is the
 * length of the verbatim prefix, so the scan stops at the first miss. That
 * definition has almost no dynamic range on novel seeds -- divergence is at the
 * very first position, so the bucket reads the same 0.06 for every configuration
 * measured so far, which is a property of the metric and not of the mechanism.
 * With this on, every lookahead position is visited with the TRUTH token fed
 * back as history, and the per-position accuracy and rank statistics are
 * computed over all of them. The prefix metrics keep their old definition
 * either way, so one run reports both readings -- exactly so while the bank is
 * off. With the bank on, the extra positions also feed ngram_bank_update, so
 * the weights differ from a prefix-mode run and the prefix metrics are then
 * comparable in meaning but not to the digit.
 */
static int ngram_gen_full(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_FULL");
    cached = (NULL != env && '\0' != *env && 0 != atoi(env)) ? 1 : 0;
  }
  return cached;
}


/**
 * Whether gen-eval also ranks the truth by the successor-side embedding, over the
 * WHOLE vocabulary and independently of what the count model proposed.
 *
 * This is the probe that decides the directed representation BEFORE any of it is
 * wired into a predictor. The count model carries the truth in its proposal on
 * only 22% of novel-context positions, so the question that matters is not
 * whether the embedding reranks better -- it is whether a TOTAL scorer can rank
 * the truth at all where the partial one had nothing to offer. Costs one
 * vocabulary scan per position, which is why it is a knob and not always on.
 */
static int ngram_gen_embrank(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_EMBRANK");
    cached = (NULL != env && '\0' != *env && 0 != atoi(env)) ? 1 : 0;
  }
  return cached;
}


/**
 * How many embedding-proposed successors join the candidate set per step, 0 =
 * off.
 *
 * OFF BY DEFAULT BECAUSE IT CHANGES THE PROMISE, not because it is unfinished.
 * Every other generation tier emits a successor the n-gram ATTESTED in this
 * context, which is what makes the output traceable to corpus text; the one
 * existing exception, recombination's composer, is opt-in for exactly this
 * reason. A candidate proposed by the low-rank completion was never observed
 * after this context, so with this on the path SYNTHESIZES rather than SELECTS
 * and the caller has to have asked for that.
 *
 * Secondary consequence to keep in mind when reading the numbers: a position the
 * grounding floor would have suppressed becomes scoreable once proposals exist,
 * so the count candidates below that floor are admitted alongside them and the
 * prefix metrics are not comparable across this knob.
 */
static int ngram_gen_embcand(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_EMBCAND");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
    if (cached > GEN_CAND_MAX) cached = GEN_CAND_MAX;
  }
  return cached;
}


/**
 * Whether rendering has to supply the word separator itself.
 *
 * The modeling units (native chunks, syllable pieces, byte-pair pieces) bake the
 * boundary into the piece text, so concatenating them reproduces the source and
 * a separator here would duplicate it. The lexical view cannot: word ids come
 * from the shared normalized vocabulary that grounded QA matches terms against,
 * which must stay bare. So this is not a granularity table but the one-bit
 * question "modeling view or lexical view", and it belongs in exactly one place:
 * every site that re-derived it was a chance to render text no corpus contains.
 */
static int ngram_render_separated(void)
{
  return (0 == ngram_native_mode()) ? 1 : 0;
}


/**
 * Append one piece's text at pos, inserting the separator first when the view
 * needs one and the piece may be preceded by something. Returns the new end, or
 * pos unchanged when the piece does not fit or has no text.
 */
static size_t ngram_render_append(char* out, size_t out_size, size_t pos,
  const char* piece, int len, int leading)
{
  size_t result = pos;
  if (NULL != out && NULL != piece && 0 < len
    && pos + (size_t)len + 1 < out_size)
  {
    if (0 != leading && 0 != ngram_render_separated()) out[result++] = ' ';
    memcpy(out + result, piece, (size_t)len);
    result += (size_t)len;
    out[result] = '\0';
  }
  return result;
}


/**
 * Choose among the count-ranked successors with the byte model, returning the
 * index to take. Falls back to the n-gram's own first choice whenever the model
 * is absent or nothing could be scored, so enabling this can only reorder
 * candidates the n-gram already offered.
 */
static int ngram_gen_select(libxs_lexicon_t* lexicon,
  const unsigned int ids[], int nids, const char* context, int context_len)
{
  int result = 0;
  if (1 < nids && NULL != answer_hier_model) {
    const char* candidates[GEN_CAND_MAX];
    int lengths[GEN_CAND_MAX];
    char words[GEN_CAND_MAX][64];
    int slot, chosen;
    for (slot = 0; slot < nids; ++slot) {
      int len = 0;
      const char* word = libxs_lexicon_text(lexicon, ids[slot], &len, NULL);
      /* A candidate always follows the context, so it always takes a leading. */
      const size_t at = ngram_render_append(words[slot], sizeof(words[slot]), 0,
        word, len, 1);
      candidates[slot] = (0 < at) ? words[slot] : NULL;
      lengths[slot] = (int)at;
    }
    chosen = converse_hier_choose(answer_hier_model, context, context_len,
      candidates, lengths, nids);
    if (0 <= chosen) result = chosen;
  }
  return result;
}


/** Text of the first n ids, as scoring context for the byte model. */
static int ngram_gen_join(libxs_lexicon_t* lexicon,
  const unsigned int ids[], int n, char* out, size_t out_size)
{
  size_t pos = 0;
  int slot;
  if (0 < out_size) out[0] = '\0';
  for (slot = 0; slot < n; ++slot) {
    int len = 0;
    const char* word = libxs_lexicon_text(lexicon, ids[slot], &len, NULL);
    pos = ngram_render_append(out, out_size, pos, word, len, (0 < pos) ? 1 : 0);
  }
  return (int)pos;
}


static int ngram_gen_minorder(void)
{
  int result = 2;
  const char* env = getenv("CONVERSE_GEN_MINORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}


/**
 * Mean-order floor a continuation must clear to be shown as attested. The
 * per-step gate (ngram_gen_minorder) is dominated by the low-order seeding
 * transient, so a whole continuation is judged by its MEAN grounding order:
 * at maxorder 2 every run averages 2.0 (pure bigram drift, suppressed); with
 * deeper context (-x) genuinely attested passages average much higher.
 */
/**
 * Whether the -c cascade may synthesize. Off by default: every other tier emits
 * text that occurs verbatim in the corpus, and this one does not, so it is a
 * different promise to the caller and should be asked for.
 */
static int recomb_compose_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    cached = (NULL != getenv("CONVERSE_RECOMB_COMPOSE")) ? 1 : 0;
  }
  return cached;
}


static double ngram_gen_contfloor(void)
{
  double result = 3.0;
  const char* env = getenv("CONVERSE_GEN_CONTORDER");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v >= 1.0 && v <= (double)NGRAM_ORDER_MAX) result = v;
  }
  return result;
}


/**
 * Grounded greedy generation over the deep k-context store. Each step takes
 * the top successor from the longest attested context and reports that
 * context order; generation stops when the order falls below the grounding
 * floor (the generative form of "abstain rather than invent"), at a repeat,
 * or at the length budget. Prints the continuation plus the mean/min order
 * that quantifies how well grounded it was.
 */
/**
 * Greedy grounded continuation of the prompt into out[] (space-joined words).
 * Returns the number of generated tokens (0 = nothing cleared the grounding
 * floor). Optional order_mean/order_min report how attested the run was.
 */
static int ngram_generate(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  char* out, size_t out_size, double* order_mean, int* order_min_out)
{
  enum { GEN_MAX = 40 };
  unsigned int hist[NGRAM_ORDER_MAX];
  unsigned int recent[GEN_MAX];
  int hlen, maxorder = ngram_maxorder();
  int minorder = ngram_gen_minorder();
  int step, nrecent = 0;
  long order_sum = 0;
  int order_min = NGRAM_ORDER_MAX + 1;
  size_t pos = 0;
  if (NULL != out && out_size > 0) out[0] = '\0';
  hlen = ngram_history(lexicon, rules, nrules, text, text_len, hist);
  for (step = 0; hlen > 0 && step < GEN_MAX; ++step) {
    unsigned int ids[GEN_CAND_MAX];
    int got_order = 0, len = 0, i, repeat = 0, pick = 0;
    const char* word;
    int n = ngramk_predict_order(model, hist, hlen, maxorder, ids,
      ngram_gen_ncand(), &got_order);
    if (n <= 0 || got_order < minorder) break;
    if (1 < n && NULL != answer_hier_model) {
      char context[COMPOSE_MAXTEXT];
      int context_len = text_len;
      if (context_len >= (int)sizeof(context)) {
        context_len = (int)sizeof(context) - 1;
      }
      memcpy(context, text + text_len - context_len, (size_t)context_len);
      if (NULL != out && 0 < pos) {
        context_len = (int)ngram_render_append(context, sizeof(context),
          (size_t)context_len, out, (int)pos, 1);
      }
      pick = ngram_gen_select(lexicon, ids, n, context, context_len);
    }
    if (0 != pick) {
      const unsigned int chosen = ids[pick];
      ids[pick] = ids[0];
      ids[0] = chosen;
    }
    for (i = 0; i < nrecent; ++i) if (recent[i] == ids[0]) ++repeat;
    if (repeat >= 3) break;
    word = libxs_lexicon_text(lexicon, ids[0], &len, NULL);
    if (NULL == word || len <= 0) break;
    pos = ngram_render_append(out, out_size, pos, word, len,
      (0 < pos) ? 1 : 0);
    order_sum += got_order;
    if (got_order < order_min) order_min = got_order;
    if (nrecent < GEN_MAX) recent[nrecent++] = ids[0];
    if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = ids[0];
    else {
      int s;
      for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
      hist[NGRAM_ORDER_MAX - 1] = ids[0];
    }
  }
  if (NULL != order_mean) {
    *order_mean = (step > 0) ? (double)order_sum / (double)step : 0.0;
  }
  if (NULL != order_min_out) *order_min_out = (step > 0) ? order_min : 0;
  return step;
}


static void ngram_complete(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len)
{
  char out[COMPOSE_MAXTEXT];
  double order_mean = 0.0;
  int order_min = 0;
  int ntok = ngram_generate(model, lexicon, rules, nrules, text, text_len,
    out, sizeof(out), &order_mean, &order_min);
  int minorder = ngram_gen_minorder();
  LIBXS_UNUSED(order);
  if (ntok > 0) {
    printf("generate: %s\n", out);
    printf("grounding: %d tokens, mean order %.1f, min order %d\n",
      ntok, order_mean, order_min);
  }
  else printf("(no grounded continuation at min order %d)\n", minorder);
}


/**
 * Completion-mode response: for a question, answer from the corpus (facts +
 * retrieval) first, then add a grounded continuation generated over the whole
 * corpus n-gram model and seeded from the ANSWER, so -c is never empty and the
 * continuation extends what was said. Non-questions keep the pure generation
 * probe. Continuation is labeled and its grounding reported.
 */
/**
 * Minimum content-word overlap for a continuation to be shown with an answer.
 * Zero keeps the historical behaviour (attestation only), which is bit-exact.
 */
static double gen_relevance_min(void)
{
  double result = 0.0;
  const char* env = getenv("CONVERSE_GEN_RELEVANCE");
  if (NULL != env && '\0' != *env) {
    const double v = atof(env);
    if (v >= 0.0 && v <= 1.0) result = v;
  }
  return result;
}


/**
 * Content-word overlap between two texts, as a fraction of the smaller content
 * set. The same measure recomb_overlap applies to entry pairs, over lexeme
 * streams instead: the seam here joins an answer to a generated continuation,
 * neither of which is a corpus entry.
 *
 * The grounding gate already in place verifies that a continuation is deeply
 * ATTESTED, which is not the same as being about the same thing -- a deep context
 * match on a common word leads wherever that phrasing went in training, so a
 * fluent continuation can drift to an unrelated scene. Shared content beyond stop
 * words is the cheapest evidence of aboutness, and it needs no new machinery.
 */
static double gen_overlap(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* a, int alen,
  const char* b, int blen)
{
  double result = 0.0;
  libxs_lexeme_stream_t sa, sb;
  libxs_lexeme_stream_init(&sa);
  libxs_lexeme_stream_init(&sb);
  if (EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &sa,
      (const unsigned char*)a, (size_t)alen, rules, nrules, answer_lexnorms,
      answer_lexnorms_size, 0)
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &sb,
      (const unsigned char*)b, (size_t)blen, rules, nrules, answer_lexnorms,
      answer_lexnorms_size, 0))
  {
    size_t ia, ib;
    int na = 0, nb = 0, shared = 0;
    for (ia = 0; ia < sa.size; ++ia) {
      const libxs_lexeme_t* la = sa.data + ia;
      if (0 == la->id || 0 != (la->flags & LIBXS_LEXEME_STOP)) continue;
      if (0 == (la->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))) continue;
      ++na;
      for (ib = 0; ib < sb.size; ++ib) {
        if (sb.data[ib].id == la->id) {
          ++shared;
          break;
        }
      }
    }
    for (ib = 0; ib < sb.size; ++ib) {
      const libxs_lexeme_t* lb = sb.data + ib;
      if (0 != lb->id && 0 == (lb->flags & LIBXS_LEXEME_STOP)
        && 0 != (lb->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER)))
      {
        ++nb;
      }
    }
    { const int smaller = (na < nb) ? na : nb;
      if (0 < smaller) result = (double)shared / (double)smaller;
    }
  }
  libxs_lexeme_stream_release(&sa);
  libxs_lexeme_stream_release(&sb);
  return result;
}


static void complete_respond(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile, int budget,
  const char* text, int text_len)
{
  if (0 != is_question_query(text, (size_t)text_len, lexicon, rules, nrules)) {
    char answer[COMPOSE_MAXTEXT];
    char gen[COMPOSE_MAXTEXT];
    double order_mean = 0.0;
    int order_min = 0;
    int answered = answer_query(corpus, text, (size_t)text_len, budget,
      lexicon, rules, nrules, answer_model, profile, answer, sizeof(answer));
    const char* seed = (0 != answered && '\0' != answer[0]) ? answer : text;
    int seed_len = (0 != answered && '\0' != answer[0])
      ? (int)strlen(answer) : text_len;
    int ntok = ngram_generate(converse_ngram.store, lexicon, rules, nrules,
      seed, seed_len, gen, sizeof(gen), &order_mean, &order_min);
    int grounded = (ntok > 0 && order_mean >= ngram_gen_contfloor()) ? 1 : 0;
    int composed = 0;
    double relevance = -1.0;
    /**
     * Attested is not the same as relevant. The grounding gate above checks that
     * the continuation is deep replay; this checks that it is about the same
     * thing as the answer it is being appended to. Off by default (threshold 0),
     * so the historical output is unchanged unless the gate is asked for.
     */
    if (0 != grounded) {
      const double need = gen_relevance_min();
      relevance = gen_overlap(lexicon, rules, nrules, seed, seed_len, gen,
        (int)strlen(gen));
      if (relevance < need) grounded = 0;
    }
    if (0 != grounded) {
      printf("continuation: %s\n", gen);
      printf("grounding: %d tokens, mean order %.1f, min order %d,"
        " relevance %.2f\n", ntok, order_mean, order_min, relevance);
    }
    /**
     * Composition is the LAST tier, tried only once replay has failed. A
     * continuation is verbatim attested text, so it is better grounded than a
     * splice: offering a synthesized sentence while an attested one was available
     * would trade grounding for novelty without saying so. Seeded from the answer
     * when there is one, because that is the text retrieval already judged
     * relevant to the query.
     */
    if (0 == grounded && 0 != recomb_compose_on()) {
      char syn[COMPOSE_MAXTEXT];
      int nfront = 0, ncand = 0;
      const int slen = converse_recomb_compose_best(corpus, lexicon, rules,
        nrules, seed, seed_len, syn, sizeof(syn), &nfront, &ncand);
      if (0 < slen) {
        composed = 1;
        printf("composed: %s\n", syn);
        /**
         * The provenance line is not decoration. This sentence is in the corpus
         * NOWHERE, so a reader has to be able to tell it from the attested replies
         * above it, and the front size says whether the objective actually chose
         * it or was indifferent among that many.
         */
        printf("synthesis: spliced at a shared term, %d candidate%s,"
          " %d on the front (not attested; every word is)\n",
          ncand, (1 == ncand) ? "" : "s", nfront);
        /**
         * State what was NOT checked. Five separate gate ideas were measured and
         * none can tell a true join from a fluent false one: the evidence that
         * would decide it lies at clause scale, where only 2% of four-word content
         * spans recur even at 27x this corpus. So the honest output is a caveat
         * rather than a confidence, and it is worded as the specific thing left
         * unverified -- "sense of the shared term" -- because a generic disclaimer
         * on every line is quickly ignored.
         *
         * The front size carries the rest: when the objective is indifferent among
         * most candidates it did not really choose, and saying so is what keeps
         * this consistent with the abstention discipline the QA side uses.
         */
        printf("unverified: the shared term may carry a different sense in each"
          " half; no gate here checks that%s\n",
          (nfront > 1 && nfront * 2 >= ncand)
            ? ", and the objective was near-indifferent among these" : "");
      }
    }
    if (0 == answered && 0 == grounded && 0 == composed) {
      printf("I do not know from the corpus.\n");
    }
  }
  else {
    ngram_complete(converse_ngram.store, lexicon, rules, nrules, 0,
      text, text_len);
  }
}


/**
 * Held-out generation quality: seed each test sentence with its first GEN_SEED
 * content tokens, greedily extend, and count how many of the sentence's actual
 * remaining tokens the generator reproduces before diverging. A gradient-free,
 * generation-native counterpart to BPC (which only scores one-step prediction).
 * Also reports the mean grounding order over generated tokens.
 */
/**
 * Reorder generative candidates by the expert bank instead of by raw count
 * rank. Generation is where a slot has to earn its keep for real: BPC scores a
 * distribution against a known target, whereas here the model must put the
 * right word FIRST with no target in sight. A slot that lowers BPC but cannot
 * change the argmax has not changed what the system produces.
 *
 * Weights are carried across positions exactly as in scoring, but the update is
 * driven by the CHOSEN token, not the true one -- no target information may
 * enter generation. The bank therefore adapts to its own trajectory here, which
 * is the honest online setting.
 */
static int ngram_gen_bank_rank(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int ids[], int n,
  unsigned int slots, const libxs_predict_t* store, void* context,
  int use_emb, int vocabulary, double weight[])
{
  int result = 0;
  if (1 < n) {
    double best = -1.0;
    int i;
    for (i = 0; i < n; ++i) {
      ngram_expert_t expert[NGRAM_BANK_MAX];
      double pooled;
      ngram_bank_experts(hist, hlen, maxorder, ids[i], slots, store, context,
        use_emb, vocabulary, expert);
      pooled = ngram_bank_pool(weight, expert);
      if (pooled > best) {
        best = pooled;
        result = i;
      }
    }
  }
  return result;
}


/**
 * One bucket of the per-position generation report.
 *
 * inlist is the share of scored positions where the truth appears ANYWHERE in
 * the offered candidates. It separates the two ways a generator can be wrong:
 * top1 - inlist is what better ranking could still win, and 100 - inlist is what
 * no reranking can reach because the truth was never proposed. The count model
 * offers only attested successors, so on novel context the second term is the
 * one that binds, and a mechanism aimed at the first would be misdirected.
 */
static void gen_bucket_report(const char* label, long n, long top1,
  long inlist, long abst, double rr)
{
  fprintf(stderr, "%s n=%ld top1=%.2f%% mrr=%.4f inlist=%.2f%%"
    " abstain=%.2f%%\n", label, n,
    (n > 0) ? 100.0 * (double)top1 / (double)n : 0.0,
    (n > 0) ? rr / (double)n : 0.0,
    (n > 0) ? 100.0 * (double)inlist / (double)n : 0.0,
    (0 < n + abst) ? 100.0 * (double)abst / (double)(n + abst) : 0.0);
}


static void gen_embrank_report(const char* label, long n, long top1,
  long top10, double rr)
{
  fprintf(stderr, "%s n=%ld top1=%.2f%% top10=%.2f%% mrr=%.4f\n", label, n,
    (n > 0) ? 100.0 * (double)top1 / (double)n : 0.0,
    (n > 0) ? 100.0 * (double)top10 / (double)n : 0.0,
    (n > 0) ? rr / (double)n : 0.0);
}


static int ngram_gen_eval(libxs_registry_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int holdout, const char* kind,
  const libxs_predict_t* store, int use_emb)
{
  enum { GEN_SEED = 3, GEN_LOOK = 20 };
  int result = EXIT_FAILURE;
  long nsent = 0, sum_repro = 0, gen_tokens = 0, order_sum = 0, index = 0;
  long nsent_att = 0, sum_repro_att = 0, nsent_nov = 0, sum_repro_nov = 0;
  int maxorder = ngram_maxorder();
  int minorder = ngram_gen_minorder();
  const int gen_full = ngram_gen_full();
  const int gen_ncand = ngram_gen_ncand();
  long pos_n = 0, pos_top1 = 0, pos_abst = 0, pos_order = 0, pos_inl = 0;
  long pos_n_att = 0, pos_top1_att = 0, pos_abst_att = 0, pos_inl_att = 0;
  long pos_n_nov = 0, pos_top1_nov = 0, pos_abst_nov = 0, pos_inl_nov = 0;
  double pos_rr = 0.0, pos_rr_att = 0.0, pos_rr_nov = 0.0;
  long ctx_n_att = 0, ctx_top1_att = 0, ctx_abst_att = 0, ctx_inl_att = 0;
  long ctx_n_nov = 0, ctx_top1_nov = 0, ctx_abst_nov = 0, ctx_inl_nov = 0;
  double ctx_rr_att = 0.0, ctx_rr_nov = 0.0;
  const int gen_embrank = ngram_gen_embrank();
  const int gen_embcand = ngram_gen_embcand();
  const int gen_embctx = ngram_emb_ctx();
  /**
   * Candidate budget is FIXED at GEN_CAND_MAX, so proposals take slots from the
   * counts rather than enlarging the list. Two reasons. Without it the counts can
   * fill the list at exactly the positions proposals are for -- a match at order
   * 1 still returns a full set while failing the grounding floor, so there was no
   * room left and 53249 positions kept abstaining. And a longer list would raise
   * inlist for free, which would make the comparison against the attested-only
   * baseline meaningless: at matched width, a change in inlist is a real
   * substitution.
   */
  const int gen_ncand_cnt = (0 < gen_embcand)
    ? ((GEN_CAND_MAX - gen_embcand) > 1 ? (GEN_CAND_MAX - gen_embcand) : 1)
    : 0;
  long pos_embadd = 0, pos_embpick = 0, pos_embpick_nov = 0, pos_resc = 0;
  long emb_n_att = 0, emb_top1_att = 0, emb_top10_att = 0;
  long emb_n_nov = 0, emb_top1_nov = 0, emb_top10_nov = 0;
  long emb_n_dec = 0, emb_top1_dec = 0, emb_top10_dec = 0;
  long emb_n_spk = 0, emb_top1_spk = 0, emb_top10_spk = 0;
  double emb_rr_att = 0.0, emb_rr_nov = 0.0;
  double emb_rr_dec = 0.0, emb_rr_spk = 0.0;
  long gen_ranked = 0, gen_reordered = 0;
  const int gen_bank = ngram_bank_probe();
  const unsigned int bank_slots = ngram_bank_slots();
  const int gen_vocab = (int)libxs_lexicon_size(lexicon);
  double bank_weight[NGRAM_BANK_MAX];
  void* gen_context = NULL;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  { int k, nslot = 0;
    for (k = 1; k <= NGRAM_BANK_LAST; ++k) {
      if (0 != ngram_bank_enabled(bank_slots, k)) ++nslot;
    }
    for (k = 0; k < NGRAM_BANK_MAX; ++k) {
      bank_weight[k] = (k >= 1 && k <= NGRAM_BANK_LAST
        && 0 != ngram_bank_enabled(bank_slots, k)) ? 1.0 / (double)nslot : 0.0;
    }
  }
  if (0 != gen_bank && 0 != (bank_slots & (1u << NGRAM_BANK_PREDICT))
    && NULL != store)
  {
    gen_context = libxs_predict_prob_create(store);
  }

  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_t stream;
    const int native = ngram_native_mode();
    libxs_lexeme_stream_init(&stream);
    /**
     * The truth tokens must come from the SAME unit the model was trained on.
     * Reading word lexemes while the model holds syllable or byte-pair ids makes
     * every prediction a miss and reports a reproduction of zero, which reads as
     * a catastrophically bad generator rather than as a unit mismatch.
     */
    if (0 != is_test && entry->text_len > 0
      && (0 != native || EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
        &stream, (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, answer_lexnorms, answer_lexnorms_size, 0)))
    {
      unsigned int truth[GEN_SEED + GEN_LOOK];
      libxs_lexeme_t nat[COMPOSE_MAXTEXT];
      int ntruth = 0;
      int ntok, ti;
      ntok = (0 != native)
        ? ngram_native_tokens(lexicon, entry->text, entry->text_len, nat,
          NULL, COMPOSE_MAXTEXT, 0)
        : (int)stream.size;
      for (ti = 0; ti < ntok && ntruth < GEN_SEED + GEN_LOOK; ++ti) {
        if (0 != native) {
          if (0 != nat[ti].id) truth[ntruth++] = nat[ti].id;
        }
        else {
          const libxs_lexeme_t* lex = stream.data + ti;
          if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 != lex->id)
          {
            truth[ntruth++] = lex->id;
          }
        }
      }
      if (ntruth > GEN_SEED) {
        unsigned int hist[NGRAM_ORDER_MAX];
        const int seedcap = (GEN_SEED < maxorder) ? GEN_SEED : maxorder;
        int hlen = 0, t, repro = 0, diverged = 0, seed_attested = -1;
        int stop = 0, spoke = 0, hit = 0, abst = 0, ordsum = 0, inl = 0;
        double rrsum = 0.0;
        for (t = 0; t < GEN_SEED; ++t) hist[hlen++] = truth[t];
        for (t = GEN_SEED; t < ntruth && 0 == stop; ++t) {
          unsigned int ids[GEN_CAND_MAX];
          unsigned int embids[GEN_CAND_MAX];
          int got_order = 0;
          int ctxatt = 0, nemb = 0, cdecl;
          int n = ngramk_predict_order(model, hist, hlen, maxorder, ids,
            (0 < gen_embcand && gen_ncand > gen_ncand_cnt)
              ? gen_ncand_cnt : gen_ncand, &got_order);
          /* Proposals join BEFORE the bank so the pool arbitrates the union
             rather than being handed a decision already made. */
          if (0 < gen_embcand) {
            const int nctx = (gen_embctx < hlen) ? gen_embctx : hlen;
            int k;
            nemb = token_emb_succ_append(hist + (hlen - nctx), nctx,
              (unsigned int)gen_vocab, ngram_emb_temp(), ids, n, GEN_CAND_MAX,
              gen_embcand);
            for (k = 0; k < nemb; ++k) embids[k] = ids[n + k];
            n += nemb;
            pos_embadd += nemb;
          }
          if (1 < n && NULL != answer_hier_model) {
            char context[COMPOSE_MAXTEXT];
            const int context_len = ngram_gen_join(lexicon, truth, t,
              context, sizeof(context));
            const int pick = ngram_gen_select(lexicon, ids, n, context,
              context_len);
            if (0 != pick) {
              const unsigned int chosen = ids[pick];
              ids[pick] = ids[0];
              ids[0] = chosen;
            }
          }
          if (0 != gen_bank && 1 < n) {
            const int pick = ngram_gen_bank_rank(model, hist, hlen, maxorder,
              ids, n, bank_slots, store, gen_context, use_emb, gen_vocab,
              bank_weight);
            ++gen_ranked;
            if (0 != pick) ++gen_reordered;
            if (0 != pick) {
              const unsigned int chosen = ids[pick];
              ids[pick] = ids[0];
              ids[0] = chosen;
            }
            /* Adapt on the CHOSEN token: no target may enter generation. */
            { ngram_expert_t expert[NGRAM_BANK_MAX];
              double pooled;
              ngram_bank_experts(hist, hlen, maxorder, ids[0], bank_slots,
                store, gen_context, use_emb, gen_vocab, expert);
              pooled = ngram_bank_pool(bank_weight, expert);
              ngram_bank_update(bank_weight, expert, pooled,
                ngram_bank_rate(), ngram_bank_share());
            }
          }
          /**
           * Classify the sentence by its SEED context only, before any token is
           * generated, so the split cannot be influenced by how far generation
           * then runs: was the whole seed attested as a context in training?
           */
          if (seed_attested < 0) {
            seed_attested = (n > 0 && got_order >= seedcap) ? 1 : 0;
          }
          /**
           * Abstention is a position the model declined, not a wrong answer, so
           * it is counted apart from accuracy rather than folded in as a miss.
           * Under the prefix definition it also ends the sentence, which is why
           * the old scan could report a short run without saying why it stopped.
           */
          /**
           * Second split, per POSITION rather than per seed: did the FULL-ORDER
           * context of THIS position recur in training. That is the definition
           * ngram_eval's attested/novel BPC split already uses, so the two
           * instruments become readable together; the seed split classifies a
           * whole sentence by its first three words and stops discriminating
           * once true history is fed back.
           */
          ctxatt = (got_order >= maxorder) ? 1 : 0;
          /**
           * Whether the COUNTS declined, judged on the count-supplied part of the
           * list only. With no proposals nemb is zero and this is exactly the
           * historical condition, which is what keeps the knob-off path
           * bit-exact; with proposals it separates "counts had nothing" from
           * "nobody had anything".
           */
          cdecl = ((n - nemb) <= 0 || got_order < minorder) ? 1 : 0;
          /**
           * Scored at EVERY visited position, including the ones the count model
           * declined: a total scorer must be judged where the partial one was
           * silent, or the comparison quietly excludes the positions the whole
           * mechanism is for.
           */
          if (0 != gen_embrank) {
            const int nctx = (gen_embctx < hlen) ? gen_embctx : hlen;
            const int erank = token_emb_succ_rank(hist + (hlen - nctx), nctx,
              truth[t], (unsigned int)gen_vocab);
            /**
             * The control that decides whether a total scorer is worth anything:
             * its accuracy ON THE POSITIONS THE COUNT MODEL DECLINED. Averaged
             * over all positions it could earn its score entirely where counts
             * already speak, which would make it a redundant expert rather than a
             * reach into the bucket that has no evidence at all.
             */
            if (0 <= erank) {
              if (0 != cdecl) {
                ++emb_n_dec;
                if (0 == erank) ++emb_top1_dec;
                if (10 > erank) ++emb_top10_dec;
                emb_rr_dec += 1.0 / (double)(erank + 1);
              }
              else {
                ++emb_n_spk;
                if (0 == erank) ++emb_top1_spk;
                if (10 > erank) ++emb_top10_spk;
                emb_rr_spk += 1.0 / (double)(erank + 1);
              }
              if (0 != ctxatt) {
                ++emb_n_att;
                if (0 == erank) ++emb_top1_att;
                if (10 > erank) ++emb_top10_att;
                emb_rr_att += 1.0 / (double)(erank + 1);
              }
              else {
                ++emb_n_nov;
                if (0 == erank) ++emb_top1_nov;
                if (10 > erank) ++emb_top10_nov;
                emb_rr_nov += 1.0 / (double)(erank + 1);
              }
            }
          }
          if (0 != cdecl && 0 == nemb) {
            ++abst;
            if (0 != ctxatt) ++ctx_abst_att;
            else ++ctx_abst_nov;
            /**
             * The prefix definition ends the run here, so the prefix metrics
             * must stop accumulating even when the scan continues -- otherwise
             * positions past an abstention extend mean-reproduced and the two
             * readings stop being comparable (measured: 7.48 -> 7.68 on grimm).
             */
            diverged = 1;
            if (0 == gen_full) stop = 1;
          }
          else {
            /**
             * Rank of the truth within the offered candidates, AFTER every
             * reranking, so it scores the ranking the model actually produces.
             * It is -1 when the truth was not offered at all: the count model
             * proposes only attested successors, so on novel contexts that is
             * the common case and it must not contribute to the reciprocal rank.
             */
            int rank = -1, c;
            for (c = 0; c < n && rank < 0; ++c) {
              if (ids[c] == truth[t]) rank = c;
            }
            if (0 != cdecl) ++pos_resc;
            /* Whether what generation actually EMITTED is a successor no count
               context attested here -- the share of output that is synthesized
               rather than selected, which is the cost side of this mode. */
            { int fromemb = 0, k;
              for (k = 0; k < nemb && 0 == fromemb; ++k) {
                if (embids[k] == ids[0]) fromemb = 1;
              }
              if (0 != fromemb) {
                ++pos_embpick;
                if (0 == ctxatt) ++pos_embpick_nov;
              }
            }
            if (0 == diverged) {
              ++gen_tokens;
              order_sum += got_order;
              if (0 == rank) ++repro;
              else diverged = 1;
            }
            ++spoke;
            ordsum += got_order;
            if (0 == rank) ++hit;
            if (0 <= rank) {
              ++inl;
              rrsum += 1.0 / (double)(rank + 1);
            }
            if (0 != ctxatt) {
              ++ctx_n_att;
              if (0 == rank) ++ctx_top1_att;
              if (0 <= rank) {
                ++ctx_inl_att;
                ctx_rr_att += 1.0 / (double)(rank + 1);
              }
            }
            else {
              ++ctx_n_nov;
              if (0 == rank) ++ctx_top1_nov;
              if (0 <= rank) {
                ++ctx_inl_nov;
                ctx_rr_nov += 1.0 / (double)(rank + 1);
              }
            }
            if (0 == gen_full) stop = diverged;
          }
          if (0 == stop) {
            if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = truth[t];
            else {
              int s;
              for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
              hist[NGRAM_ORDER_MAX - 1] = truth[t];
            }
          }
        }
        sum_repro += repro;
        ++nsent;
        pos_n += spoke;
        pos_top1 += hit;
        pos_abst += abst;
        pos_order += ordsum;
        pos_inl += inl;
        pos_rr += rrsum;
        if (1 == seed_attested) {
          sum_repro_att += repro;
          ++nsent_att;
          pos_n_att += spoke;
          pos_top1_att += hit;
          pos_abst_att += abst;
          pos_inl_att += inl;
          pos_rr_att += rrsum;
        }
        else {
          sum_repro_nov += repro;
          ++nsent_nov;
          pos_n_nov += spoke;
          pos_top1_nov += hit;
          pos_abst_nov += abst;
          pos_inl_nov += inl;
          pos_rr_nov += rrsum;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  libxs_predict_prob_destroy(gen_context);
  if (nsent > 0) {
    fprintf(stdout,
      "gen-eval[%s%s]: sentences=%ld mean-reproduced=%.2f "
      "mean-order=%.2f minorder=%d%s\n",
      kind, (holdout > 0) ? ":heldout" : "", nsent,
      (double)sum_repro / (double)nsent,
      (gen_tokens > 0) ? (double)order_sum / (double)gen_tokens : 0.0,
      minorder, (0 != gen_bank) ? " bank" : "");
    if (0 != gen_bank) {
      fprintf(stderr, "  bank rerank: %ld of %ld positions reordered"
        " (%.1f%%)\n", gen_reordered, gen_ranked,
        (gen_ranked > 0) ? 100.0 * (double)gen_reordered / (double)gen_ranked
          : 0.0);
    }
    fprintf(stderr, "  seed split: attested %.1f%% of sentences"
      " (mean-reproduced=%.2f) | novel %.1f%% (mean-reproduced=%.2f)\n",
      100.0 * (double)nsent_att / (double)nsent,
      (nsent_att > 0) ? (double)sum_repro_att / (double)nsent_att : 0.0,
      100.0 * (double)nsent_nov / (double)nsent,
      (nsent_nov > 0) ? (double)sum_repro_nov / (double)nsent_nov : 0.0);
    /**
     * mrr is the mean reciprocal rank of the truth over the offered candidates,
     * so at ncand=1 (the default) it equals top1 and carries no extra
     * information: widen CONVERSE_GEN_NCAND to give it range. ncand is printed
     * because it decides how to read the column.
     */
    fprintf(stderr, "  per-position[%s]: order=%.2f ncand=%d\n",
      (0 != gen_full) ? "full" : "prefix",
      (pos_n > 0) ? (double)pos_order / (double)pos_n : 0.0, gen_ncand);
    gen_bucket_report("    all:            ", pos_n, pos_top1, pos_inl,
      pos_abst, pos_rr);
    gen_bucket_report("    seed attested:  ", pos_n_att, pos_top1_att,
      pos_inl_att, pos_abst_att, pos_rr_att);
    gen_bucket_report("    seed novel:     ", pos_n_nov, pos_top1_nov,
      pos_inl_nov, pos_abst_nov, pos_rr_nov);
    gen_bucket_report("    ctx attested:   ", ctx_n_att, ctx_top1_att,
      ctx_inl_att, ctx_abst_att, ctx_rr_att);
    gen_bucket_report("    ctx novel:      ", ctx_n_nov, ctx_top1_nov,
      ctx_inl_nov, ctx_abst_nov, ctx_rr_nov);
    if (0 < gen_embcand) {
      fprintf(stderr, "  emb proposals[ncand=%d]: added=%ld rescued=%ld"
        " of %ld scored | EMITTED an unattested successor %.2f%% of scored"
        " (ctx-novel %.2f%%)\n", gen_embcand, pos_embadd, pos_resc, pos_n,
        (pos_n > 0) ? 100.0 * (double)pos_embpick / (double)pos_n : 0.0,
        (ctx_n_nov > 0)
          ? 100.0 * (double)pos_embpick_nov / (double)ctx_n_nov : 0.0);
    }
    if (0 != gen_embrank && 0 < emb_n_att + emb_n_nov) {
      fprintf(stderr, "  embedding rank of truth over vocab=%d"
        " (directed=%d, chance top1=%.4f%%):\n", gen_vocab,
        token_emb_directed(),
        (gen_vocab > 0) ? 100.0 / (double)gen_vocab : 0.0);
      gen_embrank_report("    ctx attested:   ", emb_n_att, emb_top1_att,
        emb_top10_att, emb_rr_att);
      gen_embrank_report("    ctx novel:      ", emb_n_nov, emb_top1_nov,
        emb_top10_nov, emb_rr_nov);
      gen_embrank_report("    counts spoke:   ", emb_n_spk, emb_top1_spk,
        emb_top10_spk, emb_rr_spk);
      gen_embrank_report("    counts DECLINED:", emb_n_dec, emb_top1_dec,
        emb_top10_dec, emb_rr_dec);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


/**
 * Coverage probe for slot abstraction (does NOT predict anything).
 *
 * The question it answers: on positions whose exact context did NOT recur in
 * training -- the only positions that measure generalization -- would a context
 * with ONE content word replaced by a typed hole have recurred? If yes for a
 * large share, then patterns-with-holes are a funded mechanism: they let a novel
 * context match training evidence, which is the type-level sharing that a
 * transformer gets from weights and that exact-context lookup cannot express.
 * If coverage is thin, abstraction cannot reach these positions either and the
 * mechanism is retired for one run instead of one session.
 *
 * A token is a SLOT CANDIDATE when its corpus frequency is at or below
 * CONVERSE_SLOT_MAXFREQ (default 20): frequent tokens are function words that
 * carry the pattern, rare ones are the content the pattern is about. This is
 * deliberately not keyed on the relation rules, so the result is not circular
 * with hand-written vocabulary.
 *
 * "Novel" here matches the definition already used by ngram_eval's attested
 * split: the full-order CONTEXT did not recur, independently of the successor.
 * Coverage, by contrast, requires the abstracted context to have been seen with
 * the SAME successor -- the abstraction has to predict the actual next token, not
 * merely exist. The asymmetry is deliberate: it makes coverage a claim about
 * useful evidence rather than about pattern frequency.
 */
#define SLOT_HOLE_ID 0xFFFFFFFEu


typedef struct slot_key_t {
  unsigned int context[NGRAM_ORDER_MAX];
  unsigned int next;
  int order;
} slot_key_t;

/** Same as slot_key_t without the successor: "was this context ever seen?" */
typedef struct slot_ctx_key_t {
  unsigned int context[NGRAM_ORDER_MAX];
  int order;
} slot_ctx_key_t;


static int slot_maxfreq(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_SLOT_MAXFREQ");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 20;
    if (cached < 1) cached = 1;
  }
  return cached;
}


static void slot_key_build(slot_key_t* key, const unsigned int hist[],
  int hlen, int order, unsigned int next, int hole)
{
  int pos;
  memset(key, 0, sizeof(*key));
  key->order = order;
  key->next = next;
  for (pos = 0; pos < order; ++pos) {
    const unsigned int id = hist[hlen - order + pos];
    key->context[pos] = (pos == hole) ? SLOT_HOLE_ID : id;
  }
}


/**
 * One pass over the corpus, calling back with every (history, successor) pair at
 * the configured order. Shared by the slot probe's train and test passes so both
 * see exactly the same tokenization the n-gram was built from.
 */
static void slot_scan(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout, int want_test,
  void (*visit)(const unsigned int*, int, unsigned int, void*), void* udata)
{
  const void* key = NULL;
  size_t cursor = 0;
  long index = 0;
  void* value = libxs_registry_begin(corpus, &key, &cursor);
  const int maxorder = ngram_maxorder();
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    const int is_test = (0 != predict_is_test(index, holdout)) ? 1 : 0;
    libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    /**
     * Sentence scale only. The corpus holds each text at BOTH sentence and
     * paragraph scale, so scanning every entry sees each sentence twice -- and a
     * paragraph copy of a training sentence would then make that sentence's own
     * contexts look attested, inflating the exact-match share far above what the
     * n-gram model actually holds.
     */
    if (SCALE_SENTENCE == entry->scale
      && (0 == holdout || is_test == want_test) && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
        (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
    {
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      size_t pos;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (hlen >= maxorder) visit(hist, hlen, lex->id, udata);
          ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, lex->id);
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
}


typedef struct slot_probe_t {
  libxs_registry_t* patterns;
  libxs_registry_t* contexts;
  libxs_registry_t* freq;
  libxs_lexicon_t* lexicon;
  int order;
  int building;
  long nnovel;
  long ncovered;
  long nexact;
  long nslotless;
} slot_probe_t;


static long slot_freq_of(const slot_probe_t* probe, unsigned int id)
{
  const long* count = (const long*)libxs_registry_get(probe->freq, &id,
    sizeof(id), NULL);
  return (NULL != count) ? *count : 0;
}


static void slot_count_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata)
{
  slot_probe_t* probe = (slot_probe_t*)udata;
  long* count = (long*)libxs_registry_get(probe->freq, &next, sizeof(next),
    NULL);
  LIBXS_UNUSED(hist); LIBXS_UNUSED(hlen);
  if (NULL != count) ++*count;
  else {
    const long one = 1;
    libxs_registry_set(probe->freq, &next, sizeof(next), &one, sizeof(one),
      NULL);
  }
}


static void slot_train_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata)
{
  slot_probe_t* probe = (slot_probe_t*)udata;
  const int order = probe->order;
  const char present = 1;
  int hole;
  slot_key_t key;
  if (hlen >= order) {
    slot_ctx_key_t ctx;
    int pos;
    memset(&ctx, 0, sizeof(ctx));
    ctx.order = order;
    for (pos = 0; pos < order; ++pos) ctx.context[pos] = hist[hlen - order + pos];
    libxs_registry_set(probe->contexts, &ctx, sizeof(ctx), &present,
      sizeof(present), NULL);
    slot_key_build(&key, hist, hlen, order, next, -1);
    libxs_registry_set(probe->patterns, &key, sizeof(key), &present,
      sizeof(present), NULL);
    for (hole = 0; hole < order; ++hole) {
      if (slot_freq_of(probe, hist[hlen - order + hole]) <= slot_maxfreq()) {
        slot_key_build(&key, hist, hlen, order, next, hole);
        libxs_registry_set(probe->patterns, &key, sizeof(key), &present,
          sizeof(present), NULL);
      }
    }
  }
}


static void slot_test_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata)
{
  slot_probe_t* probe = (slot_probe_t*)udata;
  const int order = probe->order;
  slot_key_t key;
  slot_ctx_key_t ctx;
  int hole, nslots = 0, covered = 0, pos;
  if (hlen < order) return;
  memset(&ctx, 0, sizeof(ctx));
  ctx.order = order;
  for (pos = 0; pos < order; ++pos) ctx.context[pos] = hist[hlen - order + pos];
  if (NULL != libxs_registry_get(probe->contexts, &ctx, sizeof(ctx), NULL)) {
    ++probe->nexact;
    return;
  }
  ++probe->nnovel;
  for (hole = 0; hole < order; ++hole) {
    if (slot_freq_of(probe, hist[hlen - order + hole]) <= slot_maxfreq()) {
      ++nslots;
      slot_key_build(&key, hist, hlen, order, next, hole);
      if (NULL != libxs_registry_get(probe->patterns, &key, sizeof(key), NULL)) {
        covered = 1;
      }
    }
  }
  if (0 == nslots) ++probe->nslotless;
  if (0 != covered) ++probe->ncovered;
}


/**
 * Report what fraction of novel-context positions a one-hole abstraction would
 * have covered. Reported on stdout so it lands beside the other eval lines.
 */
static void slot_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  slot_probe_t probe;
  memset(&probe, 0, sizeof(probe));
  probe.patterns = libxs_registry_create();
  probe.contexts = libxs_registry_create();
  probe.freq = libxs_registry_create();
  probe.lexicon = lexicon;
  probe.order = ngram_maxorder();
  if (NULL != probe.patterns && NULL != probe.contexts
    && NULL != probe.freq)
  {
    slot_scan(corpus, lexicon, rules, nrules, holdout, 0, slot_count_visit,
      &probe);
    slot_scan(corpus, lexicon, rules, nrules, holdout, 0, slot_train_visit,
      &probe);
    slot_scan(corpus, lexicon, rules, nrules, holdout, 1, slot_test_visit,
      &probe);
    { const long total = probe.nexact + probe.nnovel;
      fprintf(stdout, "slot-probe[order=%d maxfreq=%d]: n=%ld"
        " exact=%.1f%% novel=%.1f%%"
        " | of novel: covered=%.1f%% no-slot=%.1f%%\n",
        probe.order, slot_maxfreq(), total,
        (0 < total) ? 100.0 * (double)probe.nexact / (double)total : 0.0,
        (0 < total) ? 100.0 * (double)probe.nnovel / (double)total : 0.0,
        (0 < probe.nnovel)
          ? 100.0 * (double)probe.ncovered / (double)probe.nnovel : 0.0,
        (0 < probe.nnovel)
          ? 100.0 * (double)probe.nslotless / (double)probe.nnovel : 0.0);
    }
  }
  libxs_registry_destroy(probe.patterns);
  libxs_registry_destroy(probe.contexts);
  libxs_registry_destroy(probe.freq);
}


/**
 * Which scoring mode the predict slot uses: 0 = adaptive (default), 1 = warm up
 * over the training entries, commit the converged weights, then score FROZEN.
 *
 * Adaptive scoring carries one context across the run, so the escape bank
 * converges but the figure becomes a function of the order entries are
 * iterated -- reproducible here, not comparable across shuffles, and the
 * content-hash key already changed that order once. Frozen scoring is
 * order-independent by construction, but only says something once the weights
 * it freezes have converged, which is what the warm-up pass is for.
 */
static int ngram_bank_frozen(void)
{
  const char* env = getenv("CONVERSE_NGRAM_BANK_FROZEN");
  return (NULL != env && 0 != atoi(env)) ? 1 : 0;
}


/**
 * Converge the escape bank on the TRAINING data, then publish the weights so
 * frozen scoring has something converged to freeze.
 *
 * The entries pushed into the store are the training split already -- pushing
 * is gated on predict_is_test in token_predict_build -- so replaying them
 * reads no held-out data and needs none of the tokenization the eval loop does.
 * Passing NULL for values/probs/info still advances the bank: prob_observe
 * learns from the candidate, and the distribution is only reported if asked
 * for.
 *
 * Weights move only while this runs. Afterwards the model carries them and
 * scoring is strictly read-only, which is the property that makes the reported
 * figure independent of iteration order.
 */
static int ngram_bank_warmup(libxs_predict_t* store, int vocabulary)
{
  int result = EXIT_FAILURE;
  void* context = (NULL != store) ? libxs_predict_prob_create(store) : NULL;
  if (NULL != context) {
    libxs_predict_query_t info;
    long observed = 0;
    int i, nwarm;
    memset(&info, 0, sizeof(info));
    libxs_predict_query(store, &info);
    /**
     * How many entries the warm-up observes. The escape bank needs hundreds of
     * observations to settle, not hundreds of thousands, but each observation
     * costs a scan of every entry in the cluster -- so observing the whole store
     * is quadratic in the store size while buying convergence that a prefix
     * already reached. A bound makes that trade measurable rather than assumed;
     * 0 keeps the full pass.
     */
    nwarm = info.nentries;
    { const char* env = getenv("CONVERSE_NGRAM_BANK_WARMUP");
      if (NULL != env) {
        const int n = atoi(env);
        if (0 < n && n < nwarm) nwarm = n;
      }
    }
    for (i = 0; i < nwarm; ++i) {
      /* sized for the widest input vector either profile uses */
      double in[2 * TOKEN_EMB_DIM];
      double out[1];
      out[0] = 0.0;
      libxs_predict_get(store, i, in, out);
      if (0 < libxs_predict_prob_observe(NULL, store, context, in, 0,
        out, NULL, NULL, 0, NULL, NULL, vocabulary, 1))
      {
        ++observed;
      }
    }
    result = libxs_predict_prob_commit(store, context);
    /* nscan is the candidates ONE observation walks, so the product is the real
       cost of this pass -- printed because a bound that looks small can still be
       quadratic against a large cluster. */
    fprintf(stderr, "predict slot: warm-up observed %ld of %i entries"
      " (bound %i, %i clusters, scan max=%i avg=%.0f mean=%.0f"
      " => %.1fM pair-ops), commit %s\n",
      observed, info.nentries, nwarm, info.nclusters, info.nscan, info.escan,
      (0 < info.nclusters) ? ((double)info.nentries / info.nclusters) : 0.0,
      1e-6 * (double)observed * info.escan,
      (EXIT_SUCCESS == result) ? "ok" : "FAILED");
    libxs_predict_prob_destroy(context);
  }
  return result;
}


static int ngram_eval(libxs_registry_t* model, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind, const libxs_predict_t* store,
  int use_emb)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, index = 0;
  long ndeep = 0, ndeep_top1 = 0, nshallow = 0, nshallow_top1 = 0;
  double deep_bits = 0.0, deep_bytes = 0.0;
  double shallow_bits = 0.0, shallow_bytes = 0.0;
  double sum_bits = 0.0, sum_bytes = 0.0;
  long novel_match[NGRAM_ORDER_MAX + 1];
  double novel_match_bits[NGRAM_ORDER_MAX + 1];
  double novel_match_bytes[NGRAM_ORDER_MAX + 1];
  long novel_short = 0;
  double novel_short_bits = 0.0;
  long novel_sat = 0;
  double novel_sat_bits = 0.0;
  long ntrunc = 0, ntrunc_top1 = 0;
  double trunc_bits = 0.0, trunc_bytes = 0.0;
  double oracle_bits = 0.0;
  /**
   * The oracle split by whether the full-order context recurred. Aggregate
   * calibration gains in this project have consistently come from the attested
   * side while the novel side stayed put, so a bound that does not separate them
   * cannot say whether reweighting has anything left to offer where generation
   * actually fails.
   */
  double oracle_deep_bits = 0.0, oracle_shallow_bits = 0.0;
  long oracle_extra = 0;
  double expert_bits[NGRAM_ORDER_MAX + 1];
  double expert_bytes[NGRAM_ORDER_MAX + 1];
  long oracle_pick[NGRAM_ORDER_MAX + 1];
  long oracle_mixed = 0;
  long sel_total[NGRAM_ORDER_MAX + 1];
  long sel_mixed[NGRAM_ORDER_MAX + 1];
  long sel_low[NGRAM_ORDER_MAX + 1];
  /**
   * How often the KIND-DIFFERENT slot is the cheapest member, bucketed by the
   * matched context length the model already reports. The oracle reads the
   * target, so its 0.188 bits are only reachable if an OBSERVABLE feature
   * separates those positions: if the share is flat across buckets, no selector
   * keyed on this feature can find them and the headroom is not reachable.
   */
  long sel_extra[NGRAM_ORDER_MAX + 1];
  double sel_bits = 0.0;
  const int sel_order = ngram_select_order();
  double entry_bytes = 0.0;
  double bank_bits = 0.0;
  double bank_weight[NGRAM_BANK_MAX];
  double bank_wsum[NGRAM_BANK_MAX];
  long bank_active[NGRAM_BANK_MAX];
  /**
   * Each slot's STANDALONE cost, over exactly the positions where it spoke.
   * Without this a collapsed weight is uninterpretable: it cannot be told apart
   * from a plumbing fault. Same per-expert denominator discipline as the order
   * experts -- an abstaining slot must not be charged for positions it never
   * scored, nor credited with them.
   */
  double bank_slot_bits[NGRAM_BANK_MAX];
  double bank_slot_bytes[NGRAM_BANK_MAX];
  long bank_n = 0;
  /* The bank under the attested/novel split: two mechanisms so far had their
     headline reversed by bucket-splitting, so every slot set reports both. */
  double bank_deep_bits = 0.0, bank_shallow_bits = 0.0;
  const int bank = ngram_bank_probe();
  const unsigned int bank_slots = ngram_bank_slots();
  const int bank_geo = ngram_bank_geometric();
  /**
   * One scoring context for the whole run. The escape bank needs hundreds of
   * observations to commit, so a per-entry stream would be all transient and no
   * convergence; a per-run stream converges but makes the figure a function of
   * ENTRY ORDER -- reproducible for a fixed corpus and iteration, not comparable
   * across shuffles. A size of 0 means the model cannot be scored, in which case
   * the slot abstains rather than silently falling back to frozen weights.
   */
  void* bank_context = NULL;
  const int bank_frozen = ngram_bank_frozen();
  const int bank_vocab = (int)libxs_lexicon_size(lexicon);
  const double bank_rate = ngram_bank_rate();
  const double bank_share = ngram_bank_share();
  const int oracle = ngram_oracle_probe();
  const double inv_log2 = 1.0 / log(2.0);
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  FILE* file;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  memset(expert_bits, 0, sizeof(expert_bits));
  memset(expert_bytes, 0, sizeof(expert_bytes));
  memset(oracle_pick, 0, sizeof(oracle_pick));
  if (0 != bank && 0 != (bank_slots & (1u << NGRAM_BANK_PREDICT))
    && NULL != store && 0 == bank_frozen)
  {
    /**
     * NULL means the model cannot be scored, which is a different situation
     * from passing NULL as the context (that selects frozen scoring on a model
     * that is fine), so the slot abstains rather than silently freezing.
     */
    bank_context = libxs_predict_prob_create(store);
    if (NULL == bank_context) {
      fprintf(stderr, "predict slot: model cannot be scored, slot abstains\n");
    }
  }
  memset(bank_wsum, 0, sizeof(bank_wsum));
  memset(bank_active, 0, sizeof(bank_active));
  memset(bank_slot_bits, 0, sizeof(bank_slot_bits));
  memset(bank_slot_bytes, 0, sizeof(bank_slot_bytes));
  { int k;
    /**
     * Uniform prior over the ENABLED slots; the update is what differentiates
     * them. A disabled slot starts at zero and stays there, which is what makes
     * the order-only bank bit-exact to the pre-slot version.
     */
    int nslot = 0;
    for (k = 1; k <= NGRAM_BANK_LAST; ++k) {
      if (0 != ngram_bank_enabled(bank_slots, k)) ++nslot;
    }
    for (k = 0; k < NGRAM_BANK_MAX; ++k) {
      bank_weight[k] = (k >= 1 && k <= NGRAM_BANK_LAST
        && 0 != ngram_bank_enabled(bank_slots, k)) ? 1.0 / (double)nslot : 0.0;
    }
  }
  memset(sel_total, 0, sizeof(sel_total));
  memset(sel_mixed, 0, sizeof(sel_mixed));
  memset(sel_low, 0, sizeof(sel_low));
  memset(sel_extra, 0, sizeof(sel_extra));
  memset(novel_match, 0, sizeof(novel_match));
  memset(novel_match_bits, 0, sizeof(novel_match_bits));
  memset(novel_match_bytes, 0, sizeof(novel_match_bytes));
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    int native = ngram_native_mode();
    libxs_lexeme_stream_init(&stream);
    { const int ds = ngram_dedup_scale();
      const int want = (2 == ds) ? SCALE_PARAGRAPH : SCALE_SENTENCE;
      /**
       * Each source byte must be counted once, or BPC is not comparable across
       * granularities -- which is the one job the metric exists to do. Clause
       * fragments are stored at sentence scale ALONGSIDE their parent sentence,
       * so keeping both scored ~3x the real text.
       */
      if (0 != ds && (want != entry->scale
        || 0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT)))
      {
        is_test = 0;
      }
    }
    if (0 != is_test && entry->text_len > 0
      && (0 != native || EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 0)))
    {
      int wctx = ngram_wordctx();
      libxs_lexeme_t nat[COMPOSE_MAXTEXT];
      unsigned int word_ids[COMPOSE_MAXTEXT];
      int ntok = (0 != native)
        ? ngram_native_tokens(lexicon, entry->text, entry->text_len,
          nat, (0 != wctx) ? word_ids : NULL, COMPOSE_MAXTEXT, 0)
        : (int)stream.size;
      int ti;
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      int maxorder = ngram_maxorder();
      /* Source bytes this entry contributes, whether or not every token in it
         is scored -- the ceiling the denominator should approach. */
      entry_bytes += (double)entry->text_len;
      for (ti = 0; ti < ntok; ++ti) {
        unsigned int cur;
        unsigned short curlen;
        int is_content;
        if (0 != wctx) {
          hlen = ngram_wordctx_hist(nat, word_ids, ti, wctx, hist,
            NGRAM_ORDER_MAX);
        }
        if (0 != native) {
          cur = nat[ti].id;
          curlen = nat[ti].length;
          is_content = (0 != cur) ? 1 : 0;
        }
        else {
          const libxs_lexeme_t* lex = stream.data + ti;
          cur = lex->id;
          curlen = lex->length;
          is_content = (0 != (lex->flags
            & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER)) && 0 != cur) ? 1 : 0;
        }
        if (0 != is_content) {
          if (hlen > 0) {
            unsigned int ids[NGRAM_TOPK];
            int matched = 0;
            int n = ngramk_predict_order(model, hist, hlen, maxorder, ids,
              NGRAM_TOPK, &matched);
            double p = ngramk_prob(model, hist, hlen, maxorder, cur);
            double bits = -log(p) * inv_log2;
            int rank, top1 = 0;
            int deep = (matched >= maxorder && hlen >= maxorder) ? 1 : 0;
            /**
             * Per-position order selection, bounded from above. Each expert is
             * the SAME interpolated model given only its last k words of
             * history, so expert k is what a model capped at order k would say.
             * The oracle takes the cheapest expert per position: not achievable
             * (it reads the answer), but it bounds what ANY per-position mixture
             * of these predictors can win. The byte side found local order
             * selection worth 0.5-0.8 bits cross-document, and this asks whether
             * the same headroom exists at the word level before anything is
             * built to exploit it.
             */
            if (0 != oracle) {
              int k;
              double best = bits;
              int bestk = 0, best_extra = 0;
              const double w = (curlen > 0) ? (double)curlen : 1.0;
              for (k = 1; k <= maxorder && k <= hlen; ++k) {
                double pk = ngramk_prob(model, hist + (hlen - k), k, k, cur);
                double bk = -log((pk > 0.0) ? pk : 1e-12) * inv_log2;
                /**
                 * Per-expert byte denominator: expert k only exists where the
                 * history reaches k, so dividing its bits by the TOTAL bytes
                 * would flatter every high order by omitting the positions it
                 * cannot score. That artifact made order 6 look like the best
                 * expert when it had simply skipped the hard short-history
                 * positions.
                 */
                expert_bits[k] += bk;
                expert_bytes[k] += w;
                if (bk < best) {
                  best = bk;
                  bestk = k;
                }
              }
              /**
               * The bank: linear pool over the expert slots, with weights
               * carried across positions and updated causally (score first,
               * then update), so no target information enters the score. The
               * slots are the fixed orders plus, when enabled, experts of a
               * different KIND -- the 2.018 oracle below says order selection
               * alone cannot reach the target, so mixing kinds is the only
               * direction with measured room.
               */
              if (0 != bank) {
                ngram_expert_t expert[NGRAM_BANK_MAX];
                double pooled, pbits;
                int b;
                ngram_bank_experts(hist, hlen, maxorder, cur, bank_slots,
                  store, bank_context, use_emb, bank_vocab, expert);
                pooled = (0 != bank_geo)
                  ? ngram_bank_pool_geo(model, hist, hlen, maxorder, cur,
                    bank_weight, expert)
                  : ngram_bank_pool(bank_weight, expert);
                pbits = -log(pooled) * inv_log2;
                bank_bits += pbits;
                if (0 != deep) bank_deep_bits += pbits;
                else bank_shallow_bits += pbits;
                ++bank_n;
                for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
                  bank_wsum[b] += bank_weight[b];
                  if (0 != expert[b].active) {
                    const double pb = expert[b].probability;
                    ++bank_active[b];
                    bank_slot_bits[b] += -log((pb > 0.0) ? pb : 1e-12)
                      * inv_log2;
                    bank_slot_bytes[b] += w;
                  }
                }
                /**
                 * Extend the bound to the WHOLE pool, not just the fixed orders.
                 * Any per-position reweighting can at best put all its mass on
                 * the cheapest member available at that position, so taking the
                 * minimum over every active slot -- including the ones that
                 * differ in kind -- bounds what ANY mixing rule over this
                 * evidence could reach. Without the extra slots the bound would
                 * flatter a pool it does not cover.
                 */
                for (b = NGRAM_ORDER_MAX + 1; b <= NGRAM_BANK_LAST; ++b) {
                  if (0 != expert[b].active) {
                    const double pb = expert[b].probability;
                    const double bb = -log((pb > 0.0) ? pb : 1e-12) * inv_log2;
                    if (bb < best) {
                      best = bb;
                      best_extra = 1;
                      ++oracle_extra;
                    }
                  }
                }
                ngram_bank_update(bank_weight, expert, pooled, bank_rate,
                  bank_share);
              }
              oracle_bits += best;
              if (0 != deep) oracle_deep_bits += best;
              else oracle_shallow_bits += best;
              /* bestk==0 means no single-order expert beat the mixture. */
              if (0 == bestk) ++oracle_mixed;
              else ++oracle_pick[bestk];
              /**
               * Can an OBSERVABLE feature separate "o1 wins" from "the mixture
               * wins"? The natural candidate costs nothing: the matched context
               * length, which the predictor already reports and which does not
               * read the target. Accumulate the oracle's choice against it, so
               * the confusion is visible rather than assumed. Secs 16/19/20 each
               * found a signal that separated classes in hindsight and died when
               * computed without the answer, so this is checked before any
               * chooser is written.
               */
              { const int mm = (matched < 0) ? 0
                  : ((matched > NGRAM_ORDER_MAX) ? NGRAM_ORDER_MAX : matched);
                ++sel_total[mm];
                if (0 != best_extra) ++sel_extra[mm];
                if (0 == bestk) ++sel_mixed[mm];
                else if (1 == bestk) ++sel_low[mm];
                /**
                 * The ACHIEVABLE rule the confusion above suggests: take the
                 * order-1 expert when a short context matched (matched >= 2),
                 * otherwise keep the mixture. Uses only the reported match
                 * length, never the target, so unlike the oracle this is a
                 * mechanism and its bits are honest.
                 */
                if (mm >= 2 && hlen >= sel_order) {
                  double ps = ngramk_prob(model, hist + (hlen - sel_order),
                    sel_order, sel_order, cur);
                  sel_bits += -log((ps > 0.0) ? ps : 1e-12) * inv_log2;
                }
                else sel_bits += bits;
              }
            }
            ++npairs;
            sum_bits += bits;
            sum_bytes += (curlen > 0) ? (double)curlen : 1.0;
            for (rank = 0; rank < n; ++rank) {
              if (ids[rank] == cur) {
                if (0 == rank) {
                  ++ntop1;
                  top1 = 1;
                }
                ++ntopk;
                break;
              }
            }
            /**
             * Split by whether the FULL-ORDER context was attested in training.
             * The store is built from non-test entries only, so a match at the
             * maximum order means this exact context recurs verbatim; those
             * positions are recall, not generalization. Reported separately so
             * the depth result cannot be read as generalization it is not.
             */
            if (0 != deep) {
              ++ndeep;
              ndeep_top1 += top1;
              deep_bits += bits;
              deep_bytes += (curlen > 0) ? (double)curlen : 1.0;
            }
            else {
              /**
               * Decompose the novel bucket by how much context DID match. The
               * predictor already reports that length, so this costs no extra
               * lookup. It bounds what a total function can buy: mass at
               * matched==0 is scored by the unigram prior alone and is the only
               * part where no count evidence exists at any order, while mass at
               * matched>0 is already carried by a shorter attested context.
               */
              const int m = (matched < 0) ? 0
                : ((matched > NGRAM_ORDER_MAX) ? NGRAM_ORDER_MAX : matched);
              ++nshallow;
              nshallow_top1 += top1;
              shallow_bits += bits;
              shallow_bytes += (curlen > 0) ? (double)curlen : 1.0;
              ++novel_match[m];
              novel_match_bits[m] += bits;
              novel_match_bytes[m] += (curlen > 0) ? (double)curlen : 1.0;
              /**
               * A position whose history is shorter than the order cannot be
               * full-order attested, so it lands in the novel bucket by sentence
               * position rather than by unattested context. Counted separately:
               * without this the short-context mass reads as weak evidence when
               * part of it is merely an early position with nothing to look up.
               */
              if (hlen < maxorder) {
                ++novel_short;
                novel_short_bits += bits;
              }
              /**
               * The distinction that decides how to read the short-context mass.
               * SATURATED means the model matched ALL the history that existed,
               * so nothing was unattested and the only remedy is more history --
               * an early sentence position. TRUNCATED means history was
               * available and the longer context was not in the store, which is
               * the genuinely unattested case. Lumping them reads a young
               * sentence as weak evidence.
               */
              { const int avail = (hlen < maxorder) ? hlen : maxorder;
                if (m >= avail) {
                  ++novel_sat;
                  novel_sat_bits += bits;
                }
                /**
                 * The REPAIRED generalization bucket: a full-order history was
                 * available AND the full-order context was not attested. This is
                 * the only set where the model was asked something its evidence
                 * could have covered and did not. The wider novel bucket mixes
                 * these with early-sentence positions that had nothing longer to
                 * look up, so a mechanism can move it without generalizing.
                 */
                else if (hlen >= maxorder) {
                  ++ntrunc;
                  ntrunc_top1 += top1;
                  trunc_bits += bits;
                  trunc_bytes += (curlen > 0) ? (double)curlen : 1.0;
                }
              }
            }
          }
          if (0 == wctx) ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, cur);
        }
        if (0 == native && 0 == wctx) {
          const libxs_lexeme_t* lex = stream.data + ti;
          if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-ngram[%s%s]: top1=%.1f%% top%d=%.1f%% n=%ld bpc=%.3f\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
    /**
     * The BPC denominator, printed because it is the only thing that makes the
     * number comparable ACROSS granularities: the same test text must yield the
     * same byte count whatever the unit. A zero-length token (a sub-word
     * continuation carries the whole word's bytes on its first piece) falls back
     * to 1.0, which would silently inflate the denominator and understate BPC
     * for the finer units, so the count is reported rather than trusted.
     */
    fprintf(stderr, "  bpc denominator: %.0f source bytes of %.0f in scored"
      " entries (%.1f%% covered, %.2f bytes/position)\n", sum_bytes,
      entry_bytes, (entry_bytes > 0.0) ? 100.0 * sum_bytes / entry_bytes : 0.0,
      (npairs > 0) ? sum_bytes / (double)npairs : 0.0);
    fprintf(stderr, "  attested-context split: verbatim %.1f%% of positions"
      " (top1=%.1f%% bpc=%.3f) | novel %.1f%% (top1=%.1f%% bpc=%.3f)\n",
      100.0 * (double)ndeep / (double)npairs,
      (ndeep > 0) ? 100.0 * (double)ndeep_top1 / (double)ndeep : 0.0,
      (deep_bytes > 0.0) ? deep_bits / deep_bytes : 0.0,
      100.0 * (double)nshallow / (double)npairs,
      (nshallow > 0) ? 100.0 * (double)nshallow_top1 / (double)nshallow : 0.0,
      (shallow_bytes > 0.0) ? shallow_bits / shallow_bytes : 0.0);
    if (0 != oracle) {
      int k;
      fprintf(stderr, "  order experts (bpc):");
      for (k = 1; k <= NGRAM_ORDER_MAX; ++k) {
        if (expert_bytes[k] > 0.0) {
          fprintf(stderr, " o%d=%.3f", k,
            expert_bits[k] / expert_bytes[k]);
        }
      }
      fprintf(stderr, " | ORACLE=%.3f (mixed=%.3f)\n",
        (sum_bytes > 0.0) ? oracle_bits / sum_bytes : 0.0,
        (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
      /**
       * The bound where it matters. A mechanism that only improves the attested
       * side cannot help generation on unseen material, and the novel column is
       * what any reweighting of this evidence could reach there.
       */
      fprintf(stderr, "  ORACLE split: attested=%.3f (of %.3f) |"
        " novel=%.3f (of %.3f) | extra-slot wins=%ld\n",
        (deep_bytes > 0.0) ? oracle_deep_bits / deep_bytes : 0.0,
        (deep_bytes > 0.0) ? deep_bits / deep_bytes : 0.0,
        (shallow_bytes > 0.0) ? oracle_shallow_bits / shallow_bytes : 0.0,
        (shallow_bytes > 0.0) ? shallow_bits / shallow_bytes : 0.0,
        oracle_extra);
      fprintf(stderr, "  oracle picks:");
      for (k = 1; k <= NGRAM_ORDER_MAX; ++k) {
        if (oracle_pick[k] > 0) {
          fprintf(stderr, " o%d=%.1f%%", k,
            100.0 * (double)oracle_pick[k] / (double)npairs);
        }
      }
      fprintf(stderr, " mixed=%.1f%%\n",
        100.0 * (double)oracle_mixed / (double)npairs);
      fprintf(stderr, "  oracle choice vs matched context (o1-wins%%/mixed%%):");
      for (k = 0; k <= NGRAM_ORDER_MAX; ++k) {
        if (sel_total[k] > 0) {
          fprintf(stderr, " m%d=%.0f/%.0f", k,
            100.0 * (double)sel_low[k] / (double)sel_total[k],
            100.0 * (double)sel_mixed[k] / (double)sel_total[k]);
        }
      }
      /**
       * The reachability test for the oracle's remaining headroom. A share that
       * varies across buckets is a selector waiting to be written; a flat share
       * means this feature cannot tell those positions apart and the headroom
       * stays with the oracle.
       */
      if (0 < oracle_extra) {
        fprintf(stderr, "\n  kind-different slot cheapest, by matched context:");
        for (k = 0; k <= NGRAM_ORDER_MAX; ++k) {
          if (sel_total[k] > 0) {
            fprintf(stderr, " m%d=%.0f%%(n=%ld)", k,
              100.0 * (double)sel_extra[k] / (double)sel_total[k],
              sel_total[k]);
          }
        }
      }
      if (0 != bank && bank_n > 0) {
        int b, nslot = 0;
        for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
          if (0 != ngram_bank_enabled(bank_slots, b)) ++nslot;
        }
        fprintf(stderr, "  BANK(%s rate=%.2f share=%.3f slots=%d)=%.3f"
          " | attested %.3f novel %.3f | mean weight (active%%):",
          (0 != bank_geo) ? "geo" : "linear", bank_rate, bank_share, nslot,
          (sum_bytes > 0.0) ? bank_bits / sum_bytes : 0.0,
          (deep_bytes > 0.0) ? bank_deep_bits / deep_bytes : 0.0,
          (shallow_bytes > 0.0) ? bank_shallow_bits / shallow_bytes : 0.0);
        /**
         * Mean weight alone cannot be read without the active share: a slot
         * that abstains often keeps its weight while contributing nothing, so a
         * healthy-looking weight can belong to an expert that almost never
         * spoke.
         */
        for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
          if (bank_wsum[b] > 0.0) {
            fprintf(stderr, " %s=%.3f(%.0f%%)", ngram_bank_slotname(b),
              bank_wsum[b] / (double)bank_n,
              100.0 * (double)bank_active[b] / (double)bank_n);
          }
        }
        if (predict_nscored > 0) {
          /**
           * Frozen scoring reports no entropy: info aliases the context and
           * there is none, so the point query cannot return it. Saying FROZEN
           * rather than printing 0.000 keeps a missing diagnostic from reading
           * as a converged one.
           */
          if (0 != bank_frozen) {
            fprintf(stderr, "\n  predict slot: vocab=%d scored=%ld FROZEN"
              " (order-independent)", bank_vocab, predict_nscored);
          }
          else {
            const double ent = predict_entropy_sum / (double)predict_nscored;
            fprintf(stderr, "\n  predict slot: support=%d vocab=%d scored=%ld"
              " escape entropy=%.3f%s", predict_support_last, bank_vocab,
              predict_nscored, ent,
              (ent > NGRAM_PREDICT_ENTROPY_MAX) ? " (UNSETTLED)" : "");
          }
          /**
           * How much of the slot is the kNN vote. Bits are per position, not per
           * byte, so they compare only with each other -- the question is
           * whether the positions the vote reached are cheaper than the rest and
           * how many there are.
           */
          if (0 < predict_att_n + predict_unatt_n) {
            const long tot = predict_att_n + predict_unatt_n;
            fprintf(stderr, "\n  predict slot local evidence: attested %.1f%%"
              " (%.3f bits/pos) | none %.1f%% (%.3f bits/pos)",
              100.0 * (double)predict_att_n / (double)tot,
              (0 < predict_att_n)
                ? predict_att_bits / (double)predict_att_n : 0.0,
              100.0 * (double)predict_unatt_n / (double)tot,
              (0 < predict_unatt_n)
                ? predict_unatt_bits / (double)predict_unatt_n : 0.0);
          }
        }
        fprintf(stderr, "\n  bank slots standalone (bpc where active):");
        for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
          if (bank_slot_bytes[b] > 0.0) {
            fprintf(stderr, " %s=%.3f", ngram_bank_slotname(b),
              bank_slot_bits[b] / bank_slot_bytes[b]);
          }
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, " | RULE(m>=2 -> o%d)=%.3f\n", sel_order,
        (sum_bytes > 0.0) ? sel_bits / sum_bytes : 0.0);
    }
    if (nshallow > 0) {
      int m;
      fprintf(stderr, "  novel bucket by matched context:");
      for (m = 0; m <= NGRAM_ORDER_MAX; ++m) {
        if (novel_match[m] > 0) {
          fprintf(stderr, " n%d=%.1f%%/bpc=%.3f/bits=%.1f%%", m,
            100.0 * (double)novel_match[m] / (double)nshallow,
            (novel_match_bytes[m] > 0.0)
              ? novel_match_bits[m] / novel_match_bytes[m] : 0.0,
            (shallow_bits > 0.0)
              ? 100.0 * novel_match_bits[m] / shallow_bits : 0.0);
        }
      }
      fprintf(stderr, " | truncated %.1f%% of all (top1=%.1f%% bpc=%.3f)\n",
        100.0 * (double)ntrunc / (double)npairs,
        (ntrunc > 0) ? 100.0 * (double)ntrunc_top1 / (double)ntrunc : 0.0,
        (trunc_bytes > 0.0) ? trunc_bits / trunc_bytes : 0.0);
      fprintf(stderr, "  novel bucket composition:"
        " short-history %.1f%%/bits=%.1f%%"
        " saturated %.1f%%/bits=%.1f%% (n=%ld)\n",
        100.0 * (double)novel_short / (double)nshallow,
        (shallow_bits > 0.0) ? 100.0 * novel_short_bits / shallow_bits : 0.0,
        100.0 * (double)novel_sat / (double)nshallow,
        (shallow_bits > 0.0) ? 100.0 * novel_sat_bits / shallow_bits : 0.0,
        nshallow);
    }
    result = EXIT_SUCCESS;
  }
  file = fopen(converse_path_predict_eval, "r");
  if (NULL != file) {
    long hnpairs = 0, htop1 = 0, htopk = 0;
    char line[EVAL_LINE_MAX];
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      char* context;
      char* expected;
      char* sep;
      unsigned int prev2 = 0, prev1 = 0;
      unsigned int ids[NGRAM_TOPK];
      int n, rank;
      context = eval_trim(line);
      if ('\0' == *context || '#' == *context) continue;
      sep = strchr(context, '|');
      if (NULL == sep) continue;
      *sep = '\0';
      expected = eval_trim(sep + 1);
      context = eval_trim(context);
      if ('\0' == *context || '\0' == *expected) continue;
      ngram_last_context(lexicon, rules, nrules, context,
        (int)strlen(context), &prev2, &prev1);
      n = ngram_predict(model, prev2, prev1, order, ids, NGRAM_TOPK);
      ++hnpairs;
      for (rank = 0; rank < n; ++rank) {
        char word[LIBXS_LEXEME_MAXBYTES + 1];
        int len = 0;
        const char* text = libxs_lexicon_text(lexicon, ids[rank], &len, NULL);
        if (NULL != text && len > 0 && len < (int)sizeof(word)) {
          memcpy(word, text, (size_t)len);
          word[len] = '\0';
          if (0 != text_contains_word_ci(expected, (int)strlen(expected),
            word))
          {
            if (0 == rank) ++htop1;
            ++htopk;
            break;
          }
        }
      }
    }
    if (hnpairs > 0) {
      fprintf(stdout,
        "predict-ngram[%s-fixture]: top1=%.1f%% top%d=%.1f%% n=%ld\n",
        kind, 100.0 * (double)htop1 / (double)hnpairs, NGRAM_TOPK,
        100.0 * (double)htopk / (double)hnpairs, hnpairs);
    }
    fclose(file);
  }
  libxs_predict_prob_destroy(bank_context);
  return result;
}


static libxs_predict_t* token_predict_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const answer_predict_profile_t* profile, int use_emb, int holdout,
  int ctxlen)
{
  int ninputs = (0 != use_emb) ? 2 * TOKEN_EMB_DIM : 2;
  libxs_predict_t* model = libxs_predict_create(ninputs, 1);
  double decay = knnlm_decay();
  long pushed = 0;
  if (ctxlen < 2) ctxlen = 2;
  if (NULL != model && NULL != corpus && NULL != profile) {
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value;
    libxs_predict_set_mode(model, profile->mode);
    libxs_predict_set_decompose(model, profile->decompose);
    if (0 != use_emb) {
      double weights[2 * TOKEN_EMB_DIM];
      int dim;
      for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
        weights[dim] = 1.0;
        weights[TOKEN_EMB_DIM + dim] = 2.0;
      }
      libxs_predict_set_weights(model, weights);
    }
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value && pushed < TOKEN_PREDICT_TRAIN_MAX) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      libxs_lexeme_stream_t stream;
      int is_train = (0 == predict_is_test(index, holdout));
      libxs_lexeme_stream_init(&stream);
      if (0 != is_train && entry->text_len > 0
        && EXIT_SUCCESS == libxs_lexeme_stream_encode(
        lexicon, &stream, (const unsigned char*)entry->text,
        (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
      {
        size_t pos;
        unsigned int prev1 = 0, prev2 = 0;
        unsigned int hist[TOKEN_CTX_MAX];
        int hlen = 0;
        for (pos = 0; pos < stream.size && pushed < TOKEN_PREDICT_TRAIN_MAX;
          ++pos)
        {
          const libxs_lexeme_t* lex = stream.data + pos;
          if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 != lex->id)
          {
            if (0 != prev1) {
              double in[2 * TOKEN_EMB_DIM];
              double out[1];
              if (ctxlen > 2 && 0 != use_emb) {
                knnlm_ctx_vector(hist, hlen, ctxlen, decay, in);
              }
              else token_input_vector(prev2, prev1, use_emb, in);
              out[0] = (double)lex->id;
              if (EXIT_SUCCESS == libxs_predict_push(NULL, model, in, out)) {
                ++pushed;
              }
            }
            prev2 = prev1;
            prev1 = lex->id;
            ngram_hist_push(hist, &hlen, TOKEN_CTX_MAX, lex->id);
          }
          if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
            prev1 = 0;
            prev2 = 0;
            hlen = 0;
          }
        }
      }
      libxs_lexeme_stream_release(&stream);
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
    if (pushed <= 0 || EXIT_SUCCESS != libxs_predict_build(model,
      profile->clusters, profile->order, profile->quality))
    {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  else {
    libxs_predict_destroy(model);
    model = NULL;
  }
  return model;
}


static unsigned int token_predict_next(const libxs_predict_t* model,
  unsigned int prev2, unsigned int prev1, int use_emb)
{
  double in[2 * TOKEN_EMB_DIM];
  double out[1];
  token_input_vector(prev2, prev1, use_emb, in);
  out[0] = 0.0;
  libxs_predict_eval(NULL, model, in, out, NULL, 1);
  return (out[0] > 0.0) ? (unsigned int)(out[0] + 0.5) : 0;
}


static int token_predict_eval(const libxs_predict_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int use_emb, int holdout,
  const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, seen = 0, index = 0;
  int stride = predict_eval_stride();
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
    {
      size_t pos;
      unsigned int prev1 = 0, prev2 = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (0 != prev1) {
            if (0 == (seen % stride)) {
              unsigned int pred = token_predict_next(model, prev2, prev1,
                use_emb);
              ++npairs;
              if (pred == lex->id) ++ntop1;
            }
            ++seen;
          }
          prev2 = prev1;
          prev1 = lex->id;
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
          prev1 = 0;
          prev2 = 0;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout, "predict-%s%s: top1=%.1f%% n=%ld (stride=%d)\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, npairs,
      stride);
    result = EXIT_SUCCESS;
  }
  return result;
}


static void token_complete(const libxs_predict_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int use_emb, const char* text, int text_len)
{
  unsigned int prev2 = 0, prev1 = 0;
  unsigned int pred;
  int len = 0;
  const char* word;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  if (0 == prev1) {
    printf("(no continuation)\n");
    return;
  }
  pred = token_predict_next(model, prev2, prev1, use_emb);
  word = (0 != pred) ? libxs_lexicon_text(lexicon, pred, &len, NULL) : NULL;
  printf("next:");
  if (NULL != word && len > 0) printf(" %.*s", len, word);
  printf("\n");
  { unsigned int step2 = prev2, step1 = prev1;
    int step;
    printf("greedy:");
    for (step = 0; step < 12; ++step) {
      unsigned int p = token_predict_next(model, step2, step1, use_emb);
      int l = 0;
      const char* w = (0 != p) ? libxs_lexicon_text(lexicon, p, &l, NULL)
        : NULL;
      if (NULL == w || l <= 0) break;
      printf(" %.*s", l, w);
      step2 = step1;
      step1 = p;
    }
    printf("\n");
  }
}


static int ngram_candidates(libxs_registry_t* model, unsigned int prev2,
  unsigned int prev1, int order, unsigned int ids[], double relfreq[],
  int provenance[], int* ctx_total, int k)
{
  int result = 0;
  const ngram_entry_t* entry = NULL;
  int prov = 0;
  if (order >= 2 && 0 != prev2) {
    entry = ngram_lookup(model, prev2, prev1);
    if (NULL != entry) prov = 2;
  }
  if (NULL == entry) {
    entry = ngram_lookup(model, 0, prev1);
    if (NULL != entry) prov = 1;
  }
  if (NULL != ctx_total) *ctx_total = (NULL != entry) ? (int)entry->total : 0;
  if (NULL != entry) {
    unsigned int taken[NGRAM_SUCC_MAX];
    unsigned int nsucc = entry->nsucc;
    unsigned int slot;
    double total = (entry->total > 0) ? (double)entry->total : 1.0;
    for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) {
      taken[slot] = 0;
    }
    while (result < k) {
      int best = -1;
      for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) {
        if (0 == taken[slot]
          && (best < 0 || entry->succ[slot].count > entry->succ[best].count))
        {
          best = (int)slot;
        }
      }
      if (best < 0) break;
      taken[best] = 1;
      ids[result] = entry->succ[best].id;
      relfreq[result] = (double)entry->succ[best].count / total;
      provenance[result] = prov;
      ++result;
    }
  }
  else {
    int slot;
    for (slot = 0; slot < k && slot < converse_ngram.backoff_count; ++slot) {
      ids[slot] = converse_ngram.backoff_ids[slot];
      relfreq[slot] = 0.0;
      provenance[slot] = 0;
      ++result;
    }
  }
  return result;
}


static int token_is_stop(libxs_lexicon_t* lexicon, unsigned int id)
{
  unsigned int flags = 0;
  int len = 0;
  if (NULL != lexicon && 0 != id) {
    libxs_lexicon_text(lexicon, id, &len, &flags);
  }
  return (0 != (flags & LIBXS_LEXEME_STOP)) ? 1 : 0;
}


static void rerank_features(libxs_registry_t* ngram, unsigned int prev1,
  unsigned int id, double relfreq, int rank, int provenance, int ctx_total,
  double ctx_top, libxs_lexicon_t* lexicon, double inputs[RERANK_INPUTS])
{
  double prior = ngram_unigram_prior(id);
  double reliability = (double)ctx_total / RERANK_RELIABILITY;
  double lift = relfreq / (prior + 1e-6);
  if (reliability > 1.0) reliability = 1.0;
  if (lift > RERANK_LIFT_MAX) lift = RERANK_LIFT_MAX;
  inputs[0] = relfreq;
  inputs[1] = (double)rank / (double)NGRAM_SUCC_MAX;
  inputs[2] = (double)provenance / 2.0;
  inputs[3] = (double)token_is_stop(lexicon, id);
  inputs[4] = ngram_pair_relfreq(ngram, 0, prev1, id);
  inputs[5] = prior;
  inputs[6] = reliability;
  inputs[7] = ctx_top;
  inputs[8] = lift;
}


static libxs_predict_t* rerank_build(const libxs_registry_t* corpus,
  libxs_registry_t* ngram, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order,
  const answer_predict_profile_t* profile, int holdout)
{
  libxs_predict_t* model = libxs_predict_create(RERANK_INPUTS, 1);
  long pushed = 0;
  if (NULL != model && NULL != corpus && NULL != ngram && NULL != profile) {
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value;
    static const double weights[RERANK_INPUTS] = {
      3.0, 0.5, 1.0, 0.4, 1.5, 0.6, 0.8, 1.0, 2.0
    };
    libxs_predict_set_mode(model, profile->mode);
    libxs_predict_set_decompose(model, profile->decompose);
    libxs_predict_set_weights(model, weights);
    if (0.0 != profile->smooth) libxs_predict_set_smooth(model,
      profile->smooth);
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value && pushed < TOKEN_PREDICT_TRAIN_MAX) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      libxs_lexeme_stream_t stream;
      int is_train = (0 == predict_is_test(index, holdout));
      libxs_lexeme_stream_init(&stream);
      if (0 != is_train && entry->text_len > 0
        && EXIT_SUCCESS == libxs_lexeme_stream_encode(
        lexicon, &stream, (const unsigned char*)entry->text,
        (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
      {
        size_t pos;
        unsigned int prev1 = 0, prev2 = 0;
        for (pos = 0; pos < stream.size && pushed < TOKEN_PREDICT_TRAIN_MAX;
          ++pos)
        {
          const libxs_lexeme_t* lex = stream.data + pos;
          if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 != lex->id)
          {
            if (0 != prev1) {
              unsigned int ids[NGRAM_SUCC_MAX];
              double relfreq[NGRAM_SUCC_MAX];
              int prov[NGRAM_SUCC_MAX];
              int ctx_total = 0;
              int n = ngram_candidates(ngram, prev2, prev1, order, ids,
                relfreq, prov, &ctx_total, NGRAM_SUCC_MAX);
              double ctx_top = (n > 0) ? relfreq[0] : 0.0;
              int rank;
              for (rank = 0; rank < n; ++rank) {
                double in[RERANK_INPUTS];
                double out[1];
                rerank_features(ngram, prev1, ids[rank], relfreq[rank], rank,
                  prov[rank], ctx_total, ctx_top, lexicon, in);
                out[0] = (ids[rank] == lex->id) ? 1.0 : 0.0;
                libxs_predict_push(NULL, model, in, out);
                ++pushed;
              }
            }
            prev2 = prev1;
            prev1 = lex->id;
          }
          if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
            prev1 = 0;
            prev2 = 0;
          }
        }
      }
      libxs_lexeme_stream_release(&stream);
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
    if (pushed <= 0 || EXIT_SUCCESS != libxs_predict_build(model,
      profile->clusters, profile->order, profile->quality))
    {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  else {
    libxs_predict_destroy(model);
    model = NULL;
  }
  return model;
}


static int rerank_topk(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  unsigned int prev2, unsigned int prev1, int order, unsigned int out_ids[],
  int k)
{
  unsigned int ids[NGRAM_SUCC_MAX];
  double relfreq[NGRAM_SUCC_MAX];
  int prov[NGRAM_SUCC_MAX];
  double score[NGRAM_SUCC_MAX];
  int taken[NGRAM_SUCC_MAX];
  int ctx_total = 0;
  int n = ngram_candidates(ngram, prev2, prev1, order, ids, relfreq, prov,
    &ctx_total, NGRAM_SUCC_MAX);
  double ctx_top = (n > 0) ? relfreq[0] : 0.0;
  int result = 0;
  int rank;
  for (rank = 0; rank < n; ++rank) {
    double in[RERANK_INPUTS];
    double out[1];
    rerank_features(ngram, prev1, ids[rank], relfreq[rank], rank, prov[rank],
      ctx_total, ctx_top, lexicon, in);
    out[0] = 0.0;
    libxs_predict_eval(NULL, reranker, in, out, NULL, 1);
    score[rank] = out[0];
    taken[rank] = 0;
  }
  while (result < k && result < n) {
    int best = -1;
    for (rank = 0; rank < n; ++rank) {
      if (0 == taken[rank] && (best < 0 || score[rank] > score[best])) {
        best = rank;
      }
    }
    if (best < 0) break;
    taken[best] = 1;
    out_ids[result] = ids[best];
    ++result;
  }
  return result;
}


static int rerank_eval(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, seen = 0, index = 0;
  int stride = predict_eval_stride();
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  if (NULL == ngram || NULL == reranker || NULL == corpus || NULL == lexicon) {
    return EXIT_FAILURE;
  }
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
    {
      size_t pos;
      unsigned int prev1 = 0, prev2 = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (0 != prev1) {
            if (0 == (seen % stride)) {
              unsigned int ids[NGRAM_TOPK];
              int n = rerank_topk(ngram, reranker, lexicon, prev2, prev1,
                order, ids, NGRAM_TOPK);
              int rank;
              ++npairs;
              for (rank = 0; rank < n; ++rank) {
                if (ids[rank] == lex->id) {
                  if (0 == rank) ++ntop1;
                  ++ntopk;
                  break;
                }
              }
            }
            ++seen;
          }
          prev2 = prev1;
          prev1 = lex->id;
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
          prev1 = 0;
          prev2 = 0;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-%s%s: top1=%.1f%% top%d=%.1f%% n=%ld (stride=%d)\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      stride);
    result = EXIT_SUCCESS;
  }
  return result;
}


static void rerank_complete(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len)
{
  unsigned int prev2 = 0, prev1 = 0;
  unsigned int ids[NGRAM_TOPK];
  int n, i;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  n = rerank_topk(ngram, reranker, lexicon, prev2, prev1, order, ids,
    NGRAM_TOPK);
  if (n <= 0) {
    printf("(no continuation)\n");
    return;
  }
  printf("next:");
  for (i = 0; i < n; ++i) {
    int len = 0;
    const char* word = libxs_lexicon_text(lexicon, ids[i], &len, NULL);
    if (NULL != word && len > 0) printf(" %.*s", len, word);
  }
  printf("\n");
  { unsigned int step2 = prev2, step1 = prev1;
    int step;
    printf("greedy:");
    for (step = 0; step < 12; ++step) {
      unsigned int step_ids[1];
      int len = 0;
      const char* word;
      if (0 == rerank_topk(ngram, reranker, lexicon, step2, step1, order,
        step_ids, 1)) break;
      word = libxs_lexicon_text(lexicon, step_ids[0], &len, NULL);
      if (NULL == word || len <= 0) break;
      printf(" %.*s", len, word);
      step2 = step1;
      step1 = step_ids[0];
    }
    printf("\n");
  }
}


static const libxs_predict_t* knnlm_cache_model = NULL;
static double* knnlm_cache_in = NULL;
static unsigned int* knnlm_cache_next = NULL;
static int knnlm_cache_size = 0;
static double* knnlm_dyn_in = NULL;
static unsigned int* knnlm_dyn_next = NULL;
static int knnlm_dyn_size = 0;
static int knnlm_dyn_cap = 0;
static uint64_t* knnlm_ann_code = NULL;
static int* knnlm_ann_order = NULL;
static int knnlm_ann_size = 0;


static void knnlm_cache_free(void)
{
  free(knnlm_cache_in);
  free(knnlm_cache_next);
  free(knnlm_dyn_in);
  free(knnlm_dyn_next);
  free(knnlm_ann_code);
  free(knnlm_ann_order);
  knnlm_cache_in = NULL;
  knnlm_cache_next = NULL;
  knnlm_dyn_in = NULL;
  knnlm_dyn_next = NULL;
  knnlm_ann_code = NULL;
  knnlm_ann_order = NULL;
  knnlm_cache_size = 0;
  knnlm_dyn_size = 0;
  knnlm_dyn_cap = 0;
  knnlm_ann_size = 0;
  knnlm_cache_model = NULL;
}


/**
 * Approximate NN over the static datastore is enabled by CONVERSE_KNNLM_ANN;
 * default off keeps the exact brute-force scan (bit-identical results).
 */
static int knnlm_ann_mode(void)
{
  const char* env = getenv("CONVERSE_KNNLM_ANN");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


/**
 * Quantize a context vector's leading dims to a Hilbert code: locality in the
 * embedding space becomes locality along the 1-D code so a window around the
 * query code holds its near neighbors. Values are the prev1 half of the input
 * (the more predictive token, per the 4x weighting in knnlm_scan).
 */
static uint64_t knnlm_ann_encode(const double* in)
{
  unsigned int coords[KNNLM_ANN_DIMS];
  const double* v = in + TOKEN_EMB_DIM;
  int d;
  double scale = (double)((1 << KNNLM_ANN_BITS) - 1);
  for (d = 0; d < KNNLM_ANN_DIMS; ++d) {
    double x = (d < TOKEN_EMB_DIM) ? v[d] : 0.0;
    double u = 0.5 * (x + 1.0);
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    coords[d] = (unsigned int)(u * scale + 0.5);
  }
  return libxs_hilbert_bits(coords, KNNLM_ANN_DIMS, KNNLM_ANN_BITS);
}


static int knnlm_ann_cmp(const void* a, const void* b)
{
  int ia = *(const int*)a, ib = *(const int*)b;
  uint64_t ca = knnlm_ann_code[ia], cb = knnlm_ann_code[ib];
  if (ca < cb) return -1;
  if (ca > cb) return 1;
  return 0;
}


/* Build the sorted Hilbert index over the current static cache. */
static void knnlm_ann_build(void)
{
  free(knnlm_ann_code);
  free(knnlm_ann_order);
  knnlm_ann_code = NULL;
  knnlm_ann_order = NULL;
  knnlm_ann_size = 0;
  if (knnlm_cache_size > 0) {
    knnlm_ann_code = (uint64_t*)malloc((size_t)knnlm_cache_size
      * sizeof(*knnlm_ann_code));
    knnlm_ann_order = (int*)malloc((size_t)knnlm_cache_size
      * sizeof(*knnlm_ann_order));
    if (NULL != knnlm_ann_code && NULL != knnlm_ann_order) {
      int i;
      for (i = 0; i < knnlm_cache_size; ++i) {
        knnlm_ann_code[i] = knnlm_ann_encode(knnlm_cache_in
          + (size_t)i * 2 * TOKEN_EMB_DIM);
        knnlm_ann_order[i] = i;
      }
      qsort(knnlm_ann_order, (size_t)knnlm_cache_size,
        sizeof(*knnlm_ann_order), knnlm_ann_cmp);
      knnlm_ann_size = knnlm_cache_size;
    }
    else {
      free(knnlm_ann_code);
      free(knnlm_ann_order);
      knnlm_ann_code = NULL;
      knnlm_ann_order = NULL;
    }
  }
}


/**
 * Exact-distance rerank of one candidate against the query into the running
 * top-K (same metric and heap discipline as knnlm_scan).
 */
static void knnlm_ann_consider(const double* in, int idx,
  unsigned int near_next[], double near_dist[], int* nnear)
{
  const double* entry = knnlm_cache_in + (size_t)idx * 2 * TOKEN_EMB_DIM;
  unsigned int cand = knnlm_cache_next[idx];
  double dist = 0.0;
  int dim, slot;
  if (0 == cand) return;
  for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
    double d2 = in[dim] - entry[dim];
    double d1 = in[TOKEN_EMB_DIM + dim] - entry[TOKEN_EMB_DIM + dim];
    dist += d2 * d2 + 4.0 * d1 * d1;
  }
  for (slot = *nnear; slot > 0; --slot) {
    if (dist >= near_dist[slot - 1]) break;
  }
  if (slot < KNNLM_K) {
    int move = (*nnear < KNNLM_K) ? *nnear : (KNNLM_K - 1);
    for (; move > slot; --move) {
      near_next[move] = near_next[move - 1];
      near_dist[move] = near_dist[move - 1];
    }
    near_next[slot] = cand;
    near_dist[slot] = dist;
    if (*nnear < KNNLM_K) ++*nnear;
  }
}


/**
 * Retrieve the static-cache neighbors of the query from a window around its
 * Hilbert code, then exact-rerank them. Replaces the O(N) scan of the static
 * cache; the dynamic store is still scanned brute-force by the caller.
 */
static void knnlm_ann_scan(const double* in, unsigned int near_next[],
  double near_dist[], int* nnear)
{
  if (knnlm_ann_size > 0) {
    uint64_t q = knnlm_ann_encode(in);
    int lo = 0, hi = knnlm_ann_size - 1, mid, left, right, taken;
    while (lo <= hi) {
      mid = (lo + hi) / 2;
      if (knnlm_ann_code[knnlm_ann_order[mid]] < q) lo = mid + 1;
      else hi = mid - 1;
    }
    if (lo > knnlm_ann_size - 1) lo = knnlm_ann_size - 1;
    left = lo - 1;
    right = lo;
    taken = 0;
    while (taken < KNNLM_ANN_WINDOW && (left >= 0 || right < knnlm_ann_size)) {
      int pick_left = 0;
      if (left >= 0 && right < knnlm_ann_size) {
        uint64_t cl = knnlm_ann_code[knnlm_ann_order[left]];
        uint64_t cr = knnlm_ann_code[knnlm_ann_order[right]];
        uint64_t dl = (q > cl) ? (q - cl) : (cl - q);
        uint64_t dr = (cr > q) ? (cr - q) : (q - cr);
        pick_left = (dl <= dr) ? 1 : 0;
      }
      else if (left >= 0) pick_left = 1;
      if (0 != pick_left) {
        knnlm_ann_consider(in, knnlm_ann_order[left], near_next, near_dist,
          nnear);
        --left;
      }
      else {
        knnlm_ann_consider(in, knnlm_ann_order[right], near_next, near_dist,
          nnear);
        ++right;
      }
      ++taken;
    }
  }
}


static void knnlm_dyn_reset(void)
{
  knnlm_dyn_size = 0;
}


static void knnlm_dyn_insert(const unsigned int hist[], int hlen, int ctxlen,
  unsigned int next)
{
  unsigned int prev1 = (hlen > 0) ? hist[hlen - 1] : 0;
  unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
  if (0 != prev1 && 0 != next) {
    if (knnlm_dyn_size >= knnlm_dyn_cap) {
      int cap = (knnlm_dyn_cap > 0) ? (2 * knnlm_dyn_cap) : 4096;
      double* nin = (double*)realloc(knnlm_dyn_in,
        (size_t)cap * 2 * TOKEN_EMB_DIM * sizeof(double));
      unsigned int* nnext = (unsigned int*)realloc(knnlm_dyn_next,
        (size_t)cap * sizeof(unsigned int));
      if (NULL == nin || NULL == nnext) {
        free(nin); free(nnext);
        return;
      }
      knnlm_dyn_in = nin;
      knnlm_dyn_next = nnext;
      knnlm_dyn_cap = cap;
    }
    if (ctxlen > 2) {
      knnlm_ctx_vector(hist, hlen, ctxlen, knnlm_decay(),
        knnlm_dyn_in + (size_t)knnlm_dyn_size * 2 * TOKEN_EMB_DIM);
    }
    else token_input_vector(prev2, prev1, 1,
      knnlm_dyn_in + (size_t)knnlm_dyn_size * 2 * TOKEN_EMB_DIM);
    knnlm_dyn_next[knnlm_dyn_size] = next;
    ++knnlm_dyn_size;
  }
}


/**
 * Number of retrieval heads (CONVERSE_KNNLM_HEADS, default 1 = bit-exact). With
 * h heads the embedding is split into h contiguous subspaces of TOKEN_EMB_DIM/h
 * dimensions; each head retrieves its own top-K over its own subspace distance,
 * and the votes are summed. This is multi-head attention's decomposition minus
 * the learned projections: the subspaces are fixed slices of the factorization
 * rather than anything W_Q/W_K could rotate into place.
 */
static int knnlm_heads(void)
{
  int result = 1;
  const char* env = getenv("CONVERSE_KNNLM_HEADS");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= TOKEN_EMB_DIM && 0 == (TOKEN_EMB_DIM % v)) result = v;
  }
  return result;
}


static void knnlm_scan_head(const double* in, const double* cin,
  const unsigned int* cnext, int count, int dim_begin, int dim_end,
  unsigned int near_next[], double near_dist[], int* nnear)
{
  int i, dim;
  for (i = 0; i < count; ++i) {
    const double* entry = cin + (size_t)i * 2 * TOKEN_EMB_DIM;
    double dist = 0.0;
    int slot;
    if (0 == cnext[i]) continue;
    for (dim = dim_begin; dim < dim_end; ++dim) {
      double d2 = in[dim] - entry[dim];
      double d1 = in[TOKEN_EMB_DIM + dim] - entry[TOKEN_EMB_DIM + dim];
      dist += d2 * d2 + 4.0 * d1 * d1;
    }
    for (slot = *nnear; slot > 0; --slot) {
      if (dist >= near_dist[slot - 1]) break;
    }
    if (slot < KNNLM_K) {
      int move = (*nnear < KNNLM_K) ? *nnear : (KNNLM_K - 1);
      for (; move > slot; --move) {
        near_next[move] = near_next[move - 1];
        near_dist[move] = near_dist[move - 1];
      }
      near_next[slot] = cnext[i];
      near_dist[slot] = dist;
      if (*nnear < KNNLM_K) ++*nnear;
    }
  }
}


/**
 * Diagonal learned projection for retrieval (CONVERSE_KNNLM_PROJ=1, default off
 * and bit-exact). Retrieval wants contexts that share a next token close and
 * contexts with different next tokens far, which is Fisher's criterion: weight
 * each dimension by sqrt(between-class / within-class scatter) over next-token
 * classes. Regression onto token ids would be meaningless -- ids carry no metric
 * -- so the objective is discriminative, not least-squares. Fitted from the
 * TRAINING entries only (the datastore already excludes held-out text) and
 * applied to BOTH the query and the stored keys, since a distance is only
 * meaningful when both sides live in one space.
 *
 * Diagonal first by design: it tests whether reweighting dimensions by
 * discriminative power helps at all, before paying for the symmetric generalized
 * eigenproblem a full matrix needs (no such solver exists in libxs_math). A
 * rotation can only add over a diagonal if the informative directions are
 * misaligned with the axes, and the PPMI factorization already delivers
 * variance-ordered axes.
 */
static double token_proj[2 * TOKEN_EMB_DIM];
static int token_proj_ready = 0;


static int token_proj_mode(void)
{
  const char* env = getenv("CONVERSE_KNNLM_PROJ");
  return (NULL != env && '0' != *env) ? 1 : 0;
}


static void token_proj_apply(double* vec)
{
  if (0 != token_proj_ready) {
    int d;
    for (d = 0; d < 2 * TOKEN_EMB_DIM; ++d) vec[d] *= token_proj[d];
  }
}


static void token_proj_build(const libxs_predict_t* store)
{
  enum { PROJ_CLASS_MAX = 256, PROJ_MINCOUNT = 20 };
  const int dims = 2 * TOKEN_EMB_DIM;
  libxs_predict_query_t stats;
  token_proj_ready = 0;
  if (0 == token_proj_mode() || NULL == store) return;
  libxs_predict_query(store, &stats);
  if (stats.nentries < PROJ_CLASS_MAX) return;
  { /* two passes: class means, then scatter about them */
    static double mean[PROJ_CLASS_MAX][2 * TOKEN_EMB_DIM];
    static double grand[2 * TOKEN_EMB_DIM];
    static double between[2 * TOKEN_EMB_DIM], within[2 * TOKEN_EMB_DIM];
    unsigned int class_id[PROJ_CLASS_MAX];
    long count[PROJ_CLASS_MAX];
    long freq_total = 0;
    int nclasses = 0, ci, d, i;
    /**
     * Class granularity: a next token becomes its own class only once it recurs
     * often enough to estimate a mean; everything rarer is pooled into class 0.
     * With thousands of distinct next tokens over tens of thousands of pairs the
     * within-class scatter would otherwise be estimated from single members and
     * be degenerate.
     */
    memset(mean, 0, sizeof(mean));
    memset(grand, 0, sizeof(grand));
    memset(between, 0, sizeof(between));
    memset(within, 0, sizeof(within));
    memset(count, 0, sizeof(count));
    class_id[0] = 0; /* pooled rare-token class */
    nclasses = 1;
    for (i = 0; i < stats.nentries; ++i) {
      double in[2 * TOKEN_EMB_DIM], out[1] = { 0 };
      unsigned int next;
      long seen;
      libxs_predict_get(store, i, in, out);
      next = (out[0] > 0.0) ? (unsigned int)(out[0] + 0.5) : 0;
      if (0 == next) continue;
      seen = (long)(ngram_unigram_prior(next) * (double)stats.nentries);
      ci = 0;
      if (seen >= PROJ_MINCOUNT) {
        int found = 0;
        for (ci = 1; ci < nclasses; ++ci) {
          if (class_id[ci] == next) { found = 1; break; }
        }
        if (0 == found) {
          if (nclasses < PROJ_CLASS_MAX) {
            class_id[nclasses] = next;
            ci = nclasses++;
          }
          else ci = 0;
        }
      }
      ++count[ci];
      ++freq_total;
      for (d = 0; d < dims; ++d) {
        mean[ci][d] += in[d];
        grand[d] += in[d];
      }
    }
    if (freq_total < PROJ_CLASS_MAX || nclasses < 2) return;
    for (ci = 0; ci < nclasses; ++ci) {
      if (count[ci] > 0) {
        for (d = 0; d < dims; ++d) mean[ci][d] /= (double)count[ci];
      }
    }
    for (d = 0; d < dims; ++d) grand[d] /= (double)freq_total;
    for (ci = 0; ci < nclasses; ++ci) {
      for (d = 0; d < dims; ++d) {
        const double diff = mean[ci][d] - grand[d];
        between[d] += (double)count[ci] * diff * diff;
      }
    }
    for (i = 0; i < stats.nentries; ++i) {
      double in[2 * TOKEN_EMB_DIM], out[1] = { 0 };
      unsigned int next;
      long seen;
      libxs_predict_get(store, i, in, out);
      next = (out[0] > 0.0) ? (unsigned int)(out[0] + 0.5) : 0;
      if (0 == next) continue;
      seen = (long)(ngram_unigram_prior(next) * (double)stats.nentries);
      ci = 0;
      if (seen >= PROJ_MINCOUNT) {
        for (ci = 1; ci < nclasses; ++ci) {
          if (class_id[ci] == next) break;
        }
        if (ci >= nclasses) ci = 0;
      }
      for (d = 0; d < dims; ++d) {
        const double diff = in[d] - mean[ci][d];
        within[d] += diff * diff;
      }
    }
    { double wmax = 0.0;
      for (d = 0; d < dims; ++d) {
        const double ratio = (within[d] > 0.0) ? (between[d] / within[d]) : 0.0;
        token_proj[d] = sqrt(ratio);
        if (token_proj[d] > wmax) wmax = token_proj[d];
      }
      if (wmax <= 0.0) return;
      for (d = 0; d < dims; ++d) token_proj[d] /= wmax;
      token_proj_ready = 1;
      fprintf(stderr, "projection: %d classes, weights", nclasses);
      for (d = 0; d < dims; d += TOKEN_EMB_DIM / 4) {
        fprintf(stderr, " %.2f", token_proj[d]);
      }
      fprintf(stderr, " (every %d dims)\n", TOKEN_EMB_DIM / 4);
    }
  }
}


static void knnlm_cache_build(const libxs_predict_t* store)
{
  libxs_predict_query_t stats;
  knnlm_cache_free();
  libxs_predict_query(store, &stats);
  if (stats.nentries > 0) {
    knnlm_cache_in = (double*)malloc((size_t)stats.nentries
      * 2 * TOKEN_EMB_DIM * sizeof(double));
    knnlm_cache_next = (unsigned int*)malloc((size_t)stats.nentries
      * sizeof(unsigned int));
    if (NULL != knnlm_cache_in && NULL != knnlm_cache_next) {
      int i;
      for (i = 0; i < stats.nentries; ++i) {
        double out[1] = { 0 };
        libxs_predict_get(store, i,
          knnlm_cache_in + (size_t)i * 2 * TOKEN_EMB_DIM, out);
        knnlm_cache_next[i] = (out[0] > 0.0)
          ? (unsigned int)(out[0] + 0.5) : 0;
      }
      knnlm_cache_size = stats.nentries;
      knnlm_cache_model = store;
      /* Fit the projection from the datastore, then map the keys into it; the
         query is mapped in knnlm_vote so both sides share one space. */
      token_proj_build(store);
      if (0 != token_proj_ready) {
        for (i = 0; i < stats.nentries; ++i) {
          token_proj_apply(knnlm_cache_in + (size_t)i * 2 * TOKEN_EMB_DIM);
        }
      }
    }
    else knnlm_cache_free();
  }
}


/**
 * Control distributions replacing the retrieved vote, to test whether the
 * kNN-LM gain comes from the learned metric or merely from interpolating some
 * smoother distribution into a sparse backoff estimator.
 * 1 = global unigram prior (no context, no retrieval at all).
 * 2 = K neighbors drawn by a coprime stride over the datastore (retrieval
 *     machinery and vote arithmetic intact, metric replaced by an arbitrary
 *     but reproducible selection).
 */
/**
 * Softmax temperature for the retrieval vote (CONVERSE_KNNLM_TEMP). Zero keeps
 * the historical inverse-distance kernel 1/(eps+d), bit-exact. A positive value
 * weights neighbor i by exp(-d_i/T) after subtracting the minimum distance for
 * numerical stability, which is attention's weighting rather than an ad-hoc
 * one: the temperature then controls how peaked the vote is, exactly as it
 * does in a transformer's attention.
 */
static double knnlm_temp(void)
{
  double result = 0.0;
  const char* env = getenv("CONVERSE_KNNLM_TEMP");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v > 0.0) result = v;
  }
  return result;
}


static int knnlm_control(void)
{
  const char* env = getenv("CONVERSE_KNNLM_CONTROL");
  int result = 0;
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 0 && v <= 2) result = v;
  }
  return result;
}


static int knnlm_vote_control(int mode, unsigned int position,
  unsigned int vote_ids[], double vote_p[], int maxvote)
{
  int result = 0;
  if (1 == mode) {
    const unsigned int vocab = converse_ngram.unifreq_size;
    unsigned int id;
    double total = 0.0;
    for (id = 1; id <= vocab; ++id) {
      const double p = ngram_unigram_prior(id);
      if (p > 0.0) {
        int slot = (result < maxvote) ? result : (maxvote - 1);
        if (result < maxvote) ++result;
        else if (vote_p[slot] >= p) continue;
        while (slot > 0 && vote_p[slot - 1] < p) {
          vote_ids[slot] = vote_ids[slot - 1];
          vote_p[slot] = vote_p[slot - 1];
          --slot;
        }
        vote_ids[slot] = id;
        vote_p[slot] = p;
      }
    }
    for (id = 0; id < (unsigned int)result; ++id) total += vote_p[id];
    if (total > 0.0) {
      for (id = 0; id < (unsigned int)result; ++id) vote_p[id] /= total;
    }
  }
  else if (2 == mode && knnlm_cache_size > 0) {
    const size_t stride = libxs_coprime_bias((size_t)knnlm_cache_size, -1.0);
    unsigned int uniq_id[KNNLM_K];
    double uniq_w[KNNLM_K];
    int nuniq = 0, i, j;
    double wtotal = 0.0;
    for (i = 0; i < KNNLM_K; ++i) {
      const size_t at = (stride * (position + (unsigned int)i) + 1)
        % (size_t)knnlm_cache_size;
      const unsigned int next = knnlm_cache_next[at];
      if (0 == next) continue;
      for (j = 0; j < nuniq; ++j) {
        if (uniq_id[j] == next) break;
      }
      if (j == nuniq) {
        uniq_id[j] = next;
        uniq_w[j] = 0.0;
        ++nuniq;
      }
      uniq_w[j] += 1.0;
      wtotal += 1.0;
    }
    while (result < maxvote && result < nuniq) {
      int best = -1;
      for (j = 0; j < nuniq; ++j) {
        if (0.0 <= uniq_w[j] && (best < 0 || uniq_w[j] > uniq_w[best])) {
          best = j;
        }
      }
      if (best < 0 || uniq_w[best] <= 0.0) break;
      vote_ids[result] = uniq_id[best];
      vote_p[result] = uniq_w[best] / wtotal;
      uniq_w[best] = -1.0;
      ++result;
    }
  }
  return result;
}


static int knnlm_vote(const libxs_predict_t* store, const unsigned int hist[],
  int hlen, int ctxlen, unsigned int vote_ids[], double vote_p[], int maxvote)
{
  unsigned int prev1 = (hlen > 0) ? hist[hlen - 1] : 0;
  unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
  int result = 0;
  if (NULL != store && 0 != prev1) {
    double in[2 * TOKEN_EMB_DIM];
    unsigned int near_next[KNNLM_K];
    double near_dist[KNNLM_K];
    unsigned int uniq_id[KNNLM_K];
    double uniq_w[KNNLM_K];
    int nnear = 0, nuniq = 0, i, j;
    double wtotal = 0.0, dmin = 0.0;
    const double temp = knnlm_temp();
    int ann = knnlm_ann_mode();
    const int control = knnlm_control();
    if (knnlm_cache_model != store) {
      knnlm_cache_build(store);
      if (0 != ann) knnlm_ann_build();
    }
    if (0 != control) {
      result = knnlm_vote_control(control, prev1 + 31u * prev2, vote_ids,
        vote_p, maxvote);
      nnear = 0;
      nuniq = 0;
    }
    else {
      const int heads = knnlm_heads();
      const int span = TOKEN_EMB_DIM / heads;
      int head;
      if (ctxlen > 2) knnlm_ctx_vector(hist, hlen, ctxlen, knnlm_decay(), in);
      else token_input_vector(prev2, prev1, 1, in);
      token_proj_apply(in);
      /**
       * One retrieval per head over its own subspace, votes accumulated into
       * the shared distribution. With heads==1 the loop performs exactly the
       * single full-width scan it replaced (and keeps using the ANN index,
       * which is built over the full vector).
       */
      for (head = 0; head < heads; ++head) {
        nnear = 0;
        if (1 == heads && 0 != ann) {
          knnlm_ann_scan(in, near_next, near_dist, &nnear);
        }
        else {
          knnlm_scan_head(in, knnlm_cache_in, knnlm_cache_next,
            knnlm_cache_size, head * span, (head + 1) * span,
            near_next, near_dist, &nnear);
        }
        knnlm_scan_head(in, knnlm_dyn_in, knnlm_dyn_next, knnlm_dyn_size,
          head * span, (head + 1) * span, near_next, near_dist, &nnear);
        /* Softmax needs the minimum distance first (stability); the historical
           inverse-distance kernel needs no such pass. */
        if (temp > 0.0) {
          for (i = 0; i < nnear; ++i) {
            if (i == 0 || near_dist[i] < dmin) dmin = near_dist[i];
          }
        }
        for (i = 0; i < nnear; ++i) {
          unsigned int next = near_next[i];
          double w = (temp > 0.0)
            ? exp(-(near_dist[i] - dmin) / temp)
            : (1.0 / (0.05 + near_dist[i]));
          for (j = 0; j < nuniq; ++j) {
            if (uniq_id[j] == next) break;
          }
          if (j == nuniq && nuniq < KNNLM_K) {
            uniq_id[j] = next;
            uniq_w[j] = 0.0;
            ++nuniq;
          }
          if (j < nuniq) {
            uniq_w[j] += w;
            wtotal += w;
          }
        }
      }
      while (result < maxvote && result < nuniq) {
        int best = -1;
        for (j = 0; j < nuniq; ++j) {
          if (0.0 <= uniq_w[j] && (best < 0 || uniq_w[j] > uniq_w[best])) {
            best = j;
          }
        }
        if (best < 0 || uniq_w[best] <= 0.0) break;
        vote_ids[result] = uniq_id[best];
        vote_p[result] = uniq_w[best] / wtotal;
        uniq_w[best] = -1.0;
        ++result;
      }
    }
  }
  return result;
}


static int knnlm_topk(libxs_registry_t* ngram, const libxs_predict_t* store,
  const unsigned int hist[], int hlen, int ctxlen, int order,
  unsigned int out_ids[], int k, unsigned int target, double* target_prob)
{
  unsigned int prev1 = (hlen > 0) ? hist[hlen - 1] : 0;
  unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
  unsigned int ids[NGRAM_SUCC_MAX + KNNLM_VOTE_MAX];
  double relfreq[NGRAM_SUCC_MAX];
  double score[NGRAM_SUCC_MAX + KNNLM_VOTE_MAX];
  int prov[NGRAM_SUCC_MAX];
  int taken[NGRAM_SUCC_MAX + KNNLM_VOTE_MAX];
  unsigned int vote_ids[KNNLM_VOTE_MAX];
  double vote_p[KNNLM_VOTE_MAX];
  int ctx_total = 0;
  int n = ngram_candidates(ngram, prev2, prev1, order, ids, relfreq, prov,
    &ctx_total, NGRAM_SUCC_MAX);
  int nvote = knnlm_vote(store, hist, hlen, ctxlen, vote_ids, vote_p,
    KNNLM_VOTE_MAX);
  double lambda;
  int result = 0, rank, v;
  const char* env = getenv("CONVERSE_KNNLM_LAMBDA");
  if (NULL != env && '\0' != *env) {
    lambda = atof(env);
  }
  else {
    double t = (double)ctx_total;
    int prov0 = (n > 0) ? prov[0] : 0;
    if (2 == prov0) lambda = t / (t + 1.0);
    else if (1 == prov0) lambda = 0.5;
    else lambda = 0.1;
  }
  if (lambda < 0.0) lambda = 0.0;
  if (lambda > 1.0) lambda = 1.0;
  for (rank = 0; rank < n; ++rank) {
    score[rank] = lambda * relfreq[rank];
    taken[rank] = 0;
  }
  for (v = 0; v < nvote; ++v) {
    for (rank = 0; rank < n; ++rank) {
      if (ids[rank] == vote_ids[v]) break;
    }
    if (rank < n) {
      score[rank] += (1.0 - lambda) * vote_p[v];
    }
    else if (n < NGRAM_SUCC_MAX + KNNLM_VOTE_MAX) {
      ids[n] = vote_ids[v];
      score[n] = (1.0 - lambda) * vote_p[v];
      taken[n] = 0;
      ++n;
    }
  }
  while (result < k && result < n) {
    int best = -1;
    for (rank = 0; rank < n; ++rank) {
      if (0 == taken[rank] && (best < 0 || score[rank] > score[best])) {
        best = rank;
      }
    }
    if (best < 0) break;
    taken[best] = 1;
    out_ids[result] = ids[best];
    ++result;
  }
  /**
   * A normalized probability for one target, needed to score this model in bits
   * rather than only by rank. The candidate scores above rank on relative
   * frequency within the retained successor list, which does not sum to one over
   * the vocabulary; the classic kNN-LM mixture does, because the n-gram term is
   * normalized and the vote mass sums to one over the neighbours. So lambda=1
   * must reproduce the n-gram baseline bit for bit, and that is the control
   * which says this is wired correctly rather than merely plausible.
   */
  if (NULL != target_prob) {
    double p = lambda * ngramk_prob(ngram, hist, hlen, ngram_maxorder(),
      target);
    for (v = 0; v < nvote; ++v) {
      if (vote_ids[v] == target) {
        p += (1.0 - lambda) * vote_p[v];
        break;
      }
    }
    *target_prob = p;
  }
  return result;
}


static int knnlm_eval(libxs_registry_t* ngram, const libxs_predict_t* store,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, int holdout,
  const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, seen = 0, index = 0;
  long nnullq = 0, nnulltgt = 0;
  long ndeep = 0, ndeep_top1 = 0, nshallow = 0, nshallow_top1 = 0;
  double deep_bits = 0.0, deep_bytes = 0.0;
  double shallow_bits = 0.0, shallow_bytes = 0.0;
  double sum_bits = 0.0, sum_bytes = 0.0;
  long ntrunc = 0, ntrunc_top1 = 0;
  double trunc_bits = 0.0, trunc_bytes = 0.0;
  const double inv_log2 = 1.0 / log(2.0);
  double ord_inner = 0.0, ord_cross = 0.0;
  long nord_inner = 0, nord_cross = 0;
  const char* ord_env = getenv("CONVERSE_KNNLM_ORDERPROBE");
  int order_probe = (NULL != ord_env && '0' != ord_env[0]) ? 1 : 0;
  int stride = predict_eval_stride();
  int ctxlen = knnlm_ctxlen();
  const char* dyn_env = getenv("CONVERSE_KNNLM_DYN");
  int dynamic = (NULL != dyn_env && '0' != dyn_env[0]) ? 1 : 0;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  if (NULL == ngram || NULL == store || NULL == corpus || NULL == lexicon) {
    return EXIT_FAILURE;
  }
  if (0 != dynamic) knnlm_dyn_reset();
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != ngram_dedup_scale() && (SCALE_SENTENCE != entry->scale
      || 0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT)))
    {
      is_test = 0;
    }
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
    {
      size_t pos;
      unsigned int hist[TOKEN_CTX_MAX];
      int hlen = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (hlen > 0) {
            if (0 == (seen % stride)) {
              unsigned int ids[NGRAM_TOPK];
              double p = 0.0;
              int n = knnlm_topk(ngram, store, hist, hlen, ctxlen, order, ids,
                NGRAM_TOPK, lex->id, &p);
              int rank, top1 = 0;
              /**
               * Whether the FULL-order context recurred in training, by the same
               * definition ngram_eval uses, so the two models' buckets are the
               * same set of positions. Retrieval has only ever been reported as
               * a top-k delta over all positions; the project's own criterion is
               * that a mechanism moving only the verbatim bucket has achieved
               * nothing, and that criterion had never been applied here.
               */
              int matched = 0;
              double bits;
              int deep;
              ngramk_predict_order(ngram, hist, hlen, ngram_maxorder(), NULL, 0,
                &matched);
              deep = (matched >= ngram_maxorder()
                && hlen >= ngram_maxorder()) ? 1 : 0;
              if (!(p > 0.0)) p = 1e-12;
              bits = -log(p) * inv_log2;
              ++npairs;
              sum_bits += bits;
              sum_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
              if (0 != token_emb_isnull(hist[hlen - 1])) ++nnullq;
              if (0 != token_emb_isnull(lex->id)) ++nnulltgt;
              if (0 != order_probe) {
                knnlm_order_probe(hist, hlen, ctxlen, knnlm_decay(),
                  &ord_inner, &ord_cross, &nord_inner, &nord_cross);
              }
              for (rank = 0; rank < n; ++rank) {
                if (ids[rank] == lex->id) {
                  if (0 == rank) {
                    ++ntop1;
                    top1 = 1;
                  }
                  ++ntopk;
                  break;
                }
              }
              if (0 != deep) {
                ++ndeep;
                ndeep_top1 += top1;
                deep_bits += bits;
                deep_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
              }
              else {
                const int mo = ngram_maxorder();
                const int avail = (hlen < mo) ? hlen : mo;
                ++nshallow;
                nshallow_top1 += top1;
                shallow_bits += bits;
                shallow_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
                /* The repaired generalization bucket; see ngram_eval. */
                if (matched < avail && hlen >= mo) {
                  ++ntrunc;
                  ntrunc_top1 += top1;
                  trunc_bits += bits;
                  trunc_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
                }
              }
            }
            ++seen;
            if (0 != dynamic) knnlm_dyn_insert(hist, hlen, ctxlen, lex->id);
          }
          ngram_hist_push(hist, &hlen, TOKEN_CTX_MAX, lex->id);
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-%s%s%s: top1=%.1f%% top%d=%.1f%% n=%ld (stride=%d) bpc=%.3f\n",
      kind, (0 != dynamic) ? ":dyn" : "", (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      stride, (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
    fprintf(stderr, "  attested-context split: verbatim %.1f%% of positions"
      " (top1=%.1f%% bpc=%.3f) | novel %.1f%% (top1=%.1f%% bpc=%.3f)\n",
      100.0 * (double)ndeep / (double)npairs,
      (ndeep > 0) ? 100.0 * (double)ndeep_top1 / (double)ndeep : 0.0,
      (deep_bytes > 0.0) ? deep_bits / deep_bytes : 0.0,
      100.0 * (double)nshallow / (double)npairs,
      (nshallow > 0) ? 100.0 * (double)nshallow_top1 / (double)nshallow : 0.0,
      (shallow_bytes > 0.0) ? shallow_bits / shallow_bytes : 0.0);
    fprintf(stderr, "  truncated %.1f%% of all (top1=%.1f%% bpc=%.3f)\n",
      100.0 * (double)ntrunc / (double)npairs,
      (ntrunc > 0) ? 100.0 * (double)ntrunc_top1 / (double)ntrunc : 0.0,
      (trunc_bytes > 0.0) ? trunc_bits / trunc_bytes : 0.0);
    fprintf(stderr, "embedding coverage: null query vectors %.2f%%,"
      " null targets %.2f%% (n=%ld)\n",
      100.0 * (double)nnullq / (double)npairs,
      100.0 * (double)nnulltgt / (double)npairs, npairs);
    if (0 != order_probe && nord_cross > 0) {
      fprintf(stderr, "order sensitivity (cosine of swapped context, 1.0 ="
        " order discarded): inner=%.4f (n=%ld) cross=%.4f (n=%ld)\n",
        (nord_inner > 0) ? (ord_inner / (double)nord_inner) : 1.0, nord_inner,
        ord_cross / (double)nord_cross, nord_cross);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


static void knnlm_complete(libxs_registry_t* ngram,
  const libxs_predict_t* store, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len)
{
  unsigned int prev2 = 0, prev1 = 0;
  unsigned int hist[2];
  unsigned int ids[NGRAM_TOPK];
  int n, i;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  hist[0] = prev2;
  hist[1] = prev1;
  n = knnlm_topk(ngram, store, hist, 2, 2, order, ids, NGRAM_TOPK, 0, NULL);
  if (n <= 0) {
    printf("(no continuation)\n");
    return;
  }
  printf("next:");
  for (i = 0; i < n; ++i) {
    int len = 0;
    const char* word = libxs_lexicon_text(lexicon, ids[i], &len, NULL);
    if (NULL != word && len > 0) printf(" %.*s", len, word);
  }
  printf("\n");
  { unsigned int step[2];
    int s;
    step[0] = prev2;
    step[1] = prev1;
    printf("greedy:");
    for (s = 0; s < 12; ++s) {
      unsigned int step_ids[1];
      int len = 0;
      const char* word;
      if (0 == knnlm_topk(ngram, store, step, 2, 2, order, step_ids, 1,
        0, NULL))
      {
        break;
      }
      word = libxs_lexicon_text(lexicon, step_ids[0], &len, NULL);
      if (NULL == word || len <= 0) break;
      printf(" %.*s", len, word);
      step[0] = step[1];
      step[1] = step_ids[0];
    }
    printf("\n");
  }
}


static int corpus_profile_for_path(const char* path)
{
  int result = PROFILE_PROSE;
  if (0 <= converse_profile_override) result = converse_profile_override;
  else if (NULL != path) {
    size_t len = strlen(path);
    if (len >= 3 && '.' == path[len - 3]
      && ('m' == path[len - 2] || 'M' == path[len - 2])
      && ('d' == path[len - 1] || 'D' == path[len - 1]))
    {
      result = PROFILE_MARKDOWN;
    }
  }
  return result;
}


/**
 * Index prose markdown blocks at sentence scale in addition to paragraph scale.
 *
 * Off by default, and the default is not tidiness: the n-gram trainer scans every
 * corpus entry without filtering on scale, so a sentence indexed alongside its
 * paragraph is trained on twice and every bits-per-character figure for the
 * documentation moves. Enabling this therefore changes a published baseline, and
 * the mechanisms that need it (recombination, which gates on sentence scale and
 * otherwise finds no host at all) are probes rather than defaults.
 *
 * Code blocks and tables are never split: they have no sentences, and splicing
 * them would produce text the word-level seam gates cannot judge.
 */
static int corpus_md_sentences(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_MD_SENTENCES");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


static int corpus_md_store(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like)
{
  int result = 0;
  int min_bytes = (0 != code_like) ? 2 : 8;
  int min_words = (0 != code_like) ? 0 : 3;
  while (len > 0 && 0 != isspace((unsigned char)text[len - 1])) --len;
  if (len >= min_bytes && len < COMPOSE_MAXTEXT
    && count_words(text, len) >= min_words)
  {
    corpus_entry_t entry;
    if (EXIT_SUCCESS == corpus_entry_build(&entry, text, len,
      SCALE_PARAGRAPH, lexicon, rules, nrules))
    {
      corpus_entry_set_section(&entry, section, section_len);
      if (0 != corpus_store_entry(corpus, &entry)) result = 1;
    }
    if (0 == code_like && 0 != corpus_md_sentences()) {
      int begin = 0, at;
      for (at = 0; at <= len; ++at) {
        const int is_end = (at == len)
          ? 1 : is_sentence_end_text(text, (size_t)len, (size_t)at);
        if (0 != is_end) {
          int end = (at < len) ? at + 1 : at;
          int span;
          size_t close_size = text_closer_size(text, (size_t)len, (size_t)end);
          while (0 != close_size) {
            end += (int)close_size;
            close_size = text_closer_size(text, (size_t)len, (size_t)end);
          }
          span = end - begin;
          while (0 < span && 0 != isspace((unsigned char)text[begin])) {
            ++begin;
            --span;
          }
          while (0 < span
            && 0 != isspace((unsigned char)text[begin + span - 1])) --span;
          /* A block that is one sentence is already stored above. */
          if (8 < span && span < len && 3 <= count_words(text + begin, span)
            && EXIT_SUCCESS == corpus_entry_build(&entry, text + begin, span,
              SCALE_SENTENCE, lexicon, rules, nrules))
          {
            corpus_entry_set_section(&entry, section, section_len);
            corpus_store_entry(corpus, &entry);
          }
          begin = end;
        }
      }
    }
  }
  return result;
}


static int corpus_md_emit_block(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like)
{
  int result = 0;
  if (len < COMPOSE_MAXTEXT) {
    result = corpus_md_store(corpus, text, len, section, section_len,
      lexicon, rules, nrules, code_like);
  }
  else {
    int line_start = 0, i;
    for (i = 0; i <= len; ++i) {
      if (i == len || '\n' == text[i]) {
        if (i > line_start) {
          result += corpus_md_store(corpus, text + line_start,
            i - line_start, section, section_len, lexicon, rules, nrules,
            code_like);
        }
        line_start = i + 1;
      }
    }
  }
  return result;
}


static int corpus_ingest_markdown(libxs_registry_t* corpus,
  const unsigned char* text, size_t text_size, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int nblocks = 0;
  char section[ENTRY_SECTION_MAX];
  int section_len = 0;
  size_t line_start = 0, block_start = 0;
  int in_fence = 0, block_code = 0, block_has = 0;
  size_t i;
  section[0] = '\0';
  for (i = 0; i <= text_size; ++i) {
    if (i == text_size || '\n' == text[i]) {
      size_t ls = line_start;
      int line_len = (int)(i - line_start);
      int fence = 0;
      while (ls < i && 0 != isspace((unsigned char)text[ls])) ++ls;
      fence = (i - ls >= 3 && '`' == text[ls] && '`' == text[ls + 1]
        && '`' == text[ls + 2]) ? 1 : 0;
      if (0 != in_fence) {
        block_has = 1;
        if (0 != fence) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(i - block_start), section, section_len, lexicon, rules,
            nrules, 1);
          in_fence = 0;
          block_start = i + 1;
          block_has = 0;
        }
      }
      else if (0 != fence) {
        if (0 != block_has) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), section, section_len, lexicon,
            rules, nrules, block_code);
        }
        in_fence = 1;
        block_start = line_start;
        block_code = 1;
        block_has = 0;
      }
      else if ('#' == (ls < i ? text[ls] : 0)) {
        size_t h = ls;
        if (0 != block_has) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), section, section_len, lexicon,
            rules, nrules, block_code);
          block_has = 0;
        }
        while (h < i && '#' == text[h]) ++h;
        while (h < i && 0 != isspace((unsigned char)text[h])) ++h;
        section_len = (int)(i - h);
        if (section_len >= ENTRY_SECTION_MAX) section_len = ENTRY_SECTION_MAX - 1;
        if (section_len > 0) {
          memcpy(section, text + h, (size_t)section_len);
          section[section_len] = '\0';
        }
        else section[0] = '\0';
        block_start = i + 1;
        block_code = 0;
      }
      else if (0 == line_len) {
        size_t bs = block_start;
        int header_only = 0;
        while (bs < line_start && 0 != isspace((unsigned char)text[bs])) ++bs;
        if (0 != block_has && line_start - bs > 7
          && 0 == strncmp((const char*)text + bs, "Header:", 7))
        {
          header_only = 1;
        }
        if (0 != block_has && 0 == header_only) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), section, section_len, lexicon,
            rules, nrules, block_code);
          block_has = 0;
        }
        if (0 == header_only) {
          block_start = i + 1;
          block_code = 0;
        }
      }
      else {
        int table = ('|' == text[ls]) ? 1 : 0;
        if (0 == block_has) block_code = table;
        block_has = 1;
      }
      line_start = i + 1;
    }
  }
  if (0 != block_has) {
    nblocks += corpus_md_emit_block(corpus, text + block_start,
      (int)(text_size - block_start), section, section_len, lexicon, rules,
      nrules, block_code);
  }
  fprintf(stderr, "  ingested %s: %d markdown blocks\n", path, nblocks);
  return (nblocks > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}


static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = EXIT_FAILURE;
  int profile;
  FILE* f;
  unsigned char* text = NULL;
  size_t text_size = 0;
  if (NULL == corpus || NULL == path) return EXIT_FAILURE;
  if (corpus_source_id < 0xffff) ++corpus_source_id;
  profile = corpus_profile_for_path(path);
  f = fopen(path, "rb");
  if (NULL != f) {
    fseek(f, 0, SEEK_END);
    text_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    if (text_size > 0) {
      text = (unsigned char*)malloc(text_size + 1);
      if (NULL != text) {
        if (fread(text, 1, text_size, f) == text_size) {
          text[text_size] = 0;
          result = EXIT_SUCCESS;
        }
      }
    }
    fclose(f);
  }
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0
    && PROFILE_MARKDOWN != profile)
  {
    unsigned char* reflowed = NULL;
    size_t reflowed_size = 0;
    if (EXIT_SUCCESS == libxs_text_reflow(text, text_size,
      &reflowed, &reflowed_size))
    {
      free(text);
      text = reflowed;
      text_size = reflowed_size;
    }
  }
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0
    && PROFILE_MARKDOWN == profile)
  {
    result = corpus_ingest_markdown(corpus, text, text_size, path,
      lexicon, rules, nrules);
  }
  else if (EXIT_SUCCESS == result && NULL != text && text_size > 0) {
    size_t sent_start = 0;
    int nsentences = 0, nparagraphs = 0;
    char current_section[ENTRY_SECTION_MAX];
    int current_section_len = 0;
    size_t i;
    current_section[0] = '\0';
    for (i = 0; i <= text_size; ++i) {
      int is_sent_end = (i == text_size)
        ? 1 : is_sentence_end_text(text, text_size, i);
      if (0 != is_sent_end && i > sent_start) {
          size_t sent_end = (i < text_size) ? i + 1 : i;
          int len = (int)(sent_end - sent_start);
          while (sent_end < text_size) {
            size_t close_size = text_closer_size(text, text_size, sent_end);
            if (0 == close_size) break;
            sent_end += close_size;
          }
          len = (int)(sent_end - sent_start);
          while (len > 0 && 0 != isspace(text[sent_start + len - 1])) --len;
          if (len > 8) {
            char title[ENTRY_SECTION_MAX];
            int title_len = corpus_title_prefix(text + sent_start, len,
              title, (int)sizeof(title));
            if (title_len > 0) {
              memcpy(current_section, title, (size_t)title_len + 1);
              current_section_len = title_len;
            }
          }
          if (len >= COMPOSE_MAXTEXT) {
            size_t frag_start = sent_start;
            while (frag_start < sent_end) {
              size_t scan = frag_start;
              size_t frag_end = frag_start;
              while (scan < sent_end && scan - frag_start < 240) {
                if (',' == text[scan] || ';' == text[scan]
                  || ':' == text[scan] || '.' == text[scan]
                  || '?' == text[scan] || '!' == text[scan])
                {
                  frag_end = scan + 1;
                }
                ++scan;
              }
              if (frag_end > frag_start) {
                size_t trim_start = frag_start;
                int frag_len;
                while (trim_start < frag_end
                  && 0 != isspace(text[trim_start])) ++trim_start;
                frag_len = (int)(frag_end - trim_start);
                while (frag_len > 0
                  && 0 != isspace(text[trim_start + frag_len - 1])) --frag_len;
                if (frag_len > 24 && frag_len < COMPOSE_MAXTEXT
                  && count_words(text + trim_start, frag_len) >= 4)
                {
                  corpus_entry_t entry;
                  if (EXIT_SUCCESS == corpus_entry_build(&entry,
                    text + trim_start, frag_len, SCALE_SENTENCE,
                    lexicon, rules, nrules))
                  {
                    /* A clause cut from the sentence stored below: its bytes
                       are already covered by that entry. */
                    entry.lexical_flags |= ENTRY_LEX_FRAGMENT;
                    corpus_entry_set_section(&entry, current_section,
                      current_section_len);
                    if (0 != corpus_store_entry(corpus, &entry)) ++nsentences;
                  }
                }
              }
              while (frag_start < sent_end && ',' != text[frag_start]
                && ';' != text[frag_start] && ':' != text[frag_start]
                && '.' != text[frag_start] && '?' != text[frag_start]
                && '!' != text[frag_start]) ++frag_start;
              if (frag_start < sent_end) ++frag_start;
              while (frag_start < sent_end && 0 != isspace(text[frag_start])) {
                ++frag_start;
              }
            }
          }
          if (len > 8 && len < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + sent_start, len);
            if (nwords >= 3) {
              corpus_entry_t entry;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + sent_start, len, SCALE_SENTENCE,
                lexicon, rules, nrules))
              {
                corpus_entry_set_section(&entry, current_section,
                  current_section_len);
                if (0 != corpus_store_entry(corpus, &entry)) ++nsentences;
              }
            }
          }
          sent_start = sent_end;
          while (sent_start < text_size && 0 != isspace(text[sent_start])) {
            ++sent_start;
          }
          i = (sent_start > 0) ? sent_start - 1 : 0;
      }
    }
    { size_t para_start = 0, p;
      char para_section[ENTRY_SECTION_MAX];
      int para_section_len = 0;
      para_section[0] = '\0';
      for (p = 0; p < text_size; ++p) {
        if ('\n' == text[p] && p + 1 < text_size && '\n' == text[p + 1]) {
          int plen = (int)(p - para_start);
          while (plen > 0 && 0 != isspace(text[para_start + plen - 1]))
            --plen;
          if (plen > 8) {
            char title[ENTRY_SECTION_MAX];
            int title_len = corpus_title_prefix(text + para_start, plen,
              title, (int)sizeof(title));
            if (title_len > 0) {
              memcpy(para_section, title, (size_t)title_len + 1);
              para_section_len = title_len;
            }
          }
          if (plen >= COMPOSE_MAXTEXT) {
            size_t frag_start = para_start;
            size_t para_end = para_start + (size_t)plen;
            while (frag_start < para_end) {
              size_t scan = frag_start;
              size_t frag_end = frag_start;
              while (scan < para_end && scan - frag_start < 240) {
                if (',' == text[scan] || ';' == text[scan]
                  || ':' == text[scan] || '.' == text[scan]
                  || '?' == text[scan] || '!' == text[scan])
                {
                  frag_end = scan + 1;
                }
                ++scan;
              }
              if (frag_end > frag_start) {
                size_t trim_start = frag_start;
                int frag_len;
                while (trim_start < frag_end
                  && 0 != isspace(text[trim_start])) ++trim_start;
                frag_len = (int)(frag_end - trim_start);
                while (frag_len > 0
                  && 0 != isspace(text[trim_start + frag_len - 1])) --frag_len;
                if (frag_len > 24 && frag_len < COMPOSE_MAXTEXT
                  && count_words(text + trim_start, frag_len) >= 4)
                {
                  corpus_entry_t entry;
                  if (EXIT_SUCCESS == corpus_entry_build(&entry,
                    text + trim_start, frag_len, SCALE_SENTENCE,
                    lexicon, rules, nrules))
                  {
                    /* A clause cut from the sentence stored below: its bytes
                       are already covered by that entry. */
                    entry.lexical_flags |= ENTRY_LEX_FRAGMENT;
                    corpus_entry_set_section(&entry, para_section,
                      para_section_len);
                    if (0 != corpus_store_entry(corpus, &entry)) ++nsentences;
                  }
                }
              }
              while (frag_start < para_end && ',' != text[frag_start]
                && ';' != text[frag_start] && ':' != text[frag_start]
                && '.' != text[frag_start] && '?' != text[frag_start]
                && '!' != text[frag_start]) ++frag_start;
              if (frag_start < para_end) ++frag_start;
              while (frag_start < para_end && 0 != isspace(text[frag_start])) {
                ++frag_start;
              }
            }
          }
          if (plen > 40 && plen < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + para_start, plen);
            if (nwords >= 8) {
              corpus_entry_t entry;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + para_start, plen, SCALE_PARAGRAPH,
                lexicon, rules, nrules))
              {
                corpus_entry_set_section(&entry, para_section,
                  para_section_len);
                if (0 != corpus_store_entry(corpus, &entry)) ++nparagraphs;
              }
            }
          }
          while (p + 1 < text_size && '\n' == text[p + 1]) ++p;
          para_start = p + 1;
        }
      }
    }
    fprintf(stderr, "  ingested %s: %d sentences, %d paragraphs\n",
      path, nsentences, nparagraphs);
  }
  free(text);
  return result;
}


static int corpus_ingest_basename(libxs_registry_t* corpus,
  const char* basename, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules)
{
  static const char* const separators[] = { "-", "_", "." };
  size_t basename_len;
  size_t ningested = 0;
  unsigned int part;
  int sep_index;
  int result = EXIT_SUCCESS;
  if (NULL == corpus || NULL == basename || '\0' == basename[0]) {
    return EXIT_FAILURE;
  }
  basename_len = strlen(basename);
  { char* path = (char*)malloc(basename_len + 5);
    if (NULL == path) return EXIT_FAILURE;
    sprintf(path, "%s", basename);
    if (EXIT_SUCCESS == corpus_ingest_file(corpus, path, lexicon,
      rules, nrules)) ++ningested;
    sprintf(path, "%s.txt", basename);
    if (EXIT_SUCCESS == corpus_ingest_file(corpus, path, lexicon,
      rules, nrules)) ++ningested;
    free(path);
  }
  for (part = 1; part <= CORPUS_BASENAME_PART_MAX; ++part) {
    for (sep_index = 0; sep_index < 3; ++sep_index) {
      char digits[16];
      char* path;
      size_t path_size;
      sprintf(digits, "%u", part);
      path_size = basename_len + strlen(separators[sep_index])
        + strlen(digits) + 5;
      path = (char*)malloc(path_size);
      if (NULL == path) {
        result = EXIT_FAILURE;
        part = CORPUS_BASENAME_PART_MAX;
        break;
      }
      sprintf(path, "%s%s%s.txt", basename, separators[sep_index], digits);
      if (EXIT_SUCCESS == corpus_ingest_file(corpus, path, lexicon,
        rules, nrules)) ++ningested;
      free(path);
    }
  }
  if (0 == ningested && EXIT_SUCCESS == result) {
    fprintf(stderr, "basename %s: no companion text files (name only)\n",
      basename);
    result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == result) {
    fprintf(stderr, "basename %s: %lu files\n", basename,
      (unsigned long)ningested);
  }
  return result;
}


static double converse_score(const corpus_fprint_t* candidate,
  const libxs_fprint_t* query, const corpus_fprint_t* prev)
{
  double low = 0, high = 0, coherence = 0;
  int kmax = candidate->order < query->order
    ? candidate->order : query->order;
  int k;
  for (k = 0; k <= kmax; ++k) {
    double va = (candidate->nk[k] > 0)
      ? candidate->acc_sq[k] / candidate->nk[k] : 0;
    double vb = (query->nk[k] > 0)
      ? query->acc_sq[k] / query->nk[k] : 0;
    double ma = (candidate->nk[k] > 0)
      ? candidate->acc_sum[k] / candidate->nk[k] : 0;
    double mb = (query->nk[k] > 0)
      ? query->acc_sum[k] / query->nk[k] : 0;
    double dv = va - vb, dm = ma - mb;
    double d2 = dv * dv + dm * dm;
    if (k <= 1) low += d2;
    else high += d2;
  }
  if (NULL != prev) {
    for (k = 0; k <= kmax; ++k) {
      double va = (candidate->nk[k] > 0)
        ? candidate->acc_sq[k] / candidate->nk[k] : 0;
      double vb = (prev->nk[k] > 0)
        ? prev->acc_sq[k] / prev->nk[k] : 0;
      double ma = (candidate->nk[k] > 0)
        ? candidate->acc_sum[k] / candidate->nk[k] : 0;
      double mb = (prev->nk[k] > 0)
        ? prev->acc_sum[k] / prev->nk[k] : 0;
      double dv = va - vb, dm = ma - mb;
      coherence += dv * dv + dm * dm;
    }
  }
  return (low + 0.2 * coherence) / (1.0 + 0.1 * high);
}


static int entry_used(const corpus_entry_t* e,
  const corpus_entry_t* const used[], int nused)
{
  int result = 0, u;
  for (u = 0; u < nused && 0 == result; ++u) {
    if (used[u]->text_len == e->text_len
      && 0 == libxs_memcmp(used[u]->text, e->text, (size_t)e->text_len))
    {
      result = 1;
    }
  }
  return result;
}


static const corpus_entry_t* select_best(void* const candidates[],
  int ncandidates, const corpus_entry_t* const used[], int nused,
  const libxs_fprint_t* query, const corpus_fprint_t* prev,
  int require_scale, unsigned char preferred_scale)
{
  const corpus_entry_t* result = NULL;
  double best_score = 1e30;
  int ci;
  for (ci = 0; ci < ncandidates; ++ci) {
    const corpus_entry_t* e = (const corpus_entry_t*)candidates[ci];
    if (e->text_len >= 40
      && (0 == require_scale || e->scale == preferred_scale)
      && 0 == entry_used(e, used, nused))
    {
      double score = converse_score(&e->fprint, query, prev);
      if (score < best_score) {
        best_score = score;
        result = e;
      }
    }
  }
  return result;
}


static uint64_t query_hilbert_code(const libxs_fprint_t* fp)
{
  unsigned int coords[COMPOSE_NDIMS];
  int k;
  for (k = 0; k <= 4 && k <= fp->order; ++k) {
    double v = fp->l2[k], m = fp->mean[k];
    if (v < 0) v = 0;
    if (v > 1.0) v = 1.0;
    if (m < -1.0) m = -1.0;
    if (m > 1.0) m = 1.0;
    coords[k] = (unsigned int)(v * ((1 << COMPOSE_BITS) - 1));
    coords[5 + k] = (unsigned int)((m + 1.0) * 0.5 * ((1 << COMPOSE_BITS) - 1));
  }
  for (k = fp->order + 1; k <= 4; ++k) {
    coords[k] = 0;
    coords[5 + k] = 0;
  }
  return libxs_hilbert_bits(coords, COMPOSE_NDIMS, COMPOSE_BITS);
}


static int respond(const libxs_spatial_t* spatial,
  const libxs_registry_t* corpus, const char* query_text,
  size_t query_len, const libxs_fprint_t* query, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile)
{
  int result = EXIT_SUCCESS;
  char output[65536];
  size_t out_pos = 0;
  int step;
  corpus_fprint_t prev_fprint;
  int have_prev = 0;
  const corpus_entry_t* used[64];
  int nused = 0;
  void* candidates[256];
  int ncandidates;
  uint64_t qcode = query_hilbert_code(query);
  unsigned char preferred_scale = (query->n > 60)
    ? SCALE_PARAGRAPH : SCALE_SENTENCE;

  if (0 != is_question_query(query_text, query_len, lexicon, rules, nrules)) {
    char rewritten[COMPOSE_MAXTEXT];
    const char* q = query_text;
    size_t qlen = query_len;
    if (EXIT_SUCCESS == conv_rewrite(query_text, query_len, rewritten,
      sizeof(rewritten)))
    {
      q = rewritten;
      qlen = strlen(rewritten);
    }
    if (0 != answer_query(corpus, q, qlen, budget,
      lexicon, rules, nrules, answer_model, profile, NULL, 0))
    {
      conv_remember(q, qlen);
      return result;
    }
    { char gen[COMPOSE_MAXTEXT];
      double order_mean = 0.0;
      int ntok = ngram_generate(converse_ngram.store, lexicon, rules, nrules,
        q, qlen, gen, sizeof(gen), &order_mean, NULL);
      if (ntok > 0 && order_mean >= ngram_gen_contfloor()) {
        printf("%s\n", gen);
        return result;
      }
    }
    printf("I do not know from the corpus.\n");
    return result;
  }

  ncandidates = libxs_spatial_nearest(spatial, qcode, 256, candidates);

  for (step = 0; step < budget; ++step) {
    const corpus_fprint_t* prev = (0 != have_prev) ? &prev_fprint : NULL;
    const corpus_entry_t* best_entry = select_best(candidates, ncandidates,
      used, nused, query, prev, 1, preferred_scale);

    if (NULL == best_entry) {
      best_entry = select_best(candidates, ncandidates,
        used, nused, query, prev, 0, preferred_scale);
    }
    if (NULL == best_entry) break;
    if (nused < 64) used[nused++] = best_entry;

    { const char* txt = best_entry->text;
      int tlen = best_entry->text_len;
      while (tlen > 0 && 0 != isspace((unsigned char)*txt)) {
        ++txt; --tlen;
      }
      while (tlen > 0 && 0 != isspace((unsigned char)txt[tlen - 1])) {
        --tlen;
      }
      if (tlen <= 0) continue;
      if (out_pos > 0 && out_pos + 1 < sizeof(output)) {
        output[out_pos++] = '\n';
      }
      if (out_pos + (size_t)tlen < sizeof(output)) {
        memcpy(output + out_pos, txt, (size_t)tlen);
        out_pos += (size_t)tlen;
      }
    }

    prev_fprint = best_entry->fprint;
    have_prev = 1;
  }

  if (out_pos > 0) {
    printf("%.*s\n", (int)out_pos, output);
  }
  return result;
}
