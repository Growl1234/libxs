#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_malloc.h>

#include "converse_qa.h"
#include "converse_recomb.h"

#define ANSWER_MAX 4

#define ANSWER_MIN_SCORE 0.35

#define BRIDGE_LINE_MAX 2048

/** Fact-field marker: this expectation depends on loaded rules. */
#define EVAL_RULE_GOVERNED "-"

#define CONV_TOPIC_MAX 64


typedef struct answer_bridge_t {
  const char* name;
  const char* query;
  const char* evidence;
  const char* reply;
  double score;
} answer_bridge_t;

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


static answer_bridge_t* answer_bridge_loaded = NULL;
static size_t answer_bridge_loaded_size = 0;

/**
 * Sections of the most recent fact reply, so a citation can be printed without
 * threading an out-parameter through five resolver signatures. Cleared by
 * answer_fact_reply before dispatch and set by whichever resolver answered;
 * empty means the fact carried no section and no citation is printed.
 *
 * Several, because a reply can rest on several facts, and the two paths that
 * aggregate over facts cited NOTHING. For a claim about attribution an
 * unattributed answer is worse than a misattributed one, since nothing signals
 * it. Each contributing source is named; the usual case contributes one and
 * reads exactly as a single citation always did.
 */
static char answer_fact_section[4 * ENTRY_SECTION_MAX];
static int answer_fact_section_len = 0;
/**
 * Class term the most recent fact reply rested on, and where that term came
 * from, when it was not asserted in the rule file. Carried the same way the
 * citation is, and for the same reason: five resolvers would otherwise each grow
 * an out-parameter. Empty means the reply rests on asserted rules only.
 */
static char answer_fact_learned[64];
static int answer_fact_learned_len = 0;
static int answer_fact_learned_from = RELATION_RULE_ASSERTED;
static libxs_lexicon_t* answer_query_lexicon = NULL;
static const libxs_lexrule_t* answer_query_rules = NULL;
static int answer_query_nrules = 0;
/**
 * Per-token case census: occurrences that carried an initial upper-case letter,
 * and occurrences in total.
 *
 * LIBXS_LEXEME_ENTITY is exactly that per OCCURRENCE (the default rule set marks
 * WORD_INITIAL_UPPER), and the lexicon interns the LOWER-CASED form, so both
 * surface forms of a word share one id. Aggregating the flag by id is therefore
 * what separates a name from a common word: a name is never written lower-case
 * and a common noun is -- and the test needs no threshold, because one
 * lower-case occurrence is enough to settle it.
 */
static unsigned int* answer_case_upper = NULL;
static unsigned int* answer_case_total = NULL;
static unsigned int answer_case_size = 0;
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

static long answer_hier_nreorder = 0;
static long answer_hier_nchanged = 0;


static void answer_bridge_free_const(const char* ptr);
static char* answer_bridge_copy_trim(const char* text);
static void answer_bridge_free_loaded(void);
static int answer_bridge_append_loaded(const char* name, const char* query,
  const char* evidence, const char* score, const char* reply);
static int answer_bridge_parse_line(char* line);
static size_t answer_bridge_load_file(const char* path);
static void answer_bridge_report(FILE* stream);
static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len);
static int text_starts_sentence(const char* text, int text_len);
static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id);
static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text);
static int lexeme_stream_has_similar_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  int tolerance);
static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon);
static int query_type_prefers_sentence(int query_type);
static int corpus_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* corpus);
static int answer_query_section(const char* query_text, size_t query_len,
  char* title, int title_size);
static int corpus_entry_section_copy(const corpus_entry_t* entry,
  size_t entry_size, char* title, int title_size);
static int corpus_entry_section_match(const corpus_entry_t* entry,
  size_t entry_size, const char* title, int title_len);
static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size);
static void answer_case_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static void answer_case_free(void);
static int answer_word_is_name(const char* word, int word_len);
static int answer_query_relation_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size);
static int answer_relation_copy_name(char* output, int output_size,
  const char* text, int text_len, int begin, int end);
static int answer_relation_actor_has_token(const char* actor, int actor_len,
  const char* token, int token_len);
static int answer_relation_copy_section_head(char* output, int output_size,
  const corpus_entry_t* entry);
static int answer_relation_copy_antecedent(char* output, int output_size,
  const char* text, int text_len, int cue_pos);
static int answer_relation_find_person_before(const char* text, int text_len,
  int limit, int* term_pos, int* term_len);
/** Only a person answers a "who" question: a name, or a person-class term. */
static int answer_relation_answer_is_person(const char* text, int text_len);
static int answer_relation_copy_person_before(char* output, int output_size,
  const char* text, int text_len, int limit);
static int answer_relation_match_query(const char* query_text,
  size_t query_len, int query_type, const corpus_entry_t* entry,
  answer_relation_match_t* match);
static int answer_relation_section_title(char* output, int output_size,
  const corpus_entry_t* entry, const answer_relation_match_t* match);
static int answer_relation_same_answer(const char* lhs, int lhs_len,
  const char* rhs, int rhs_len);
static void answer_relation_facts_free(void);
static int answer_relation_fact_append(const corpus_entry_t* entry,
  const answer_relation_match_t* match);
static int answer_relation_fact_extract_actor(const char* text, int text_len,
  int verb_pos, int* scan_pos, char* actor, int actor_size);
static int answer_relation_fact_extract_made(const corpus_entry_t* entry,
  int made_pos);
static size_t answer_relation_facts_build(const libxs_registry_t* corpus);
static void answer_relation_facts_report(FILE* stream);
static void answer_identity_facts_free(void);
static int answer_identity_word_is_name(const char* word, int word_len);
static int answer_identity_fact_append(const char* name, int name_len,
  const char* role, int role_len, const corpus_entry_t* entry, double score);
static int answer_identity_is_connective(const char* word, int word_len);
static size_t answer_identity_facts_build(const libxs_registry_t* corpus);
static void answer_identity_facts_report(FILE* stream);
static int answer_identity_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_describe_facts_free(void);
static int answer_describe_fact_append(const char* role, int role_len,
  const char* text, int text_len, const corpus_entry_t* entry, double score);
static int answer_describe_word_is_article(const char* word, int word_len);
static size_t answer_describe_facts_build(const libxs_registry_t* corpus);
static void answer_describe_facts_report(FILE* stream);
static int answer_describe_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_docdef_facts_free(void);
static int answer_docdef_header(const char* text, int text_len,
  char* header, int header_size, int* header_len);
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus);
static void answer_docdef_facts_report(FILE* stream);
static int answer_docdef_term(const char* query_text, size_t query_len,
  char* term, int term_size);
static int answer_docdef_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void conv_reset(void);
static void conv_remember(const char* query_text, size_t query_len);
static int conv_word_is_pronoun(const char* word, int len);
static int conv_rewrite(const char* query_text, size_t query_len,
  char* out, size_t out_size);
static int answer_relation_fact_relation_match(const char* query_relation,
  const answer_relation_fact_t* fact);
static int answer_relation_fact_actor_match(const char* query_actor,
  int query_actor_len, const answer_relation_fact_t* fact);
static int answer_relation_fact_section_match(const char* query_section,
  int query_section_len, const answer_relation_fact_t* fact);
static int answer_relation_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static int answer_relation_aggregate_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size);
static double answer_identity_score(const char* query_text, size_t query_len,
  int query_type, const corpus_entry_t* entry);
static int answer_features(const libxs_lexeme_stream_t* query,
  const corpus_entry_t* entry, size_t entry_size, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS]);
static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile);
static double answer_predict_score(const libxs_predict_t* model,
  const double inputs[ANSWER_PREDICT_INPUTS], double base_score);
static int answer_bridge_query_group_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const char* group, int group_len);
static int answer_bridge_query_match(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* spec);
static int answer_bridge_evidence_group_match(const char* text, int text_len,
  const char* group, int group_len);
static int answer_bridge_evidence_match(const corpus_entry_t* entry,
  const char* spec);
static const answer_bridge_t* answer_bridge_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const corpus_entry_t* entry);
static double answer_semantic_bridge_score(const answer_bridge_t* bridge);
static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type);
static int answer_select(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX]);
static int answer_reply_role(char* output, size_t output_size,
  const char* name, int name_len, const char* role);
static void answer_strip_heading_prefix(const char** text, int* text_len);
static size_t answer_append_clean(char* output, size_t output_size,
  size_t output_pos, const char* text, int text_len);
static int answer_frame_after(char* output, size_t output_size,
  size_t* output_pos, const char* text, int text_len, const char* marker);
static int answer_frame_keywords_after(char* output, size_t output_size,
  size_t* output_pos, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  const char* marker);
static int answer_bridge_expand_reply(const answer_bridge_t* bridge,
  const char* text, int text_len, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, char* output, size_t output_size);
static int answer_reply_what_is(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  char* output, size_t output_size);
static int answer_reply(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_evidence_sentence(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_query_is_negated(const char* query_text, size_t query_len);
static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output, size_t output_size);
static int answer_citation_len(const char* section, int section_len);
static void answer_print_citation(const char* section, int section_len);
static void answer_fact_section_set(const char* section, int section_len);
/** Name one more source, for a reply that rests on several facts. */
static void answer_fact_section_add(const char* section, int section_len);
static void answer_fact_learned_set(const char* term, int term_len,
  int provenance);
static void answer_print_learned(void);
static int answer_hier_reorder(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX],
  int answer_count);
/** What a reader is shown for an answer list, printed unless print is 0. */
static int answer_render(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[], int answer_count,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int print, char* output, size_t output_size);
static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  char* out_reply, size_t out_size);
static int text_contains_ci(const char* text, int text_len, const char* term);
static int text_find_word_ci(const char* text, int text_len, const char* term);
static int eval_parse_line(char* line, char* fields[5]);
static int eval_terms_empty(const char* spec);
static int eval_terms_match_text(const char* text, int text_len,
  const char* spec);
static int eval_terms_match_answers(const corpus_entry_t* entries[],
  int nanswers, const char* spec, int top_only);
static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
static int recomb_compose_on(void);
static double ngram_gen_contfloor(void);
static double gen_relevance_min(void);
static double gen_overlap(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* a, int alen,
  const char* b, int blen);
static void complete_respond(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile, int budget,
  const char* text, int text_len);
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


static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len)
{
  int result = -1;
  size_t rule_pos;
  if (NULL != relation && NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < converse_rules_size() && result < 0;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
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
      converse_lexnorms(), converse_lexnorms_size(), 1))
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


/**
 * The entry's section, for a caller that must remember it rather than test it.
 * An entry is stored at its actual text length, so entry_size is what says the
 * fixed-offset fields are present at all.
 */
static int corpus_entry_section_copy(const corpus_entry_t* entry,
  size_t entry_size, char* title, int title_size)
{
  int result = 0;
  if (NULL != title && 0 < title_size) {
    title[0] = '\0';
    if (NULL != entry && entry_size >= CORPUS_ENTRY_META_SIZE) {
      result = entry->section_len;
      if (result >= title_size) result = title_size - 1;
      if (0 < result) {
        memcpy(title, entry->section, (size_t)result);
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


/**
 * Count, per token id, how many occurrences carried an initial upper-case letter
 * and how many there were. One corpus pass, and the only pass that has to see
 * the SURFACE form rather than the interned one.
 */
static void answer_case_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  answer_case_free();
  if (NULL == corpus || NULL == lexicon) return;
  answer_case_size = libxs_lexicon_size(lexicon) + 1;
  answer_case_upper = (unsigned int*)calloc(answer_case_size,
    sizeof(*answer_case_upper));
  answer_case_total = (unsigned int*)calloc(answer_case_size,
    sizeof(*answer_case_total));
  if (NULL == answer_case_upper || NULL == answer_case_total) {
    answer_case_free();
    return;
  }
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    if (SCALE_SENTENCE == entry->scale && 0 < entry->text_len
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
        (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
    {
      size_t pos;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lexeme = stream.data + pos;
        if (0 != (lexeme->flags & LIBXS_LEXEME_WORD) && 0 != lexeme->id
          && lexeme->id < answer_case_size)
        {
          ++answer_case_total[lexeme->id];
          if (0 != (lexeme->flags & LIBXS_LEXEME_ENTITY)) {
            ++answer_case_upper[lexeme->id];
          }
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    value = libxs_registry_next(corpus, &key, &cursor);
  }
}


static void answer_case_free(void)
{
  free(answer_case_upper);
  free(answer_case_total);
  answer_case_upper = NULL;
  answer_case_total = NULL;
  answer_case_size = 0;
}


/**
 * Does the corpus use this word as a NAME? True when every occurrence carried an
 * initial upper-case letter, which is a construction rule rather than a
 * threshold: a single lower-case occurrence makes the word a common one.
 *
 * The word arrives as the questioner typed it, so it is lower-cased before the
 * lookup -- libxs_lexicon_id matches the stored bytes, and the lexicon stores the
 * normalized (lower-case) form, so passing a capitalized word would simply
 * miss.
 */
static int answer_word_is_name(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word && 0 < word_len && word_len <= LIBXS_LEXEME_MAXBYTES
    && NULL != answer_case_total && NULL != answer_query_lexicon)
  {
    char lower[LIBXS_LEXEME_MAXBYTES + 1];
    unsigned int id;
    int pos;
    for (pos = 0; pos < word_len; ++pos) {
      lower[pos] = (char)tolower((unsigned char)word[pos]);
    }
    lower[word_len] = '\0';
    id = libxs_lexicon_id(answer_query_lexicon, lower, word_len, 0, 0);
    if (0 != id && id < answer_case_size && 0 != answer_case_total[id]) {
      result = (answer_case_upper[id] == answer_case_total[id]) ? 1 : 0;
    }
  }
  return result;
}


/**
 * The word a "who/what is X" question asks about, skipping any leading skip|
 * term. Capitalization is NOT reported: it used to be, and the three resolvers
 * routed on it -- identity required an upper-case initial, the relation paths
 * required a lower-case one. That made the user's typing decide which layer
 * answered, and it was wrong in both directions: a name typed lower-case was
 * refused an answer its capitalized spelling gets, while a capitalized role word
 * escaped the relation layer and fell through to raw evidence under a
 * misattributed citation. Whether a word NAMES something is a property of the
 * corpus, which the fact index already established when it built its names, so
 * the resolvers consult that instead.
 */
static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size)
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


/**
 * Is this token one of the WORDS of the actor phrase? Word-bounded, because a
 * substring test let a short actor match inside a longer, unrelated word: the
 * sentence was scanned for the letters rather than for the word, so the reply
 * asserted a relation to an actor that is not in it, and cited the source.
 */
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
    result = text_contains_word_ci(actor, actor_len, token_buf);
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
    for (rule_pos = 0; rule_pos < converse_rules_size(); ++rule_pos) {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
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


/**
 * Can this text answer a "who" question at all? Either it is written as a proper
 * noun HERE, or a rule puts it in the person class. Nothing else is a person,
 * and the relation layer serves QUERY_WHO only.
 *
 * Capitalization decides because this is CORPUS text, where a capital mid-
 * sentence is the author saying "name" -- the same signal the entity lexrule
 * reads. That is the opposite of the query-side defect fixed earlier, where
 * capitalization of the QUERY routed an answer: a reader's shift key is not
 * evidence about the world, and an author's is. The object always follows a
 * verb, so it is never capitalized merely by standing first in its sentence.
 */
static int answer_relation_answer_is_person(const char* text, int text_len)
{
  return (NULL != text && 0 < text_len
    && (0 != isupper((unsigned char)text[0])
      || 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON, text,
        text_len))) ? 1 : 0;
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
  int relation_len;
  int actor_len;
  if (NULL == query_text || NULL == entry || NULL == match
    || QUERY_WHO != query_type) return 0;
  memset(match, 0, sizeof(*match));
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation));
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  if (relation_len <= 0 || 0 != answer_word_is_name(relation, relation_len)) {
    return 0;
  }
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
    int actor_seen = text_contains_word_ci(entry->text, entry->text_len,
      actor);
    if (rel_pos < 0) alt_pos = answer_relation_rule_alias_pos(relation,
      entry->text, entry->text_len, &alt_len);
    verb_pos = (rel_pos >= 0) ? rel_pos : alt_pos;
    verb_len = (rel_pos >= 0) ? relation_len : alt_len;
    /**
     * The actor must be ATTESTED IN THIS SENTENCE. There used to be an escape
     * hatch here: if the actor was absent but the word "he" stood before the
     * verb, the actor counted as seen -- treating a pronoun as an anaphor for
     * whatever the reader happened to ask about. It left the actor
     * unconstrained, so the reply asserted a relation to an actor the corpus
     * never mentions AND attached a citation to it. Resolving the pronoun would
     * need anaphora; refusing to claim an actor the sentence never names needs
     * nothing, and is the truth about the sentence.
     */
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
      /**
       * Only a PERSON can answer a "who" question, so the direct object has to
       * be one. This used to be a blacklist of five pronouns, which is the
       * judge-it-afterwards pattern that has never held: a first-person pronoun
       * object and an adjective left behind after the real object was skipped
       * both became answers, and both were then aggregated into an otherwise
       * correct sentence. A pronoun object still falls through to the search
       * before the verb below; an object that is neither a person nor a pronoun
       * now yields no fact at all, which is the truth about it.
       */
      if (obj_end > obj_begin
        && 0 != answer_relation_answer_is_person(entry->text + obj_begin,
          obj_end - obj_begin))
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
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
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
    for (rule_pos = 0; rule_pos < converse_rules_size(); ++rule_pos) {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
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
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
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
  int query_section_len;
  const answer_identity_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_identity_facts_size) return EXIT_FAILURE;
  name_len = answer_query_be_word(query_text, query_len, name,
    (int)sizeof(name));
  if (name_len <= 0 || 0 == answer_word_is_name(name, name_len)) {
    return EXIT_FAILURE;
  }
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
    answer_fact_learned_set(best->role, best->role_len,
      answer_relation_rule_provenance(RELATION_RULE_PERSON, best->role,
        best->role_len));
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
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
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
                  && ';' != *end && '!' != *end && '?' != *end
                  && '\n' != *end && '\r' != *end) ++end;
                /**
                 * CUT OFF by a line break rather than closed by punctuation or by
                 * the end of the entry: the clause is incomplete ("A young fox,
                 * who said:" -- the speech was on the next line) and describes
                 * nothing, so it is not a description. Rejected the same way a
                 * clause-less role is, by never becoming a fact.
                 */
                if (end < text_end && ('\n' == *end || '\r' == *end)) {
                  end = clause;
                }
                else score = 2.0;
              }
            }
            /**
             * A described role needs a DESCRIPTION. Without the relative clause
             * `end` still points at the role itself, so the fact would be the
             * article plus the queried word and the reply would restate the
             * question: "Who is the wife?" -> "A wife." Such a fact cannot inform
             * any answer, so it is never stored, and the query abstains -- which
             * is the truth, since the corpus describes no wife.
             */
            if (end > clause
              && EXIT_SUCCESS == answer_describe_fact_append(token, token_len,
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
    (int)sizeof(role));
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
    answer_fact_learned_set(best->role, best->role_len,
      answer_relation_rule_provenance(RELATION_RULE_PERSON, best->role,
        best->role_len));
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


/**
 * Does the question ask about the relation this fact states? The relation is one
 * word on both sides, so this is EQUALITY, directly or through an alias rule.
 *
 * It used to be a substring test, which is the same defect the actor matching
 * carried: a longer word containing the relation borrowed the fact, so a
 * question about a different verb -- including one built by negating this verb
 * with a prefix -- was answered by it, asserted, and cited. A relation that
 * merely CONTAINS another relation is a different relation.
 */
static int answer_relation_fact_relation_match(const char* query_relation,
  const answer_relation_fact_t* fact)
{
  int result = 0;
  size_t rule_pos;
  if (NULL == query_relation || NULL == fact) return 0;
  if (0 != libxs_striequal(query_relation, strlen(query_relation),
    fact->relation, (size_t)fact->relation_len))
  {
    result = 1;
  }
  for (rule_pos = 0; rule_pos < converse_rules_size() && 0 == result;
    ++rule_pos)
  {
    const answer_relation_rule_t* rule = converse_rules() + rule_pos;
    if (RELATION_RULE_ALIAS == rule->kind
      && 0 != libxs_striequal(rule->relation, strlen(rule->relation),
        fact->relation, (size_t)fact->relation_len)
      && 0 != libxs_striequal(query_relation, strlen(query_relation),
        rule->term, strlen(rule->term)))
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
  const answer_relation_fact_t* answer_facts[RELATION_FACT_MAX];
  int answer_lens[RELATION_FACT_MAX];
  int answer_made[RELATION_FACT_MAX];
  double answer_scores[RELATION_FACT_MAX];
  int relation_len;
  int actor_len;
  int query_section_len;
  int count = 0;
  int slot;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_relation_facts_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation));
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (relation_len <= 0 || 0 != answer_word_is_name(relation, relation_len)) {
    return EXIT_FAILURE;
  }
  for (slot = 0; slot < RELATION_FACT_MAX; ++slot) {
    answers[slot][0] = '\0';
    answer_facts[slot] = NULL;
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
            answer_facts[slot] = fact;
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
          answer_facts[insert] = answer_facts[insert - 1];
          answer_lens[insert] = answer_lens[insert - 1];
          answer_made[insert] = answer_made[insert - 1];
          answer_scores[insert] = answer_scores[insert - 1];
          --insert;
        }
        memcpy(answers[insert], fact->answer,
          (size_t)fact->answer_len + 1);
        answer_facts[insert] = fact;
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
    /* The emitted answers ARE the class terms this reply rests on, so the rule
       layer is asked about them rather than a flag being cached per fact. */
    for (item = 0; item < count && 0 == answer_fact_learned_len; ++item) {
      answer_fact_learned_set(answers[item], answer_lens[item],
        answer_relation_rule_provenance(RELATION_RULE_PERSON, answers[item],
          answer_lens[item]));
    }
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
      /* Named once the reply exists, so a resolver that assembles nothing
         leaves no citation behind. Every fact that reached the reply names its
         source, and a reply resting on two tales is credited to both rather
         than to neither. */
      for (item = 0; item < count; ++item) {
        if (NULL != answer_facts[item]) {
          answer_fact_section_add(answer_facts[item]->section,
            answer_facts[item]->section_len);
        }
      }
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
  char answer_sections[RELATION_AGG_MAX][ENTRY_SECTION_MAX];
  int answer_section_lens[RELATION_AGG_MAX];
  int answer_lens[RELATION_AGG_MAX];
  double answer_scores[RELATION_AGG_MAX];
  int count = 0;
  char relation[64];
  char actor[64];
  int relation_len;
  int actor_len;
  if (NULL == corpus || NULL == query_text || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation));
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (query_section_len > 0 && relation_len > 0
    && 0 == answer_word_is_name(relation, relation_len)
    && actor_len > 0)
  {
    int slot;
    for (slot = 0; slot < RELATION_AGG_MAX; ++slot) {
      answers[slot][0] = '\0';
      answer_sections[slot][0] = '\0';
      answer_section_lens[slot] = 0;
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
              answer_section_lens[slot] = corpus_entry_section_copy(entry,
                entry_size, answer_sections[slot], ENTRY_SECTION_MAX);
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
            memcpy(answer_sections[insert], answer_sections[insert - 1],
              (size_t)answer_section_lens[insert - 1] + 1);
            answer_section_lens[insert] = answer_section_lens[insert - 1];
            answer_lens[insert] = answer_lens[insert - 1];
            answer_scores[insert] = answer_scores[insert - 1];
            --insert;
          }
          memcpy(answers[insert], candidate, (size_t)candidate_len + 1);
          answer_section_lens[insert] = corpus_entry_section_copy(entry,
            entry_size, answer_sections[insert], ENTRY_SECTION_MAX);
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
        /* The query named a section and every contributor had to match it, so
           these agree in the ordinary case; naming them all is still what makes
           the aggregate say where it came from. */
        for (item = 0; item < count; ++item) {
          answer_fact_section_add(answer_sections[item],
            answer_section_lens[item]);
        }
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
  int word_len;
  answer_relation_match_t relation_match;
  if (QUERY_WHO != query_type || NULL == entry) return 0.0;
  word_len = answer_query_be_word(query_text, query_len, word,
    (int)sizeof(word));
  if (word_len <= 0) return 0.0;
  if (0 == answer_word_is_name(word, word_len)
    && 0 != answer_relation_match_query(query_text,
    query_len, query_type, entry, &relation_match))
  {
    result = relation_match.score;
  }
  if (0 == text_contains_word_ci(entry->text, entry->text_len, word)) {
    return result;
  }
  if (0 != answer_word_is_name(word, word_len)) {
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
      rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 1))
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
            (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 1))
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


/**
 * Which slot already holds this text, or -1.
 *
 * Ingest keys an entry on its text AND its section, and stores the same span at
 * more than one scale, so one sentence can exist as several entries: a stock
 * phrase recurs across sources, and every sentence short enough to be a
 * paragraph of its own is stored twice. The answer list is what the reader sees,
 * so the same sentence must not occupy two of its slots -- which is what returned
 * one reply twice at `-n 2`. Rejected here rather than at print time, so the
 * reply path, the evaluation and the recomb host all see one answer list.
 */
/**
 * Has this exact text already been printed in this reply? Records it if not.
 *
 * An answer list free of repeated entries is still not a reply free of repeated
 * sentences: evidence extraction pulls the matching sentence out of a paragraph
 * as readily as out of the sentence stored beside it, so two different entries
 * render one sentence. The reader sees printed text, so printed text is what is
 * compared.
 */
static int answer_shown_repeat(char shown[][COMPOSE_MAXTEXT], int* nshown,
  const char* text, int text_len)
{
  int result = 0;
  int pos;
  for (pos = 0; pos < *nshown && 0 == result; ++pos) {
    if ((int)strlen(shown[pos]) == text_len
      && 0 == libxs_memcmp(shown[pos], text, (size_t)text_len))
    {
      result = 1;
    }
  }
  if (0 == result && *nshown < ANSWER_MAX && text_len < COMPOSE_MAXTEXT) {
    memcpy(shown[*nshown], text, (size_t)text_len);
    shown[*nshown][text_len] = '\0';
    ++(*nshown);
  }
  return result;
}


static int answer_slot_with_text(const corpus_entry_t* const entries[],
  int limit, const corpus_entry_t* entry)
{
  int result = -1;
  int slot;
  for (slot = 0; slot < limit && 0 > result; ++slot) {
    if (NULL != entries[slot] && entries[slot]->text_len == entry->text_len
      && 0 == libxs_memcmp(entries[slot]->text, entry->text,
        (size_t)entry->text_len))
    {
      result = slot;
    }
  }
  return result;
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
  int slot, held;
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
      converse_lexnorms(), converse_lexnorms_size(), 1))
  {
    query_type = query_type_of(&query, lexicon);
    query_be_len = answer_query_be_word(rank_query_text, rank_query_len,
      query_be_word, (int)sizeof(query_be_word));
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
        && 0 == answer_word_is_name(query_be_word, query_be_len)
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
        held = answer_slot_with_text(entries, limit, entry);
        if (0 <= held) {
          /* Keep the better-scoring occurrence: the score carries the scale
             preference, so this is what picks a sentence over the fragment cut
             out of it. */
          if (score > scores[held]) {
            entries[held] = entry;
            scores[held] = score;
            while (0 < held && scores[held] > scores[held - 1]) {
              const corpus_entry_t* swap_entry = entries[held - 1];
              const double swap_score = scores[held - 1];
              entries[held - 1] = entries[held];
              scores[held - 1] = scores[held];
              entries[held] = swap_entry;
              scores[held] = swap_score;
              --held;
            }
          }
        }
        else for (slot = 0; slot < limit; ++slot) {
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


/**
 * Drop the heading a stored entry opens with: ingest stores the first sentence of
 * a section from its heading onward, so the heading is part of that entry's text
 * and would otherwise open the reply.
 *
 * corpus_title_len decides what a heading is. The scan that used to stand here
 * had its own answer and got it wrong in both directions: a one-word title was
 * kept, while the initial capital of the sentence beneath it counted as the
 * title's second word and was eaten, so the reply began mid-word.
 */
static void answer_strip_heading_prefix(const char** text, int* text_len)
{
  if (NULL != text && NULL != *text && NULL != text_len && 0 < *text_len) {
    const int title_len = corpus_title_len(*text, *text_len);
    if (0 < title_len) {
      const char* next = *text + title_len;
      int remaining = *text_len - title_len;
      while (0 < remaining && 0 != isspace((unsigned char)*next)) {
        ++next;
        --remaining;
      }
      if (0 < remaining) {
        *text = next;
        *text_len = remaining;
      }
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
      rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 1))
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
      converse_lexnorms(), converse_lexnorms_size(), 1))
  {
    query_type = query_type_of(&query, lexicon);
    bridge = answer_bridge_match(&query, lexicon, entry);
    be_len = answer_query_be_word(query_text, query_len, be_word,
      (int)sizeof(be_word));
  }
  if (NULL != bridge && NULL != bridge->reply) {
    result = answer_bridge_expand_reply(bridge, text, text_len, lexicon,
      rules, nrules, output, output_size);
  }
  else if (QUERY_WHO == query_type && be_len > 0
    && 0 == answer_word_is_name(be_word, be_len)
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
    converse_lexnorms(), converse_lexnorms_size(), 1))
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
 * <role>") is not groundable from evidence that states only positives. The
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
  if (NULL != answer_query_lexicon && answer_query_nrules > 0
    && NULL != query_text && query_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(answer_query_lexicon,
      &query, (const unsigned char*)query_text, query_len,
      answer_query_rules, answer_query_nrules,
      converse_lexnorms(), converse_lexnorms_size(), 0))
  {
    for (rule_pos = 0; rule_pos < converse_rules_size() && 0 == result;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
      if (RELATION_RULE_NEGATE == rule->kind
        && 0 != lexeme_stream_has_text(&query, answer_query_lexicon,
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
  answer_fact_learned_set(NULL, 0, RELATION_RULE_ASSERTED);
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
/**
 * Say when a reply rests on a class term that was not asserted. This is the
 * whole point of admitting the learned bands: the reply is offered, and the
 * reader is told which word it hangs on, so one that should not be in the class
 * reads as a guess rather than as an assertion.
 */
static void answer_print_learned(void)
{
  if (0 < answer_fact_learned_len) {
    printf("%s: rests on the %s class term \"%s\"\n",
      (RELATION_RULE_PROPOSED == answer_fact_learned_from)
        ? "speculative" : "learned",
      (RELATION_RULE_PROPOSED == answer_fact_learned_from)
        ? "proposed" : "learned", answer_fact_learned);
  }
}


/**
 * Bytes of a section that form the citation: ONE line.
 *
 * A section captured before the heading map was built could carry the lines
 * that followed the heading, and printing it whole named two sources at once.
 * Stopping at the first line break recovers the intended title. The map makes
 * that unreachable for a freshly ingested corpus, but a persisted one predating
 * it still has such sections stored.
 *
 * Shared with the evaluation so a checked citation is the SAME text the reader is
 * shown, rather than the raw section a fixture would then have to describe.
 */
static int answer_citation_len(const char* section, int section_len)
{
  int result = 0;
  if (NULL != section) {
    while (result < section_len && '\n' != section[result]
      && '\r' != section[result] && '\0' != section[result]) ++result;
  }
  return result;
}


static void answer_print_citation(const char* section, int section_len)
{
  const int len = answer_citation_len(section, section_len);
  if (0 < len) printf("citation: %.*s\n", len, section);
}


static void answer_fact_learned_set(const char* term, int term_len,
  int provenance)
{
  answer_fact_learned_len = 0;
  answer_fact_learned[0] = '\0';
  answer_fact_learned_from = RELATION_RULE_ASSERTED;
  if (NULL != term && term_len > 0 && RELATION_RULE_ASSERTED != provenance
    && term_len < (int)sizeof(answer_fact_learned))
  {
    memcpy(answer_fact_learned, term, (size_t)term_len);
    answer_fact_learned[term_len] = '\0';
    answer_fact_learned_len = term_len;
    answer_fact_learned_from = provenance;
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


/**
 * Name one more source this reply rests on. Already-named and oversized sources
 * are dropped rather than truncated: half a title is a citation to nothing.
 */
static void answer_fact_section_add(const char* section, int section_len)
{
  static const char joiner[] = "; ";
  if (NULL != section && 0 < section_len
    && NULL == libxs_strimem(answer_fact_section,
      (size_t)answer_fact_section_len, section, (size_t)section_len))
  {
    const int join_len = (0 < answer_fact_section_len)
      ? (int)(sizeof(joiner) - 1) : 0;
    if (answer_fact_section_len + join_len + section_len
      < (int)sizeof(answer_fact_section))
    {
      memcpy(answer_fact_section + answer_fact_section_len, joiner,
        (size_t)join_len);
      answer_fact_section_len += join_len;
      memcpy(answer_fact_section + answer_fact_section_len, section,
        (size_t)section_len);
      answer_fact_section_len += section_len;
      answer_fact_section[answer_fact_section_len] = '\0';
    }
  }
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
  if (0 != converse_judge_active() && 1 < answer_count) {
    const char* candidates[ANSWER_MAX];
    int lengths[ANSWER_MAX];
    double bits[ANSWER_MAX];
    int slot, nvalid = 0;
    for (slot = 0; slot < answer_count; ++slot) {
      candidates[slot] = (NULL != entries[slot]) ? entries[slot]->text : NULL;
      lengths[slot] = (NULL != entries[slot]) ? entries[slot]->text_len : 0;
    }
    if (EXIT_SUCCESS == converse_judge_rescore(query_text, (int)query_len,
      candidates, lengths, answer_count, bits))
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
        if (1 < converse_judge_verbose()) {
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


static size_t answer_visible_append(char* output, size_t output_size,
  size_t output_pos, const char* text, int text_len)
{
  size_t result = output_pos;
  if (NULL != output && 0 < text_len
    && output_pos + (size_t)text_len + 2 < output_size)
  {
    if (0 < output_pos) output[result++] = '\n';
    memcpy(output + result, text, (size_t)text_len);
    result += (size_t)text_len;
    output[result] = '\0';
  }
  return result;
}


/**
 * Render an answer list the way a reader sees it; print it unless asked not to.
 *
 * ONE renderer, because the evaluation used to check `answer_reply` -- which
 * FAILS for most retrieved answers, those being shown as evidence instead -- so
 * everything the reply path does after that was ungated, and a reply that began
 * mid-word passed. What the reader is shown and what the fixture scores are now
 * the same bytes by construction.
 *
 * Returns 2 when a composed reply answered, 1 when evidence was shown, 0 when
 * nothing could be rendered; the caller's out-parameter carries only a composed
 * reply, which is what it has always carried.
 */
static int answer_render(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[], int answer_count,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int print, char* output, size_t output_size)
{
  int result = 0;
  int slot;
  size_t pos = 0;
  char reply[COMPOSE_MAXTEXT];
  char shown[ANSWER_MAX][COMPOSE_MAXTEXT];
  int nshown = 0;
  if (NULL != output && 0 < output_size) output[0] = '\0';
  if (0 < answer_count && NULL != entries[0]
    && EXIT_SUCCESS == answer_reply(query_text, query_len, entries[0],
      lexicon, rules, nrules, reply, sizeof(reply)))
  {
    pos = answer_visible_append(output, output_size, pos, reply,
      (int)strlen(reply));
    if (0 != print) {
      printf("%s\n", reply);
      answer_print_citation(entries[0]->section, entries[0]->section_len);
    }
    result = 2;
  }
  for (slot = 0; 0 == result && slot < answer_count
    && NULL != entries[slot]; ++slot)
  {
    const char* text = entries[slot]->text;
    int text_len = entries[slot]->text_len;
    if (EXIT_SUCCESS == answer_evidence_sentence(query_text, query_len,
      entries[slot], lexicon, rules, nrules, reply, sizeof(reply)))
    {
      if (0 == answer_shown_repeat(shown, &nshown, reply,
        (int)strlen(reply)))
      {
        pos = answer_visible_append(output, output_size, pos, reply,
          (int)strlen(reply));
        if (0 != print) {
          if (nshown > 1) printf("\n");
          printf("%s\n", reply);
          answer_print_citation(entries[slot]->section,
            entries[slot]->section_len);
        }
      }
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
          && 0 != text_ends_sentence(text, text_len)))
      && 0 == answer_shown_repeat(shown, &nshown, text, text_len))
    {
      pos = answer_visible_append(output, output_size, pos, text, text_len);
      if (0 != print) {
        if (nshown > 1) printf("\n");
        printf("%.*s\n", text_len, text);
      }
    }
  }
  if (0 == result && 0 < nshown) result = 1;
  return result;
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
  int rendered;
  char reply[COMPOSE_MAXTEXT];
  char visible[4 * COMPOSE_MAXTEXT];
  if (NULL != out_reply && out_size > 0) out_reply[0] = '\0';
  if (EXIT_SUCCESS == answer_fact_reply(corpus, query_text, query_len,
    reply, sizeof(reply)))
  {
    printf("%s\n", reply);
    answer_print_citation(answer_fact_section, answer_fact_section_len);
    answer_print_learned();
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
  rendered = answer_render(query_text, query_len, entries, answer_count,
    lexicon, rules, nrules, 1, visible, sizeof(visible));
  if (2 == rendered && NULL != out_reply && out_size > 0) {
    size_t rn = strlen(visible);
    if (rn >= out_size) rn = out_size - 1;
    memcpy(out_reply, visible, rn);
    out_reply[rn] = '\0';
  }
  LIBXS_UNUSED(scores);
  return (answer_count > 0) ? 1 : 0;
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


/**
 * question|evidence-terms|reply-terms|fact-terms|citation-terms
 *
 * The last two are optional, so a three- or four-field line behaves exactly as
 * before. The citation field is what makes ATTRIBUTION testable: the fixture
 * states which source the answer must be credited to, and a reply that is right
 * about the world but wrong about where it came from now fails.
 */
static int eval_parse_line(char* line, char* fields[5])
{
  int result = EXIT_FAILURE;
  char* cursor;
  int field_pos;
  if (NULL != fields) {
    for (field_pos = 0; field_pos < 5; ++field_pos) fields[field_pos] = NULL;
  }
  if (NULL != line && NULL != fields) {
    cursor = eval_trim(line);
    if ('\0' != *cursor && '#' != *cursor) {
      /* The trailing segment belongs to the field it reached, so a line with
         three, four or five fields fills exactly those and leaves the rest
         unset. */
      for (field_pos = 0; field_pos < 4 && NULL != cursor; ++field_pos) {
        char* sep = strchr(cursor, '|');
        if (NULL != sep) {
          *sep = '\0';
          fields[field_pos] = eval_trim(cursor);
          cursor = sep + 1;
        }
        else {
          fields[field_pos] = eval_trim(cursor);
          cursor = NULL;
        }
      }
      if (NULL != cursor) fields[4] = eval_trim(cursor);
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
  int npass = 0, ntop = 0, nany = 0, nreply = 0, nfact = 0, ncite = 0;
  int ncases = 0;
  int have_facts = (0 != answer_relation_facts_size
    || 0 != answer_docdef_facts_size) ? 1 : 0;
  FILE* file;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL == corpus || NULL == lexicon || NULL == rules) return EXIT_FAILURE;
  file = fopen(converse_eval_path(), "r");
  if (NULL == file) {
    fprintf(stderr, "eval: no %s file found\n", converse_eval_path());
  }
  while (NULL != file) {
    char line[EVAL_LINE_MAX];
    char* fields[5];
    const corpus_entry_t* entries[ANSWER_MAX];
    double scores[ANSWER_MAX];
    int nanswers;
    int top_pass;
    int any_pass;
    int reply_pass;
    int fact_pass;
    int fact_checked;
    int have_fact;
    int cite_pass;
    int cite_len;
    const char* cite = NULL;
    int pass;
    char reply[COMPOSE_MAXTEXT];
    char visible[4 * COMPOSE_MAXTEXT];
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
    cite_len = 0;
    /* Unconditionally, because an interactive query tries the fact resolvers
       FIRST: what the reader is shown is this reply whenever it succeeds, and
       whether the fixture states fact-terms has nothing to do with it. */
    have_fact = (EXIT_SUCCESS == answer_fact_reply(corpus, qtext, qlen,
      reply, sizeof(reply))) ? 1 : 0;
    if (0 != fact_checked) {
      if (0 != have_fact) {
        fact_pass = eval_terms_match_text(reply, (int)strlen(reply),
          fields[3]);
        /* Whichever resolver answered published its section, so the citation
           under test is the one the reader would have been shown. */
        cite_len = answer_citation_len(answer_fact_section,
          answer_fact_section_len);
        cite = answer_fact_section;
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
    /* Before the fact-only and abstention branches, both of which return early:
       a fact reply is exactly the kind of answer whose attribution matters. */
    if (0 == cite_len && 0 < nanswers && NULL != entries[0]) {
      cite = entries[0]->section;
      cite_len = answer_citation_len(entries[0]->section,
        entries[0]->section_len);
    }
    cite_pass = 1;
    if (NULL != fields[4] && 0 == eval_terms_empty(fields[4])) {
      cite_pass = (0 < cite_len)
        ? eval_terms_match_text(cite, cite_len, fields[4]) : 0;
      fprintf(stdout, "%s cite %s\n", (0 != cite_pass) ? "PASS" : "FAIL",
        fields[0]);
      if (0 != cite_pass) ++ncite;
      else if (0 < cite_len) {
        fprintf(stdout, "     cited \"%.*s\", expected \"%s\"\n", cite_len,
          cite, fields[4]);
      }
      else fprintf(stdout, "     cited nothing, expected \"%s\"\n", fields[4]);
    }
    if (0 != eval_terms_empty(fields[1])) {
      if (0 != fact_checked) {
        pass = (0 != fact_pass && 0 != cite_pass) ? 1 : 0;
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
        && 0 == converse_rules_size())
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
    /**
     * The reply expectation is scored against THE TEXT THE READER IS SHOWN, not
     * against answer_reply -- which fails for most retrieved answers, those
     * being shown as evidence instead, so the old check could only ever be
     * written for the few queries that compose a reply. answer_render is the
     * same renderer the interactive path prints with, called with printing off.
     */
    if (0 != have_fact) {
      size_t rn = strlen(reply);
      if (rn >= sizeof(visible)) rn = sizeof(visible) - 1;
      memcpy(visible, reply, rn);
      visible[rn] = '\0';
    }
    else {
      answer_render(qtext, qlen, entries, nanswers, lexicon, rules, nrules, 0,
        visible, sizeof(visible));
    }
    if (0 == eval_terms_empty(fields[2])) {
      reply_pass = eval_terms_match_text(visible, (int)strlen(visible),
        fields[2]);
    }
    pass = (0 != any_pass && 0 != reply_pass && 0 != fact_pass
      && 0 != cite_pass) ? 1 : 0;
    fprintf(stdout, "%s top %s\n", (0 != top_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s any %s\n", (0 != any_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s reply %s\n", (0 != reply_pass) ? "PASS" : "FAIL",
      fields[0]);
    /* Say what was replied, the way the citation check says what was cited: a
       reply expectation is otherwise unwritable without rebuilding to look. */
    if (0 == reply_pass && 0 == eval_terms_empty(fields[2])) {
      fprintf(stdout, "     replied \"%s\", expected \"%s\"\n", visible,
        fields[2]);
    }
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
    "eval[%s]: %d/%d passed (top=%d, any=%d, reply=%d, fact=%d, cite=%d)\n",
    profile->name, npass, ncases, ntop, nany, nreply, nfact, ncite);
  if (0 != converse_judge_active()) {
    fprintf(stderr, "  hier rescore: %ld rankings, top-1 changed on %ld\n",
      answer_hier_nreorder, answer_hier_nchanged);
  }
  if (NULL != file) fclose(file);
  if (ncases > 0 && npass == ncases) result = EXIT_SUCCESS;
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
      (const unsigned char*)a, (size_t)alen, rules, nrules, converse_lexnorms(),
      converse_lexnorms_size(), 0)
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &sb,
      (const unsigned char*)b, (size_t)blen, rules, nrules, converse_lexnorms(),
      converse_lexnorms_size(), 0))
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
    int ntok = ngram_generate(converse_ngram_handle()->store, lexicon, rules, nrules,
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
    ngram_complete(converse_ngram_handle()->store, lexicon, rules, nrules, 0,
      text, text_len);
  }
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
      int ntok = ngram_generate(converse_ngram_handle()->store, lexicon, rules, nrules,
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


/**
 * Recombination over the corpus. The byte model and the word probability are
 * DIAGNOSTICS here, not prerequisites: both only produce seam scores, and a seam
 * score has never separated a true join from a fluent false one -- the last
 * attempt ranked the false continuation 50x above the true one -- so
 * grammaticality is enforced by the clause constraint instead. Running without
 * them drops the bpc and penalty columns and nothing else, which is also what
 * lets this half link without the byte model at all.
 */
static void converse_qa_recomb(converse_run_t* run,
  libxs_registry_t* ngram_model)
{
  const char* env = getenv("CONVERSE_RECOMB");
  const int limit = (NULL != env && '\0' != *env && 0 < atoi(env))
    ? atoi(env) : 50;
  converse_recomb_host_t host;
  host.ends_sentence = text_ends_sentence;
  host.is_wordchar = libxs_lexeme_is_word_char;
  host.entry_build = corpus_entry_build;
  host.word_prob = (NULL != ngram_model) ? recomb_word_prob : NULL;
  host.seam_bits = (0 != converse_judge_active())
    ? converse_judge_seam_bits : NULL;
  host.maxorder = ngram_maxorder();
  if (0 == converse_judge_active()) {
    fprintf(stderr, "recomb: no seam judge"
      " (CONVERSE_HIER_RESCORE=1 adds the bpc columns)\n");
  }
  converse_recomb_probe_run(run->corpus, run->lexicon, run->rules, run->nrules,
    limit, &host);
}


/**
 * Read prompts and print grounded continuations. Questions answer from the
 * corpus first (with a continuation on top); everything else replays the
 * n-gram. A prediction kind under -c is served by the other half, so this path
 * only ever holds the grounded generator.
 */
static void converse_qa_complete(converse_run_t* run,
  libxs_registry_t* ngram_model)
{
  converse_recomb_host_t compose;
  char line[4096];
  int compose_ready = 0;
  if (0 != recomb_compose_on()) {
    compose.ends_sentence = text_ends_sentence;
    compose.is_wordchar = libxs_lexeme_is_word_char;
    compose.entry_build = corpus_entry_build;
    compose.word_prob = (NULL != ngram_model) ? recomb_word_prob : NULL;
    compose.seam_bits = (0 != converse_judge_active())
      ? converse_judge_seam_bits : NULL;
    compose.maxorder = ngram_maxorder();
    if (0 == converse_judge_active()) {
      fprintf(stderr, "compose: no seam judge"
        " (CONVERSE_HIER_RESCORE=1 adds the bpc columns)\n");
    }
    /* One corpus pass, kept for the session: per-query would dominate. */
    if (EXIT_SUCCESS == converse_recomb_open(run->corpus, run->lexicon,
      &compose))
    {
      compose_ready = 1;
    }
    else fprintf(stderr, "compose: pivot index could not be built\n");
  }
  printf("> ");
  fflush(stdout);
  while (NULL != fgets(line, (int)sizeof(line), stdin)) {
    size_t len = strlen(line);
    while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
    if (0 < len) {
      if (0 != is_question_query(line, len, run->lexicon, run->rules,
        run->nrules))
      {
        complete_respond(run->corpus, run->lexicon, run->rules, run->nrules,
          run->answer_model, run->profile, run->budget, line, (int)len);
      }
      else ngram_complete(ngram_model, run->lexicon, run->rules, run->nrules,
        run->ngram_order, line, (int)len);
    }
    printf("> ");
    fflush(stdout);
  }
  if (0 != compose_ready) converse_recomb_close();
}


/** Interactive answering: the corpus is searched, never sampled from. */
static int converse_qa_answer(converse_run_t* run)
{
  libxs_spatial_t spatial;
  int result = corpus_spatial_build(&spatial, run->corpus);
  if (EXIT_SUCCESS == result) {
    libxs_registry_t* ngram_model = ngram_build(run->corpus, run->lexicon,
      run->rules, run->nrules, 0);
    char line[4096];
    ngram_backoff_build(ngram_model, run->lexicon);
    conv_reset();
    printf("> ");
    fflush(stdout);
    while (NULL != fgets(line, (int)sizeof(line), stdin)) {
      size_t len = strlen(line);
      libxs_fprint_t query;
      size_t shape;
      while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
      if (0 < len) {
        shape = len;
        libxs_fprint(&query, LIBXS_DATATYPE_U8, line, 1,
          &shape, NULL, FPRINT_ORDER, 0, 0, 0);
        respond(&spatial, run->corpus, line, len, &query, run->budget,
          run->lexicon, run->rules, run->nrules, run->answer_model,
          run->profile);
      }
      printf("> ");
      fflush(stdout);
    }
    libxs_spatial_destroy(&spatial);
  }
  return result;
}


int converse_qa_run(converse_run_t* run)
{
  int result = EXIT_SUCCESS;
  answer_query_lexicon = run->lexicon;
  answer_query_rules = run->rules;
  answer_query_nrules = run->nrules;
  answer_bridge_load_file(converse_bridge_path());
  answer_bridge_report(stderr);
  /* Before the facts and before any query: the resolvers ask it which words the
     corpus uses as names, which is what decides who answers. */
  answer_case_build(run->corpus, run->lexicon, run->rules, run->nrules);
  converse_stage_end("f_case");
  answer_relation_facts_build(run->corpus);
  answer_relation_facts_report(stderr);
  converse_stage_end("f_relation");
  answer_identity_facts_build(run->corpus);
  answer_identity_facts_report(stderr);
  converse_stage_end("f_identity");
  answer_describe_facts_build(run->corpus);
  answer_describe_facts_report(stderr);
  converse_stage_end("f_describe");
  answer_docdef_facts_build(run->corpus);
  answer_docdef_facts_report(stderr);
  converse_stage_end("f_docdef");
  converse_judge_open(run->corpus);
  if (0 != run->eval_mode) {
    result = eval_converse(run->corpus, run->lexicon, run->rules, run->nrules,
      run->answer_model, run->profile);
  }
  else if (0 != run->predict_eval_mode || 0 != run->complete_mode) {
    libxs_registry_t* ngram_model;
    converse_bpe_prepare(run->corpus, run->ngram_holdout);
    converse_stage_end("test_ingest");
    ngram_model = ngram_build(run->corpus, run->lexicon, run->rules,
      run->nrules, run->ngram_holdout);
    ngram_backoff_build(ngram_model, run->lexicon);
    converse_stage_end("ngram_build");
    converse_stage_begin();
    if (0 != run->predict_eval_mode) {
      converse_qa_recomb(run, ngram_model);
      ngram_stats(ngram_model);
      converse_stage_end("eval");
      converse_stage_report();
    }
    else converse_qa_complete(run, ngram_model);
  }
  else result = converse_qa_answer(run);
  converse_judge_close();
  answer_case_free();
  answer_bridge_free_loaded();
  answer_relation_facts_free();
  answer_identity_facts_free();
  answer_describe_facts_free();
  answer_docdef_facts_free();
  return result;
}
