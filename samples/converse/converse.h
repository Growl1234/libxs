#ifndef CONVERSE_H
#define CONVERSE_H

#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>

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

#endif /*CONVERSE_H*/