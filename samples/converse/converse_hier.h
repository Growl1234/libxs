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

#endif /*CONVERSE_HIER_H*/