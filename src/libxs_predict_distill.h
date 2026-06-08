LIBXS_API_INLINE void internal_libxs_predict_distill(
  libxs_predict_t* model, int nclusters, int order)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  const int keep_denom = model->distill_keep;
  const int nkeep = (keep_denom > 1) ? (p / keep_denom) : p;
  const int train_count = LIBXS_MAX((int)(p * 0.8 + 0.5), 2);
  const int fold_size = p - train_count;
  const int nfolds = (p + fold_size - 1) / fold_size;
  double* predicted = (double*)malloc((size_t)p * (size_t)n * sizeof(double));
  if (NULL != predicted && fold_size > 0 && nkeep > 0) {
    int fold, i, ok = 1;
    for (fold = 0; fold < nfolds && 0 != ok; ++fold) {
      const int fold_begin = fold * fold_size;
      const int fold_end = LIBXS_MIN(fold_begin + fold_size, p);
      libxs_predict_t* fm = libxs_predict_create(m, n);
      if (NULL != fm) {
        fm->eval_mode = model->eval_mode;
        fm->decompose = model->decompose;
        fm->diff_mode = -1;
        for (i = 0; i < p; ++i) {
          if (i >= fold_begin && i < fold_end) continue;
          libxs_predict_push(NULL, fm,
            model->entries[i].inputs, model->entries[i].outputs);
        }
        if (EXIT_SUCCESS == libxs_predict_build(fm, nclusters, order)) {
          for (i = fold_begin; i < fold_end; ++i) {
            libxs_predict_eval(NULL, fm, model->entries[i].inputs,
              predicted + (size_t)i * n, NULL, 1);
          }
        }
        else {
          ok = 0;
        }
        libxs_predict_destroy(fm);
      }
      else {
        ok = 0;
      }
    }
    if (0 != ok) {
      const size_t nc = (size_t)p;
      const size_t coprime = libxs_coprime2(nc);
      for (i = 0; i < p; ++i) {
        memcpy(model->entries[i].outputs, predicted + (size_t)i * n,
          (size_t)n * sizeof(double));
      }
      if (nkeep < p) {
        internal_libxs_predict_entry_t* kept =
          (internal_libxs_predict_entry_t*)malloc(
            (size_t)nkeep * sizeof(*kept));
        if (NULL != kept) {
          for (i = 0; i < nkeep; ++i) {
            const int idx = (int)LIBXS_SHUFFLE_INDEX(
              (size_t)i, nc, coprime, 0);
            kept[i] = model->entries[idx];
          }
          for (i = 0; i < p; ++i) {
            int j, found = 0;
            for (j = 0; j < nkeep && 0 == found; ++j) {
              if (kept[j].inputs == model->entries[i].inputs) found = 1;
            }
            if (0 == found) {
              free(model->entries[i].inputs);
              free(model->entries[i].outputs);
            }
          }
          memcpy(model->entries, kept,
            (size_t)nkeep * sizeof(*kept));
          model->nentries = nkeep;
          free(kept);
        }
      }
    }
  }
  free(predicted);
  model->distill_keep = 0;
}
