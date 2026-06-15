#if !defined(LIBXS_PREDICT_HKNN_MINLEAF)
#  define LIBXS_PREDICT_HKNN_MINLEAF 0
#endif


LIBXS_API_INLINE void internal_libxs_predict_hknn_partition(
  libxs_predict_t* model, int* nclusters_out)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  const int target_nc = (int)(sqrt((double)p) + 0.5);
  const int min_leaf = (0 < LIBXS_PREDICT_HKNN_MINLEAF)
    ? LIBXS_PREDICT_HKNN_MINLEAF
    : LIBXS_MAX(p * 2 / (target_nc * 3), 3);
  int* const out_assign = model->hknn_assignments;
  int pairs_pool = 0, order_pool = 0;
  internal_libxs_predict_rf_pair_t* pairs =
    (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(internal_libxs_predict_rf_pair_t), pairs_pool);
  int* order = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(int), order_pool);
  int nleaves = 0;
  if (NULL != pairs && NULL != order) {
    int stack_begin[64], stack_count[64];
    double stack_imbal[64];
    int sp, i;
    for (i = 0; i < p; ++i) order[i] = i;
    stack_begin[0] = 0;
    stack_count[0] = p;
    stack_imbal[0] = 1.0;
    sp = 1;
    while (sp > 0) {
      const int begin = stack_begin[--sp];
      const int nc = stack_count[sp];
      const double imbal = stack_imbal[sp];
      int best_feat = -1, best_pos = -1, j;
      double best_score = -1;
      if (nc <= min_leaf || nleaves >= target_nc) {
        for (i = begin; i < begin + nc; ++i) {
          out_assign[order[i]] = nleaves;
        }
        ++nleaves;
      }
      else {
        const int ideal_half = nc / 2;
        const double allowed_dev = 0.22 / LIBXS_MAX(imbal, 0.5);
        const int band_lo = LIBXS_MAX(
          (int)(ideal_half - nc * allowed_dev), min_leaf);
        const int band_hi = LIBXS_MIN(
          (int)(ideal_half + nc * allowed_dev), nc - min_leaf);
        for (j = 0; j < m; ++j) {
          double sum_all[128], sum2_all[128];
          double sum_left[128], sum2_left[128];
          int oi;
          for (i = 0; i < nc; ++i) {
            pairs[i].val = model->entries[order[begin + i]].inputs[j];
            pairs[i].idx = order[begin + i];
          }
          libxs_sort(pairs, nc, sizeof(pairs[0]),
            internal_libxs_predict_rf_pair_cmp, NULL);
          memset(sum_all, 0, (size_t)n * sizeof(double));
          memset(sum2_all, 0, (size_t)n * sizeof(double));
          for (i = 0; i < nc; ++i) {
            for (oi = 0; oi < n; ++oi) {
              const double v = model->entries[pairs[i].idx].outputs[oi];
              sum_all[oi] += v;
              sum2_all[oi] += v * v;
            }
          }
          memset(sum_left, 0, (size_t)n * sizeof(double));
          memset(sum2_left, 0, (size_t)n * sizeof(double));
          for (i = 0; i < nc - 1; ++i) {
            const int nleft = i + 1, nright = nc - nleft;
            for (oi = 0; oi < n; ++oi) {
              const double v = model->entries[pairs[i].idx].outputs[oi];
              sum_left[oi] += v;
              sum2_left[oi] += v * v;
            }
            if (pairs[i].val != pairs[i + 1].val
              && nleft >= band_lo && nleft <= band_hi)
            {
              double fisher = 0;
              const double penalty =
                1.0 + 4.0 * LIBXS_FABS((double)nleft / nc - 0.5);
              for (oi = 0; oi < n; ++oi) {
                const double ml = sum_left[oi] / nleft;
                const double mr = (sum_all[oi] - sum_left[oi]) / nright;
                const double vl = sum2_left[oi] / nleft - ml * ml;
                const double vr = (sum2_all[oi] - sum2_left[oi]) / nright
                  - mr * mr;
                const double within = vl * nleft + vr * nright;
                const double between = (double)nleft * nright
                  * (ml - mr) * (ml - mr) / nc;
                if (within > 0) fisher += between / within;
              }
              { const double score = fisher / penalty;
                if (score > best_score) {
                  best_score = score;
                  best_feat = j;
                  best_pos = i;
                }
              }
            }
          }
          if (best_feat == j) {
            for (i = 0; i < nc; ++i) order[begin + i] = pairs[i].idx;
          }
        }
        if (best_feat < 0) {
          for (i = begin; i < begin + nc; ++i) {
            out_assign[order[i]] = nleaves;
          }
          ++nleaves;
        }
        else {
          if (best_feat != m - 1) {
            for (i = 0; i < nc; ++i) {
              pairs[i].val = model->entries[order[begin + i]].inputs[best_feat];
              pairs[i].idx = order[begin + i];
            }
            libxs_sort(pairs, nc, sizeof(pairs[0]),
              internal_libxs_predict_rf_pair_cmp, NULL);
            for (i = 0; i < nc; ++i) order[begin + i] = pairs[i].idx;
          }
          { const int nleft = best_pos + 1;
            const int nright = nc - nleft;
            const double imbal_left = imbal * (double)nc / (2.0 * nleft);
            const double imbal_right = imbal * (double)nc / (2.0 * nright);
            if (sp < 64) {
              stack_begin[sp] = begin + nleft;
              stack_count[sp] = nright;
              stack_imbal[sp] = imbal_right;
              ++sp;
            }
            if (sp < 64) {
              stack_begin[sp] = begin;
              stack_count[sp] = nleft;
              stack_imbal[sp] = imbal_left;
              ++sp;
            }
          }
        }
      }
    }
  }
  *nclusters_out = LIBXS_MAX(nleaves, 1);
  LIBXS_PREDICT_FREE(order, order_pool);
  LIBXS_PREDICT_FREE(pairs, pairs_pool);
}


LIBXS_API_INLINE void internal_libxs_predict_hknn_refine(
  libxs_predict_t* model, int nclusters)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int max_iter = LIBXS_MIN(LIBXS_PREDICT_MAXITER, 10);
  int pts_pool = 0, comp_pool = 0, cnt_pool = 0;
  double* pts = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * (size_t)m * sizeof(double), pts_pool);
  double* comp = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)nclusters * (size_t)m * sizeof(double), comp_pool);
  int* counts = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)nclusters * sizeof(int), cnt_pool);
  if (NULL != pts && NULL != comp && NULL != counts) {
    int i, c, j, iter;
    for (i = 0; i < p; ++i) {
      internal_libxs_predict_normalize(model,
        model->entries[i].inputs, pts + (size_t)i * m);
    }
    for (iter = 0; iter < max_iter; ++iter) {
      int changed = 0;
      for (i = 0; i < p; ++i) {
        double best = libxs_dist2(
          pts + (size_t)i * m, model->clusters[0].centroid, m);
        int bestc = 0;
        for (c = 1; c < nclusters; ++c) {
          const double d = libxs_dist2(
            pts + (size_t)i * m, model->clusters[c].centroid, m);
          if (d < best) { best = d; bestc = c; }
        }
        if (model->assignments[i] != bestc) {
          model->assignments[i] = bestc;
          changed = 1;
        }
      }
      if (0 == changed) iter = max_iter;
      else {
        memset(comp, 0, (size_t)nclusters * (size_t)m * sizeof(double));
        memset(counts, 0, (size_t)nclusters * sizeof(int));
        for (c = 0; c < nclusters; ++c) {
          memset(model->clusters[c].centroid, 0, (size_t)m * sizeof(double));
        }
        for (i = 0; i < p; ++i) {
          const int ci = model->assignments[i];
          double* cen = model->clusters[ci].centroid;
          double* cmp = comp + (size_t)ci * m;
          for (j = 0; j < m; ++j) {
            libxs_kahan_sum(pts[(size_t)i * m + j], &cen[j], &cmp[j]);
          }
          ++counts[ci];
        }
        for (c = 0; c < nclusters; ++c) {
          if (counts[c] > 0) {
            for (j = 0; j < m; ++j) {
              model->clusters[c].centroid[j] /= counts[c];
            }
          }
        }
      }
    }
  }
  LIBXS_PREDICT_FREE(counts, cnt_pool);
  LIBXS_PREDICT_FREE(comp, comp_pool);
  LIBXS_PREDICT_FREE(pts, pts_pool);
}


LIBXS_API_INLINE void internal_libxs_predict_hknn_centroids(
  libxs_predict_t* model, int nclusters)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  int counts_pool = 0, norm_pool = 0, i, c, j;
  int* counts = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)nclusters * sizeof(int), counts_pool);
  double* norm = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)m * sizeof(double), norm_pool);
  if (NULL != counts && NULL != norm) {
    memset(counts, 0, (size_t)nclusters * sizeof(int));
    for (c = 0; c < nclusters; ++c) {
      memset(model->clusters[c].centroid, 0, (size_t)m * sizeof(double));
    }
    for (i = 0; i < p; ++i) {
      const int ci = model->assignments[i];
      internal_libxs_predict_normalize(model,
        model->entries[i].inputs, norm);
      for (j = 0; j < m; ++j) {
        model->clusters[ci].centroid[j] += norm[j];
      }
      ++counts[ci];
    }
    for (c = 0; c < nclusters; ++c) {
      if (counts[c] > 0) {
        for (j = 0; j < m; ++j) {
          model->clusters[c].centroid[j] /= counts[c];
        }
      }
    }
  }
  LIBXS_PREDICT_FREE(norm, norm_pool);
  LIBXS_PREDICT_FREE(counts, counts_pool);
}
