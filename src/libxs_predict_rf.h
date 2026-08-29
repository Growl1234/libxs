LIBXS_API_INLINE int internal_libxs_predict_rf_pair_cmp(
  const void* a, const void* b, void* ctx)
{
  const double va = ((const internal_libxs_predict_rf_pair_t*)a)->val;
  const double vb = ((const internal_libxs_predict_rf_pair_t*)b)->val;
  LIBXS_UNUSED(ctx);
  return (va > vb) - (va < vb);
}


LIBXS_API_INLINE int internal_libxs_predict_rf_split(
  const internal_libxs_predict_entry_t* entries,
  const int* subset, int nsub, int nfeat, int nfeatsub,
  internal_libxs_predict_rf_node_t* node, size_t seed,
  int output_idx, int label_off)
{
  int result = 0;
  double best_gini = 2.0;
  int trial;
  int pairs_pool = 0;
  const size_t feat_coprime = libxs_coprime2((size_t)nfeat);
  internal_libxs_predict_rf_pair_t* pairs =
    (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nsub * sizeof(internal_libxs_predict_rf_pair_t), pairs_pool);
  node->feature = -1;
  node->label = -1;
  if (NULL != pairs) {
  for (trial = 0; trial < nfeatsub; ++trial) {
    const int f = (int)(LIBXS_SHUFFLE_INDEX(
      (size_t)trial, (size_t)nfeat, feat_coprime, seed) % (size_t)nfeat);
    int left_counts[128], right_counts[128];
    int nleft, nright, i, k;
    for (i = 0; i < nsub; ++i) {
      pairs[i].val = entries[subset[i]].inputs[f];
      pairs[i].idx = subset[i];
    }
    libxs_sort(pairs, nsub, sizeof(pairs[0]),
      internal_libxs_predict_rf_pair_cmp, NULL);
    memset(right_counts, 0, sizeof(right_counts));
    nright = nsub; nleft = 0;
    for (i = 0; i < nsub; ++i) {
      ++right_counts[(LIBXS_ROUNDX(int, entries[pairs[i].idx].outputs[output_idx]) + label_off) & 127];
    }
    memset(left_counts, 0, sizeof(left_counts));
    for (i = 0; i < nsub - 1; ++i) {
      const int label = (LIBXS_ROUNDX(int, entries[pairs[i].idx].outputs[output_idx]) + label_off) & 127;
      ++left_counts[label]; ++nleft;
      --right_counts[label]; --nright;
      if (pairs[i].val == pairs[i + 1].val) continue;
      { double gini_l = 1.0, gini_r = 1.0, gini;
        for (k = 0; k < 128; ++k) {
          if (left_counts[k] > 0) {
            double p = (double)left_counts[k] / nleft;
            gini_l -= p * p;
          }
          if (right_counts[k] > 0) {
            double p = (double)right_counts[k] / nright;
            gini_r -= p * p;
          }
        }
        gini = ((double)nleft * gini_l + (double)nright * gini_r) / nsub;
        if (gini < best_gini) {
          best_gini = gini;
          node->feature = f;
          node->threshold = 0.5 * (pairs[i].val + pairs[i + 1].val);
        }
      }
    }
  }
  }
  LIBXS_PREDICT_FREE(pairs, pairs_pool);
  result = (node->feature >= 0) ? 1 : 0;
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tree(
  const internal_libxs_predict_entry_t* entries,
  int* subset, int nsub, int nfeat, int max_depth, int min_leaf,
  internal_libxs_predict_rf_node_t* nodes, int max_nodes,
  int output_idx, int label_off)
{
  int stack_subset[64], stack_count[64], stack_depth[64], stack_node[64];
  int sp = 0, nnodes = 0;
  int nfeatsub = (int)(sqrt((double)nfeat) + 0.5);
  if (nfeatsub < 1) nfeatsub = 1;
  stack_subset[0] = 0;
  stack_count[0] = nsub;
  stack_depth[0] = 0;
  stack_node[0] = nnodes++;
  nodes[0].feature = -1;
  nodes[0].left = -1;
  nodes[0].right = -1;
  nodes[0].label = 0;
  sp = 1;
  while (sp > 0 && nnodes < max_nodes - 2) {
    const int si = stack_subset[--sp];
    const int nc = stack_count[sp];
    const int depth = stack_depth[sp];
    const int ni = stack_node[sp];
    int counts[128] = { 0 }, best_label = 0, best_count = 0, k;
    internal_libxs_predict_rf_node_t split;
    LIBXS_MEMZERO(&split);
    for (k = 0; k < nc; ++k) {
      ++counts[(LIBXS_ROUNDX(int, entries[subset[si + k]].outputs[output_idx]) + label_off) & 127];
    }
    for (k = 0; k < 128; ++k) {
      if (counts[k] > best_count) { best_count = counts[k]; best_label = k; }
    }
    nodes[ni].label = best_label;
    if (depth >= max_depth || nc <= min_leaf || best_count == nc
      || 0 == internal_libxs_predict_rf_split(entries, subset + si, nc,
        nfeat, nfeatsub, &split, (size_t)ni, output_idx, label_off))
    {
      nodes[ni].feature = -1;
      continue;
    }
    { int* sub = subset + si;
      int i, nleft = 0, nright = 0;
      nodes[ni].feature = split.feature;
      nodes[ni].threshold = split.threshold;
      for (i = 0; i < nc; ++i) {
        if (entries[sub[i]].inputs[split.feature] <= split.threshold) ++nleft;
      }
      nright = nc - nleft;
      if (0 == nleft || 0 == nright) { nodes[ni].feature = -1; continue; }
      { int part_pool = 0;
        int* part = (int*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(int), part_pool);
        if (NULL != part) {
          int li = 0, ri = 0;
          for (i = 0; i < nc; ++i) {
            if (entries[sub[i]].inputs[split.feature] <= split.threshold) {
              part[li++] = sub[i];
            }
            else {
              part[nleft + ri++] = sub[i];
            }
          }
          memcpy(sub, part, (size_t)nc * sizeof(int));
          LIBXS_PREDICT_FREE(part, part_pool);
        }
        else { nodes[ni].feature = -1; continue; }
      }
      /**
       * A child is created before it is known whether the stack has room to
       * process it.  Its label must be initialized here: when sp saturates the
       * node is never popped, and eval reads label unconditionally at a leaf.
       * Left uninitialized it takes whatever the scratch allocator returned,
       * which varies with allocation history and thread count - the cause of
       * run-to-run differences in RF results.  The parent's majority label is
       * the correct fallback, being what a leaf at this point would predict.
       */
      nodes[ni].left = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      nodes[nnodes].label = best_label;
      if (sp < 64) {
        stack_subset[sp] = si;
        stack_count[sp] = nleft;
        stack_depth[sp] = depth + 1;
        stack_node[sp] = nnodes;
        ++sp;
      }
      ++nnodes;
      nodes[ni].right = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      nodes[nnodes].label = best_label;
      if (sp < 64) {
        stack_subset[sp] = si + nleft;
        stack_count[sp] = nright;
        stack_depth[sp] = depth + 1;
        stack_node[sp] = nnodes;
        ++sp;
      }
      ++nnodes;
    }
  }
  return nnodes;
}


#if !defined(LIBXS_PREDICT_RF_NTREES)
#  define LIBXS_PREDICT_RF_NTREES 100
#endif
/** Trees per candidate while scoring depth: enough to average out the
 *  bootstrap, few enough that trying four depths is not four full builds. */
#if !defined(LIBXS_PREDICT_RF_PROBE)
#  define LIBXS_PREDICT_RF_PROBE 12
#endif


/**
 * Misclassification rate of a small forest grown to max_depth over the first
 * ntrain entries, measured on the rest.  Depth is scored rather than derived
 * because the derived 2*log2(p) is a function of the corpus size alone: it says
 * 20 for a corpus of 1339 whether that corpus has three features or three
 * hundred, and a tree that deep on three features is fitting the sample.
 */
LIBXS_API_INLINE double internal_libxs_predict_rf_score(
  const internal_libxs_predict_entry_t* entries, int p, int m,
  int output_idx, int label_off, int max_depth, int min_leaf, int ntrain)
{
  const int nt = LIBXS_PREDICT_RF_PROBE;
  const int max_nodes = LIBXS_MIN(ntrain / min_leaf * 2 + 1, 65536);
  int nodes_pool = 0, boot_pool = 0, nn_pool = 0;
  internal_libxs_predict_rf_node_t* nodes =
    (internal_libxs_predict_rf_node_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nt * (size_t)max_nodes
        * sizeof(internal_libxs_predict_rf_node_t), nodes_pool);
  int* bootstrap = (int*)LIBXS_PREDICT_MALLOC((size_t)ntrain * sizeof(int),
    boot_pool);
  int* nn = (int*)LIBXS_PREDICT_MALLOC((size_t)nt * sizeof(int), nn_pool);
  double result = 1.0;
  if (NULL != nodes && NULL != bootstrap && NULL != nn) {
    const size_t boot_n = (size_t)ntrain * 2 + 1;
    const size_t boot_coprime = libxs_coprime2(boot_n);
    int t, i, wrong = 0, scored = 0;
    for (t = 0; t < nt; ++t) {
      for (i = 0; i < ntrain; ++i) {
        bootstrap[i] = (int)(LIBXS_SHUFFLE_INDEX(i, boot_n, boot_coprime,
          (size_t)t * 7 + 13) % (size_t)ntrain);
      }
      nn[t] = internal_libxs_predict_rf_build_tree(entries, bootstrap, ntrain,
        m, max_depth, min_leaf, nodes + (size_t)t * max_nodes, max_nodes,
        output_idx, label_off);
    }
    for (i = ntrain; i < p; ++i) {
      const double* inputs = entries[i].inputs;
      const int label =
        (LIBXS_ROUNDX(int, entries[i].outputs[output_idx]) + label_off) & 127;
      int votes[128], best_label = 0, best_count = 0, k;
      memset(votes, 0, sizeof(votes));
      for (t = 0; t < nt; ++t) {
        const internal_libxs_predict_rf_node_t* tn = nodes + (size_t)t * max_nodes;
        int ni = 0;
        if (0 >= nn[t]) continue;
        while (ni >= 0 && ni < nn[t] && tn[ni].feature >= 0) {
          ni = (inputs[tn[ni].feature] <= tn[ni].threshold)
            ? tn[ni].left : tn[ni].right;
        }
        if (ni >= 0 && ni < nn[t]) ++votes[tn[ni].label & 127];
      }
      for (k = 0; k < 128; ++k) {
        if (votes[k] > best_count) { best_count = votes[k]; best_label = k; }
      }
      if (best_label != label) ++wrong;
      ++scored;
    }
    result = (0 < scored) ? ((double)wrong / scored) : 1.0;
  }
  LIBXS_PREDICT_FREE(nn, nn_pool);
  LIBXS_PREDICT_FREE(bootstrap, boot_pool);
  LIBXS_PREDICT_FREE(nodes, nodes_pool);
  return result;
}

LIBXS_API_INLINE void internal_libxs_predict_rf_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int n = model->noutputs;
  const int ntrees = (0 < model->rf_ntrees)
    ? model->rf_ntrees : LIBXS_PREDICT_RF_NTREES;
  internal_libxs_predict_rf_t* rf =
    (internal_libxs_predict_rf_t*)calloc(1, sizeof(internal_libxs_predict_rf_t));
  if (NULL != rf) {
    rf->trees = (internal_libxs_predict_rf_tree_t*)calloc(
      (size_t)ntrees * (size_t)n, sizeof(internal_libxs_predict_rf_tree_t));
    rf->label_offset = (int*)malloc((size_t)n * sizeof(int));
    rf->depth = (int*)malloc((size_t)n * sizeof(int));
    rf->ntrees = ntrees;
    rf->noutputs = n;
    if (NULL != rf->trees && NULL != rf->label_offset && NULL != rf->depth) {
      const int derived = (int)(2.0 * log((double)p) / log(2.0));
      const int min_leaf = 5;
      const int ntrain = (int)(p * 0.8 + 0.5);
      int oi, i;
      for (oi = 0; oi < n; ++oi) {
        int vmin = LIBXS_ROUNDX(int, model->entries[0].outputs[oi]);
        for (i = 1; i < p; ++i) {
          const int v = LIBXS_ROUNDX(int, model->entries[i].outputs[oi]);
          if (v < vmin) vmin = v;
        }
        rf->label_offset[oi] = -vmin;
      }
      if (0 < model->rf_depth) {
        for (oi = 0; oi < n; ++oi) rf->depth[oi] = model->rf_depth;
      }
      /**
       * Scoring is opt-in (a negative request), not the default, because it was
       * measured not to pay: on the shipped tuning corpus it moved exact match
       * over sixteen outputs by half a point and made the absolute error of the
       * widest three outputs worse, while costing the crystal corpus 2.6x its
       * build.  It is kept because it is the only way to find out for a corpus
       * where the derived depth is wrong, and because there was previously no
       * way to ask at all.
       */
      else if (0 == model->rf_depth || ntrain <= min_leaf || ntrain >= p) {
        for (oi = 0; oi < n; ++oi) rf->depth[oi] = derived;
      }
      else for (oi = 0; oi < n; ++oi) {
        /**
         * Scored per output, not once for the model: outputs of one corpus
         * differ in how much structure there is to fit, and the shallow
         * candidates exist because the derived depth is the overfitting end of
         * the range on a small corpus with few features.
         */
        int cand[4];
        double best_err = -1.0;
        int best = derived, ci;
        cand[0] = 3;
        cand[1] = 6;
        cand[2] = derived / 2;
        cand[3] = derived;
        for (ci = 0; ci < 4; ++ci) {
          const int d = (3 > cand[ci]) ? 3 : cand[ci];
          if (0 < ci && d == ((3 > cand[ci-1]) ? 3 : cand[ci-1])) continue;
          { const double err = internal_libxs_predict_rf_score(model->entries,
              p, model->ninputs, oi, rf->label_offset[oi], d, min_leaf, ntrain);
            if (0 > best_err || err < best_err) { best_err = err; best = d; }
          }
        }
        rf->depth[oi] = best;
      }
      model->rf = rf;
    }
    else {
      free(rf->trees);
      free(rf->label_offset);
      free(rf->depth);
      free(rf);
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_rf_build_tasks(
  libxs_predict_t* model, int tid, int ntasks)
{
  const internal_libxs_predict_rf_t* rf = model->rf;
  if (NULL != rf) {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = rf->noutputs;
    const int ntrees = rf->ntrees;
    const int total_trees = ntrees * n;
    const int min_leaf = 5;
    const int max_nodes = LIBXS_MIN(p / min_leaf * 2 + 1, 65536);
    const int chunk = (total_trees + ntasks - 1) / ntasks;
    const int begin = tid * chunk;
    const int end = LIBXS_MIN(begin + chunk, total_trees);
    int bootstrap_pool = 0;
    int* bootstrap = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), bootstrap_pool);
    if (NULL != bootstrap) {
      int ti;
      for (ti = begin; ti < end; ++ti) {
        const int oi = ti / ntrees;
        const int t = ti % ntrees;
        const int max_depth = rf->depth[oi];
        const size_t boot_n = (size_t)p * 2 + 1;
        const size_t boot_coprime = libxs_coprime2(boot_n);
        int nodes_pool = 0;
        internal_libxs_predict_rf_node_t* nodes;
        int i, nn;
        if (NULL != rf->trees[ti].nodes) continue;
        nodes = (internal_libxs_predict_rf_node_t*)LIBXS_PREDICT_MALLOC(
            (size_t)max_nodes * sizeof(internal_libxs_predict_rf_node_t),
            nodes_pool);
        for (i = 0; i < p; ++i) {
          bootstrap[i] = (int)(LIBXS_SHUFFLE_INDEX(
            i, boot_n, boot_coprime,
            (size_t)(oi * ntrees + t) * 7 + 13) % (size_t)p);
        }
        if (NULL != nodes) {
          nn = internal_libxs_predict_rf_build_tree(
            model->entries, bootstrap, p, m, max_depth, min_leaf,
            nodes, max_nodes, oi, rf->label_offset[oi]);
          rf->trees[ti].nodes = (internal_libxs_predict_rf_node_t*)malloc(
            (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
          if (NULL != rf->trees[ti].nodes) {
            memcpy(rf->trees[ti].nodes, nodes,
              (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
            rf->trees[ti].nnodes = nn;
          }
          LIBXS_PREDICT_FREE(nodes, nodes_pool);
        }
      }
      LIBXS_PREDICT_FREE(bootstrap, bootstrap_pool);
    }
  }
}


LIBXS_API_INLINE int internal_libxs_predict_rf_eval_output(
  const internal_libxs_predict_rf_t* rf, int output_idx,
  const double* inputs, double* confidence)
{
  int votes[128] = { 0 };
  int best_label = 0, best_count = 0, t, k;
  const int base = output_idx * rf->ntrees;
  for (t = 0; t < rf->ntrees; ++t) {
    const internal_libxs_predict_rf_tree_t* tree = &rf->trees[base + t];
    int ni = 0;
    if (NULL == tree->nodes || 0 == tree->nnodes) continue;
    while (ni >= 0 && ni < tree->nnodes && tree->nodes[ni].feature >= 0) {
      const internal_libxs_predict_rf_node_t* nd = &tree->nodes[ni];
      ni = (inputs[nd->feature] <= nd->threshold) ? nd->left : nd->right;
    }
    if (ni >= 0 && ni < tree->nnodes) {
      ++votes[tree->nodes[ni].label & 127];
    }
  }
  for (k = 0; k < 128; ++k) {
    if (votes[k] > best_count) { best_count = votes[k]; best_label = k; }
  }
  if (NULL != confidence) {
    *confidence = (rf->ntrees > 0) ? (double)best_count / rf->ntrees : 0.0;
  }
  return best_label - rf->label_offset[output_idx];
}
