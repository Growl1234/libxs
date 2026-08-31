LIBXS_API_INLINE int internal_libxs_predict_rf_pair_cmp(
  const void* a, const void* b, void* ctx)
{
  const double va = ((const internal_libxs_predict_rf_pair_t*)a)->val;
  const double vb = ((const internal_libxs_predict_rf_pair_t*)b)->val;
  LIBXS_UNUSED(ctx);
  return (va > vb) - (va < vb);
}


/**
 * Best split of the subset over a random subset of the features. A folded
 * output is split on Gini impurity over its class counts; a real-valued one on
 * the residual sum of squares either side of the threshold, which running sums
 * carry from one candidate threshold to the next in constant time instead of
 * the sweep over 128 counts that impurity needs.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_split(
  const internal_libxs_predict_entry_t* entries,
  const int* subset, int nsub, int nfeat, int nfeatsub,
  internal_libxs_predict_rf_node_t* node, size_t seed,
  int output_idx, int label_off, int regress)
{
  int result = 0;
  /** Negative until a candidate is seen: unlike impurity, a sum of squares has
   *  no upper bound that could serve as the initial best. */
  double best_score = -1.0;
  double mu = 0;
  int trial, i;
  int pairs_pool = 0;
  const size_t feat_coprime = libxs_coprime2((size_t)nfeat);
  internal_libxs_predict_rf_pair_t* pairs =
    (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nsub * sizeof(internal_libxs_predict_rf_pair_t), pairs_pool);
  node->feature = -1;
  node->label = -1;
  if (NULL != pairs) {
  /**
   * Deviations are taken about the subset mean rather than about zero: the
   * sums of squares of an output that is large and narrow differ in their
   * trailing digits only, and a split's improvement is lost in the
   * cancellation. Subtracting the mean first puts that improvement in the
   * leading digits.
   */
  if (0 != regress && 0 < nsub) {
    for (i = 0; i < nsub; ++i) mu += entries[subset[i]].outputs[output_idx];
    mu /= nsub;
  }
  for (trial = 0; trial < nfeatsub; ++trial) {
    const int f = (int)(LIBXS_SHUFFLE_INDEX(
      (size_t)trial, (size_t)nfeat, feat_coprime, seed) % (size_t)nfeat);
    int nleft, nright;
    for (i = 0; i < nsub; ++i) {
      pairs[i].val = entries[subset[i]].inputs[f];
      pairs[i].idx = subset[i];
    }
    libxs_sort(pairs, nsub, sizeof(pairs[0]),
      internal_libxs_predict_rf_pair_cmp, NULL);
    if (0 != regress) {
      double sum_l = 0, sqr_l = 0, sum_t = 0, sqr_t = 0;
      for (i = 0; i < nsub; ++i) {
        const double d = entries[pairs[i].idx].outputs[output_idx] - mu;
        sum_t += d;
        sqr_t += d * d;
      }
      nright = nsub; nleft = 0;
      for (i = 0; i < nsub - 1; ++i) {
        const double d = entries[pairs[i].idx].outputs[output_idx] - mu;
        sum_l += d; sqr_l += d * d; ++nleft;
        --nright;
        if (pairs[i].val == pairs[i + 1].val) continue;
        /** The right side is the total less the left rather than a second
         *  running sum: subtracting each element in turn would accumulate the
         *  cancellation of every step, and the right side ends near zero. */
        { const double sum_r = sum_t - sum_l;
          const double sqr_r = sqr_t - sqr_l;
          const double sse = (sqr_l - sum_l * sum_l / nleft)
            + (sqr_r - sum_r * sum_r / nright);
          if (0 > best_score || sse < best_score) {
            best_score = sse;
            node->feature = f;
            node->threshold = 0.5 * (pairs[i].val + pairs[i + 1].val);
          }
        }
      }
    }
    else {
      int left_counts[128], right_counts[128];
      int k;
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
          if (0 > best_score || gini < best_score) {
            best_score = gini;
            node->feature = f;
            node->threshold = 0.5 * (pairs[i].val + pairs[i + 1].val);
          }
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
  int output_idx, int label_off, int regress)
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
  nodes[0].value = 0;
  sp = 1;
  while (sp > 0 && nnodes < max_nodes - 2) {
    const int si = stack_subset[--sp];
    const int nc = stack_count[sp];
    const int depth = stack_depth[sp];
    const int ni = stack_node[sp];
    int best_label = 0, best_count = 0, pure = 0, k;
    double mean = 0, dev = 0;
    internal_libxs_predict_rf_node_t split;
    LIBXS_MEMZERO(&split);
    if (0 != regress) {
      for (k = 0; k < nc; ++k) {
        mean += entries[subset[si + k]].outputs[output_idx];
      }
      if (0 < nc) mean /= nc;
      for (k = 0; k < nc; ++k) {
        const double d = entries[subset[si + k]].outputs[output_idx] - mean;
        dev += d * d;
      }
      /** A constant subset has nothing left to split on; without this the
       *  best split of no variance is still taken and grows dead nodes. */
      pure = (0 == dev) ? 1 : 0;
    }
    else {
      int counts[128] = { 0 };
      for (k = 0; k < nc; ++k) {
        ++counts[(LIBXS_ROUNDX(int, entries[subset[si + k]].outputs[output_idx]) + label_off) & 127];
      }
      for (k = 0; k < 128; ++k) {
        if (counts[k] > best_count) { best_count = counts[k]; best_label = k; }
      }
      mean = (double)best_label;
      pure = (best_count == nc) ? 1 : 0;
    }
    nodes[ni].label = best_label;
    nodes[ni].value = mean;
    if (depth >= max_depth || nc <= min_leaf || 0 != pure
      || 0 == internal_libxs_predict_rf_split(entries, subset + si, nc,
        nfeat, nfeatsub, &split, (size_t)ni, output_idx, label_off, regress))
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
       * process it. Its read-out must be initialized here: when sp saturates
       * the node is never popped, and eval reads it unconditionally at a leaf.
       * Left uninitialized it takes whatever the scratch allocator returned,
       * which varies with allocation history and thread count - the cause of
       * run-to-run differences in RF results. The parent's own read-out is
       * the correct fallback, being what a leaf at this point would predict.
       */
      nodes[ni].left = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      nodes[nnodes].label = best_label;
      nodes[nnodes].value = mean;
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
      nodes[nnodes].value = mean;
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
 * Error of a small forest grown to max_depth over the first ntrain entries,
 * measured on the rest: the misclassification rate of a folded output, the mean
 * absolute error of a real-valued one. The two are never compared against each
 * other, only across the depth candidates of one output. Depth is scored
 * rather than derived because the derived 2*log2(p) is a function of the corpus
 * size alone: it says 20 for a corpus of 1339 whether that corpus has three
 * features or three hundred, and a tree that deep on three features is fitting
 * the sample.
 */
LIBXS_API_INLINE double internal_libxs_predict_rf_score(
  const internal_libxs_predict_entry_t* entries, int p, int m,
  int output_idx, int label_off, int max_depth, int min_leaf, int ntrain,
  int regress)
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
    double err = 0;
    for (t = 0; t < nt; ++t) {
      for (i = 0; i < ntrain; ++i) {
        bootstrap[i] = (int)(LIBXS_SHUFFLE_INDEX(i, boot_n, boot_coprime,
          (size_t)t * 7 + 13) % (size_t)ntrain);
      }
      nn[t] = internal_libxs_predict_rf_build_tree(entries, bootstrap, ntrain,
        m, max_depth, min_leaf, nodes + (size_t)t * max_nodes, max_nodes,
        output_idx, label_off, regress);
    }
    for (i = ntrain; i < p; ++i) {
      const double* inputs = entries[i].inputs;
      const int label =
        (LIBXS_ROUNDX(int, entries[i].outputs[output_idx]) + label_off) & 127;
      int votes[128], best_label = 0, best_count = 0, k, nvalid = 0;
      double sum = 0;
      memset(votes, 0, sizeof(votes));
      for (t = 0; t < nt; ++t) {
        const internal_libxs_predict_rf_node_t* tn = nodes + (size_t)t * max_nodes;
        int ni = 0;
        if (0 >= nn[t]) continue;
        while (ni >= 0 && ni < nn[t] && tn[ni].feature >= 0) {
          ni = (inputs[tn[ni].feature] <= tn[ni].threshold)
            ? tn[ni].left : tn[ni].right;
        }
        if (ni >= 0 && ni < nn[t]) {
          if (0 != regress) { sum += tn[ni].value; ++nvalid; }
          else ++votes[tn[ni].label & 127];
        }
      }
      if (0 != regress) {
        if (0 < nvalid) {
          err += LIBXS_FABS(sum / nvalid - entries[i].outputs[output_idx]);
        }
      }
      else {
        for (k = 0; k < 128; ++k) {
          if (votes[k] > best_count) { best_count = votes[k]; best_label = k; }
        }
        if (best_label != label) ++wrong;
      }
      ++scored;
    }
    if (0 < scored) {
      result = (0 != regress) ? (err / scored) : ((double)wrong / scored);
    }
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
    rf->regress = (int*)malloc((size_t)n * sizeof(int));
    rf->depth = (int*)malloc((size_t)n * sizeof(int));
    rf->ntrees = ntrees;
    rf->noutputs = n;
    if (NULL != rf->trees && NULL != rf->label_offset && NULL != rf->depth
      && NULL != rf->regress)
    {
      const int derived = (int)(2.0 * log((double)p) / log(2.0));
      const int min_leaf = 5;
      const int ntrain = (int)(p * 0.8 + 0.5);
      int oi, i;
      /**
       * An output is a class only if it is written like one: every value
       * integral, and the whole range representable in the 128 labels the
       * folding leaves. Everything else is a quantity, split on variance and
       * read out as a mean. Both halves of the test matter. Without the
       * first, a magnitude of 4.5 is answered as 5 and the forest cannot beat
       * its own rounding. Without the second, an integral output spanning more
       * than the fold wraps two distant values onto one label and the forest is
       * left predicting a class it invented.
       */
      for (oi = 0; oi < n; ++oi) {
        double vmin = model->entries[0].outputs[oi];
        double vmax = vmin;
        int integral = 1;
        for (i = 0; i < p; ++i) {
          const double v = model->entries[i].outputs[oi];
          if (0 != LIBXS_NOTNAN(v)) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
          /** Rounding is only defined over the range of the integer it goes
           *  through; beyond it the value is a quantity in any case. */
          if (0 != integral && (0 == LIBXS_NOTNAN(v)
            || 2147483647.0 < LIBXS_FABS(v)
            || 0.0 != v - (double)LIBXS_ROUNDX(long long, v)))
          {
            integral = 0;
          }
        }
        rf->regress[oi] = (0 == integral || 127.0 < vmax - vmin) ? 1 : 0;
        /** Only a class has an offset, and only a class is in range for it. */
        rf->label_offset[oi] = (0 == rf->regress[oi])
          ? -LIBXS_ROUNDX(int, vmin) : 0;
      }
      if (0 < model->rf_depth) {
        for (oi = 0; oi < n; ++oi) rf->depth[oi] = model->rf_depth;
      }
      /**
       * Scoring is opt-in (a negative request), not the default, because it was
       * measured not to pay: on the shipped tuning corpus it moved exact match
       * over sixteen outputs by half a point and made the absolute error of the
       * widest three outputs worse, while costing the crystal corpus 2.6x its
       * build. It is kept because it is the only way to find out for a corpus
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
              p, model->ninputs, oi, rf->label_offset[oi], d, min_leaf, ntrain,
              rf->regress[oi]);
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
      free(rf->regress);
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
    int begin, end, bootstrap_pool = 0;
    int* bootstrap = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), bootstrap_pool);
    internal_libxs_predict_split(total_trees, tid, ntasks, &begin, &end);
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
            nodes, max_nodes, oi, rf->label_offset[oi], rf->regress[oi]);
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


/**
 * Forest read-out for one output: the mean of the leaves reached for a
 * real-valued output, the majority of the labels reached for a folded one.
 * Both descend the same trees, so the read-out is a choice of what to
 * accumulate rather than a second traversal.
 */
LIBXS_API_INLINE double internal_libxs_predict_rf_eval_output(
  const internal_libxs_predict_rf_t* rf, int output_idx,
  const double* inputs, double* confidence, double* variance)
{
  const int regress = (NULL != rf->regress) ? rf->regress[output_idx] : 0;
  const int base = output_idx * rf->ntrees;
  int votes[128];
  int best_label = 0, best_count = 0, nvalid = 0, t, k;
  double sum = 0, sqr = 0, result;
  if (0 == regress) memset(votes, 0, sizeof(votes));
  for (t = 0; t < rf->ntrees; ++t) {
    const internal_libxs_predict_rf_tree_t* tree = &rf->trees[base + t];
    int ni = 0;
    if (NULL == tree->nodes || 0 == tree->nnodes) continue;
    while (ni >= 0 && ni < tree->nnodes && tree->nodes[ni].feature >= 0) {
      const internal_libxs_predict_rf_node_t* nd = &tree->nodes[ni];
      ni = (inputs[nd->feature] <= nd->threshold) ? nd->left : nd->right;
    }
    if (ni >= 0 && ni < tree->nnodes) {
      if (0 != regress) {
        const double v = tree->nodes[ni].value;
        sum += v;
        sqr += v * v;
        ++nvalid;
      }
      else ++votes[tree->nodes[ni].label & 127];
    }
  }
  if (0 != regress) {
    result = (0 < nvalid) ? (sum / nvalid) : 0.0;
    if (NULL != variance) {
      /** Clamped because the closed form can fall below zero by rounding when
       *  the trees agree; it never can mathematically. */
      const double v = (0 < nvalid) ? (sqr / nvalid - result * result) : 0.0;
      *variance = (0 < v) ? v : 0.0;
    }
    /**
     * The spread across trees is reported as variance and not folded into the
     * confidence: a dispersion-derived confidence was measured to be worse than
     * a pinned one, and callers gate on the variance instead.
     */
    if (NULL != confidence) *confidence = 1.0;
  }
  else {
    for (k = 0; k < 128; ++k) {
      if (votes[k] > best_count) { best_count = votes[k]; best_label = k; }
    }
    if (NULL != confidence) {
      *confidence = (rf->ntrees > 0) ? (double)best_count / rf->ntrees : 0.0;
    }
    if (NULL != variance) *variance = 0;
    result = (double)(best_label - rf->label_offset[output_idx]);
  }
  return result;
}
