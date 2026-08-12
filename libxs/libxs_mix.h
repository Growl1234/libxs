/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MIX_H
#define LIBXS_MIX_H

#include "libxs_macros.h"

/**
 * Fixed-share mixture over a bank of experts: a linear pool whose weights are
 * carried across observations and updated multiplicatively from realized log
 * loss, with a uniform share redistributed each step so an expert that was
 * wrong for a stretch can recover.
 *
 * NOT to be confused with libxs_predict_set_series_bank, which averages several
 * views of one window with FIXED equal weight. This learns the weights.
 *
 * The pool is scored strictly BEFORE the weights move, which is what makes a
 * stream figure honest: weights that had already seen the outcome yield a code
 * length better than the truth, with no symptom to notice. libxs_mix_observe
 * enforces that ordering rather than asking the caller to get it right.
 *
 * An expert may ABSTAIN at a position (no opinion, rather than a low opinion).
 * Abstention is not the same as a probability of zero: the pool renormalizes
 * over the experts that spoke, so a silent expert neither dilutes the mixture
 * nor is punished for staying quiet. With every expert active the
 * renormalization is a no-op, because the weights sum to one.
 *
 * A weight of zero means the slot is DISABLED, permanently: a bank sized for the
 * widest configuration holds the slots it does not use at zero, and share mass
 * must not resurrect them. A bank that instead wants every slot revivable --
 * libxs_predict's escape-rate bank is one, because its slots are a fixed ladder
 * rather than an optional set -- is not a client of this primitive, and saying so
 * is cheaper than a flag that makes one contract mean two things.
 */
LIBXS_EXTERN_C typedef struct libxs_mix_t {
  /** Per-slot weight, summing to one over slots that have ever been active. */
  double* weight;
  /** Number of slots. */
  int nslot;
  /** Multiplicative learning rate (exponent on the likelihood ratio). */
  double rate;
  /** Uniform share redistributed each step (0 disables recovery). */
  double share;
  /**
   * Floor substituted for a likelihood ratio that is not positive. Without it
   * an expert that once gave the outcome no mass at all is multiplied by zero
   * and can never recover, because the share term only reaches slots that still
   * hold mass. Pass 0 to disable, which is a deliberate choice to let such an
   * expert die permanently.
   */
  double relmin;
} libxs_mix_t;


/**
 * Initialize a mixture over nslot experts with uniform weights.
 * Returns EXIT_SUCCESS, or EXIT_FAILURE on invalid arguments or allocation
 * failure. The caller owns the struct; libxs_mix_destroy releases the weights.
 *
 * The struct is plain data, so a caller that already owns a weight array may
 * instead fill the fields directly and point weight at it -- useful for a bank
 * that lives on the stack or inside a larger record. Such a view must NOT be
 * passed to libxs_mix_destroy, which would free memory it does not own.
 */
LIBXS_API int libxs_mix_create(libxs_mix_t* mix, int nslot,
  double rate, double share, double relmin);

/** Release the weights (NULL is accepted; the struct itself is caller-owned). */
LIBXS_API void libxs_mix_destroy(libxs_mix_t* mix);

/**
 * Reset to a uniform prior over the slots marked active (NULL = all slots).
 * A slot left inactive here starts at zero and stays there, which is how a bank
 * sized for the widest configuration behaves identically to one built only for
 * the slots in use.
 */
LIBXS_API void libxs_mix_reset(libxs_mix_t* mix, const int active[]);

/**
 * Pooled probability without touching the weights, for reporting or for ranking
 * candidates at one position. prob[] holds each expert's probability for the
 * outcome under consideration; active[] may be NULL (all experts speak).
 * Returns 0 when no expert speaks -- callers that take a logarithm must floor
 * the result themselves, because the right floor is theirs to choose.
 */
LIBXS_API double libxs_mix_pool(const libxs_mix_t* mix, const double prob[],
  const int active[]);

/**
 * Advance the weights toward the experts that beat a mixture the CALLER
 * computed. Use this when the pool is not libxs_mix_pool's: a caller may mix in
 * experts this bank does not carry, or floor the pooled value before taking its
 * logarithm, and the update must then be relative to the value actually used --
 * not to a recomputed one that differs in the last bits or in the slot set.
 *
 * Passing libxs_mix_pool's own result here is exactly libxs_mix_observe.
 */
LIBXS_API void libxs_mix_update(libxs_mix_t* mix, const double prob[],
  const int active[], double mixture);

/**
 * Pool, then advance the weights toward the experts that beat the pool.
 * Returns the pooled probability computed BEFORE the update -- the value that
 * may be reported or accumulated into a code length. Preferred for new code:
 * the ordering that keeps a stream figure honest cannot be got wrong.
 */
LIBXS_API double libxs_mix_observe(libxs_mix_t* mix, const double prob[],
  const int active[]);

/* header-only: include implementation */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_MIX_H*/
