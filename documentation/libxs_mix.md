# Mix

Header: `libxs_mix.h`

Fixed-share mixture over a bank of experts: a linear pool whose per-expert
weights are carried across observations and updated from realized log loss, so
the mixture tracks whichever expert is currently right. Experts may abstain per
observation.

Not to be confused with `libxs_predict_set_series_bank`, which averages several
views of one window with fixed equal weight; this learns the weights.

## Types

```C
typedef struct libxs_mix_t {
  double* weight;  /* per-slot weight [nslot] */
  int nslot;       /* number of experts */
  double rate;     /* exponent on the likelihood ratio */
  double share;    /* uniform share redistributed per step */
  double relmin;   /* floor for a non-positive ratio (0 disables) */
} libxs_mix_t;
```

The struct is plain data. A caller that already owns a weight array may fill the
fields directly and point `weight` at it, which suits a bank living on the stack
or inside a larger record; such a view must not be passed to
`libxs_mix_destroy`.

## Functions

```C
int libxs_mix_create(libxs_mix_t* mix, int nslot,
  double rate, double share, double relmin);
void libxs_mix_destroy(libxs_mix_t* mix);
void libxs_mix_reset(libxs_mix_t* mix, const int active[]);
```

`create` starts from a uniform prior and returns `EXIT_SUCCESS` or
`EXIT_FAILURE`. `reset` restores a uniform prior over the slots marked active
(NULL selects all). `destroy` releases the weights; the struct itself is
caller-owned.

```C
double libxs_mix_pool(const libxs_mix_t* mix,
  const double prob[], const int active[]);
void libxs_mix_update(libxs_mix_t* mix,
  const double prob[], const int active[], double mixture);
double libxs_mix_observe(libxs_mix_t* mix,
  const double prob[], const int active[]);
```

`prob[k]` is expert `k`'s probability for the outcome under consideration.
`active` may be NULL (every expert speaks). `pool` reports the mixture without
touching the weights. `update` advances the weights against a mixture the caller
supplies. `observe` is `pool` followed by `update` with that pool, and returns
the pooled value taken *before* the update.

Prefer `observe`: a pooled value taken after `update` silently understates the
code length. Use `update` only when the caller's mixture differs from `pool`'s,
for instance because it mixes in experts this bank does not carry or was floored
before taking a logarithm.

## Abstention and Disabled Slots

Abstention (`active[k] == 0`) is not a probability of zero. The pool renormalizes
over the experts that spoke, so a silent expert does not dilute the mixture.
An update renormalizes the experts that spoke to sum to one, while a silent
expert keeps its absolute weight: its weight relative to the others changes
while it is quiet, and the weights as a whole need not sum to one.

A weight of zero disables the slot permanently (share mass does not revive it),
which suits the unused slots of a bank sized for the widest configuration.

With `relmin` of zero, an expert that once gave the outcome no probability drops
to zero weight for good; a positive `relmin` lets it recover.

## Example

```C
libxs_mix_t mix;
double prob[3];
if (EXIT_SUCCESS == libxs_mix_create(&mix, 3, 0.15, 0.005, 1e-4)) {
  double bits = 0;
  while (next_observation(prob)) {
    const double p = libxs_mix_observe(&mix, prob, NULL);
    bits -= log(0 < p ? p : 1e-12) / log(2.0);
  }
  libxs_mix_destroy(&mix);
}
```
