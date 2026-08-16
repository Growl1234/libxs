# Converse Sample

Converse is a small, self-contained language sample built on libxs metatokens,
lexemes, fingerprints, registries, n-grams, and parameter prediction. It provides
grounded extractive question answering, learned relation and identity facts, an
experimental next-token predictor, and evaluation fixtures. For design and
measured comparisons, see `~/papers/converse/paper.tex`.

## Build

Build the library first so that the sample can link against it:

```bash
cd ../..
make PEDANTIC=2
cd samples/converse
make PEDANTIC=2
```

Or from the `libxs` root:

```bash
make PEDANTIC=2 samples/converse
```

## Which binary

The build produces three, and each links only what it serves:

| binary | modes | contains |
| :--- | :--- | :--- |
| `converse-qa.x` | interactive, `-e`, `-c`, `-L` | grounded answering and recombination |
| `converse-lm.x` | `-E`, `-c -K KIND`, `-L` | next-token models and the byte model |
| `converse.x` | all of the above | both halves |

`-L` only writes the shared corpus, lexicon and predictor, so either half can do
it. A mode the binary does not serve is rejected before any corpus work, naming
the binary that does serve it. `converse.x` accepts every documented command and
is what the reproduction script uses.

The byte model (`converse_hier.c`) reaches the grounded half only as an installed
judge, so `converse-qa.x` does not contain it: `CONVERSE_HIER_RESCORE=1` there
answers without rescoring and the recombination probe prints `no seam judge` with
its bpc columns at zero, which every other column reproduces regardless. Use
`converse.x` when those columns are wanted.

## Summarize and compose

Summarize a file by repeatedly fusing adjacent sentences (`-n` sets the target
sentence count):

```bash
./summarize.x texts/prose1.txt
./summarize.x -n 3 texts/prose1.txt
printf '%s\n' 'First sentence. Second related sentence.' | ./summarize.x -
```

Compose mode (`-g`) ingests one or more files into `converse.dat`, fingerprints
and tokenizes the first input as the target, and emits a short assembled text
(`-n` sets the phrase budget). Remove `converse.dat` to rebuild from scratch:

```bash
./summarize.x -g -n 8 texts/prose1.txt texts/prose2.txt
```

## Interactive question answering

Run the interactive sample with one or more corpus files:

```bash
./converse.x -n 3 texts/prejudice.txt texts/prose1.txt
./converse.x -b texts/grimm
```

The `-b` option treats its argument as a file prefix and probes known filenames
without scanning the directory: `name`, `name.txt`, and numbered `.txt` parts
using `name-N.txt`, `name_N.txt`, or `name.N.txt`. Missing siblings are fine as
long as at least one candidate file is usable.

Question-shaped prompts are answered extractively and abstain
(`I do not know from the corpus.`) when coverage is too low. Questions of the
form `In Title, ...` are ranked only against a matching uppercase story heading.
Non-question prompts use the fingerprint/Hilbert composition path.

## Answer evaluation

Run a data-driven evaluation over a local fixture. The sample reads
`converse.eval` from the current directory; keep it next to the fixture text it
describes:

```bash
./converse.x -e -b texts/grimm
./converse.x -P temporal -e texts/prose1.txt
```

Each non-comment `converse.eval` line has three required pipe-separated fields
and an optional fourth:

```text
question|expected-evidence-terms|expected-reply-terms|expected-fact-terms
```

Expected terms are comma-separated. An empty evidence field marks an abstention
case; an empty reply field skips the concise-reply check. The optional fourth
field checks the learned-fact reply path and is evaluated only when relation
facts were learned this run (that is, when a `converse.relations` file is
present), so the same fixture passes with or without rules. Three-field lines
behave exactly as before.

## Local rule files

All corpus-specific vocabulary stays in local, ignored rule files rather than in
`converse.c`.

Relation rules (`converse.relations`) keep aliases, person-like terms, and
filler words out of source. Each non-comment line is one of:

```text
alias|query-relation|evidence-verb
person|term
skip|term
where|term
place|term
topic|term
copula|term
article|term
prep|term
own|term
poss|shape
aux|term
agent|term
```

For example `alias|eaten|devoured`, `person|grandmother`, `skip|the`. `where|in`
declares a location MARKER and `place|forest` a place noun, and the two together
give `Where ...?` questions a proposition to answer with: the reply is the actor
followed by a verbatim span of the actor's own sentence, cited. `why|because` and
`how|by` declare the corresponding markers. All of these are language facts rather
than corpus facts, so they belong in the shared `converse.rules`; a corpus whose
own `<corpus>.rules` replaces that file must state its own. After ingestion the
sample rebuilds an in-memory fact index and reports `relation facts: N learned`,
`identity facts: N learned` and `location facts: N learned`. `CONVERSE_FACTS_LIST=1`
prints the relation and location facts themselves, which is the only way to judge
them.

`topic|about` marks the subject of an attribute question, so
`What do we know about Hansel?` answers with several cited propositions collected
from the fact layers rather than with one retrieved sentence. Attested facts are
stated plainly; a speculative one speaks only when nothing about that name is
attested, and is labelled when it does.

Relation questions
consult this index before falling back to raw evidence. Identity facts of the
form `name is the role` draw their role words from the `person|term` rules and
bind a role to a name only within a single sentence; a `Who is X?` question then
answers from the highest-scoring identity fact for X, or abstains.

`copula|is`, `article|a` and `prep|of` are the syntactic classes the type shapes
need: with them, `Who is Aristotle?` can answer from the copular ("X is a Y") and
appositive ("X, a Y, ...") shapes prose states definitions in. `own|belongs` marks a
possession question, and `poss|apostrophe-s` declares how the language WRITES
possession -- English marks it with an apostrophe and an s, German with a bare s and
no apostrophe at all -- so `What belongs to Curdken?` enumerates what is attested,
each item cited. Declaring the wrong shape costs silence rather than error, because
the name census still has to recognize what remains once the mark is removed.

`aux|had` and `agent|by` are what the passive shape needs. The auxiliaries are
declared for one reason: the word an auxiliary governs is a VERB, so the class of
verbs is DERIVED from the corpus rather than listed (`verbs derived: N from the
auxiliary frame`). That class is incomplete by construction -- English narrative is
past simple, which no auxiliary governs -- so it is used only to REJECT: a name the
corpus puts after a verb is that verb's object, not the subject of its clause.
`agent|by` then makes any passive readable without a rule per verb, so
`X was visited by Y` becomes an edge between two entities.

The same derived class carries the ACTIVE shape -- a name, a verb, and either
another name or an article-headed phrase, as in `Agassi won the Australian Open` --
where it is a REQUIREMENT rather than a rejection: a word the corpus never puts
after an auxiliary is not read as a verb, so the shape declines to look rather than
guessing. What it declines costs a fact that is never stated, which is why the same
incomplete class is safe in both polarities. Such a fact is re-emitted in the voice
the corpus used, since restating what the corpus stated cannot be ungrammatical.

Bridge rules (`converse.bridges`) provide optional evidence-backed answer frames.
Each non-comment line has five pipe-separated fields:

```text
name|query-groups|evidence-groups|score|reply
```

Within query and evidence groups, whitespace separates required groups and `/`
separates alternatives; evidence terms can use `_` for a literal space. The
reply may be literal text or a small frame such as `{after:lighthouse had}` or
`{keywords-after:recorded everything:}`.

## Next-token prediction

A separate, experimental next-token predictor is trained from the ordered token
stream of the ingested corpus and kept out of the grounded QA path. It is
exercised through its own flags:

```bash
./converse.x -E -b texts/grimm            # next-token accuracy (default trigram)
./converse.x -E -K bigram -b texts/grimm  # choose the model
./converse.x -E -K hier -b texts/grimm    # hierarchy and byte PPM evaluation
printf 'the little\n' | ./converse.x -c -b texts/grimm   # suggestions + greedy
CONVERSE_GRAN=meta-word ./converse.x -E -b texts/grimm   # metatoken policy
```

`-E` reports next-token accuracy; `-c` reads prompts and prints the top few
next-token suggestions plus a short greedy continuation. `-H N` evaluates on a
held-out split (train on the other sentences, test on 1-in-N) for an honest,
non-memorized accuracy. The model is chosen with `-K`:

- `bigram` -- previous token to next token.
- `trigram` (default) -- previous two tokens, backing off to bigram then to a
  global unigram distribution.
- `predict` -- `libxs_predict` over the previous token IDs, `-P PROFILE`.
- `embed` -- `predict` over distributional token embeddings instead of raw IDs.
- `rerank` -- `libxs_predict` reranks the trigram's candidate successors,
  `-P PROFILE`.
- `hier` -- report hierarchical metatoken, contextual-byte, and exact byte-PPM
  BPC. This model is evaluation-only.

`CONVERSE_GRAN` accepts `word`, `native`, `syllable`, `bpe`, `meta-native`,
`meta-word`, or `meta-syllable`.

For `-K hier`, `CONVERSE_HIER_MINCOUNT` sets the minimum known-unit count and
`CONVERSE_HIER_CLOCK_ORDER` sets byte/state context order from 1 through 6
(default 2). `CONVERSE_HIER_STATE_DECAY` sets fixed recurrent-state decay from
zero up to, but not including, one. `CONVERSE_HIER_TOP_STRIDE` controls how
often PPM top-1/top-3 is evaluated (default 40; use 1 for every byte).
`CONVERSE_HIER_EXPERT_ORDER` sets the highest mixed PPM order (default 6),
while `CONVERSE_HIER_EXPERT_RATE` and `CONVERSE_HIER_EXPERT_SHARE` control the
online fixed-share mixer (defaults 0.15 and 0.005).

An optional `converse.predict` fixture of `context|expected-next` lines adds a
curated check to `-E`. The predictor profiles for `-K predict|embed|rerank` are
selected with `-P` (`raw`, `poly2`, `smooth`, `temporal`, `rf`, `fisher`,
`hknn`); see the paper for which profiles suit prediction and why.

## Local state files

The sample keeps all state in the sample directory and reuses corpus files by
refreshing existing entries instead of duplicating them. The following are local
and ignored by version control:

- `texts/` -- corpus files.
- `converse.dat` -- persisted corpus registry.
- `converse.par` -- parent texts the corpus refers to. Written beside the corpus
  and required with it: delete one and the other is rebuilt.
- `converse.lex` -- persisted lexicon and token IDs.
- `converse.prd` -- persisted answer reranker.
- `converse.eval` -- evaluation fixture.
- `converse.relations`, `converse.bridges`, `converse.predict` -- optional rule
  and fixture files.

The sample is intentionally experimental and does not add a stable public
summarization API.
