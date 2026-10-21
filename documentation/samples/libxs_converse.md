# Converse Sample

Converse reads plain text or Markdown files and then answers questions about them,
summarizes them, assembles new text from them, and trains small next-token models on
them. Answers are extracted from the ingested text and carry a citation, or the sample
abstains; nothing is generated from outside the corpus unless you ask for it.

It is built only on libxs primitives (lexemes, metatokens, fingerprints, registries,
n-grams, parameter prediction) and is self-contained: nothing is downloaded, no
external service is contacted, and no data is used beyond the files you give it. It
is experimental and adds no stable public API.

## Build

```bash
cd samples/converse
make PEDANTIC=2
```

This builds `converse.x`, `converse-qa.x`, `converse-lm.x`, `converse-norm.x` and
`summarize.x`. Without a built library the sample compiles header-only; with one
(`make PEDANTIC=2` in the `libxs` root first), it links it. From the `libxs` root,
`make PEDANTIC=2 samples/converse/` does the same.

## Which binary

| binary | serves | use it for |
| :--- | :--- | :--- |
| `converse-qa.x` | interactive, `-e`, `-c`, `-L` | questions, evaluation, recombination |
| `converse-lm.x` | `-E`, `-c -K KIND`, `-L` | next-token models, byte model |
| `converse-norm.x` | prompt, `-e`, `-t`, `-r`, `-m`, `-L` | spelling normalization |
| `converse.x` | everything above but normalization | anything, and all documented examples |

`converse.x` accepts every command below. The split binaries link only what they
serve, and a mode a binary does not serve is rejected up front, naming the one that
does. The recombination probe fills its bpc columns only under `converse.x` with
`CONVERSE_HIER_RESCORE=1`; otherwise it prints `no seam judge` and leaves them at
zero.

Run any binary without arguments to print its options and the main environment
variables.

## Getting text in

Put files anywhere; `texts/` is the conventional place and is ignored by git. Two
structures are understood:

- **prose** — paragraphs of running text, blank-line separated. Uppercase lines are
  read as section headings and become the citation for everything under them.
- **Markdown** — headings, paragraphs and code blocks; `#` headings become the
  citation.

The structure is chosen by file extension (`.md` is Markdown, everything else is
prose) and can be forced with `-p prose|markdown`.

Pass files directly, or use `-b PREFIX` to pick up a numbered set without listing it:

```bash
./converse.x texts/prose1.txt texts/prose2.txt
./converse.x -b texts/grimm                  # texts/grimm.txt, grimm-1.txt, ...
./converse.x -b docs ../../documentation/*.md
```

`-b PREFIX` probes `PREFIX`, `PREFIX.txt` and numbered parts (`PREFIX-N.txt`,
`PREFIX_N.txt`, `PREFIX.N.txt`); missing siblings are fine as long as one file is
usable. Markdown files are not probed, so pass them explicitly (a glob is fine, as
above).

`-b` also names the **state files**: the basename of the prefix is the prefix of
everything the run persists, so `-b docs` keeps `docs.dat`, `docs.lex` and so on, and
several corpora can live side by side in one directory. When `-b` is given more than
once, the first one names the state. Without `-b`, state is named after the current
directory (`converse.*` when run from the sample directory).

Any UTF-8 text works. Suitable sources are Project Gutenberg books, a Wikipedia
extract, your own notes, or the documentation of this repository. Larger corpora are
supported; memory is the practical limit.

### Warm and cold runs

State is reused: a second run over the same files loads the persisted corpus, lexicon
and reranker instead of re-ingesting, which is much faster. To force a full rebuild,
delete the state files for that prefix:

```bash
rm -f grimm.dat grimm.par grimm.src grimm.lex grimm.prd grimm.facts
```

Do this before comparing two runs — a warm run answers from state built by the
previous one.

To build the state without asking anything — useful before a batch of runs, or to
time ingestion on its own — use `-L`:

```bash
./converse.x -L -b texts/grimm
```

## Asking questions

```bash
./converse.x -b texts/grimm
```

The sample then reads one question per line:

```text
> Who is Gretel?
Gretel is the girl.
citation: HANSEL AND GRETEL (texts/grimm.txt:2544)
> Where did Hans go?
Hans went into the stable.
citation: CLEVER HANS (texts/grimm.txt:6031)
> Who was eaten by the wolf?
Grandmother and Little Red-Cap were eaten by the wolf.
citation: LITTLE RED-CAP [LITTLE RED RIDING HOOD] (texts/grimm.txt:3104-3112)
> What belongs to Curdken?
hat belongs to Curdken.
citation: THE GOOSE-GIRL (texts/grimm.txt:1771)
> What do we know about Hansel?
Hansel is the boy. Hansel stood still and peeped back at the house.
citation: HANSEL AND GRETEL (texts/grimm.txt:2544-2579)
> Who is Sherlock Holmes?
I do not know from the corpus.
```

What is answerable depends on the rule files (next section). With the shipped rules
the sample answers:

- `Who is X?` — an identity or a definition ("X is a Y", "X, a Y, ...").
- `Where did X go?` / `Where is X?` — a place, as a proposition from X's own sentence.
- `Who was V by X?` — a relation, including passives it was never told about.
- `What belongs to X?` — an enumeration, each item cited.
- `What do we know about X?` — several cited propositions collected about one name.
- `How are X and Y connected?` / `What connects X and Y?` — the path between two
  entities, stated as the propositions it is made of.
- anything else question-shaped — the best matching sentence, ranked and cited.

Every answer is followed by a citation naming where it came from: the section title
when the corpus has headings, and always the file and the line. An answer assembled
from several propositions cites the range of lines it rests on
(`texts/grimm.txt:2544-2579`), one range per file, and a single line when it is one
line. Titles are only printed when the text really carries them — a corpus of flat
prose is cited by file and line alone rather than by a sentence that happened to look
like a heading.

A connection question is answered by stating the facts that join the two entities,
each with its own citation, and never by inventing a relation between them:

```text
> How are McClellan and Douglas connected?
Lincoln restored McClellan. Lincoln forced Douglas.
citation: texts/wiki2m.txt:813-895
```

If nothing in the corpus joins them, the sample says so rather than answering with a
sentence about one of them.

Three behaviours to expect:

- The sample **abstains** (`I do not know from the corpus.`) rather than guessing when
  the corpus does not support an answer.
- A question whose KIND it does not recognize — one using none of the declared `ask|`
  words — still gets an answer, introduced by `I did not recognize the question. The
  closest the corpus comes:` so a relevant sentence is never mistaken for a direct
  answer. That is different from abstaining: the question was not read, rather than
  read and unsupported.
- A question of the form `In TITLE, ...` is ranked only against the matching heading,
  which is how to ask about one document in a corpus of many.

Questions are case-insensitive.

Non-question input is treated as a composition request and answered from the
fingerprint path instead. `-n N` sets how many sentences a response may use (default 1).

## Checking answers against a fixture

`-e` runs a fixture of questions and expected terms and prints a pass count, which is
how to tell whether a rule-file or corpus change helped:

```bash
./converse.x -e -b texts/grimm
```

The fixture is `<prefix>.eval` (so `grimm.eval` for `-b grimm`). Each non-comment line has three required fields and two optional ones:

```text
question|evidence-terms|reply-terms|fact-terms|citation-terms
```

The fifth field checks the citation, and it is matched against the whole citation —
title, file and line — so a fixture can assert `texts/grimm.txt:2544`.

Terms are comma-separated and matched case-insensitively. An empty evidence field
marks an abstention case (the question SHOULD go unanswered); an empty reply field
skips the concise-reply check. The fact field is checked only when relation facts were
learned, so one fixture works with and without rule files.

`-P PROFILE` selects the profile used to rank candidate answers (`raw`, `poly2`,
`smooth`, `temporal`, `rf`, `fisher`, `hknn`), for both interactive use and `-e`:

```bash
./converse.x -P temporal -e texts/prose1.txt
```

## Normalizing spellings

`converse-norm.x` reads the corpus vocabulary as a codebook and maps a surface
form to the nearest word in it, so whatever it answers is a word the corpus
contains. With no mode it prompts, one word per line, and prints the decision
together with the words it chose between:

```bash
./converse-norm.x -b texts/grimm
```

```text
> rabbet
rabbet -> rabbit  (distance 1)
    rabbit (261)
> quene
quene: abstained, nearest at 2 with 4 tied
    queen (373)
    quite (242)
```

`-r` sets the largest edit distance a correction may span (default 1) and `-m`
how far the runner-up must be beyond the chosen word (default 1, which accepts
everything). `-e` reports how densely the vocabulary packs and how often a
corrupted word is recovered, which is how to pick the two:

```bash
./converse-norm.x -e -r 1 -m 2 -b texts/grimm
```

A wrong correction silently replaces the word that was typed: a larger `-r`
recovers more but corrects wrongly more often, and a larger `-m` abstains more
often. Run `-e` on your own corpus before trusting a setting.

`-t` reads surface forms from a file, one per line, and writes the accepted
corrections to `<prefix>.norms`:

```bash
printf 'rabbet\nqueeen\n' > forms.txt
./converse-norm.x -t forms.txt -b texts/grimm
```

Every binary loads that table on its next run and applies it while encoding, so a
question asked with a corrected spelling answers as the corrected one would.
Forms that decode to themselves, and those with no word in reach, are left out.

## Summarizing and composing

`summarize.x` is a separate, smaller entry point:

```bash
./summarize.x texts/prose1.txt              # fuse adjacent sentences
./summarize.x -n 3 texts/prose1.txt         # to a target sentence count
printf 'First sentence. Second one.' | ./summarize.x -
./summarize.x -g -r -n 8 texts/prose1.txt texts/prose2.txt   # compose
```

- default: summarize one file by repeatedly fusing adjacent sentences. A single
  argument that is not a readable file, or several arguments, are summarized as
  literal text (joined by spaces); no argument or `-` reads stdin.
- `-n N`: target sentence count; without it, fusion continues while a pair can be
  fused and each fusion is printed.
- `-g`: compose mode — ingest all files, take the first as the target, and emit a
  short assembled text (`-n` is the phrase budget, default 12). The ingested
  corpus is kept in `summarize.dat` in the working directory, and the next run
  loads and extends it.
- `-r`: reflow the input first, joining cosmetic line breaks (useful for Gutenberg
  files); in compose mode only the target is reflowed.
- `-h` (or `--help`) as the first argument: print the usage.

Progress and diagnostics go to stderr, the result to stdout.

## Recombining text from the corpus

An experimental mode splices clauses from different sentences into new ones, keeping
each half verbatim and cutting only at clause boundaries:

```bash
CONVERSE_RECOMB=50 CONVERSE_HIER_RESCORE=1 ./converse.x -E -x -b texts/grimm
```

`CONVERSE_RECOMB=N` attempts N joins (50 when N is not positive) and reports how
many were made, with coherence and seam statistics. `CONVERSE_HIER_RESCORE=1`
trains the hierarchical byte model (`-K hier`) and uses it to score each seam (the
bpc columns) and to reorder candidate answers; it takes effect only in the binaries
that link that model (`converse.x`, `converse-lm.x`). `CONVERSE_RECOMB_COMPOSE=1`
additionally lets the interactive `-c` mode answer with a spliced sentence. It is off by default, since it emits text
that does not occur verbatim in the corpus.

## Next-token prediction

An experimental predictor is trained from the token stream and kept out of the
question-answering path:

```bash
./converse.x -E -b texts/grimm              # accuracy, default trigram
./converse.x -E -K bigram -b texts/grimm    # pick the model
./converse.x -E -H 10 -b texts/grimm        # held-out: train on 9/10, test on 1/10
printf 'the little\n' | ./converse.x -c -b texts/grimm   # suggestions + continuation
CONVERSE_GRAN=meta-word ./converse.x -E -b texts/grimm   # token unit
```

- `-E` reports accuracy, `-c` prints the top few successors plus a short continuation.
- `-H N` evaluates on a held-out split, `-T PREFIX` on a separate corpus, which is
  what to use for an honest, non-memorized figure.
- `-K` selects the model: `bigram`, `trigram` (default), `predict`, `embed`,
  `rerank`, `knnlm`, or `hier` (evaluation-only: metatoken, byte and PPM bits per
  character).
- `-P PROFILE` selects the predictor profile for `predict|embed|rerank` (`raw`,
  `poly2`, `smooth`, `temporal`, `rf`, `fisher`, `hknn`).
- `-x` uses the deepest n-gram context.
- `CONVERSE_GRAN` sets the token unit: `word`, `native`, `syllable`, `bpe`,
  `meta-native`, `meta-word`, `meta-syllable`.

An optional `<prefix>.predict` fixture of `context|expected-next` lines adds a curated
check to `-E`. Many further knobs exist; run the binary with no arguments to list them.

## Teaching it a language and a corpus

Everything corpus- and language-specific lives in local rule files rather than in the
source, and they are what decide which questions can be answered. Two files, both
optional:

- `converse.rules` — the LANGUAGE: function words and syntactic classes. The English
  file is part of the repository. A `<prefix>.rules` file REPLACES it (for a corpus in
  another language).
- `<prefix>.relations` — this CORPUS: aliases, role words, place names. It EXTENDS
  the language file, so it need not restate function words.

Each non-comment line is `kind|term`:

| kind | declares | example |
| :--- | :--- | :--- |
| `alias` | a query verb and the verb the text uses | `alias\|eaten\|devoured` |
| `person` | a role word that can be bound to a name | `person\|grandmother` |
| `place` | a place noun | `place\|forest` |
| `where`, `why`, `how` | the markers those questions turn on | `where\|in` |
| `topic` | the marker of an attribute question | `topic\|about` |
| `own` | the verb a possession question uses | `own\|belongs` |
| `poss` | how the language WRITES possession | `poss\|apostrophe-s` |
| `copula`, `article`, `prep` | the syntactic classes definitions need | `copula\|is` |
| `aux` | the auxiliaries | `aux\|had` |
| `agent` | the word a passive names its agent with | `agent\|by` |
| `genitive` | the word that marks a possessor after the thing | `genitive\|of` |
| `join` | how the language writes a joiner inside a name | `join\|ampersand` |
| `link` | the words that ask how two entities relate | `link\|connected` |
| `ask` | a question KIND and this language's word for it | `ask\|who\|who` |
| `pron` | the back-reference pronouns a follow-up points with | `pron\|it` |
| `result` | the light verb introducing a resulting state | `result\|made` |
| `skip`, `negate` | filler words and negators | `skip\|the` |

A few notes that matter in use:

- `where|in` plus `place|forest` is what makes `Where ...?` answerable; declaring only
  one of the two answers nothing.
- `aux|had` and `agent|by` are enough to read passives and active clauses generally —
  the sample derives which words the corpus uses as verbs and as nouns from those
  frames and reports `verbs derived: N` and `nouns derived: N`. No verb list is needed.
- `poss|apostrophe-s` (English) against `poss|apostrophe` (German): declare the wrong
  one and possession answers go silent rather than wrong.
- `ask|` is how the sample knows a question's KIND. The first field is the kind (`who`,
  `what`, `where`, `when`, `why`, `how`, `yesno`), the second is the word — so a German
  rule file writes `ask|who|wer`. A question using none of the declared words still gets
  an answer, prefixed with a note that the question was not recognized.
- `person|father` plus `genitive|of` is what makes kinship answerable in BOTH forms —
  "Lincoln's father Thomas" and "Aegeus, the father of Theseus". A role word you do not
  declare is simply not read, so extend the `person|` class for the kinds of relation
  your text states.

The sample can also PROPOSE rules instead of only reading them:
`CONVERSE_RULES_LEARN=N` offers up to N `person|` class members found in the corpus,
which is a way to bootstrap a rule file for new text. What it proposes is speculative,
so it is reported separately and, if a `<prefix>.learn.eval` fixture exists, that
fixture is used for `-e` while learning is on.

After ingestion the sample reports what it learned (`relation facts: N learned`,
`identity facts`, `location facts`, `type facts`). Set `CONVERSE_FACTS_LIST=1` to
print the facts themselves, plus a `graph reach:` line (edges, entities, pairs joined
by one middle, largest degree). Few edges means connection questions will mostly go
unanswered on that corpus.

Optional `<prefix>.bridges` adds evidence-backed answer frames, five fields per line:

```text
name|query-groups|evidence-groups|score|reply
```

Whitespace separates required groups and `/` separates alternatives; `_` stands for a
literal space in evidence terms. The reply is literal text or a small frame such as
`{after:lighthouse had}`.

## Environment variables

The variables used above, all unset by default:

| variable | effect |
| :--- | :--- |
| `CONVERSE_PREDICT=1` | train the answer reranker and save it as `<prefix>.prd` (skipped otherwise) |
| `CONVERSE_GRAN=UNIT` | token unit for next-token models (`word` by default) |
| `CONVERSE_NGRAM_ORDER=N` | n-gram context order 1–6 (default 2; `-x` selects 6) |
| `CONVERSE_RECOMB=N` | run the recombination probe over N joins |
| `CONVERSE_RECOMB_COMPOSE=1` | let `-c` answer with a spliced sentence |
| `CONVERSE_HIER_RESCORE=1` | score seams and reorder answers with the hierarchical model |
| `CONVERSE_RULES_LEARN=N` | propose up to N `person\|` class members as rules |
| `CONVERSE_FACTS_LIST=1` | print the learned facts and the `graph reach:` line |

Running a binary without arguments lists these and the further tuning variables
of the next-token models.

## Files the sample writes

All state is kept in the working directory, named after the `-b` prefix (after the
directory without one), and all of it except `.norms` is ignored by git:

| file | holds |
| :--- | :--- |
| `<prefix>.dat` | the ingested corpus |
| `<prefix>.par` | the source texts the corpus refers to; required with `.dat` |
| `<prefix>.lex` | lexicon and token IDs |
| `<prefix>.prd` | the answer reranker, trained only with `CONVERSE_PREDICT=1` |
| `<prefix>.src` | the file names the corpus refers to, for citations |
| `<prefix>.facts` | the derived fact layers, rebuilt when the corpus changes |
| `<prefix>.norms` | spelling corrections, written by `converse-norm.x -t` |

Delete one of `.dat` / `.par` and both are rebuilt. The fixtures and rule files
(`<prefix>.eval`, `<prefix>.learn.eval`, `<prefix>.predict`, `converse.rules`,
`<prefix>.relations`, `<prefix>.bridges`) are inputs you write; the sample never
modifies them.
