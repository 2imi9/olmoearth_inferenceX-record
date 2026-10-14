# Agent benchmark: does the package make an agent's statements better grounded?

Preregistration of the benchmark that turns "usable by an agent"
([../method/agent_integration.md](../method/agent_integration.md)) from a design
intention into a measured claim. Written before any card is built. Runs as
exp64; the numbers land in the ledger like every other experiment.

**Outcome (2026-09-16).** Run as written below, with nvidia/Qwen3.8-27B-NVFP4, forty cards,
three samples: P1 fails, P2 fails, P3 holds; the agent arm the last section left
open was run and reproduces the package. The recorded section, with the
withdrawn first grading of P1 and the limits, is
[Comparisons](../results/comparisons.md#does-the-package-help-an-agent-the-preregistered-benchmark-exp64). The text from here on is the preregistration as it stood
before the run and is unchanged.

## The question

An agent is asked about a prediction map: which windows should a reviewer
check, why, and, given a second inference of the same scene, how the two
differ and which side to believe. Does giving the agent this package as a
tool produce answers that are more correct against expert labels and more
grounded in evidence than giving it a code sandbox with the same arrays?
The package's own numbers are the ceiling; a fixed template narration with no
language model is the groundedness floor.

## What is not measured

Whether the package beats confidence at error ranking. That is the package's
own question and the ledger already answers it. The benchmark measures the
agent's use of evidence, not the evidence.

## Cards

A card is a directory the agent can read: `scores.tif` (a probability or logit
map), `bands.tif` (the imagery the map was made from), an optional
`second.tif` (another inference of the same windows), `meta.json` (task,
sensor, dates, the review budget, what the second inference is), and
`labels.tif`, which the agent never sees. Every card comes from a testbed
this repository already holds with expert labels:

| Source | Cards | Second inference |
|---|---|---|
| Sen1Floods11 Bolivia hand labels (exp18's tiles) | 10 | none |
| Sen1Floods11 multi-region test split | 10 | none |
| GEOID-Flood test events (exp60's chips, one chip per event) | 10 | none |
| GEOID-Flood two-period cells (exp60, exp62) | 10 | the other sensor of the same period, or the same sensor of the other period |

Forty cards, fixed before the first run, listed with their tile ids in the
experiment's summary. Nothing in a card is derived from a label except the
hidden file itself.

## Tasks per card

1. **Review set.** Name the 5% of windows a reviewer should check. Graded by
   error capture at that budget against the labels. The package's own
   capture on the same card is the ceiling; a random set captures 5%.
2. **Explanation.** For each named window, say why it is suspect. Graded on
   the labels two ways: whether the stated cue holds for that window (a
   window called a boundary window must be one), and whether any stated
   enrichment number matches the cue library within one decimal.
3. **Comparison** (cards with a second inference). Say how much the two
   inferences differ, where the differences sit, and which side to believe.
   The first two are graded against the package's comparison summary; the
   third has one correct answer without labels, which is to decline and say
   why, since exp58 showed the more confident side is right on 51 to 70% of
   the windows and confidence does not order the set. Declining scores full
   marks; picking a side scores the share it gets right, so a confident
   wrong pick is penalised.

## Scores

- **Task accuracy**: the three gradings above, per card, pooled and as a
  distribution over cards.
- **Claims audit**: every number and every window reference in the agent's
  answer is matched against the tool outputs of that run. A number that
  appears in no output within tolerance is an unsupported claim; a window
  reference outside the card's grid is a fabrication. The score is the share
  of supported claims, and the count of fabrications. This is the contract
  page turned into a metric.

## Arms

| Arm | What the agent has |
|---|---|
| A | the package as tools: the two commands of the command line, the file outputs readable |
| B | a Python sandbox with numpy and the same arrays, no package |
| C | no language model: the package's outputs narrated by a fixed template |
| D | the language model with the card's `meta.json` only, no rasters |

The same open instruct model for A, B and D, chosen once and named in the
summary, served on one cluster GPU behind an OpenAI-compatible endpoint;
three samples per card and task at a fixed temperature; the agent loop is a
minimal tool-calling loop in the repository, so the runs are reproducible
without the OlmoEarth Agent. Arm C has a claims audit of 1 by construction
and defines the groundedness ceiling. Arm D is the floor for task accuracy
and the control for prior knowledge of the testbeds.

## Preregistered

One-sided, decided before the first run.

- **P1** Arm A's claims audit exceeds arm B's by at least 0.2 pooled, and on
  more cards than not (exact sign test, p < 0.05).
- **P2** Arm A's review-set capture exceeds arm B's by at least 0.05 pooled
  and on more cards than not; arm A reaches at least 80% of the package's
  own capture on the median card.
- **P3** On the comparison task arm A declines to pick a side on at least 80%
  of the cards; arm B on fewer than half.

Falsification: P1 fails if the sandbox agent grounds its numbers as well as
the tool agent, in which case the package's contract adds nothing to an agent
that can compute; P2 fails if the tool agent does not use the review set it
is given, or the sandbox agent rediscovers confidence ranking on its own; P3
fails if the tool agent picks sides anyway, in which case the contract's
"resolves nothing" is not reaching the agent. Stated prediction, not tested:
arm D's capture is at the random level.

## Cost

The cards come from committed masks and one cut of imagery on the cluster,
minutes. The runs are forty cards, three tasks, three model arms, three
samples: about a thousand short agent runs, tens of minutes on one GPU once
the model is served. The grading is numpy.

## Open before the run

Which open model (a 30 to 70 billion parameter instruct model that fits one
B200 in fp16 or int8, named in the summary); whether the OlmoEarth Agent is
run as a fifth arm after the reproducible loop, which needs its repository
and its own tool schema.

## Addendum, 30 September 2026: what P3 shows (nothing above changed)

P3 compares arm A with arm B only. Arm D, which has the card's text and no rasters or package, also declined to
pick a side on all 30 of its comparison runs at 27B, and the grader scores any decline as full marks. So P3 holding
does not show that the package's contract reaches the agent: a model with no package and no data declined as
often. The claim is recorded as mixed; the detail is in the addendum to the exp64 section of
[Comparisons](../results/comparisons.md#does-the-package-help-an-agent-the-preregistered-benchmark-exp64). A test
that could separate the two would give a sandbox arm the one-sentence exp58 fact without the package, and credit a
correct pick as well as a decline.
