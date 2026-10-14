# Integration with the OlmoEarth Agent

**Division of labour.** This repo owns the *evidence*: experiments, the
ledger, and a small library of pure functions that turn a prediction into an
assessment. The OlmoEarth Agent owns the *conversation*: it wraps those
functions as tools and narrates their summaries. The agent never generates
evidence itself.

Index at [../TECHNIQUES.md](../TECHNIQUES.md); recipe at
[recipe.md](recipe.md).

## With labels: `calibrate`

When the caller holds labels for a model, `calibrate.fit_ranker` and
`calibrate.fit_side` fit the label-free readings to them and return a `Fusion`
plus a held-out report (cross-fitted by tile or event; excess AURC, capture,
calibration, or share right against the raw rule, always-a, always-b and the
coin, with a sign test over groups). Inputs: named per-window readings, the
decisions, validity, labels, group ids, and the model family as a string.
Outputs: the fusion (JSON-serialisable) and the report. Contract: the fusion is
bound to the family it was fitted on and raises when asked to score another,
because a rule fitted on frozen heads does not transfer to a fine-tuned model
(exp59); an agent must refit per model and report the held-out numbers, not the
in-sample ones.

## As a tool call

An agent that cannot import the package can call the two commands and read the
JSON back: `oe-inferencex assess <scores> --out <dir>` returns the review sets
and the explanation, `oe-inferencex compare <a> <b> --out <dir>` the difference
of two decisions on identical windows, both label-free unless a reference or
label raster is given. The files are the contract: `assessment.json` and
`explanation.json` for the first, `comparison.json` for the second, each with
an `inputs` block naming what was read and a `files` block naming what was
written. The Usage page lists every field. An agent comparing two maps passes
the dates they describe (`--date-a`, `--date-b`) and narrates
`comparison.json`'s `dates.reading`: across dates a difference can be real
change on the ground, and the command refuses to say which map is right
against one reference unless `--labels-date` is given; with only one of the
two map dates it refuses to grade at all. The caution travels with the tool
rather than depending on the model knowing it.

The input-condition caution travels the same way. Without a condition layer,
`assessment.json`, the estimate JSON and the zone JSON carry `scope`: what a
whole-map ranking, rate or zone does not show when part of the map was read
from other inputs, as under cloud. A model run on an input combination it was
not trained on can be confidently wrong there, so its errors come late in the
order; a model trained on each input can still err more under one than under
another (exp88). The OlmoEarth Agent does not
pass a condition layer yet. Once it supports one, it should pass the layer as
`--condition` to `assess` and `sample` whenever the map records each pixel's
input condition. The outputs then rank, estimate and certify each condition on
its own (`conditions`, `per_condition`). A sample drawn with the layer is
certified per condition, and the zone's top-level fields are null.

## What this repo provides

| Module | Provides |
|---|---|
| `oe_inferencex.assess` | `assess_prediction` (logits or probabilities), `assess_classmap` (hard class map plus an exported confidence band), both with `order="confidence"` or `"boundary_first"` for the review sets (exp36) and an optional input-condition layer (`condition=`, exp88), `summary` (JSON-safe view) |
| `oe_inferencex.taskcard` | What task, legend, goal and audit settings a fine-tuned model has |
| `oe_inferencex.lcc` | HTTP range reader for the served change rasters |
| `oe_inferencex.metrics` | Tie-aware AURC, excess AURC, error capture at a budget, selective accuracy, ECE; torch-free |
| `oe_inferencex.signals` | The supported signals and controls as pure functions: confidence, prediction-boundary indicator, aligned tile-phase, NDWI-gradient and the other pixel controls, the U+ combination |
| `oe_inferencex.stats` | Exact sign tests, the one-vote-per-cluster test, sign-flip permutation, block and cluster bootstraps |
| `oe_inferencex.explain` | Why a review window is suspect: a library of label-free cues with their measured share among error and correct windows (`CUES`), `explain_review_set` (which cues fire on each window of an assessment's review sets, co-occurrence, the windows no cue explains), `cue_enrichment` (the validation: share among errors against share among correct, cluster bootstrap) |
| `oe_inferencex.compare` | How two inferences of the same scene differ, on the windows both predicted: `disagreement` (the rate pooled and per group, and the mask), `where` (each cue's share among the disagreement windows against its share among the agreement windows, and the ratio), `stability` (pairwise phi of several disagreement sets), `over_groups` (a per-group statistic as a distribution: median, share of groups where the sign flips, one-sided exact sign test); with labels `crosstab` (the errors one side corrects, the errors it adds, the errors both make, their phi) and `which_side` (how often each side matches the label where they disagree); `compare_inferences` (all of these in one summary, label-free unless labels are given) |

**Rule:** pure functions; no network except the task-card resolvers and the
raster reader. Arrays are returned, never serialized into text. `tests/`
checks these modules against the numbers recorded in `exp/out` (`uv run
pytest`), and against the experiment modules when the encoder is installed.

## What the agent provides

| Component | Location |
|---|---|
| Tools `olmoearth_task_card`, `olmoearth_assess_prediction` | `src/olmoearth_agent/tools/inference_assessment.py` |
| Skill 18 `olmoearth-inference-assessment` | vendored skills submodule |
| Egress capability `inferencex-configs` | agent config |

**Rule:** summary statistics only cross the tool boundary. Per-window
rasters, the flagged-window GeoJSON and the summary JSON are written under
`exports/` and referenced by handle. The five recipe caveats are returned
verbatim. A sixth, the input-condition caution above, is an agent-side change
not yet made: returning `scope` among the caveats, and having the agent's
certify tool read `per_condition` rather than the whole-map fields, belong to
the agent repository.

The agent imports this package lazily and reports `{"available": false}`
with install instructions when it is missing, so the agent runs without it.

## What the tool does with a prediction

1. **Confidence per window** — negative logit margin, or the exported band
   for the production case, pooled to 4-px windows.
2. **Boundary fraction per window** of the model's own class map — the
   triage cue.
3. **Review sets** at 1, 5 and 10% budgets, most suspicious first, with the
   boundary share of each set.
4. **With a reference map** — error rate, tie-aware AURC of confidence and
   of boundary, oracle and random, error capture at each budget, and the
   reference caveat (exp18).

5. **Why each review window is suspect** — `explain_review_set` lists, per
   window, the label-free cues that fire (on a prediction boundary; among
   the least confident 20%; and any cue the caller derives from other
   inputs, such as tiling instability, NDWI ambiguity, JRC seasonality),
   each quoted with its measured enrichment and the experiment that
   measured it (exp37 on identical windows), plus the co-occurrence of cues
   in the set and the windows no cue explains.

6. **Compare two inferences of the same scene.** `compare_inferences`
   takes two hard decision maps of identical windows (probabilities
   thresholded, class scores argmaxed, before the call), the validity mask
   of the windows both predicted, and optionally an id per window (tile,
   event), a label map and label-free cues of the same shape. It returns
   the disagreement rate pooled and per group, each cue's enrichment on
   the disagreement windows against the agreement windows, and the
   disagreement mask among the arrays; `stability` gives the pairwise phi
   of several disagreement sets, the test that two runs flag the same
   windows. With labels it adds the graded block: the cross-tab of the two
   error maps (the errors one side corrects, the errors it adds, the
   errors both make, and their phi), which side matches the label on the
   disagreement windows, and per group the same cross-tab with
   `over_groups` of the net correction. Everything before the graded block
   is label-free.

Nothing is fused; nothing is learned. The narration says which signal ranked
the windows, which cues explain each one and with what evidence, and which
caveats apply. The explanation layer returns structured evidence; the
sentences are templates, and the low-confidence cue is the ranker itself,
so inside a confidence review set it always fires.

## Is it measured?

Yes, since 2026-09-16: [../plan/agent_benchmark.md](../plan/agent_benchmark.md)
(exp64), forty cards, three tasks, a claims audit against the tool outputs,
recorded in [Comparisons](../results/comparisons.md#does-the-package-help-an-agent-the-preregistered-benchmark-exp64). Handed these tools, Qwen3.8-27B-NVFP4
reproduces the package's review set and declines the side question on every
comparison card, and the OlmoEarth Agent as shipped finds the tools on its own
and does the same. Two of the three preregistered predictions failed: a numpy
sandbox with the same arrays captures 0.904 of the package's
errors by itself and grounds 94.2% of its numbers, so the package does not add
the ranking to a model of that strength. The third holds as graded, and it does
not show what the package adds either: an arm with no package and no rasters
also declined to pick between two maps on every comparison run, and the grader
scores any decline as full marks. What separated the tool arm from the sandbox
was reliability, which was not preregistered: the sandbox gave no gradeable
answer in 27 of 120 runs, and its stated cues held on 86.2% of its windows
against 99.5%.

## Status (2026-09-16)

**Agent side.** Skill #18 `olmoearth-review-set` is on the agent repository's
branch `2imi9/feature-inferencex-review-set`, merged into main on 2026-09-18 (pull request 155): four tools
(`olmoearth_review_set`, `olmoearth_compare_review`, `olmoearth_grade_review_rule`,
`olmoearth_review_budget_ceiling`), scores taken inline or from a `.json` file
under `OLMOEARTH_SCORES_ROOT`, row and column returned with a grid; 645 tests
passing there. exp72 showed the port bit-exact against this package on real
inference. The agent's LLM client also gained a defensive decode for
double-encoded tool arguments, which the 7B pilot exposed.


**Trial through Studio (2026-09-24).** The shipped agent (main at a26a5c7) was run on four briefs against the
user's own Studio project with Qwen3.8-27B-NVFP4, as an outside user would. It lists models and declines to say
which of two maps is right without labels, as exp64 measured. Three gaps came out, each with its fix assigned: the
comparison tool counted Studio's no-data value (−1) as data, which turned a correlation of −0.017 into 0.946; no
tool turns a Studio prediction into the review set's input, so the model ranked by hand and put the most confident
windows first; and without this package's `sample` and `estimate` the model quoted a simple-random-sample interval
for a targeted design. Studio returns no probabilities or raster download for these models, only map tiles and a
point lookup. The log, with each run and its attribution (harness, tool, model, Studio or setup), is
`exp/out/agent_trial_2026-09-24.md`.
