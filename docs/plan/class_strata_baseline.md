# The practitioner's design: confidence strata against map-class strata for a map's error rate (exp105 preregistration)

**Status: frozen on 11 October 2026, before any arm's interval was computed on the suite's units.** Written 11
October 2026. The owner chose the predictions' form on 11 October 2026 (P1: both a count of wins and a median width
ratio; P2: two thirds of the tasks), confirmed the count of gradable tasks (18 at 300 labels) and the branches'
reverse-win threshold (10 of the 18). Changed after two independent reviews and before freezing: B's allocation now carries the package's
floor (it is the shipped design), the median ratio is coverage-first like the win count, the branches read the
reverse win counts, and the disclosure of what is already known is corrected. The page is frozen before any arm's
interval is computed on the suite's units.

**Result, 11 October 2026 (added at recording; the frozen text is unchanged).** Run once at 76b7a3c (job 1287676),
recorded after an independent audit ([comparisons](../results/comparisons.md#the-practitioners-design-confidence-strata-against-map-class-strata-exp105)).
P1 holds, at its bar: B wins against C on 14 of the 18 tasks, median ratio 0.853; the count turns on B's coverage at
m-brick-kiln (0.941 here, about 0.934 pooled over 30 reruns, 7 of which gave 13). P2 holds: 16 of 18, median 0.863.
C wins against B on 4 and D on 2, so no branch changes a recommendation. Report-only: at 100 labels the confidence
design covers 0.913 and 0.933 on two 200-window AWF maps.

- **Allowed before freezing:** the map-class stratum sizes of each task (the map's classes only, no label: done on 11
  October 2026, below), the smoke on synthetic units.
- **Not allowed before freezing:** any interval, coverage or width of arms C, D or E on the suite's units.
- **Already known:** exp78 recorded a confidence-strata arm (its D2c/E1: Wald interval, Neyman allocation without the
  0.02 floor) and a Wilson simple random arm on the seven segmentation tasks at 300 labels only. exp85 recorded the
  same two arms on 15 of the 18 gradable tasks at 300 (all but the three Nandi tasks); that unfloored B covered below
  0.93 on four: m-eurosat (0.271), m-brick-kiln (0.890), m-so2sat (0.903) and CropHarvest China 6 S1+S2 (0.911).
  exp81 recorded the shipped confidence design's coverage of overall accuracy, and per-class user's accuracy under it
  and under a random sample, on all 24 tasks at 300 and 1,000. Arms C, D and E have never been run on these units.

## The question

The package's default design for "how wrong is the map?" stratifies by the model's confidence (`sample --design
confidence`: five margin quintiles, Neyman allocation from the model's own confidence). The record graded it against
simple random sampling only (exp78, exp81). What a practitioner does instead is stratify by the map's classes (Olofsson
et al. 2014; Stehman 2014), and Skakun (2025, *RSE* 326, 114805) gives the efficiency of map-class strata over simple
random sampling for area estimation in terms of the map's accuracies. Does the confidence design give a narrower valid interval for a
map's overall error rate than map-class strata, at the same budget, and is the gain from the strata themselves or only
from the allocation?

## The populations

exp78's per-unit exports of the embedding suite (`exp/out/exp78_units/`, OlmoEarth Base probe maps, 24 tasks): per
unit the map's class (`dec`), the margin, the top-1 probability (`p1`) and the error against the benchmark label (every
unit labelled, so each design is graded exactly against the true rate). The units are those exp78 exported (24
tasks; exp78 graded the seven segmentation tasks, exp81 all 24).

Map-class strata, counted label-free on 11 October 2026: 2 to 19 classes per task; the smallest class holds 1 unit
(AWF), 2 (m-cashew-plant) or 8 (MADOS), and 0.03% to 47% of a task's units.

## The arms (R = 2,000 draws per task, budget and arm; seed 105)

| Arm | Strata | Allocation | Interval |
|---|---|---|---|
| A | none (simple random sample) | — | the package's exact hypergeometric interval |
| B | five margin quintiles (the shipped `confidence` design) | Neyman from the model's confidence: spread sqrt(q(1 − q)), q = max(0.02, the stratum's mean 1 − p1), as `sample_for_estimation` allocates | `stratified_interval_wilson` |
| C | the map's classes | proportional to class size (Olofsson's stratified estimator) | `stratified_interval_wilson` |
| D | the map's classes | Neyman from the model's confidence, as B | `stratified_interval_wilson` |
| E (report-only) | the map's classes | equal, by water-filling (`equal_allocation`) | `stratified_interval_wilson` |

Every stratified arm uses the package's `neyman_allocation` or `equal_allocation` (a floor of two labels per stratum,
no stratum asked for more units than it has, so a class smaller than its share is labelled in full) and the package's
stratified estimator and variance. B and D differ only in the strata; C and D only in the allocation.

## Predictions (overall window error rate, 95% intervals; budget 300)

A task is gradable at a budget B when it has at least 2B units (at 300 labels: 18 of the 24 tasks; the three AWF tasks,
200 units, are below the budget and not run at 300, and the three CropHarvest Togo tasks, 306, are reported, not
graded). An arm **wins** a task against another when its coverage is at least 0.93 and either the other's coverage is
below 0.93 or its mean half-width is no larger. The **ratio** of B's half-width over another arm's is taken coverage
first in the same way: infinite on a task where B covers below 0.93, zero where only the other arm does.

- **P1 (the default against practice):** B wins against C on at least 14 of the 18 gradable tasks, and the median over
  those tasks of the ratio (B over C) is at most 0.90. Both are required.
- **P2 (the strata themselves):** B wins against D on at least 12 of the 18.
- **Reported, not graded:** budgets 100 and 1,000 (the same gradability rule); arm E; coverage, mean half-width, bias
  ratio, the variance of the estimates and the share of draws with a starved stratum (a one-unit class labelled in
  full is a census, not counted) of every arm; the reverse win counts C over B, D over B, D over C and C over D; each
  map class's user's accuracy with its Wilson interval from that class's labels under C and E, the use map-class
  strata are designed for; on the binary tasks, Var(C)/Var(A) predicted by Cochran's stratified variance from the
  population's class sizes and class error rates (on a binary map the same per-stratum variance as Skakun's area case;
  the simple random denominator here is the error rate's θ(1 − θ), not the class share's) beside the measured ratio.

## What would change the conclusion

- **P1 and P2 hold:** the default design is better than practice for the overall error rate, and because of its strata;
  the docs say so beside the map-class design's use for per-class accuracy.
- **P1 holds, P2 does not:** the strata are not shown to be the source of the gain. If D also wins against B on at
  least 10 of the 18 tasks, the docs recommend D's allocation over map-class strata as an option; otherwise nothing
  changes.
- **P1 does not hold:** the default is not shown better than practice by the preregistered margin. If C wins against
  B on at least 10 of the 18 tasks, the docs recommend map-class strata for the overall rate (`sample --condition`
  given the class raster splits labels equally; a `--design class` would allocate proportionally) and the record says
  so; otherwise no recommendation changes.

## What it cannot show

- Maps other than linear probes of one encoder on benchmark splits; the splits are not landscape samples.
- Area estimation of a class, which map-class strata are built for: the estimand here is the overall error rate.
- Labels taken as right; reviewer error is not modelled.
- Independence across tasks: the 18 gradable tasks are 12 label sets (PASTIS, CropHarvest China 6 and Nandi three
  times each, read from different inputs), so the counts are not over 18 independent maps.

## Outputs

`exp/out/exp105_summary.json` (per task, budget and arm: coverage, mean half-width, bias ratio, starved rate; the
predictions' verdicts). Script `exp/exp105_class_strata.py` (numpy only), run as a cluster CPU job pinned to a commit.
