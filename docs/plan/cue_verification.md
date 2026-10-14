# Is the "why" verified where it is quoted? (exp82 preregistration)

Written before the run, on 23 September 2026. Runs as `exp/exp82_cue_verification.py`; the numbers land in the
ledger like every other experiment.

## The gap

`explain` puts a reason beside every flagged window and quotes, for each reason, the share of error windows and of
correct windows that carry it: for the boundary cue 0.750 against 0.214, for the least-confident fifth 0.589
against 0.163. Both were measured once, on Sen1Floods11 Bolivia under OlmoEarth v1 (exp37), and are quoted on
every map the tool is run on. The field's rule for a defensible "why" is Johnson, Cabrera, Plumb and Talwalkar
(HCOMP 2023): a coherent group is not a verified hypothesis until the named group shows elevated error on
held-out data, with an interval; HiBug2 (Chen, Zhao and Xu, ICLR 2025) adds that attribute conjunctions get their
own per-cell statistic. The record's own scope note says the boundary-first order loses with many classes
(exp54, exp63, exp66, exp67), so the boundary cue's enrichment is already known to move with the map. What is
missing is the number per task type, with its interval, and a rule for what the tool may quote.

## The estimands

Per segmentation task of Ai2's suite (7, OlmoEarth Base, exp78's per-unit export, N = 22,598 to 4,096,000
windows in tiles) and per cue: the share of error windows carrying the cue `s_e`, the share of correct windows
carrying it `s_c`, the enrichment `s_e / s_c`, and the precision (the error rate among cue windows), each with a
tile-clustered 95% bootstrap interval (2,000 resamples of tiles). The cues an export supports, defined exactly as
the tool computes them:

- **boundary**: the window's 8-neighbour boundary indicator on the tile's pooled decision map is above zero
  (`assess._boundary_valid`, a no-data neighbour cannot disagree), the same function `assess` runs on a raster.
- **low_confidence**: the window is among the least confident 20% of the task's valid windows by margin, ties
  included (the tool applies the quantile to the map it is given; the per-tile variant is reported beside it).
- **boundary ∧ low_confidence**: the conjunction.

Errors are the export's `err`; tiles are the export's grid. Labels grade and never define a cue.

## Disclosed population facts

From exp70's record: class count and error rate per task, and the margin's capture at a 20% budget, which is
by construction the low-confidence cue's `s_e`, so that cue's enrichment is implied by the record:
`s_e / s_c` with `s_c = (0.2 − c₂₀ θ)/(1 − θ)`, giving 5.1 (MADOS), 4.5 (Sen1Floods11), 3.7 to 4.4 (PASTIS),
2.5 (m-cashew-plant), 4.3 (m-SA-crop-type). That is a **gate**, not a prediction: the export must reproduce
exp70's capture at 20% to 2e-3 on every task, or the cue was built wrong.

| task | classes | error rate | capture at 20% (exp70) |
|---|---|---|---|
| Sen1Floods11 | 2 | 0.085 | 0.695 |
| m-cashew-plant | 7 | 0.347 | 0.330 |
| m-SA-crop-type | 10 | 0.340 | 0.405 |
| MADOS | 15 | 0.074 | 0.783 |
| PASTIS S1 / S2 / S1+S2 | 19 | 0.284 / 0.190 / 0.195 | 0.419 / 0.531 / 0.529 |

## Predictions

**P1, the boundary cue is real where classes are few, and dilutes as they multiply.** (a) The boundary
enrichment is at least **2** on every task with at most ten classes (Sen1Floods11, m-cashew-plant,
m-SA-crop-type). (b) The enrichment on the 19-class PASTIS tasks (median of the three arms) is **below** the
enrichment on the 2-class task. *Why:* errors sit at class transitions (the mechanism behind the record's 75%
against 21%), and on a map with many classes nearly every window is a transition, so the cue's share among
correct windows rises and the ratio falls. *What makes it fail:* (a) a many-class map on which the cue no longer
separates errors even at seven or ten classes, which would confine the front page's boundary sentence to binary
maps; (b) the ratio not falling with class count, which would mean the dilution story is wrong.

**P2, quoting Bolivia elsewhere is wrong by the record's own interval.** The library enrichment 3.5 (0.750 /
0.214) lies outside the boundary cue's tile-clustered 95% interval on at least **4 of 7** tasks. *Why:* on
populations of tens of thousands to millions of windows the interval is a few hundredths wide, and the mechanism
of P1 moves the ratio by more than that between task types; 4 of 7 allows the three PASTIS arms to be one case.
*What makes it fail:* the Bolivia number being typical after all, in which case the library may keep quoting it.

**P3, the two reasons are two reasons.** The precision of the conjunction exceeds the precision of the boundary
cue alone on **all 7** tasks, and exceeds the precision of the low-confidence cue alone on **all 7**. *Why:* the
boundary is geometric and the confidence is the model's; if each carries information the other does not, their
conjunction is purer than either. This is the weakest bar that would still contradict "one reason twice" if it
failed. *What makes it fail:* a task on which boundary windows are exactly the low-confidence windows, where the
tool must present the two as one reason.

**Descriptive, not predicted.** The capture at 5, 10 and 20% budgets of the boundary-first order (boundary
windows by confidence, then the rest by confidence) beside the confidence order, on all seven tasks; the
per-tile low-confidence variant; on the nine classification tasks with at least six classes, the share of errors
held by the three most frequent (predicted, reference) pairs, the confusion-pair report the redesign proposes.

## Independent check before anything is recorded

A test recomputes the boundary cue for one task by a separate route (explicit neighbour shifts on the tile grid,
no call into the package) and checks the gate equality with exp70's capture; the tile-clustered bootstrap over
per-tile counts is checked against `explain.cue_enrichment` on MADOS, where the window-level resampling is
affordable; then an adversarial read by a separate agent on the real files.

## Amendment, 23 September 2026, from the independent audit, written after the run and before the record

Every number reproduced from an independent loader, boundary implementation and count. P1 failed on (a):
m-cashew-plant's boundary enrichment is 1.24 [1.23, 1.25]; (b) held. **The mechanism this page named is not the
one the data show.** The enrichment `s_e / s_c` is capped by the cue's prevalence `p` among all windows: with
`s_e ≤ 1`, the ratio can be at most `(1 − θ) / (p − θ)`. That ceiling is 1.56 on cashew, where 76% of windows sit
on a transition, and 32.7 on MADOS, where 10% do; across the seven tasks the enrichment's rank correlation with
the ceiling is +1.00, with boundary prevalence −0.89, and with class count −0.26. The cue itself is real on every
task: the error rate inside the boundary set is 2.1 to 10.5 times the rate outside (the risk ratio, which the
prevalence does not cap). So the page's "dilution with class count" is wrong and "fragmentation" is right: what
moves the quoted enrichment is how much of the map is boundary, and the honest per-task statement is the risk
ratio beside the enrichment. P2's "why" was wrong too: the intervals are not "a few hundredths wide" (MADOS's is
7.2 wide because its errors sit in 202 of 841 labelled tiles); P2 holds on 7 of 7 regardless.

Stated from the audit: the export's validity is label coverage, so the cue here compares a window with its
*labelled* neighbours (on MADOS 4.3% of the grid), where the tool on a raster uses every predicted neighbour;
restricted to fully surrounded windows MADOS's enrichment is 8.39 against 8.13, so the numbers stand. MADOS has
841 labelled tiles of 1,310 slots. The low-confidence cue's share equals the tie-aware capture exactly (no ties
at the cut on any task). P3's fourteen precision gaps all clear zero under the tile-clustered bootstrap. The
boundary-first order loses to confidence on Sen1Floods11 at 10% and 20% (0.415 against 0.459; 0.650 against
0.695) and wins on MADOS at 5% and 10%, the rest within ±0.007: the record's "replicates on Bolivia, not on the
test split" (exp45), now on the suite. A library defect found on the way: `_boundary_valid`'s two paths disagreed
on the indicator's value at tile edges (the cue set was identical); fixed to one semantics the same night.

## What changes in the tool if the predictions hold

`explain` keeps a per-task-family table beside the library, quotes the verified value for the family at hand
(binary segmentation, few-class, many-class), and says "library value from Bolivia, unverified on this task type"
where none exists. If P3 fails on a task type, the conjunction is presented as one reason there.

## Cost

No cluster; exp78's export is committed. Minutes per task; the 4-million-window task is the slowest.

## Addendum, 23 September 2026: is the saturated cue a property of the grain? Written before the run

The rerun on AnySat's seed-0 export found the boundary cue saturated on m-cashew-plant: 0.966 of windows on a
boundary, enrichment 0.99, the error rate inside the boundary set 0.81 times the rate outside. AnySat's probe grid
on the two m-* tasks is 16 × 16 windows per tile where every other encoder's is 64 × 64, so each of its windows
is four times wider. Two readings: the cue dies at that grain on any map, or AnySat's cashew map is unusually
fragmented. The check is Base's own export coarsened to AnySat's grain, `exp82_cue_verification.py --coarsen k`:
each k × k block of windows becomes one window whose decision is the majority class over its valid fine windows
(ties to the smallest class), whose reference is the majority reference the same way, whose validity is any valid
fine window, and whose confidence is the mean fine margin (a proxy, used for the low-confidence cue only; nothing
below is graded on it). The boundary cue is then the tool's own `_boundary_valid` on the coarse grid. Base's seven
segmentation tasks at k = 2 and k = 4 (MADOS 20 × 20 → 5 × 5 at k = 4; the others divide evenly).

**G1, the saturation is the grain's.** Base's cashew export at k = 4 has boundary prevalence at least **0.90** and
enrichment at most **1.10**. *What makes it fail:* prevalence below 0.9 under majority pooling, which would make
AnySat's saturation a property of its map rather than of the window size, and the record's note would say so.

**G2, the ceiling governs across grains as it did across tasks.** Over the 21 (task, k) cells the enrichment's
rank correlation with the ceiling `(1 − θ)/(p − θ)` is at least **0.9**. *What makes it fail:* a correlation
below 0.9, meaning the grain changes the enrichment through something other than the prevalence.

**G3, the threshold the tool now uses is right.** On every cell with prevalence at least **0.9** the risk ratio
inside/outside is at most **1.3**, and on every cell below 0.9 it is above **1.3**. This is the check of
`explain.BOUNDARY_SATURATED = 0.9`. *What makes it fail:* a saturated cell where the cue is still real (the
constant overstates) or an unsaturated cell where it is not (the constant is too high).

Descriptive: prevalence, enrichment, risk ratio and the ceiling per (task, k); the gate and P1–P3 are not graded
on coarsened exports (the recorded capture is at the fine grain). Cost: minutes, no cluster.

**Result, written after the run.** G1 fails as written, by 0.004: Base's cashew map at k = 4 has prevalence 0.896
against the 0.90 bar, with enrichment 1.04 inside its bar. Most of AnySat's saturation is therefore the grain
(0.765 → 0.896 for the same map) and the rest is its map (0.966). G2 holds: Spearman 0.987 over the 21 cells
(MADOS at k = 4 has a prevalence below its error rate, so its ceiling is unbounded and ranks highest). G3 fails
by 5e-5: the cell at 0.896 has a risk ratio of 1.2999, so the cue is already down to 1.3 just below the threshold,
and no cell reaches 0.9 to test the first clause. What the run shows without a bar is monotone: coarsening lowers
the enrichment and the risk ratio on all seven tasks at both steps (cashew 1.24 → 1.08 → 1.04 and 2.14 → 1.50 →
1.30; MADOS 8.13 → 7.30 → 6.06 and 10.5 → 8.3 → 5.7) and raises the prevalence on six of seven (MADOS's falls at
k = 4, where majority pooling erases its small marine objects). The recordable statement is that one, and the
tool's clause was reworded from "not a reason" to "at most a weak reason", with the two measured points (1.3 at
90%, below 1 at 97%) in place of a bar the data did not reach.
