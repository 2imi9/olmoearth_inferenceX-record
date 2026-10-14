# Is a confident error a window that agrees with its scene? (exp77 preregistration)

Written before the run, from a hypothesis this project did not invent. Runs as
`exp/exp77_scene_contamination.py`; the numbers land in the ledger like every other experiment.

## The gap this addresses

The record's own limit is stated in [Findings](../Findings.md#ranking-errors-without-labels): the margin takes a median 0.68 of the
gap between a random and a perfect ranking, labels buy a fifth to a third of the rest, and *what remains is
errors the model makes confidently, which no label-free reading tried has seen*. Eleven readings from the
encoder's internals, its pretraining objective, a posterior over the head, feature typicality, a second model of
the same family and flip-and-rotate consistency have all failed to see them
([TECHNIQUES](../TECHNIQUES.md)); so did a five-seed ensemble, nearest-neighbour typicality and a Mahalanobis
distance on all 24 tasks of the suite (exp73), and every form of the confidence family (exp76).

Those readings share an assumption: that a confident error is *unusual* in some measurable way. A mechanism that
would make confident errors **usual** has never been tested here.

## The hypothesis, and where it comes from

"Vision Transformers Need More Than Registers" (Shi, Yu and Yang, arXiv 2602.22394) reports that a ViT trained
with image-level supervision and global attention diffuses foreground semantics into background tokens: the
global representation is assembled from patches that do not carry the class, a shortcut they call *lazy
aggregation*. Their probe is the similarity between a patch and the CLS token; their fix changes pre-training.

Neither transfers directly. OlmoEarth has no CLS token, ships zero register tokens in both released Base
checkpoints, and its main loss is per token, with an image-level contrastive term weighted 0.1 (v1) and 0.05
(v1.2); and a pre-training change is Ai2's to make, not this project's. What does transfer is the mechanism,
because a token can be compared with its own tile's mean whether or not a CLS token exists:

> **If scene-level semantics have diffused into tokens that do not carry them, then a window whose token looks
> like its scene will be labelled by what its neighbourhood is rather than by what it shows. Such a window is
> confidently wrong, and the thing that marks it is not that it is unusual but that it is too typical of its
> scene.**

This is testable on frozen features, which keeps it inside the audit's scope: no encoder is trained.

## What is measured

On the seven segmentation tasks of Ai2's published suite (the tasks whose embeddings are token grids, so the
tokens are readable): for tile `n` with tokens `x_i`, let `m` be the tile's mean token and `m_hat` its unit
vector.

| Quantity | Definition | Role |
|---|---|---|
| scene typicality | `cos(x_i, m)`, averaged onto the 4-px windows everything here is graded on | a label-free reading |
| decontaminated token | `x_i - lambda * (x_i . m_hat) m_hat` | the intervention |
| arm A | exp70's recipe unchanged: Ai2's linear probe on the raw tokens, seed 0 | the baseline |
| arm A' | the same, seed 1 | the floor a gain must clear |
| arm B | the same recipe on decontaminated tokens | the intervention's map |

`lambda` is chosen from {0.5, 1.0} on a 20% holdout of the **train** tiles. The test split never chooses it.

## The predictions

1. **P1, the mechanism.** Inside the more confident half of the windows, the mean scene typicality of the error
   windows exceeds that of the correct ones, on at least 5 of the 7 tasks.
2. **P2, the intervention.** Arm B's window accuracy exceeds arm A's by at least 0.5 points, and by more than
   that task's own seed floor, on at least 4 of the 7 tasks.
3. **P3, the diagnosis.** Across the seven tasks, the size of P1's gap ranks positively with the size of P2's
   gain (Spearman rho > 0). This is the prediction that matters for the tool: it is the claim that an audit
   quantity measured *without labels* says where a fix will pay.
4. **P4, the honesty check.** No reading introduced here beats the margin at ranking all errors on more than 2
   of the 7 tasks. The record says the margin is the best single ranker; a run that quietly overturned that
   without saying so would be the failure mode to guard against.

P1 and P2 can hold while P3 fails; that would mean the mechanism is real and the audit cannot locate it, which
is a negative result worth recording. P2 failing while P1 holds would mean the contamination is measurable but
not removable by a linear projection, which is also worth recording.

## What would make this wrong

- **A gain inside the noise.** Two seeds of arm A differ by some amount on every task; a gain below that is not
  a gain. P2 is gated on it explicitly.
- **Choosing `lambda` on the test split.** It is chosen on a train holdout; the final fit uses the full train
  split and is scored once.
- **Reading the result as an improvement to OlmoEarth.** It is not. The encoder is frozen; this changes how a
  *probe* reads its tokens, on the suite's own probe recipe. Whether it would survive a fine-tuned head, or
  another encoder, is not tested here.
- **One family of tasks.** All seven are segmentation, and three are PASTIS variants; the sign test over tasks
  overstates independence, as [exp70](../results/comparisons.md#tasks-we-did-not-choose-ai2s-whole-published-suite-exp70)
  already notes for the suite.

## Cost

CPU only, the cached embeddings the suite already uses, five probe fits per task (two baseline seeds, two
lambda choices on the holdout, one final). No new dependency, fp32 throughout.

## Amendment, 20 September 2026, written before the run's numbers were read

Two defects in the text above, found by an adversarial read of this page against the code while job 1003587 was
still running. The text above is left as it stood; this section corrects it, and the corrections bind the grading.

**1. P3's quantity is not label-free, and the sentence claiming it was is wrong.** P1's gap is the mean scene
typicality of the *error* windows minus that of the *correct* ones, inside the confident half. Separating error
from correct needs the labels: `contamination_gap` splits on `err`, which comes from the majority label per
window. So P3 as written correlates a **labelled** diagnostic with the gain, and the sentence above calling it
"the claim that a quantity measured *without labels* says where a fix will pay" is false of its own quantity.
P3 is still worth grading, but it is the weaker claim, and it will be reported as the weaker claim.

The label-free claim is worth a test, because it is the one a reviewer of a map could act on. It needs a
quantity this run does not store: the distribution of window scene typicality per map, which needs no labels and
no probe. **P3-label-free**, defined here before any result is read: across the seven tasks, the mean window
scene typicality of a task ranks positively with that task's P2 gain (Spearman rho > 0). It is graded only if P2
holds, since a gain of zero everywhere has nothing to rank against. The quantity comes from a follow-up job that
loads the same cached embeddings, computes the cosines and summarises them; it fits no probe and trains nothing.

**2. P1 has no control for class frequency, so a positive gap will not identify the mechanism.** Within a tile,
a window the model gets confidently wrong is disproportionately a minority-class window predicted as the tile's
dominant class, and such a window's token sits near the tile mean for reasons that have nothing to do with
diffusion of semantics. Class rarity is not a hypothetical alternative here: it is the best no-model control on
six of these seven tasks (exp/out/exp70_summary.json; the seventh, Sen1Floods11, prefers embedding distance).
A positive P1 is therefore consistent with the LaSt-ViT mechanism **and** with plain class imbalance, and no
sentence in the record may pick between them on this run's evidence. P1's statement is amended to what it can
support: *confident errors are more typical of their own tile than confident correct windows are*, with the
mechanism left open.

**Checks before any prediction is graded.** Arm A is exp70's recipe by construction, so it must reproduce
exp70's recorded numbers on all seven tasks or the baseline has moved and no gain is comparable to the record:

| task | accuracy | windows | margin excess AURC |
|---|---|---|---|
| mados | 0.9264 | 22,598 | 0.007793 |
| sen1floods11 | 0.9155 | 592,385 | 0.020199 |
| pastis_sentinel1 | 0.7157 | 458,638 | 0.071459 |
| pastis_sentinel2 | 0.8099 | 458,638 | 0.038639 |
| pastis_sentinel1_sentinel2 | 0.8052 | 458,638 | 0.039449 |
| m_cashew_plant | 0.6528 | 204,800 | 0.130784 |
| m_sa_crop_type | 0.6600 | 4,096,000 | 0.067186 |

Also: every count is reported against seven whether or not seven tasks ran, and by the five distinct sources as
well, since the three PASTIS variants are one source; a task whose gap is undefined (too few confident errors)
is counted as not supporting P1 and named; `same_valid_windows` is checked per task before any accuracy
comparison; the seed floor is one reseed, not a distribution, and is called that; and P4's margins are reported
signed and beside the margin's own lead over the best no-model control on that task, which runs from 0.0165 to
0.1272 here, so that a lead of 1e-9 is not read as a result.

**The lambda grid cannot choose zero.** It is {0.5, 1.0}, so arm B always removes at least half the scene
direction and the holdout cannot decide to leave the tokens alone. A task whose best answer was "do nothing" can
only appear as a loss. This is a limit on P2, and the chosen lambda and both holdout accuracies are reported per
task so a reader can see it.

