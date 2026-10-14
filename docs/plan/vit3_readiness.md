# Readiness for an encoder that adapts at inference

Written 18 September 2026, before any such checkpoint exists, and before its
design is known. Nothing here assumes one architecture: the gates below are
properties any encoder that adapts per input will have, whether by test-time
training, fast weights, retrieval or tuning at inference, and the code takes
only decisions, margins and per-unit readings. ViT3 is the worked example
because it is the most concrete published design. ViT3 (Han et al., arXiv
2512.01643) replaces attention with a test-time-training layer: a
small inner model is refit on every input sequence from all of its key-value
pairs, in a single full-batch step at learning rate 1.0 under a mean-squared
reconstruction loss, and the adapted weights produce the output. The paper's
own accounting puts one inner epoch at about four forward-equivalents per
layer. If OlmoEarth moves to this design, the audit does not change, because
it reads predictions, not encoders. Four properties of the encoder do change,
and each has a gate prepared here.

## What changes, and the gate for it

| ViT3 property | Consequence for an audit | Gate, and where it lives |
|---|---|---|
| The inner model is fit on all tokens of the input, so a window's features depend on the crop it was inferred in | Tiling instability becomes structural; the value measured on a frozen ViT does not transfer | `signals.crop_dependence`: the share of windows whose decision flips across the four offsets, a map property reported beside the reseed floor |
| One full-batch inner step at learning rate 1.0 | Two engines or two precisions can drift more than a frozen ViT does | `compare.determinism_check`: the same tiles inferred twice, disagreement and margin drift against the family's reseed floor, pass or fail |
| A self-supervised reconstruction residual exists per token at inference, for free | A new label-free reading people will want to use as an error score | `scripts/suite_regression.py --signal`: any per-unit reading scored beside the margin and the two no-model controls on the 24 tasks |
| The adapted weights move by a measurable amount per input | A per-tile novelty score, the typicality family in new form | The same entry point; and, on the comparison side, a cue for where two inferences differ |

The fit is per sequence, not per batch, so batch composition is not expected
to move a map; the determinism gate compares engines and precisions.

## Preregistered, before the run

Stated for a test-time-training OlmoEarth; the thresholds are the record's
existing bars, so they carry over to any adaptive design unchanged.

- **P1** The margin of a test-time-training OlmoEarth beats the best no-model
  control on at least 75% of the tasks it carries, sign test p < 0.05,
  exp74's bar. The suite regression runner decides it.
- **P2** The test-time reconstruction residual, pooled to windows, beats the
  margin on at most 6 of 24 tasks, as the pretraining objective's own residual
  did not (exp28, exp32 to exp34): a self-supervised residual measures how
  unusual the input is, not whether the head is wrong.
- **P3** The crop-dependence rate is higher than frozen OlmoEarth Base's on
  the same tiles and offsets. Base's number is recorded first, from exp18's
  cache, and the factor is stated then.
- **P4** Under bf16 the map disagrees with the fp32 map on no more than the
  family's reseed floor (exp57: 2 to 4% of windows). If it does, that engine
  is not used for an audit, and the record says so.

Also on the record before the run: a test-time-training OlmoEarth is a new
model family, so every fusion fitted in exp65 is refit or dropped, and the
readout gap is reported on adapted features before any ranking number is
trusted.

## Compute budget per reading

Every audit here scales with inference, not with labels, and a ViT3 forward
pass costs about four times an attention pass. Readings and their price:

| Reading | Forward passes | Status in the record |
|---|---|---|
| Margin, entropy, boundary indicator, spectral cues | 1 | the ranking, supported |
| Crop dependence, tiling instability | S (4 here) | a map property; rejected as a ranker |
| Shift-averaged decision | S (4 here) | the decision design, supported |
| Five-seed ensemble readings | K (5 here) | rejected |
| Test-time reconstruction residual, adaptation magnitude | 0 extra on a ViT3 | untested, P2 |

The single-pass readings stay the default. The rest are for measuring the
map, not for ranking it.

## How to run it when a checkpoint exists

1. Publish the checkpoint's embeddings for the suite, or run the encoder on
   the 24 tasks, and run `scripts/suite_regression.py --model <dir>`; that is P1.
2. Export the reconstruction residual per token as a per-unit reading and run
   the same command with `--signal`; that is P2.
3. Infer exp18's Bolivia tiles at the four offsets, pool the decisions, and
   call `crop_dependence`; compare with Base; that is P3.
4. Infer the same tiles under fp32 and bf16 and call `determinism_check` with
   the reseed floor; that is P4.

Each step is one job. None has been run; nothing here is a claim.
