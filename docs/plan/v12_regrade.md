# exp98's and exp99's frozen rules applied to the record's v1.2 fine-tune (exp103 preregistration)

**Status: frozen on 11 October 2026, before the v1.2 fine-tune has predicted any window.** Written 11 October 2026,
after exp102 trained and published the model (https://huggingface.co/2imi9/OlmoEarth-v1_2-FT-AWF-Base). Every
threshold and floor below is copied from a page the owner already confirmed and froze (docs/plan/awf_deployment.md,
frozen 7b12dc4; docs/plan/awf_transfer.md, frozen 52489b1); nothing new is set here, so the owner's go ("ok", 11
October 2026) to run the regrade freezes it.

**Result, 11 October 2026 (added at recording; the frozen text is unchanged).** Run at b83a5b9 (jobs 1281898 to
1281904; rslearn 0.1.17 predicted all 270 windows of 2023 and all 271 of 2017 with input), recorded after an
independent audit ([comparisons](../results/comparisons.md#a-second-model-of-the-same-task-exp98s-and-exp99s-rules-on-a-v12-fine-tune-exp103)).
A-P3 holds (AUROC 0.814), A-P4 holds (37.5%), A-P5 holds (0.653, lower bound 0.617). T-P1 does not hold (0.642; under
LENIENT 0.730, so the frozen reading is that only STRICT fails), T-P2 holds on the point value (0.267), T-P3 not
determined (STRICT 37.0%, LENIENT 16.6%). The same verdicts as for v1.

## The question

Every result in the record about a deployed OlmoEarth map comes from one model, Ai2's FT-AWF on OlmoEarth v1 Base. Do
the findings about the verification layer hold for a second model of the same task: the record's fine-tune of OlmoEarth
v1.2 Base with Ai2's recipe and labels (exp102)? The question is about the layer's findings (does the map's own
confidence rank its errors, at Ai2's points and under transfer), not about which model is better: no prediction compares
the two models.

## The maps

The v1.2 checkpoint (exp102: epoch 60, sha256 5ce88827...) predicts with `rslearn model predict` (rslearn 0.1.17, the
venv exp102 trained in) over the same two rslearn datasets the record's v1 maps came from, writing a new output layer
`output_v12` with the ten-channel softmax (the model.yaml's `output_probs` and float32 writer layer), and nothing else
in the datasets changes:

- **2023 (A):** exp98's deployment dataset on scratch (270 windows over Ai2's AWF request geometry, 2023 imagery).
- **2017 (T):** exp99's deployment dataset on scratch (313 windows holding the TimeSync plots, 2017 imagery; 271 with
  twelve item groups).

Each is read with `oe-inferencex from-olmoearth --layer output_v12` in parts, as exp98's and exp99's read jobs read
theirs, and graded with exp98's and exp99's own scripts at exp98's and exp99's frozen commits' logic, writing to new
output directories (`exp/out/exp103/`), never over exp98's or exp99's records.

## Predictions (copied from the frozen pages)

On map A at Ai2's validation points inside the area, as exp98 graded them (floors: at least 200 points, and at least 15
errors for P3 and P4):

- **A-P3:** the AUROC of the map's top-1 confidence for its errors is at least 0.75 (exp98's P3).
- **A-P4:** the least confident 10% of the points hold at least 25% of the errors (exp98's P4).
- **A-P5:** the points' mean percentile among the map's pixels by confidence is at least 0.55, with the one-sided 95%
  task-cluster bootstrap lower bound above 0.5 (exp98's P5).

On map T at the TimeSync plots with input, as exp99 graded them, STRICT primary and LENIENT reported (floors: at least
250 plots with input, 20 errors and 20 correct under STRICT):

- **T-P1:** the design-weighted AUROC of the confidence for the STRICT errors is at least 0.70 (exp99's P1).
- **T-P2:** the least confident 10% close at least 0.25 of the gap from random to the ceiling (exp99's P2).
- **T-P3 (descriptive):** the error rates under STRICT and LENIENT with exact intervals and exp99's transfer-gap call.

**Not applied:** exp98's P1 and P2 compare the re-run map with exp89's replica of the v1 model; for a v1.2 model they
have no meaning and are reported as not applicable. exp98's and exp99's report-only parts (accuracy at the points,
the reweighting, plan, the burned-area layer, the condition layer) are reported where their inputs exist.

## What would change the conclusion

- **All five graded predictions hold as they did for v1** (exp98's P3 to P5 held; exp99's P1 did not, P2 held at the
  point value): the layer's findings carry over to a second model, including the weaker ranking under transfer.
- **A prediction that held for v1 fails for v1.2, or the reverse:** the finding is model-specific, and the record says
  so beside exp98's and exp99's.
- **The maps cannot be produced** (rslearn 0.1.17 cannot predict on datasets prepared with rslearn 0.0.27): reported,
  nothing graded.

## What it cannot show

- Whether v1.2 is better or worse than v1: one fine-tune, one seed, and its checkpoint was chosen on the validation
  points the A predictions are graded at (as Ai2's v1 checkpoint was), so A's accuracy is optimistic for both.
- Anything about a model of another task.

## Outputs

`exp/out/exp103/awf2023/` and `exp/out/exp103/transfer2017/` (the summaries and units exp98's and exp99's scripts
write), the job records in `exp/out/exp103/`. Job `exp/jobs/e103.sh` on the aicr cluster (GPU for the predictions,
CPU for the reads and grades), pinned to a full commit.
