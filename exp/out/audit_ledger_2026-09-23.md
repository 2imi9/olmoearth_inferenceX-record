# Audit of the claim ledger's checks, 23 September 2026

Read-only audit by a separate agent of `docs/claims.yaml` (167 claims), `docs/Findings.md`, `docs/results/*.md`,
`docs/related_work.md` §7, `tests/test_claims.py`, `tests/test_recorded.py` and the artifacts under `exp/out/`.
The question: which checks could fail for a reason their author did not anticipate. Kept here so the front page's
sentence about verification can point at what was found rather than restate it.

## Headline

- 138 of 167 checks read a verdict flag (`['holds']`, `['prereg'][...] is True`) or pin a rounded number back
  from the artifact that produced it; on the front page, 77 of 85. Such a check catches drift after the fact and
  never a formula wrong from the start.
- 5 claims named a `crosscheck` before this audit; `tests/test_recorded.py` recomputed seven experiments from
  committed per-unit files (exp21, exp37 shares, exp65, exp66, exp67, exp68, exp69) without any claim naming them.
- Actioned the same night: the checks of `suite-margin-wins-every-task`, `suite-holds-under-every-encoder`,
  `suite-outside-families-hold-at-twenty-two`, `suite-margin-beats-the-strong-alternatives` and
  `suite-alternatives-clear-the-control-and-still-lose` now recompute wins, losses, sign tests and medians from the
  per-task signal fields (each verified to turn False under a one-task perturbation and to stay True when only the
  verdict block is corrupted); exp78's three claims and exp37's five carry new recomputation tests as `crosscheck`;
  the seven existing recomputations are named on the claims they cover, with their scope (subsample or direction
  only) stated in the claim notes. The headroom share is 0.680 under both the exact and the asymptotic oracle.

## 1. Read-back checks, by weight (the 24 groups the audit listed)

1. `suite-margin-wins-every-task` — rewritten.
2. `suite-holds-under-every-encoder`, `suite-outside-families-hold-at-twenty-two` — rewritten.
3. `suite-margin-beats-the-strong-alternatives` — rewritten.
4. `suite-alternatives-clear-the-control-and-still-lose` — rewritten; its statement was also wrong (the ensemble
   beats the margin on EuroSAT and AWF Landsat, where it clears the control) and is corrected.
5. `geoid-exception-rate`, `geoid-capture-effect-size` — per-event records exist for the 2/45, 1/45, 0.88 and 0.64.
6. `atlas-disagreement-is-boundary-located`, `atlas-different-differences`, `atlas-which-side` — exp57_masks.npz
   holds every pair's decisions and labels; needs a crosscheck test (the ledger loads only JSON and CSV).
7. `two-periods-sensor-axis-isolated`, `two-periods-time-only-difference`, `residue-not-seasonal-water` —
   exp60_masks.npz and exp61_layers.npz carry everything.
8. `fourth-cell-completed` — exp62_masks.npz joined to exp60 by chip index.
9. `multiclass-boundary-cue-fails-on-parcels`, `multiclass-sensor-vs-encoder-different`,
   `shared-errors-task-dependent` — exp63_masks.npz supports every number.
10. `calibrate-side-rule-held-out` — now crosschecked on all eight pairs; `calibrate-family-lock` is not
    recomputable (the forced frozen rule needs training windows not in exp65_readings.npz).
11. `dfc2020-*` — crosschecked on the 200-patch subsample for direction and magnitude; P1's NDWI control is not
    exported per window, so the 0.1576 lead is not recomputable.
12. `dw-*` — crosschecked on an every-eighth-tile subsample; the per-pixel ECE 0.199 is not recomputable.
13. `lucas-*` — the filter contrast is now crosschecked; the two-date part (917/591, 31.2%) has no second-date
    decision committed, and `lucas-overfitting-inverts-the-ranker-ordering` is recorded from a superseded run's
    numbers (its own note says so), so the front-page sentence "putting the margin last" has no artifact.
14. `eurocrops-*` — crosschecked (floors and ratio; the ranking claim by direction).
15. `agent-benchmark-*` (six claims) — per-card fields exist; a recount is one loop, not done.
16. exp78's three claims — crosschecked by a seeded re-draw from the per-unit files.
17. exp37's five claims — crosschecked from the per-window table, all exact.
18. `w1-accuracy-gain`, `mixed-label-windows-not-a-ceiling` — recomputable inside the ledger from the CSV; not done.
19. `finetune-*` — exp52 exports no per-window data; the corrected share is recomputable from exp57's masks, the
    NDWI test behind "the exception goes" is not.
20. `confidence-best-single-signal`, `bolivia-ndwi-exception` — hinge on exp45's `replicates` flag; no per-window
    artifact for exp45/47.
21. `label-fitted-fusion-*`, `shrug-signals-rejected`, `bag-not-replicated` — exp49/50 committed summaries only;
    unrecomputable by any route.
22. `modality-dominates-shared-errors`, `s1-probe-ndwi-flip`, `v12-bolivia-their-readout`,
    `cross-encoder-phi-on-their-embeddings`, `multiclass-confidence-beats-embedding-control`,
    `audit-does-not-save-labels`, `no-encoder-internal-signal-beats-confidence`, `sensor-disagreement-*`,
    `confident-errors-are-scene-typical` — no per-unit records committed.
23. `pixel-head-beats-encoder-bolivia`, `served-product-boundary-triage`, `accuracy-needs-coverage`,
    `fine-tuned-model-audit` — pure number read-backs although a CSV exists; the last two now crosschecked.
24. `margin-takes-two-thirds-of-the-ranking-headroom` (asymptotic oracle) against
    `the-ranking-ceiling-is-the-tasks-not-the-encoder` (exact): both give 0.680; left as they are, stated here.

## 2. Front-page sentences the body does not back at that strength (fixed the same night where marked)

- "beats every index that never sees the model on every task of Ai2's suite" — the suite has no pixel index and
  its class-rarity control sees the decision; "one known exception" where the body lists three. **Fixed.**
- Boundary-first "finds more errors at small review budgets" with no exception, against losses at 8, 9 and 15–19
  classes and on 60% of GEOID events. **Fixed.**
- "The package tests recompute the recorded numbers from the committed artifacts" — true of a minority. **Fixed.**
- "nine times what the same axis moved on water" divides two artifacts and no check reads both.
- "putting the margin last" (LUCAS) rests on a superseded run.
- Recorded sections never mentioned on the front page, of which five are exceptions or scope limits: exp75
  (sensor disagreement is a real signal and finds 24–25% of the margin's missed errors where the view head can do
  the task), exp06 (on the Zambezi delta every no-model statistic ranks the WorldCover disagreements as well as
  every model signal), exp35 (the NDWI control beats confidence pooled at 20% on Bolivia hand labels), exp38 (a
  spectral-ambiguity-first order beats boundary-first by ten points at 20%), exp22 (the served product's
  boundaries are quantised to the patch lattice).
- `docs/results/signals.md` states numbers and carries no markers.

## 3. Scope gaps (dimensions other than seeds, which exp79 covers)

Encoder (every front-page finding but the suite ranking is OlmoEarth v1 Base with a linear probe); boundary-first
(binary water, v1, grid window; fails on the v1.2 test split, loses at 8 and 9 classes, ties the margin on LUCAS);
cues (Bolivia only; the strongest cue is water-specific); tiling average (Sen1Floods11 S2 only); fusion (two splits
of one dataset, an NDWI feature carrying it); fine-tuning (one run per arm, one recipe); "the sensor is the largest
lever and the backbone the smallest" (one version pair; an encoder-family swap moves as much as the sensor on
multi-class tasks); the suite's "no-model" controls (one sees the decision); exp78 (one encoder, one budget in the
artifact, one cluster design); exp64 (binary water cards, one model per size); GEOID (4 of 19 shards, 9–10
activations); DFC2020 (one reference pair); LUCAS and EuroCrops (one encoder, one year, class heterogeneity not on
the front page); "an accuracy needs a coverage" (one model with 41 errors, reversed on Dynamic World).

## 4. Cheap and untested, computable from committed artifacts alone

1. Cross-family Dawid–Skene consensus from exp63_masks.npz (six encoders on MADOS and PASTIS S2) and exp57's
   encoder masks on Sen1Floods11: the record calls the within-family rejection "the designed fix, untested".
2. The deployable two-signal fusion (confidence + NDWI level, cross-fitted by tile) from exp65_readings.npz, v1 only.
3. The test-split boundary shares (73% vs 18%) quoted with exp18 from exp65_readings.npz, which would replace the
   number rather than reproduce it (shift-averaged decision, not the grid window).

Not computable from exp/out alone, although named cheap: CAPA for exp75 (aggregates only committed); the
likelihood-ratio selector (needs features); exp64's multi-class cards; the frozen joint head; head-seed variance.
