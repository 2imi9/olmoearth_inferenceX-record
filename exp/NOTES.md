# Experiment notes

**Archived 10 October 2026.** The 14 experiments exp06, exp09, exp10, exp12, exp22, exp26, exp27, exp39, exp40,
exp44, exp53, exp61, exp86 and exp87 were moved out of the repository. Their files (scripts, outputs, tests and plan
pages) are in the [v1.9.0 tree](https://github.com/2imi9/olmoearth_inferenceX/tree/v1.9.0); where an entry below
names one of their files, that is where it is.

Chronological lab log. Standing conclusions live in docs/TECHNIQUES.md.

Entries run to exp47 (9 September 2026). From exp49 on, each experiment is
recorded where it is graded: the results pages under docs/results/, the
technique ledger and the claim ledger (docs/claims.yaml), and this log was not
continued. exp48 was preregistered and not run (issue 13).

| Exp | Topic |
|---|---|
| smoke_test | checkpoint load; noise floor for cross-model agreement |
| exp01 | embedding-level agreement map, Kazungula |
| exp02 | full audit slice: heads, E_case, E_geo, baseline, risk-coverage |
| exp03 | tri-model, tile-phase, E_dist, v1_2 attempt |
| exp04 | AWF expert-label validation, official spatial split |
| exp05 | difficult scenes: wetland margins, domain shift |
| exp06 | no-model pixel-statistic controls |
| exp07 | Dawid-Skene label-free reliability (rejected within family) |
| exp08 | masking perturbation (rejected) |
| exp09 | seven-scene replication with control |
| exp10 | rater strength vs diversity (diversity wins) |
| exp11 | pre-registered 29-scene comparison with bootstrap + permutation stats |
| exp12 | fills the AWF table cells: E_dist and pixel control in-domain |
| exp13 | corrected statistics on the rule-selected set: aligned tile-phase, tie-aware excess AURC, block bootstrap, sign/permutation tests |
| exp14 | boundary ablation: tile-phase indistinguishable from prediction-boundary proximity |
| exp15 | boundary proximity + E_geo: no benefit from the conjunction; E_geo sensitivity unmeasurable under WorldCover truth |
| exp16 | boundary indicator on AWF: labelled patches not interior; score error-associated but a proxy for low margin; confidence wins |
| exp17 | evidence from inside the encoder: band-set disagreement 21/27 vs WorldCover; depth probes marginal; logit-lens, drift, attention entropy rejected |
| exp18 | Sen1Floods11 hand labels (spatial hold-out): confidence is best; tile-phase, band-set, boundary, E_case all equal or worse; WorldCover wins read as reference error |
| exp19 | v1 vs v1.2 (RoPE): instability larger under v1.2; single band-set token in v1.2; cross-version disagreement not useful |
| exp20 | first served production output (olmoearth_lcc): no class confidence exported; boundary captures 0.88 of water disagreements at a 5% budget; change ambiguity sits on flagged-region edges |
| exp21 | fine-tuned AWF model end to end from Ai2's checkpoint: accuracy 0.881 (Ai2 0.895); confidence best, tiling instability ties it; overconfident (ECE 0.080); 0.945 at 80% coverage |
| exp22 | served v1.2 product: outputs quantized to the 4-px patch lattice (19/20 profiles, p <= 6e-08); no inference-window seams at 64-512 px (limit 5-10% of rows at 128 px); WorldCover control clean |
| exp23 | WorldCover 2020 vs 2021 instability: errors 14x enriched at unstable patches but those are ~10% of errors; tile-phase advantage unchanged on stable patches (21/23); exp18 reading withdrawn |
| exp24 | 2021 imagery + 2021 map + within-year head: tile-phase still beats confidence 23/3/0 (2024 same scenes 25/1/0); year gap does not explain the WorldCover wins |
| exp26 | hand-check kit: top-12 disagreements by tiling instability and by confidence on the three scenes with the largest exp13 gain (shire_80, barotse, okavango_80); crops, cues and blank verdict columns; no claim |
| exp25 | JRC seasonal-water split: disagreements enriched on seasonal margins, but tile-phase still beats confidence without them (22/2/0 in 2024, 22/1/0 in 2021); third mismatch component ruled out |
| repro-aicr | exp02 and exp21 reproduced end to end on the AICR B200 cluster (2026-09-05): every metric identical, max abs diff 1e-5; exp17, exp18, exp20, exp23-exp26 and the transfer summary rerun there on 2026-09-06 with byte-identical tracked outputs and the river-level 8/0 (p = 0.0078) unchanged |
| bench-b200 | encoder throughput and precision on AICR: fp32 leaves the B200 tensor cores idle (18.8 ms per 128x128 window); TF32 halves it with 59 of 60 exp21 fields unchanged and one budget count moving by one window; bf16 7-10x, untested on the audit |
| exp27-gate | oracle gate on the latent-MIM target space (2026-09-06, CPU): pure-class WorldCover prototypes have pairwise cosine median 0.996, class sits in token norm, layout outweighs water fraction, ridge readout of water fraction from perfect tokens R2 0.67 raw / 0.55 normalised; prototype and retrieval readouts of the shipped decoder are unsound |
| exp28 | decoder self-consistency (native masked-token reconstruction error, whole-spectrum and token-level masks, crowding/constant/input controls): rejected on both testbeds; the preregistered confidence+decoder combination gains nothing (4/4 rivers, p = 0.64); the frozen targets are near-collinear (within-scene pairwise cosine median 0.994) |
| exp30 | last-layer Laplace on the probe head (logit variance, probit-moderated confidence, Gauss-Hermite MI and entropy, bootstrap-head std): rejected on both testbeds; the preregistered confidence+variance combination loses 0/8 rivers (p = 1.0); on the one-scene head the variance is feature norm (Spearman 0.89), on the 128k-patch head the weights are well determined and the variance is still the worst signal |
| exp31 | feature-space typicality (kNN, Ledoit-Wolf Mahalanobis, PCA residual, class-conditional Mahalanobis, ViM) against the head's training patches, the scene itself (cross-fitted) and a cross-testbed pool: rejected; the preregistered confidence+typicality combination reaches 6/2 rivers (p = 0.145) against WorldCover and hurts on hand labels (60/291 tiles); no typicality score beats confidence on either testbed |
| exp32 | the latent-MIM target space read from code and measured on CPU: the target encoder is the untouched random initialisation (EMA decay 1.0); on four scenes its targets have random-pair cosine 0.99 and centred effective rank 2.0 against 54 for the online encoder: the mechanism behind exp27 and exp28 |
| exp33 | context-prediction residual with a whitened target on the frozen encoder: the target swap makes the latent-MIM objective predictable (57% of whitened variance from context on held-out rivers, 70% on Bolivia) but the residual tracks input texture, not error: rejected on both testbeds; U+ 6/2 rivers (p = 0.145), 45/306 Bolivia tiles |
| exp34 | three further readings of the re-targeted objective: discrete-target NLL and predictive entropy (k-means, HuBERT-style), gap-masked residual (3x3 hole, Latent MIM-style), residual along the decision direction: all rejected on both testbeds; the preregistered entropy combination 0/8 rivers; the gap residual is the only variant at parity with confidence on WorldCover (13/14, 4/4 rivers) and it still tracks texture |
| exp35 | operating points at fixed review budgets (5, 10, 20%): the preregistered test (tiling instability vs confidence at 20% on Bolivia hand labels) is null (116/112/123 tiles, p = 0.42; pooled 0.707 vs 0.749); the boundary indicator captures more errors than confidence at the 5% budget on hand labels (0.286 vs 0.259, bootstrap CI excludes zero, per-tile p = 0.068), the one operating point where a constructed cue beats confidence; it reverses by 20% |
| exp36 | dihedral consistency (8 flips and rotations) and the boundary-first-then-confidence review order: dihedral consistency is not a better ranker (4/4 rivers vs confidence, 154/196 Bolivia tiles); the lexicographic order captures more errors than confidence at the 5% and 10% budgets on Bolivia hand labels (per tile 85/31/235 and 112/48/191, one-sided p = 3e-7 and 2e-7; pooled gain CI above zero), not at 20%, and gains nothing on the AWF fine-tuned model |
| exp37 | cue enrichment on identical windows for the explanation layer: on Bolivia hand labels 95% of error windows carry at least one of five label-free cues (boundary 3.5x, least confident 20% 3.6x, unstable 3.6x, NDWI-ambiguous 7.2x, flip/rotation disagreement 3.5x); inside confidence's 5% review set the boundary and NDWI cues separate error rates (0.46 vs 0.16, 0.51 vs 0.33), the tiling and dihedral cues do not; `oe_inferencex.explain` built |
| exp38 | spectral ambiguity first, then boundary, then confidence, at fixed budgets (CPU, on exp37's tables): preregistered against the exp36 order on Bolivia, per-tile budgets win at 5% and 10% (151/100/100, p = 8e-4; 146/86/119, p = 5e-5) but one pooled budget does not (0.292 vs 0.274, CI touching zero; 0.481 vs 0.494); clear only at 20% pooled (0.831 vs 0.732); loses to the boundary-first order on the WorldCover rivers (2/6, 3/5, 3/4); mixed, boundary-first stays the supported rule |
| exp39 | neighbourhood contradiction (share of a window's 32 nearest neighbours in a disjoint bank that the same head predicts differently) in three embedding spaces: P1 passed, the OlmoEarth-space score beats confidence at the 10% budget on Bolivia hand labels (169/126/56 tiles, p = 0.007; pooled 0.530 vs 0.465, CI above zero); P2, AnySat's local embedding beats confidence but not the OlmoEarth space (159/143, p = 0.19) and its 40 m patch output is position-dominated; the pixel-statistics ablation exceeds both by a wide margin (10%: 214/89/48, pooled 0.682 vs 0.465; E-AURC 219/131 and pooled 0.0093 vs 0.0105, the first hand-label E-AURC win), so the semantic-neighbourhood reading is falsified and the finding is the model's inconsistency across spectrally similar windows; null against WorldCover (rivers 4/4); replication preregistered as exp40 |
| exp40 | preregistered replication of the pixel-statistics neighbourhood contradiction on the Sen1Floods11 test split (800 tiles, other regions; Bolivia as the bank): fails, worse than confidence at the 10% budget (183/195/105, p = 0.75; pooled 0.530 vs 0.643, CI below zero), at 20% and on E-AURC (215/267, pooled 0.0149 vs 0.0096); the OlmoEarth-space score is far worse (97/301); the Bolivia win did not carry over, so neither contradiction score is supported |
| exp41 | two-view disagreement from Ai2's paper embeddings (probes per model on identical windows, partner chosen by lowest held-out error correlation): rejected. On Sen1Floods11 the three outside partners that share the 4-px grid err on 80-82% of OlmoEarth's error windows, as its own family does (phi 0.77-0.81), and disagreement captures 0.242 at the 10% budget against 0.454; on AWF the outside models are far less correlated (phi 0.32-0.48 against the family's 0.50-0.70). The gloss "the errors belong to the windows" was withdrawn: fine-tuning the encoder corrects 55.6% of a frozen probe's errors on the same AWF points (exp21) |
| exp42 | window design: four alternatives to the 4-px grid window judged at pixel level on hand labels; the shift-averaged decision (mean of the four tilings covering each pixel) raises pixel accuracy by 1.0 points on Bolivia (321/66/53 tiles, p = 1e-41) and 0.9 on the test split (583/78/139, p = 9e-97), preregistered and supported; spectral split and scale-adaptive windows are mixed; 10% of windows have mixed hand labels and carry 45% (Bolivia) and 58% (test) of the grid window's errors (error rate 0.28-0.39 against 0.02-0.05 on pure windows), the grid's limit, not the pixel map's |
| exp43 | a content-derived partition (deterministic k-means on Sentinel-2, 4-connected components) with the hard majority of W1's pixel decisions, preregistered against W1 with the corrected/broken accounting: not supported; it breaks more pixels than it corrects on both testbeds (Bolivia C 15,093 vs B 16,273, 168/209 tiles; test split C 24,525 vs B 30,951, 244/387), because 11-15% of segments are mixed and 5-12% of the pure ones carry a wrong majority, which then flips whole segments; W1's own errors are 43% (Bolivia) and 56% (test) on the mixed-label 10% of pixels |
| exp44 | sixteen crop offsets against the four diagonals, twelve extra encoder passes: the full offset set is better than the diagonals on both testbeds (201/149/90, p = 0.003; 369/202/229, p = 1.3e-12) but by +0.04 accuracy points, five times below the preregistered minimum worthwhile effect, so unsupported; an unbalanced seven-offset subset (diagonals plus horizontal phases) is worse than the diagonals, a balanced eight-offset subset is between |
| exp45 | the four supported findings repeated on OlmoEarth v1.2 Base, the encoder the served product uses: the cue enrichments and the shift-averaged window replicate on both hand-label testbeds (the window gain is larger than under v1, as v1.2's greater tiling instability predicts); the boundary-first order replicates on Bolivia and does not extend to the test split; **confidence is no longer the best ranker on Bolivia under v1.2**, where tile-phase (0.0138) and even the no-model NDWI-level control (0.0119) beat it (0.0146); v1.2 matches v1's accuracy and ranks its own errors worse; cross-version overlap: v1.2 is wrong on 71-75% of v1's error windows and on 1-3% of the rest |
| exp46 | what makes the errors shared, five arms on identical Sen1Floods11 windows: the INPUT MODALITY is by far the biggest lever (Sentinel-1 against Sentinel-2 gives phi 0.39/0.38, against 0.70 for a backbone swap), a two-layer head moves the error set only marginally more than a backbone swap (0.68/0.70) and a 3x3 neighbourhood head a little more (0.63/0.68); label impurity is not the driver (phi is the same on homogeneous and mixed windows) and neither is chip clustering; a head on fifteen per-window pixel statistics with NO encoder beats the frozen encoder on Bolivia (0.9199 against 0.9116) |
| exp47 | ranking the shift-averaged decision's own errors, the design the repository recommends and had never audited: the averaged confidence beats tiling instability everywhere (+0.0017 to +0.0022, p from 4e-10 to 2e-37) and beats the grid window's confidence on Bolivia, but on Bolivia it is matched or beaten by the no-model NDWI-level control under BOTH backbones, so exp45's finding was not an artefact of grading the grid window; preregistration fails (only v1.2 on the test split passes); every U+ combination is worse than plain averaged confidence |

## exp01 — first E_case map (2026-08-31)

Setup: one S2 L2A scene (Planetary Computer, 0-5% cloud, 2024 dry season),
128x128 @ 10m, Nano + Base embeddings (CPU, patch 4), per-patch correlation of
local cosine-similarity structure between the two models (radius 4).

Findings:
- Noise floor: ~0.59 mean agreement on pure random input (shared
  patchification/normalization). Agreement must be read against this, not 0.
- Mongu dryland window: mean 0.66. Disagreement tracks linear features and
  heterogeneous ground; homogeneous woodland ~0.9. One hard low-agreement
  horizontal band with no RGB counterpart - possible detector seam,
  E_system-flavored, unchased.
- Kazungula window (river in frame): mean 0.78. Open water agrees ~0.9+,
  disagreement concentrates on the shoreline and traces the bridge (thin
  structure). Consistent with the E_case hypothesis: cross-model disagreement
  clusters where segmentation errors would live (boundaries, thin structures),
  not randomly.

Next:
- Quantify: agreement vs distance-to-waterline / GRIT centerline overlay.
- Few-shot water head on both embeddings -> prediction disagreement (true
  E_case, not just representation structure).
- v1 vs v1_2 same-window comparison (needs olmoearth_pretrain pull for loader).

## exp02 — full minimal audit slice (2026-08-31)

Every pipeline stage present, minimal depth. Heads trained at Katima Mulilo,
evaluated at Kazungula (~110 km, spatial split). WorldCover 2021 water as weak
truth. OSM waterway=river centerline as E_geo reference (overpass mirrors:
mail.ru worked when main + kumi were down; HF_HUB_OFFLINE=1 for cached reruns).

Numbers:
- Nano head 0.973 / Base head 0.979 eval acc vs WorldCover. Base error rate 2.1%.
- E_case AURC 0.0011 vs max-softmax baseline 0.0009. BASELINE WINS on this scene.
- E_geo: 52 centerline patches, 0 consensus-dry. No river breaks, no false alarms.

Honest read: on an easy dry-season scene with a 400 m river, the model's own
confidence is enough and E_case adds nothing. This is the §5 baseline argument
made concrete on day one. The channels' claimed value is on hard cases
(correlated errors, narrow channels, flood season, wetland margins), so the
next experiment must be a deliberately hard scene, not another easy one.
Also: errors are isolated specks partly from 2021-labels-vs-2024-scene drift
(moving sandbars) - weak-truth noise, not all model error.

Next:
- Hard scene: Barotse floodplain in flood season, or a narrow (<100 m) reach
  where the river is subpixel at patch scale. Expect E_geo to activate there.
- Tri-model (add Tiny) for Dawid-Skene-shaped E_case.
- v1 vs v1_2 E_system on the same windows.

**2026-09-06, true-colour panel.** The figure gained panel (a), the
Sentinel-2 L2A window itself (S2A 2024-08-29, 0% cloud, the least-cloudy
June-September 2024 item the loader selects deterministically) with the 22
disagreement patches outlined. The window was re-fetched into the cache as
the B04/B03/B02 digital numbers plus the date; its grid transform and the
WorldCover labels are identical to the committed cache, and the committed
probabilities were left untouched. A full CPU re-run on the Mac the same day
gave the same 22 disagreement patches, the same AURCs (0.0011 / 0.0009) and
the same error rate, with probabilities differing from the committed ones by
at most 1.5e-3 (consistent with fp32 accumulation-order differences; the
B200 run of 2026-09-05 matched to 1e-5). Where the outlined patches sit,
measured by `bridge_strip_check` in the script from the cache: the reference
draws the Kazungula bridge as a line of 15 non-water patches across the
river, mostly one patch (40 m) wide, and the head reproduces it, calling
only 2 of those 15 water (mean P(water) 0.12); the bridge neither breaks nor
disappears in the prediction. Of the 22 disagreements, 2 are on that line
and 20 lie within one patch of a reference class change (18 adjacent to
one); on 17 of those 20 the head says non-water where the 2021 map says
water. 15 of the 22 carry a confident prediction (P above 0.9 or below 0.1)
and only 1 is uncertain (0.3-0.7). Nothing is fabricated: these are boundary
cells on which a majority-pooled 10 m map from 2021 and a head reading a
2024 dry-season image disagree about the water fraction.

## exp03 — four techniques, one run (2026-08-31)

Same Katima/Kazungula pair. Results in docs/TECHNIQUES.md (ledger is the
authority). Headlines: tile-phase E_system beats max-softmax (first channel to
do it); naive tri-model std worse than pairwise (Tiny pollutes); E_dist doesn't
rank in-domain errors (OOD alarm, not error proxy); v1_2 loader blocked on
current olmoearth_pretrain checkout.

Separately, allenai HF datasets answer OQ4: olmoearth_lcc production COGs ship
binary-change prob + argmax + top-1 prob (encoder v1.2-Base), and AWF/mangrove
expert labels are public. Production output is HTTP range-readable.

## exp04 — AWF expert-label validation (2026-08-31)

Harness on real partner truth (details in ledger). Baseline wins on multiclass
in-domain errors; tile-phase easy-scene win did not transfer; weak-rater
effect replicated with Nano. Herbaceous wetland weakest class (50% recall).
Features cached in exp/out/exp04_feats.npz (5 passes x 1459 windows).

## exp05 — hard scenes and domain shift (2026-08-31)

Every channel beats max-softmax on both hard AOIs (Barotse floodplain
wetland margins, Zambezi delta mangrove shift). E_case 3x better on margins,
E_dist 18x better under shift. Weak-truth caveat: delta "errors" trace a river
WorldCover likely misses. Ledger has the full statement. Eval windows cached
in exp/out/exp05_cache.npz.

## exp06 — no-model image-statistic controls (2026-08-31)

Spectral variance, NDWI ambiguity, NDWI gradient vs the same errors.
Kazungula and Barotse claims survive (E_case keeps a margin over the best
control). The delta E_dist shift claim does NOT survive: trivial NDWI stats
rank those disagreements better. Claim withdrawn in ledger and README.
Secondary: no-model stats outrank max-softmax on both hard scenes.
Results: exp/out/exp06_controls.csv.

## exp07 — Dawid-Skene label-free reliability (2026-08-31, overnight)

Negative with a useful twist: DS overestimates all three models and inverts
the order (details in ledger). The DS-vs-measured gap measures correlated
error per model. Fix is an out-of-family rater. Tiny AWF features cached in
exp/out/exp07_tiny_feats.npy; Tiny val acc 0.805.

## exp08 — masking-perturbation ensemble, GPU (overnight)

First GPU experiment (RTX 5090 laptop, torch 2.7.1+cu128; 32 occlusion
reruns x 3 scenes in under two minutes). Clean negative: occlusion
instability is the worst signal everywhere; it measures context reliance,
not error. Design rule recorded in ledger: perturb tokenization, not
content.

## exp09 — multi-scene replication with controls, GPU (overnight)

Seven river scenes, full comparison + NDWI-gradient control (details in
ledger). Confidence best on 0/7; E_case 3 wins (all surviving the control);
tile-phase 2; control wins the two reference-omission scenes as exp06
predicted; E_dist 0 legitimate wins. First AURC numbers with spread.
Iteration notes: initial water-fraction filter wrongly excluded
high-error scenes (fixed to errors>=8); four initial AOIs missed their
rivers at 1.28 km precision and were re-aimed.

## exp10 — rater strength vs diversity, GPU (overnight)

|Large-Base| worse than |Nano-Base| (0.0197 vs 0.0129 mean AURC, 3/7
scenes) despite Large being more accurate everywhere. Disagreement partners
need decorrelation, not strength. Note: exp10's per-scene win counts use a
3-signal set (baseline, two pairs) and are not comparable to exp09's
5-signal win counts.

## exp11 — pre-registered statistical hardening (2026-09-01)

Rule committed before fetching; 29 scenes. Headline revisions: baseline best
on 6/29 (exp09's 0/7 superseded); E_case advantage does not generalize
(p=0.07 toward worse); tile-phase most frequent winner but mean-zero; E_dist
only significant mean improvement (p=0.019), reference-quality confounded.
No signal dominates -> regime identification becomes the central problem.
Seed test vacuous (deterministic heads). Stats in exp/out/exp11_stats.csv.

## exp12 — AWF missing cells (2026-09-01)

E_dist 0.1104 and spectral-variability control 0.1287 on the same 51 AWF
errors; both far behind baseline 0.0367. README table n/a cells filled.

## exp13 — corrected statistics on the 29-scene set (2026-09-01)

Refinement pass caught three methodological problems: unaligned tile-phase
in exp05/09/11, raw AURC not comparable across scenes, i.i.d. patch
bootstrap ignoring spatial autocorrelation. Recomputed from cached exp11
features. Aligned tile-phase beats baseline 27/29 (sign p<0.001), control
19/29; E_dist's exp11 significance was a scale artifact; E_case unchanged.
First robust, control-surviving positive result. Also corrected the E_dist
definition in docs (distance to the head's training scene, not pretraining
data). Results: exp/out/exp13_corrected_stats.csv.

## exp14 — boundary ablation of tile-phase (2026-09-01)

Aligned tile-phase is equivalent to a zero-cost prediction-boundary
indicator (neighbor label disagreement on the shift-0 map): 13/29
head-to-head, p=0.71; the indicator alone beats confidence 23/29. The
headline is restated as boundary proximity, not perturbation. Explains the
AWF loss (interior point labels). Results: exp/out/exp14_boundary_ablation.csv.

## exp15 — boundary proximity + E_geo (2026-09-01)

Georeferencing recovered and verified for 23/29 scenes. Geo-first-then-
boundary is worse than boundary alone (5/23, p=0.011); geo alone 3/23.
Geo flags 3x error-enriched but precision zero on 8 scenes: OSM-vs-
WorldCover disagreement on narrow channels. E_geo sensitivity still
unmeasurable under WorldCover truth. Results: exp/out/exp15_boundary_geo.csv.

## audit + corrections (2026-09-01)

A 204-agent adversarial audit confirmed 53 findings. Code fixes: tie-aware
AURC (expected value under random tie-breaking) and a negative-|logit|
baseline in exp13/14/15 and evidence.py; tie-excluding sign tests; exp14
now scores against the pixel control; kafue and luangwa (cache leftovers,
not rule-selected) excluded from exp13-15 (27 scenes); AWF loader now warps
the 20 m/60 m band groups onto the 10 m grid instead of stretching them
(exp04/07/12 re-embedded); rasterize_polyline floor fix. Text fixes: the
E-AURC 'scale artifact' explanation for E_dist withdrawn (the oracle
subtraction cancels in differences); 'equivalent'/'adds nothing' softened
to 'statistically indistinguishable'; exp15 'rejected' softened to 'no
benefit shown' (5/9/13 W/L/T); E_geo enrichment restated as pooled
precision; README scope narrowed to linear probes on frozen encoders;
boundary-concentrated error positioned against prior work; AWF point-label
explanation demoted to hypothesis. Audit record: exp/out/audit_wf_119095bf.json.
Post-correction headline: aligned tile-phase beats confidence on 26/27
(sign p=4e-07); boundary indicator 19/27 vs confidence (p=0.05, marginal),
22/27 vs control; the two are indistinguishable head-to-head (15/12, p=0.70).

## AWF re-embedding after loader fix (2026-09-01)

Band groups now warped onto the 10 m grid with georeferencing (WarpedVRT)
instead of stretched. Re-run on GPU: Nano 0.753, Tiny 0.802, Base
0.817 val accuracy; 63 Base errors; AURC baseline 0.0363, E_case
0.0670, tile-phase 0.0489, E_dist 0.1338, control 0.1658. DS
reliabilities: nano 0.859 (measured 0.753, gap +0.106), tiny 0.916 (measured 0.802, gap +0.114), base 0.876 (measured 0.817, gap +0.059).
All AWF numbers in README/docs updated from this run.

## exp16 — boundary indicator on the AWF point-label task (2026-09-01)

Dense Base head over each validation crop (GPU). Errors reproduce exp04
(63). Script adversarially reviewed (40 agents) before recording; the
review corrected the analysis: interior test against the within-window
reference (labelled patches slightly less boundary-like than random
patches, sign p=0.008), conditional test of the score given the margin
(LRT p=0.002, coefficient 0.52 vs 3.53), cluster bootstrap over
30 tasks. AURC confidence 0.0363 vs boundary 0.0636. The "no
boundary context" explanation is withdrawn; the score is a low-margin proxy
on nine classes. Results: exp/out/exp16_awf_boundary.csv, exp16_summary.json.

## exp17 — evidence from inside the encoder (2026-09-01)

Hooked v1-Base (12 blocks, 768-d, 12 heads, no registers; sequence H,W,T,S
verified). Band-set disagreement (heads on the three S2 band-set tokens):
21/6 vs confidence (p=0.006), 16/27 vs control, one pass, better than the
two-model E_case. Depth probes 19/8 (p=0.052). Logit-lens settling 0/27,
drift 3/24, attention entropy 3/24: rejected. phi with final head: Nano
0.61, band-set probe 0.75, so disagreement value is not error decorrelation
alone. Cache exp/out/exp17_internals.npz (float16 per-block tokens).

## exp18 — Sen1Floods11 dense expert labels (2026-09-01)

Head trained on 600 valid-split tiles; scored on Bolivia (held-out region,
351 tiles) and 800 test tiles. Confidence best everywhere. Tile-phase
163/187 (p=0.22) on Bolivia, 173/305 (worse) on test; band-set 111/239;
boundary 134/217; E_case 84/265; E_dist 22/329; control 79/271. Errors on
boundaries 75% vs 21% (phenomenon holds). Reading at the time: the
WorldCover advantages were reference-error detection (tested and not
supported by exp23). Repository claims revised in
README, ledger, comparisons, signals, roadmap. Cache exp/out/exp18_feats.npz
(3.9 GB, ignored). Caveat: L1C chips through the L2A path.

## exp19 — v1 vs v1.2 (2026-09-01)

Isolated worktree + venv on olmoearth_pretrain main. v1 features match the
cache exactly. v1.2: rope_3d_mixed, one S2 band-set token per patch. Tile-
phase magnitude larger under v1.2 (0.046 vs 0.032; smaller on 6/31). Against
WorldCover both versions' tile-phase beats their confidence (26/1, 25/2), but
exp18 shows that advantage does not transfer to hand labels (cause open, exp23). Cross-version disagreement:
6/21 (v1 errors), 18/9 n.s. (v1.2). Cache exp/out/exp19_feats.npz (ignored).

## exp20 - served production output (2026-09-02)

First assessment of one of Ai2's own outputs. Ten 512-px windows of
allenai/olmoearth_lcc (EPSG:3857, about 9.55 m) read by HTTP range requests
(oe_inferencex/lcc.py; GDAL vsicurl stalls on the signed CDN redirect).
Product facts from the dataset card: band 1 change probability; bands 4-5
land cover classes with no confidence; bands 6-7 are change-category head
probabilities (80 to 99% at 255 over all pixels because "none" dominates).
Water map against WorldCover water: boundary AURC below random at 6/6 sites
with water; capture 0.88 at 5% budget (median); boundary share 0.92 among
disagreements against 0.01 among agreements. Full-legend disagreement median
49%: legend semantics, not error. Change probability: median 2.7% flagged,
1.2% ambiguous; low confidence on flagged-region edges 2.7% against 0.07%
interior. Two mistakes fixed on the way: the first run used band 6 as a
class confidence (it is not), and the assessor's oracle had the wrong sign
(fixed in assess.py; verified oracle 0.05, anti-oracle 0.66 on synthetic
data). Cache exp/out/exp20_windows.npz (ignored). Product gap to report:
export a class-head confidence alongside bands 4-5. The dataset's
annotated change points were first proposed as the next testbed and then
withdrawn: the card states they trained the model, so they are in-sample.

## exp21 - fine-tuned AWF model end to end (2026-09-02)

The user pointed out the fine-tuned checkpoints are public. Loaded
allenai/OlmoEarth-v1-FT-AWF-Base (Lightning ckpt from rslearn) into our
pretrain encoder strictly (231/231 keys), replicated rslearn's wrapper
(mean-pool T and S, legacy month timestamps) and head (bilinear x4 then 1x1
conv). Validation split, 344 expert points, 16-px and 32-px crops, 4 shifts.
Accuracy 0.881 / 0.878 (Ai2 reports 0.895; patch-logit reading 0.898). Probe
of exp16: 0.817; 28 of 41 fine-tuned errors are also probe errors. Signals
(16 px): confidence AURC 0.0262, tiling instability 0.0235 (cluster
bootstrap over 30 tasks: CI [-0.0068, +0.0010], P(better) 0.93), boundary
0.0765, probe disagreement 0.0852, NDVI-tstd control 0.0937; oracle 0.0076,
random 0.119. Capture at 20% budget: confidence 0.63, tile 0.71. Calibration
ECE 0.080, top bin 0.93 accurate at 0.99 confidence; selective accuracy
0.945 at 80% coverage. First run had two bugs (probe CSV key prefix; patch
logits instead of bilinear pixel logits), fixed before recording. Cache
exp/out/exp21_stacks.npz (ignored). Next: the other four fine-tuned
checkpoints for generality.

## exp22 - periodic artifacts in the served product (2026-09-02)

Label-free periodicity test on 5 windows of 4096 served px (3 tiles).
Geometry: served grid is Web Mercator zoom 14 (9.5546 units/px) warped from
UTM 10 m, so pipeline periods are predicted per window (4 UTM px ->
4.34-4.42 served px; UTM lines sheared by about 0.6 deg, corrected).
Method validated first on lattice-free synthetic maps (0/10 scan false
positives; seams on 5-10% of rows detectable); the first synthetic
generator (8x zoom) had a lattice of its own and every peak it produced was
an artifact - discarded before any product number was read; the warp-beat
prediction was corrected from 1/(ratio-1) to ratio/(ratio-1). Result:
19/20 product profiles peak on the patch lattice (max Bonferroni p 6e-08);
control never below p 0.006. Window seams (64-512 px, fundamental only, since
comb scores are confounded by the lattice): class map 64 px: 0 of 10, 128 px: 0 of 10, 256 px: 0 of 10, 512 px: 0 of 10,
gradient 64 px: 0 of 10, 128 px: 1 of 10, 256 px: 0 of 10, 512 px: 0 of 10, control 64 px: 0 of 10, 128 px: 0 of 10, 256 px: 1 of 10, 512 px: 0 of 10: the null rate.
In-situ detection limits: 128 px 5% in 3, 10% in 2 windows; 256 px 10% in 2, 20% in 3.
Warp beat at p<0.001 in 2/10 and 2/10 product profiles, 0/10 control.
Reading: no window-seam striping in the served v1.2 product at the stated
limits; outputs are patch-quantized (40 m), which bounds boundary accuracy.
Cache exp/out/exp22_windows.npz (ignored).

## exp23 - reference instability (2026-09-02)

Tests the exp18 reading directly. 24 rule scenes (kafue_20/50/80 dropped:
re-read B02 no longer matches the cache), WorldCover 2020 v100 and 2021
v200 fetched on each scene's recovered grid, signals as in exp13. T1:
unstable share among disagreements median 10.8% vs 0.8% among
agreements (17/3/4, p 0.003); pooled 10% of disagreements unstable. T2:
tile-phase's flagged errors not more often unstable than confidence's
(8/8/8, p 1.000). T3: on reference-stable patches tile-phase still
beats confidence 21/23 (p 7e-05) vs 22/23 on all patches; boundary
18/23 vs 15/23. The "reference error" explanation is withdrawn
(not supported by the measurable component); open candidates: shared
reference errors, 2021-to-2024 change, task properties. Next: rerun the
scenes with imagery from WorldCover's own year to test the temporal
mismatch. Cache exp/out/exp23_geo.npz (ignored).


## exp24 - the year gap (2026-09-02)

Same rule scenes with May-Sep 2021 S2 L2A (least cloudy, <5%), WorldCover
2021 on each 2021 grid, exp11 features (GPU, HF offline), head retrained on
Katima 2021 (seed 0; training fit 1.0 for both years' heads). 26 scenes
scored in both years; 2021 has more disagreements (2337 vs 1845; more on
20/27 scenes; imagery May-Aug 2021 against an annual map). Tile-phase vs
confidence 23/3/0 (p 9e-05, median gain +0.0172) against 25/1/0
(gain +0.0047) in 2024; boundary 20/6/0 vs 18/8/0; E_case 11/15/0 vs
10/16/0; E_dist 12/14/0 vs 12/14/0; control 12/14/0 vs 13/13/0; paired tile
gain not larger in 2024 (10/16/0, p 0.327). The year gap does not
explain the WorldCover wins. Remaining candidates: single-date image vs
annual composite (seasonal water margins), or a real task difference.
Next (exp25): split patches by JRC Global Surface Water seasonality and
test whether the advantage lives on seasonal-water patches. Caches
exp24_scenes.npz, exp24_feats.npz (ignored).

## exp25 - seasonal water (2026-09-02)

JRC GSW v1.3 seasonality (2020, 30 m) and occurrence on each scene grid
(2024 grids from exp23_geo; 2021 grids re-derived with the exp15 lookup and
B02 check); patch seasonal = any pixel 1-11 months. Median seasonal share
9%. T1: seasonal share among disagreements 39% vs 8% among
agreements (2024, 21/3/3, p 3e-04); 2021 41% vs 9%. T2 without
seasonal patches: tile-phase vs confidence 22/2/0 (2024, p 4e-05; all patches
23/1/0), 22/1/0 (2021, p 6e-06; all 20/3/0); boundary 19/5/0/20/3/0. T3 seasonal
only: 12/6/0 / 13/8/0. Not supported. Tooling: the scene rule's Overpass
coordinates are now cached in exp/out/rule_candidates.json (a mirror
failure had silently dropped the Zambezi scenes from one run); the JRC
asset hangs the interpreter at exit, so the script ends with os._exit.
Cache exp25_jrc.npz (ignored).

## exp26 - hand-check kit (2026-09-02)

Preparation, not evidence. Scenes by rule (largest exp13 tiling-instability
gain): shire_80 (188 disagreements), barotse (97), okavango_80 (76). For
each, the top-12 disagreements by tiling instability and by confidence
(overlap 2, 0 and 4) as 48-px true-colour, NDWI, WorldCover-2021 and
head-probability crops (exp/out/exp26_handcheck_<scene>_<ranker>.png) and
a CSV with model probability, reference label, NDWI cues, JRC seasonality
and occurrence, both ranks, and empty verdict/note columns (vocabulary:
model error / reference error / seasonal or date difference / ambiguous).
The reviewer's verdicts, once entered, are the input to the next analysis;
the NDWI cue is a crude aid and must not be treated as truth.

## Reproduction on AICR (2026-09-05)

Environment and pipeline check on a different machine, not new evidence.
Fresh clone at ~/olmoearth_inferenceX on the MGHPCC AICR cluster, built
inside a Slurm job (b200-devel, job 697056): `uv sync --extra encoder
--extra geo --python 3.12` resolved torch 2.7.1+cu128 in 2m38s, and
smoke_test.py gave a Nano/Base similarity-structure correlation of 0.5855
(the documented ~0.59 floor). exp02 with the committed cache moved aside
(job 697211, one B200, 32 s wall) selected the same scenes as the original
run (S2B_MSIL2A_20240926T081619_R121_T35KKA,
S2A_MSIL2A_20240829T080601_R078_T35KLA, both 0% cloud) and reproduced every
number: Nano 0.973 and Base 0.979 against WorldCover; E_case AURC 0.0011,
baseline 0.0009; error rate 0.021; 52 centerline patches, 0 flagged. The
regenerated exp02_cache.npz against the committed one: labels, transform
and CRS identical; max |dp| 9.7e-06 (Base), 4.5e-06 (Nano). A run with the
cache present prints "loaded cached probs/labels" and replays the metric
layer only; it does not exercise the encoder.

exp21 on the same clone (job 697617, one B200, 2m55s wall; the AWF dataset
fetched from allenai/olmoearth_projects_awf to /scratch and symlinked in as
data/awf, 1459 windows): 344 validation windows loaded in 48 s, four shifts
in 3 s at 16 px and 16 s at 32 px. All 60 numeric fields of
exp21_summary.json match the committed run to a max abs diff of 1.05e-05:
accuracy 0.881 (41 errors), confidence AURC 0.0262, tiling instability
0.0235, ECE 0.080, selective accuracy 0.945 at 80% coverage. A first
attempt (job 697356) fetched and extracted the dataset but aborted before
inference on a pipefail in the job script, not in the experiment.

**2026-09-06, chain rerun.** Job 706739 (b200-devel, one B200, fp32) reran
exp18, exp17, exp20, exp23, exp24, exp25, exp26 and `exp/summary_transfer.py`
in sequence from a fresh `git pull`, with the tracked `exp/out/` files reset
first. `git diff -- exp/out/` is empty afterwards: every tracked CSV, JSON and
PNG those scripts write is byte-identical to the committed version, and the
transfer summary printed the same river-level result (8/0, sign p = 0.0078,
n = 8). The run also regenerated the gitignored caches the follow-on
experiments read (exp17_internals, exp18_feats, exp23_geo, exp24_feats,
exp25_jrc). Log: `slurm/oeix-chain-706739.out` on the cluster.

## Encoder throughput and precision on AICR (2026-09-05)

Not new evidence about the signals; a measurement of how the encoder uses
the GPU and whether a faster numeric path changes the audit. Every
experiment in this repository runs fp32 with no TF32, autocast or compile.
exp/bench_encoder_b200.py (job 697668, one B200, torch 2.7.1+cu128, random
input) shows why that is slow: in fp32 attention runs on an sm80
memory-efficient SDPA kernel and the GEMMs on SIMT sgemm, so the tensor
cores are idle, and the flash kernel refuses fp32 outright. Per 128x128
window: fp32 18.8 ms; TF32 10.1 ms (pooled-feature drift rel 2.2e-4,
random-head score Spearman 1.000000); bf16 autocast 2.7 ms (rel 5.6e-3,
0.99993); bf16 + torch.compile 2.4 ms (rel 2.6e-3, 0.99998). Batched x16
reaches 1.8 ms and the AWF 32x32x12-month regime 1.1 ms. Values in
exp/out/bench_encoder_b200.json.

TF32 on the audit itself (job 697843): exp21 rerun with no script change,
torch.set_float32_matmul_precision("high") set before import, stacks cache
present. Inference 8 s against 19 s. 59 of the 60 numeric fields of
exp21_summary.json match fp32 to under 1e-4 - every accuracy, AURC, ECE and
selective-accuracy value - and one does not: error capture at the 20%
budget by tiling instability on 32-px crops moves from 27/42 (0.643) to
28/42 (0.667), one window crossing the budget cut. Continuous metrics are
insensitive to TF32; a discrete review-budget count can move by one
window, because tiling instability is a small standard deviation across
shifts. bf16 has not been tested on the audit. All numbers reported in the
docs remain the fp32 runs.

## exp27 oracle gate (2026-09-06)

Design check, not evidence about the signals. Before any decoder output is
read as a WorldCover class, exp/exp27_oracle_gate.py pushes synthetic 8x8
class-code patches through the exact saved target projection (Conv2d(1->768,
8), bias norm 2.07 against a weight-sum norm of 16.4, after the WorldCover
normaliser which maps codes to a monotone scalar, 10 -> 0.25, 80 -> 0.97).
Results in exp/out/exp27_oracle_gate.json: the eleven pure-class prototypes
have pairwise cosine min 0.935, median 0.996 (water vs wetland 0.9999);
token norm grows linearly with the code (4.6 -> 19.4), so class identity is
carried by magnitude, which the L2-normalised discrimination loss discards;
in water/grass mixtures the cosine to the water prototype is non-monotone in
water fraction (0.990 at 0%, 0.933 at 25%, 1.0 at 100%) while a change of
layout at fixed fraction moves it by 0.06-0.12; a ridge readout of water
fraction from perfect tokens reaches R2 0.67 raw and 0.55 after L2
normalisation. Consequence, recorded in docs/plan/roadmap.md: nearest-
prototype and retrieval readouts of the decoder are unsound, a learned
locked readout is bounded by that ceiling, and a decoder-versus-map
disagreement count cannot serve as the coupling statistic because it
reduces to the class indicator when the decoder collapses to one class. The
primary endpoint becomes reference specificity on identical cells, which
needs adjudicated labels on the river scenes. A CPU pilot of the exp27 draft
on one scene motivated this gate. The draft was committed as
exp/exp27_decoder_worldcover.py on 2026-09-17 (retired, never run past that
pilot), removed from the tree later that day after the full-tree audit, and
remains in tag v1.0.0; its pilot numbers are not recorded, because this gate
showed the prototype readout it relies on to be unsound.

## exp28 decoder self-consistency (2026-09-06)

Question: does the model's own pretraining objective, evaluated at inference,
rank the probe's errors? For each scene, K masks hide 25% of the 4-px patches
with all three S2 band-set tokens of a hidden patch masked together; the
encoder and decoder run once per mask, and each hidden token's decoded output
is compared with the frozen target projection of the true patch (cosine
distance; centred cosine; the patch-discrimination NLL at tau 0.1, minus
log n). Whole-spectrum masking is a deliberate intervention, not the
pretraining geometry (pretraining used modality_cross_random token masking,
not grouped across band sets), so a token-level complementary-pair variant
matching the pretraining marginal (50% of tokens hidden, cross-band tokens
visible, every token hidden 4 times) is scored from the same code path.
Controls: the target-only crowding NLL (the true target queried against the
same candidates with no decoder), a constant score, and two observed-input
controls (S2 within-patch variance, NDWI level). Preregistered inference: the
gain of U+ (mean of the within-scene midrank percentiles of confidence and of
decoder cosine distance) over confidence, per-river means, one-sided exact
sign test over the 8 river clusters. Errors are always those of the original
unmasked probe. Job 706755, one B200, fp32, 106 s, 0 failures; outputs
exp/out/exp28_summary.json and exp/out/exp28_decoder_consistency.csv.

Part A (27 rule-selected scenes; the exp13 error set as regenerated on the
cluster, provenance under exp31). Decoder cosine distance
(K = 8) has median E-AURC 0.0302 against 0.0118 for confidence and 0.0032 for
tile-phase; it loses to confidence 7/20/0 by scene and 2/6 by river
(one-sided p = 0.96), beats the constant score 22/5/0 and loses to both
observed-input controls (S2 variance 6/21/0, NDWI level 5/22/0). The
preregistered combination U+ gains nothing over confidence: 4/4/0 rivers,
p = 0.64 (11/16/0 by scene). The NLL scores (median 0.080) sit beside the
target-only crowding control (0.094): the discrimination loss is governed by
how crowded a target's neighbourhood is, not by the decoder. The token-mask
variant is indistinguishable from whole-spectrum masking (0.0303 against
0.0302); centring makes the score worse (0.054); K = 32 changes nothing
(0.0292). Within every reference-class x prediction-boundary stratum the
mean primary score on error and on correct patches differs by at most 0.045
and the sign is not consistent across strata. Tile-phase against confidence
in this pipeline: 26/1/0 by scene, 8/0/0 by river (one-sided p = 0.0039).

Part B (Sen1Floods11 Bolivia, hand labels, 351 tiles, head accuracy 0.912).
Pooled E-AURC: confidence 0.0105, tile-phase 0.0115, NDWI level 0.0119,
boundary 0.0313; decoder cosine distance 0.065 (23/328 tiles against
confidence), U+ 0.031 (81/270): the combination hurts.

Mechanism (the per-scene diagnostics in the summary, medians over the 27
scenes). The mean cosine between a decoded token and its true target is
0.469 (range 0.441-0.569) against 0.440 (0.396-0.499) for a shuffled target,
a gap of 0.030 (smallest scene 0.012); the mean pairwise cosine of the frozen
targets within a scene is 0.994 (0.975-0.999). The target space is nearly
collinear under cosine, the same kind of aliasing the exp27 oracle gate
measured for the WorldCover projection, so the residual carries little
per-patch information beyond what input texture already gives: it beats only
the constant score.

Verdict: rejected as an error signal on both testbeds; a valid negative that
closes the decoder-side family (exp08 occlusion, exp28 native masking) for
this checkpoint.

## exp30 last-layer Laplace on the probe head (2026-09-06)

Question: does a posterior over the probe head's weights, with the head's
confidence as its point estimate, carry an epistemic term that ranks the
probe's errors? Last-layer Laplace: N(theta, (H + lambda I)^-1) with H the
Hessian of the balanced binary cross-entropy at the trained head (bias
included, positive weight n_neg/n_pos as in train_logistic_head) and lambda
chosen post hoc by the Laplace marginal likelihood on a log grid (51 points,
1e-4 to 1e6, trained weights fixed). Scores: the logit variance
v = phi^T Sigma phi (preregistered primary), the probit-moderated confidence
-|mu| / sqrt(1 + pi v / 8), predictive entropy and mutual information under
l ~ N(mu, v) by 64-node Gauss-Hermite quadrature, the standard deviation of
the logits of 16 heads retrained on bootstrap resamples of the training
patches, and the feature norm as a diagnostic. References, controls, the U+
combination and the river-clustered one-sided sign test as exp28, through
exp/harness_ab.py (the exp28 scaffolding extracted verbatim; exp28 itself
unchanged). Job 706903, one B200, fp32, 192 s, 0 failures; outputs
exp/out/exp30_summary.json and exp/out/exp30_laplace_head.csv. A first run
(job 706827) integrated the two predictive scores with a 64-node
Gauss-Hermite rule, which the Codex cross-review showed mis-orders patches
once the variance exceeds a few hundred, as in part A; the rerun integrates
on a 4001-point logit grid and every other number is identical. The trained
weights are not the stationary point of the regularised objective (the head
is fitted without a prior), so the marginal likelihood is a surrogate; the
residual gradient norm is recorded (part A 0.98 against 0.24 for the
likelihood alone, part B 211 against 241).

Part A (27 rule scenes, the regenerated exp13 error set as exp28; head on
the 1024 katima patches).
The head fits its scene perfectly (training accuracy 1.0), so the Hessian
has rank 174 of 769 at 1e-8 of its largest eigenvalue, the marginal
likelihood picks lambda = 0.1 with 13.9 effective parameters, and the
variance is prior-dominated: mean 630 on the training patches, Spearman 0.89
(median over scenes) with the feature norm. As a ranker it is the worst
signal: median E-AURC 0.071 against 0.0118 for confidence, 3/24/0 by scene
and 0/8 by river, and it loses to all three controls (constant 5/22, S2
variance 2/25, NDWI level 3/24). The preregistered combination U+ loses to
confidence 7/20/0 by scene and 0/8 by river (one-sided p = 1.0). Against
confidence: moderated confidence 9/18 (1/7 rivers; Spearman 0.97 with
confidence), mutual information 6/21 (1/7; Spearman 0.92 with confidence),
predictive entropy 9/18 (1/7), bootstrap std 4/23 (2/6). The variance ranking is insensitive to lambda
(Spearman 0.97 at lambda/100, 0.95 at 100 lambda). Tile-phase against
confidence in this pipeline: 26/1 by scene, 8/0 by river (p = 0.0039), as
exp28.

Part B (Sen1Floods11 Bolivia; head on 127,840 valid-split patches, accuracy
0.912, 351 scored tiles). Here the Hessian has rank 768 of 769, lambda = 6.3 with
667 effective parameters, mean training variance 0.22, and the variance is
not norm (Spearman -0.12). It is still the worst signal: pooled E-AURC 0.168
against 0.0105 for confidence, 6/345 tiles, below the constant score
(58/293). Bootstrap std 0.121 (6/345). Moderated confidence and predictive
entropy are confidence itself (Spearman 1.00, pooled 0.0105; 77/152 with 122
tied tiles and 73/141 with 137). Mutual information 0.0114 (93/253). U+
0.0594 (21/330).

Mechanism. On the one-scene head the variance measures how far a feature
leaves the span of the training patches, the Bayesian form of exp13's E_dist
(13/27), and behaves like it. On the large head the weights are well
determined and the variance rises with the size of the logit (Spearman -0.53
with confidence on Bolivia, -0.48 on the scenes): it flags the patches the
head is surest about, since a large projection on a weight direction also
carries a large posterior variance along it. Neither a Bayesian nor a
bootstrap posterior over the last layer contains information about error
that the point estimate lacks.

Verdict: rejected as an error signal on both testbeds; the last-layer
posterior family (Laplace, bootstrap ensemble) is closed for these heads.

## exp31 feature-space typicality (2026-09-06)

Question: exp13's E_dist (mean cosine distance to the five nearest patches
of the head's training scene) found 13/27 against confidence and was the
worst signal on hand labels (22/329, exp18), but it conflates the reference
set with the density estimator. Here the pooled 768-d Base feature of each
patch is scored against three reference sets with three estimators each:
R1 the head's training patches (katima's 1024 in part A; the 127,840 valid
patches of the 600 valid-split tiles in part B), R2 the evaluated unit
itself, cross-fitted over 5 random folds so no patch is scored against a
reference containing it (819 reference patches per fold in A, 180 in B),
and R3 a cross-testbed pool as a proxy for a pretraining-distribution
reference (part A: the 414,225 Sen1Floods11 patches cached by exp18 from
11 countries; part B: the 27 rule scenes plus katima, 28,672 patches).
Estimators: mean cosine distance to the k = 5 nearest reference patches
(R1 in part A is exp13's E_dist), Mahalanobis distance under the
Ledoit-Wolf-shrunk covariance, and the PCA residual norm outside the top
d = min(192, n/2) principal directions (the residual term of ViM). On R1,
where the head's own training labels are legitimately available, also the
class-conditional Mahalanobis distance (Lee et al. 2018, tied covariance)
and ViM (Wang et al. 2022) adapted to the one-logit head, which fuses
confidence and typicality by construction. The feature norm is scored as a
diagnostic. Preregistered primary: the same-scene kNN score (label-free,
no external data); U+ = mean of the within-unit midrank percentiles of
confidence and of that score; inference = per-river mean gain over
confidence, one-sided exact sign test over the 8 rivers. References,
controls and scaffolding as exp28 (exp/harness_ab.py). Job 706963, one
B200, fp32, 47 s, 0 failures (job 706904 failed on a device-placement bug
before any score was computed; job 706911 ranked ViM by its softmax
probability, which saturates at 1.0 and tied the most atypical patches, so
the Codex cross-review had it replaced by the equivalent log-odds; every
other number is identical between 706911 and 706963); outputs
exp/out/exp31_summary.json and exp/out/exp31_feature_typicality.csv.

Part A (27 rule scenes, exp13 error set). The preregistered combination U+
beats confidence 14/13 by scene and 6/2 by river, one-sided p = 0.145:
short of the 7/8 needed, and it loses to tile-phase 2/25 (0/8 rivers). The
primary score alone loses to confidence 11/16 (2/6 rivers) and to the S2
variance control 5/22. No typicality score beats confidence: the R1 kNN
score (exp13's E_dist) is 13/14 (5/3 rivers, median E-AURC 0.0076 against
0.0118 for confidence, best on 2 scenes); the Gaussian scores run from 9/18 (R2
Mahalanobis; R2 PCA residual 8/19) through 5/22 (R1 Mahalanobis,
class-conditional Mahalanobis and PCA residual) to 1/26 (R3 Mahalanobis
and PCA residual), none with a river gain beyond 1/7; ViM is 0/27. Against the S2 patch-variance control only the
R1 kNN score wins (16/11); the R2 and R3 Gaussian scores lose 2/25 to 5/22.
Spearman with the NDWI-gradient control: R2 kNN 0.55, R2 Mahalanobis 0.47,
R3 kNN 0.27; the R1 kNN score correlates with confidence (0.56) and
tile-phase (0.55). The R1 kNN E-AURCs differ from the committed exp13
values by at most 6.3e-4 over the 27 scenes. Provenance: exp28, exp30 and
exp31 all score one error set, the seed-0 head on the exp11 features
regenerated on the cluster, and that set differs from the committed exp13
run (original laptop features) by one or two patches on four scenes
(barotse 98 against 97, kafue_20 78 against 77, shire_20 302 against 303,
vicfalls_up 21 against 23; identical on the other 23), so the exp13 numbers
quoted from this pipeline are its own, not a byte reproduction of exp13.
The 13/27 pattern of E_dist is reproduced. Tile-phase against confidence:
26/1 by scene, 8/0 by river.

Part B (Sen1Floods11 Bolivia, 351 scored tiles, head accuracy 0.912).
Every typicality score loses to confidence on at least 317 of 351 tiles:
pooled E-AURC 0.0527 for the best (R3 kNN) against 0.0105 for confidence;
R1 kNN 0.0593, R2 kNN 0.0628, R1 Mahalanobis 0.0663, class-conditional
0.0648, ViM 0.0929, R3 Mahalanobis 0.0831; the R1, R2 and R3 kNN scores
sit level with the S2 variance control (173/178, 161/190, 171/180). U+
0.0326, 60/291 tiles: the combination hurts.

Reading: atypicality of the frozen feature, whichever reference it is
measured against and whichever density estimates it, is not error
likelihood for these heads. The one partial exception, a 6/2-river gain of
the confidence+same-scene-kNN combination against WorldCover, does not
reach the preregistered threshold and reverses on hand labels, the pattern
of every WorldCover-only advantage in this repository. Roadmap item 9
(E_dist formalisation) closes: the AOA-style reference-sample question is
answered for the two references that can be built from this repository's
data, and only a true pretraining sample remains untested.

Verdict: rejected on both testbeds.

## exp32 the latent-MIM target space (2026-09-06)

A code reading and a CPU measurement, not a signal. From the installed
olmoearth_pretrain code and the checkpoint config: the train module is
ContrastiveLatentMIM with ema_decay (1.0, 1.0) and reinit_targets False, so
the target encoder is a copy of the online encoder that is never updated;
token_exit_cfg is 0 for every modality, so a target is the target encoder's
patch embedding of the raw patch with no attention; the loss is patch
discrimination (prediction and target L2-normalised, softmax at tau 0.1 over
the masked tokens of the same sample, mask_other_samples True) plus a
0.1-weighted contrastive term on pooled tokens; masking encodes and decodes
50% of the tokens, with the maps decode-only. exp/exp32_target_space.py,
outputs exp/out/exp32_target_space.csv and exp32_summary.json.

The shipped target encoder is the random initialisation: every LayerNorm
weight is exactly 1 and every bias exactly 0 (the online encoder's LayerNorm
weights have std 0.14), its matrices sit at init scale (std
0.029 against 0.040), and its three S2 patch projections have drifted to
cosine 0.48, 0.44, 0.56 from the online encoder's. The targets are therefore frozen
random linear projections of the raw 8x8x12 patch.

On four committed 128-px scenes (barotse, okavango_80, shire_80, kazungula),
three band sets each: random pairs of target tokens have cosine 0.991
(online-encoder tokens 0.60); one direction holds 99.2% of the raw
target energy (61% for the encoder); after centring the top direction
still holds 85% of the variance and the effective rank is 2.0, against 54
for the encoder outputs; the token norm varies by 9% (CV). A random
projection preserves the raw patch covariance, which is dominated by
brightness, and the targets are not normalised per patch (MAE does that;
the Latent MIM recipe of Wei et al., arXiv:2407.15837, uses an EMA target
with within-image patch InfoNCE and a gap-4 stochastic grid against local
copying), so the discrimination loss can mostly tell bright from dark.

Consequences already recorded elsewhere now have their mechanism: the
class aliasing of the exp27 gate (class in token norm, prototypes at cosine
0.996), the uninformative decoder residual of exp28 (decoded-to-true cosine
0.47 against 0.43 shuffled; NLL near log n), and the failure of every
decoder-side signal. Two follow-ups are issues #10 (a predictor with a
whitened target on frozen features, the last latent-MIM reading with a
clean target) and #11 (the pretraining-target recommendation).

## exp33 context-prediction residual with a whitened target (2026-09-07)

Question: exp28 read the shipped decoder's masked-token residual and found
nothing, and exp32 traced that to a target of effective rank 2. Does the
latent-MIM reading carry error information once the target is not
degenerate? The frozen encoder's pooled tokens are PCA-whitened in their
top 64 directions (fitted on the predictor's training units only), a small
transformer predictor (two pre-norm layers, width 128, learned mask token,
sinusoidal 2-d positions) is trained label-free to reconstruct hidden
patches from the rest of the unit (MSE, 25% random masks, Adam 1e-3, 400
steps), and each patch is scored by its whitened MSE residual over K = 8
quarter masks (every patch hidden twice, as exp28). Companions: the cosine
form, the mean predictor (the top-d Mahalanobis distance of exp31) and the
same predictor with every patch hidden (its position-only prior), which
gives the context gain 1 - MSE(25% hidden) / MSE(all hidden). Part A
cross-fits over three river-disjoint folds (Zambezi + Luangwa; Cuando +
Kafue + Okavango; Rovuma + Save + Shire) with katima, which lies on the
Zambezi, in the two pools that do not hold the Zambezi out; part B trains
on the 600 valid tiles and scores Bolivia. References, controls, U+ and the
river test as exp28 (exp/harness_ab.py). Preregistered null: the residual
is a boundary detector. Run of record: job 714859, one B200, fp32, 32 s, 0
failures. Earlier runs: 714647 failed in part A on a cache-only scene with
no river cluster; 714672 had katima in every pool, which the Codex review
of exp34 flagged as not strictly river-disjoint for the Zambezi fold
(every count below moved by at most one scene between 714672 and 714859).
Outputs exp/out/exp33_summary.json and exp/out/exp33_context_predictor.csv.

The target swap works. Whitening keeps 78% of the variance in part A and
69% in part B; the training loss falls from 1.3-1.4 to 0.29-0.33 (A) and
0.36 (B) in whitened units; on held-out rivers the predictor explains a
median 57% of the whitened variance beyond its position-only prior (48% to
66% across scenes; 52% on the Zambezi fold, whose pool lacks katima), and
70% on Bolivia. Against exp28, where the shipped
decoder's decoded-to-true cosine was 0.469 against 0.431 for a shuffled
target, the objective is now predictable from context.

The residual is not an error signal. Part A: the whitened residual against
confidence is 13/14 by scene and 2/6 by river (one-sided p = 0.96), median
E-AURC 0.0126 against 0.0118; it loses to tile-phase 4/23, to the boundary
indicator 3/24 and to the S2 patch-variance control 11/16; the
preregistered combination U+ is 14/13 by scene and 6/2 by river (p =
0.145), the same 6/2 as exp31's combination and short of the 7/8
threshold. The cosine form is 9/18, the Mahalanobis top-d 13/14, the
position-only residual 13/14. Within scenes the residual correlates with
the S2 patch-variance control (Spearman median 0.56) and the NDWI-gradient
control (0.45) more than with the boundary indicator (0.31), and not with
confidence (0.03). Tile-phase against confidence: 8/0 by river, as before.
Part B (351 scored tiles, head accuracy 0.912): the residual loses to
confidence 12/339, pooled E-AURC 0.0664 against 0.0105; U+ 45/306 (pooled
0.0331); Spearman with S2 variance 0.57 pooled, with boundary 0.22.

Reading: what the predictor cannot predict from context is input texture
(patches whose neighbourhood does not determine them), not model error; the
preregistered null named the boundary indicator, and the observed correlate
is texture more than boundary. Runs of the same design differ by one tile
in the Bolivia U+ count (45/306 in jobs 714647 and 714859, 46/305 in
714672), GPU nondeterminism in the predictor training.

Verdict: rejected as an error signal on both testbeds. The constructive
half stands for issue #11: normalising the target turns a rank-2 objective
into one that context predicts at 60-70%, so the pretraining-target
recommendation is strengthened while the residual-as-error-signal idea is
closed. A discrete-target variant would ask the same question with the
same expected answer and is not scheduled.

## exp34 three further readings of the re-targeted objective (2026-09-07)

Question: exp33 established that a whitened target makes the latent-MIM
objective predictable and that its residual ranks input texture. Do the
readings of the proposal that ask a different question fare better? (a)
Discrete target, HuBERT-style: k-means with K = 128 over the whitened
training tokens, a cluster predictor trained with cross-entropy on 25%
hidden patches, scored by the true-cluster NLL and by the predictive
entropy of the cluster distribution, which needs no true token. (b) Gap
masking, Latent MIM-style: 3x3 holes centred on a stride-6 lattice, 36
phases so every patch is a hole centre once, the residual scored at the
centre only, the predictor trained on hole masks with the loss on centres.
(c) The exp33 residual projected onto the water head's weight direction in
whitened coordinates. The exp33 residual is recomputed as the reference
variant. Whitening, predictor, folds, controls, U+ and the river test as
exp33; preregistered primary = the predictive entropy; null = every variant
tracks S2 patch variance and the boundary indicator. Job 714849, one B200,
fp32, 50 s, 0 failures (a first run, job 714823, had katima in every pool;
the Codex review pointed out that katima lies on the Zambezi, so the rerun
keeps it out of the Zambezi-held-out pool and the numbers below are the
rerun's; the two runs agree to within one scene on every count). Outputs
exp/out/exp34_summary.json and exp/out/exp34_retarget_variants.csv;
exp/exp34_retarget_variants.py. The correlations with the two controls use
tie-averaged ranks (oe_inferencex.stats.spearman), since the boundary
indicator has nine levels; the harness's per-reference Spearman columns
keep exp14's positional form.

Part A (27 rule scenes). The preregistered combination U+ (confidence +
entropy) loses to confidence 7/20 by scene and 0/8 by river (p = 1.0); the
entropy alone is 5/22 and 1/7 (median E-AURC 0.0431 against 0.0118 for
confidence and 0.0467 for the constant score), the cluster NLL 8/19 and
2/6, the decision-direction residual 11/16 and 2/6 (the retained whitened
subspace carries 98-99% of the head's training logit variance, so the
projection loses little), the exp33 residual 13/14 and 2/6 (median 0.0126,
exp33 gave 0.0125: GPU nondeterminism in the predictor training). The
gap-masked residual is the only variant at parity with confidence, 13/14
by scene and 4/4 by river (p = 0.64), median 0.0085, best on 2 scenes; it
loses to tile-phase 6/21 and to the boundary indicator 5/22 (tile-phase
against confidence 8/0 by river, as before). Correlates within scenes
(medians, tie-averaged ranks): the gap residual keeps most of the texture
correlation (Spearman 0.43 with S2 variance, 0.35 with boundary, against
0.56 and 0.37 for the exp33 residual); the entropy and the NLL are nearly
free of it (0.05 and 0.14 with S2 variance; 0.03 and 0.18 with boundary)
and of confidence (-0.19, -0.12), so their failure is not texture:
context-uncertainty about a patch's cluster is simply unrelated to the
probe's errors. Caveat for the
entropy in part A: the cluster predictor overfits its 20 training scenes
(training cross-entropy 0.2 nats against a held-out NLL of 3.9, one nat
below chance at log K = 4.85, with a predictive entropy of 1.5), so its
uncertainty is overconfident there.

Part B (Sen1Floods11 Bolivia, 351 scored tiles). The cluster predictor is
roughly calibrated here (held-out NLL 1.69 against entropy 1.88; training
cross-entropy 1.39 on 600 tiles) and the null holds: entropy 17/334
against confidence (pooled E-AURC 0.072 against 0.0105), U+ 60/290
(0.033); NLL 14/337; gap residual 22/329 (0.064; Spearman 0.60 with S2
variance); decision-direction residual 8/343 (0.080; Spearman 0.03 with S2
variance and 0.09 with boundary, so noise with respect to everything
measured; the subspace carries 95% of the training logit variance); exp33
residual 12/339.

Verdict: rejected on both testbeds, all three readings. With exp28 (frozen
target), exp33 (whitened target) and exp34 (discrete target, gap masking,
decision direction), no reading of OlmoEarth's latent-MIM objective at
inference ranks the probe's errors better than its confidence; the family
is closed for this checkpoint. What survives is the exp33 measurement that
a normalised target makes the objective predictable, which is the
pretraining recommendation of issue #11.

## exp35 operating points at fixed review budgets (2026-09-07)

Question (roadmap item 3, issue #9): every comparison so far ranks by
AURC; does any supported signal help confidence at a fixed review budget,
which is what a reviewer with a budget actually uses? Metric: the expected
fraction of a unit's errors inside its round(b n) most suspect windows
under random tie-breaking (oe_inferencex.metrics.capture_at_budget_expected,
so the nine-level boundary indicator is neither credited nor penalised for
raster order; exp21's stable-sort form is reported alongside), at budgets of
5, 10 and 20%. Preregistered: tiling instability against confidence at the
20% budget on Sen1Floods11 Bolivia, per-tile one-sided exact sign test,
because exp21's hint (0.71 against 0.63 on the fine-tuned model) was
directional. Secondary: the fine-tuned AWF model (exp21's per-window table,
bootstrap over its 30 task clusters), the WorldCover scenes (one vote per
river), and every other signal and budget with two-sided tests. Pooled
Bolivia estimates carry a bootstrap over all 440 tiles with valid patches
(the Codex review caught a first version that resampled only the 351
scored tiles). exp/exp35_operating_points.py; job 715760, one B200, fp32,
333 s, 0 failures (job 715729 ran the pre-review code with the same
per-tile and AWF numbers); outputs exp/out/exp35_summary.json and
exp/out/exp35_operating_points.csv.

Bolivia (hand labels; 351 scored tiles, 440 pooled). The preregistered
test is null: tiling instability against confidence at 20% is 116/112 with
123 ties (one-sided p = 0.42), and pooled it captures 0.707 of the errors
against 0.749 for confidence (bootstrap CI of the gain [-0.066, -0.019]);
at 5% and 10% it is 137/132 and 134/124 by tile with pooled gains again
below zero. The boundary indicator is the one signal that beats confidence
at an operating point: at the 5% budget it captures 0.286 of the errors
against 0.259 (bootstrap CI [+0.009, +0.046], P(better) 1.00; by tile
181/147/23, two-sided p = 0.068), while it loses to confidence on AURC
(131/220) and at 20% (0.667 against 0.749, CI [-0.114, -0.050]); at 5% it
also beats both pixel controls (NDWI gradient 0.188, NDWI level 0.242).
Two cautions: the finding is secondary, not preregistered, and the
NDWI-level control itself beats confidence at the 20% budget pooled (0.795
against 0.749, CI [+0.010, +0.079]; by tile 137/115/99, p = 0.19), which
says a single pooled operating point is a noisy criterion on this testbed.
The NDWI-gradient control, the S2-variance control and the constant score
lose at every budget.

AWF fine-tuned model (expert points, 344 windows, 30 tasks). On the 16-px
crops tiling instability captures 0.707 against 0.634 at 20% (gain +0.073,
task-bootstrap CI [+0.000, +0.190], P(better) 0.93), reproducing exp21's
hint; on the 32-px crops the same comparison reverses (0.643 against 0.690,
P 0.36), and at 5% and 10% it is behind on both crops. Boundary, probe
disagreement and the NDVI control lose at every budget with intervals below
zero at 20%.

WorldCover scenes (weak reference). Tiling instability beats confidence
8/0 by river at 10% and 20% and 7/1 at 5%, as on AURC; the boundary
indicator is 7/1 by river at every budget where its AURC vote is 5/3, so a
coarse score fares better at fixed budgets than under AURC, which is the
metric difference the issue named; the pixel controls sit at 4/4 to 6/2.

Reading: the operating-point view does not change the standing
conclusion. On expert labels no signal beats confidence at a fixed budget
with preregistered support; the one operating point where a constructed
cue does better, boundary at 5% on Bolivia, is small (2.7 points of
capture), secondary, and reverses at larger budgets, and a no-model control
shows a comparable pooled win at 20%. Roadmap item 3 closes; the boundary
indicator's role as a triage cue gains one qualified operating point.

## exp36 dihedral consistency, and boundary first then confidence (2026-09-07)

Two label-free candidates with separate preregistered tests. (1) Dihedral
consistency: OlmoEarth v1 was pretrained with flip-and-rotate augmentation
(the checkpoint's transform_config), so predictions on the eight flips and
rotations of a window should agree; each transformed window is encoded and
scored by the same head, the probability map is mapped back, and the signal
is the standard deviation over the eight maps. Preregistered: against
confidence, one vote per river on the 27 scenes and per tile on Bolivia; U+
= midrank mean of confidence and dihedral consistency. (2) Boundary first,
then confidence: exp35 found the boundary indicator beats confidence at a
5% budget and loses at 20%; the reviewer's rule is lexicographic, boundary
patches ordered by confidence first, then the interior by confidence, no
parameter. Preregistered: capture at the 5% and 10% budgets against
confidence, per-tile one-sided exact sign tests on Bolivia and a bootstrap
over the 30 tasks of the fine-tuned AWF model (exp21's table); 20% and the
WorldCover scenes alongside. exp/exp36_dihedral_lexicographic.py; one B200 job, 716511 on commit a28d29b,
429 s (a first run, 716293, on e7dd569 had the pooled bootstrap of the
lexicographic rule built from per-tile midranks, caught in the Codex
review; its per-tile, river and AWF numbers are identical).
Outputs exp/out/exp36_summary.json and exp/out/exp36_dihedral_lexicographic.csv.

Dihedral consistency. Part A: against confidence 17/10 by scene and 4/4 by
river (one-sided p = 0.64), median E-AURC 0.0077 against 0.0118; it loses
to tile-phase 5/22 (0/8 rivers) and to the boundary indicator 10/17;
Spearman with confidence 0.67 and with tile-phase 0.57 (medians). U+ 19/8
by scene, 4/4 by river. Part B (351 tiles): 154/196 against confidence,
pooled E-AURC 0.0110 against 0.0105; Spearman with confidence 0.94; U+ is
201/148 by tile with a worse pooled E-AURC (0.0223), so its per-tile edge
does not survive pooling. Capture at 5, 10 and 20%: below confidence at
every budget, pooled intervals below zero. The identity transform
reproduces the cached probabilities to a maximum absolute difference of
0.013 in part A (cached exp11 features against a fresh forward pass on the
B200) and 0.007 in part B (the float16 feature cache); the eight maps of
a window share one forward path, so the signal is unaffected. Verdict:
rejected as a ranker; it behaves like a smoothed confidence, not like
tile-phase.

Boundary first, then confidence. Bolivia hand labels: at the 5% budget it
beats confidence on 85 tiles, loses on 31 and ties on 235 (one-sided
p = 2.7e-7; pooled 0.274 against 0.259, tile
bootstrap CI of the gain [+0.009, +0.021]); at 10% 112/48/191 (p = 2.3e-7; pooled 0.494 against 0.465, CI [+0.015, +0.039]); at 20%
121/77/153 by tile (two-sided p = 0.002) with the pooled gain no longer
positive (0.732 against 0.749, CI [-0.054, +0.013]). The ties are the tiles where the least-confident
windows are already boundary windows, so the two orders pick the same
review set. On E-AURC it loses to confidence on Bolivia (pooled 0.0165
against 0.0105), as the boundary indicator does: the gain is an
operating-point gain, not a ranking gain. AWF fine-tuned model (exp21's
344 windows, 30 tasks): capture identical to confidence at 5% on both crops
(0.220 and 0.214; the 17 least-confident windows are all boundary windows),
equal at 10% on the 16-px crop and behind on the 32-px crop (0.381 against
0.429), behind at 20% (0.634 against 0.634 and 0.571 against 0.690);
E-AURC worse (0.037 against 0.026). WorldCover scenes: 25/2 by scene and
8/0 by river on E-AURC, 8/0 by river on capture at every budget, as the
boundary indicator's WorldCover wins predict.

Reading: the first preregistered result on expert labels where a
constructed rule beats confidence, and a bounded one. Ordering the review
by boundary first and confidence within captures 1.5 to 3 points more of
the errors at 5% and 10% budgets on Sen1Floods11 Bolivia, gains nothing on
the AWF point task where the low-margin windows are boundary windows
anyway, and is behind by 20%. It is a triage rule, consistent with the
boundary indicator's standing as a triage cue, and it costs nothing at
inference.

## exp37 cue enrichment on identical windows (2026-09-07)

The first build of the explanation layer (docs/plan/roadmap.md, "Explanation
layer"; docs/results/explanation.md): `oe_inferencex.explain` holds a cue
library with measured shares, derives the boundary and low-confidence cues
from an assessment, accepts caller-derived cues, and reports per review
window which cues fire, their co-occurrence and the windows no cue explains
(`explain_review_set`); `cue_enrichment` is the validation (share among
errors against share among correct, cluster bootstrap of the ratio). exp37
measured the five label-free cues the package derives on identical windows,
one error definition per testbed: boundary (indicator > 0), low confidence
(20% least confident of the unit, ties included), unstable (tile-phase top
20%), NDWI-ambiguous (|patch-mean NDWI| < 0.1), flip/rotation disagreement
(exp36's cached std over the 8 transforms, top 20%). Part A: 27 rule scenes,
27,648 windows, 1,844 errors, scene-clustered bootstrap, per-river shares.
Part B: Bolivia, 81,984 valid windows in 440 tiles, 7,248 errors of the
exp18 head, tile-clustered bootstrap. Descriptive; no preregistered test.
exp/exp37_cue_enrichment.py; one B200 job, 716704 on a0130a2, 24 s, 0
failures. Outputs exp/out/exp37_summary.json, exp37_cue_enrichment.csv, and
the per-window tables exp37_patches_scenes.npz and exp37_patches_bolivia.npz
(compressed) from which tests/test_recorded.py recomputes the shares. The
review sets use the assessor's own order (assess.review_order, ties by
descending raster position; the Codex review caught an ascending variant).

Bolivia shares among errors / correct and enrichment [CI]: boundary
0.750 / 0.214, 3.5x [3.2, 3.8]; low confidence 0.589 / 0.163, 3.6x [3.4,
3.9]; unstable 0.583 / 0.164, 3.6x [3.3, 3.8]; NDWI-ambiguous 0.483 /
0.067, 7.2x [6.2, 8.5], error rate among its windows 0.41 against a base
rate of 0.088; dihedral 0.579 / 0.164, 3.5x [3.3, 3.8]. Scenes: boundary
4.3x, low confidence 3.3x, unstable 4.6x, NDWI 3.8x, dihedral 3.9x; every
cue enriched on 8/8 rivers, NDWI on 7/8. 95% of Bolivia errors carry at
least one cue (93.5% on the scenes), 80% two or more, 3.0 cues per error
window against 0.8 per correct one; 65% of correct windows carry none.
Inside confidence's 5% review set (4,043 windows, error rate 0.379): 73%
boundary, 88% unstable, 95% dihedral, 27% NDWI; error rate with / without
the cue: boundary 0.46 / 0.16, NDWI 0.51 / 0.33, unstable 0.38 / 0.37,
dihedral 0.38 / 0.41; at 20%: boundary 0.38 / 0.09, NDWI 0.47 / 0.20,
unstable 0.29 / 0.19. Co-occurrence among errors: boundary with low
confidence 50%, with unstable 52%, with dihedral 49%; NDWI with each of
the others 24 to 35%.

Reading: boundary, low confidence, instability and dihedral disagreement
are one phenomenon seen four ways (a low-margin window on a class boundary
flips under any perturbation), so a typical review window's explanation is
three facets of one fact; spectral ambiguity is the one cue that adds
information inside the review set, and it is task-specific. The library in
the package now carries these shares (tests check them against the
summary). Whether an NDWI-first review order captures more errors at a
budget is a preregistered question for a later experiment.

## exp38 spectral ambiguity first, then boundary, then confidence (2026-09-08)

The order that exp37 suggested: NDWI-ambiguous windows (|patch-mean NDWI|
< 0.1) first, ordered by confidence, then boundary windows, then the rest.
Preregistered against exp36's boundary-first order on Bolivia at the 5% and
10% budgets: per-tile one-sided exact sign tests over the 351 tiles with
3 <= errors <= n - 3, and one tile bootstrap (2000 resamples over the 440
tiles with valid windows, shared by every comparison) of the pooled gain
with the scores rebuilt from a global confidence midrank per resample.
Secondary: against confidence, the 20% budget, the NDWI-then-confidence
order, and the 27 WorldCover scenes with one vote per river. Stated
caveat: the cue was chosen after seeing exp37's Bolivia enrichment on the
same tiles (its threshold was fixed before exp37; the river vote is the
check that shares no windows with the choice). Computed locally on CPU from
exp37's per-window tables, exp/exp38_ndwi_first_order.py, 340 s; the
tables reproduce exp36's per-tile counts exactly (85/31/235 and
112/48/191), recorded in the summary. Outputs exp/out/exp38_summary.json
and exp/out/exp38_ndwi_first.csv (per-unit captures).

Primary. Per tile the NDWI-first order beats the boundary-first order at
5% (151 better, 100 worse, 100 tied; one-sided p = 7.7e-4) and at 10%
(146/86/119, p = 4.9e-5). Pooled over the whole testbed it does not: 0.292
against 0.274 at 5% (CI of the gain [-0.002, +0.037]) and 0.481 against
0.494 at 10% (CI [-0.053, +0.033]). The two statistics answer different
operating modes. With a budget per tile the order falls back to boundary
then confidence wherever a tile has few ambiguous windows and gains from
them where it has many; with one budget over the whole area the ambiguous
windows of the NDWI-heavy tiles (8,500 of 82,000 windows carry the cue,
error rate 0.41) fill the 10% set and crowd out the low-confidence boundary
windows of the other tiles. Secondary: at 20% the NDWI-first order is
clearly ahead both ways (119/54/178; pooled 0.831 against 0.732, CI [+0.071,
+0.130]); against confidence it is ahead pooled at 5% (0.292 against 0.259,
CI [+0.014, +0.052]) and 20% (0.831 against 0.749), not at 10%; the
NDWI-then-confidence order without the boundary level is within 0.003 of
it everywhere. WorldCover scenes: against the boundary-first order the
rivers vote 2/6, 3/5 and 3/4 at the three budgets; against confidence 4/4,
4/4 and 6/1 (p = 0.062), while the boundary-first order stays 8/0 at every
budget.

Verdict: mixed, not supported at the preregistered level (the per-tile
tests pass, the pooled bootstrap does not). The boundary-first order
remains the supported triage rule. Spectral ambiguity is a task-specific
cue that explains errors (exp37) and, with a per-tile or a loose budget,
finds more of them on hand labels; as a global ordering at tight budgets
it does not, and on the WorldCover reference it hurts.

## exp39 neighbourhood contradiction in three embedding spaces (2026-09-08)

Origin: a brainstorm on whether a representation-learning result could give
an inference-time error signal beyond the head's margin; its first proposal
was neighbourhood contradiction in an out-of-family embedding space. For a
query window with prediction y, take its k = 32 nearest neighbours by cosine
in an embedding space from a bank of windows that share no tile or river
with the query, and score the share of neighbours the same head predicts
differently (contradiction = 1 - q(y)); the neighbour vote's binary entropy
is a secondary reading. No label enters: the bank carries the model's own
predictions. Three spaces, since the ablations are the test: the frozen
OlmoEarth features (exp11 and exp18 caches); per-window pixel statistics
(means of the twelve log bands, NDWI mean and std, standardised); AnySat
(torch.hub gastruc/anysat, base, fp32, one date, bands standardised per
testbed). Banks: the Sen1Floods11 test split as exp18 sampled it (800 tiles,
seed 1, 180,000 windows, never used to train the head) for Bolivia; the
rule scenes on other rivers for the scenes. Preregistered at the 10%
budget on Bolivia, per-tile one-sided sign tests over the 351 tiles with
3 <= errors <= n - 3 and a tile bootstrap of the pooled gain, river votes on
the scenes alongside: P1, the OlmoEarth-space contradiction beats
confidence; P2, the AnySat-space contradiction beats confidence and beats
the OlmoEarth space; falsification, no hand-label gain or a gain the
pixel-statistics ablation reproduces. One-sided only for those pairs;
everything else two-sided. exp/exp39_neighbour_contradiction.py, one
B200 job, 725563 on c379c5d, 139 s, 0 failures; Codex review before the
run fixed four blocking defects (AnySat output layout, bank/feature
alignment, NaN signals reaching the scorer, a same-river fallback).
Outputs exp/out/exp39_summary.json, exp39_neighbour_contradiction.csv,
exp39_cache.npz (per-window scores).

AnySat, a finding before the result. Its patch output at 40 m on a
single-date input is dominated by position within the tile: the features
barely move when the content shifts by one window (correlation with the
unshifted map 0.97 to 0.99 after centring, against 0.58 to 0.69 with the
correctly shifted map). Only the local half of its dense per-pixel output
moves with the content (1.00 along the right axis, 0.31 to 0.52 along the
wrong one), so that half, pooled to the window, is the P2 space and the
contextual half a secondary space; the checks are recorded in the summary.

Bolivia hand labels (351 scored tiles; capture of the errors inside the
budget, then E-AURC). P1 passed: the OlmoEarth-space contradiction beats
confidence at 10% on 169 tiles, loses on 126, ties on 56 (one-sided
p = 0.0072), pooled 0.530 against 0.465 (CI of the gain [+0.027, +0.095]);
at 5% 189/113/49, pooled 0.324 against 0.259 (CI [+0.040, +0.088]); at
20% null (140/125/86, CI spanning zero). On E-AURC it does not beat
confidence (177/173, pooled 0.0134 against 0.0105): an operating-point
gain, like the boundary rule, not a ranking gain. P2, half: AnySat's local
embedding beats confidence at 10% (172/127/52, p = 0.0054; pooled 0.513
against 0.465, CI [+0.009, +0.088]) but not the OlmoEarth space (159/143/49,
p = 0.19; pooled 0.513 against 0.530, CI [-0.054, +0.025]); the contextual
AnySat space is far below confidence (0.137 against 0.259 at 5%). The
falsification clause fired, and then some: the pixel-statistics ablation
beats confidence at every budget by a wide margin (5%: 236/70/45,
p = 3e-22, pooled 0.412 against 0.259, CI [+0.120, +0.187]; 10%:
214/89/48, p = 5e-13, pooled 0.682 against 0.465, CI [+0.177, +0.257];
20%: 176/91/84, pooled 0.891 against 0.749) and on E-AURC, per tile
219/131 (p = 3e-6) and pooled 0.0093 against 0.0105, the first hand-label
E-AURC win in the repository; it also beats tile-phase 227/123, the
boundary indicator 256/90 and the NDWI-gradient control 277/73. Its
Spearman with confidence is 0.60 (OlmoEarth space 0.70, AnySat local 0.56,
AnySat context 0.14). Strata (error rate among the top fifth by score
against the rest, ties included): within the least confident quintile
0.79 against 0.21, the next 0.32 against 0.024, the middle 0.050 against
0.004; boundary windows 0.72 against 0.13; NDWI-clear windows 0.20 against
0.006. The preregistered combination (confidence + AnySat local, U+) wins
per tile (226/123 on E-AURC) but loses pooled at every budget; the 0.75 /
0.25 weighting likewise.

WorldCover scenes (27, one vote per river). OlmoEarth-space contradiction
against confidence: E-AURC 21/6 by scene (p = 0.006), 5/3 by river
(p = 0.36); capture 7/1 rivers at every budget (p = 0.035 one-sided at 10%,
0.070 two-sided at 5 and 20%); median E-AURC 0.0063 against 0.0118. The
pixel-statistics contradiction is null here: 14/13 scenes, 4/4 rivers on
E-AURC, 5/3, 4/4, 4/4 on capture. AnySat local 17/10 scenes, 5/3 rivers,
capture 7/1, 6/2, 5/3. Best per scene most often AnySat context (11),
tile-phase (5), OlmoEarth contradiction (3).

Verdicts. P1 supported on hand labels at tight budgets, directionally
positive on WorldCover, no ranking gain. P2 rejected: an out-of-family
representation adds nothing over OlmoEarth's own, and AnySat's contextual
features are not usable at this scale. The semantic-neighbourhood reading
is falsified by its own ablation: what the mechanism measures is the
model's inconsistency across windows that look alike, and windows look
alike, for this task, in fourteen spectral statistics better than in any
768-dimensional embedding. That inconsistency is the finding: label-free,
one bank of the model's own predictions, and stronger than confidence on
hand labels at every budget and on E-AURC. It was the ablation, not a
preregistered candidate, and it is null on the WorldCover reference, so
its status is a strong secondary finding until a preregistered replication
on an independent expert testbed (exp40, the Sen1Floods11 test split as
queries with Bolivia as the bank) passes.

## exp40 replication of the pixel-statistics contradiction on the test split (2026-09-08)

The preregistered replication of exp39's ablation result on labels it had
not seen: queries are the Sen1Floods11 test split as exp18 sampled it (800
tiles from several regions and events, 171,861 valid windows, head
accuracy 0.953, 483 tiles scored), the bank is the 441 Bolivia tiles with
the same head's predictions, the head retrained from the valid split
exactly as the harness does. Primary, one-sided, the pixel-statistics
contradiction against confidence: capture at the 10% budget per tile and
the tile bootstrap of the pooled gain (95% interval excluding zero), and
E-AURC per tile; support needs all three. exp/exp40_pixel_contradiction_replication.py,
one B200 job, 725652 on 558ab52, 131 s, 0 failures; Codex review found no
blocking defect. Outputs exp/out/exp40_summary.json, exp40_pixel_contradiction_replication.csv,
exp40_cache.npz.

Result: not supported, on every primary test. At 10% the pixel contradiction
is 183 tiles better, 195 worse, 105 tied (p = 0.75), pooled 0.530 against
0.643 for confidence (CI of the gain [-0.160, -0.066]); at 20% 108/200/175,
pooled 0.722 against 0.824; at 5% it is ahead per tile (237/182/64,
two-sided p = 0.008) but not pooled (0.367 against 0.417, CI [-0.095,
+0.004]). On E-AURC it loses per tile 215/267 (one-sided p = 0.99) and
pooled 0.0149 against 0.0096. The OlmoEarth-space contradiction is far
worse (97/301 at 10%, pooled 0.334 against 0.643; E-AURC 116/365). The
pixel space still beats the OlmoEarth space here (280/121 at 10%), so the
ordering of the spaces replicates while the win over confidence does not.
Strata: the score separates error rates inside the least confident
quintile (0.43 against 0.13) and among boundary windows (0.43 against
0.10), weaker than on Bolivia (0.79 against 0.21, 0.72 against 0.13).

Reading. The two runs differ in what the bank covers. In exp39 the queries
were one event and the bank 800 tiles from many regions, so every query
had look-alikes; in exp40 the queries span many regions and the bank is
one event, so the nearest neighbours of a window from another region are
often not look-alikes at all, and their contradiction measures bank
mismatch rather than the model's inconsistency. Confidence is also
stronger on this split (error rate 4.7% against 8.8%). Whether a bank that
covers the query's regions restores the gain is the open question the
result leaves; it would be a third preregistration (queries and bank both
from the test split, the bank excluding the query's region), not a
re-reading of this one. Under the protocol both contradiction scores stand
as Bolivia-only findings: a preregistered pass on one event and a
preregistered failure on the multi-region split. Neither is supported.

## exp41 two-view disagreement from the paper embeddings (2026-09-08)

The cross-model question that exp07, exp10 and exp39 left: does an outside
representation err where OlmoEarth does not, so that disagreement between
two heads on two views flags OlmoEarth's errors? Ai2's
allenai/olmoearth-paper-embeddings (row-aligned embeddings of 26 models on
the paper's tasks, with labels) allows the test without an encoder pass.
One multinomial logistic probe per model (L-BFGS, standardised features)
fitted on 80% of the valid units; the held-out 20% selects the outside
partner with the lowest error correlation (phi) with OlmoEarth Base;
same-family references nano, tiny, large. Preregistered: the selected
partner's confidence-weighted disagreement against OlmoEarth's confidence
(top-1 minus top-2 logit), Sen1Floods11 (Sentinel-1 input in this dump,
64-px chips, 4-px windows labelled by majority, chips as clusters) at the
10% budget per chip and pooled and on E-AURC per chip, one-sided; AWF
Sentinel-2 points pooled with a sample bootstrap; falsification if the
outside partners' error correlation is not below the family references'.
Limitation stated before the run: no imagery in the dump, so no pixel
control. exp/exp41_two_view_disagreement.py, one B200 job, 726248 on
c25273f, 416 s; Codex review before the run fixed a phi overflow, an
alignment check and the memory plan. Four Sen1Floods11 partners (CROMA,
TerraMind, CopernicusFM, Satlas) embed the chips on other grids and were
dropped by the 4-px labelling; three outside partners remained (Clay,
Galileo, Panopticon). Outputs exp/out/exp41_summary.json, exp41_two_view.csv,
exp41_cache.npz.

Sen1Floods11 (592,385 test windows, 49,386 errors of the OlmoEarth probe,
accuracy 0.917; 1,579 chips scored). The falsification fired first: the
outside partners' errors are as correlated with OlmoEarth's as its own
family's. On the test split the probability that a partner is wrong where
OlmoEarth is wrong is 0.80 to 0.82 for every model, Clay, Galileo and
Panopticon alike, nano, tiny and large alike, against 0.016 to 0.021 where
OlmoEarth is right; phi 0.77 to 0.81 for all six (held-out selection: Clay
0.72, family minimum 0.69). The selected partner's weighted disagreement
captures 0.242 of the errors at 10% against 0.454 for confidence (per chip
281 better, 1,263 worse; pooled CI [-0.233, -0.193]) and has pooled E-AURC
0.0668 against 0.0215 (per chip 78/1,499); the accuracy-weighted vote over
all outside partners 0.362; U+ 0.311. Every partner, inside or outside the
family, is within 0.01 of the selected partner (the full spread across partners is 0.017).

AWF Sentinel-2 (200 test points, 9 classes, 61 errors; OlmoEarth probe
0.695, Panopticon 0.630). Panopticon was selected on a held-out phi of
-0.003, which the test split shows to be the noise of 195 samples: its
test phi is 0.62 (family 0.50 to 0.70). Weighted disagreement captures
0.180 at 10% against 0.246 (5th percentile of the bootstrap gain -0.13);
E-AURC 0.21 against 0.09; the vote 0.213. Worse on every count.

Reading, as first written and then corrected. Cross-model disagreement
cannot flag OlmoEarth's errors, because the surviving partners make the same
errors whatever family they come from. Issue 6 is answered in the negative
for both designs, neighbourhood (exp39) and head (exp41), and two-view
disagreement is rejected. That much stands.

The gloss originally attached to it, "the errors belong to the windows, not
to the model", does not, and is withdrawn. An adversarial re-reading of the
repository's own outputs (2026-09-08) established four things against it.
(1) The one end-to-end fine-tuned model we hold contradicts it directly: on
the 344 AWF points, against the frozen exp16 probe on the same points and
labels, fine-tuning corrects 35 of the probe's 63 errors, 55.6%, and the
error correlation is phi 0.475 at crop 16 and 0.466 at crop 32, against
exp41's 0.77 to 0.81 across six frozen encoders (exp/out/exp21_finetuned_awf.csv,
columns error and probe_error). Changing the readout moves the error set
about twice as much as changing the frozen encoder does. (2) The claim rests
on the Sen1Floods11 half only: on AWF the outside models sit at phi 0.32 to
0.48 against the family's 0.50 to 0.70, and exp41's own
falsification_correlation.outside_below_family is true there. The line "on
AWF the same" was right about the capture numbers and wrong about the
correlation. (3) "Every encoder" is three outside encoders; four of the seven
requested partners (CROMA, TerraMind, CopernicusFM, Satlas) never produced
Sen1Floods11 embeddings on the 4-px grid and were dropped, which is a
selection on shared tokenisation, not a logistics footnote. (4) The window
geometry cannot be carrying the weight the phrase implies: exp43's
block-constant oracle forces at most 29% (Bolivia) and 49% (test split) of
the grid window's errors, and 55% and 42% of those errors sit on windows
whose labels are entirely one class, where the window constrains nothing.

What the evidence does support, narrowly: no frozen encoder read out by a
linear probe on this input escapes these errors, whatever its family. The
shared factor may be the frozen-probe readout as much as the window; exp41
cannot separate them, and the fine-tuned comparison above says the readout
carries at least as much of it.

## exp42 window design (2026-09-08)

exp41 found that no frozen encoder read by a linear probe escapes these
errors, which made the window worth testing as a design in its own right. Four alternatives to the fixed 4-px grid window, all from exp18's
cached features at crop offsets 0 to 3 px and the same head, judged at
pixel level on hand labels over the 57 x 57 region every tiling covers:
W1 shift-averaged (each pixel takes the mean of the four windows that
cover it), W2 scale-adaptive (a window defers to its 8-px block when the
block's margin is larger), W3 spectral split (inside a window the pixels
on each side of NDWI = 0 take the mean W1 probability of their own side),
W4 abstention by tile-phase against abstention by confidence at matched
coverage within each tile; purity from the pixel labels. Preregistered
primary: W1 against W0 on pixel accuracy per tile, one-sided, on Bolivia
and on the test split, both required. exp/exp42_window_design.py, one B200
job, 726464 on 2d6dce6, minutes; Codex review before the run, no blocking
finding. Outputs exp/out/exp42_summary.json, exp42_window_design.csv.

Supported. W1 raises pixel accuracy from 0.8969 to 0.9071 on Bolivia (441
tiles: 321 better, 66 worse, 53 tied, p = 1.3e-41, mean gain +1.02 points)
and from 0.9411 to 0.9503 on the test split (800 tiles: 583/78/139,
p = 8.9e-97, +0.91 points). The cost is four forward passes instead of
one; no labels, no retraining. W1's own confidence is also a slightly
better abstention score (selective accuracy at 80% coverage 0.9576 against
0.9535 on Bolivia, 0.9798 against 0.9791 on test) and, on Bolivia, a better
ranker (E-AURC 0.0091 against 0.0105; capture 0.517 against 0.463 at 10%),
though on the test split its E-AURC is worse (0.0109 against 0.0098) while
capture at 5 and 10% is better; secondary, not preregistered.

Mixed or null. W3 helps on Bolivia (0.9084 against 0.9071; 217/158 tiles,
p = 0.003; inside boundary windows 0.790 against 0.785) and hurts on the
test split (0.9483 against 0.9503; 226/404; boundary windows 0.851 against
0.860): a spectral edge is the right edge for one flood and not for many
regions. W2 gains 0.25 points on Bolivia (175/121, p = 0.002) and nothing
on the test split (253/241), with 39 to 40% of windows deferring. Tile-
phase abstention equals confidence abstention (0.9542 against 0.9535; 0.9776
against 0.9791).

Purity. Windows whose hand labels are mixed (labelled water fraction
between 0.1 and 0.9) are 10% of the windows on both testbeds and carry 45%
of the errors on Bolivia and 58% on the test split; their error rate is
0.39 and 0.28 against 0.054 and 0.022 on pure windows. These are the grid window's errors (W0): the
statistic says where the one-class-per-block decision fails, not that
those pixels are undecidable. W1 decides per pixel, and its residual on
the same windows is a different quantity, measured in exp43.

Reading. The grid is not the right unit, and the remedy is not a cleverer
window but no fixed window: average the tilings, and the decision follows
the content by construction. It is the first change in the repository that
improves the prediction map rather than the review of it, and it is
label-free. The package now ships it (signals.shift_averaged_probability
and pool_to_windows). What remains: the mixed windows are where the grid
fails, and W1 already decides inside them; whether a content-derived
partition with a hard majority of W1's pixels does better is exp43's
preregistered question. W4 shows tile-phase does not reject more errors
than confidence does at matched coverage.

## exp43 segment majority over a content-derived partition (2026-09-08)

The candidate that came out of the Lean development on the window
question (WindowDesignProofs.lean): the exact accounting of any redesign
against W1 on the same labelled pixels, C corrected, B broken, gain
(C - B) / N, and the conditional theorem that a segment with one true
class on which W1 is right on more than half of its pixels becomes fully
correct under the strict majority of W1's hard pixel decisions. Whether an
image-derived partition meets the conditions was the conjecture graded
here. Partition, fixed before the run and built without labels: per tile,
log1p of the twelve bands standardised within the tile plus raw NDWI times
2, k-means with K = 8 initialised at the quantiles of the first principal
component (sign fixed), exactly 20 iterations, 4-connected components.
Each segment takes the strict majority of W1's hard predictions inside the
common region, ties to land. Primary: against W1 on pixel accuracy per
tile, one-sided, on both testbeds, minimum worthwhile effect +0.002 mean
gain. exp/exp43_segment_majority.py, one B200 job, 727104 on 37de255
(a first submission failed on a missing scipy; components are now
computed without it, verified equal on random maps); Codex review before
the run fixed the NDWI weight, the sign of the first component, the
iteration count, finiteness checks and atomic tile results. Outputs
exp/out/exp43_summary.json and exp43_segment_majority.csv (per tile: N, C,
B, D, gain, purity, majority correctness).

Not supported, on both testbeds. Two endpoints, kept apart: the
preregistered one is the mean over tiles of (C - B) / N, and the pooled
one is (sum C - sum B) / sum N; they differ because tiles carry different
numbers of labelled pixels. Bolivia (440 of 441 tiles; tile 47 has no
labelled pixel inside the common 57 x 57 region and is excluded):
15,093 corrected against 16,273 broken over 1,180,805 pixels; mean tile
gain -0.00288, pooled gain -0.000999; 168 tiles better, 209 worse, 63
tied (one-sided p = 0.98); accuracy 0.9042 against W1's 0.9071. Test
split (800 tiles, none excluded): 24,525 corrected against 30,951 broken
over 2,479,909 pixels; mean tile gain -0.00250, pooled -0.002591;
244/387/169 (p = 1); 0.9478 against 0.9503. The soft variant is the same
(-0.0031, -0.0023); K = 16 makes segments so small that the map is nearly
W1 again (-0.0005, +0.0000). The plain twelve-band partition, without the
NDWI feature, is no better: mean gain -0.00354 on Bolivia and -0.00229 on
the test split, so the water weighting is not what sank the candidate.

Where the damage comes from (grading only, segments with at least two
labelled pixels; singletons account for at most 7 of the corrections and
breaks). The theorem holds exactly where its conditions hold: pure
segments carrying a correct W1 majority break nothing at all, and pure
segments carrying a wrong majority correct nothing at all.

| Segment class | Bolivia: segments, pixels, C, B | Test split: segments, pixels, C, B |
|---|---|---|
| pure, correct majority | 30,699; 897,358; C 8,957; B 0 | 65,665; 2,001,090; C 8,162; B 0 |
| pure, wrong majority | 2,570; 31,337; C 0; B 4,418 | 2,434; 28,695; C 0; B 3,359 |
| mixed | 4,743; 237,768; C 6,133; B 11,848 | 8,047; 424,883; C 16,363; B 27,590 |

Mixed segments are the dominant damage: they hold 73% of the broken
pixels on Bolivia and 89% on the test split, against 27% and 11% from
wrong majorities. Purity depends on how it is counted, so all three are
reported: the fraction of segments that are single-class is 0.875 and
0.894, the pixel-weighted share is 0.796 and 0.827, and the mean dominant
label fraction is 0.969 and 0.974. A spectral partition is nearly pure by
the last measure and still mixed on 20.4% (Bolivia) and 17.3% (test split) of its pixels,
which is where a single decision per segment costs more than it gains.

The statistic exp42 did not report: W1's own pixel errors sit 43%
(Bolivia) and 56% (test split) on the mixed-label windows, which hold 10%
of the pixels. That is error concentration, not undecidability; the pixel
map does decide inside them, and its error rate there is 6.5 times the
pure-window rate on Bolivia and 11.9 times on the test split. The actual limit of one class per 4-px block is the
oracle sum over windows of min(water pixels, land pixels) among the
evaluated pixels: 35,838 of 1,180,805 pixels on Bolivia (3.0%) and 71,878
of 2,479,909 on the test split (2.9%). The grid window's measured error
rates, 10.3% and 5.9%, are far above that oracle, so most of W0's loss is
the head, not the block.

Reading. A spectral partition is not a class partition often enough. The
Lean identities did their job: the run reports exactly what would have had
to be true, and it was not. Under the protocol the candidate is
unsupported; the result does not show that no partition can help, only
that this one, chosen without labels, does not.

## exp44 sixteen crop offsets against the four diagonals (2026-09-08)

W1 averages the four diagonal crop offsets (dy = dx) that exp18 cached.
The Lean counterexample in WindowDesignProofs.lean shows that adding the
twelve horizontal and vertical phases carries no guaranteed advantage; it
does not show there is none, so this run asks the empirical question. For
every offset (dy, dx) in {0..3}^2 the tile is cropped to 60 px at that
offset and encoded by the same frozen encoder (the four diagonals from the
cache, twelve new passes here), the same exp18 head scores it, each window
map is painted to the pixels it covers, and W16 is the equal-weight mean of
the sixteen maps. Preregistered: W16 against W1 on pixel accuracy per tile,
one-sided, both testbeds, minimum worthwhile effect +0.002 mean tile gain.
exp/exp44_sixteen_offsets.py, chained with exp43 in one B200 job, 727364 on
72d5375, 3 min 14 s in total (19 s and 32 s of encoding). Checks: a fresh
pass on the (0, 0) crop reproduces the cached diagonal logits exactly, and
fresh features of the (1, 1) crop match the float16 cache to a mean
absolute difference of 8e-4, the cast. Outputs exp/out/exp44_summary.json,
exp44_sixteen_offsets.csv.

| Design | Bolivia: accuracy, tiles better/worse/tied, mean tile gain | Test split |
|---|---|---|
| W1, four diagonals | 0.90707 | 0.95028 |
| W16, all sixteen offsets | 0.90745; 201/149/90, p = 0.003; +0.00037 (pooled +0.00028) | 0.95068; 369/202/229, p = 1.3e-12; +0.00040 (pooled +0.00042) |
| W8, diagonals and anti-diagonals | 0.90734; 191/155/94; +0.00026 | 0.95053; 334/210/256; +0.00025 |
| W7, diagonals and horizontal phases | 0.90603; 132/239/69; -0.00104 | 0.94874; 181/427/192; -0.00154 |

Not supported, but not null either. W16 is better than W1 on both
testbeds with one-sided p = 0.003 and 1.3e-12, and the gain is +0.04
accuracy points, five times below the preregistered minimum worthwhile
effect of +0.2 points, for twelve extra forward passes: statistically
clear, practically negligible, and unsupported under the protocol that
fixed the threshold before the run.

The informative part is the subsets. Adding the four anti-diagonal phases,
a balanced set of eight, gives half of W16's gain. Adding only the
horizontal phases, seven distinct offsets skewed to dy = 0, is worse than
the diagonals alone on both testbeds (p = 3e-8 and 7e-24 against). So what
matters is not that more phases are sampled but that the sampled set is
balanced in both directions; an unbalanced set moves the effective window
off centre and costs more than the extra sampling gains. That also explains
why the four diagonals, a balanced set, were already close to the ceiling
of this family.

## exp45 the supported findings on OlmoEarth v1.2 Base (2026-09-08)

Every supported result in this repository was established on v1 Base, but the
served land cover change product runs v1.2 Base, and v1.2 differs
structurally: rotary position encodings and one Sentinel-2 band-set token per
patch instead of three (exp19). This run repeats the four supported findings
on v1.2, on the same tiles, with the same head protocol (the exp18 head
retrained from v1.2 features of the valid split, seed 0), four crop offsets,
on Sen1Floods11 Bolivia (441 tiles) and the test split (800 tiles). Each
replication keeps the original's direction and unit and must pass on both
testbeds. Environment: a separate venv built from upstream main (commit
95bdd47), which exposes the v1.2 identifiers the pinned release does not; the
v1 environment is untouched, so the published v1 numbers stay reproducible.
exp/exp45_v12_replication.py, one B200 job, 728864 on 5f9c985, minutes.
Outputs exp/out/exp45_summary.json and exp45_v12_replication.csv.

Head accuracy is the same as v1's: 0.9062 against 0.9116 on Bolivia, 0.9534
against 0.9528 on the test split. What changes is how well the model ranks
its own errors.

| Finding | Bolivia | Test split | Replicates |
|---|---|---|---|
| R1 confidence is the best ranker | pooled E-AURC 0.0146; tile-phase 0.0138 and NDWI level 0.0119 both beat it; per tile 152/201 against tile-phase | 0.0059, beats every reference and control | no |
| R2 boundary first at the 10% budget | 134/46/173, p = 1.8e-11; pooled 0.4611 against 0.4234, CI [+0.028, +0.053] | 143/81/271, p = 2.1e-05 per tile, pooled 0.6624 against 0.6587 with the interval spanning zero | Bolivia only |
| R3 the four cues are enriched | boundary 3.3x, least-confident 3.2x, unstable 3.3x, NDWI-ambiguous 7.1x | 4.1x, 3.7x, 3.6x, 4.9x | yes |
| R4 shift-averaged beats the grid window | 0.9067 against 0.8957, 330/69/41, mean +0.0110 | 0.9516 against 0.9420, 587/81/132, mean +0.0096 | yes |

The R1 failure is the result of the run, with one scope condition that
matters. R1, R2 and R3 all grade the GRID window: the error set is
p_shift[0] > 0.5 and the confidence is the shift-0 logit margin, exactly as
exp18, exp36 and exp37 defined them. They are therefore statements about the
decision the repository no longer recommends. The shift-averaged decision R4
endorses has never had its own errors ranked by anything, here or in exp42,
whose selective-accuracy and capture columns also scored against the grid
window's error set. exp47 measures it. On Bolivia under v1.2 the grid
window's confidence is third: tiling instability ranks its errors better
(0.0138), and so does the NDWI-level control, which uses no model at all
(0.0119, unchanged from v1 because it does not depend on the encoder). Under v1 on the same
tiles confidence led at 0.0105 with tile-phase at 0.0115 and NDWI level at
0.0119. So v1.2 kept v1's accuracy and lost ranking quality: its logit margin
carries less information about where it is wrong. On the test split confidence
still leads everything, so the loss is testbed-dependent, not universal. This
is the first time a no-model control has beaten confidence on expert labels,
and it is a property of the newer backbone, not of the audit.

R2's Bolivia gain is larger under v1.2 than under v1 (+0.038 against +0.029 at
the 10% budget). The test split was never run for R2 under v1, so its failure
there is a limit on the finding's reach rather than a contradiction; at the
20% budget on the test split the boundary-first order is clearly worse
(0.743 against 0.842).

R4 replicates with a slightly larger gain than v1 (+0.0110 and +0.0096 against
+0.0102 and +0.0091), which is what exp19's instability measurement predicts:
v1.2's tilings disagree more (mean tile-phase 0.0493 on Bolivia against about
0.030 for v1), so averaging them recovers more.

D1, cross-version error overlap on identical windows. The v1.2 probe is wrong
on 74.6% (Bolivia) and 71.3% (test split) of the windows where the v1 probe is
wrong, and on 3.1% and 1.3% of the rest; phi 0.70 in both. v1.2 changes the
position encoding, the token structure and the pretraining run, and this is
Sentinel-2 through our own pipeline rather than the Sentinel-1 embeddings of
exp41's dump. It therefore removes exp41's shared-modality explanation while
leaving the shared linear probe and the shared labels in place. The overlap is
lower than exp41's 80 to 82%, so a real architectural change does move the
error set, but not much.

## exp46 what makes the errors shared (2026-09-09)

exp41 found six frozen encoders erring on 80-82% of the same Sen1Floods11
windows, and held three things constant while doing it: one token-local
linear probe, one input modality (its embeddings are Sentinel-1), and one
hard-thresholded majority label per window. This varies them one at a time
on identical windows, with the same labels, chips and fit split. Five arms:
a head on fifteen per-window pixel statistics with no encoder at all (A0),
the token-local linear head on cached Sentinel-2 features (A1, the
reference), the same head class on Sentinel-1 features encoded here (A2), a
two-layer head on the same features as A1 (A3), and a linear head on the
3x3 token neighbourhood (A4). Reference points already measured on these
windows: a backbone swap from v1 to v1.2 gives phi 0.697 and 0.705 (exp45),
swapping the frozen encoder across families gives 0.77 to 0.81 (exp41), and
fine-tuning gives 0.475 (exp21). exp/exp46_shared_error_sources.py, one
B200 job, 729376 on d897294. Codex review before the run fixed four things,
one of which mattered: the no-encoder arm's features span four orders of
magnitude and its accuracy rose from 0.64 to 0.92 once standardised.

| Arm, against A1 | Bolivia: accuracy, phi | Test split: accuracy, phi |
|---|---|---|
| A0 no encoder, fifteen pixel statistics | 0.9199, 0.714 | 0.9425, 0.553 |
| A1 Sentinel-2 linear (reference) | 0.9116 | 0.9528 |
| A2 Sentinel-1 linear | 0.8728, 0.389 | 0.8931, 0.381 |
| A3 Sentinel-2 two-layer head | 0.9201, 0.679 | 0.9567, 0.698 |
| A4 Sentinel-2 3x3 neighbourhood | 0.9290, 0.627 | 0.9541, 0.679 |
| backbone swap, for comparison | 0.697 | 0.705 |

Both preregistered comparisons pass, but they pass very differently. The
input modality is the dominant lever: reading the same scene through radar
instead of optics moves the error set to phi 0.38-0.39, roughly twice as far
as swapping the backbone. Changing the readout barely beats a backbone swap
(0.679 and 0.698 for the two-layer head, 0.627 and 0.679 for the
neighbourhood head), which is much weaker than the fine-tuning result at
0.475; fine-tuning changes the features as well as the head, and this run
shows the head alone does not account for it.

Two mechanisms are ruled out rather than confirmed. Label impurity is not
the driver: phi is essentially the same on windows whose labels are strictly
one class and on mixed ones (0.626 against 0.665 for A3 on Bolivia, 0.760
against 0.582 on the test split). Chip-level clustering is not inflating the
pooled figure either: the within-chip median phi sits within 0.02 to 0.05 of
the pooled value on every arm.

The uncomfortable result is A0. Fifteen per-window statistics with no
encoder reach 0.9199 on Bolivia against the frozen encoder's 0.9116 under
the same head, and their errors are as correlated with the encoder's
(phi 0.714) as another backbone's are. On the multi-region test split the
encoder does earn its place (0.9528 against 0.9425). So on the single-event
testbed the foundation model buys nothing that a spectral summary does not
already provide, which is the same direction as exp47's control result
below.

## exp47 ranking the decision the repository recommends (2026-09-09)

exp45's R1 graded the grid window: its error set is the shift-0 decision and
its confidence the shift-0 margin, faithfully replicating exp18, exp36 and
exp37. But exp42 and exp45's own R4 recommend the shift-averaged decision,
which now ships as signals.shift_averaged_probability, and its errors had
never been ranked by anything; exp42's averaged-confidence columns also
scored against the grid window's error set. This run fixes that. Both
backbones, both testbeds, the averaged decision's own error set, and every
ranker graded on it. Preregistered one-sided against tiling instability and
the no-model NDWI-level control, minimum pooled lead 0.001, both testbeds
under v1.2 required. exp/exp47_served_ranker.py, same job 729376.

| Arm | averaged confidence | grid confidence | tile-phase | NDWI level (no model) |
|---|---|---|---|---|
| v1 Bolivia | 0.0094 | 0.0108 | 0.0113 | 0.0113 |
| v1 test split | 0.0109 | 0.0099 | 0.0131 | 0.0106 |
| v1.2 Bolivia | 0.0116 | 0.0149 | 0.0133 | 0.0097 |
| v1.2 test split | 0.0064 | 0.0061 | 0.0081 | 0.0106 |

Pooled excess AURC, lower is better. Not supported: only v1.2 on the test
split passes both preregistered comparisons. The averaged confidence does
beat tiling instability everywhere, by 0.0017 to 0.0022 with per-tile
p between 4e-10 and 2e-37, and it beats the grid window's confidence on
Bolivia under both backbones (0.0094 against 0.0108, 0.0116 against 0.0149),
which is the shift-averaging gain showing up in ranking as well as accuracy.

The result that matters is the control. On Bolivia the NDWI-level control,
which uses no model at all, matches the averaged confidence under v1 (a
pooled lead of +0.0019 for confidence but 142 tiles against 174 per tile)
and beats it under v1.2 (-0.0019 pooled, 142 against 182 per tile). exp45's
finding therefore survives the correction: it was not an artefact of grading
a window design we had replaced. On the multi-region test split the model's
confidence wins comfortably under v1.2 (+0.0042) and ties under v1
(-0.0003 pooled while winning 316 tiles to 108). The pattern is testbed
dependent, and the testbed where the model loses to a spectral index is the
single flood event.

Every U+ combination is worse than plain averaged confidence, by 0.0025 to
0.0153, so the fusion that exp45's result seemed to invite does not help.
The boundary-first order is also worse on excess AURC (0.0175 against
0.0094 on v1 Bolivia), which is consistent rather than contradictory: it was
only ever claimed at fixed review budgets, never as a ranker.

## exp116 — the target rule, the class design and the certificate on real maps (2026-10-24)

Preregistered and frozen in debc794 (docs/plan/target_class_certificate.md) at the owner's "fix all", after the
final three-rater score named these three parts as untested on real maps. Arms A and B ran on the cluster (job
1496625, 32 cores, 27 minutes) against the package at 9bf2ecb; arm C locally on the package's Dynamic World tile.
The cluster's exp79_units holds exp79's RTX rerun of OlmoEarth Base beside the 16 encoders, and the script's glob
read it: 380 files where the plan fixed 356. exp97 guarded against exactly this and exp116 did not; the summary
reports the planned 356 as the result and the 380 beside it, and every prediction holds on both. An independent
second route (wf_8dbe7de7-5db, from the plan alone) reproduced every arm-A cell and arm-B map of the planned files.
P1 to P4 hold; P5 is mixed (the class-design sample on the tile certified nothing at 600 labels over nine classes).
Read after the run: on 46 of 413 cells, more than delta of the target rule's issued zones were wrong, though no
cell's share of all draws passed the bar; the package now says what delta counts wherever a zone is certified.

## exp117 — grading an agency's stratified sample (2026-10-26)

Not preregistered. After 1.17.2 the owner chose the agency workflow first of three items: grade a map with an outside
agency's stratified reference sample, NLCD 2016 AA and GFC2020, with intervals valid at any sample size. A first design
workflow ran its computations on the laptop and was stopped ("turn off the fans"); from then on nothing ran locally.
Sizing (aicr 1523908) showed union bounds over strata valid and useless. A second design workflow (three designs, a
judge) chose the stratified betting e-value; the judge's job (1525456) ran the self-tests and enumerations, which all
held, but its widths step and simulation reached their time limits (the certified ratio scan took up to 2,327 s per
ratio), so its selection rule, fixed before the run, returned the fallback with three gates unevaluated. The box of two
betting intervals for ratios and a simulation followed (1531551: coverage 0.997 to 1.000, normal theory 0.937 to
0.970), and the betting interval and the box were chosen after seeing the widths: a departure, stated in the results.
The package's `grade` (branch agency, 1.18.0) ports the engine; a review workflow (four angles, adversarially verified)
confirmed 26 defects (input handling, notes, tests, and a difference bound wider than it needed to be; none that made an
interval invalid on a correctly read sample), fixed before the end-to-end run
(1533266, at 0d9de25), whose every interval equals the design study's. A second review round confirmed 8 more (the table
reader too strict for pandas' and R's files, a stratum column that is a map's refused, the sizes file's header, and weights
whose printing allows several sizes, where a near-census could be read as a census and the pure-stratum rule return a
point that missed the truth); fixed in 5d8b434 (1.18.0), whose end-to-end run (1534186) printed the same numbers. Two
job-script slips on the way: three NLCD
alternate labels outside the legend (50, 91) broke the 8-class crosswalk, and `uv run` outside the project found no
command; both fixed in the script, not in the package.

**exp117, 26 October 2026, later: NLCD's Table 4 reconciled.** The 1.18.0 score (three raters, median 81) asked why
the primary-or-alternate per-class accuracies were up to 4.75 points off Table 4. The paper's PDF could not be read on
the laptop and nothing runs there; its pages were rendered on aicr (pymupdf, jobs 1539878 and 1539905) and Table 4 read
from the image: the transcription was right. The agency's own reference file (the shapefile's dbf) holds the same
labels as exp94 for all 4,629 points (1540024). A battery of agreement rules (1540024, 1540059, 1540086;
`exp/exp117_agency/table4.py` keeps the decisive ones, from the record's points alone) found the table's
primary-or-alternate user's accuracies all reproduced only when the 2011 primary label is accepted too (overall
86.955%, the caption's 86.9% truncated; the text's 86.4% is the one-alternate rule `grade` uses by default); no rule
reproduced its producer's accuracies. The package's `--alternate` now takes several columns (1.18.1), and
`grade --alternate alt p11` gives the same 15 user's accuracies (1540166). The Elsevier page images are not committed.

## exp118 — grade's intervals by a second route, and their coverage at NLCD's shape (2026-10-26)

Preregistered and frozen in 000bdf7 at the owner's "one by one, follow the prerequisite principle", after the 1.18.0
score. Arm A: an agent wrote the second route from the plan's specification alone (no package, no design-study
scripts); it ran on aicr (1541331, 34 s) and reproduced every interval within the plan's bars, the pure-stratum rule
within 1.1e-7 where the hypergeometric tails are decided in floating point. Arm B1: 5,000 draws at NLCD's shape with
1.18.1 from PyPI (1541238, 64 cores, 27 minutes; the smoke run 1541082 had timed it): every valid interval held; normal
theory fell below the bar on 33 of 91 quantities, all per-class, and held on the overall accuracies and the difference.
That last point corrects how exp117's NLCD-LCMAP finding reads: at n = 4,629 normal theory covers that difference, and
the disagreement is the valid interval's width. Arm B2 (GFC2020) was cancelled after seven minutes (1541240), before any
result was read: the owner asked whether the compute was necessary, and B2 tests the with-replacement case the
interval is built for. The owner then asked about a model-internal version for OlmoEarth v1.3 (weights public on the
Hub); decided to finish the general product first (the report, step 3), then design that study.

## exp119 — what 1.20.0's new intervals cost: release checks (2026-10-27)

Descriptive, no predictions. The 1.20.0 CHANGELOG and Usage quoted measured costs (aicr jobs 1554014 and 1553048) that
had no artifact in the record; the 1.20.0 Fable rating named that gap. The raw outputs are now committed in
`exp/out/exp119/`, summarised by `exp/exp119_summary.py` on aicr (job 1556952), and checked by a second route in
`tests/test_exp119.py`. Recomputing found two medians off by one hundredth (0.83 and 1.74 in the CHANGELOG; 0.82 and
1.73 from the JSON, the 1.74 having been against normal theory's interval cut to 0 to 100%), corrected in the
package's 1.20.1. The head-to-head with Spertus-Stark and ALPHA once planned under this number was dropped by the
owner on 27 October ("only do the meaningful things"), so the id is used here.

## exp120 — estimate --valid and the per-profile verdicts on fully labelled real maps (2026-10-29)

Preregistered and frozen in 1c41b86 (docs/plan/valid_and_profiles_on_real_maps.md) at the owner's "记录" (record it),
after the 1.20.0 Fable rating named two gaps: `--valid`'s coverage measured on no fully labelled map, and the per-profile
verdicts never run on a real map. Before the plan, an unregistered first check on the Dynamic World tile (aicr 1580392,
500 draws per cell, package 1.20.2 with the mask fix) had passed every bar; the plan says so, and arm B reruns that tile
with fresh seeds. The run used olmoearth-inferencex 1.20.3 from PyPI, installed clean in the job (1580985; a first
submission at 64 cores, 1580923, was cancelled while still pending for a free node, and the job ran at 48 cores,
arm A in 60 minutes, arm B in 1). Arm A ran exp116's 356 unit files (sha256 checked against exp/out/exp116_ab.json);
arm B ran the command line end to end. P1 to P3 hold: `--valid` covered on at least 0.9225 of draws in every cell, a
wrong verdict on at most 0.0225, an interval miss on at most 0.13. The second route (`exp/exp120_crosscheck.py`) was
written by an agent from the plan and the package's source without opening the first script (wf_a000f14d-8a5) and
checked against the plan by a second agent; it ran on aicr (1581168), and its counts equal
the run's on all 2,833 cells of the 357 maps (`exp/exp120_compare.py`, no difference). The writing agent ran
`python3 -c "print(1)"` on the laptop once, against the owner's rule;
it touched no file. Read after the run: the interval given without `--valid` fell below the bar on 11 cells (10 of them
the confidence design at 300 labels), where `--valid` fell on none (`exp/exp120_describe.py`, aicr 1582199).
