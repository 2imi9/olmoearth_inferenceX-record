# Do the package's three outputs hold on Ai2's other fine-tuned models? (exp89 preregistration)

**Status: frozen on 1 October 2026, before any prediction on a validation window.** Written the same day. The owner
confirmed every threshold, chose arm A (FT-AWF) and a torch MLP for K3, then amended the plan: arms F (Forest Loss
Driver) and A are report-only, and arm M (Mangrove) waits for Ai2's validation split. At freezing the owner also
confirmed arm F's population, input, controls, 1° clusters and α levels (0.10 and 0.15) and arm A's 2.0-point
alignment tolerance. Arm F's inventory (the 42 GB download) was still running: its count of validation windows
against Ai2's 109 is reported beside the alignment check and changes no rule. Arm M's split reader and imagery
fetch come later as a dated amendment, with its predictions and thresholds unchanged.

- **Allowed before freezing:**
  - the smoke tests on synthetic data;
  - the inventory job: it lists an arm's tar and the checkpoint keys, and counts windows and labels per split. It runs no model on a validation window;
  - the model smoke on training windows only. It is in-sample and never graded.
- **Not allowed before freezing:** any prediction on a validation window. That includes the accuracy gate and, for arms F and A, the alignment check. So arms F and A run once the page is frozen.
- **Open before freezing:**
  - Done at freezing: the owner confirmed arm A's alignment tolerance (2.0 points, below) and arm F's readings: its population, input, controls, clusters and α levels.
  - Arm F's inventory checks that the tar's validation windows match Ai2's: 109 windows with `split` "val", all eight image layers and the label completed, and a label among the ten classes. Ai2's matrix gives 44 agriculture, 30 burned, 18 none, 6 hurricane, 4 logging, 4 road and one each of landslide, mining and river (if its rows are the true class, not verified).
  - Arm F's tar SHA-256 is recorded by its first download and then pinned in the run script beside its size and MD5.
  - Arm M: when Ai2 shares its split, its reader and a fetch mode for its imagery are written and pinned here, as a dated amendment, before its gate runs. Its predictions and thresholds do not change.
  - The owner confirms the section "Readings fixed by the run script".

## Amendment of 1 October 2026: arms F and A report-only, arm M waits

The owner's decision. Ai2's public fine-tuned evaluation sets are too small to grade exp89 as written.

- **Arm F (Forest Loss Driver, new) and arm A (FT-AWF) are report-only.**
  - They run with the same measures, controls, clusters and studies as a graded arm. Every number and every interval is reported.
  - No prediction is graded on them, and no pass or fail verdict is drawn. Arm A's grading on P2 and P3 is withdrawn.
  - Why: arm F's validation set holds 109 windows and arm A's 344 points. At Ai2's accuracies (76.1% and 89.5%) that is about 26 and about 36 errors, below the error floor of 40. And N/5 is 21.8 and 68.8, below the smallest budget of 300 (a budget is graded only where B ≤ N/5 and N ≥ 1,500).
  - What is reported, for each prediction, is the numbers it would read, with no threshold applied:
    - P1: confidence's capture of the errors at 5%, 10% and 20%, beside random and the attainable ceiling;
    - P2: the best informative control and its AUROC;
    - P3: confidence minus that control at 5% and 10%, and in AURC, each with its cluster-bootstrap 95% interval (and the one-sided 5th percentile at 10%);
    - P4: the estimate's coverage and median widths per design, where a budget fits (arm A at B = 300; arm F at neither budget);
    - P5: the certificate's violation rate, median coverage and share certifying nothing, beside K3a's median coverage, where a budget fits; and c*(α) from all labels for both orders, on both arms.
  - Measures 1 to 7 are reported in full.
- **The accuracy gate stays for arms F and A as a reported alignment check, with its tolerance.**
  - Arm F: within 2.0 points of Ai2's 76.1% (83 of 109 windows), accuracy per window, two-sided. The accuracy under each of rslearn's four flips is reported beside it (Ai2's validation flipped at random); only the unflipped accuracy is checked.
  - Arm A: within 2.0 points of Ai2's 89.5%, accuracy per point, two-sided (exp21's replica was 1.4 points off).
  - It keeps the gate's rules: run after freezing, logged to the ledger outside the checkout, at most three attempts in all, on the pinned checkpoint.
  - A check outside the tolerance does not stop the run. The arm's numbers are then reported as "replica not aligned". The record and the run say "aligned" or "replica not aligned", never pass or fail.
- **Arm M (Mangrove) keeps its graded predictions (P1 to P5) and waits for Ai2's validation split.**
  - The public tar holds 100,000 points and 2,000 task areas, with no train/val split and no imagery (the inventory on the cluster, 1 October 2026).
  - Ai2 was asked for the split in Slack on 1 October 2026. Arm M runs once the split is shared and pinned in the run script; until then every mode that scores a window reports it as not run.
  - Its imagery will have to be fetched, so its gate uses the 1.0-point tolerance for refetched imagery.
- **Arm N (Nandi) stays not run:** its checkpoint and dataset are not public.
- **Arm A's data.** Arm A reads the pinned AWF tar (`allenai/olmoearth_projects_awf` at da8eb6aa, sha256 d0837f14...) that the job downloads and extracts under its scratch data directory. It no longer reads exp21's `data/awf`, whose files the 30-day scratch purge had partly removed, which is why the AWF inventory failed on 1 October. Every extraction keeps a manifest of its files. A file the purge removes later is found before any window is read, and the tar is extracted again.

## The question

The package makes three claims about a model's map:

1. **Review order.** The model's confidence ranks the windows to check, so a small review finds many errors.
2. **Estimate.** A designed sample of labels gives the error rate, with an interval that covers it.
3. **Certify.** A random sample of labels certifies the most confident part of the map at an error rate α, with a guarantee.

On Ai2's fine-tuned models the record tests only the first claim. It does so on one model (FT-AWF, exp21), against a no-model control that was near chance. exp89 asks whether all three claims hold on Ai2's other public fine-tuned models. It uses an informative control and takes Ai2's validation labels as the truth.

## What the record holds, and what it lacks

- **exp21 (FT-AWF).** The replica reached 0.881 on Ai2's 344 validation points, against Ai2's 0.895, with 41 errors. Confidence caught 22% of the errors in a 5% review and 39% in a 10% review (claim fine-tuned-capture-at-budgets).
  - Its only no-model control was the temporal standard deviation of NDVI. Its AUROC is 0.553, recomputed on 1 October from the committed `exp/out/exp21_finetuned_awf.csv`. So "beats the control" on AWF says little (red team, 30 September).
  - exp21 was not preregistered.
- **The suite's controls are near chance** (red team). They close a median 0.094 of the gap from random to perfect. The margin's AUROC is above 0.5 in all 3,560 exp79 cells. Beating random is therefore expected and shows little.
- **The estimate's designs were graded only at error rates of 5.5% and above.** exp78's tasks ran from 7.4% to 34.7%, and exp79's from 5.5% (checked in `exp/out/exp78_summary.json` and `exp79_summary.json`). A map at 2% to 3% error is untested. exp80's zones went as low as 1.8% (m_eurosat).
- **Every graded estimate and zone used probes on frozen embeddings,** never a fine-tuned model.

## Which models, and why

Checked on 1 October 2026 with HF API calls only.

- **Mangrove is public.**
  - Model `allenai/OlmoEarth-v1-FT-Mangrove-Base` at 57ee738b; `model.ckpt` is 1,041,362,118 bytes.
  - Dataset `datasets/allenai/olmoearth_projects_mangrove` at 3a878b8a; `mangrove.tar` is 62,433,280 bytes and `annotation_features.geojson` 33,933,316 bytes.
  - Ai2's doc links the dataset without `datasets/` in the path. That link returns 401.
- **Nandi is not public.** The model `allenai/OlmoEarth-v1-FT-Nandi-Base` and the dataset `datasets/allenai/olmoearth_projects_nandi` both return 401. Ai2's doc links them anyway.
- **Mangrove was declined before.** `docs/plan/roadmap.md` item 6 declined it for the ranking questions, for two reasons:
  - it classifies 2x2-pixel windows with no spatial context, so the boundary and tiling signals are undefined;
  - its split assigns 2x2 cells by hash, so it does not hold out space.

  exp89 does not test the boundary or tiling signals. It tests the model's confidence, the estimate and the certificate, and none of these needs spatial context. The hash split stays a limit (below). Mangrove does not meet the roadmap's need for "a second fine-tuned dense task with expert labels and a spatial split". Nandi would meet part of that need.
- **Forest Loss Driver is public (added with the amendment).**
  - Model `allenai/OlmoEarth-v1-FT-ForestLossDriver-Base` at 15502f8a; `model.ckpt` is 381,184,175 bytes, sha256 682c20b9.... Its training config is rslearn_projects `data/forest_loss_driver/20251104/config.yaml`, which that project's README ties to the Hub checkpoint.
  - Dataset: one tar on Google Cloud Storage, `ai2-olmoearth-projects-public-data/projects/forest_loss_driver/20251029/dataset.tar`, 42,214,604,800 bytes, object generation 1761857427036506, MD5 6abc5b94... (an HTTP HEAD on 1 October 2026).
  - Ai2's 76.1% is 83 of 109 validation windows, micro accuracy, read from its confusion matrix.

exp89 has four arms:

- **Arm M, Mangrove.** Graded on P1 to P5. Waits for Ai2's validation split (the amendment).
- **Arm N, Nandi.** Frozen with this page under the same thresholds. It runs only if Ai2 publishes the checkpoint and the dataset; until then it is reported as not run. Its thresholds do not change after arm M's result.
- **Arm A, FT-AWF (run: the owner's choice on 1 October).** Report-only since the amendment. It had been graded on P2 and P3 only, because exp21's confidence result has already been seen.
- **Arm F, Forest Loss Driver (added with the amendment).** Report-only.

## Design

### Data and units

- **Mangrove.**
  - Population: the windows of group `sample_100K` tagged `split=val` in `mangrove.tar` at the pinned revision, after the label rules below. The size is unknown until the inventory runs. It is about 12,500 if the group holds 100,000 windows (not verified).
  - Unit: one 2x2-pixel window at 10 m, which is one point and four pixels. The model gives one prediction per window.
  - Labels: Global Mangrove Watch v4 reference points (Zenodo 17394267, CC BY 4.0). The classes are 1 mangrove, 2 water and 3 other; 0 is invalid.
  - Split: Ai2 hashes each 2x2 cell into train (87.5%) or val (12.5%). Validation windows therefore have training neighbours.
- **Nandi (if published).**
  - Population: group `spatial_split`, `split=val`. The size is unknown.
  - Unit: one 63x63 window holding one labelled pixel. This is inferred from AWF, which uses the same preparer; it is not verified.
  - Labels come from three sources:
    - points from 819 CGIAR/IFPRI polygons;
    - water and built-up points derived from WorldCover;
    - tree points annotated in Studio.
  - Class indices (from `olmoearth_run.yaml`): 0 coffee, 1 grassland, 2 trees, 3 maize, 4 sugarcane, 5 tea, 6 vegetables, 7 legumes, 8 water, 9 builtup, and 10 nodata.
  - Split: 128-px (1.28 km) cells, 75/25.
- **AWF (arm A).** exp21's 344 validation points, read from the pinned tar (the amendment).
- **Forest Loss Driver (arm F)** (confirmed at freezing):
  - Population: the validation set as Ai2's training config reads it. Every window, in any group, whose option `split` is "val", whose eight image layers (`pre_sentinel2` and `.1` to `.3`, `post_sentinel2` and `.1` to `.3`) and label layer carry rslearn's completed marker, and whose label is one of the ten classes. Ai2's split script gives "val" only to the Brazil and Colombia phase 1 and 2 groups, by the first hex digit of the SHA-256 of the window's name (0 to 3).
  - The option `olmoearth_evals_split` belongs to the OlmoEarth paper's evaluation, not to this checkpoint. It never chooses a window; the inventory only reports it beside `split`.
  - Unit: one window, one prediction.
  - Labels: `new_label` of the first feature of `layers/label/data.geojson` that holds one of the ten classes, in the order agriculture, mining, airstrip, road, logging, burned, landslide, hurricane, river, none. This is rslearn's ClassificationTask with unknown categories skipped. A window with no such feature is dropped and counted. `label.json` and `old_label` never give a label; the inventory only counts the windows whose `label.json` names another label.
  - Split: by a hash of the window's name within four groups, not by space, so a validation window can have training neighbours.

### Clusters (for the bootstrap only)

- **Mangrove:** the 0.1° longitude/latitude cell. olmoearth_run partitions Mangrove requests on the same 0.1° grid. The 1° cell is reported. (confirmed)
- **Nandi:** the 128-px split cell. If the windows record a polygon id, cells that share a source polygon are merged. The 0.05° cell is reported. (confirmed)
- **AWF:** exp21's 30 annotation tasks.
- **Forest Loss Driver:** the 1° cell of the window's centre (confirmed at freezing). The window group cannot be the cluster: validation windows come from four groups only. The group is the stratum of the by-stratum report.

The estimate and certify study draws from the finite validation set, so its coverage needs no clusters. The clusters matter only for the bootstrap of the ranking comparison.

### The replica: how each window's scores are computed

The replica does not use rslearn, as in exp21. The checkpoint's encoder keys are loaded strictly into OlmoEarth v1-Base (`olmoearth-pretrain` 0.1.0, revision 4bd1392a).

**Common to both models:**

- Input: 12 monthly Sentinel-2 L2A mosaics in the band order B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09, as stored in the tar. If the tar stores band groups at their native resolution, the 20 m and 60 m groups are resampled bilinearly onto the 10 m grid, as exp21 did.
- Normalisation: OlmoEarth's computed statistics with std multiplier 2. This matches rslearn's `OlmoEarthNormalize` default, and Mangrove sets 2.0 explicitly.
- Timestamps: rslearn's legacy values, which are day 1, month index 0 to 11, year 2024. exp21 used day 15 and year 2023. The model smoke reports the largest logit difference between the two conventions.
- Encoder: run with `fast_pass=True`. Tokens are mean-pooled over timesteps and band sets, which is rslearn's `token_pooling` default.
- Precision: fp32 with TF32 off.

**Mangrove:**

- Input: the 2x2 block at the label, at patch size 2, so one token per timestep and band set.
- Head: the pooling decoder's amax over a 1x1 map (which does nothing), then `Linear(768, 4)`. The expected keys are `model.decoders.mangrove_classification.0.output_layer.{weight,bias}` (inferred, not verified).
- Output: four logits per window.

**Forest Loss Driver (arm F):**

- Input: the eight layers in the config's order (four pre, then four post), 12 bands each in OlmoEarth's order, cropped by rslearn's centre Pad to 64 px (pixels 32 to 95 of a 128-px window) on the raw values, then normalised.
- Encoder: rslearn's SimpleTimeSeries with 48 channels per image runs the encoder twice at patch size 4, once on the four pre layers and once on the four post layers. Each pass has timestamps day 1, months 0 to 3, year 2024, and is mean-pooled over timesteps and band sets. The two 768x16x16 maps are concatenated, pre then post.
- Head: rslearn's PoolingDecoder, loaded strictly from `model.decoder.0`: a 3x3 convolution to 128 channels with ReLU, the maximum over the map, two linear layers of 512 with ReLU, and a linear layer to 10.
- The encoder keys sit under `model.encoder.0.encoder.model.` (one level deeper if the wrapper kept the whole model). The strict load decides which.
- Ai2's validation used random flips and bfloat16 autocast. The replica uses no flip and fp32. The alignment check reports the accuracy under each flip; S2 reports the bfloat16 difference on training windows.
- Output: ten logits per window. All ten channels are trained.

**Nandi:**

- Input: the 16x16 crop of the 63x63 window that puts the label pixel at row 8, column 8 (exp21's rule with shift 0), at patch size 1. Ai2's validation used a random crop; ours is fixed.
- Head: a 1x1 conv from 768 to 11 channels, with no upsampling (scale factor 1). The expected keys are `model.decoders.segment.1.layer.{weight,bias}`, by analogy with AWF (not verified).
- Output: 11 logits, read at the label pixel.

### Prediction and confidence

- **Prediction:** the argmax over all output channels, as rslearn takes it. A window predicted as an untrained channel (Mangrove 0, Nandi 10) counts as an error. The count is reported; it is expected to be zero.
- **Confidence, graded:** the top-1 softmax probability over the trained channels only (Mangrove 1 to 3, Nandi 0 to 9; AWF 0 to 8 and arm F 0 to 9, reported). For arms F and A the same confidence is computed and reported, not graded. It is the package's `form="top1"`, through `assess_prediction` at patch 1. The reason: exp76 found top-1 better than the margin on 14 of 16 multi-class tasks, and the package warns about the margin for multi-class logits. (confirmed)
  - The package computes it from its logarithm, which has no ties, but returns the probability. That rounds to exactly 1.0 when the top trained logit leads the others by about 37 or more, so such windows tie.
  - The ranking measures are tie-aware. The certificate's order breaks the ties by window index, which is fixed before any label is seen, so the certificate stays valid.
  - The number of tied windows is reported (`n_p1_saturated`).
- **Reported, not graded:**
  - the logit margin, which is the default on the command line;
  - the entropy;
  - for Nandi, context-shift instability: the spread of the shift-0 class's probability over crop shifts of 0 to 3 px. At patch size 1 there is no sub-patch lattice, so this is not exp21's tiling instability.

### Controls

All controls are fixed here and computed from the same 12 mosaics. None is tuned on the validation set.

- **K1, random order.** The baseline, computed analytically.
- **K2, class rarity.** The training frequency of the predicted class; the rarer the class, the more suspect the window. This is the record's control.
- **K3, a no-encoder classifier. This is the informative control.**
  - Features, per month: the 12 normalised bands at the label pixel (Nandi) or the 2x2 block mean (Mangrove), plus NDVI. Mangrove also gets MNDWI (from B03 and B11).
  - Empty months: filled with the unit's median of that band over months. The count of empty months is also a feature.
  - Model: an MLP in torch, so no new dependency. Two hidden layers of 256, ReLU, dropout 0.1, Adam at learning rate 1e-3, weight decay 1e-4, batch 512, 30 epochs, seed 89. The last epoch is used, with no early stopping and no tuning.
  - Training data: the train split only.
    - Mangrove: up to 20,000 training windows drawn with seed 89. The cap is set from the fetch pilot before freezing, and is never below 5,000.
    - Nandi: all training windows.
    - Under Mangrove's hash split, K3's training windows are as close to the validation windows as the fine-tuned model's were.
  - Signals:
    - K3a: one minus K3's top-1 probability;
    - K3b: K3's disagreement with the fine-tuned model, binary and ranked tie-aware.
  - A gradient-boosted classifier would need scikit-learn added to the project. That is the owner's call, made before freezing.
- **K4, index controls.**
  - Mangrove:
    - inundation ambiguity min(f, 1 - f), where f is the share of valid months with MNDWI above 0;
    - how close the annual median MNDWI is to 0;
    - the monthly standard deviation of NDVI.
  - Nandi:
    - the temporal standard deviation of NDVI at the label pixel (exp21's control);
    - the 3x3 standard deviation of NDVI around the label pixel.
- **K5, cloud and missing data.** The number of months that are empty or whose B02 reflectance exceeds 0.2. In the harmonised L2A digital numbers the tar stores, that is 2,000. (confirmed with the other thresholds; the draft's marker was left unchanged by mistake)
- **Arm F's K3, K4 and K5** (confirmed at freezing; arm F has eight timesteps, not twelve months, and a loss to read):
  - K3's features (128 per window) come from the 64-px crop the model sees. For each stack (pre and post): the per-pixel median over its non-empty timesteps, as 12 bands plus NDVI (B08, B04) and NBR (B08, B12); their spatial mean and standard deviation over the crop and their mean over the centre 16 px. Then post minus pre of those, then each stack's count of empty timesteps. The MLP and its settings are K3's above; it is fitted on every training window that passes the population's layer rule.
  - K4: minus the NDVI drop and minus the NBR drop from the pre to the post composite at the centre 16 px. A small or negative drop is a weak loss signal, so it is suspect.
  - K5: the number of the eight timesteps that are empty or whose centre mean B02 exceeds 2,000.
  - K2 is the record's class rarity, from the training windows' labels.

**The best informative control** is whichever of K2 to K5 has the lowest AURC on the validation set. Choosing it on the graded data favours the control, so the comparison is conservative for the package.

For certification the comparator is K3a, the best continuous control. A binary order would break its ties by index.

Two candidate controls are not used:

- distance to the edge of the GMW 2020 map, because that map was built from these same points;
- disagreement with WorldCover for Nandi, because WorldCover produced the water and built-up labels and was a pretraining target.

### Label handling

- **Labels are taken as right.** No label is checked or corrected. An "error" is a disagreement with Ai2's label, including where the label itself is wrong.
- **Mangrove:**
  - A window is kept when its valid label pixels all agree and none is 0. Other windows are dropped and counted.
  - The gate also computes Ai2's pixel-level micro accuracy over all valid label pixels.
- **Nandi:**
  - A window is kept when it has exactly one labelled pixel, with a class from 0 to 9.
  - Results are also reported by label source (CGIAR polygon, WorldCover, Studio trees) if the windows record it.
  - The Studio tree points were added to fix a known model error, so they are not a random sample.

### The estimate and certify study

The validation labels are the truth for the whole population of N windows, and θ is the share of errors. The only randomness is the draw of labels, as in exp78 and exp80.

- **Budgets:** B = 300 and B = 1,000 labels. A budget is graded only where B ≤ N/5. Where N < 1,500 no budget is graded, and the study is reported only.
- **Draws:** R = 2,000 per (arm, design, B). Each draw's seed comes from `np.random.SeedSequence([89, arm, design, B])`.
- **Estimate:**
  - Two designs: `random`, with the exact hypergeometric interval, and `confidence`, the package default (margin quintiles, Neyman allocation from the top-1 probability).
  - Both run through `sample_for_estimation` and `estimate_error_rate`, unchanged.
  - Per cell: the coverage of the 95% interval, the median width, the bias and the RMSE.
- **Certify:**
  - Call: `certify_zone(confidence, indices, wrong, alpha, delta=0.10, rule="prefix")` on the random design's draws. The Bonferroni rule is reported beside it.
  - α levels: Mangrove 0.02 (graded) and 0.01 (reported); Nandi 0.05 (graded) and 0.10 (reported). (confirmed) AWF 0.05 and 0.10, and arm F 0.10 and 0.15, all reported (confirmed at freezing; Ai2's error rates are 10.5% and 23.9%). At 109 windows arm F fits neither budget, so only its c*(α) is reported.
  - Each graded α is below Ai2's reported error rate (2.4% and 12.7%). So the whole map cannot be certified, and only the order can help.
  - At δ = 0.10 a zone needs at least 114 labels at α = 0.02, 230 at 0.01, 45 at 0.05 and 22 at 0.10. So at B = 300, α = 0.01 cannot certify any zone smaller than 80% of the map.
  - Per cell:
    - the violation rate: the share of draws where the certified zone's true error rate exceeds α;
    - the median certified coverage;
    - the share of draws that certify nothing;
    - the best coverage possible, c*(α), from all labels.
  - The same is computed with K3a's order.
  - The review-set guard refuses a draw that looks enriched. A random draw almost always passes; refusals are counted.

### Review budgets and bootstrap

- **Review budgets:** 5% and 10% are graded, and 20% is reported. Capture is measured with the tie-aware `capture_at_budget_expected`.
- **Cluster bootstrap:** 2,000 resamples of the clusters, seed 89.
  - AURC differences use `cluster_bootstrap_difference`.
  - Capture differences use the same resampling, coded in the exp89 script.

## Smoke and the alignment gate

- **S1, numpy smoke.** Synthetic logits, labels and controls with known answers: one case where confidence beats the control and one where it does not. Every grading function must return the known verdict. It runs on the Mac and in the tests.
- **S2, model smoke.** Runs before freezing, on training windows only.
  - The checkpoint loads strictly, every key is listed, and the head has the expected shape.
  - One batch at patch size 2 (Mangrove) or 1 (Nandi) gives the expected output shape.
  - Accuracy on 512 training windows, drawn with seed 89, is at least 0.95 for Mangrove. This is in-sample and only checks the loading. For arms F and A it is reported with no bar.
  - Arm F: the logits under bfloat16 autocast beside fp32, as their largest difference and the number of predictions that change.
  - The untrained channel is never the argmax.
  - The largest logit difference between the two timestamp conventions is reported.
- **G, the alignment gate.** Runs after freezing, and reads accuracy only.
  - The gate job reads the validation windows and computes the predictions. It writes only the window count, the error count and the accuracy. It writes no confidence and no control.
  - **Mangrove passes** when our recomputed pixel-level micro accuracy is within 0.5 points of Ai2's 97.6% if the imagery came from the tar. If the imagery had to be fetched again, the tolerance is 1.0 point. (confirmed)
  - **Nandi passes** within 2.0 points of Ai2's 87.3%. exp21's AWF replica was 1.4 points from Ai2's figure. (confirmed)
  - **AWF (arm A) passes** within 2.0 points of Ai2's 89.5%, as Nandi does, on the accuracy per window. exp21's replica was 1.4 points off. (confirmed at freezing) Since the amendment this is arm A's alignment check: it is reported, and outside the tolerance the arm's numbers read "replica not aligned".
  - **Arm F's alignment check** is within 2.0 points of Ai2's 76.1%, on the accuracy per window, reported in the same way, with the accuracy under each flip beside it.
  - The gate is two-sided: a replica far above Ai2's figure is as suspect as one below it.
  - A failed gate may be retried after a fix to the replica. There are at most three attempts in all: the first and two retries.
  - Each attempt is logged with its commit, the page's status and the checkpoint's sha256. It goes to a ledger kept outside the git checkout before it goes to the gate file, so a reset of the checkout cannot lower the count. The run accepts a pass only if the ledger holds it, it was made on the frozen page, and it scored the pinned checkpoint.
  - When no kept validation window has imagery, the gate refuses and spends no attempt. Windows without imagery are counted.
  - Nothing is graded until the gate passes.
- **Error floor.** The ranking predictions (P1 to P3) are graded only if the population holds at least 40 errors, exp21's level. Below that they are reported only.

## Measures

1. Accuracy (pixel micro and per window), per-class recall, the confusion matrix, the error count, and the number of clusters that hold an error.
2. For each signal (confidence, margin, and each control):
   - the AUROC for errors;
   - the capture at 5%, 10% and 20%, against random (0.05, 0.10 and 0.20) and against the best possible ranking (`attainable_ceiling`);
   - the excess AURC, and the share of the gap from random to perfect that the signal closes.
3. Confidence minus the best informative control: the capture at 5% and 10%, and the AURC, each with its cluster-bootstrap 95% interval.
4. Selective accuracy at 50%, 80%, 90% and 100% coverage, and the ECE over 10 bins of the top-1 probability.
5. The estimate study (above), per design and budget.
6. The certify study (above), per α and budget, for confidence and for K3a.
7. Reported by stratum, not graded: Mangrove and AWF by label class, Nandi by label source, and arm F by window group.

## Predictions

All predictions are one-sided. Each is graded per arm, never pooled.

- **P1, the review order beats random by a useful margin.** In a 10% review, confidence captures at least 40% of the errors on Mangrove and at least 30% on Nandi. (confirmed)
  - For scale: random catches 10%, and exp21 caught 39% on AWF at 11.9% error.
- **P2, the informative control is informative.** The best informative control's AUROC for the model's errors is at least 0.65. (confirmed)
  - If P2 fails, P3 is still graded, but it is reported as a comparison with a weak control, which shows little.
- **P3, confidence beats the informative control (the question).** Both must hold (confirmed):
  - in a 10% review, confidence captures at least 5 points more of the errors than the best informative control;
  - the one-sided 95% cluster-bootstrap lower bound of that difference is above 0.
- **P4, at a low error rate the estimate still covers, and the confidence design gives a narrower interval.** Both must hold:
  - at each graded budget, the confidence design's 95% interval covers θ on at least 0.93 of the 2,000 draws (the package's bar, exp79);
  - at B = 300, its median width is at most 0.85 of the random design's. exp78's median was 0.80. (confirmed)
- **P5, certification is valid and useful.** At B = 1,000 (or at B = 300 where only that budget is graded), at the graded α, all three must hold:
  - the violation rate is at most 0.12, which is δ = 0.10 plus three Monte Carlo standard errors at R = 2,000;
  - the median certified coverage is at least 0.50 on Mangrove and at least 0.30 on Nandi (confirmed);
  - it exceeds the median coverage certified with K3a's order by at least 0.10 (confirmed).
- **Arms F and A:** no prediction is graded (the amendment). The numbers each prediction reads are reported with their intervals.

## Readings fixed before freezing

- **What can fail by construction and what cannot.**
  - The random design's exact interval covers at least 95% by construction.
  - The prefix rule is valid on any map under a random draw. So P5's validity part checks the implementation more than it tests a hypothesis.
  - P4's coverage part is a real test. The stratified interval is not exact, and it has not been graded below 5.5% error.
- **What a pass on P1 alone means:** little. On the suite, confidence beats random everywhere. The informative result is P3 together with P2.
- **The one fact read early:** Ai2's reported accuracies (97.6% and 87.3%) set α and P1's thresholds. They are public figures, not our result.
- **Nandi's thresholds** are frozen now, before arm M runs, and do not change after it.

## Readings fixed by the run script

`exp/exp89_finetuned_checkpoints.py` reads this page as below where the page leaves a detail open. None changes a threshold. The owner confirms them when freezing.

- **Mangrove's label rule.** A window is kept when all four pixels of its 2x2 block, after rslearn's centre pad to 2, hold the same class from 1 to 3. The gate's pixel micro accuracy counts every valid (non-zero) label pixel of every validation window with imagery, kept or not, as Ai2's metric does.
- **The margin.** The logit margin is taken over the trained channels. The estimate's confidence design stratifies by it (margin quintiles) and allocates by the top-1 probability, as `sample --logits` does. A window whose top two channels include the untrained one is counted.
- **The best informative control.** The candidates are K2, K3a, K3b, each K4 index and K5, each counted as one candidate.
- **P3's bound.** The one-sided 95% lower bound is the 5th percentile of the cluster bootstrap of the capture difference at 10%.
- **Draws and the certificate.**
  - `SeedSequence([89, arm, design, B]).generate_state(R)` gives one seed per draw. The arm is Mangrove 0, Nandi 1, AWF 2 or Forest Loss Driver 3; the design is random 0 or confidence 1.
  - The certificate reuses the random design's draws.
  - A violation is a certified zone whose true error rate exceeds α. A draw that certifies nothing, or that the review-set guard refuses, has coverage 0 and no violation. Refusals are counted.
  - c*(α) is the largest grid coverage whose zone, in that order, has a true error rate of at most α.
- **K3's features** are z-scored with the training split's mean and standard deviation. Nothing is fitted on the validation split.
- **K4's indices.**
  - Nandi and AWF use exp21's NDVI over all twelve months, (B08 - B04) / max(B08 + B04, 1e-6). So AWF's temporal control equals exp21's.
  - The 3x3 control is the spatial standard deviation of each pixel's median NDVI over the months.
  - Mangrove's indices use the valid months only, as f is defined over valid months.
- **Imagery.**
  - Imagery is read from the tar only. There is no fetch mode yet (see "Open before freezing").
  - A validation window with no completed Sentinel-2 item group is dropped and counted.
  - A window with fewer than 12 completed item groups is run with its own number of timesteps, batched by that number. This is what rslearn's masked pooling over missing timesteps reduces to.
- **The smoke** certifies on 300 of the 2,000 draws and bootstraps 300 resamples, so it finishes in about a minute. The run uses 2,000 everywhere.
- **Nandi's windows.** The polygon id is looked for under the option keys `polygon_id`, `source_polygon`, `polygon` and `source_id`, a guess until the windows are seen. The label source is read from the option key `source`.
- **The report-only arms' record.** The gate's record for arms F and A holds `aligned` where a graded arm holds `pass`, and `alignment` reads "aligned" or "replica not aligned". The run needs a recorded check, made on the frozen page and the pinned checkpoint and held by the ledger, but not an aligned one. The summary holds the reported numbers under `reported`, with no threshold and no verdict, and the arm's `alignment`.
- **Arm F's data.** The job downloads the tar at its pinned object generation with a resumable curl. The script checks its size and MD5 against the pin, records its SHA-256, and streams it once, writing only the eight image layers, the label layer, each window's own files and every file outside the windows (the dataset's config). Every other layer (Landsat, Sentinel-1, Sentinel-2 groups `.4` and `.5`, the masks) is counted and skipped. Groups are not filtered, because the split is read per window.
- **Arm F's training windows** are every window with `split` "train" that passes the population's layer rule; K2's frequencies and K3's fit read them.

## What follows, whatever the outcome

- **The gate fails three times:** nothing is graded. exp89 records the gap and its likely cause (imagery, crop or timestamps). No claim is made.
- **P1 fails:** the review order is weak on that model. The README's statement about fine-tuned models stays at AWF and gets a scope note.
- **P2 fails:** the comparison is again with a weak control. The record says P3 shows little, and roadmap item 6 stays open.
- **P2 and P3 hold:** a new claim. On the named model, its confidence finds more errors in a 10% review than an informative no-encoder classifier does, by the measured amount. The README can then name two fine-tuned checkpoints: AWF with a weak control and Mangrove with an informative one.
- **P2 holds and P3 fails:** a cheap classifier with no encoder ranks this model's errors as well as the model does. The package's ranking adds nothing over it here, and the record and the docs say so.
- **P4's coverage fails:** the package's default design has a defect at low error rates. It is fixed in a later release and regraded on exp78, exp79 and here. Until then the docs warn below the tested error rate.
- **P4's width fails:** the confidence design gives no narrower interval on this map. The default stays, and the docs give the measured range.
- **P5's validity fails:** an implementation defect, which is fixed and regraded.
- **P5's usefulness fails:** at 1,000 labels, certification returns little on this map. The docs state what it costs.
- **Arms F and A, whatever their numbers:** no claim is made from them. They are recorded as report-only, beside their alignment ("aligned" or "replica not aligned"), and the docs may cite them only as such.

## Limits stated in advance

- **Labels are assumed right.**
  - GMW points are labels from a reference product, not field checks.
  - Nandi mixes field polygons, points derived from WorldCover and points made in Studio.
  - Reviewer and label error are not modelled. In practice they break interval coverage.
- **Points are not independent.**
  - Nandi's points are sampled inside 819 polygons.
  - Mangrove's validation windows sit beside training windows because of the hash split, so they are close to in-sample.
  - The bootstrap resamples clusters. The estimate study is about the finite validation set, not about a mapped area.
- **One checkpoint per task,** chosen by Ai2 on this same validation set. Its reported accuracy is optimistic, and its errors are those of a selected checkpoint.
- **Ai2's own validation split is the population.** It is not a random sample of any map, so the estimate and certify results are statements about these points only.
- **Mangrove has no spatial context.** The boundary and tiling signals are not tested. The roadmap's reasons for declining Mangrove still hold for those questions.
- **Arm F's validation set is small and was used to choose the checkpoint.** Its 109 windows come from four groups in Brazil and Colombia; Peru is never validated. The training config kept the best validation accuracy, and Ai2's figure was measured with random flips.
- **The replica is ours, not rslearn's.** The gate bounds the difference in accuracy, not in which windows are wrong.
- **Package functions, not the command line.** Points are not a raster map. exp89 therefore calls `assess_prediction`, `sample_for_estimation`, `estimate_error_rate` and `certify_zone` directly. It does not run the `assess`, `sample`, `estimate` or `certify` commands end to end.
- **No correction for multiple predictions.** Each prediction is graded on its own and reported as such.

## Changes after freezing

- The predictions, thresholds, controls, clusters, budgets and seeds do not change once the gate has run.
- A bug in a reader may be fixed if the fix is shown on the smoke cases and every arm is regraded. The change is listed beside the verdict.
- A change of rule starts a new experiment.

## Addendum of 1 October 2026, after freezing and before any result was read

A reader bug fixed under "Changes after freezing". The synthetic smoke of job 1151248 wrote its gate attempt (a
synthetic checkpoint, made while the page was a draft) into the real gate ledger of arm A, and the run guard then
refused arm A's run (job 1151373) although its real alignment check was made on the frozen page and the pinned
checkpoint. Two changes, each with a test that failed first: the synthetic smoke keeps its ledger in a temporary
directory; and an attempt made before freezing on another checkpoint than the pinned one no longer counts towards the
limit or blocks the run. It stays in the ledger as history. No rule, threshold or reading changed, and no number had
been read.

## Result, appended 2 October 2026 (nothing above changed)

Arms A and F ran on the frozen page and the pinned checkpoints; both replicas are aligned (FT-AWF 88.1% against
Ai2's 89.5%, the same predictions as exp21 on all 344 points; Forest Loss Driver 77.1% against 76.1% on Ai2's 109
windows). As the amendment fixed, no prediction is graded and no claim of support is drawn. The numbers each
prediction reads, with their intervals, are in
[comparisons.md](../results/comparisons.md#ai2s-own-fine-tuned-models-report-only-ft-awf-and-forest-loss-driver-exp89)
and the ledger entry `exp89-finetuned-report-only`. Arm M waits for Ai2's Mangrove split; arm N is not public.
