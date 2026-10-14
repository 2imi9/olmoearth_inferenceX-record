"""exp89: do the review order, the estimate and the certificate hold on Ai2's other public fine-tuned models?

Preregistered in docs/plan/finetuned_checkpoints.md; read that page first, this file is the run. Four arms, as the
owner's decision of 1 October 2026 amends the page:
  mangrove  allenai/OlmoEarth-v1-FT-Mangrove-Base, graded on P1 to P5. Not run until Ai2 shares its validation split:
            the public mangrove.tar holds 100,000 points and 2,000 task areas, with no split and no imagery. Every mode
            that scores a window reports it as not run while ARMS["mangrove"]["split_source"] is None;
  nandi     frozen under the same thresholds and not run: its checkpoint and dataset return 401 on the Hub. Every
            mode reports it as not run unless both become readable and their revisions are pinned here;
  awf       FT-AWF on Ai2's 344 validation points (exp21's), REPORT-ONLY: every measure, control, cluster and study,
            every number and interval, and no verdict on any prediction;
  fld       allenai/OlmoEarth-v1-FT-ForestLossDriver-Base on Ai2's validation windows (options.split "val", 109 in
            Ai2's confusion matrix), REPORT-ONLY as arm A.
For the report-only arms the accuracy gate is an alignment check: it is reported with its tolerance, and a check
outside it labels every number "replica not aligned". Nothing in their outputs is a pass or a fail.

    python exp/exp89_finetuned_checkpoints.py --smoke                      # S1: numpy only, synthetic units
    python exp/exp89_finetuned_checkpoints.py --smoke-torch                # S2 code on a synthetic tar and checkpoint
    python exp/exp89_finetuned_checkpoints.py --smoke-torch --real --arm fld        # S2 on 512 real TRAINING windows
    python exp/exp89_finetuned_checkpoints.py --inventory --arm fld --tar T  # files, keys, windows and labels per split
    python exp/exp89_finetuned_checkpoints.py --gate --arm fld             # accuracy only; refused until frozen
    python exp/exp89_finetuned_checkpoints.py --arm fld                    # the full run: frozen AND gate recorded

What is reused, not rewritten. Every statistic is the package's: the ranking measures are oe_inferencex.metrics
(aurc_expected, excess_aurc, oracle_aurc, capture_at_budget_expected, attainable_ceiling, weighted_auroc with unit
weights as exp70 reads the AUROC, selective_accuracy, expected_calibration_error); the AURC bootstrap is
oe_inferencex.stats.cluster_bootstrap_difference, and the capture bootstrap draws the same resamples in the same order;
the estimate and the certificate are oe_inferencex.estimate (sample_for_estimation, estimate_error_rate,
exact_coverage_srs, certify_zone, zone_order, min_labels_to_certify), called unchanged. The confidence is
oe_inferencex.assess.assess_prediction at patch 1 with form "top1" on the trained channels; the logit margin is the
same call with form "margin". Arm A reads its windows with oe_inferencex.awf (list_windows, given the root of the
pinned tar extracted under E89_DATA, and load_window_full) and runs exp21's forward pass (logits_grid) unchanged. Its
crop is this file's crop_at, which applies exp21's rule at H = W = 63 (tested against exp21's numbers). Only the
checkpoint is loaded here, at the pinned revision, and the encoder is built
from OlmoEarth v1-Base's config.json at the revision the page names (ENCODER_REVISIONS).

Interpretations of the plan page, each flagged in the commit that added this file; none changes a threshold:
  a. Mangrove's label rule "a window is kept when its valid label pixels all agree and none is 0" is read as: all
     pixels of the 2x2 block (after rslearn's centre Pad to 2) carry the same class, from 1 to 3. The gate's pixel
     micro accuracy is over every valid (non-zero) label pixel of every validation window with imagery, kept or not,
     as Ai2's metric counts them.
  b. The logit margin is taken over the trained channels, like the graded top-1 probability, so the estimate's
     confidence design stratifies by it (margin quintiles) and allocates by the top-1 probability, as the command line's
     default `sample --logits` does. A window whose top two include the untrained channel is counted and reported.
  c. The best informative control is the lowest-AURC signal among K2, K3a, K3b, each K4 index and K5, each counted
     as one candidate.
  d. P3's one-sided 95% lower bound is the 5th percentile of the cluster bootstrap of the capture difference at 10%.
  e. The estimate draws: SeedSequence([89, arm, design, B]).generate_state(R) gives one seed per draw; arm is
     mangrove 0, nandi 1, awf 2, fld 3 and design random 0, confidence 1. The certificate reuses the random design's
     draws.
     A draw whose zone is certified and whose zone's true error rate exceeds alpha is a violation; a draw that
     certifies nothing, or that the review-set guard refuses, counts as coverage 0 and no violation (refusals are
     counted). c*(alpha) is the largest grid coverage whose zone, in that order, has a true error rate of at most alpha.
  f. K3's features are z-scored with the training split's mean and standard deviation before the MLP. The page fixes
     the network and the optimiser; it says nothing on feature scaling, and the empty-month count (0 to 12) would
     otherwise sit on another scale than the normalised bands. Nothing is fitted on the validation split.
  g. K4's NDVI controls for Nandi and AWF use exp21's formula (all twelve months, (B08 - B04) / max(B08 + B04, 1e-6)),
     so AWF's temporal control equals exp21's recorded one. The 3x3 control is the spatial standard deviation of each
     pixel's median NDVI over the months, in the 3x3 block around the label pixel. Mangrove's indices use the valid
     months only, as the page defines f over valid months.
  h. K5's threshold, B02 above 0.2 reflectance, is 2,000 in the harmonised L2A digital numbers the tar stores. The
     owner confirmed it with the other thresholds; the page's marker now says so.
  i. Arm A's gate is read as Nandi's: within 2.0 points of Ai2's 89.5% for AWF (exp21's replica was 1.4 points off),
     two-sided. Since the amendment of 1 October 2026 it is arm A's alignment check (see j). Agreement with exp21's
     recorded predictions is reported in the run's summary, not in the gate file, which holds counts and accuracy only.
  j. The gate counts attempts: at most three in all, the first and two retries. A third failed attempt closes the gate
     and nothing is graded. A passed gate is not rerun. Each attempt is appended to a ledger before the gate file is
     written. On the cluster the ledger lives outside the git checkout (E89_GATE_LEDGER), so the job's hard reset cannot
     roll the count back: a gate file behind its ledger is restored from it, and one that holds an attempt the ledger
     lacks is refused. The run checks the pass itself (require_gate): made on a frozen page, on the pinned checkpoint,
     and held by the ledger. For the report-only arms (A and F) the same machinery is an alignment check: an attempt
     records "aligned" (true or false) where a graded arm records "pass", the file says "aligned" or "replica not
     aligned", and the run needs a recorded check, not an aligned one. A check outside the tolerance labels every
     number of the run "replica not aligned"; a second or third attempt is allowed after a fix to the replica.
  k. --smoke runs the estimate study at the run's R = 2,000 draws but certifies on the first SMOKE_DRAWS of the random
     design's draws and bootstraps SMOKE_BOOT resamples, so it finishes in under a minute; the run uses R = 2,000
     and 2,000 resamples everywhere, as the page fixes.
  l. Imagery is read from the tar only. There is no fetch mode yet: if the inventory finds no Sentinel-2 layers in
     mangrove.tar, a fetch mode, and the gate's 1.0-point tolerance for refetched imagery, come before freezing. The
     tar's size makes that likely: 62,433,280 bytes is 121,940 tar blocks, and a 2x2 window with 12 item groups takes
     about 60 to 200 of them, so the tar can carry imagery for at most about 2,000 windows, not the page's expected
     12,500 validation windows. A validation window with no completed item group is dropped and counted (the gate
     records the count), and the gate refuses, recording no attempt, when no kept window has imagery.
  m. A window with fewer than 12 completed item groups is run with its own number of timesteps (batched by it), which
     is what rslearn's masked pooling over the missing timesteps reduces to; the inventory and the run count them.
  n. Nandi's polygon id is looked for under the option keys in NANDI_POLYGON_KEYS, a guess until its windows are seen,
     and its label source under the option key "source".
Arm F's readings (added with the amendment of 1 October 2026; the page marks them proposed, owner to confirm):
  o. The population is rslearn's validation set as Ai2's training config reads it: every window, in any group, whose
     options.split is "val", whose eight image layers (pre_sentinel2 and .1 to .3, post_sentinel2 and .1 to .3) and
     label layer carry rslearn's completed marker (check_window with load_all_layers), and whose label is valid. The
     label is ClassificationTask's: the first feature of layers/label/data.geojson whose new_label is one of the ten
     classes (skip_unknown_categories, allow_invalid). label.json and old_label never give a label (the inventory
     counts the windows whose label.json names another one). options.olmoearth_evals_split belongs to the paper's
     evaluation, not to the checkpoint: it never chooses a window (the inventory reports it beside options.split).
  p. The model's input is rslearn's: the eight layers in the config's order, 12 bands each in OlmoEarth's order,
     Pad(64, center) on the raw values, then OlmoEarth's normalisation. SimpleTimeSeries(image_channels=48,
     groups=[[0], [1]]) runs the encoder once on the four pre layers and once on the four post layers (timestamps day
     1, months 0 to 3, 2024, both passes), mean-pools each over timesteps and band sets, and concatenates pre then
     post (1,536 channels on a 16x16 map at patch 4). The head is rslearn's PoolingDecoder (conv 3x3 to 128 + ReLU,
     amax over the map, Linear 512 + ReLU twice, Linear to 10), loaded strictly from model.decoder.0. Ai2's
     validation used random flips and bf16 autocast; the replica uses no flip and fp32. The alignment check records
     the accuracy under each of the four flips beside it, and S2 reports the bf16 difference on training windows.
  q. K3's features (128 per window) come from the 64-px crop the model sees. Per stack (pre, post): the per-pixel
     median over its non-empty timesteps (a timestep is empty when every pixel and band is 0), its 12 bands plus NDVI
     (B08, B04) and NBR (B08, B12); their spatial mean and standard deviation over the crop and their mean over the
     centre 16 px (42 per stack); then post minus pre of those 42; then each stack's count of empty timesteps. K4 is
     the loss signal at the centre: minus the NDVI drop and minus the NBR drop from the pre to the post composite (a
     small or negative drop is suspect). K5 counts the eight timesteps that are empty or whose centre mean B02
     exceeds 2,000. K3 is fitted on every training window (options.split "train") that passes o's layer rule.
  r. The bootstrap's cluster is the 1 degree cell of the window's centre. The window group cannot be one: Ai2's split
     script gives validation windows to four groups only, and four clusters cannot carry a bootstrap. The group is the
     stratum of the by-stratum report.
  s. The tar is on Google Cloud Storage, not the Hub. It is pinned by its object generation, size and MD5 (an HTTP
     HEAD on 1 October 2026). The job downloads it with a resumable curl; verify_tar hashes it once (MD5 and SHA-256
     in one pass), refuses it unless the size and the MD5 match the pin, and records the SHA-256, which is checked
     too once it is pinned here. extract_layers then streams it once and writes only what the run reads: the eight
     image layers, the label layer, every window's top-level files and every file outside the windows (the
     dataset's config). Every other layer (Landsat, Sentinel-1, Sentinel-2 groups .4 and .5, the masks) is counted
     and skipped. Groups are not filtered: the split is read per window, and group names are not verified until the
     inventory.
  t. Certification for arm F is studied at alpha 0.10 and 0.15 (Ai2's error rate is 23.9%). At 109 windows neither
     budget (300, 1,000) fits, so its estimate and certify cells are reported as not run; c*(alpha) from all labels
     is reported for both orders.
Arm A's data (fixed with the amendment): arm A reads the pinned AWF tar the job downloads and extracts under
E89_DATA/awf, never the old data/awf (exp21's location, which the scratch purge has partly removed). Every arm's
extraction keeps a manifest; a file the purge removed later is found before anything is read, and the tar is
extracted again.

Safety, not interpretation. The checkpoint is checked against its pinned sha256, a local --ckpt included, and loaded
with weights_only=True; any other global it pickles becomes an inert stub class, and a global naming code execution is
refused. The scored checkpoint's sha256 is recorded in the replica's info and in every gate attempt.
predict_population is the one function that scores validation windows, and it refuses before the page is frozen;
compute_units and run_arm refuse without a verified gate pass (graded arms) or a verified alignment check (report-only
arms).

Outputs (exp/out): exp89_inventory_<arm>.json (--inventory), exp89_s2_<arm>.json (--smoke-torch --real),
exp89_gate_<arm>.json (--gate) and its ledger exp89_gate_<arm>.ledger.jsonl (in E89_GATE_LEDGER when set, else beside
it), exp89_units_<arm>.npz and exp89_summary.json (the run); exp89_summary_smoke.json from --smoke. --out-dir moves
them. Hub downloads go to HF_HOME (scratch on the cluster) and are checked against the pinned sha256 before use; the
tar is extracted under --data. Arm F's tar is given with --tar (the job downloads it) and checked against its pinned
size and MD5.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import traceback
import warnings

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (os.path.join(ROOT, "scripts"), ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import assess, estimate as est, metrics, stats   # noqa: E402
import exp78_error_rate_estimation as e78                           # noqa: E402  (jsonable; numpy only)

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "finetuned_checkpoints.md")

# ----------------------------------------------------------------------------- the pinned upstream (HF API, 1 Oct 2026)
MANGROVE_MODEL = "allenai/OlmoEarth-v1-FT-Mangrove-Base"
MANGROVE_DATA = "allenai/olmoearth_projects_mangrove"
NANDI_MODEL = "allenai/OlmoEarth-v1-FT-Nandi-Base"
NANDI_DATA = "allenai/olmoearth_projects_nandi"
AWF_MODEL = "allenai/OlmoEarth-v1-FT-AWF-Base"
AWF_DATA = "allenai/olmoearth_projects_awf"
FLD_MODEL = "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base"
# Arm F's dataset is on Google Cloud Storage, not the Hub. Pinned by an HTTP HEAD on 1 October 2026: the object
# generation (a GET at another generation returns 404), the size and the MD5 of x-goog-hash (equal to the ETag).
FLD_DATA_URL = ("https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/forest_loss_driver/"
                "20251029/dataset.tar")
# olmoearth_projects (GitHub) at the commit whose olmoearth_run_data configs the replica below follows
CONFIG_REVISION = "589bce143f17fb3e522a8b09ba6dfa8d36aa36ae"
# Arm F's training config: rslearn_projects data/forest_loss_driver/20251104/config.yaml, which its README ties to the
# Hub checkpoint; olmoearth_projects cec89c1660 added the 76.1% to docs/forest_loss_driver.md (30 October 2025)
FLD_CONFIG = "rslearn_projects data/forest_loss_driver/20251104/config.yaml"
FLD_CLASSES = ("agriculture", "mining", "airstrip", "road", "logging", "burned", "landslide", "hurricane", "river",
               "none")
FLD_PRE = ("pre_sentinel2", "pre_sentinel2.1", "pre_sentinel2.2", "pre_sentinel2.3")
FLD_POST = ("post_sentinel2", "post_sentinel2.1", "post_sentinel2.2", "post_sentinel2.3")
FLD_LAYERS = FLD_PRE + FLD_POST                  # rslearn concatenates them in this order: pre timesteps, then post
FLD_KEEP_LAYERS = FLD_LAYERS + ("label",)        # what extract_layers writes; every other layer is counted and skipped
FLD_CROP = 64                                    # Pad(size=64, mode="center") on the 128-px window
FLD_CENTRE = 16                                  # K3's and K4's centre support: the centre 16 px of the crop
# Ai2's split script (rslearn_projects rslp/forest_loss_driver/scripts/assign_split.py at eef0353f) gives "val" only to
# these groups, by sha256(window name)[0] in 0 to 3; reported, the population is read from options.split
FLD_VAL_GROUPS = ("20250428_brazil_phase1", "20250428_colombia_phase1", "20250428_brazil_phase2",
                  "20250428_colombia_phase2")
FLD_AI2_VAL_WINDOWS = 109                        # the sum of Ai2's confusion matrix (83 of 109 = 76.1%), reported
FLD_ENCODER_PREFIX = "model.encoder.0.encoder.model."
FLD_HEAD_PREFIX = "model.decoder.0."
FLD_HEAD_KEYS = ("model.decoder.0.conv_layers.0.0.weight", "model.decoder.0.conv_layers.0.0.bias",
                 "model.decoder.0.fc_layers.0.0.weight", "model.decoder.0.fc_layers.0.0.bias",
                 "model.decoder.0.fc_layers.1.0.weight", "model.decoder.0.fc_layers.1.0.bias",
                 "model.decoder.0.output_layer.weight", "model.decoder.0.output_layer.bias")
FLIPS = ("none", "h", "v", "hv")                 # rslearn's Flip: h reverses the columns, v the rows

# ----------------------------------------------------------------------------- fixed by the page
SEED = 89
REVIEW_BUDGETS = (0.05, 0.10, 0.20)          # 5% and 10% graded, 20% reported
EST_BUDGETS = (300, 1000)
MIN_N_GRADED = 1500                          # below it no budget is graded
DRAWS = 2000
BOOT = 2000
SMOKE_DRAWS = 300
SMOKE_BOOT = 300
DELTA = 0.10
ERROR_FLOOR = 40
P2_AUROC = 0.65
P3_POINTS = 0.05
P4_COVER = 0.93
P4_WIDTH = 0.85
P5_VIOLATION = 0.12
P5_LEAD = 0.10
EPS = 1e-12
K5_B02 = 2000.0                              # 0.2 reflectance in harmonised L2A digital numbers
MAX_GATE_ATTEMPTS = 3
ARM_INDEX = {"mangrove": 0, "nandi": 1, "awf": 2, "fld": 3}
ARM_LETTER = {"mangrove": "M", "nandi": "N", "awf": "A", "fld": "F"}
DESIGN_INDEX = {"random": 0, "confidence": 1}
OLMO_BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09")
N_MONTHS = 12
# K3, the no-encoder classifier: fixed on the page, no tuning
K3_HIDDEN, K3_DROPOUT, K3_LR, K3_WD, K3_BATCH, K3_EPOCHS = 256, 0.1, 1e-3, 1e-4, 512, 30
K3_CAP_MANGROVE = 20000
S2_WINDOWS = 512
S2_MIN_ACCURACY = 0.95                       # Mangrove only; in-sample, a loading check
PREDICTIONS = ("P1", "P2", "P3", "P4", "P5")

ARMS = {
    "mangrove": {
        "model": MANGROVE_MODEL, "model_revision": "57ee738b5b4474ad61391ed612e4fbc2a034353e",
        "model_file": "model.ckpt", "model_sha256": "ffb51fbb58cf96219df848303268445a710a804465c2e57a823e26ec40b3eb2b",
        "model_bytes": 1041362118,
        "data": MANGROVE_DATA, "data_revision": "3a878b8a9e54ef627939d48cab334f74e89fa634",
        "data_file": "mangrove.tar", "data_sha256": "ef9dba6a18ec03217e233abf9652423612d8f42999d69637920aa31d1e2d3594",
        "data_bytes": 62433280,
        "extra_files": {"annotation_features.geojson": "e87bc7919b3a1684aca1c1e3d5fd0b76f85c61c77f5cec1d610cce7ed72fa2a1"},
        "group": "sample_100K", "n_out": 4, "trained": (1, 2, 3), "untrained": (0,),
        "class_names": {1: "mangrove", 2: "water", 3: "other"},
        "label_layer": ("label_raster", "label"), "label_fill": 0, "unit": "block2", "patch": 2,
        "head_keys": ("model.decoders.mangrove_classification.0.output_layer.weight",
                      "model.decoders.mangrove_classification.0.output_layer.bias"),
        "head_kind": "linear",
        "ai2_accuracy": 0.976, "gate_tolerance": {"tar": 0.005, "fetched": 0.010},
        "graded": ("P1", "P2", "P3", "P4", "P5"), "p1_capture": 0.40, "alpha": 0.02, "alpha_reported": (0.01,),
        "p5_coverage": 0.50, "k3_cap": K3_CAP_MANGROVE, "clusters": "0.1 degree cell", "clusters_reported": "1 degree cell",
        "report_only": False,
        # where Ai2's validation split for the sample_100K windows comes from; None until Ai2 shares it (asked in Slack
        # on 1 October 2026). While None, every mode that scores a window reports arm M as not run.
        "split_source": None,
        "not_run_reason": "the public mangrove.tar (allenai/olmoearth_projects_mangrove at 3a878b8a) holds 100,000 "
                          "labelled points and 2,000 task areas, with no train/val split and no imagery (the inventory "
                          "on the cluster, 1 October 2026), so Ai2's 97.6% cannot be matched. Ai2 was asked for the "
                          "validation split on 1 October 2026; arm M keeps its graded predictions and runs once the "
                          "split is shared and pinned here",
    },
    "nandi": {
        "model": NANDI_MODEL, "model_revision": None, "model_file": "model.ckpt", "model_sha256": None,
        "data": NANDI_DATA, "data_revision": None, "data_file": None, "data_sha256": None, "extra_files": {},
        "group": "spatial_split", "n_out": 11, "trained": tuple(range(10)), "untrained": (10,),
        "class_names": {0: "coffee", 1: "grassland", 2: "trees", 3: "maize", 4: "sugarcane", 5: "tea",
                        6: "vegetables", 7: "legumes", 8: "water", 9: "builtup"},
        "label_layer": ("label", "category"), "label_fill": 10, "unit": "pixel", "patch": 1, "crop": 16,
        "head_keys": ("model.decoders.segment.1.layer.weight", "model.decoders.segment.1.layer.bias"),
        "head_kind": "conv1x1",
        "ai2_accuracy": 0.873, "gate_tolerance": {"tar": 0.020, "fetched": 0.020},
        "graded": ("P1", "P2", "P3", "P4", "P5"), "p1_capture": 0.30, "alpha": 0.05, "alpha_reported": (0.10,),
        "p5_coverage": 0.30, "k3_cap": None, "clusters": "128-px split cell, merged by source polygon",
        "clusters_reported": "0.05 degree cell", "report_only": False,
        "not_run_reason": "allenai/OlmoEarth-v1-FT-Nandi-Base and allenai/olmoearth_projects_nandi returned 401 on the "
                          "Hub on 1 October 2026; arm N runs only if Ai2 publishes both and their revisions are pinned",
    },
    "awf": {
        "model": AWF_MODEL, "model_revision": "a347b1546ab881c92aa125400ed5acc8126394ca", "model_file": "model.ckpt",
        "model_sha256": "1036b76bf7dfde819479dfa02507634f59a5637f14d4071bc3bf18f4d3d4bab2", "model_bytes": 1041443421,
        "data": AWF_DATA, "data_revision": "da8eb6aab65b518cb8223912ba9091cbaa5dad2f",
        "data_file": "dataset.tar", "data_sha256": "d0837f14235f9ef3207af6d84c428bb9dcb81bd18ee6a96d03ce4b5fff992090",
        "data_bytes": 1869219840, "extra_files": {},
        "group": "spatial_split", "n_out": 10, "trained": tuple(range(9)), "untrained": (9,),
        "class_names": None, "label_layer": ("label", "category"), "label_fill": 9, "unit": "pixel", "patch": 4,
        "crop": 16, "head_keys": ("model.decoders.segment.1.layer.weight", "model.decoders.segment.1.layer.bias"),
        "head_kind": "conv1x1_up4",
        "ai2_accuracy": 0.895, "gate_tolerance": {"tar": 0.020, "fetched": 0.020},
        # report-only since the amendment of 1 October 2026: 344 validation points, about 36 errors at Ai2's 89.5%
        "graded": (), "report_only": True, "p1_capture": None, "alpha": 0.05, "alpha_reported": (0.10,),
        "p5_coverage": None, "k3_cap": None, "clusters": "exp21's annotation task", "clusters_reported": None,
    },
    "fld": {
        "model": FLD_MODEL, "model_revision": "15502f8acb4caed6e3a7d777b0fb569c1f7eb791", "model_file": "model.ckpt",
        "model_sha256": "682c20b99f44a81769ad7d3ce1b64d933fae3810bcf5978f9999cfebf1c11855", "model_bytes": 381184175,
        "data": None, "data_revision": None, "data_file": "dataset.tar", "extra_files": {},
        "data_url": FLD_DATA_URL, "data_generation": "1761857427036506",
        "data_md5": "6abc5b944c64330f0a545d5ed7ddddf4", "data_bytes": 42214604800,
        # not known before the first download (Cloud Storage gives MD5 and CRC32C only); verify_tar records it and
        # checks it once it is pinned here
        "data_sha256": None,
        "group": None, "n_out": 10, "trained": tuple(range(10)), "untrained": (),
        "class_names": dict(enumerate(FLD_CLASSES)), "label_layer": ("label", "data.geojson"), "label_fill": None,
        "unit": "window", "patch": 4, "crop": FLD_CROP, "head_keys": FLD_HEAD_KEYS, "head_kind": "pooling_decoder",
        "ai2_accuracy": 0.761, "gate_tolerance": {"tar": 0.020, "fetched": 0.020},
        # report-only: 109 validation windows, about 26 errors at Ai2's 76.1%
        "graded": (), "report_only": True, "p1_capture": None, "alpha": 0.10, "alpha_reported": (0.15,),
        "p5_coverage": None, "k3_cap": None, "clusters": "1 degree cell", "clusters_reported": None,
    },
}
# Where the encoder's keys sit. Arm F wraps OlmoEarth in SimpleTimeSeries, so its keys sit one level deeper: with
# rslearn 0.0.12's wrapper (which keeps the model's encoder as .model) under model.encoder.0.encoder.model., and one
# level deeper still if the wrapper kept the whole model. A strict load into the encoder decides which one holds.
ENCODER_PREFIXES = ("model.encoder.0.model.", "model.encoder.0.model.encoder.", FLD_ENCODER_PREFIX,
                    FLD_ENCODER_PREFIX + "encoder.", "model.encoder.0.")
# The encoder skeletons' config.json at the revisions exp/out/upstream_revisions.json records (the page names Base's)
ENCODER_REVISIONS = {"OLMOEARTH_V1_BASE": "4bd1392a4539404d2c74276c39f3cb4cfff466cc",
                     "OLMOEARTH_V1_NANO": "529248a4dc3c54014c56b7504641cec98de31d1c"}
# A Lightning checkpoint pickles globals besides tensors (rslearn and jsonargparse objects in hyper_parameters, numpy
# scalars in callback state). Each is mapped to an inert stub class under its own name: nothing is imported or run.
# A global that names code execution has no place in a checkpoint and is refused outright.
REFUSED_PREFIXES = ("os.", "posix.", "nt.", "subprocess.", "sys.", "builtins.", "shutil.", "socket.", "importlib.",
                    "runpy.", "pickle.", "marshal.", "code.", "ctypes.", "pty.", "multiprocessing.", "webbrowser.")
GATE_KEYS = ("arm", "prereg_status", "ai2_accuracy", "metric", "tolerance_points", "imagery_route", "attempts",
             "passed", "closed", "max_attempts")
ATTEMPT_KEYS = ("attempt", "utc", "commit", "dirty", "prereg_status", "checkpoint_sha256", "n_windows", "n_errors",
                "accuracy", "n_windows_dropped", "n_windows_no_imagery", "n_pixels", "n_pixel_errors", "accuracy_pixel",
                "accuracy_by_flip", "gap_points", "pass")
# the report-only arms' alignment check: the same record with "aligned" where a graded arm has "pass" and "passed"
ALIGN_KEYS = tuple(k for k in GATE_KEYS if k != "passed") + ("aligned", "alignment", "report_only")
ALIGN_ATTEMPT_KEYS = tuple(k for k in ATTEMPT_KEYS if k != "pass") + ("aligned",)
NOT_ALIGNED = "replica not aligned"
GATE_LEDGER_ENV = "E89_GATE_LEDGER"          # the gate's ledger directory; exp/jobs/e89.sh puts it outside the checkout


class GateRefused(RuntimeError):
    """The gate, or a run that needs its pass, refuses: nothing is scored or graded."""


# ----------------------------------------------------------------------------- small helpers
def prereg_status(text=None):
    """'frozen', 'draft' or 'unknown', from the plan page's status line (the first line starting '**Status'). Copied
    from exp88: "frozen" must come first after the label, so "not frozen" never matches."""
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    for line in text.splitlines():
        if line.startswith("**Status"):
            low = line.lower()
            if re.match(r"\*\*status:?\*?\*?\s*frozen", low):
                return "frozen"
            if "draft" in low or "not frozen" in low:
                return "draft"
            return "unknown"
    return "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return commit, dirty


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(e78.jsonable(obj), f, indent=1)


def fail(record, part, ex):
    record.setdefault("failures", []).append({"part": part, "error": repr(ex), "traceback": traceback.format_exc()})
    print(f"{part} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)


def bkey(b):
    """A review budget as the summary's key: '0.05', '0.1', '0.2'."""
    return repr(float(b))


def is_report_only(arm):
    """Arms A and F since the amendment of 1 October 2026: every measure is reported, no prediction is graded."""
    return bool(ARMS[arm].get("report_only"))


def waiting_reason(arm):
    """Why arm M cannot score a window yet (None once Ai2's split is pinned in ARMS["mangrove"]["split_source"])."""
    if arm == "mangrove" and ARMS["mangrove"].get("split_source") is None:
        return ARMS["mangrove"]["not_run_reason"]
    return None


# ----------------------------------------------------------------------------- the channel and label rules per arm
def pad_center(a, size, fill=0):
    """rslearn's Pad(size, mode="center") on the last two axes: pad equally (the extra pixel after) when smaller,
    crop when larger, with the same floor rule rslearn uses for an odd difference."""
    a = np.asarray(a)
    out = a
    for axis in (-2, -1):
        extra = size - out.shape[axis]
        before = extra // 2
        after = extra - before
        if extra > 0:
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, after)
            out = np.pad(out, pad, constant_values=fill)
        elif extra < 0:
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(-before, out.shape[axis] + after)
            out = out[tuple(sl)]
    return out


def mangrove_unit_label(lab2):
    """(class or None, reason): a 2x2 label block is a unit when all four pixels hold the same class from 1 to 3."""
    v = np.asarray(lab2).ravel()
    if (v == 0).any():
        return None, "a label pixel is 0"
    if not (v == v[0]).all():
        return None, "label pixels disagree"
    if int(v[0]) not in ARMS["mangrove"]["trained"]:
        return None, "class outside 1 to 3"
    return int(v[0]), None


def single_pixel_label(lab, fill, trained):
    """(row, col, class) or (None, reason): Nandi and AWF keep a window with exactly one labelled pixel."""
    lab = np.asarray(lab)
    rows, cols = np.nonzero(lab != fill)
    if len(rows) != 1:
        return None, f"{len(rows)} labelled pixels"
    c = int(lab[rows[0], cols[0]])
    if c not in trained:
        return None, f"class {c} outside the trained classes"
    return (int(rows[0]), int(cols[0]), c), None


def fld_label_from_features(features):
    """(class index or None, drop reason or None, the new_label read: the valid one, else the first one seen):
    rslearn's ClassificationTask with
    property_name new_label, skip_unknown_categories and allow_invalid. A feature without properties or without
    new_label is passed over, and so is one whose new_label is not one of the ten classes; the first feature with one
    of the ten classes gives the window's class. A window with none is invalid (masked out of Ai2's accuracy)."""
    first = None
    for ft in features or []:
        props = (ft or {}).get("properties")
        if not props or "new_label" not in props:
            continue
        v = props["new_label"]
        if first is None:
            first = v
        if v in FLD_CLASSES:
            return FLD_CLASSES.index(v), None, v
    return None, ("no feature has new_label" if first is None else "new_label outside the ten classes"), first


def fld_label(wdir):
    """Arm F's label from the vector layer `label` (layers/label/data.geojson), which rslearn reads. label.json at the
    window's top is never read (rslearn does not read it, and Ai2's label sync rewrites data.geojson only). The label
    layer must carry rslearn's completed marker, as check_window requires of the targets."""
    ldir = os.path.join(wdir, "layers", "label")
    if not os.path.exists(os.path.join(ldir, "completed")):
        return None, "label layer not completed", None
    path = os.path.join(ldir, "data.geojson")
    if not os.path.exists(path):
        return None, "no data.geojson in the label layer", None
    with open(path) as f:
        return fld_label_from_features(json.load(f).get("features"))


def fld_layers_completed(wdir):
    """Arm F's eight image layers that carry rslearn's completed marker. check_window with load_all_layers keeps a
    window only when all eight do."""
    return [n for n in FLD_LAYERS if os.path.exists(os.path.join(wdir, "layers", n, "completed"))]


def crop_at(H, W, r, c, size, shift=0):
    """exp21's crop rule on an H x W window: the label pixel at (size // 2 - shift) when the crop fits, clamped at the
    edges. Returns (r0, c0, row in crop, col in crop)."""
    r0 = min(max(r - size // 2 + shift, 0), H - size)
    c0 = min(max(c - size // 2 + shift, 0), W - size)
    return r0, c0, r - r0, c - c0


def readings(logits, arm):
    """Per unit: the prediction (argmax over every output channel, as rslearn takes it), the graded confidence (top-1
    softmax probability over the trained channels, assess_prediction form "top1" at patch 1), the logit margin over
    the trained channels (form "margin"), the entropy over the trained channels, and two counts the page asks for."""
    A = ARMS[arm]
    L = np.asarray(logits, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] != A["n_out"]:
        raise ValueError(f"{arm}: logits are {L.shape}, expected (N, {A['n_out']})")
    trained = list(A["trained"])
    pred = L.argmax(1)
    scores = L[:, trained].T[:, None, :]                       # (C_trained, 1, N): one window per unit at patch 1
    top1 = assess.assess_prediction(scores, is_logit=True, patch=1, budgets=(0.05,), form="top1")
    mar = assess.assess_prediction(scores, is_logit=True, patch=1, budgets=(0.05,), form="margin")
    p1 = np.asarray(top1["arrays"]["confidence"], dtype=np.float64).ravel()
    margin = np.asarray(mar["arrays"]["confidence"], dtype=np.float64).ravel()
    z = L[:, trained] - L[:, trained].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    top2 = np.argsort(-L, axis=1, kind="stable")[:, :2]
    return {"pred": pred, "p1": p1, "margin": margin, "entropy": entropy,
            "n_pred_untrained": int((~np.isin(pred, trained)).sum()),
            "n_untrained_in_top2": int(np.isin(top2, list(A["untrained"])).any(1).sum()),
            "n_p1_saturated": int((p1 >= 1.0).sum())}


# ----------------------------------------------------------------------------- controls from the twelve mosaics
def unit_indices(series):
    """series (N, 12 months, 12 bands) raw values in OlmoEarth band order -> NDVI and MNDWI per month, and the empty
    months (every band 0: no mosaic covered the unit, or the item group is absent)."""
    s = np.asarray(series, dtype=np.float64)
    b = {name: s[..., i] for i, name in enumerate(OLMO_BANDS)}
    empty = (s == 0).all(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"])
        mndwi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
    return ndvi, mndwi, empty


def exp21_ndvi(series):
    """exp21's NDVI: every month, (B08 - B04) / max(B08 + B04, 1e-6)."""
    s = np.asarray(series, dtype=np.float64)
    red, nir = s[..., OLMO_BANDS.index("B04")], s[..., OLMO_BANDS.index("B08")]
    return (nir - red) / np.maximum(nir + red, 1e-6)


def index_controls(arm, series, ndvi3x3=None):
    """K4 and K5 per unit, each a suspicion (higher = more suspect). `series` is the unit's (N, 12, 12) raw series
    (Mangrove: the 2x2 block mean; Nandi and AWF: the label pixel); `ndvi3x3` is (N, 3, 3, 12) exp21-NDVI around the
    label pixel for Nandi and AWF."""
    ndvi, mndwi, empty = unit_indices(series)
    valid = ~empty
    out = {}
    if arm == "mangrove":
        n_valid = valid.sum(1)
        with np.errstate(invalid="ignore"):
            f = np.where(n_valid > 0, ((mndwi > 0) & valid).sum(1) / np.maximum(n_valid, 1), np.nan)
        med = np.array([np.median(m[v]) if v.any() else np.nan for m, v in zip(mndwi, valid)])
        std = np.array([np.std(n[v]) if v.any() else np.nan for n, v in zip(ndvi, valid)])
        out["k4_inundation_ambiguity"] = np.minimum(f, 1 - f)
        out["k4_mndwi_near_zero"] = -np.abs(med)
        out["k4_ndvi_month_std"] = std
    else:
        out["k4_ndvi_temporal_std"] = np.nanstd(exp21_ndvi(series), axis=1)
        if ndvi3x3 is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)              # NaN pixels outside a window edge
                med = np.nanmedian(np.asarray(ndvi3x3, dtype=np.float64), axis=-1)   # each pixel's median over months
                out["k4_ndvi_3x3_std"] = np.nanstd(med.reshape(med.shape[0], -1), axis=1)
    b02 = np.asarray(series, dtype=np.float64)[..., OLMO_BANDS.index("B02")]
    out["k5_cloud_months"] = (empty | (b02 > K5_B02)).sum(1).astype(np.float64)
    # a unit with no valid month has no index: it is put at the suspect end (it is suspect), not dropped
    for k, v in out.items():
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.nanmax(v[~bad]) + 1.0 if (~bad).any() else 0.0
    return out


def k3_features(series, normalize, arm):
    """K3's inputs per unit: per month the 12 normalised bands, NDVI (and MNDWI for Mangrove), each empty month filled
    with the unit's median of that feature over its non-empty months; then the count of empty months.
    `normalize` maps raw (N, 12, 12) to OlmoEarth's computed normalisation (std multiplier 2)."""
    s = np.asarray(series, dtype=np.float64)
    ndvi, mndwi, empty = unit_indices(s)
    parts = [np.asarray(normalize(s), dtype=np.float64), ndvi[..., None]]
    if arm == "mangrove":
        parts.append(mndwi[..., None])
    x = np.concatenate(parts, axis=-1)                         # (N, 12, F)
    x = np.where(empty[..., None], np.nan, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # a unit with every month empty
        med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True) if x.size else x
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isfinite(x), x, np.broadcast_to(med, x.shape))
    return np.concatenate([x.reshape(x.shape[0], -1), empty.sum(1, keepdims=True).astype(np.float64)], axis=1)


def fit_no_encoder_classifier(x_train, y_train, classes, x_eval, seed=SEED):
    """K3: an MLP in torch with the page's fixed settings (two hidden layers of 256, ReLU, dropout 0.1, Adam at 1e-3
    with weight decay 1e-4, batch 512, 30 epochs, seed 89, the last epoch). CPU, so the fit is deterministic. Returns
    the class probabilities on x_eval, (N_eval, len(classes)), columns in the order of `classes`, and the fit record."""
    import torch
    classes = list(classes)
    pos = {c: i for i, c in enumerate(classes)}
    y = np.array([pos[int(v)] for v in y_train], dtype=np.int64)
    xt = np.asarray(x_train, dtype=np.float64)
    mu, sd = xt.mean(0), xt.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    xt = ((xt - mu) / sd).astype(np.float32)
    xe = ((np.asarray(x_eval, dtype=np.float64) - mu) / sd).astype(np.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], K3_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(K3_DROPOUT),
                              torch.nn.Linear(K3_HIDDEN, K3_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(K3_DROPOUT),
                              torch.nn.Linear(K3_HIDDEN, len(classes)))
    opt = torch.optim.Adam(net.parameters(), lr=K3_LR, weight_decay=K3_WD)
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.from_numpy(xt), torch.from_numpy(y)
    losses = []
    net.train()
    for _ in range(K3_EPOCHS):
        order = torch.randperm(len(Y), generator=g)
        tot = 0.0
        for i in range(0, len(Y), K3_BATCH):
            idx = order[i:i + K3_BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        losses.append(tot / len(Y))
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(xe)), dim=1).double().numpy()
        train_acc = float((net(X).argmax(1) == Y).double().mean())
    return prob, {"n_train": int(len(Y)), "n_features": int(xt.shape[1]), "classes": classes,
                  "loss_last_epoch": losses[-1], "loss_first_epoch": losses[0], "train_accuracy": train_acc}


def k3_signals(prob, classes, pred):
    """K3a = 1 - K3's top-1 probability; K3b = 1 where K3's class differs from the fine-tuned model's prediction."""
    prob = np.asarray(prob, dtype=np.float64)
    k3_class = np.asarray(classes)[prob.argmax(1)]
    return {"k3a_no_encoder_uncertainty": 1.0 - prob.max(1),
            "k3b_no_encoder_disagreement": (k3_class != np.asarray(pred)).astype(np.float64)}, k3_class


def class_rarity(pred, train_labels, trained):
    """K2: minus the training frequency of the predicted class (rarer = more suspect); an untrained class has 0."""
    lab = np.asarray(train_labels)
    freq = {int(c): float((lab == c).mean()) if lab.size else 0.0 for c in trained}
    return -np.array([freq.get(int(p), 0.0) for p in pred], dtype=np.float64), freq


# ----------------------------------------------------------------------------- arm F's controls, from the model's crop
def _ratio(a, b):
    """(a - b) / max(a + b, 1e-6), exp21's guard, for NDVI and NBR."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (a - b) / np.maximum(a + b, 1e-6)


def fld_composite(stack4):
    """One stack (S, S, 4, 12) raw -> its per-pixel median over the non-empty timesteps (S, S, 12) and the number of
    empty timesteps. A timestep is empty when every pixel and band is 0; a stack with no non-empty timestep is 0."""
    s = np.asarray(stack4, dtype=np.float64)
    empty = (s == 0).all(axis=(0, 1, 3))
    if empty.all():
        return np.zeros(s.shape[:2] + (s.shape[3],)), int(empty.sum())
    return np.median(s[:, :, ~empty, :], axis=2), int(empty.sum())


def fld_window_features(crop):
    """Arm F's no-encoder features (K3) and index inputs (K4, K5) from the model's crop (S, S, 8, 12) raw, pre
    timesteps 0 to 3 then post 4 to 7, bands in OlmoEarth's order. Per stack: the composite's 12 bands, NDVI (B08, B04)
    and NBR (B08, B12); their spatial mean and standard deviation over the crop and their mean over the centre 16 px
    (42 per stack). Then post minus pre of those 42, then each stack's count of empty timesteps: 128 features."""
    crop = np.asarray(crop, dtype=np.float64)
    if crop.ndim != 4 or crop.shape[2] != len(FLD_LAYERS) or crop.shape[3] != len(OLMO_BANDS):
        raise ValueError(f"arm F's crop is (S, S, 8, 12), not {crop.shape}")
    S = crop.shape[0]
    c0 = max((S - FLD_CENTRE) // 2, 0)
    centre = slice(c0, c0 + FLD_CENTRE)
    bi = {b: OLMO_BANDS.index(b) for b in ("B02", "B04", "B08", "B12")}
    per, idx = {}, {}
    for name, sl in (("pre", slice(0, 4)), ("post", slice(4, 8))):
        comp, n_empty = fld_composite(crop[:, :, sl, :])
        ndvi = _ratio(comp[..., bi["B08"]], comp[..., bi["B04"]])
        nbr = _ratio(comp[..., bi["B08"]], comp[..., bi["B12"]])
        img = np.concatenate([comp, ndvi[..., None], nbr[..., None]], axis=-1)            # (S, S, 14)
        cen = img[centre, centre]
        per[name] = np.concatenate([img.mean((0, 1)), img.std((0, 1)), cen.mean((0, 1))])
        idx[name] = {"ndvi_centre": float(cen[..., 12].mean()), "nbr_centre": float(cen[..., 13].mean()),
                     "n_empty": n_empty}
    x = np.concatenate([per["pre"], per["post"], per["post"] - per["pre"],
                        [idx["pre"]["n_empty"], idx["post"]["n_empty"]]])
    empty_t = (crop == 0).all(axis=(0, 1, 3))
    b02 = crop[centre, centre, :, bi["B02"]].mean((0, 1))
    idx["cloud_timesteps"] = int((empty_t | (b02 > K5_B02)).sum())
    return x, idx


def _fld_feature_names():
    """The 128 names of fld_window_features' vector, in its order."""
    base = list(OLMO_BANDS) + ["NDVI", "NBR"]
    one = [f"{s}_{b}" for s in ("crop_mean", "crop_std", "centre_mean") for b in base]
    return ([f"pre_{n}" for n in one] + [f"post_{n}" for n in one] + [f"diff_{n}" for n in one]
            + ["pre_empty_timesteps", "post_empty_timesteps"])


FLD_FEATURE_NAMES = _fld_feature_names()


def fld_index_controls(indices):
    """K4 and K5 per window for arm F (higher = more suspect): minus the centre's NDVI drop and minus its NBR drop
    from the pre to the post composite (a small or negative drop is a weak loss signal), and the timesteps that are
    empty or cloudy at the centre (K5)."""
    pre_ndvi = np.array([d["pre"]["ndvi_centre"] for d in indices], dtype=np.float64)
    post_ndvi = np.array([d["post"]["ndvi_centre"] for d in indices], dtype=np.float64)
    pre_nbr = np.array([d["pre"]["nbr_centre"] for d in indices], dtype=np.float64)
    post_nbr = np.array([d["post"]["nbr_centre"] for d in indices], dtype=np.float64)
    return {"k4_ndvi_drop_weak": -(pre_ndvi - post_ndvi), "k4_nbr_drop_weak": -(pre_nbr - post_nbr),
            "k5_cloud_timesteps": np.array([d["cloud_timesteps"] for d in indices], dtype=np.float64)}


# ----------------------------------------------------------------------------- measures 1-4: accuracy and ranking
def signal_table(u, err):
    u = np.asarray(u, dtype=np.float64)
    e = np.asarray(err, dtype=np.float64)
    n, k = e.size, int(e.sum())
    theta = k / n
    aurc = metrics.aurc_expected(u, e)
    oracle = metrics.oracle_aurc(n, k)
    cap = metrics.capture_at_budget_expected(u, e, REVIEW_BUDGETS)
    return {"auroc": float(metrics.weighted_auroc(u, e, np.ones(n))), "aurc": aurc,
            "excess_aurc": float(metrics.excess_aurc(u, e)),
            "gap_closed": float((theta - aurc) / (theta - oracle)) if theta > oracle else float("nan"),
            "capture": {bkey(b): float(v) for b, v in cap.items()},
            "capture_random": {bkey(b): float(max(1, int(round(b * n))) / n) for b in REVIEW_BUDGETS},
            "ceiling": {bkey(b): metrics.attainable_ceiling(b, theta, n) if k else float("nan") for b in REVIEW_BUDGETS}}


def capture_bootstrap(score_a, score_b, errors, clusters, budget=0.10, n_boot=BOOT, seed=SEED):
    """Cluster bootstrap of capture(a) - capture(b) at a review budget, positive favours a. The resampling is
    stats.cluster_bootstrap_difference's, draw for draw: the same generator, the same calls, the same skipped
    resamples with no error."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(score_a, dtype=np.float64).ravel(), np.asarray(score_b, dtype=np.float64).ravel()
    err, cl = np.asarray(errors, dtype=np.float64).ravel(), np.asarray(clusters).ravel()
    ids = np.unique(cl)
    idx_by = {c: np.flatnonzero(cl == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        if err[sel].sum() == 0:
            continue
        ca = metrics.capture_at_budget_expected(a[sel], err[sel], (budget,))[budget]
        cb = metrics.capture_at_budget_expected(b[sel], err[sel], (budget,))[budget]
        diffs.append(ca - cb)
    d = np.asarray(diffs)
    if d.size == 0:
        return {"n_resamples": 0, "lower_one_sided_95": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "p_a_better": float("nan")}
    return {"n_resamples": int(d.size), "lower_one_sided_95": float(np.percentile(d, 5)),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_a_better": float((d > 0).mean())}


def control_names(controls):
    return [k for k in controls if k.startswith(("k2_", "k3a_", "k3b_", "k4_", "k5_"))]


def ranking(err, conf_u, margin_u, entropy_u, controls, clusters, n_boot=BOOT, clusters_coarse=None):
    """Measures 2 and 3: every signal's AUROC, captures, ceilings, excess AURC and gap closed; the best informative
    control (lowest AURC of K2 to K5); confidence minus that control at 5% and 10% and in AURC, with cluster-bootstrap
    intervals (and the coarse clusters' beside them, reported)."""
    err = np.asarray(err, dtype=np.float64)
    sig = {"confidence": conf_u, "margin": margin_u, "entropy": entropy_u, **controls}
    table = {k: signal_table(v, err) for k, v in sig.items()}
    cands = control_names(controls)
    best = min(cands, key=lambda k: (table[k]["aurc"], k)) if cands else None
    out = {"signals": table, "candidates": cands, "best_control": best}
    if best is None or err.sum() == 0:
        return out
    conf, ctl = np.asarray(conf_u, dtype=np.float64), np.asarray(controls[best], dtype=np.float64)
    lead = {}
    for b in (0.05, 0.10):
        lead[bkey(b)] = {"difference": table["confidence"]["capture"][bkey(b)] - table[best]["capture"][bkey(b)],
                         "bootstrap": capture_bootstrap(conf, ctl, err, clusters, b, n_boot, SEED)}
    lo, hi, p = stats.cluster_bootstrap_difference(conf, ctl, err, clusters, n_boot=n_boot, seed=SEED)
    lead["aurc"] = {"difference": table["confidence"]["aurc"] - table[best]["aurc"],
                    "bootstrap": {"lo95": lo, "hi95": hi, "p_confidence_better": p}}
    out["confidence_minus_best_control"] = lead
    if clusters_coarse is not None:
        lo, hi, p = stats.cluster_bootstrap_difference(conf, ctl, err, clusters_coarse, n_boot=n_boot, seed=SEED)
        out["coarse_clusters_reported"] = {
            "capture_0.1": capture_bootstrap(conf, ctl, err, clusters_coarse, 0.10, n_boot, SEED),
            "aurc": {"lo95": lo, "hi95": hi, "p_confidence_better": p}, "n_clusters": int(np.unique(clusters_coarse).size)}
    return out


def accuracy_block(label, pred, err, p1, clusters, n_out, class_names, pixel=None):
    """Measures 1 and 4: accuracy, per-class recall, the confusion matrix, errors, clusters with an error, selective
    accuracy and ECE."""
    label, pred = np.asarray(label), np.asarray(pred)
    correct = 1.0 - np.asarray(err, dtype=np.float64)
    out = {"n_units": int(label.size), "n_errors": int(err.sum()), "accuracy_window": float(correct.mean()),
           "error_rate": float(np.mean(err)),
           "n_clusters": int(np.unique(clusters).size),
           "n_clusters_with_an_error": int(np.unique(np.asarray(clusters)[np.asarray(err) > 0]).size),
           "per_class_recall": {}, "confusion": {}}
    for c in np.unique(label):
        m = label == c
        name = (class_names or {}).get(int(c), str(int(c)))
        out["per_class_recall"][name] = {"class": int(c), "n": int(m.sum()), "recall": float((pred[m] == c).mean())}
        out["confusion"][str(int(c))] = np.bincount(pred[m], minlength=n_out).tolist()
    if pixel is not None:
        out["accuracy_pixel_micro"] = pixel
    out["selective_accuracy"] = {bkey(c): v for c, v in
                                 metrics.selective_accuracy(1.0 - np.asarray(p1), correct).items()}
    ece, rows = metrics.expected_calibration_error(np.asarray(p1), correct, bins=10)
    out["ece_10_bins"] = ece
    out["reliability"] = rows
    return out


# ----------------------------------------------------------------------------- measures 5-6: estimate and certify
def graded_budgets(N):
    """The page: a budget is graded only where B <= N/5, and none where N < 1,500."""
    return [B for B in EST_BUDGETS if N >= MIN_N_GRADED and B <= N / 5]


def draw_seeds(arm, design, B, draws):
    return np.random.SeedSequence([SEED, ARM_INDEX[arm], DESIGN_INDEX[design], int(B)]).generate_state(draws)


def estimate_study(err, margin, p1, arm, draws=DRAWS):
    """Both designs at every budget the population holds, through sample_for_estimation and estimate_error_rate
    unchanged. Returns (cells, the random design's draws {B: (R, B) indices}) for the certificate."""
    err = np.asarray(err, dtype=np.float64)
    N, K = err.size, int(err.sum())
    theta = K / N
    cells, random_draws = {}, {}
    for design in ("random", "confidence"):
        for B in EST_BUDGETS:
            if B > N:
                cells[f"{design}/{B}"] = {"run": False, "reason": f"the budget exceeds the {N} units"}
                continue
            seeds = draw_seeds(arm, design, B, draws)
            ests, lows, highs = np.empty(draws), np.empty(draws), np.empty(draws)
            idx_all = np.empty((draws, B), dtype=np.int32) if design == "random" else None
            for r, s in enumerate(seeds):
                smp = est.sample_for_estimation(margin, B, design=design, p1=p1, seed=int(s))
                idx = np.asarray(smp["indices"], int)
                res = est.estimate_error_rate(smp, err[idx])
                ests[r], lows[r], highs[r] = res["estimate"], res["low"], res["high"]
                if idx_all is not None:
                    idx_all[r] = idx
            cover = (lows <= theta) & (theta <= highs)                 # as exp78 and exp88 grade it
            width = highs - lows
            cell = {"run": True, "design": design, "budget": B, "draws": draws, "theta": theta,
                    "coverage": float(cover.mean()), "median_width": float(np.median(width)),
                    "mean_width": float(width.mean()), "bias": float(ests.mean() - theta),
                    "rmse": float(np.sqrt(((ests - theta) ** 2).mean())), "mean_estimate": float(ests.mean())}
            if design == "random":
                cell["exact_coverage"] = float(est.exact_coverage_srs(N, K, B, interval=est.hypergeom_interval))
                random_draws[B] = idx_all
            cells[f"{design}/{B}"] = cell
    return cells, random_draws


def best_coverage(score, err, alpha):
    """c*(alpha): the largest grid coverage whose zone in this order (zone_order: descending score, ties by index)
    has a true error rate of at most alpha, from all labels."""
    order, _ = est.zone_order(score)
    err = np.asarray(err, dtype=np.float64)
    best = 0.0
    for c in est.ZONE_GRID:
        n = max(1, int(round(c * order.size)))
        if err[order[:n]].mean() <= alpha + EPS:
            best = c
    return best


def certify_study(err, orders, random_draws, alphas):
    """certify_zone on the random design's draws, per budget, alpha and order; the prefix rule for every order and
    the Bonferroni rule beside it for the confidence order."""
    err = np.asarray(err, dtype=np.float64)
    out = {}
    for B, idx_all in random_draws.items():
        for alpha in alphas:
            for oname, score in orders.items():
                for rule in (("prefix", "bonferroni") if oname == "confidence" else ("prefix",)):
                    cov, viol, nothing, refused, rates = [], 0, 0, 0, []
                    for idx in idx_all:
                        try:
                            res = est.certify_zone(score, idx, err[idx], alpha, delta=DELTA, rule=rule,
                                                   cut="standard")             # recorded with the standard cut
                        except ValueError as ex:
                            if "random sample" not in str(ex) and "enriched" not in str(ex):
                                raise
                            refused += 1
                            cov.append(0.0)
                            nothing += 1
                            continue
                        if res["coverage"] is None:
                            cov.append(0.0)
                            nothing += 1
                            continue
                        rate = float(err[res["zone_indices_in_order"]].mean())
                        rates.append(rate)
                        cov.append(float(res["coverage"]))
                        viol += rate > alpha + EPS
                    R = len(idx_all)
                    out[f"{oname}/{rule}/{B}/{alpha:g}"] = {
                        "order": oname, "rule": rule, "budget": int(B), "alpha": float(alpha), "draws": R,
                        "violation_rate": viol / R, "median_coverage": float(np.median(cov)),
                        "mean_coverage": float(np.mean(cov)), "share_certifying_nothing": nothing / R,
                        "refused_by_review_set_guard": refused,
                        "median_true_rate_of_certified_zones": float(np.median(rates)) if rates else float("nan"),
                        "best_coverage_possible": best_coverage(score, err, alpha),
                        "min_labels_to_certify": est.min_labels_to_certify(alpha, DELTA)}
    return out


# ----------------------------------------------------------------------------- the predictions
def _not_graded(reason, **kw):
    return {"holds": None, "graded": False, "reason": reason, **kw}


def grade_p1(rank, n_err, arm):
    A = ARMS[arm]
    cap = rank["signals"]["confidence"]["capture"][bkey(0.10)]
    if "P1" not in A["graded"]:
        return _not_graded("not graded on this arm", capture_10=cap)
    if n_err < ERROR_FLOOR:
        return _not_graded(f"{n_err} errors, below the floor of {ERROR_FLOOR}: reported only", capture_10=cap,
                           threshold=A["p1_capture"])
    return {"holds": bool(cap >= A["p1_capture"] - EPS), "graded": True, "capture_10": cap,
            "threshold": A["p1_capture"], "random": rank["signals"]["confidence"]["capture_random"][bkey(0.10)]}


def grade_p2(rank, n_err, arm):
    best = rank.get("best_control")
    au = rank["signals"][best]["auroc"] if best else float("nan")
    if "P2" not in ARMS[arm]["graded"]:
        return _not_graded("not graded on this arm", best_control=best, auroc=au)
    if best is None:
        return _not_graded("no control was computed")
    if n_err < ERROR_FLOOR:
        return _not_graded(f"{n_err} errors, below the floor of {ERROR_FLOOR}: reported only", best_control=best,
                           auroc=au, threshold=P2_AUROC)
    return {"holds": bool(au >= P2_AUROC - EPS), "graded": True, "best_control": best, "auroc": au,
            "threshold": P2_AUROC}


def grade_p3(rank, n_err, arm, p2):
    lead = rank.get("confidence_minus_best_control")
    if "P3" not in ARMS[arm]["graded"]:
        return _not_graded("not graded on this arm")
    if lead is None:
        return _not_graded("no control or no error")
    d = lead[bkey(0.10)]["difference"]
    lb = lead[bkey(0.10)]["bootstrap"]["lower_one_sided_95"]
    base = {"best_control": rank["best_control"], "capture_difference_10": d, "lower_one_sided_95": lb,
            "threshold_points": P3_POINTS}
    if n_err < ERROR_FLOOR:
        return _not_graded(f"{n_err} errors, below the floor of {ERROR_FLOOR}: reported only", **base)
    holds = bool(d >= P3_POINTS - EPS and np.isfinite(lb) and lb > 0)
    out = {"holds": holds, "graded": True, **base,
           "points_met": bool(d >= P3_POINTS - EPS), "bound_above_zero": bool(np.isfinite(lb) and lb > 0)}
    if p2.get("holds") is False:
        out["reading"] = "P2 failed: a comparison with a weak control, which shows little"
    return out


def grade_p4(cells, N, arm):
    if "P4" not in ARMS[arm]["graded"]:
        return _not_graded("not graded on this arm")
    gb = graded_budgets(N)
    if not gb:
        return _not_graded(f"no budget is graded: N = {N} (budgets are graded where B <= N/5 and N >= {MIN_N_GRADED})")
    cov = {B: cells[f"confidence/{B}"]["coverage"] for B in gb}
    ratio = cells["confidence/300"]["median_width"] / cells["random/300"]["median_width"]
    cov_ok = bool(all(v >= P4_COVER - EPS for v in cov.values()))
    width_ok = bool(ratio <= P4_WIDTH + EPS)
    return {"holds": cov_ok and width_ok, "graded": True, "graded_budgets": gb, "coverage": cov,
            "coverage_ok": cov_ok, "coverage_bar": P4_COVER, "width_ratio_300": ratio, "width_ok": width_ok,
            "width_bar": P4_WIDTH}


def grade_p5(cert, N, arm):
    A = ARMS[arm]
    if "P5" not in A["graded"]:
        return _not_graded("not graded on this arm")
    gb = graded_budgets(N)
    if not gb:
        return _not_graded(f"no budget is graded: N = {N}")
    B = 1000 if 1000 in gb else 300
    a = A["alpha"]
    c = cert[f"confidence/prefix/{B}/{a:g}"]
    k = cert[f"k3a/prefix/{B}/{a:g}"]
    lead = c["median_coverage"] - k["median_coverage"]
    valid = bool(c["violation_rate"] <= P5_VIOLATION + EPS)
    useful = bool(c["median_coverage"] >= A["p5_coverage"] - EPS)
    beats = bool(lead >= P5_LEAD - EPS)
    return {"holds": valid and useful and beats, "graded": True, "budget": B, "alpha": a,
            "violation_rate": c["violation_rate"], "valid": valid, "violation_bar": P5_VIOLATION,
            "median_coverage": c["median_coverage"], "useful": useful, "coverage_bar": A["p5_coverage"],
            "median_coverage_k3a": k["median_coverage"], "lead_over_k3a": lead, "beats_k3a": beats,
            "lead_bar": P5_LEAD}


def grade_units(units, arm, draws=DRAWS, n_boot=BOOT, certify_draws=None):
    """Everything after the logits: readings, accuracy, ranking, the estimate and certify studies, P1 to P5.
    `units`: label, logits, clusters, controls {name: suspicion}, k3_top_prob, optional clusters_coarse, stratum,
    pixel accuracy. numpy only."""
    A = ARMS[arm]
    label = np.asarray(units["label"])
    rd = readings(units["logits"], arm)
    err = (rd["pred"] != label).astype(np.float64)
    n_err = int(err.sum())
    clusters = np.asarray(units["clusters"])
    controls = {k: np.asarray(v, dtype=np.float64) for k, v in units["controls"].items()}
    res = {"arm": arm, "n_units": int(label.size), "n_errors": n_err, "draws": draws, "bootstrap_resamples": n_boot,
           "certify_draws": certify_draws or draws,
           "prediction_counts": {k: rd[k] for k in ("n_pred_untrained", "n_untrained_in_top2", "n_p1_saturated")}}
    res["accuracy"] = accuracy_block(label, rd["pred"], err, rd["p1"], clusters, A["n_out"], A["class_names"],
                                     units.get("pixel"))
    conf_u, margin_u = 1.0 - rd["p1"], -rd["margin"]
    if n_err == 0:
        res["ranking"] = {"note": "no error: nothing to rank"}
        rank = None
    else:
        rank = ranking(err, conf_u, margin_u, rd["entropy"], controls, clusters, n_boot,
                       units.get("clusters_coarse"))
        res["ranking"] = rank
    if units.get("stratum") is not None and rank is not None:
        res["by_stratum_reported"] = by_stratum(err, conf_u, controls.get(rank["best_control"]),
                                                np.asarray(units["stratum"]))
    N = int(label.size)
    cells, random_draws = estimate_study(err, rd["margin"], rd["p1"], arm, draws)
    alphas = (A["alpha"],) + tuple(A["alpha_reported"])
    orders = {"confidence": rd["p1"], "k3a": 1.0 - controls["k3a_no_encoder_uncertainty"]}
    if certify_draws is not None:
        random_draws = {B: v[:certify_draws] for B, v in random_draws.items()}
    res["certify"] = certify_study(err, orders, random_draws, alphas)
    # c*(alpha) from all labels, per order: reported on every arm, and the only certify number an arm whose population
    # is below both budgets has
    res["certify_best_coverage"] = {f"{o}/{a:g}": best_coverage(sc, err, a) for o, sc in orders.items() for a in alphas}
    if is_report_only(arm):
        res["estimate"] = {"budgets_run": [B for B in EST_BUDGETS if B <= N], "cells": cells}
        res["report_only"] = True
        res["reported"] = report_measures(rank, cells, res["certify"], res["certify_best_coverage"], N, n_err, arm)
        return res
    res["estimate"] = {"graded_budgets": graded_budgets(N), "cells": cells}
    if rank is None:
        res["prereg"] = {p: _not_graded("no error in the population") for p in ("P1", "P2", "P3", "P4", "P5")}
    else:
        p2 = grade_p2(rank, n_err, arm)
        res["prereg"] = {"P1": grade_p1(rank, n_err, arm), "P2": p2, "P3": grade_p3(rank, n_err, arm, p2),
                         "P4": grade_p4(cells, N, arm), "P5": grade_p5(res["certify"], N, arm)}
    res["prereg"]["complete"] = all(res["prereg"][p]["holds"] is not None for p in A["graded"])
    res["prereg"]["graded_on_this_arm"] = list(A["graded"])
    return res


def report_measures(rank, cells, cert, best_cov, N, n_err, arm):
    """The report-only arms (A and F): the numbers each prediction reads, with their intervals, and nothing else. No
    threshold is applied and nothing here says whether a prediction would hold: the page amended on 1 October 2026
    draws no verdict on these arms. Why: their validation sets are too small for the page's error floor and budget
    rule (stated in `why`, with the counts)."""
    A = ARMS[arm]
    out = {"why": (f"report-only by the owner's decision of 1 October 2026: Ai2's validation set is too small to "
                   f"grade ({N} units, {n_err} errors here; the page's error floor is {ERROR_FLOOR} errors, and a "
                   f"budget is graded only where it is at most N/5 with N at least {MIN_N_GRADED}). Every number "
                   "below is reported; none is graded"),
           "n_units": N, "n_errors": n_err}
    if rank is not None:
        sig = rank["signals"]
        out["review_order"] = {k: sig["confidence"][k] for k in ("auroc", "capture", "capture_random", "ceiling",
                                                                  "excess_aurc", "gap_closed")}
        best = rank.get("best_control")
        if best is not None:
            out["best_informative_control"] = {"name": best, "auroc": sig[best]["auroc"],
                                               "capture": sig[best]["capture"], "aurc": sig[best]["aurc"]}
        lead = rank.get("confidence_minus_best_control")
        if lead is not None:
            out["confidence_minus_best_control"] = lead
    else:
        out["review_order"] = {"note": "no error: nothing to rank"}
    est_out = {}
    for B in EST_BUDGETS:
        c, r = cells.get(f"confidence/{B}", {}), cells.get(f"random/{B}", {})
        if not (c.get("run") and r.get("run")):
            est_out[str(B)] = {"run": False, "reason": c.get("reason") or r.get("reason")}
            continue
        est_out[str(B)] = {"run": True, "theta": c["theta"], "confidence_coverage": c["coverage"],
                           "random_coverage": r["coverage"], "random_exact_coverage": r.get("exact_coverage"),
                           "confidence_median_width": c["median_width"], "random_median_width": r["median_width"],
                           "width_ratio": c["median_width"] / r["median_width"] if r["median_width"] else float("nan")}
    out["estimate"] = est_out
    cert_out = {}
    for key, c in cert.items():
        order, rule, B, a = key.split("/")
        if order != "confidence" or rule != "prefix":
            continue
        k = cert.get(f"k3a/prefix/{B}/{a}")
        cert_out[f"{B}/{a}"] = {"budget": int(B), "alpha": float(a), "violation_rate": c["violation_rate"],
                                "median_coverage": c["median_coverage"],
                                "share_certifying_nothing": c["share_certifying_nothing"],
                                "median_coverage_k3a": k["median_coverage"] if k else None,
                                "coverage_minus_k3a": c["median_coverage"] - k["median_coverage"] if k else None}
    out["certify"] = cert_out or {"run": False, "reason": f"no budget fits the {N} units"}
    out["certify_best_coverage"] = best_cov
    out["alphas"] = [A["alpha"], *A["alpha_reported"]]
    return out


def by_stratum(err, conf_u, ctl_u, stratum):
    out = {}
    for s in np.unique(stratum):
        m = stratum == s
        e = err[m]
        row = {"n": int(m.sum()), "n_errors": int(e.sum()), "error_rate": float(e.mean())}
        if 0 < e.sum() < e.size:
            row["confidence_auroc"] = float(metrics.weighted_auroc(conf_u[m], e, np.ones(e.size)))
            row["confidence_capture_10"] = float(metrics.capture_at_budget_expected(conf_u[m], e, (0.10,))[0.10])
            if ctl_u is not None:
                row["best_control_auroc"] = float(metrics.weighted_auroc(ctl_u[m], e, np.ones(e.size)))
        out[str(s)] = row
    return out


# ----------------------------------------------------------------------------- S1: the numpy smoke
# what the synthetic cases must grade; tests/test_exp89.py reads these
SMOKE_EXPECTED = {
    "beats": {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True},
    "control_wins": {"P1": True, "P2": True, "P3": False, "P4": True, "P5": False},
    "weak": {"P1": False, "P2": False, "P3": False, "P4": False, "P5": False},
}
# the report-only arms' cases: every measure is computed and nothing is graded, whatever the numbers say
SMOKE_REPORT_ONLY = ("awf_report", "fld_report")
# (arm, N, error rate, confidence errors' suspicion band, K3 errors' band)
SMOKE_CASES = {"beats": ("mangrove", 6000, 0.024, (0.93, 1.0), (0.50, 1.0)),
               "control_wins": ("mangrove", 6000, 0.024, (0.82, 1.0), (0.97, 1.0)),
               "weak": ("mangrove", 6000, 0.024, (0.0, 1.0), (0.05, 1.0)),
               "awf_report": ("awf", 400, 0.12, (0.90, 1.0), (0.50, 1.0)),
               "fld_report": ("fld", 120, 0.24, (0.80, 1.0), (0.50, 1.0))}
# what a pass/fail verdict looks like in an output: report-only arms must hold none of these keys or words
VERDICT_KEYS = ("holds", "pass", "passed", "verdict", "graded", "complete", "coverage_ok", "width_ok", "valid",
                "useful", "beats", "beats_k3a", "points_met", "bound_above_zero", "accuracy_ok", "ok",
                "graded_budgets", "graded_on_this_arm", "prereg")
VERDICT_WORDS = ("PASS", "FAIL", "holds", "passed", "failed")


def verdicts_in(obj, path=""):
    """Every place in a JSON-like object that reads as a pass/fail verdict: a key from VERDICT_KEYS, or a string value
    holding a word from VERDICT_WORDS. The report-only arms' outputs must give an empty list."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in VERDICT_KEYS:
                found.append(f"{path}/{k}")
            found += verdicts_in(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += verdicts_in(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if any(re.search(rf"\b{w}\b", obj) for w in VERDICT_WORDS):
            found.append(f"{path}={obj[:60]!r}")
    return found


def logits_from_p1(p1, pred, n_out, trained, delta=1.0):
    """Logits whose argmax is `pred` and whose top-1 probability over the trained channels is `p1`: the predicted
    channel at 0, the next trained channel at -g, the rest of the trained channels at -(g + delta), the untrained
    channels far below. g solves p1 = 1 / (1 + e^-g (1 + (m - 2) e^-delta)) for m trained channels."""
    m = len(trained)
    rest = 1.0 + (m - 2) * np.exp(-delta)
    g = np.log(rest * p1 / (1.0 - p1))
    L = np.full((p1.size, n_out), -60.0)
    for i in range(p1.size):
        others = [c for c in trained if c != pred[i]]
        L[i, pred[i]] = 0.0
        L[i, others[0]] = -g[i]
        for c in others[1:]:
            L[i, c] = -(g[i] + delta)
    return L


def synthetic_units(case, seed=SEED):
    """Units with known answers. Correct units' suspicions are uniform on (0, 1); the errors' suspicions are uniform on
    a band near the suspect end, one band for the model's confidence and one for K3. The top-1 probability falls with
    the suspicion steeply (1 - p1 = 0.002 + 0.4 s^8), so the least confident quintile is the one the Neyman rule
    feeds. K2, K4 and K5 are weak by construction."""
    arm, N, theta, band_c, band_k = SMOKE_CASES[case]
    A = ARMS[arm]
    rng = np.random.default_rng([seed, list(SMOKE_CASES).index(case)])
    trained = list(A["trained"])
    K = int(round(theta * N))
    is_err = np.zeros(N, bool)
    is_err[rng.choice(N, K, replace=False)] = True
    prior = np.full(len(trained), 1.0 / len(trained))
    pred = np.asarray(trained)[rng.choice(len(trained), N, p=prior)]
    label = pred.copy()
    for i in np.flatnonzero(is_err):
        label[i] = rng.choice([c for c in trained if c != pred[i]])
    s = rng.random(N)
    s[is_err] = rng.uniform(*band_c, K)
    p1 = 1.0 - (0.002 + 0.4 * s ** 8)
    logits = logits_from_p1(p1, pred, A["n_out"], trained)
    t = rng.random(N)
    t[is_err] = rng.uniform(*band_k, K)
    k3_top = 0.995 - 0.6 * t ** 3
    controls = {"k2_class_rarity": class_rarity(pred, rng.choice(trained, 5000, p=prior), trained)[0],
                "k3a_no_encoder_uncertainty": 1.0 - k3_top,
                "k3b_no_encoder_disagreement": (t > 0.97).astype(np.float64)}
    k4 = {"mangrove": ("k4_inundation_ambiguity", "k4_mndwi_near_zero", "k4_ndvi_month_std"),
          "fld": ("k4_ndvi_drop_weak", "k4_nbr_drop_weak")}.get(arm, ("k4_ndvi_temporal_std", "k4_ndvi_3x3_std"))
    for name in k4:
        controls[name] = rng.standard_normal(N) + 0.1 * is_err
    k5 = "k5_cloud_timesteps" if arm == "fld" else "k5_cloud_months"
    controls[k5] = rng.poisson(1.0 + 0.2 * is_err).astype(np.float64)
    n_cl = {"mangrove": 300, "fld": 25}.get(arm, 30)
    clusters = rng.integers(0, n_cl, N)
    return {"label": label, "logits": logits, "clusters": clusters.astype(str),
            "clusters_coarse": (clusters // 10).astype(str), "controls": controls,
            "stratum": np.asarray([str(v) for v in label])}


def planted_checks():
    """Every grade at and beside its threshold on planted numbers, the budget rules and the error floor."""
    def rank_with(cap_conf, cap_ctl, auroc, lb):
        sig = {"confidence": {"capture": {bkey(0.10): cap_conf}, "capture_random": {bkey(0.10): 0.10}},
               "k3a_no_encoder_uncertainty": {"auroc": auroc, "capture": {bkey(0.10): cap_ctl}}}
        return {"signals": sig, "best_control": "k3a_no_encoder_uncertainty",
                "confidence_minus_best_control": {bkey(0.10): {"difference": cap_conf - cap_ctl,
                                                               "bootstrap": {"lower_one_sided_95": lb}}}}

    def cells_with(cov300, cov1000, w_conf, w_rand):
        return {"confidence/300": {"coverage": cov300, "median_width": w_conf},
                "confidence/1000": {"coverage": cov1000, "median_width": w_conf},
                "random/300": {"coverage": 0.95, "median_width": w_rand}}

    def cert_with(B, viol, cov, cov_k, a=0.02):
        return {f"confidence/prefix/{B}/{a:g}": {"violation_rate": viol, "median_coverage": cov},
                f"k3a/prefix/{B}/{a:g}": {"median_coverage": cov_k}}

    c = {}
    c["P1_at_threshold"] = grade_p1(rank_with(0.40, 0.1, 0.7, 0.1), 40, "mangrove")["holds"] is True
    c["P1_below"] = grade_p1(rank_with(0.3999, 0.1, 0.7, 0.1), 40, "mangrove")["holds"] is False
    c["P1_nandi_threshold"] = grade_p1(rank_with(0.30, 0.1, 0.7, 0.1), 40, "nandi")["holds"] is True
    c["P1_floor_reported_only"] = grade_p1(rank_with(0.9, 0.1, 0.7, 0.1), 39, "mangrove")["holds"] is None
    c["P2_at_threshold"] = grade_p2(rank_with(0.5, 0.1, 0.65, 0.1), 40, "mangrove")["holds"] is True
    c["P2_below"] = grade_p2(rank_with(0.5, 0.1, 0.6499, 0.1), 40, "mangrove")["holds"] is False
    for arm in ("awf", "fld"):                                     # report-only: no grade, whatever the numbers
        c[f"none_graded_on_{arm}"] = (
            grade_p1(rank_with(0.9, 0.1, 0.7, 0.1), 400, arm)["holds"] is None
            and grade_p2(rank_with(0.9, 0.1, 0.9, 0.1), 400, arm)["holds"] is None
            and grade_p3(rank_with(0.9, 0.1, 0.9, 0.1), 400, arm, {"holds": True})["holds"] is None
            and grade_p4(cells_with(0.95, 0.95, 0.5, 1.0), 5000, arm)["holds"] is None
            and grade_p5(cert_with(1000, 0.0, 0.9, 0.1, a=ARMS[arm]["alpha"]), 5000, arm)["holds"] is None)
    p2 = {"holds": True}
    c["P3_at_threshold"] = grade_p3(rank_with(0.45, 0.40, 0.7, 0.001), 40, "mangrove", p2)["holds"] is True
    c["P3_points_short"] = grade_p3(rank_with(0.4499, 0.40, 0.7, 0.001), 40, "mangrove", p2)["holds"] is False
    c["P3_bound_at_zero"] = grade_p3(rank_with(0.60, 0.40, 0.7, 0.0), 40, "mangrove", p2)["holds"] is False
    c["P3_weak_control_reading"] = "weak control" in grade_p3(rank_with(0.6, 0.4, 0.6, 0.1), 40, "mangrove",
                                                              {"holds": False})["reading"]
    c["P4_at_thresholds"] = grade_p4(cells_with(0.93, 0.93, 0.85, 1.0), 5000, "mangrove")["holds"] is True
    c["P4_coverage_short"] = grade_p4(cells_with(0.9299, 0.95, 0.5, 1.0), 5000, "mangrove")["holds"] is False
    c["P4_coverage_short_at_1000"] = grade_p4(cells_with(0.95, 0.9299, 0.5, 1.0), 5000, "mangrove")["holds"] is False
    c["P4_1000_not_graded_below_5000"] = grade_p4(cells_with(0.95, 0.10, 0.5, 1.0), 4999, "mangrove")["holds"] is True
    c["P4_width_over"] = grade_p4(cells_with(0.95, 0.95, 0.8501, 1.0), 5000, "mangrove")["holds"] is False
    c["P4_none_below_1500"] = grade_p4(cells_with(0.95, 0.95, 0.5, 1.0), 1499, "mangrove")["holds"] is None
    c["P5_at_thresholds"] = grade_p5(cert_with(1000, 0.12, 0.50, 0.40), 5000, "mangrove")["holds"] is True
    c["P5_invalid"] = grade_p5(cert_with(1000, 0.1201, 0.9, 0.1), 5000, "mangrove")["holds"] is False
    c["P5_not_useful"] = grade_p5(cert_with(1000, 0.0, 0.4999, 0.1), 5000, "mangrove")["holds"] is False
    c["P5_lead_short"] = grade_p5(cert_with(1000, 0.0, 0.9, 0.8001), 5000, "mangrove")["holds"] is False
    c["P5_at_300_when_only_300_graded"] = grade_p5(cert_with(300, 0.0, 0.9, 0.1), 2000, "mangrove")["budget"] == 300
    c["P5_nandi_alpha_and_bar"] = grade_p5(cert_with(1000, 0.0, 0.30, 0.20, a=0.05), 5000, "nandi")["holds"] is True
    c["budgets"] = (graded_budgets(1499) == [] and graded_budgets(1500) == [300] and graded_budgets(4999) == [300]
                    and graded_budgets(5000) == [300, 1000])
    return c


def smoke(out_dir=None, draws=DRAWS, certify_draws=SMOKE_DRAWS, n_boot=SMOKE_BOOT):
    """S1: the synthetic cases through every measure and every grade, the planted grades, the freeze guard's reading."""
    out_dir = out_dir or OUT
    t0 = time.time()
    cases = {}
    for case in (*SMOKE_EXPECTED, *SMOKE_REPORT_ONLY):
        arm = SMOKE_CASES[case][0]
        cases[case] = grade_units(synthetic_units(case), arm, draws=draws, n_boot=n_boot, certify_draws=certify_draws)
    out = {"experiment": "exp89 smoke S1: synthetic units with known answers", "smoke": True, "draws": draws,
           "certify_draws": certify_draws, "bootstrap_resamples": n_boot, "cases": cases, "planted": planted_checks(),
           "prereg_status_reading": {
               "frozen": prereg_status("**Status: frozen on 2 October 2026, before any run.**"),
               "frozen_naming_the_draft": prereg_status("**Status: frozen on 2 October 2026; the draft of 1 October "
                                                        "is in the history.**"),
               "draft": prereg_status("**Status: DRAFT, not frozen.** Written 1 October 2026.")}}
    assert out["prereg_status_reading"] == {"frozen": "frozen", "frozen_naming_the_draft": "frozen", "draft": "draft"}, \
        out["prereg_status_reading"]
    bad = [k for k, v in out["planted"].items() if v is not True]
    if bad:
        raise AssertionError(f"planted grades wrong: {bad}")
    out["seconds"] = time.time() - t0
    dump(out, os.path.join(out_dir, "exp89_summary_smoke.json"))
    for case, want in SMOKE_EXPECTED.items():
        g = cases[case]["prereg"]
        got = {p: g[p]["holds"] for p in want}
        print(f"smoke case {case}: " + ", ".join(f"{p} {v}" for p, v in got.items()) +
              f" | errors {cases[case]['n_errors']} of {cases[case]['n_units']}", flush=True)
        if got != want:
            raise AssertionError(f"smoke case {case}: graded {got}, the synthetic design implies {want}")
    for case in SMOKE_REPORT_ONLY:
        res = cases[case]
        found = verdicts_in(res)
        if found or not res.get("report_only") or "reported" not in res:
            raise AssertionError(f"smoke case {case}: a report-only arm must carry no verdict, found {found}")
        print(f"smoke case {case}: report-only, nothing graded | errors {res['n_errors']} of {res['n_units']}",
              flush=True)
    print(f"smoke OK in {out['seconds']:.1f}s: every graded case grades as designed, planted grades hold at their "
          "thresholds, the report-only cases carry no verdict, the freeze guard reads the status line", flush=True)
    return out


# ----------------------------------------------------------------------------- the rslearn dataset on disk
def find_windows_root(data_dir):
    """The `windows` directory of an extracted rslearn dataset (at most three levels down)."""
    for depth in range(4):
        pattern = os.path.join(data_dir, *(["*"] * depth), "windows")
        import glob
        hits = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no rslearn windows directory under {data_dir}")


def list_windows(windows_root, group):
    """[{dir, name, group, split, meta}] for every window of the group, sorted by name."""
    gdir = os.path.join(windows_root, group)
    out = []
    for name in sorted(os.listdir(gdir)):
        wdir = os.path.join(gdir, name)
        mpath = os.path.join(wdir, "metadata.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            meta = json.load(f)
        out.append({"dir": wdir, "name": name, "group": group,
                    "split": (meta.get("options") or {}).get("split"), "meta": meta})
    return out


def window_grid(meta):
    """(H, W, crs, transform) of a window from rslearn's metadata: bounds in projection pixels."""
    from rasterio.transform import Affine
    x0, y0, x1, y1 = meta["bounds"]
    pr = meta["projection"]
    xr, yr = float(pr["x_resolution"]), float(pr["y_resolution"])
    return int(y1 - y0), int(x1 - x0), pr["crs"], Affine(xr, 0.0, x0 * xr, 0.0, yr, y0 * yr)


def window_lonlat(meta):
    """Longitude and latitude of the window's centre."""
    import rasterio.warp
    H, W, crs, tr = window_grid(meta)
    x, y = tr * (W / 2.0, H / 2.0)
    lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


def completed_groups(wdir, layer):
    """Sorted item-group indices of a layer whose directory holds rslearn's `completed` marker, as rslearn's
    load_all_item_groups discovers them: `layer` is group 0, `layer.k` group k."""
    ldir = os.path.join(wdir, "layers")
    if not os.path.isdir(ldir):
        return []
    out = []
    for d in os.listdir(ldir):
        if d == layer:
            k = 0
        elif d.startswith(layer + ".") and d[len(layer) + 1:].isdigit():
            k = int(d[len(layer) + 1:])
        else:
            continue
        if os.path.exists(os.path.join(ldir, d, "completed")):
            out.append(k)
    return sorted(out)


def read_band_sets(gdir, H, W, crs, transform):
    """{band: (H, W) float32} from every band-set directory of one item group, warped bilinearly onto the window's
    10 m grid where a band set is stored at another resolution (exp21's rule)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    bands = {}
    for bs in sorted(os.listdir(gdir)):
        path = os.path.join(gdir, bs, "geotiff.tif")
        if not os.path.exists(path):
            continue
        names = bs.split("_")
        with rasterio.open(path) as src:
            if (src.height, src.width) == (H, W):
                data = src.read()
            else:
                with WarpedVRT(src, crs=crs, transform=transform, width=W, height=H,
                               resampling=Resampling.bilinear) as vrt:
                    data = vrt.read()
        if data.shape[0] != len(names):
            raise RuntimeError(f"{path}: {data.shape[0]} bands for the band set {bs}")
        for i, n in enumerate(names):
            bands[n] = data[i].astype(np.float32)
    return bands


def read_stack(wdir, meta, layer="sentinel2"):
    """(H, W, 12, 12) raw stack in OlmoEarth band order and the number of completed item groups. Months beyond the
    completed groups stay 0 (absent), and are counted."""
    H, W, crs, tr = window_grid(meta)
    groups = completed_groups(wdir, layer)
    stack = np.zeros((H, W, N_MONTHS, len(OLMO_BANDS)), dtype=np.float32)
    for t, k in enumerate(groups[:N_MONTHS]):
        name = layer if k == 0 else f"{layer}.{k}"
        bands = read_band_sets(os.path.join(wdir, "layers", name), H, W, crs, tr)
        missing = [b for b in OLMO_BANDS if b not in bands]
        if missing:
            raise RuntimeError(f"{wdir} {name}: bands {missing} are missing")
        for j, b in enumerate(OLMO_BANDS):
            stack[:, :, t, j] = bands[b]
    return stack, len(groups)


def read_fld_stack(wdir, meta):
    """(H, W, 8, 12) raw stack of arm F's eight layers in the config's order (pre_sentinel2, .1 to .3, then
    post_sentinel2, .1 to .3), bands in OlmoEarth's order, warped onto the window's 10 m grid where a band set is
    stored at another resolution."""
    H, W, crs, tr = window_grid(meta)
    stack = np.zeros((H, W, len(FLD_LAYERS), len(OLMO_BANDS)), dtype=np.float32)
    for t, name in enumerate(FLD_LAYERS):
        bands = read_band_sets(os.path.join(wdir, "layers", name), H, W, crs, tr)
        missing = [b for b in OLMO_BANDS if b not in bands]
        if missing:
            raise RuntimeError(f"{wdir} {name}: bands {missing} are missing")
        for j, b in enumerate(OLMO_BANDS):
            stack[:, :, t, j] = bands[b]
    return stack


def fld_unit(wdir, meta, with_stack=True):
    """One arm F window by rslearn's rules (reading o): all eight image layers and the label layer completed, and a
    label among the ten classes; else dropped with the reason. With the stack: the 64-px centre crop the model sees
    (rslearn's Pad on the raw values) and the K3 features and K4/K5 indices read from it."""
    rec = {"label": None, "drop": None}
    done = fld_layers_completed(wdir)
    rec["n_layers_completed"] = len(done)
    lab, why, raw = fld_label(wdir)
    rec["new_label"] = raw
    if not done:
        rec["no_imagery"] = True
    if len(done) < len(FLD_LAYERS):
        rec["drop"] = f"{len(FLD_LAYERS) - len(done)} of the 8 image layers not completed"
    elif lab is None:
        rec["drop"] = why
    else:
        rec["label"] = lab
    if not with_stack or rec["drop"]:
        return rec
    stack = read_fld_stack(wdir, meta)
    crop = np.moveaxis(pad_center(np.moveaxis(stack, (0, 1), (-2, -1)), FLD_CROP, fill=0), (-2, -1), (0, 1))
    rec["input"] = np.ascontiguousarray(crop)                                   # (64, 64, 8, 12) raw
    rec["n_groups"] = len(FLD_LAYERS)
    rec["features"], rec["indices"] = fld_window_features(crop)
    return rec


def has_input(rec):
    """A unit record that carries the model's input (Mangrove and arm F: "input"; Nandi and AWF: "crops")."""
    return "input" in rec or "crops" in rec


def read_label(wdir, arm):
    import rasterio
    layer, band = ARMS[arm]["label_layer"]
    path = os.path.join(wdir, "layers", layer, band, "geotiff.tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def unit_of(arm, wdir, meta, with_stack=True):
    """One window read by the arm's rules: its label unit (or the reason it is dropped), its model input and the
    series the controls read. Mangrove: rslearn's centre Pad to 2 on the image and the label. Nandi: the 16-px crop
    at shifts 0 to 3 (shift 0 is the graded input) and the 3x3 NDVI block around the label pixel."""
    A = ARMS[arm]
    if arm == "fld":
        return fld_unit(wdir, meta, with_stack)
    lab = read_label(wdir, arm)
    rec = {"label": None, "drop": None}
    if lab is None:
        rec["drop"] = "no label raster"
        return rec
    if arm == "mangrove":
        lab2 = pad_center(lab, A["patch"], fill=0)
        rec["label"], rec["drop"] = mangrove_unit_label(lab2)
        rec["pixel_labels"] = lab2.ravel()
    else:
        got, why = single_pixel_label(lab, A["label_fill"], A["trained"])
        if got is None:
            rec["drop"] = why
        else:
            rec["rc"], rec["label"] = got[:2], got[2]
    if not with_stack:
        return rec
    stack, n_groups = read_stack(wdir, meta)
    rec["n_groups"] = n_groups
    if n_groups == 0:
        rec["drop"] = rec["drop"] or "no imagery"
        rec["no_imagery"] = True
        return rec
    if arm == "mangrove":
        s2 = pad_center(np.moveaxis(stack, (0, 1), (-2, -1)), A["patch"], fill=0)       # (T, B, 2, 2)
        rec["input"] = np.moveaxis(s2, (-2, -1), (0, 1))                                # (2, 2, T, B)
        rec["series"] = rec["input"].reshape(-1, N_MONTHS, len(OLMO_BANDS)).mean(0)     # the block mean
    elif rec["label"] is not None:
        H, W = stack.shape[:2]
        r, c = rec["rc"]
        rec["crops"], rec["locs"] = [], []
        for s in range(4):
            r0, c0, pr, pc = crop_at(H, W, r, c, A["crop"], s)
            rec["crops"].append(stack[r0:r0 + A["crop"], c0:c0 + A["crop"]])
            rec["locs"].append((pr, pc))
        rec["series"] = stack[r, c]
        rec["ndvi3x3"] = ndvi_block3(stack, r, c)
    return rec


def ndvi_block3(stack, r, c):
    """(3, 3, 12) exp21-NDVI around (r, c); pixels outside the window are NaN, so a block clipped at an edge is read
    over the pixels it has."""
    H, W = stack.shape[:2]
    out = np.full((3, 3, stack.shape[2]), np.nan)
    for i in range(3):
        for j in range(3):
            rr, cc = r - 1 + i, c - 1 + j
            if 0 <= rr < H and 0 <= cc < W:
                out[i, j] = exp21_ndvi(stack[rr, cc])
    return out


def cluster_ids(arm, rec_meta, name, lonlat):
    """The bootstrap's cluster and the reported coarser one."""
    if arm == "fld":
        lon, lat = lonlat
        return f"{int(np.floor(lon))}_{int(np.floor(lat))}", None
    if arm == "mangrove":
        lon, lat = lonlat
        return (f"{int(np.floor(lon / 0.1))}_{int(np.floor(lat / 0.1))}",
                f"{int(np.floor(lon))}_{int(np.floor(lat))}")
    if arm == "nandi":
        x0, y0 = rec_meta["bounds"][:2]
        cell = f"{rec_meta['projection']['crs']}_{int(np.floor(x0 / 128))}_{int(np.floor(y0 / 128))}"
        lon, lat = lonlat
        return cell, f"{int(np.floor(lon / 0.05))}_{int(np.floor(lat / 0.05))}"
    return name.split("_point_")[0], None


NANDI_POLYGON_KEYS = ("polygon_id", "source_polygon", "polygon", "source_id")


def merge_by_polygon(cells, options):
    """Nandi: cells that share a source polygon are merged (union-find), when the windows record a polygon id."""
    parent = {c: c for c in cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for c, o in zip(cells, options):
        pid = next((o[k] for k in NANDI_POLYGON_KEYS if k in (o or {})), None)
        if pid is None:
            continue
        if pid in first:
            parent[find(c)] = find(first[pid])
        else:
            first[pid] = c
    return [find(c) for c in cells]


# ----------------------------------------------------------------------------- the Hub, the checkpoint
def hub_file(repo, filename, revision, repo_type, sha=None, local_dir=None):
    """A pinned file from the Hub (into HF_HOME), checked against its pinned sha256."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type=repo_type)
    got = sha256(path)
    if sha and got != sha:
        raise RuntimeError(f"{repo}@{revision[:8]}/{filename}: sha256 {got}, pinned {sha}")
    return path, got


def arm_readable(arm):
    """(readable, detail): both the model and the dataset answer the Hub API without a token (public, as the page's
    "if Ai2 publishes" requires), and both revisions are pinned here."""
    A = ARMS[arm]
    detail = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for kind, repo in (("model", A["model"]), ("dataset", A["data"])):
            try:
                info = api.model_info(repo, token=False) if kind == "model" else api.dataset_info(repo, token=False)
                detail[kind] = {"repo": repo, "readable": True, "sha": info.sha}
            except Exception as ex:                                  # 401, 404, offline
                detail[kind] = {"repo": repo, "readable": False, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
    except ImportError as ex:
        return False, {"error": repr(ex)}
    pinned = bool(A["model_revision"] and A["data_revision"])
    detail["pinned"] = pinned
    ok = all(detail[k]["readable"] for k in ("model", "dataset")) and pinned
    return ok, detail


def nandi_status():
    ok, detail = arm_readable("nandi")
    return {"status": "readable and pinned" if ok else "not run", "reason": None if ok else ARMS["nandi"]["not_run_reason"],
            "checked_utc": utc_now(), "hub": detail}


def _stub_class(fullname):
    module, _, name = fullname.rpartition(".")

    def init(self, *args, **kwargs):
        self.__dict__["_stub_args"] = (args, kwargs)

    def setstate(self, state):
        self.__dict__["_stub_state"] = state
    return type(name, (), {"__module__": module, "__qualname__": name, "__init__": init, "__setstate__": setstate,
                           "__doc__": f"inert stand-in for {fullname}, never imported"})


def load_checkpoint(path):
    """(state_dict, how, top-level keys). torch.load with weights_only=True first. If the checkpoint pickles other
    globals (rslearn's, Lightning's, jsonargparse's), each is mapped to an inert stub class under its own name and
    the load is retried with weights_only=True, so nothing from those packages is imported or run; only tensors are
    kept. A global that names code execution (os, subprocess, builtins, ...) is refused."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        how = "weights_only"
    except Exception as first:                                     # an unsupported global
        unsafe = list(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        bad = [g for g in unsafe if g.startswith(REFUSED_PREFIXES)]
        if bad or not unsafe:
            raise RuntimeError(f"{path}: refused; it pickles {bad or 'no listed global'} ({first!r})") from None
        stubs = [(_stub_class(g), g) for g in unsafe]
        with torch.serialization.safe_globals(stubs):
            ck = torch.load(path, map_location="cpu", weights_only=True)
        how = f"weights_only with {len(stubs)} inert stub classes: {sorted(unsafe)}"
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if not isinstance(sd, dict):
        raise RuntimeError(f"{path}: no state_dict")
    return {k: v for k, v in sd.items() if hasattr(v, "shape")}, how, sorted(ck) if isinstance(ck, dict) else None


def checkpoint_inventory(sd, arm):
    A = ARMS[arm]
    prefixes = {p: sum(k.startswith(p) for k in sd) for p in ENCODER_PREFIXES}
    groups = {}
    for k in sd:
        g = ".".join(k.split(".")[:4])
        groups[g] = groups.get(g, 0) + 1
    return {"n_tensors": len(sd), "n_parameters": int(sum(int(np.prod(v.shape)) for v in sd.values())),
            "keys": {k: list(v.shape) for k, v in sd.items()}, "key_groups": groups,
            "encoder_prefix_counts": prefixes,
            "head_expected": {k: (list(sd[k].shape) if k in sd else None) for k in A["head_keys"]},
            "decoder_keys": {k: list(v.shape) for k, v in sd.items()
                             if k.startswith(("model.decoders.", "model.decoder."))}}


# ----------------------------------------------------------------------------- the replica (torch)
def encoder_skeleton(model_id):
    """OlmoEarth's model with random weights, built from its config.json at the pinned revision. The package's loader
    by model id reads the config at the Hub's current main; a changed config that keeps every parameter shape would
    still load strictly and change the model without a word."""
    from huggingface_hub import hf_hub_download
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_path
    cfg = hf_hub_download(ModelID[model_id].repo_id(), "config.json", revision=ENCODER_REVISIONS[model_id])
    return load_model_from_path(os.path.dirname(cfg), load_weights=False)


def make_pooling_decoder(in_channels, out_channels, num_conv_layers, num_fc_layers, conv_channels, fc_channels):
    """rslearn's PoolingDecoder (rslearn.models.pooling_decoder, 0.0.12) with its parameter names, so a checkpoint's
    model.decoder.0.* keys load into it strictly: 3x3 convolutions with ReLU, the amax over the map, fully connected
    layers with ReLU, the output layer."""
    import torch

    class PoolingDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            conv, prev = [], in_channels
            for _ in range(num_conv_layers):
                conv.append(torch.nn.Sequential(torch.nn.Conv2d(prev, conv_channels, 3, padding=1), torch.nn.ReLU()))
                prev = conv_channels
            self.conv_layers = torch.nn.Sequential(*conv)
            fc = []
            for _ in range(num_fc_layers):
                fc.append(torch.nn.Sequential(torch.nn.Linear(prev, fc_channels), torch.nn.ReLU()))
                prev = fc_channels
            self.fc_layers = torch.nn.Sequential(*fc)
            self.output_layer = torch.nn.Linear(prev, out_channels)

        def forward(self, x):                                     # (B, C, H', W') -> (B, out_channels)
            x = self.conv_layers(x)
            x = torch.amax(x, dim=(2, 3))
            return self.output_layer(self.fc_layers(x))
    return PoolingDecoder()


def load_pooling_decoder(sd, prefix, out_channels, in_channels):
    """The head under `prefix`, its shape read from its own tensors and checked, loaded strictly."""
    sub = {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    if "output_layer.weight" not in sub:
        raise RuntimeError(f"no pooling decoder under {prefix}; decoder keys: "
                           f"{[k for k in sd if k.startswith(('model.decoder.', 'model.decoders.'))]}")
    n_conv = len({k.split(".")[1] for k in sub if k.startswith("conv_layers.")})
    n_fc = len({k.split(".")[1] for k in sub if k.startswith("fc_layers.")})
    conv_ch = sub["conv_layers.0.0.weight"].shape[0] if n_conv else in_channels
    fc_ch = sub["fc_layers.0.0.weight"].shape[0] if n_fc else conv_ch
    first_in = sub["conv_layers.0.0.weight"].shape[1] if n_conv else sub["output_layer.weight"].shape[1]
    if first_in != in_channels:
        raise RuntimeError(f"the head reads {first_in} channels; the encoder gives {in_channels} (pre and post)")
    head = make_pooling_decoder(in_channels, out_channels, n_conv, n_fc, conv_ch, fc_ch)
    head.load_state_dict(sub, strict=True)
    return head, {"num_conv_layers": n_conv, "num_fc_layers": n_fc, "conv_channels": int(conv_ch),
                  "fc_channels": int(fc_ch), "in_channels": int(in_channels), "out_channels": int(out_channels),
                  "n_keys": len(sub)}


class Replica:
    """The fine-tuned model without rslearn: the encoder keys loaded strictly into olmoearth_pretrain's encoder
    (OlmoEarth v1-Base for the real checkpoints), tokens mean-pooled over timesteps and band sets, the arm's head.
    Mangrove: patch 2 on the 2x2 block, the pooling decoder's amax over a 1x1 map, Linear(D, 4). Nandi: patch 1 on
    the 16-px crop, a 1x1 conv to 11 channels read at the label pixel. AWF: exp21's logits_grid (patch 4, the 1x1
    conv on patch features, bilinear x4) at the label pixel, unchanged. Arm F: SimpleTimeSeries' two passes (the four
    pre layers, then the four post layers) at patch 4 on the 64-px crop, concatenated, then rslearn's PoolingDecoder.
    fp32, TF32 off."""

    def __init__(self, arm, ckpt_path, model_id="OLMOEARTH_V1_BASE", device=None, checkpoint_sha256=None):
        import torch
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.arm, self.A = arm, ARMS[arm]
        if tuple(Modality.SENTINEL2_L2A.band_order) != OLMO_BANDS:
            raise RuntimeError(f"the encoder's band order {Modality.SENTINEL2_L2A.band_order} is not {OLMO_BANDS}")
        # E89_DEVICE=cpu forces the CPU: on aicr's GPUs the encoder's band-index buffers stayed on the CPU while the
        # sample went to cuda (job 1151050, index_select device mismatch); the validation sets are small enough for CPU
        self.device = device or os.environ.get("E89_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_sha256 = checkpoint_sha256 or sha256(ckpt_path)
        self.model_id, self.encoder_revision = model_id, ENCODER_REVISIONS[model_id]
        sd, self.load_how, self.top_level = load_checkpoint(ckpt_path)
        model = encoder_skeleton(model_id)
        self.encoder_prefix, errors = None, {}
        for pre in ENCODER_PREFIXES:
            sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if not sub:
                continue
            try:
                model.encoder.load_state_dict(sub, strict=True)
                self.encoder_prefix, self.n_encoder_keys = pre, len(sub)
                break
            except RuntimeError as ex:
                errors[pre] = str(ex)[:300]
        if self.encoder_prefix is None:
            raise RuntimeError(f"no prefix loads the encoder strictly: {errors}")
        self.n_checkpoint_keys = len(sd)
        self.normalizer = Normalizer(Strategy.COMPUTED)                  # std multiplier 2, rslearn's default
        self.modality = Modality.SENTINEL2_L2A
        self.n_band_sets = len(Modality.SENTINEL2_L2A.band_sets)
        D = int(getattr(model.encoder, "embedding_size", 0) or 0)
        if self.A["head_kind"] == "pooling_decoder":
            if not D:
                raise RuntimeError("the encoder does not say its embedding size")
            head, self.head_info = load_pooling_decoder(sd, FLD_HEAD_PREFIX, self.A["n_out"], 2 * D)
            used = {k for k in sd if k.startswith(self.encoder_prefix) or k.startswith(FLD_HEAD_PREFIX)}
            self.unused_keys = sorted(set(sd) - used)
            self.head_shape = [self.A["n_out"], 2 * D]
            self.model = model.to(self.device).eval()
            self.head = head.to(self.device).eval()
            return
        wk, bk = self.A["head_keys"]
        if wk not in sd or bk not in sd:
            raise RuntimeError(f"head keys {wk}, {bk} not in the checkpoint; decoder keys: "
                               f"{[k for k in sd if k.startswith('model.decoders.')]}")
        w, b = sd[wk].float(), sd[bk].float()
        if w.ndim == 4:
            if w.shape[2:] != (1, 1):
                raise RuntimeError(f"{wk}: {tuple(w.shape)} is not a 1x1 convolution")
            w = w[:, :, 0, 0]
        D = D or int(w.shape[1])
        if tuple(w.shape) != (self.A["n_out"], D) or tuple(b.shape) != (self.A["n_out"],):
            raise RuntimeError(f"head {tuple(w.shape)} + {tuple(b.shape)}, expected ({self.A['n_out']}, {D})")
        self.head_shape = list(w.shape)
        self.head_info, self.unused_keys = None, None
        self.model = model.to(self.device).eval()
        self.w, self.b = w.to(self.device), b.to(self.device)

    def normalize(self, raw):
        return self.normalizer.normalize(self.modality, np.asarray(raw, dtype=np.float64))

    def sample(self, stacks, timestamps="rslearn"):
        """(B, S, S, T, 12) raw -> MaskedOlmoEarthSample. rslearn's legacy timestamps are day 1, month index, 2024;
        exp21's were day 15, month index, 2023."""
        import torch
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        x = self.normalize(stacks)
        B, S, S2, T, _ = x.shape
        day, year = (1, 2024) if timestamps == "rslearn" else (15, 2023)
        ts = torch.tensor([[day, m, year] for m in range(T)], device=self.device)[None].repeat(B, 1, 1)
        return MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=self.device),
            sentinel2_l2a_mask=(torch.ones((B, S, S2, T, self.n_band_sets), device=self.device)
                                * MaskValue.ONLINE_ENCODER.value),
            timestamps=ts)

    def logits(self, stacks, locs=None, timestamps="rslearn", precision="fp32"):
        """(B, n_out) logits at each unit. precision "bf16" runs arm F's encoder under bfloat16 autocast, as Ai2's
        validation did (the head stays fp32); it is reported by S2, never scored."""
        import contextlib
        import torch
        stacks = np.asarray(stacks, dtype=np.float32)
        with torch.no_grad():
            if self.arm == "fld":
                ctx = (torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16)
                       if precision == "bf16" else contextlib.nullcontext())
                maps = []
                for part in (slice(0, len(FLD_PRE)), slice(len(FLD_PRE), len(FLD_LAYERS))):
                    with ctx:
                        out = self.model.encoder(self.sample(stacks[..., part, :], timestamps), fast_pass=True,
                                                 patch_size=self.A["patch"])
                    tok = out["tokens_and_masks"].sentinel2_l2a.float().mean(dim=[3, 4])      # (B, H', W', D)
                    maps.append(tok.permute(0, 3, 1, 2))
                return self.head(torch.cat(maps, dim=1)).double().cpu().numpy()      # pre channels, then post
            if precision != "fp32":
                raise ValueError("only arm F reports a bf16 pass")
            if self.arm == "awf":
                import exp21_finetuned_awf as exp21
                from oe_inferencex import awf
                awf.CROP = stacks.shape[1]                                     # the mask size stacks_to_sample uses
                _, pix = exp21.logits_grid(self.model, self.w, self.b, list(stacks))
                return np.stack([pix[i][r, c] for i, (r, c) in enumerate(locs)]).astype(np.float64)
            out = self.model.encoder(self.sample(stacks, timestamps), fast_pass=True, patch_size=self.A["patch"])
            feat = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])       # (B, H', W', D)
            if self.arm == "mangrove":
                v = torch.amax(feat, dim=(1, 2))                                 # the pooling decoder on a 1x1 map
            else:
                v = torch.stack([feat[i, r, c] for i, (r, c) in enumerate(locs)])
            return (v @ self.w.T + self.b).double().cpu().numpy()

    def info(self):
        return {"arm": self.arm, "device": str(self.device), "checkpoint_sha256": self.checkpoint_sha256,
                "encoder_config": {"model_id": self.model_id, "revision": self.encoder_revision},
                "load": self.load_how, "top_level": self.top_level,
                "encoder_prefix": self.encoder_prefix, "n_encoder_keys": self.n_encoder_keys,
                "n_checkpoint_keys": self.n_checkpoint_keys, "head_shape": self.head_shape, "dtype": "float32",
                "tf32": False, "patch": self.A["patch"], "head": self.head_info,
                "unused_checkpoint_keys": self.unused_keys}


def batched_logits(rep, inputs, locs, T_groups, batch, timestamps="rslearn", precision="fp32"):
    """Logits for every unit, batched by the number of item groups so each batch has one T and runs with
    fast_pass (a window with fewer groups is read with its own T, as rslearn's masked pooling reads it)."""
    n_out = rep.A["n_out"]
    out = np.zeros((len(inputs), n_out))
    T_groups = np.asarray(T_groups)
    for T in np.unique(T_groups):
        idx = np.flatnonzero(T_groups == T)
        for i in range(0, idx.size, batch):
            j = idx[i:i + batch]
            st = np.stack([inputs[k][..., :T, :] for k in j])
            kw = {"precision": precision} if precision != "fp32" else {}
            out[j] = rep.logits(st, [locs[k] for k in j] if locs is not None else None, timestamps, **kw)
    return out


BATCH = {"mangrove": 1024, "nandi": 2, "awf": 32, "fld": 16}


# ----------------------------------------------------------------------------- reading an arm's windows
MARKER = ".exp89_extracted"
MANIFEST = ".exp89_manifest.json"


def manifest_missing(data_dir):
    """Files of the extraction's manifest that are gone or changed size (the scratch purge removes files by age), or
    None when the extraction has no manifest (one made before the manifest existed)."""
    path = os.path.join(data_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        files = json.load(f)
    bad = []
    for rel, size in files.items():
        full = os.path.join(data_dir, rel)
        try:
            if os.path.getsize(full) != size:
                bad.append(rel)
        except OSError:
            bad.append(rel)
    return bad


def resolve_data(arm, data_dir, download=True, tar=None, log=print):
    """The arm's extracted dataset under data_dir, extracted once from its pinned tar and checked against the
    extraction's manifest on every use. Arm M, N and A: the tar is fetched from the Hub (when `download`) or given
    with --tar, and checked against its pinned sha256. Arm F: the tar is downloaded by the job (exp/jobs/e89.sh) and
    given with --tar; verify_tar checks its size and MD5, and only the layers the run reads are extracted. An
    extraction with a file missing, or with no manifest, is extracted again from its tar; none of the old locations
    (data/awf) is read."""
    marker = os.path.join(data_dir, MARKER)
    if os.path.exists(marker):
        with open(marker) as f:
            info = json.load(f)
        missing = manifest_missing(data_dir)
        if missing == []:
            return find_windows_root(data_dir), info
        why = ("it has no manifest" if missing is None else
               f"{len(missing)} of its files are missing or changed, e.g. {missing[:3]}")
        log(f"{data_dir}: the extraction is extracted again: {why}")
        tar = tar or (info.get("tar") if info.get("tar") and os.path.exists(info["tar"]) else None)
        os.remove(marker)
    A = ARMS[arm]
    if arm == "fld":
        if not tar:
            raise FileNotFoundError(f"{data_dir} holds no complete extraction of arm F's dataset and no --tar was "
                                    "given: the job downloads it (exp/jobs/e89.sh, E89_MODE=inv)")
        info = extract_layers(tar, data_dir, FLD_KEEP_LAYERS, verify_tar(tar, arm, log=log), log=log)
        return find_windows_root(data_dir), info
    if tar:
        got = sha256(tar)
        if A["data_sha256"] and got != A["data_sha256"]:
            raise RuntimeError(f"--tar {tar}: sha256 {got}, pinned {A['data_sha256']} ({A['data']}@"
                               f"{A['data_revision'][:8]})")
        path = tar
    elif not download:
        raise FileNotFoundError(f"{data_dir} holds no complete extracted dataset and downloads are off")
    else:
        path, got = hub_file(A["data"], A["data_file"], A["data_revision"], "dataset", A["data_sha256"])
    info = extract_tar(path, data_dir, got)
    return find_windows_root(data_dir), info


def _safe_member(m):
    if m.name.startswith("/") or ".." in m.name.split("/"):
        raise RuntimeError(f"unsafe member in the tar: {m.name}")


def _write_extraction(data_dir, info, manifest):
    """The manifest first, then the marker: a marker always has its manifest. Every extracted file is first dated now:
    tarfile gives each file its member's own time, which for a dataset packed months ago is older than the scratch
    purge's 30 days, so an extraction was removed at the next sweep (exp100's first job, 1271993, found 113,134 of the
    AWF files gone a day after they were extracted again)."""
    for rel in manifest:
        os.utime(os.path.join(data_dir, rel), None)
    with open(os.path.join(data_dir, MANIFEST), "w") as f:
        json.dump(manifest, f)
    info = {**info, "n_files": len(manifest), "extracted_utc": utc_now()}
    with open(os.path.join(data_dir, MARKER), "w") as f:
        json.dump(info, f)
    return info


def extract_tar(path, data_dir, sha=None):
    """Extract a dataset tar under data_dir, refusing absolute or parent paths, and leave a manifest of its files
    and a marker."""
    os.makedirs(data_dir, exist_ok=True)
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        for m in members:
            _safe_member(m)
        # the data filter refuses links out of the directory and special files where Python has it (3.12, 3.11.4)
        tf.extractall(data_dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    manifest = {m.name: m.size for m in members if m.isfile()}
    return _write_extraction(data_dir, {"tar": path, "sha256": sha or sha256(path), "n_members": len(members)},
                             manifest)


def verify_tar(path, arm, log=print):
    """Arm F's tar against its pin: the size, then the MD5 and the SHA-256 in one pass. The size and the MD5 are pinned
    (Cloud Storage's metadata); the SHA-256 is recorded, and checked once ARMS["fld"]["data_sha256"] pins it."""
    A = ARMS[arm]
    size = os.path.getsize(path)
    if size != A["data_bytes"]:
        raise RuntimeError(f"{path}: {size} bytes, pinned {A['data_bytes']} (an incomplete download: the job's curl "
                           "resumes it)")
    md5, sha = hashlib.md5(), hashlib.sha256()
    t0 = time.time()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
            sha.update(chunk)
    got_md5, got_sha = md5.hexdigest(), sha.hexdigest()
    log(f"{path}: {size} bytes, md5 {got_md5}, sha256 {got_sha} ({time.time() - t0:.0f}s)")
    if got_md5 != A["data_md5"]:
        raise RuntimeError(f"{path}: md5 {got_md5}, pinned {A['data_md5']} (generation {A['data_generation']})")
    if A["data_sha256"] and got_sha != A["data_sha256"]:
        raise RuntimeError(f"{path}: sha256 {got_sha}, pinned {A['data_sha256']}")
    return {"tar": path, "url": A.get("data_url"), "generation": A.get("data_generation"), "bytes": size,
            "md5": got_md5, "md5_pinned": A["data_md5"], "sha256": got_sha, "sha256_pinned": A["data_sha256"]}


def member_layer(name):
    """The rslearn layer a tar member belongs to (.../windows/<group>/<window>/layers/<layer>/...), or None for a
    window's own files and anything outside the windows."""
    parts = name.split("/")
    if "windows" not in parts:
        return None
    j = parts.index("windows")
    if len(parts) > j + 4 and parts[j + 3] == "layers":
        return parts[j + 4]
    return None


def extract_layers(path, data_dir, keep_layers, info, log=print):
    """Stream a dataset tar once and write only the members the run reads: the layers in `keep_layers`, every
    window's own files (metadata.json, items.json) and every file outside the windows (the dataset's config). Every
    other layer is counted (members and bytes) and skipped. Links and special files are skipped and counted. Leaves a manifest and a marker."""
    os.makedirs(data_dir, exist_ok=True)
    filt = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    kept, skipped, other = {}, {}, {"links_or_special": 0}
    manifest = {}
    t0 = time.time()
    with tarfile.open(path, "r|*") as tf:
        for i, m in enumerate(tf):
            _safe_member(m)
            layer = member_layer(m.name)
            if layer is not None and layer not in keep_layers:
                e = skipped.setdefault(layer, {"members": 0, "bytes": 0})
                e["members"] += 1
                e["bytes"] += m.size
                continue
            if m.isdir():
                continue
            if not m.isfile():
                other["links_or_special"] += 1
                continue
            tf.extract(m, data_dir, **filt)
            manifest[m.name] = m.size
            e = kept.setdefault(layer or "window and dataset files", {"members": 0, "bytes": 0})
            e["members"] += 1
            e["bytes"] += m.size
            if (i + 1) % 200000 == 0:
                log(f"  {i + 1} members read, {len(manifest)} written, {time.time() - t0:.0f}s")
    return _write_extraction(data_dir, {**info, "kept_layers": list(keep_layers), "kept": kept,
                                        "skipped": skipped, **other}, manifest)


def arm_windows(arm, windows_root):
    """Every window of the arm's group with its split, sorted by name; every group for arm F (its population is read
    from each window's options.split). Arm A lists through oe_inferencex.awf.list_windows, given the extracted tar's
    group directory, so the module's ROOT (exp21's data/awf) is neither read nor changed."""
    if arm == "awf":
        from oe_inferencex import awf
        out = []
        for wdir, split, r, c, cat in awf.list_windows(os.path.join(windows_root, ARMS["awf"]["group"])):
            with open(os.path.join(wdir, "metadata.json")) as f:
                meta = json.load(f)
            out.append({"dir": wdir, "name": os.path.basename(wdir), "group": ARMS["awf"]["group"], "split": split,
                        "rc": (r, c), "label": cat, "meta": meta})
        return out
    if arm == "fld":
        out = []
        for g in sorted(d for d in os.listdir(windows_root) if os.path.isdir(os.path.join(windows_root, d))):
            out += list_windows(windows_root, g)
        return out
    return list_windows(windows_root, ARMS[arm]["group"])


def read_units(arm, windows, need="model", log=print):
    """Read windows into unit records. `need`: "label" (labels only), "model" (labels, input, series) or
    "features" (as "model", then arm F's crop is dropped once its K3 features and indices are read, so thousands of
    training windows fit in memory)."""
    out = []
    t0 = time.time()
    for i, w in enumerate(windows):
        if arm == "awf":
            from oe_inferencex import awf
            rec = {"label": w["label"], "drop": None, "rc": w["rc"]}
            if need == "model":
                stack = awf.load_window_full(w["dir"])
                r, c = w["rc"]
                rec["n_groups"] = N_MONTHS
                r0, c0, pr, pc = crop_at(63, 63, r, c, ARMS["awf"]["crop"], 0)
                rec["crops"] = [stack[r0:r0 + 16, c0:c0 + 16]]
                rec["locs"] = [(pr, pc)]
                rec["series"] = stack[r, c]
                rec["ndvi3x3"] = ndvi_block3(stack, r, c)
        else:
            rec = unit_of(arm, w["dir"], w["meta"], with_stack=(need in ("model", "features")))
            if need == "features":
                rec.pop("input", None)
        rec["name"], rec["split"], rec["meta"], rec["group"] = w["name"], w["split"], w["meta"], w.get("group")
        out.append(rec)
        if (i + 1) % 2000 == 0:
            log(f"  read {i + 1}/{len(windows)} windows, {time.time() - t0:.0f}s")
    return out


def flip_input(a, flip):
    """rslearn's Flip on one unit's (S, S, T, C) input: "h" reverses the columns, "v" the rows."""
    if flip in ("h", "hv"):
        a = a[:, ::-1]
    if flip in ("v", "hv"):
        a = a[::-1]
    return np.ascontiguousarray(a)


def model_inputs(arm, recs, shift=0, flip="none"):
    if flip != "none" and arm != "fld":
        raise ValueError("flips are read for arm F only (its validation used rslearn's Flip)")
    if arm in ("mangrove", "fld"):
        return [flip_input(r["input"], flip) if flip != "none" else r["input"] for r in recs], None
    return [r["crops"][shift] for r in recs], [r["locs"][shift] for r in recs]


def pixel_accuracy(recs, pred):
    """Mangrove: Ai2's pixel micro accuracy over every valid label pixel of every window, kept or not."""
    n = k = 0
    for r, p in zip(recs, pred):
        v = r["pixel_labels"][r["pixel_labels"] != 0]
        n += v.size
        k += int((v != p).sum())
    return {"n_pixels": int(n), "n_pixel_errors": int(k), "accuracy_pixel": 1.0 - k / n if n else float("nan")}


# ----------------------------------------------------------------------------- inventory (no model on any window)
def cmd_inventory(args):
    arm = args.arm
    out_dir = args.out_dir or OUT
    inv = {"experiment": "exp89 inventory: files, checkpoint keys, windows and labels per split; no model is run",
           "arm": arm, "utc": utc_now(), "prereg_status": prereg_status(), "commit": git_state()[0],
           "config_revision": CONFIG_REVISION, "report_only": is_report_only(arm),
           "nandi": nandi_status() if not args.offline else {"status": "not checked (offline)"}}
    if arm == "nandi" and inv["nandi"]["status"] != "readable and pinned" and not getattr(args, "synthetic", False):
        inv["status"] = "not run"
        dump(inv, os.path.join(out_dir, f"exp89_inventory_{arm}.json"))
        print(f"arm N not run: {ARMS['nandi']['not_run_reason']}", flush=True)
        return 0
    A = ARMS[arm]
    if waiting_reason(arm):
        inv["waiting"] = waiting_reason(arm)                  # the inventory may run; scoring a window may not
    data_dir = args.data or default_data_dir(arm)
    files = {}
    windows_root, extracted = resolve_data(arm, data_dir, download=not args.offline, tar=getattr(args, "tar", None))
    if not args.offline and arm != "fld":
        for fname, sha in [(A["data_file"], A["data_sha256"])] + list(A["extra_files"].items()):
            p, got = hub_file(A["data"], fname, A["data_revision"], "dataset", sha)
            files[fname] = {"path": p, "bytes": os.path.getsize(p), "sha256": got, "sha256_pinned": sha,
                            "matches": got == sha}
    inv["files"], inv["extracted"], inv["windows_root"] = files, extracted, windows_root
    tar_path = (extracted or {}).get("tar")
    if arm != "fld" and tar_path and os.path.exists(tar_path):     # arm F's 42 GB tar: its extraction counted it
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
        top = {}
        for n in names:
            top[n.split("/")[0]] = top.get(n.split("/")[0], 0) + 1
        inv["tar"] = {"n_members": len(names), "top_level": top, "first_members": names[:20]}
    ckpt = args.ckpt
    if ckpt is None and not args.offline:
        ckpt, got = hub_file(A["model"], A["model_file"], A["model_revision"], "model", A["model_sha256"])
        files[A["model_file"]] = {"path": ckpt, "bytes": os.path.getsize(ckpt), "sha256": got,
                                  "sha256_pinned": A["model_sha256"],
                                  "matches": (got == A["model_sha256"]) if A["model_sha256"] else None}
    if ckpt:
        if args.ckpt:                                   # a local file: listed, never scored, and its hash recorded
            got = sha256(ckpt)
            files["--ckpt"] = {"path": ckpt, "bytes": os.path.getsize(ckpt), "sha256": got,
                               "sha256_pinned": A["model_sha256"],
                               "matches": (got == A["model_sha256"]) if A["model_sha256"] else None}
        sd, how, top = load_checkpoint(ckpt)
        inv["checkpoint"] = {"load": how, "top_level": top, **checkpoint_inventory(sd, arm)}
    if "annotation_features.geojson" in files:
        inv["geojson"] = geojson_summary(files["annotation_features.geojson"]["path"])
    inv["dataset"] = dataset_summary(arm, windows_root)
    dump(inv, os.path.join(out_dir, f"exp89_inventory_{arm}.json"))
    d = inv["dataset"]
    keys = ("windows_per_split", "units_per_split", "item_groups_per_window", "window_px") + (
        ("val_units_per_group", "ai2_val_windows") if arm == "fld" else ())
    print(json.dumps(e78.jsonable({k: d.get(k) for k in keys}), indent=1))
    return 0


def geojson_summary(path):
    with open(path) as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    keys, geoms = {}, {}
    for ft in feats:
        for k in (ft.get("properties") or {}):
            keys[k] = keys.get(k, 0) + 1
        t = (ft.get("geometry") or {}).get("type")
        geoms[t] = geoms.get(t, 0) + 1
    return {"n_features": len(feats), "property_keys": keys, "geometry_types": geoms}


def _count(d, k, n=1):
    d[k] = d.get(k, 0) + n


def dataset_summary(arm, windows_root):
    """Windows per group and split, window sizes, CRSs, time ranges, layers, item groups, labels per split. Reads
    metadata and label rasters only, never imagery into a model."""
    A = ARMS[arm]
    groups = sorted(d for d in os.listdir(windows_root) if os.path.isdir(os.path.join(windows_root, d)))
    per_group = {}
    for g in groups:
        ws = list_windows(windows_root, g)
        sp = {}
        for w in ws:
            _count(sp, str(w["split"]))
        per_group[g] = {"n_windows": len(ws), "splits": sp}
    windows = arm_windows(arm, windows_root)
    out = {"groups": per_group, "group": A["group"], "windows_per_split": {}, "units_per_split": {},
           "dropped_per_split": {}, "unit_classes_per_split": {}, "pixel_classes_per_split": {}, "window_px": {},
           "crs": {}, "time_range": {"min_start": None, "max_end": None, "durations_days": {}}, "layers": {},
           "item_groups_per_window": {}, "items_json": 0, "band_sets": {}, "option_keys": {}}
    import datetime as dt
    for w in windows:
        sp = str(w["split"])
        _count(out["windows_per_split"], sp)
        meta = w["meta"]
        H, W = meta["bounds"][3] - meta["bounds"][1], meta["bounds"][2] - meta["bounds"][0]
        _count(out["window_px"], f"{H}x{W}")
        _count(out["crs"], meta["projection"]["crs"])
        for k in (meta.get("options") or {}):
            _count(out["option_keys"], k)
        tr = meta.get("time_range")
        if tr:
            a, b = tr
            out["time_range"]["min_start"] = min(filter(None, [out["time_range"]["min_start"], a]))
            out["time_range"]["max_end"] = max(filter(None, [out["time_range"]["max_end"], b]))
            try:
                days = (dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).days
                _count(out["time_range"]["durations_days"], str(days))
            except ValueError:
                pass
        ldir = os.path.join(w["dir"], "layers")
        if os.path.isdir(ldir):
            for d in os.listdir(ldir):
                base = d.split(".")[0]
                e = out["layers"].setdefault(base, {"dirs": 0, "completed": 0})
                e["dirs"] += 1
                e["completed"] += os.path.exists(os.path.join(ldir, d, "completed"))
                if base in ("sentinel2", "pre_sentinel2", "post_sentinel2") and "." not in d:
                    for bs in os.listdir(os.path.join(ldir, d)):
                        if os.path.isdir(os.path.join(ldir, d, bs)):
                            _count(out["band_sets"], bs)
        if arm == "fld":                                  # of the eight layers the model reads, how many completed
            _count(out["item_groups_per_window"], str(len(fld_layers_completed(w["dir"]))))
        else:
            _count(out["item_groups_per_window"], str(len(completed_groups(w["dir"], "sentinel2"))))
        out["items_json"] += os.path.exists(os.path.join(w["dir"], "items.json"))
        if arm == "awf":
            rec = {"label": w["label"], "drop": None}
        else:
            rec = unit_of(arm, w["dir"], meta, with_stack=False)
        if rec["label"] is None:
            _count(out["dropped_per_split"].setdefault(sp, {}), rec["drop"])
        else:
            _count(out["units_per_split"], sp)
            _count(out["unit_classes_per_split"].setdefault(sp, {}), str(rec["label"]))
        if arm == "fld":
            fld_window_counts(out, w, sp, rec)
        if "pixel_labels" in rec:
            for v, n in zip(*np.unique(rec["pixel_labels"], return_counts=True)):
                _count(out["pixel_classes_per_split"].setdefault(sp, {}), str(int(v)), int(n))
    return out


def fld_window_counts(out, w, sp, rec):
    """Arm F's inventory, per window (reported, never a filter): the units per group in Ai2's validation split (the
    decisive check: the sum should be Ai2's 109 if the tar's splits are those of training), the raw new_label values,
    the paper's olmoearth_evals_split beside options.split, and windows whose label.json names another label than
    data.geojson (label.json is never read for the label)."""
    g = str(w.get("group"))
    if sp == "val" and rec["label"] is not None:
        _count(out.setdefault("val_units_per_group", {}), g)
    _count(out.setdefault("new_label_values_per_split", {}).setdefault(sp, {}), str(rec.get("new_label")))
    opts = w["meta"].get("options") or {}
    if "olmoearth_evals_split" in opts:
        _count(out.setdefault("olmoearth_evals_split_beside_split", {}),
               f"split={sp} evals={opts['olmoearth_evals_split']}")
    lj = os.path.join(w["dir"], "label.json")
    if os.path.exists(lj):
        out["label_json"] = out.get("label_json", 0) + 1
        try:
            with open(lj) as f:
                d = json.load(f)
            v = d.get("new_label", d.get("label")) if isinstance(d, dict) else None
        except (OSError, ValueError):
            v = "unreadable"
        if v is not None and v != rec.get("new_label"):
            out["label_json_differs"] = out.get("label_json_differs", 0) + 1
    out["ai2_val_windows"] = FLD_AI2_VAL_WINDOWS
    out["ai2_val_groups"] = list(FLD_VAL_GROUPS)


def default_data_dir(arm):
    """E89_DATA/<arm> (the job sets E89_DATA on scratch), else data/exp89_<arm>. Never data/awf, exp21's location."""
    base = os.environ.get("E89_DATA")
    if base:
        return os.path.join(base, arm)
    return os.path.join(ROOT, "data", f"exp89_{arm}")


# ----------------------------------------------------------------------------- S2 on real training windows
def model_smoke(arm, rep, windows, out_path, n=S2_WINDOWS, log=print):
    """S2 on TRAINING windows only: strict load (done by Replica), one batch's shape, in-sample accuracy on n windows
    drawn with seed 89, the untrained channel never the argmax, the largest logit difference between the two
    timestamp conventions."""
    train = [w for w in windows if w["split"] == "train"]
    recs = read_units(arm, train, need="label", log=log)
    usable = [w for w, r in zip(train, recs) if r["label"] is not None]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(usable), min(n, len(usable)), replace=False).tolist())
    chosen = [usable[i] for i in pick]
    assert all(w["split"] == "train" for w in chosen), "S2 reads training windows only"
    recs = [r for r in read_units(arm, chosen, need="model", log=log) if has_input(r) and r["label"] is not None]
    inputs, locs = model_inputs(arm, recs)
    T = [r["n_groups"] for r in recs]
    lg = batched_logits(rep, inputs, locs, T, BATCH[arm])
    k = 2 if len(T) > 1 and T[0] == T[1] else 1                         # one batch of one T
    first = rep.logits(np.stack(inputs[:k])[..., :T[0], :], locs[:k] if locs else None)
    label = np.array([r["label"] for r in recs])
    pred = lg.argmax(1)
    acc = float((pred == label).mean()) if label.size else float("nan")
    res = {"experiment": "exp89 S2: the real checkpoint on training windows (in-sample, never graded)", "arm": arm,
           "utc": utc_now(), "commit": git_state()[0], "replica": rep.info(), "n_windows": len(recs),
           "one_batch_output_shape": list(first.shape), "expected_columns": ARMS[arm]["n_out"],
           "accuracy_in_sample": acc, "accuracy_bar": S2_MIN_ACCURACY if arm == "mangrove" else None,
           "accuracy_ok": bool(acc >= S2_MIN_ACCURACY) if arm == "mangrove" else None,
           "n_pred_untrained": int((~np.isin(pred, ARMS[arm]["trained"])).sum()),
           "n_groups": {str(t): int(c) for t, c in zip(*np.unique(T, return_counts=True))} if T else {}}
    if arm != "awf":
        lg15 = batched_logits(rep, inputs, locs, T, BATCH[arm], timestamps="exp21")
        res["timestamp_conventions"] = {"rslearn": "day 1, month index, 2024", "exp21": "day 15, month index, 2023",
                                        "max_abs_logit_difference": float(np.abs(lg - lg15).max()) if lg.size else None,
                                        "n_predictions_changed": int((lg15.argmax(1) != pred).sum())}
    if arm == "fld":
        # Ai2's validation ran the encoder under bf16 autocast; the replica is fp32. Reported on training windows only.
        lgb = batched_logits(rep, inputs, locs, T, BATCH[arm], precision="bf16")
        res["bf16_autocast"] = {"max_abs_logit_difference": float(np.abs(lg - lgb).max()) if lg.size else None,
                                "n_predictions_changed": int((lgb.argmax(1) != pred).sum()),
                                "accuracy_in_sample_bf16": (float((lgb.argmax(1) == label).mean()) if label.size
                                                            else None)}
        res["k3_features"] = {"n": len(FLD_FEATURE_NAMES), "finite": bool(all(np.isfinite(r["features"]).all()
                                                                              for r in recs))}
    res["ok"] = bool(res["one_batch_output_shape"][-1] == ARMS[arm]["n_out"] and res["n_pred_untrained"] == 0
                     and (res["accuracy_ok"] is not False))
    dump(res, out_path)
    log(f"S2 {arm}: {len(recs)} training windows, in-sample accuracy {acc:.4f}, untrained argmax "
        f"{res['n_pred_untrained']}, ok {res['ok']}")
    return res


# ----------------------------------------------------------------------------- the gate (accuracy only)
def gate_record(arm, out_dir):
    path = os.path.join(out_dir, f"exp89_gate_{arm}.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f), path


def gate_passed(arm, out_dir=None):
    """(whether the last attempt passed, or for a report-only arm whether the replica is aligned; the record)."""
    rec, _ = gate_record(arm, out_dir or OUT)
    return bool(rec and rec.get(gate_keys(arm)[1]) is True), rec


def gate_keys(arm):
    """(the attempt's key, the record's key): ("pass", "passed") for a graded arm, ("aligned", "aligned") for a
    report-only arm, whose gate is an alignment check and never a pass or a fail."""
    return ("aligned", "aligned") if is_report_only(arm) else ("pass", "passed")


def alignment_label(attempts):
    """A report-only arm's label for its numbers: "aligned" when the last check is within the tolerance, else
    "replica not aligned" (also when no check is recorded)."""
    return "aligned" if attempts and attempts[-1].get("aligned") is True else NOT_ALIGNED


def ledger_path(arm, out_dir):
    """The gate's append-only ledger, one JSON line per attempt. In E89_GATE_LEDGER when set (the cluster job sets it
    outside the git checkout, so a hard reset cannot touch it), else beside the gate file."""
    return os.path.join(os.environ.get(GATE_LEDGER_ENV) or out_dir, f"exp89_gate_{arm}.ledger.jsonl")


def read_ledger(arm, out_dir):
    path = ledger_path(arm, out_dir)
    if not os.path.exists(path):
        return [], path
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()], path


def append_ledger(arm, out_dir, attempt):
    path = ledger_path(arm, out_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(e78.jsonable(attempt), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def _canon(attempts):
    return [json.dumps(e78.jsonable(a), sort_keys=True) for a in attempts]


def reconcile_gate(arm, out_dir, rec):
    """The gate file's attempts against the ledger's. The ledger is the count: a file behind it (a reset restored an
    older committed file) is restored from it; a file holding an attempt the ledger does not hold, in order, was
    edited or written elsewhere, and is refused. Returns (the ledger's attempts, whether the file was behind)."""
    ledger, lpath = read_ledger(arm, out_dir)
    held = (rec or {}).get("attempts") or []
    if _canon(held) != _canon(ledger)[:len(held)]:
        raise GateRefused(f"exp89_gate_{arm}.json holds {len(held)} attempts that the gate's ledger ({lpath}, "
                          f"{len(ledger)} attempts) does not hold: the file was edited or written elsewhere. Nothing "
                          "is scored or graded until the two agree.")
    return ledger, len(ledger) > len(held)


def counted_attempts(arm, attempts):
    """The attempts that count: all but a synthetic smoke's (made on a draft page and another checkpoint than the pinned
    one). A synthetic smoke once wrote its gate attempt
    into the real ledger (job 1151248's smoke, a synthetic checkpoint on a draft page, found by job 1151373); such an
    attempt stays in the ledger as history but neither counts towards the limit nor blocks the run."""
    pinned = ARMS[arm]["model_sha256"]
    # dropped only when both hold: made before freezing (so it scored no validation window, which predict_population
    # refuses before freezing) and on another checkpoint than the pinned one (so it was the synthetic smoke's)
    return [a for a in attempts
            if not (pinned and a.get("prereg_status") != "frozen" and a.get("checkpoint_sha256") != pinned)]


def require_gate(arm, out_dir, ledger_required=True):
    """The run's own check of the gate, beside cmd_run's: a pass on the last attempt (a graded arm) or a recorded
    alignment check, aligned or not (a report-only arm), within the attempt limit, every attempt made on a frozen
    page, the last attempt on the pinned checkpoint, and (when the run scores windows) the same attempts in the
    ledger. Returns the gate record or raises GateRefused."""
    ro = is_report_only(arm)
    what = "alignment check" if ro else "gate"
    rec, _ = gate_record(arm, out_dir)
    if rec is None:
        raise GateRefused(f"no {what} record")
    full = rec.get("attempts") or []
    att = counted_attempts(arm, full)
    if ro:
        if not att:
            raise GateRefused("no alignment check is recorded")
    else:
        if rec.get("closed"):
            raise GateRefused("the gate is closed")
        if rec.get("passed") is not True or not att or att[-1].get("pass") is not True:
            raise GateRefused("the gate has not passed")
    if len(att) > MAX_GATE_ATTEMPTS:
        raise GateRefused(f"the {what} record holds {len(att)} attempts, more than the {MAX_GATE_ATTEMPTS} allowed")
    if any(a.get("prereg_status") != "frozen" for a in att):
        raise GateRefused("an attempt was made while the page was not frozen")
    pinned = ARMS[arm]["model_sha256"]
    if pinned and att[-1].get("checkpoint_sha256") != pinned:
        raise GateRefused(f"the last attempt scored {att[-1].get('checkpoint_sha256')}, not the pinned checkpoint "
                          f"{pinned}")
    if ledger_required:
        ledger, lpath = read_ledger(arm, out_dir)
        if _canon(ledger) != _canon(full):
            raise GateRefused(f"the gate's ledger ({lpath}) does not hold the attempts of exp89_gate_{arm}.json")
    return rec


def predict_population(arm, rep, windows, log=print, synthetic=False, flip="none"):
    """Every validation window with imagery: (records, logits). Used by the gate and the run, after freezing. The only
    function that scores a validation window; it refuses before the page is frozen, except on the synthetic smoke's
    windows. `flip` (arm F only) applies rslearn's Flip to every input, for the gate's reported accuracy by flip."""
    if not synthetic and prereg_status() != "frozen":
        raise RuntimeError(f"{os.path.relpath(PLAN, ROOT)} is not frozen: no validation window is scored")
    if not synthetic and waiting_reason(arm):
        raise RuntimeError(f"arm {ARM_LETTER[arm]} not run: {waiting_reason(arm)}")
    val = [w for w in windows if w["split"] == "val"]
    recs = read_units(arm, val, need="model", log=log)
    have = [r for r in recs if "series" in r or has_input(r)]
    inputs, locs = model_inputs(arm, have, flip=flip)
    lg = batched_logits(rep, inputs, locs, [r["n_groups"] for r in have], BATCH[arm])
    return recs, have, lg


def run_gate(arm, rep, windows, out_dir, imagery_route="tar", log=print, synthetic=False):
    """G: predictions on the validation windows; only the window counts, the error counts and the accuracy are kept.
    Each attempt goes to the ledger first, then to the gate file (see reconcile_gate). For a report-only arm the same
    record is an alignment check: "aligned" where a graded arm has "pass", and "alignment" says "aligned" or
    "replica not aligned"; it never says pass or fail."""
    A = ARMS[arm]
    ro = is_report_only(arm)
    akey, rkey = gate_keys(arm)
    rec, path = gate_record(arm, out_dir)
    attempts, behind = reconcile_gate(arm, out_dir, rec)
    rec = rec or {"arm": arm, "ai2_accuracy": A["ai2_accuracy"],
                  "metric": ("pixel micro accuracy over valid label pixels" if arm == "mangrove"
                             else "accuracy per window" + ("" if arm == "fld" else " (one label pixel)")),
                  "tolerance_points": A["gate_tolerance"][imagery_route] * 100, "imagery_route": imagery_route,
                  "attempts": [], rkey: False, "closed": False, "max_attempts": MAX_GATE_ATTEMPTS}
    rec["attempts"] = attempts
    counted = counted_attempts(arm, attempts)
    rec[rkey] = bool(counted and counted[-1][akey] is True)
    rec["closed"] = (not rec[rkey]) and len(counted) >= MAX_GATE_ATTEMPTS
    rec["prereg_status"] = prereg_status()
    if ro:
        rec["report_only"], rec["alignment"] = True, alignment_label(attempts)
    word = (lambda ok: "aligned" if ok else NOT_ALIGNED) if ro else (lambda ok: "PASS" if ok else "FAIL")
    if behind:
        dump(rec, path)
        log(f"gate {arm}: the gate file was behind its ledger; restored to the ledger's {len(attempts)} attempts")
    if rec[rkey]:
        log(f"gate {arm}: {'aligned' if ro else 'already passed'} on attempt {rec['attempts'][-1]['attempt']}; "
            "not rerun")
        return rec
    if rec["closed"]:
        dump(rec, path)
        log(f"gate {arm}: closed after {len(rec['attempts'])} attempts outside the tolerance; " +
            ("every number of the run is labelled 'replica not aligned'" if ro else "nothing is graded"))
        return rec
    recs, have, lg = predict_population(arm, rep, windows, log, synthetic=synthetic)
    pred = lg.argmax(1)
    kept = [i for i, r in enumerate(have) if r["label"] is not None]
    n_blind = sum(1 for r in recs if r.get("no_imagery"))
    if not kept:
        raise GateRefused(f"gate {arm}: none of the {len(recs)} validation windows has both imagery and a kept label "
                          f"({n_blind} have no imagery). Nothing to gate on; no attempt is recorded.")
    label = np.array([have[i]["label"] for i in kept])
    n_err = int((pred[kept] != label).sum())
    acc_w = 1.0 - n_err / len(kept)
    by_flip = None
    if arm == "fld":                                 # Ai2's validation flipped at random; reported beside, not gated
        by_flip = {"none": acc_w}
        for f in FLIPS[1:]:
            pf = predict_population(arm, rep, windows, log, synthetic=synthetic, flip=f)[2].argmax(1)
            by_flip[f] = float((pf[kept] == label).mean())
    commit, dirty = git_state()
    att = {"attempt": len(rec["attempts"]) + 1, "utc": utc_now(), "commit": commit, "dirty": dirty,
           "prereg_status": prereg_status(), "checkpoint_sha256": getattr(rep, "checkpoint_sha256", None),
           "n_windows": len(kept), "n_errors": n_err, "accuracy": acc_w,
           "n_windows_dropped": len(recs) - len(kept), "n_windows_no_imagery": n_blind,
           "n_pixels": None, "n_pixel_errors": None, "accuracy_pixel": None, "accuracy_by_flip": by_flip}
    if arm == "mangrove":
        att.update(pixel_accuracy(have, pred))
        gated = att["accuracy_pixel"]
    else:
        gated = acc_w
    att["gap_points"] = (gated - A["ai2_accuracy"]) * 100
    att[akey] = bool(abs(gated - A["ai2_accuracy"]) <= A["gate_tolerance"][imagery_route] + EPS)
    att = json.loads(json.dumps(e78.jsonable(att)))                 # the file's and the ledger's form, identical
    append_ledger(arm, out_dir, att)                                # the count first, so a crash cannot lose it
    rec["attempts"].append(att)
    rec[rkey] = att[akey]
    rec["closed"] = (not att[akey]) and len(counted_attempts(arm, rec["attempts"])) >= MAX_GATE_ATTEMPTS
    if ro:
        rec["alignment"] = alignment_label(rec["attempts"])
    dump(rec, path)
    log(f"gate {arm} attempt {att['attempt']}: accuracy {gated:.4f} against Ai2's {A['ai2_accuracy']:.3f} "
        f"(gap {att['gap_points']:+.2f} points, tolerance {rec['tolerance_points']:.1f}): {word(att[akey])}")
    return rec


def cmd_gate(args):
    status = prereg_status()
    if status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} says the preregistration is {status}. The gate scores "
              "validation windows and runs only after freezing.", flush=True)
        return 2
    if not_run_message(args.arm):
        print(not_run_message(args.arm), flush=True)
        return 0
    out_dir = args.out_dir or OUT
    akey, rkey = gate_keys(args.arm)
    try:
        # the ledger is read before any download or model load: a passed or closed gate needs no model, a file that
        # disagrees with its ledger is refused
        attempts, _ = reconcile_gate(args.arm, out_dir, gate_record(args.arm, out_dir)[0])
        counted = counted_attempts(args.arm, attempts)
        if (counted and counted[-1][akey] is True) or len(counted) >= MAX_GATE_ATTEMPTS:
            rec = run_gate(args.arm, None, [], out_dir)
        else:
            rep, windows = build_arm(args)
            rec = run_gate(args.arm, rep, windows, out_dir)
    except GateRefused as ex:
        print(f"refused: {ex}", flush=True)
        return 2
    if is_report_only(args.arm):
        return 0                                    # an alignment check is reported, whichever way it reads
    return 0 if rec["passed"] else 1


def not_run_message(arm):
    """The line every scoring mode prints for an arm that cannot run: N while the Hub answers 401, M until Ai2's
    validation split is pinned. None for an arm that can run."""
    if arm == "nandi":
        st = nandi_status()
        if st["status"] != "readable and pinned":
            return f"arm N not run: {st['reason']}"
    if waiting_reason(arm):
        return f"arm M not run: {waiting_reason(arm)}"
    return None


def checkpoint_path(args):
    """(path, sha256) of the arm's checkpoint: the pinned download, or a local --ckpt that must hash to the pin."""
    A = ARMS[args.arm]
    if not args.ckpt:
        return hub_file(A["model"], A["model_file"], A["model_revision"], "model", A["model_sha256"])
    got = sha256(args.ckpt)
    if A["model_sha256"] and got != A["model_sha256"]:
        raise RuntimeError(f"--ckpt {args.ckpt}: sha256 {got}, pinned {A['model_sha256']} "
                           f"({A['model']}@{A['model_revision'][:8]}); only the pinned checkpoint is scored")
    return args.ckpt, got


def build_arm(args, model_id="OLMOEARTH_V1_BASE"):
    ckpt, sha = checkpoint_path(args)                       # refused off the pin before any data is read
    windows_root, _ = resolve_data(args.arm, args.data or default_data_dir(args.arm), download=not args.offline,
                                   tar=getattr(args, "tar", None))
    rep = Replica(args.arm, ckpt, model_id=model_id, checkpoint_sha256=sha)
    return rep, arm_windows(args.arm, windows_root)


# ----------------------------------------------------------------------------- the full run
def compute_units(arm, rep, windows, log=print, k3_cap=None, synthetic=False, out_dir=None):
    """Logits, controls and K3 on the validation windows; K2's frequencies and K3's fit on the training windows.
    Refused without a verified gate pass (a graded arm) or a recorded alignment check (a report-only arm), except on
    the synthetic smoke's windows."""
    A = ARMS[arm]
    if not synthetic:
        require_gate(arm, out_dir or OUT)
    recs, have, lg = predict_population(arm, rep, windows, log, synthetic=synthetic)
    rd_pred = lg.argmax(1)
    keep = np.array([r["label"] is not None for r in have])
    units = [r for r, k in zip(have, keep) if k]
    logits = lg[keep]
    pred = rd_pred[keep]
    label = np.array([r["label"] for r in units])
    # the training windows: labels for K2, imagery for K3
    train = [w for w in windows if w["split"] == "train"]
    tr_lab = read_units(arm, train, need="label", log=log)
    tr_ok = [w for w, r in zip(train, tr_lab) if r["label"] is not None]
    tr_labels = np.array([r["label"] for r in tr_lab if r["label"] is not None])
    k2, freq = class_rarity(pred, tr_labels, A["trained"])
    cap = k3_cap if k3_cap is not None else A["k3_cap"]
    rng = np.random.default_rng(SEED)
    if cap is not None and len(tr_ok) > cap:
        pick = np.sort(rng.choice(len(tr_ok), cap, replace=False))
        tr_ok = [tr_ok[i] for i in pick]
    if arm == "fld":
        # K3 on the 128 crop statistics of reading q, the training windows read without keeping their crops
        tr_recs = [r for r in read_units(arm, tr_ok, need="features", log=log)
                   if "features" in r and r["label"] is not None]
        xtr = np.stack([r["features"] for r in tr_recs])
        xva = np.stack([r["features"] for r in units])
    else:
        tr_recs = [r for r in read_units(arm, tr_ok, need="model", log=log)
                   if "series" in r and r["label"] is not None]
        xtr = k3_features(np.stack([r["series"] for r in tr_recs]), rep.normalize, arm)
        series = np.stack([r["series"] for r in units])
        xva = k3_features(series, rep.normalize, arm)
    ytr = np.array([r["label"] for r in tr_recs])
    prob, k3_fit = fit_no_encoder_classifier(xtr, ytr, A["trained"], xva)
    k3, k3_class = k3_signals(prob, A["trained"], pred)
    if arm == "fld":
        indices = fld_index_controls([r["indices"] for r in units])
    else:
        indices = index_controls(arm, series, np.stack([r["ndvi3x3"] for r in units]) if arm != "mangrove" else None)
    controls = {"k2_class_rarity": k2, **k3, **indices}
    names = [r["name"] for r in units]
    lonlat = [window_lonlat(r["meta"]) for r in units]
    cl = [cluster_ids(arm, r["meta"], r["name"], ll) for r, ll in zip(units, lonlat)]
    clusters = [c[0] for c in cl]
    if arm == "nandi":
        clusters = merge_by_polygon(clusters, [r["meta"].get("options") for r in units])
    out = {"names": np.array(names), "lonlat": np.array(lonlat), "label": label, "logits": logits,
           "clusters": np.array(clusters), "clusters_coarse": np.array([c[1] for c in cl]) if cl and cl[0][1] else None,
           "controls": controls, "k3_prob": prob, "k3_class": k3_class,
           "n_groups": np.array([r["n_groups"] for r in units]),
           "stratum": (np.array([str((r["meta"].get("options") or {}).get("source", "unrecorded")) for r in units])
                       if arm == "nandi" else
                       np.array([str(r.get("group")) for r in units]) if arm == "fld" else
                       np.array([str(v) for v in label]))}
    meta = {"n_val_windows": len(recs), "n_val_with_imagery": len(have), "n_units": int(keep.sum()),
            "dropped": {}, "k2_training_frequency": freq, "k3": k3_fit, "k3_cap": cap,
            "n_train_windows_labelled": int(tr_labels.size), "n_train_for_k3": len(tr_recs)}
    for r in recs:
        if r["label"] is None:
            _count(meta["dropped"], r["drop"])
    if arm == "mangrove":
        out["pixel"] = pixel_accuracy(have, rd_pred)
    if arm == "nandi":
        # reported, not graded, never a control candidate (control_names reads the k2_ to k5_ prefixes only)
        controls["reported_context_shift_instability"] = context_shift(rep, units, pred)
    if arm == "awf":
        meta["exp21_agreement"] = exp21_agreement(names, pred)
    if arm == "fld":
        meta["k3_feature_names"] = FLD_FEATURE_NAMES
        meta["val_units_per_group"] = {str(g): int(n) for g, n in zip(*np.unique(out["stratum"], return_counts=True))}
    return out, meta


def exp21_agreement(names, pred, path=os.path.join(OUT, "exp21_finetuned_awf.csv")):
    """Arm A, reported: how many windows get exp21's recorded prediction (its 16-px crop)."""
    import csv
    if not os.path.exists(path):
        return {"note": f"{path} not found"}
    rec = {r["window"].replace("\\", "/").split("/")[-1]: int(r["pred"]) for r in csv.DictReader(open(path))
           if r["crop"] == "16"}
    both = [(rec[n], int(p)) for n, p in zip(names, pred) if n in rec]
    return {"n_compared": len(both), "n_same_prediction": int(sum(a == b for a, b in both)),
            "n_not_in_exp21": int(sum(n not in rec for n in names))}


def context_shift(rep, units, pred):
    """Nandi, reported: the spread (standard deviation) of the shift-0 class's probability over crop shifts 0 to 3."""
    probs = []
    for s in range(4):
        inputs, locs = model_inputs("nandi", units, shift=s)
        lg = batched_logits(rep, inputs, locs, [r["n_groups"] for r in units], BATCH["nandi"])
        z = lg - lg.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        probs.append(p[np.arange(len(pred)), pred])
    return np.std(np.stack(probs), axis=0)


def write_units(units, path):
    flat = {}
    for k, v in units.items():
        if v is None:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}/{kk}"] = np.asarray(vv)
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(path, **{k: v for k, v in flat.items() if v.dtype != object})
    return {"path": os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, "bytes": os.path.getsize(path),
            "sha256": sha256(path), "arrays": sorted(flat)}


def read_units_file(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    units = {"controls": {}, "pixel": None}
    for k, v in d.items():
        if k.startswith("controls/"):
            units["controls"][k.split("/", 1)[1]] = v
        elif k.startswith("pixel/"):
            units.setdefault("pixel_raw", {})[k.split("/", 1)[1]] = v.item()
        else:
            units[k] = v
    units["pixel"] = units.pop("pixel_raw", None)
    return units


def summary_path(out_dir):
    return os.path.join(out_dir, "exp89_summary.json")


def run_arm(arm, out_dir, rep=None, windows=None, units_file=None, draws=DRAWS, n_boot=BOOT, log=print, k3_cap=None,
            synthetic=False):
    """The run past both guards: units (computed, or read from a units file), the grading, the summary. It checks the
    gate itself (require_gate); grading a units file needs the pass but not the ledger, which stays on the cluster.
    A report-only arm needs a recorded alignment check, aligned or not, and its numbers carry the check's label."""
    t0 = time.time()
    if not synthetic:
        if prereg_status() != "frozen":
            raise GateRefused(f"{os.path.relpath(PLAN, ROOT)} is not frozen: nothing is graded")
        require_gate(arm, out_dir, ledger_required=not units_file)
    if units_file:
        units, meta = read_units_file(units_file), {"from_units_file": units_file}
        units_info = {"path": units_file, "sha256": sha256(units_file)}
    else:
        units, meta = compute_units(arm, rep, windows, log, k3_cap=k3_cap, synthetic=synthetic, out_dir=out_dir)
        units_info = write_units(units, os.path.join(out_dir, f"exp89_units_{arm}.npz"))
        meta["replica"] = rep.info()
    res = grade_units(units, arm, draws=draws, n_boot=n_boot)
    gate = gate_passed(arm, out_dir)[1]
    res.update({"inputs": meta, "units_file": units_info, "seconds": time.time() - t0, "gate": gate})
    if is_report_only(arm):
        # "aligned", or "replica not aligned" when the last check is outside the tolerance or none is recorded
        res["alignment"] = alignment_label((gate or {}).get("attempts") or [])
    path = summary_path(out_dir)
    summary = json.load(open(path)) if os.path.exists(path) else {
        "experiment": "exp89: the package's three outputs on Ai2's fine-tuned models",
        "prereg_page": os.path.relpath(PLAN, ROOT), "arms": {}}
    commit, dirty = git_state()
    summary.update({"prereg_status": prereg_status(), "updated_utc": utc_now(), "commit": commit, "dirty": dirty,
                    "config": {"seed": SEED, "draws": draws, "bootstrap": n_boot, "review_budgets": REVIEW_BUDGETS,
                               "estimate_budgets": EST_BUDGETS, "delta": DELTA, "error_floor": ERROR_FLOOR,
                               "p2_auroc": P2_AUROC, "p3_points": P3_POINTS, "p4_cover": P4_COVER,
                               "p4_width": P4_WIDTH, "p5_violation": P5_VIOLATION, "p5_lead": P5_LEAD,
                               "k5_b02": K5_B02, "config_revision": CONFIG_REVISION}})
    summary["arms"][arm] = res
    summary["arms"]["nandi"] = summary["arms"].get("nandi") if arm == "nandi" else nandi_status_safe()
    if arm != "mangrove" and waiting_reason("mangrove"):
        summary["arms"]["mangrove"] = {"status": "not run", "reason": waiting_reason("mangrove")}
    dump(summary, path)
    if is_report_only(arm):
        rep_ = res["reported"]
        cap = (rep_.get("review_order") or {}).get("capture", {}).get(bkey(0.10))
        best = rep_.get("best_informative_control") or {}
        au = best.get("auroc")
        log(f"{arm}: report-only, {res['alignment']} | {res['n_errors']} errors of {res['n_units']} | confidence "
            f"captures {cap if cap is None else round(cap, 3)} of the errors in a 10% review; best informative "
            f"control {best.get('name')} (AUROC {au if au is None else round(au, 3)}) | {res['seconds']:.0f}s")
        return res
    g = res["prereg"]
    log(f"{arm}: " + ", ".join(f"{p} {g[p]['holds']}" for p in ("P1", "P2", "P3", "P4", "P5")) +
        f" | {res['n_errors']} errors of {res['n_units']} | {res['seconds']:.0f}s")
    return res


def nandi_status_safe():
    try:
        return nandi_status()
    except Exception as ex:                                            # offline: say so, never guess
        return {"status": "not run", "reason": ARMS["nandi"]["not_run_reason"], "hub": {"error": repr(ex)}}


def cmd_run(args):
    status = prereg_status()
    if status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} says the preregistration is {status}. No validation window is "
              "scored before freezing; --inventory and --smoke-torch --real (training windows) are allowed.", flush=True)
        return 2
    out_dir = args.out_dir or OUT
    if not_run_message(args.arm):
        print(not_run_message(args.arm), flush=True)
        return 0
    try:
        require_gate(args.arm, out_dir, ledger_required=not args.from_units)
        if args.from_units:
            res = run_arm(args.arm, out_dir, units_file=args.from_units)
        else:
            rep, windows = build_arm(args)
            res = run_arm(args.arm, out_dir, rep, windows)
    except GateRefused as ex:
        tail = ("Nothing is run until an alignment check is recorded." if is_report_only(args.arm)
                else "Nothing is graded until it passes.")
        print(f"refused: {ex} for arm {args.arm} (exp89_gate_{args.arm}.json). {tail}", flush=True)
        return 2
    if is_report_only(args.arm):
        return 0
    return 0 if res["prereg"]["complete"] else 1


# ----------------------------------------------------------------------------- S2 code on synthetic data
SYN_CRS = "EPSG:32737"


def write_tif(path, arr, crs, transform):
    import rasterio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.asarray(arr)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype.name, crs=crs, transform=transform) as dst:
        dst.write(arr)


def synthetic_dataset(root, arm, n_train=24, n_val=16, seed=0, size=None, dataset_bands="one_set"):
    """An rslearn dataset in the layout the investigation describes: windows/<group>/<name>/metadata.json (projection,
    bounds in pixels, time_range, options.split), layers/sentinel2[.k]/<band set>/geotiff.tif with `completed` markers
    for 12 item groups, items.json, and the label raster. Mangrove: 2x2 windows (and a few 4x4, to exercise the centre
    crop), one band set B01..B12 at 10 m, label_raster/label with classes 1..3 (a few mixed or 0 blocks). Nandi and
    AWF: `size` px windows with one labelled pixel in label/category (fill 10 or 9), the three band groups at 10, 20
    and 60 m (AWF's tar layout); one window with two labelled pixels and one month with no mosaic."""
    if arm == "fld":
        return synthetic_fld_dataset(root, n_train=n_train, n_val=n_val, seed=seed, size=size or 72)
    from rasterio.transform import Affine
    A = ARMS[arm]
    rng = np.random.default_rng(seed)
    wroot = os.path.join(root, "dataset", "windows", A["group"])
    groups = {"one_set": [("B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12", 1)],
              "three": [("B02_B03_B04_B08", 1), ("B05_B06_B07_B8A_B11_B12", 2), ("B01_B09", 6)]}[dataset_bands]
    names = []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        if arm == "mangrove":
            S = 4 if i % 11 == 5 else 2
        else:
            S = size or 24
        x0, y0 = 50000 + 7 * i, -100000 - 5 * i
        name = f"syn_{i:04d}" if arm != "awf" else f"task_{i % 4:02d}_point_{i}"
        wdir = os.path.join(wroot, name)
        os.makedirs(wdir, exist_ok=True)
        meta = {"group": A["group"], "name": name, "projection": {"crs": SYN_CRS, "x_resolution": 10, "y_resolution": -10},
                "bounds": [x0, y0, x0 + S, y0 + S], "time_range": ["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"],
                "options": {"split": split}}
        with open(os.path.join(wdir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        tr = Affine(10, 0, x0 * 10, 0, -10, y0 * -10)              # rslearn: geo = pixel bounds x resolution
        cls = int(rng.choice(A["trained"]))
        if arm == "mangrove":
            lab = np.full((S, S), cls, dtype=np.int32)
            if i % 13 == 7:
                lab[0, 0] = 0
            if i % 17 == 3:
                lab[-1, -1] = (cls % 3) + 1
        else:
            lab = np.full((S, S), A["label_fill"], dtype=np.int32)
            r, c = int(rng.integers(0, S)), int(rng.integers(0, S))
            lab[r, c] = cls
            if i == 2:
                lab[(r + 1) % S, c] = cls
        layer, band = A["label_layer"]
        write_tif(os.path.join(wdir, "layers", layer, band, "geotiff.tif"), lab[None], SYN_CRS, tr)
        open(os.path.join(wdir, "layers", layer, "completed"), "w").close()
        for t in range(N_MONTHS):
            lname = "sentinel2" if t == 0 else f"sentinel2.{t}"
            for bs, factor in groups:
                bnames = bs.split("_")
                h, w = max(1, -(-S // factor)), max(1, -(-S // factor))
                base = 300 + 2500 * (cls == 1) + 150 * t
                data = (base + rng.integers(0, 400, (len(bnames), h, w))).astype(np.uint16)
                if arm != "mangrove" and i == 3 and t == 5:
                    data[:] = 0                                            # a month no mosaic covered
                write_tif(os.path.join(wdir, "layers", lname, bs, "geotiff.tif"), data, SYN_CRS,
                          Affine(10 * factor, 0, x0 * 10, 0, -10 * factor, y0 * -10))
            open(os.path.join(wdir, "layers", lname, "completed"), "w").close()
        with open(os.path.join(wdir, "items.json"), "w") as f:
            json.dump([{"layer_name": "sentinel2", "serialized_item_groups": []}], f)
        names.append(name)
    return os.path.join(root, "dataset"), names


FLD_SYN_GROUPS = ("20250428_brazil_phase1", "peru3")
FLD_SYN_EXTRA_LAYERS = ("pre_sentinel2.4", "post_sentinel1", "mask")      # in the tar, never extracted
FLD_BAND_SET = "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12"


def synthetic_fld_dataset(root, n_train=10, n_val=6, seed=0, size=72):
    """Arm F's rslearn layout as the investigation describes it: windows/<group>/<name>/metadata.json (options.split
    beside a contradicting olmoearth_evals_split), the eight image layers pre_sentinel2[.k] and post_sentinel2[.k]
    (k 1 to 3), each one band set of the 12 bands at 10 m with rslearn's `completed` marker, the vector label layer
    layers/label/data.geojson (new_label) with its marker, and a stale label.json at the window's top that must never
    be read. Also layers the run never reads (pre_sentinel2.4, post_sentinel1, mask). Special windows: the first
    validation window lacks one image layer's marker (rslearn drops it), the second has new_label "unknown" (invalid),
    the third has a first feature without properties, and the fourth's pre_sentinel2.2 is all 0 (an empty timestep).
    Post layers have less NIR than pre layers where the class is not "none" (a loss signal for K4)."""
    from rasterio.transform import Affine
    rng = np.random.default_rng(seed)
    names = []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        j = i - n_train                                              # the validation index, negative for training
        group = FLD_SYN_GROUPS[0] if (split == "val" or i % 2) else FLD_SYN_GROUPS[1]
        name = f"fld_{i:04d}"
        wdir = os.path.join(root, "dataset", "windows", group, name)
        os.makedirs(wdir, exist_ok=True)
        x0, y0 = 30000 + 200 * i, -900000 - 150 * i
        meta = {"group": group, "name": name,
                "projection": {"crs": "EPSG:32722", "x_resolution": 10, "y_resolution": -10},
                "bounds": [x0, y0, x0 + size, y0 + size],
                "time_range": ["2024-03-01T00:00:00+00:00", "2024-03-02T00:00:00+00:00"],
                "options": {"split": split, "olmoearth_evals_split": "train" if split == "val" else "val"}}
        with open(os.path.join(wdir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        cls = int(rng.integers(0, len(FLD_CLASSES)))
        new_label = "unknown" if j == 1 else FLD_CLASSES[cls]
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [x0 * 10.0, y0 * -10.0]},
                  "properties": {"new_label": new_label, "old_label": "logging"}}]
        if j == 2:
            feats.insert(0, {"type": "Feature", "geometry": None, "properties": None})
        ldir = os.path.join(wdir, "layers", "label")
        os.makedirs(ldir, exist_ok=True)
        with open(os.path.join(ldir, "data.geojson"), "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        open(os.path.join(ldir, "completed"), "w").close()
        with open(os.path.join(wdir, "label.json"), "w") as f:
            json.dump({"new_label": "mining" if new_label != "mining" else "road"}, f)    # stale, never read
        tr = Affine(10, 0, x0 * 10, 0, -10, y0 * -10)
        loss = FLD_CLASSES[cls] != "none"
        for t, lname in enumerate(FLD_LAYERS + FLD_SYN_EXTRA_LAYERS):
            post = lname.startswith("post")
            base = rng.integers(200, 800, (12, size, size))
            nir = 3200 - (1600 if (post and loss) else 0) + 30 * t
            data = base.astype(np.uint16)
            data[FLD_BAND_SET.split("_").index("B08")] = nir + rng.integers(0, 200, (size, size))
            data[FLD_BAND_SET.split("_").index("B04")] = 400 + 600 * (post and loss)
            if j == 3 and lname == "pre_sentinel2.2":
                data[:] = 0                                              # an empty timestep
            write_tif(os.path.join(wdir, "layers", lname, FLD_BAND_SET, "geotiff.tif"), data, "EPSG:32722", tr)
            if not (j == 0 and lname == "post_sentinel2.2"):
                open(os.path.join(wdir, "layers", lname, "completed"), "w").close()
        names.append(name)
    with open(os.path.join(root, "dataset", "config.json"), "w") as f:
        json.dump({"layers": {n: {"type": "raster"} for n in FLD_LAYERS} | {"label": {"type": "vector"}}}, f)
    return os.path.join(root, "dataset"), names


def synthetic_tar(root, arm, **kw):
    """The synthetic dataset packed as a tar the way the Hub's mangrove.tar is fetched and extracted."""
    ds, names = synthetic_dataset(os.path.join(root, "src"), arm, **kw)
    path = os.path.join(root, f"synthetic_{arm}.tar")
    with tarfile.open(path, "w") as tf:
        tf.add(ds, arcname="dataset")
    return path, names


def synthetic_checkpoint(path, arm, model_id="OLMOEARTH_V1_NANO", seed=0):
    """A Lightning-style checkpoint with the encoder keys of a randomly initialised OlmoEarth encoder under rslearn's
    prefix, the arm's head keys, and a hyper_parameters entry pickling an rslearn object (which the loader must stub)."""
    import sys as _sys
    import types
    import torch
    torch.manual_seed(seed)
    enc = encoder_skeleton(model_id).encoder
    D = int(enc.embedding_size)
    A = ARMS[arm]
    if A["head_kind"] == "pooling_decoder":
        # SimpleTimeSeries around the wrapper: the encoder one level deeper, rslearn's PoolingDecoder at decoder.0
        sd = {FLD_ENCODER_PREFIX + k: v.clone() for k, v in enc.state_dict().items()}
        head = make_pooling_decoder(2 * D, A["n_out"], 1, 2, 128, 512)
        sd.update({FLD_HEAD_PREFIX + k: v.clone() for k, v in head.state_dict().items()})
    else:
        sd = {ENCODER_PREFIXES[0] + k: v.clone() for k, v in enc.state_dict().items()}
        wk, bk = A["head_keys"]
        w = torch.randn(A["n_out"], D) * 0.05
        sd[wk] = w if A["head_kind"] == "linear" else w[:, :, None, None]
        b = torch.zeros(A["n_out"])
        b[list(A["untrained"])] = -100.0                                    # the untrained channel is never chosen
        sd[bk] = b
    # a stand-in rslearn package, registered only while saving, so the checkpoint pickles an rslearn global as a real
    # one does; it is gone before anything loads the file
    added = [m for m in ("rslearn", "rslearn.fake_config") if m not in _sys.modules]
    pkg, mod = types.ModuleType("rslearn"), types.ModuleType("rslearn.fake_config")
    pkg.__path__ = []
    cls = type("FakeConfig", (), {"__module__": "rslearn.fake_config"})
    mod.FakeConfig = cls
    pkg.fake_config = mod
    for m, obj in (("rslearn", pkg), ("rslearn.fake_config", mod)):
        if m in added:
            _sys.modules[m] = obj
    try:
        obj = cls()
        obj.patch_size = A["patch"]
        torch.save({"state_dict": sd, "hyper_parameters": {"config": obj}, "epoch": 1}, path)
    finally:
        for m in added:
            del _sys.modules[m]
    return path, D


def smoke_torch(args):
    """S2's code end to end on synthetic data, or (with --real) on the pinned checkpoint and real training windows. The
    synthetic run keeps its gate ledger in a temporary directory, never in E89_GATE_LEDGER."""
    if args.real:
        return _smoke_torch(args)
    import tempfile
    old = os.environ.get(GATE_LEDGER_ENV)
    with tempfile.TemporaryDirectory() as ledger_tmp:
        os.environ[GATE_LEDGER_ENV] = ledger_tmp
        try:
            return _smoke_torch(args)
        finally:
            if old is None:
                os.environ.pop(GATE_LEDGER_ENV, None)
            else:
                os.environ[GATE_LEDGER_ENV] = old


def _smoke_torch(args):
    """The body of smoke_torch."""
    import tempfile
    out_dir = args.out_dir or OUT
    if args.real:
        if not_run_message(args.arm):
            print(not_run_message(args.arm), flush=True)
            return 0
        rep, windows = build_arm(args)
        res = model_smoke(args.arm, rep, windows, os.path.join(out_dir, f"exp89_s2_{args.arm}.json"))
        return 0 if res["ok"] else 1
    import torch
    t0 = time.time()
    # 1. the restricted loader: training-stack globals become inert stubs, a code-execution global is refused
    with tempfile.TemporaryDirectory() as tmp:
        ck, D = synthetic_checkpoint(os.path.join(tmp, "m.ckpt"), "mangrove")
        sd, how, top = load_checkpoint(ck)
        assert "stub" in how and "rslearn.fake_config.FakeConfig" in how, how
        assert set(top) == {"state_dict", "hyper_parameters", "epoch"}
        assert "rslearn" not in sys.modules, "a stub must not import the package it names"
        torch.save({"state_dict": {}, "x": os.system}, os.path.join(tmp, "bad.ckpt"))
        try:
            load_checkpoint(os.path.join(tmp, "bad.ckpt"))
        except RuntimeError as ex:
            assert "refused" in str(ex), ex
        else:
            raise AssertionError("a checkpoint that pickles os.system must be refused")
    report = {"loader": "stubs training-stack globals, refuses code-execution globals"}
    # 2. each arm: synthetic tar -> extraction -> inventory -> S2 -> gate -> run, past the freeze guard only
    for arm, kw in (("mangrove", {"n_train": 30, "n_val": 22}),
                    ("nandi", {"n_train": 6, "n_val": 4, "size": 24, "dataset_bands": "three"}),
                    ("awf", {"n_train": 10, "n_val": 6, "size": 63, "dataset_bands": "three"}),
                    ("fld", {"n_train": 10, "n_val": 7, "size": 72})):
        with tempfile.TemporaryDirectory() as tmp:
            tar_path, names = synthetic_tar(tmp, arm, **kw)
            data_dir = os.path.join(tmp, "data")
            if arm == "fld":                       # the layer filter; the size and MD5 pins are the real tar's
                ext = extract_layers(tar_path, data_dir, FLD_KEEP_LAYERS, {"tar": tar_path, "synthetic": True})
                assert set(ext["skipped"]) == set(FLD_SYN_EXTRA_LAYERS), ext["skipped"]
            else:
                extract_tar(tar_path, data_dir)
            ck, D = synthetic_checkpoint(os.path.join(tmp, "m.ckpt"), arm)
            sub_out = os.path.join(tmp, "out")
            # synthetic=True lets arm N's code run here, on synthetic data only; every real mode keeps refusing it
            ns = argparse.Namespace(arm=arm, data=data_dir, ckpt=ck, out_dir=sub_out, offline=True, real=False,
                                    from_units=None, synthetic=True, tar=None)
            assert cmd_inventory(ns) == 0
            inv = json.load(open(os.path.join(sub_out, f"exp89_inventory_{arm}.json")))
            d = inv["dataset"]
            # oe_inferencex.awf lists only windows with one labelled pixel, so arm A never sees the two-pixel window
            n_train = kw["n_train"] - (arm == "awf")
            assert d["windows_per_split"] == {"train": n_train, "val": kw["n_val"]}, d["windows_per_split"]
            if arm == "fld":
                assert d["item_groups_per_window"] == {"8": n_train + kw["n_val"] - 1, "7": 1}
                assert d["units_per_split"]["val"] == kw["n_val"] - 2, "a layer missing and an unknown label"
                assert d["label_json_differs"] == n_train + kw["n_val"], "label.json is stale and never read"
            else:
                assert d["item_groups_per_window"] == {"12": n_train + kw["n_val"]}
            assert all(v is not None for v in inv["checkpoint"]["head_expected"].values())
            if arm == "mangrove":
                assert sum(d["dropped_per_split"].get("val", {}).values()) + d["units_per_split"]["val"] == kw["n_val"]
                assert "4x4" in d["window_px"]
            elif arm == "nandi":
                assert d["dropped_per_split"]["train"]["2 labelled pixels"] == 1
            rep = Replica(arm, ck, model_id="OLMOEARTH_V1_NANO", device="cpu")
            if arm == "nandi":
                # the input path is exp21's when given exp21's timestamps: same normalisation, mask and stamps
                from oe_inferencex import awf
                x = np.random.default_rng(1).uniform(0, 4000, (2, 16, 16, 12, 12)).astype(np.float32)
                awf.CROP = 16
                ours, theirs = rep.sample(x, "exp21"), awf.stacks_to_sample(list(x), device="cpu")
                assert torch.equal(ours.sentinel2_l2a, theirs.sentinel2_l2a)
                assert torch.equal(ours.sentinel2_l2a_mask.float(), theirs.sentinel2_l2a_mask.float())
                assert torch.equal(ours.timestamps.long(), theirs.timestamps.long())
                legacy = rep.sample(x, "rslearn").timestamps[0]
                assert legacy[:, 0].tolist() == [1] * 12 and legacy[:, 1].tolist() == list(range(12))
                assert legacy[:, 2].tolist() == [2024] * 12, "rslearn's legacy stamps: day 1, month index, 2024"
            windows = arm_windows(arm, find_windows_root(data_dir))
            s2 = model_smoke(arm, rep, windows, os.path.join(sub_out, f"exp89_s2_{arm}.json"), n=8)
            assert s2["one_batch_output_shape"][-1] == ARMS[arm]["n_out"] and s2["n_pred_untrained"] == 0, s2
            if arm != "awf":
                assert s2["timestamp_conventions"]["max_abs_logit_difference"] is not None
            if arm == "fld":
                assert s2["bf16_autocast"]["max_abs_logit_difference"] is not None and s2["k3_features"]["finite"]
            g = run_gate(arm, rep, windows, sub_out, synthetic=True)
            ro = is_report_only(arm)
            assert set(g) == set(ALIGN_KEYS if ro else GATE_KEYS), sorted(g)
            assert set(g["attempts"][0]) == set(ALIGN_ATTEMPT_KEYS if ro else ATTEMPT_KEYS), sorted(g["attempts"][0])
            if arm == "fld":
                assert set(g["attempts"][0]["accuracy_by_flip"]) == set(FLIPS)
                assert g["attempts"][0]["n_windows"] == kw["n_val"] - 2
            res = run_arm(arm, sub_out, rep, windows, draws=20, n_boot=20, k3_cap=12, synthetic=True)
            if ro:
                assert res["report_only"] and res["alignment"] in ("aligned", NOT_ALIGNED)
                assert not verdicts_in({k: v for k, v in res.items() if k != "gate"}), verdicts_in(res)
            try:
                predict_population(arm, rep, windows)
            except RuntimeError as ex:
                assert "not frozen" in str(ex) or prereg_status() == "frozen"
            else:
                assert prereg_status() == "frozen", "a validation window was scored before freezing"
            assert os.path.exists(os.path.join(sub_out, f"exp89_units_{arm}.npz"))
            again = run_arm(arm, sub_out, units_file=os.path.join(sub_out, f"exp89_units_{arm}.npz"), draws=20,
                            n_boot=20, synthetic=True)
            assert again["inputs"]["from_units_file"]
            if prereg_status() != "frozen":
                # the guards on a draft page, on the synthetic outputs: no grading and no scoring without a pass
                for call in (lambda: run_arm(arm, sub_out, units_file=os.path.join(sub_out, f"exp89_units_{arm}.npz")),
                             lambda: compute_units(arm, rep, windows, out_dir=sub_out)):
                    try:
                        call()
                    except GateRefused:
                        pass
                    else:
                        raise AssertionError("a run on a draft page must be refused")
            assert g["attempts"][0]["checkpoint_sha256"] == rep.checkpoint_sha256 == sha256(ck)
            assert again["n_units"] == res["n_units"] and again["n_errors"] == res["n_errors"]
            if arm == "mangrove":
                assert res["accuracy"]["accuracy_pixel_micro"]["n_pixels"] > 0
            report[arm] = {"n_units": res["n_units"], "gate_accuracy": g["attempts"][0]["accuracy"],
                           "embedding": D, "s2_windows": s2["n_windows"]}
            print(f"smoke-torch {arm}: inventory, S2 ({s2['n_windows']} training windows), gate and run on a "
                  f"synthetic tar and checkpoint; {res['n_units']} units", flush=True)
    # 3. K3 is deterministic at seed 89
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((600, 20)), rng.integers(1, 4, 600)
    p_a, _ = fit_no_encoder_classifier(x, y, (1, 2, 3), x[:50])
    p_b, _ = fit_no_encoder_classifier(x, y, (1, 2, 3), x[:50])
    assert np.array_equal(p_a, p_b), "K3's fit must repeat exactly at seed 89"
    report["seconds"] = time.time() - t0
    print(f"smoke-torch OK in {report['seconds']:.0f}s: the restricted loader, every arm's reader, replica, S2, gate "
          "and run on synthetic data, K3 deterministic", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARMS), default="mangrove")
    ap.add_argument("--smoke", action="store_true", help="S1: numpy only, synthetic units with known answers")
    ap.add_argument("--smoke-torch", action="store_true", help="S2's code on a synthetic tar and checkpoint")
    ap.add_argument("--real", action="store_true", help="with --smoke-torch: S2 on the pinned checkpoint and real "
                                                        "TRAINING windows")
    ap.add_argument("--inventory", action="store_true", help="files, checkpoint keys, windows and labels per split")
    ap.add_argument("--gate", action="store_true", help="accuracy only; refused until the page is frozen")
    ap.add_argument("--from-units", default=None, help="the full run from a units file (grading only)")
    ap.add_argument("--data", default=None, help="where the arm's dataset is (or is extracted to)")
    ap.add_argument("--ckpt", default=None, help="a local checkpoint instead of the pinned download")
    ap.add_argument("--tar", default=None, help="the arm's dataset tar on disk (arm F: required for the first "
                                                "extraction; the job downloads it), checked against its pin")
    ap.add_argument("--offline", action="store_true", help="no Hub access: use --data and --ckpt as they are")
    ap.add_argument("--out-dir", default=None, help="where the outputs go (default: exp/out)")
    args = ap.parse_args(argv)
    if args.smoke:
        smoke(args.out_dir)
        return 0
    if args.smoke_torch:
        return smoke_torch(args)
    if args.inventory:
        return cmd_inventory(args)
    if args.gate:
        return cmd_gate(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
