#!/usr/bin/env python
"""exp55: the window protocol on GEOID-Flood, a rate over flood events instead of one Bolivia exception.

Why. Every "exception" this repository reports is one flood event: on Sen1Floods11 Bolivia the no-model NDWI-level
control out-ranked the served backbone's grid confidence (exp45), the averaged decision recovered the lead (exp47), and
under Ai2's own Sentinel-1 probe the pre-event NDWI prior out-ranked the probe's confidence (exp51). One event cannot
say how often that happens. GEOID-Flood (issue #15; Hub links-ads/geoid-flood, CC-BY-4.0) holds 219 CEMS activations
in 65 countries, 2016-2026, 14,282 tiles of 1024 x 1024 px at 10 m, splits assigned per area of interest (AoI) so no
event straddles them (train 8,938 / val 1,241 / test 2,674 tiles). Each tile carries a pre-event cloud-filtered
Sentinel-2 L2A composite, pre- and post-event Sentinel-1 GRD, a three-class label (0 background, 1 permanent water,
2 flooded water, 255 ignore) and a cloud mask over the composite (0 clear, 1 thick cloud, 2 thin cloud, 3 shadow).
This run grades the shift-averaged decision (exp42, exp47) and its rankers event by event, and reports the share of
events on which a no-model control beats the model's own confidence.

Design.
  Data (full run). A subset of the uncompressed tar shards ({tree}/shards/{split}/{layer}/{split}-{layer}-NNNNN-of-NNNNN.tar,
  members at canonical paths such as EMSR151-1/s1grd/EMSR151-1-0_s1grd_post_<date>.tif): for the test split the first
  --test-shards (4) shards of s2l2a and of s1grd plus the single label and cloudmask shards; for the val split, used
  only to fit the heads, the first --val-shards (2) shards of s2l2a and s1grd plus label and cloudmask. Shards are 2 GB
  each, so the default subset is ~24 GB. Each shard is downloaded with hf_hub_download into $HF_HOME/geoid_flood, its
  wanted members (s1grd: the post pass only) extracted with tarfile into a per-split tree root, and deleted. The Hub's
  shard_index.json.gz says the first four test shards of both imagery layers share 598 tiles in 55 AoIs and the first
  two val shards 293 tiles in 12 AoIs; tiles with all four layers present (s2l2a pre, s1grd post, label, cloudmask) are
  kept, and the AoIs and tile counts that landed are recorded.
  Chips. From each 1024 x 1024 tile, 64 x 64 chips on a stride-256 grid (16 per tile), read as rasterio windows. A chip
  is kept if >= 50% of its label pixels are not 255; for the S2 task also if >= 90% of its cloudmask pixels are 0. The
  kept chips are capped at --chips-per-event (400, random, seed 0) per AoI on both splits, so no event dominates the
  test statistics or the heads; the S2 task's chips are the clear subset of the capped set, so both sensors' heads are
  graded on identical windows wherever the S2 task is defined.
  Tasks, each a binary water map graded with exp47's protocol (4-px windows over the 60-px crop at crop offsets 0..3):
    A  permanent water from the pre-event Sentinel-2: truth = (label == 1), flooded pixels count as not-water, 255 is
       ignored. Encoder exp18.embed on the twelve bands reordered to OlmoEarth's band order (the mapping below);
       GEOID's uint16 L2A reflectance is on the same 0-10000 scale as Sen1Floods11's DN and is passed through unchanged.
    B  water after the event from the post-event Sentinel-1: truth = (label in {1, 2}). Encoder exp46.embed_s1, which
       crops to the 60-px window at offset 0 internally, on the chips pre-shifted by s px for the four offsets (the
       pre-cropped-tile route of exp44), after converting linear sigma0 to dB as 10 log10(max(x, 1e-6)) on read (non-
       finite values and the loader's > 1e3 fill values set to nodata first), which is what the OlmoEarth normaliser and
       exp46 expect (Sen1Floods11's S1 is in dB; the dataset's loader does the same with float32 epsilon as the floor).
    C  secondary, flooded water only from the post-event Sentinel-1: truth = (label == 2), permanent water not-water.
  Heads and decision. Per task a balanced logistic head (evidence.train_logistic_head, seed 0, exp47.head_from) on the
  labelled windows of the val-split chips at shift 0; on the test-split chips the four shift maps, the W1 decision
  (pool_to_windows(shift_averaged_probability(.), offset=0), windows 1..G-1, exp47/exp49), its error set, and the
  rankers graded on it: the averaged confidence -|W1 - 0.5|; aligned tile-phase; the no-model controls NDWI level from
  the pre-event S2 (for tasks B and C a cross-sensor prior, exp51's sensor flip) and S1 level, -|window-mean VV_dB -
  median VV_dB over the val chips| (exp51.s1_level; for task A a cross-sensor control); the boundary indicator of the
  averaged hard map; and the boundary-first order (assess.boundary_first_score). Also exp51's cross-tab between task
  A's S2-head errors and task B's S1-head errors on identical windows (phi, P(S1 wrong | S2 wrong)).
  Scoring, per event (AoI) and pooled over the test chips: pooled excess AURC per ranker (aurc_expected - oracle_aurc),
  per-chip exact sign tests against the averaged confidence (exp47's score, chips with 3 <= errors <= n - 3, one-sided
  for the task's primary control), tie-aware capture at the 5 / 10 / 20% budgets (the boundary-first order rebuilt
  with one midrank over the set, exp36), and exp36's per-chip test of boundary-first against confidence at the 5%
  budget. Across events: for each control, the number and share of scored events (3 <= pooled errors <= n - 3) on
  which the control's pooled excess AURC is below confidence's, the "exception rate", with a one-sided exact sign test
  over events (sign_test(w, l, "greater"), w = events confidence wins); the median over events of confidence's capture
  at the 5% budget and of its ratio to random (0.05).

Band mapping. GEOID's s2l2a GeoTIFFs carry no band descriptions; the dataset code (github.com/links-ads/geoid-flood,
src/geoid_flood/datasets/geoid.py) selects S2L2A_BAND_SELECT_INDICES = [1..8, 10, 11] with the comment "B02..B08, B8A,
B11, B12 (skip B01, B09)" and its configs name those ten BLUE, GREEN, RED, RED_EDGE_1/2/3, NIR_BROAD, NIR_NARROW,
SWIR_1, SWIR_2, so the on-disk order is B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 (the twelve L2A
bands in ascending order). OlmoEarth's order is B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09, so
BAND_IDX = [1, 2, 3, 7, 4, 5, 6, 8, 10, 11, 0, 9], the analogue of exp18.BAND_IDX. Caveat: the composites come from the
Planetary Computer L2A collection (the dataset code's tools/download_emsr/download_s2l2a_emsr_planetary.py); scenes
processed with baseline 04.00 (January 2022 on) carry Sen2Cor's +1000 BOA offset, which the dataset keeps (the 2023
EMSR712 sample's band minima sit near 1000); it is passed through unchanged, and the 1st percentile of B02 per event is
recorded so the effect on the NDWI control can be read off.

Preregistered, stated before the run:
  P1  task A (permanent water, S2 head): the averaged confidence beats the NDWI-level control on more events than it
      loses to, one-sided sign test over events p < 0.05, and pooled lead >= 0.001.
  P2  task B (post-event water, S1 head): the averaged confidence beats the S1-level control on more events than it
      loses to (p < 0.05) and pooled lead >= 0.001.
  P3  effect size: the median over events of confidence's error capture at the 5% budget is at least 0.25 (five times
      random) on both tasks A and B.
Stated predictions, descriptive: the exception rate of the NDWI control on task A is below one third of events; on task
B the pre-event NDWI prior out-ranks the S1 head's confidence on a majority of events (exp51's sensor flip);
boundary-first beats confidence at the 5% budget on a majority of events (exp36).

Inputs: the Hub dataset links-ads/geoid-flood (downloaded into $HF_HOME/geoid_flood; ~24 GB of shards for the defaults,
each deleted after extraction). Outputs: exp/out/exp55_summary.json (config with shards, AoIs, tiles and chips per split
and task, per-event and pooled results, exception rates, prereg), exp/out/exp55_geoid_flood.csv (one row per task x
event plus a pooled row), and exp/out/exp55_summary.partial.json checkpointed after each stage.
--smoke: CPU, no shards: three tiles of sample/geoid-flood/EMSR712-3 (test) and three of EMSR712-10 (the val stand-in),
chosen as the tiles with the most balanced labels after downloading every label of the sample tree (9 KB each), then
the four layers per tile (~200 MB, cached under HF_HOME), 4 chips per tile (the most balanced labelled ones), one event
per split, the whole pipeline, _smoke outputs.
"""
import csv
import json
import os
import re
import sys
import tarfile
import time
import traceback
from collections import Counter

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp46_shared_error_sources as e46  # noqa: E402
import exp51_their_probe as e51  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import boundary_indicator, ndwi_level, pool_to_windows, shift_averaged_probability  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

REPO, TREE = "links-ads/geoid-flood", "geoid-flood"
LAYERS = ("s2l2a", "s1grd", "label", "cloudmask")
PASS = {"s2l2a": "pre", "s1grd": "post", "cloudmask": "pre", "label": None}     # the pass each task reads
SMOKE_AOIS = {"EMSR712-3": "test", "EMSR712-10": "val"}                           # the sample tree's two AoIs
TILE_PX, CHIP, STRIDE = 1024, 64, 256
POSITIONS = [(r, c) for r in range(0, TILE_PX, STRIDE) for c in range(0, TILE_PX, STRIDE)]
PATCH, CROP, G, SHIFTS = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS
DEV = exp18.DEV
MIN_VALID, MIN_CLEAR, S1_FLOOR, S1_FILL_MAX = 0.5, 0.9, 1e-6, 1e3
MIN_LEAD, BUDGETS, MIN_CAPTURE5 = 0.001, (0.05, 0.10, 0.20), 0.25
GEOID_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]   # on-disk order
BAND_IDX = [GEOID_S2_BANDS.index(b) for b in exp18.OE_BANDS]                     # GEOID order -> OlmoEarth order
assert BAND_IDX == [1, 2, 3, 7, 4, 5, 6, 8, 10, 11, 0, 9], BAND_IDX
CONF, TILE, CTRL_NDWI, CTRL_S1, BOUND, LEX = ("averaged confidence", "tile-phase", "control NDWI level", "control S1 level",
                                              "boundary indicator", "boundary first, then confidence")
RANKERS = [CONF, TILE, CTRL_NDWI, CTRL_S1, BOUND, LEX]
CONTROLS = [TILE, CTRL_NDWI, CTRL_S1, BOUND, LEX]
TASKS = {"A": ("permanent water from the pre-event S2", "s2"), "B": ("water after the event from the post-event S1", "s1"),
         "C": ("flooded water only from the post-event S1", "s1")}
PRIMARY = {"A": CTRL_NDWI, "B": CTRL_S1, "C": CTRL_S1}
NAME_RE = re.compile(r"^(?P<aoi>[A-Za-z0-9]+-\d+)-(?P<idx>\d+)_(?P<layer>[a-z0-9]+)(?:_(?P<pas>pre|post)_(?P<date>[^.]+))?\.tif$")


# ----------------------------------------------------------------------------- labels
def task_labels(lab, task):
    """The three-class label chips (uint8: 0 background, 1 permanent, 2 flooded, 255 ignore) -> the task's map in {-1, 0, 1}."""
    water = {"A": lab == 1, "B": (lab == 1) | (lab == 2), "C": lab == 2}[task]
    return np.where(lab == 255, -1, water.astype(np.int64))


# ----------------------------------------------------------------------------- data: the Hub tree
def parse_name(name):
    m = NAME_RE.match(os.path.basename(name))
    return m.groupdict() if m else None


def wanted(name):
    """A member or file this run reads: one of the four layers, in the pass the tasks use."""
    d = parse_name(name)
    return d is not None and d["layer"] in LAYERS and (PASS[d["layer"]] is None or d["pas"] == PASS[d["layer"]])


def tile_table(paths):
    """{file name: local path} -> {tile id: {"aoi", layer: path}} for the tiles holding all four layers; the first date wins
    when a layer has several files for one tile."""
    tiles = {}
    for name in sorted(paths):
        if not wanted(name):
            continue
        d = parse_name(name)
        tiles.setdefault(f"{d['aoi']}-{d['idx']}", {"aoi": d["aoi"]}).setdefault(d["layer"], paths[name])
    return {k: v for k, v in sorted(tiles.items()) if all(layer in v for layer in LAYERS)}


def hub_files():
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(REPO, repo_type="dataset")


def sample_tiles(files, n_tiles):
    """The smoke's tiles: every label of the sample tree (9 KB each) is downloaded, the n_tiles tiles per AoI with the largest
    min(water, background) labelled pixel count are chosen, and their four layers downloaded (cached under HF_HOME)."""
    from huggingface_hub import hf_hub_download
    names = sorted(f for f in files if f.startswith("sample/geoid-flood/") and f.endswith(".tif"))
    balance = {}
    for f in [f for f in names if "/label/" in f]:
        with rasterio.open(hf_hub_download(REPO, f, repo_type="dataset")) as src:
            a = src.read(1)
        d = parse_name(f)
        balance[f"{d['aoi']}-{d['idx']}"] = (d["aoi"], int(min(((a == 1) | (a == 2)).sum(), (a == 0).sum())))
    out = {}
    for aoi, split in SMOKE_AOIS.items():
        chosen = sorted([t for t, (a, _) in balance.items() if a == aoi], key=lambda t: (-balance[t][1], t))[:n_tiles]
        paths = {}
        for f in names:
            d = parse_name(f)
            if d and f"{d['aoi']}-{d['idx']}" in chosen and wanted(f):
                paths[f] = hf_hub_download(REPO, f, repo_type="dataset")
        out[split] = tile_table(paths)
        print(f"  smoke {split}: {aoi} tiles {chosen} (min(water, background) pixels {[balance[t][1] for t in chosen]})", flush=True)
    return out


def fetch_split(files, split, n_shards, local_dir, root):
    """The first n_shards shards of s2l2a and s1grd and the single label and cloudmask shards of a split: each downloaded with
    hf_hub_download into local_dir, its wanted members extracted into root/split, then deleted; a marker skips finished shards."""
    from huggingface_hub import hf_hub_download
    dest = os.path.join(root, split)
    done_dir = os.path.join(dest, ".done")
    os.makedirs(done_dir, exist_ok=True)
    shards = []
    for layer in LAYERS:
        names = sorted(f for f in files if f.startswith(f"{TREE}/shards/{split}/{layer}/") and f.endswith(".tar"))
        shards += names[:n_shards] if layer in ("s2l2a", "s1grd") else names
    got = []
    for sh in shards:
        marker = os.path.join(done_dir, os.path.basename(sh) + ".done")
        if os.path.exists(marker):
            with open(marker) as f:
                got.append({"shard": sh, "members_extracted": int(f.read() or 0), "cached": True})
            print(f"  {sh}: already extracted", flush=True)
            continue
        t0 = time.time()
        path = hf_hub_download(REPO, sh, repo_type="dataset", local_dir=local_dir)
        t1 = time.time()
        with tarfile.open(path) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and wanted(m.name)]
            try:
                tar.extractall(path=dest, members=members, filter="data")
            except TypeError:                                            # Python without the extraction filter
                tar.extractall(path=dest, members=members)
        size = os.path.getsize(path)
        os.remove(path)
        with open(marker, "w") as f:
            f.write(str(len(members)))
        got.append({"shard": sh, "size_gb": size / 1e9, "members_extracted": len(members), "download_seconds": t1 - t0, "extract_seconds": time.time() - t1})
        print(f"  {sh}: {size / 1e9:.2f} GB in {t1 - t0:.0f}s, {len(members)} members extracted in {time.time() - t1:.0f}s", flush=True)
    return got


def scan_split(root, split):
    paths = {}
    for dirpath, _, fnames in os.walk(os.path.join(root, split)):
        for fn in fnames:
            if fn.endswith(".tif"):
                paths[fn] = os.path.join(dirpath, fn)
    return tile_table(paths)


# ----------------------------------------------------------------------------- chips
def read_chip(src, r, c):
    return src.read(window=Window(c, r, CHIP, CHIP))


def chip_candidates(tiles, split):
    """Pass 1, labels and cloud masks only: every grid position of every tile, kept when >= MIN_VALID of its label pixels are labelled."""
    rows = []
    for tid, t in tiles.items():
        with rasterio.open(t["label"]) as sl, rasterio.open(t["cloudmask"]) as sc:
            for r, c in POSITIONS:
                lab, cm = read_chip(sl, r, c)[0], read_chip(sc, r, c)[0]
                valid = float((lab != 255).mean())
                if valid < MIN_VALID:
                    continue
                rows.append({"split": split, "aoi": t["aoi"], "tile": tid, "r": r, "c": c, "valid": valid, "clear": float((cm == 0).mean()),
                             "balance": int(min(((lab == 1) | (lab == 2)).sum(), (lab == 0).sum())), "lab": lab})
    return rows


def select_chips(rows, cap, seed=0, smoke_chips=None):
    """The per-AoI cap (random, seed 0); in the smoke the smoke_chips most balanced labelled chips per tile instead."""
    if smoke_chips:
        by_tile = {}
        for x in rows:
            by_tile.setdefault(x["tile"], []).append(x)
        return [x for t in sorted(by_tile) for x in sorted(by_tile[t], key=lambda x: (-x["balance"], x["r"], x["c"]))[:smoke_chips]]
    rng = np.random.default_rng(seed)
    by_aoi = {}
    for x in rows:
        by_aoi.setdefault(x["aoi"], []).append(x)
    out = []
    for aoi in sorted(by_aoi):
        xs = by_aoi[aoi]
        if len(xs) > cap:
            xs = [xs[i] for i in sorted(rng.choice(len(xs), cap, replace=False))]
        out += xs
    return out


def read_imagery(tiles, chips):
    """Pass 2: the S2 windows (OlmoEarth band order, float32 DN unchanged) and post-event S1 windows (dB) of the selected chips."""
    N = len(chips)
    s2 = np.zeros((N, 12, CHIP, CHIP), np.float32)
    s1 = np.zeros((N, 2, CHIP, CHIP), np.float32)
    lab = np.stack([x["lab"] for x in chips]) if N else np.zeros((0, CHIP, CHIP), np.uint8)
    by_tile = {}
    for i, x in enumerate(chips):
        by_tile.setdefault(x["tile"], []).append(i)
    for tid, idx in by_tile.items():
        with rasterio.open(tiles[tid]["s2l2a"]) as src:
            for i in idx:
                s2[i] = read_chip(src, chips[i]["r"], chips[i]["c"])[BAND_IDX].astype(np.float32)
        with rasterio.open(tiles[tid]["s1grd"]) as src:
            for i in idx:
                x = read_chip(src, chips[i]["r"], chips[i]["c"]).astype(np.float32)
                x = np.where(np.isfinite(x) & (x <= S1_FILL_MAX), x, 0.0)              # the loader's nodata rule, then dB
                s1[i] = 10.0 * np.log10(np.maximum(x, S1_FLOOR))
    return s2, s1, lab


# ----------------------------------------------------------------------------- encoder, heads, rankers
def features(model, sensor, x, shift, chunk=256):
    """(N, G, G, D) float32 pooled tokens at crop offset `shift`: exp18.embed for S2 (chunked, so its band-set output never
    holds more than `chunk` chips), exp46.embed_s1 on the pre-shifted chips for S1 (it crops to CROP at offset 0 itself)."""
    if sensor == "s2":
        return np.concatenate([np.asarray(exp18.embed(model, x[i:i + chunk], shift)[0], dtype=np.float32) for i in range(0, len(x), chunk)])
    return np.asarray(e46.embed_s1(model, x[:, :, shift:shift + CROP, shift:shift + CROP]), dtype=np.float32)


def fit_head(feats, lab_task):
    """exp47.head_from: the balanced logistic head on the labelled windows, seed 0."""
    y, ok = exp18.patch_labels(lab_task[:, :CROP, :CROP])
    torch.manual_seed(0)
    return train_logistic_head(torch.tensor(feats.reshape(-1, feats.shape[-1])[ok.flatten()]), y.flatten()[ok.flatten()])


def arm(p_shift, s2, s1, lab_task, med_vv):
    """exp47's construction: the averaged decision's error set on windows 1..G-1 of the shift-0 grid and every ranker on it."""
    N = len(lab_task)
    y, ok = exp18.patch_labels(lab_task[:, :CROP, :CROP])
    w1_full = np.stack([pool_to_windows(shift_averaged_probability(p_shift[:, t], patch=PATCH)[0], patch=PATCH, offset=0) for t in range(N)])
    bound_full = boundary_indicator(w1_full, probabilities=True)          # neighbours taken before the crop
    sl = slice(1, G)
    w1, yy, okk = w1_full[:, sl, sl], y[:, sl, sl], ok[:, sl, sl]
    err = ((w1 > 0.5) != (yy > 0.5)).astype(np.float64)
    sig = {CONF: -np.abs(w1 - 0.5),
           TILE: exp18.aligned_tile_phase(p_shift)[:, sl, sl],
           CTRL_NDWI: np.stack([ndwi_level(t, patch=PATCH, size=CROP) for t in s2])[:, sl, sl],
           CTRL_S1: e51.s1_level(s1, med_vv, PATCH, CROP)[:, sl, sl],
           BOUND: bound_full[:, sl, sl]}
    sig[LEX] = np.stack([boundary_first_score(sig[CONF][t], sig[BOUND][t]) for t in range(N)])
    return sig, err, okk


def score(sig, err, ok, primary):
    """exp47's score on a set of chips (per-chip excess AURC and sign tests against the averaged confidence, one-sided for
    `primary`), the pooled excess AURC per ranker, the tie-aware capture at BUDGETS pooled over the set (the boundary-first
    order rebuilt with one midrank over the set, as exp36 pooled it), and exp36's per-chip capture test at 5%."""
    n, k = int(ok.sum()), int(err[ok].sum())
    out = {"n_chips": int(len(err)), "n_windows": n, "n_errors": k, "accuracy": float(1 - k / n) if n else None,
           "scorable": bool(n > 0 and 3 <= k <= n - 3), "n_chips_scored": 0,
           "pooled_eaurc": {kk: None for kk in sig}, "tests": {}, "capture": {kk: {b: None for b in BUDGETS} for kk in sig},
           "lex_vs_conf_capture5": None}
    if n == 0:
        return out
    scored = [t for t in range(len(err)) if ok[t].any() and 3 <= err[t][ok[t]].sum() <= ok[t].sum() - 3]
    e = err[ok]
    pooled_sig = {kk: np.asarray(v)[ok] for kk, v in sig.items()}
    pooled_sig[LEX] = boundary_first_score(sig[CONF][ok], sig[BOUND][ok])
    out["pooled_eaurc"] = {kk: aurc_expected(v, e) - oracle_aurc(n, k) for kk, v in pooled_sig.items()}
    out["capture"] = {kk: capture_at_budget_expected(v, e, BUDGETS) for kk, v in pooled_sig.items()}
    per = {kk: [aurc_expected(np.asarray(sig[kk][t])[ok[t]], err[t][ok[t]]) - oracle_aurc(int(ok[t].sum()), int(err[t][ok[t]].sum())) for t in scored] for kk in sig}
    for kk in sig:
        if kk == CONF:
            continue
        g = np.array(per[kk]) - np.array(per[CONF])                          # positive: the averaged confidence is better
        w, l, t_ = wins_losses_ties(g)
        one = kk == primary
        out["tests"][kk] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                            "median_gain": float(np.median(g)) if len(g) else None,
                            "pooled_lead": out["pooled_eaurc"][kk] - out["pooled_eaurc"][CONF]}
    cap5 = lambda kk, t: capture_at_budget_expected(np.asarray(sig[kk][t])[ok[t]], err[t][ok[t]], (0.05,))[0.05]   # noqa: E731
    g = np.array([cap5(LEX, t) - cap5(CONF, t) for t in scored])
    w, l, t_ = wins_losses_ties(g)
    out["lex_vs_conf_capture5"] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater"), "one_sided": True,
                                   "median_gain": float(np.median(g)) if len(g) else None}
    out["n_chips_scored"] = len(scored)
    return out


def across_events(per_event):
    """The event-level statistics: per control the number and share of scored events on which it beats the averaged confidence
    (pooled excess AURC within the event), a one-sided sign test over events (confidence better), the median over events of
    confidence's capture at 5% and its ratio to random, and boundary-first against confidence at 5% over events."""
    ev = [e for e, r in per_event.items() if r["scorable"]]
    out = {"n_events": len(per_event), "n_events_scored": len(ev), "events_scored": ev}
    for kk in CONTROLS:
        d = np.array([per_event[e]["pooled_eaurc"][kk] - per_event[e]["pooled_eaurc"][CONF] for e in ev])   # positive: confidence better
        w, l, t_ = wins_losses_ties(d)
        out[kk] = {"confidence_wins": w, "control_wins": l, "ties": t_, "exception_rate": (l / len(ev)) if ev else None,
                   "sign_p": sign_test(w, l, "greater"), "median_lead": float(np.median(d)) if len(d) else None,
                   "exceptions": [e for e, x in zip(ev, d) if x < -1e-12]}
    cap = np.array([per_event[e]["capture"][CONF][0.05] for e in ev])
    out["median_capture5_confidence"] = float(np.median(cap)) if len(cap) else None
    out["median_capture5_ratio_to_random"] = float(np.median(cap / 0.05)) if len(cap) else None
    g = np.array([per_event[e]["capture"][LEX][0.05] - per_event[e]["capture"][CONF][0.05] for e in ev])
    w, l, t_ = wins_losses_ties(g)
    out["boundary_first_vs_confidence_capture5"] = {"lex_wins": w, "confidence_wins": l, "ties": t_, "share_lex_wins": (w / len(ev)) if ev else None,
                                                    "sign_p": sign_test(w, l, "greater"), "median_gain": float(np.median(g)) if len(g) else None}
    return out


def row(task, event, r):
    return {"task": task, "event": event, "n_chips": r["n_chips"], "n_chips_scored": r["n_chips_scored"], "n_windows": r["n_windows"],
            "n_errors": r["n_errors"], "window_acc": r["accuracy"], "scorable": r["scorable"],
            **{f"eaurc {kk}": r["pooled_eaurc"].get(kk) for kk in RANKERS},
            "capture5 confidence": r["capture"][CONF][0.05], "capture5 boundary first": r["capture"][LEX][0.05],
            "controls beating confidence": ";".join(kk for kk in CONTROLS if r["scorable"] and r["pooled_eaurc"][kk] < r["pooled_eaurc"][CONF]),
            "primary control": PRIMARY[task], "primary sign_p per chip": (r["tests"].get(PRIMARY[task]) or {}).get("sign_p"),
            "s2_b02_p01": r.get("s2_b02_p01")}


def checkpoint(summary, suffix):
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp55_summary{suffix}.partial.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)


def fail(summary, part, ex):
    summary["failures"].append({"part": part, "error": repr(ex), "traceback": traceback.format_exc()})
    print(f"{part} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--test-shards", type=int, default=4, help="shards of s2l2a and of s1grd to take from the test split")
    ap.add_argument("--val-shards", type=int, default=2, help="shards of s2l2a and of s1grd to take from the val split (heads only)")
    ap.add_argument("--chips-per-event", type=int, default=400, help="cap on kept chips per AoI, random with seed 0")
    ap.add_argument("--smoke-chips", type=int, default=4, help="chips per tile in the smoke")
    ap.add_argument("--data-dir", default=None, help="where shards land and the tree is extracted (default $HF_HOME/geoid_flood)")
    ap.set_defaults(smoke_tiles=3)
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    data_dir = args.data_dir or os.path.join(hf_home, "geoid_flood")
    t_start = time.time()
    summary = {"experiment": "exp55 the window protocol on GEOID-Flood, event by event", "smoke": args.smoke, "device": DEV,
               "config": {"repo": REPO, "tree": TREE, "layers": list(LAYERS), "passes": PASS, "chip_px": CHIP, "stride_px": STRIDE, "crop": CROP, "patch": PATCH,
                          "shifts": list(SHIFTS), "min_valid": MIN_VALID, "min_clear": MIN_CLEAR, "s1_db": f"10 log10(max(x, {S1_FLOOR})), non-finite and > {S1_FILL_MAX} set to nodata",
                          "s2_band_idx": BAND_IDX, "s2_on_disk_order": GEOID_S2_BANDS, "s2_encoder_order": list(exp18.OE_BANDS),
                          "test_shards": args.test_shards, "val_shards": args.val_shards, "chips_per_event": args.chips_per_event,
                          "tasks": {k: {"name": v[0], "sensor": v[1], "primary_control": PRIMARY[k]} for k, v in TASKS.items()},
                          "rankers": RANKERS, "min_lead": MIN_LEAD, "budgets": list(BUDGETS), "min_capture5": MIN_CAPTURE5,
                          "prereg": "P1 task A: confidence beats NDWI level on more events than it loses (one-sided sign test over events p < 0.05) and pooled lead >= 0.001; "
                                    "P2 task B: the same against S1 level; P3 median over events of confidence's capture at 5% >= 0.25 on tasks A and B"},
               "results": {}, "failures": [], "timing": {}}
    rows = []

    # ---- stage 1: the tiles
    t0 = time.time()
    try:
        files = hub_files()
        if args.smoke:
            tiles = sample_tiles(files, args.smoke_tiles)
        else:
            root = os.path.join(data_dir, "tree")
            summary["config"]["shards"] = {split: fetch_split(files, split, n, data_dir, root) for split, n in (("test", args.test_shards), ("val", args.val_shards))}
            tiles = {split: scan_split(root, split) for split in ("test", "val")}
        summary["config"]["data"] = {split: {"n_tiles": len(t), "n_aois": len({v["aoi"] for v in t.values()}),
                                             "tiles_per_aoi": dict(sorted(Counter(v["aoi"] for v in t.values()).items()))} for split, t in tiles.items()}
        for split, t in tiles.items():
            print(f"{split}: {len(t)} tiles with all four layers in {summary['config']['data'][split]['n_aois']} AoIs", flush=True)
    except Exception as ex:  # noqa: BLE001
        fail(summary, "data", ex)
        checkpoint(summary, suffix)
        raise SystemExit(1)
    summary["timing"]["data_seconds"] = time.time() - t0
    checkpoint(summary, suffix)

    # ---- stage 2: the chips
    t0 = time.time()
    data = {}
    try:
        summary["config"]["chips"] = {}
        for split in ("test", "val"):
            cand = chip_candidates(tiles[split], split)
            sel = select_chips(cand, args.chips_per_event, smoke_chips=args.smoke_chips if args.smoke else None)
            s2, s1, lab = read_imagery(tiles[split], sel)
            aoi = np.array([x["aoi"] for x in sel])
            clear = np.array([x["clear"] >= MIN_CLEAR for x in sel], dtype=bool)
            data[split] = {"s2": s2, "s1": s1, "lab": lab, "aoi": aoi, "clear": clear, "tile": [x["tile"] for x in sel]}
            per_aoi = dict(sorted(Counter(aoi.tolist()).items()))
            summary["config"]["chips"][split] = {"candidates": len(cand), "kept": len(sel), "clear": int(clear.sum()), "per_aoi": per_aoi,
                                                 "per_aoi_clear": dict(sorted(Counter(aoi[clear].tolist()).items())),
                                                 "label_share": {c: float((lab == c).mean()) for c in (0, 1, 2, 255)}}
            print(f"{split}: {len(cand)} labelled chips from {len(tiles[split])} tiles, {len(sel)} kept after the per-AoI cap ({int(clear.sum())} clear for the S2 task), "
                  f"label share water {float(((lab == 1) | (lab == 2)).mean()):.3f} (permanent {float((lab == 1).mean()):.3f}, flooded {float((lab == 2).mean()):.3f}), ignore {float((lab == 255).mean()):.3f}", flush=True)
    except Exception as ex:  # noqa: BLE001
        fail(summary, "chips", ex)
        checkpoint(summary, suffix)
        raise SystemExit(1)
    summary["timing"]["chips_seconds"] = time.time() - t0
    checkpoint(summary, suffix)

    # ---- stage 3: heads on the val chips at shift 0, four shift maps on the test chips; features held per shift only
    t0 = time.time()
    val, te = data["val"], data["test"]
    med_vv = float(np.median(val["s1"][:, 0, :CROP, :CROP]))
    summary["config"]["median_vv_db_val"] = med_vv
    heads, p_shift, logit0, sel_te = {}, {}, {}, {}
    try:
        model = hb.load_model()
        summary["config"]["encode"] = {}
        for sensor in ("s2", "s1"):
            t1 = time.time()
            tasks = [k for k, (_, s) in TASKS.items() if s == sensor]
            sel_val = val["clear"] if sensor == "s2" else np.ones(len(val["lab"]), bool)
            f = features(model, sensor, val[sensor][sel_val], 0)
            for task in tasks:
                lab_t = task_labels(val["lab"][sel_val], task)
                heads[task] = fit_head(f, lab_t)
                y, ok = exp18.patch_labels(lab_t[:, :CROP, :CROP])
                summary["config"]["encode"][task] = {"train_chips": int(sel_val.sum()), "train_windows": int(ok.sum()), "train_water_share": float(y[ok].mean()) if ok.any() else None}
            del f
            sel_te[sensor] = te["clear"] if sensor == "s2" else np.ones(len(te["lab"]), bool)
            maps = {task: [] for task in tasks}
            for s in SHIFTS:
                f = features(model, sensor, te[sensor][sel_te[sensor]], s)
                for task in tasks:
                    p, lg = exp18.head_prob_logit(f, *heads[task])
                    maps[task].append(p)
                    if s == 0:
                        logit0[task] = lg
                del f
            for task in tasks:
                p_shift[task] = np.stack(maps[task])                         # (S, N, G, G)
            if DEV == "cuda":
                torch.cuda.empty_cache()
            summary["config"]["encode"][sensor] = {"val_chips": int(sel_val.sum()), "test_chips": int(sel_te[sensor].sum()), "seconds": time.time() - t1}
            print(f"{sensor}: heads {tasks} fitted on {int(sel_val.sum())} val chips (" + ", ".join(f"{k} {summary['config']['encode'][k]['train_windows']} windows, water share {summary['config']['encode'][k]['train_water_share']:.3f}" for k in tasks) +
                  f"); {int(sel_te[sensor].sum())} test chips x {len(SHIFTS)} shifts encoded in {time.time() - t1:.0f}s", flush=True)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
    except Exception as ex:  # noqa: BLE001
        fail(summary, "encode", ex)
        checkpoint(summary, suffix)
        raise SystemExit(1)
    summary["timing"]["encode_seconds"] = time.time() - t0
    checkpoint(summary, suffix)

    # ---- stage 4: per task, the error set, the rankers, per-event and pooled scores, the event-level rates
    errs, idx_task = {}, {}
    for task, (name, sensor) in TASKS.items():
        t0 = time.time()
        try:
            idx = np.flatnonzero(sel_te[sensor])
            idx_task[task] = idx
            lab_t = task_labels(te["lab"][idx], task)
            sig, err, ok = arm(p_shift[task], te["s2"][idx], te["s1"][idx], lab_t, med_vv)
            aois = te["aoi"][idx]
            per_event = {}
            for aoi in sorted(set(aois.tolist())):
                m = np.flatnonzero(aois == aoi)
                per_event[aoi] = score({kk: v[m] for kk, v in sig.items()}, err[m], ok[m], PRIMARY[task])
                per_event[aoi]["s2_b02_p01"] = float(np.percentile(te["s2"][idx][m][:, 0], 1))
                rows.append(row(task, aoi, per_event[aoi]))
            pooled = score(sig, err, ok, PRIMARY[task])
            pooled["s2_b02_p01"] = float(np.percentile(te["s2"][idx][:, 0], 1)) if len(idx) else None
            rows.append(row(task, "POOLED", pooled))
            across = across_events(per_event)
            errs[task] = (err, ok)
            summary["results"][task] = {"name": name, "sensor": sensor, "primary_control": PRIMARY[task], "n_chips": int(len(idx)),
                                        "per_event": per_event, "pooled": pooled, "across_events": across, "seconds": time.time() - t0}
            tp = pooled["tests"].get(PRIMARY[task]) or {}
            print(f"task {task} ({name}): {len(idx)} chips in {across['n_events']} events ({across['n_events_scored']} scored) | pooled {pooled['n_windows']} windows, "
                  f"{pooled['n_errors']} errors, window acc {pooled['accuracy']:.4f} | pooled E-AURC " + ", ".join(f"{kk} {v:.4f}" for kk, v in pooled["pooled_eaurc"].items()) +
                  f" | lead over {PRIMARY[task]} {tp.get('pooled_lead', float('nan')):+.4f} (per-chip p={tp.get('sign_p', float('nan')):.2g}, {tp.get('w')}/{tp.get('l')}/{tp.get('t')})", flush=True)
            if across["n_events_scored"]:
                print("  over events: " + ", ".join(f"{kk} beats confidence on {across[kk]['control_wins']}/{across['n_events_scored']} ({across[kk]['exception_rate']:.2f}, p={across[kk]['sign_p']:.2g})" for kk in CONTROLS) +
                      f" | median capture@5% {across['median_capture5_confidence']:.3f} ({across['median_capture5_ratio_to_random']:.1f}x random)"
                      f" | boundary-first beats confidence at 5% on {across['boundary_first_vs_confidence_capture5']['lex_wins']}/{across['n_events_scored']} events", flush=True)
            else:
                print("  no event reaches 3 errors and 3 correct windows; event-level rates undefined", flush=True)
        except Exception as ex:  # noqa: BLE001
            fail(summary, f"task {task}", ex)
        checkpoint(summary, suffix)

    # ---- stage 5: exp51's cross-tab, task A's S2-head errors against task B's S1-head errors on identical windows
    try:
        if "A" in errs and "B" in errs:
            ia, pos = idx_task["A"], {t: i for i, t in enumerate(idx_task["B"])}
            ib = np.array([pos[t] for t in ia], dtype=int)
            ea, oa = errs["A"]
            eb, ob = errs["B"][0][ib], errs["B"][1][ib]
            both = oa & ob
            aois = te["aoi"][ia]
            per = {}
            for aoi in sorted(set(aois.tolist())):
                m = both & (aois == aoi)[:, None, None]
                if m.sum() >= 50:
                    per[aoi] = e46.phi(ea[m] > 0, eb[m] > 0)
            finite = [v for v in per.values() if v is not None and np.isfinite(v)]
            summary["cross_tab_A_vs_B"] = {"n_chips": int(len(ia)), "n_windows": int(both.sum()), **e46.overlap(ea[both] > 0, eb[both] > 0),
                                           "acc_A_S2_head": float(1 - ea[both].mean()), "acc_B_S1_head": float(1 - eb[both].mean()),
                                           "per_event_phi": per, "median_within_event_phi": float(np.median(finite)) if finite else None}
            x = summary["cross_tab_A_vs_B"]
            print(f"cross-tab A (S2 head) vs B (S1 head) on {x['n_windows']} identical windows: phi {x['phi']}, P(S1 wrong | S2 wrong) {x['p_b_wrong_given_a_wrong']}, "
                  f"P(S1 wrong | S2 right) {x['p_b_wrong_given_a_right']}, acc {x['acc_A_S2_head']:.4f} vs {x['acc_B_S1_head']:.4f}, median within-event phi {x['median_within_event_phi']}", flush=True)
    except Exception as ex:  # noqa: BLE001
        fail(summary, "cross-tab", ex)
    checkpoint(summary, suffix)

    # ---- prereg
    try:
        def across(task, kk):
            return summary["results"].get(task, {}).get("across_events", {}).get(kk)

        def pooled_lead(task, kk):
            return (summary["results"].get(task, {}).get("pooled", {}).get("tests", {}).get(kk) or {}).get("pooled_lead")

        def holds(task, kk):
            a, lead = across(task, kk), pooled_lead(task, kk)
            return None if a is None or lead is None else bool(a["sign_p"] < 0.05 and lead >= MIN_LEAD)

        med = {t: summary["results"].get(t, {}).get("across_events", {}).get("median_capture5_confidence") for t in ("A", "B")}
        p3 = None if any(v is None for v in med.values()) else bool(all(v >= MIN_CAPTURE5 for v in med.values()))
        descriptive = {}
        a = across("A", CTRL_NDWI)
        if a and a["exception_rate"] is not None:
            descriptive["A_ndwi_exception_rate_below_one_third"] = {"rate": a["exception_rate"], "holds": bool(a["exception_rate"] < 1 / 3)}
        b = across("B", CTRL_NDWI)
        if b and b["exception_rate"] is not None:
            descriptive["B_ndwi_prior_beats_s1_confidence_on_majority"] = {"rate": b["exception_rate"], "holds": bool(b["exception_rate"] > 0.5)}
        for t in TASKS:
            x = summary["results"].get(t, {}).get("across_events", {}).get("boundary_first_vs_confidence_capture5")
            if x and x["share_lex_wins"] is not None:
                descriptive[f"{t}_boundary_first_beats_confidence_at_5pct_on_majority"] = {"share": x["share_lex_wins"], "holds": bool(x["share_lex_wins"] > 0.5)}
        summary["prereg"] = {"P1": holds("A", CTRL_NDWI), "P2": holds("B", CTRL_S1), "P3": p3, "P3_median_capture5": med,
                             "descriptive": descriptive, "complete": not summary["failures"]}
        summary["prereg"]["supported"] = None if any(summary["prereg"][k] is None for k in ("P1", "P2", "P3")) else bool(all(summary["prereg"][k] for k in ("P1", "P2", "P3")))
    except Exception as ex:  # noqa: BLE001
        fail(summary, "prereg", ex)

    summary["n_failures"] = len(summary["failures"])
    summary["timing"]["total_seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        keys = sorted({kk for r in rows for kk in r}, key=lambda s: (("task", "event").index(s) if s in ("task", "event") else 2, s))
        with open(os.path.join(hb.OUT, f"exp55_geoid_flood{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp55_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp55_summary{suffix}.json; prereg {summary.get('prereg')}; failures {summary['n_failures']}; {summary['timing']['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
