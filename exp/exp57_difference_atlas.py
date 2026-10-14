#!/usr/bin/env python
"""exp57: the difference atlas, oe_inferencex.compare on every pair of inferences the repository holds.

Why. The repository has measured every difference between two inferences of the same scene (shifted crops, exp42 and
exp44; v1 against v1.2, exp45 and exp47; the Sentinel-2 head against the Sentinel-1 head, exp46 and exp51; frozen
against fine-tuned, exp52; seven encoders on Ai2's embeddings, exp51; the two sensors per GEOID-Flood event, exp55),
and each time turned the difference into a ranker and graded it, or reported one phi. The difference itself, how
large it is, where it sits, whether two differences are the same set, was never the object, and the arithmetic (phi,
the corrected-and-broken cross-tab, the per-event rate) was copied from script to script. exp57 promotes that
arithmetic to oe_inferencex.compare and applies it to every pair on identical windows: label-free statistics first,
the label bridge second.

Design. A pair is two hard decisions (a, b) on identical 4-px windows with a validity mask, a tile or event id per
window, the labels, and four label-free cues of the first inference a: on a prediction boundary of a's own window
map (boundary_indicator of the hard map > 0); among the least confident 20% of the testbed's valid windows under a;
among the 20% most unstable under a sub-patch shift of a's tiling (aligned tile-phase; only where a has four crop
offsets); spectrally ambiguous (|NDWI| < 0.1). Per pair: the disagreement rate pooled and per group; each cue's share
among the disagreement windows against the agreement windows and their ratio (compare.where); the stability of the
disagreement set across three head draws (the full training set and two 80% subsets of its tiles, seeds 1 and 2)
and across pairs on the same grid (compare.stability); then, with labels, which side is right on the disagreement
windows (compare.which_side) and exp52's cross-tab of the two error maps (compare.crosstab). Sen1Floods11 pairs sit
on the exp47 W1 grid, the shift-averaged decision pooled to windows 1..14 of the offset-0 grid, on Bolivia (441
tiles) and the multi-region test split (800 tiles, exp18's sample):
  offsets    the tiling at crop offset 0 against the tilings at offsets 1, 2, 3, each pooled to the offset-0 windows
             (exp42's paint; "0 vs 2" is the pair carried into the cross-pair table); exp18's v1 feature cache
  backbones  OlmoEarth v1 against v1.2 Base (exp47's arms, v1.2 encoded here; ~oe12/.venv)
  sensors    the S2 head against the S1 head on v1 (exp46's arms A1 and A2, S1 encoded at the four offsets)
  finetune   the frozen S2 head against FT-S2 v1, fine-tuned once more with exp52's recipe (12 epochs)
  encoders   on Ai2's published Sen1Floods11 embeddings and their probe (exp51's arm D; test split, 2,419 tiles, 16 x
             16 windows per tile): olmoearth_base against galileo, croma, terramind, clay, satlas, anysat, panopticon;
             the NDWI cue from our test tiles matched by label hash; stability among the seven disagreement sets
  geoid      per GEOID-Flood event (exp55's shard subset, 55 events): the pre-event S2 head (permanent water) against
             the post-event S1 head (water after the event) on the clear chips both predicted; each error map against
             its own task label; the share of the disagreement windows that are flooded by the label (real change,
             not error). Needs the pinned ~/olmoearth_inferenceX/.venv (rasterio); runs as a second job that merges
             into the same summary (--pairs geoid).

Preregistered (one-sided, stated before the run, from what exp18/exp36 (boundary share 75% vs 21% among errors),
exp45/exp46 (backbone-swap phi 0.70, sensor phi 0.38) and exp52 already imply):
  P1  the disagreement windows of every pair are boundary-enriched: share among disagreement windows over share
      among agreement windows > 2 for offsets (0 vs 2), backbones, sensors and finetune on both testbeds, for every
      encoder pair on the test split, and on the median GEOID-Flood event.
  P2  the sensor difference and the backbone difference are different differences: phi between the S2-vs-S1
      disagreement mask and the v1-vs-v1.2 disagreement mask is below 0.5 pooled on both testbeds, and per tile the
      tiles with phi < 0.5 outnumber the rest under a one-sided exact sign test (p < 0.05; tiles with an undefined
      phi dropped).
  Falsification: P1 fails if any listed enrichment is at or below 2 (the disagreement would then sit in the interior
  as often as on the boundary, and the boundary cue would not describe it); P2 fails if either testbed's pooled phi
  is at or above 0.5 or the sign test does not reach 0.05 (the two differences would then be one set of unstable
  windows, and one comparison would stand for the other).

Inputs. exp/out/exp18_feats.npz (v1 pooled features, train / bolivia / test at the four offsets); data/floods
(valid, bolivia, test, train .pt); v1 and v1.2 Base checkpoints; $HF_HOME/paper_embeddings (exp51's cache);
$HF_HOME/geoid_flood/tree (exp55's extracted shards, re-fetched by exp55's loader if the purge removed them).
Outputs. exp/out/exp57_summary.json (per pair and testbed: the compare summary, the draws' stability, the cross-pair
stability matrices, prereg), exp/out/exp57_difference_atlas.csv (one row per pair x testbed), exp/out/exp57_masks.npz
(the hard decisions, validity and labels of every pair, so the package tests recompute the atlas from the artifact).
--pairs selects pairs; a run with a subset merges into an existing summary and mask file.
Smoke (CPU, no downloads): exp42's smoke cache for offsets and the frozen head; v1 Nano as the second backbone;
the four Bolivia tiles' S1 bands; a one-epoch fine-tune on four tiles; exp41's synthetic embeddings for the
encoders; the GEOID sample tiles from the local Hub cache (skipped with a note when absent).
"""
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp42_window_design as exp42  # noqa: E402
import exp46_shared_error_sources as e46  # noqa: E402
import exp47_served_ranker as e47  # noqa: E402
import exp49_shrug_signals as e49  # noqa: E402
import exp51_their_probe as e51  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.assess import summary as json_view  # noqa: E402
from oe_inferencex.compare import compare_inferences, crosstab, phi, stability  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.explain import top_fraction  # noqa: E402
from oe_inferencex.signals import boundary_indicator, ndwi_level  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G, SHIFTS = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS
DEV = exp18.DEV
SL = slice(1, G)                                   # windows 1..14 of the offset-0 grid, inside every tiling
TILE_PX = 64
PAIRS = ("offsets", "backbones", "sensors", "finetune", "encoders", "geoid")
TESTBEDS = ("bolivia", "test")
DRAWS = (0, 1, 2)                                  # head draws: the full training set, then 80% subsets (seeds 1, 2)
DRAW_FRACTION = 0.8
LOW_FRACTION = 0.2
NDWI_AMBIGUOUS = -0.1                              # ndwi_level = -|NDWI|; above -0.1 means |NDWI| < 0.1
MIN_ENRICHMENT, MAX_PHI = 2.0, 0.5
CROSS_PAIRS = ("offsets 0 vs 2", "backbones", "sensors", "finetune")
OFFSET_PAIR = "offsets 0 vs 2"
QUANTILES = (0.1, 0.5, 0.9)


# ----------------------------------------------------------------------------- the pair machinery
def w1(p_shift):
    """(S, N, G, G) shift maps -> the shift-averaged probability on windows 1..G-1 of the offset-0 grid, (N, G-1, G-1)."""
    return e49.w1_windows(p_shift)[:, SL, SL]


def offset_windows(p, s):
    """One tiling's (N, G, G) window map at crop offset s, pooled back to windows 1..G-1 of the offset-0 grid."""
    from oe_inferencex.signals import pool_to_windows
    return np.stack([pool_to_windows(exp42.paint(p[t], s, TILE_PX), patch=PATCH, offset=0)[SL, SL] for t in range(len(p))])


def labels_of(lab):
    y, ok = exp18.patch_labels(lab[:, :CROP, :CROP])
    return (y[:, SL, SL] > 0.5), ok[:, SL, SL]


def cues_of(a_hard, conf, ok, tile_phase=None, ndwi=None):
    """The four label-free cues of the first inference, boolean arrays of window shape."""
    cues = {"boundary": boundary_indicator(a_hard) > 0,
            "low_confidence": top_fraction(conf, LOW_FRACTION, valid=ok)}
    if tile_phase is not None:
        cues["unstable"] = top_fraction(tile_phase, LOW_FRACTION, valid=ok)
    if ndwi is not None:
        cues["ndwi_ambiguous"] = (ndwi > NDWI_AMBIGUOUS) & ok
    return cues


def ndwi_cue_input(s2, size=CROP, crop=True):
    v = np.stack([ndwi_level(t, patch=PATCH, size=size) for t in s2])
    return v[:, SL, SL] if crop else v


def quantiles(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return {"n": 0}
    out = {"n": int(v.size), "mean": float(v.mean())}
    out.update({f"q{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES})
    return out


def compact(out, keep_groups):
    """The compare summary for the artifact: JSON-safe, per-group detail replaced by its distribution unless kept
    (events are few, tiles are many)."""
    res = json_view(out)
    if res.get("per_group"):
        rates = [v["rate"] for v in res["per_group"].values()]
        res["per_group_rate"] = quantiles(rates)
        if not keep_groups:
            res["per_group"] = None
    g = res.get("graded")
    if g and g.get("per_group"):
        g["per_group_phi"] = quantiles([v["phi"] for v in g["per_group"].values()])
        g["per_group_share_corrected"] = quantiles([v["share_corrected"] for v in g["per_group"].values() if v["errors_a"] > 0])
        if not keep_groups:
            g["per_group"] = None
    return res


def compare_pair(a, b, ok, y, groups, cues, keep_groups=False):
    """compare_inferences on a pair plus the two accuracies; returns (compact summary, disagreement mask)."""
    out = compare_inferences(a, b, ok, groups=groups, labels=y, cues=cues)
    res = compact(out, keep_groups)
    res["accuracy_a"] = float((a == y)[ok].mean()) if ok.any() else None
    res["accuracy_b"] = float((b == y)[ok].mean()) if ok.any() else None
    return res, out["arrays"]["disagree"]


def draw_indices(n, draw, rng_seed):
    """The training tiles of a head draw: all of them for draw 0, a DRAW_FRACTION subset for the others."""
    if draw == 0:
        return np.arange(n)
    return np.sort(np.random.default_rng(rng_seed).choice(n, int(round(DRAW_FRACTION * n)), replace=False))


def fit_head(feats, lab, idx):
    """exp47.head_from on the chosen training tiles (seed 0 in torch; the draw is the tile subset)."""
    y, ok = exp18.patch_labels(lab[idx, :CROP, :CROP])
    f = feats[idx]
    torch.manual_seed(0)
    return train_logistic_head(torch.tensor(f.reshape(-1, f.shape[-1])[ok.flatten()]), y.flatten()[ok.flatten()])


def probs(feats_by_shift, head):
    """{shift: (N, G, G, D)} -> (S, N, G, G) probability maps under one head."""
    return np.stack([exp18.head_prob_logit(feats_by_shift[s], *head)[0] for s in SHIFTS])


def row_of(pair, testbed, res, extra=None):
    w = res.get("where") or {}
    g = res.get("graded") or {}
    ct, ws = g.get("crosstab") or {}, g.get("which_side") or {}
    r = {"pair": pair, "testbed": testbed, "n_windows": res["n_windows"], "n_disagree": res["n_disagree"], "rate": res["disagreement_rate"],
         "per_group_median_rate": (res.get("per_group_rate") or {}).get("q50"),
         **{f"enrichment {k}": (w.get(k) or {}).get("enrichment") for k in ("boundary", "low_confidence", "unstable", "ndwi_ambiguous")},
         "share_a_right": ws.get("share_a_right"), "share_b_right": ws.get("share_b_right"), "share_neither": (ws["neither"] / ws["n_disagree"]) if ws.get("n_disagree") else None,
         "accuracy_a": res.get("accuracy_a"), "accuracy_b": res.get("accuracy_b"),
         "errors_phi": ct.get("phi"), "share_a_errors_corrected_by_b": ct.get("share_corrected"), "broken_by_b": ct.get("broken"),
         "stability_across_draws_median_phi": (res.get("stability_across_draws") or {}).get("median_pairwise_phi")}
    if extra:
        r.update(extra)
    return r


def fmt(v, spec=".2f"):
    """Progress-line number: NaN and None print as such instead of raising."""
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def enr(res, cue):
    return fmt(((res.get("where") or {}).get(cue) or {}).get("enrichment"))


def fail(summary, part, ex):
    summary["failures"].append({"part": part, "error": repr(ex), "traceback": traceback.format_exc()})
    print(f"{part} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)


def checkpoint(summary, masks, suffix):
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp57_summary{suffix}.partial.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    np.savez_compressed(os.path.join(hb.OUT, f"exp57_masks{suffix}.partial.npz"), **masks)


# ----------------------------------------------------------------------------- Sen1Floods11: shared inputs
def load_floods(args, summary):
    """The three splits with S1 (exp51.load_ours; same tile draws as exp47 / exp52) and exp18's v1 feature cache."""
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    n_tr = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES
    tr = e51.load_ours(floods_dir, "valid", n_tr, seed=0)                        # (s1, s2, lab)
    bo = e51.load_ours(floods_dir, "bolivia", args.smoke_tiles if args.smoke else None, seed=1)
    if args.smoke:
        bo = tuple(a[:args.smoke_tiles] for a in e51.load_ours(floods_dir, "bolivia", None))   # the cache holds the first tiles
        te = None
    else:
        te = e51.load_ours(floods_dir, "test", exp18.N_TEST_TILES, seed=1)
    splits = {"train": tr, "bolivia": bo}
    if te is not None:
        splits["test"] = te
    cache = np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz" if args.smoke else "exp18_feats.npz"))
    key = {"train": "tr_base", "bolivia": "bolivia_base", "test": "test_base"}
    feats = {"train": {0: np.asarray(cache["tr_base"], dtype=np.float32)}}
    if len(feats["train"][0]) != len(tr[0]):
        raise RuntimeError(f"train cache holds {len(feats['train'][0])} tiles, the split {len(tr[0])}")
    for name in [k for k in splits if k != "train"]:
        feats[name] = {}
        for s in SHIFTS:
            k = f"{key[name]}{s}"
            if k not in cache.files or len(cache[k]) != len(splits[name][0]):
                raise RuntimeError(f"cache key {k} missing or of the wrong length")
            feats[name][s] = np.asarray(cache[k], dtype=np.float32)
    summary["config"]["floods"] = {k: int(len(v[0])) for k, v in splits.items()}
    return floods_dir, splits, feats


def encode_s2(model, s2, shifts=SHIFTS):
    return {s: np.asarray(exp18.embed(model, s2, s)[0], dtype=np.float32) for s in shifts}


def encode_s1(model, s1, shifts=SHIFTS):
    return {s: np.asarray(e46.embed_s1(model, s1[:, :, s:s + CROP, s:s + CROP]), dtype=np.float32) for s in shifts}


def testbed_inputs(splits, name, feats_v1):
    """Labels, tile ids, NDWI and the v1 S2 head's four shift maps' scaffolding for one testbed."""
    s1, s2, lab = splits[name]
    y, ok = labels_of(lab)
    return {"y": y, "ok": ok, "groups": np.arange(len(lab)), "ndwi": ndwi_cue_input(s2), "s2": s2, "s1": s1, "lab": lab, "feats": feats_v1[name]}


def draws_of(train_feats_shift0, train_lab, n_draws=len(DRAWS)):
    """The head draws as tile-index sets; the head itself is fitted where the features are known."""
    n = len(train_lab)
    return {d: draw_indices(n, d, rng_seed=d) for d in DRAWS[:n_draws]}


def with_draws(summary, pair, testbed, res, masks_by_draw):
    """Attach the stability across head draws to a pair's summary."""
    if len(masks_by_draw) > 1:
        res["stability_across_draws"] = stability({f"draw {d}": m for d, m in masks_by_draw.items()})
    return res


# ----------------------------------------------------------------------------- the pairs
def pair_offsets(ctx, summary, rows, masks, cross):
    """The tiling at offset 0 against the tilings at offsets 1..3 on the same windows, under three v1 head draws."""
    heads = {d: fit_head(ctx["feats"]["train"][0], ctx["splits"]["train"][2], idx) for d, idx in ctx["draws"].items()}
    for name in ctx["testbeds"]:
        t = ctx["inputs"][name]
        res_by_pair, mask_by_draw = {}, {}
        for d, head in heads.items():
            p = probs(t["feats"], head)                                        # (S, N, G, G)
            a = offset_windows(p[0], 0)
            a_hard = a > 0.5
            conf = -np.abs(a - 0.5)
            cues = cues_of(a_hard, conf, t["ok"], tile_phase=exp18.aligned_tile_phase(p)[:, SL, SL], ndwi=t["ndwi"])
            for s in SHIFTS[1:]:
                b_hard = offset_windows(p[s], s) > 0.5
                key = f"offsets 0 vs {s}"
                res, mask = compare_pair(a_hard, b_hard, t["ok"], t["y"], t["groups"], cues)
                if d == 0:
                    res_by_pair[key] = res
                    masks[f"{name}_offset{s}"] = b_hard
                    masks[f"{name}_offset0"], masks[f"{name}_ok"], masks[f"{name}_y"] = a_hard, t["ok"], t["y"]
                    if key == OFFSET_PAIR:
                        cross[name][key] = mask
                if key == OFFSET_PAIR:
                    mask_by_draw[d] = mask
        res_by_pair[OFFSET_PAIR] = with_draws(summary, "offsets", name, res_by_pair[OFFSET_PAIR], mask_by_draw)
        summary["results"]["offsets"][name] = res_by_pair
        for key, res in res_by_pair.items():
            rows.append(row_of(key, name, res))
        r = res_by_pair[OFFSET_PAIR]
        print(f"offsets/{name}: 0 vs 1/2/3 disagree on " + ", ".join(f"{res_by_pair[f'offsets 0 vs {s}']['disagreement_rate']:.4f}" for s in SHIFTS[1:]) +
              f" of {r['n_windows']} windows | 0 vs 2: boundary {enr(r, 'boundary')}x, low-confidence {enr(r, 'low_confidence')}x, "
              f"unstable {enr(r, 'unstable')}x, NDWI {enr(r, 'ndwi_ambiguous')}x | a right {fmt(r['graded']['which_side']['share_a_right'], '.3f')}, "
              f"b right {fmt(r['graded']['which_side']['share_b_right'], '.3f')} | across draws phi {fmt((r.get('stability_across_draws') or {}).get('median_pairwise_phi'), '.3f')}", flush=True)


def second_backbone(args):
    """v1.2 Base in the full run; v1 Nano as the smoke's stand-in."""
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
    if args.smoke:
        return load_model_from_id(ModelID.OLMOEARTH_V1_NANO).to(DEV).eval().float(), "v1 Nano (smoke stand-in)"
    if not hasattr(ModelID, "OLMOEARTH_V1_2_BASE"):
        raise RuntimeError("OLMOEARTH_V1_2_BASE unavailable: run the backbones pair with ~/oe12/.venv")
    return load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(DEV).eval().float(), "v1.2 Base"


def pair_backbones(ctx, summary, rows, masks, cross, args):
    """OlmoEarth v1 against the second backbone, each read by its own head, W1 decisions on identical windows."""
    t0 = time.time()
    model, label = second_backbone(args)
    tr_feats = encode_s2(model, ctx["splits"]["train"][1], shifts=(0,))[0]
    feats2 = {name: encode_s2(model, ctx["splits"][name][1]) for name in ctx["testbeds"]}
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    summary["results"]["backbones"]["second"] = label
    summary["results"]["backbones"]["encode_seconds"] = time.time() - t0
    for name in ctx["testbeds"]:
        t = ctx["inputs"][name]
        mask_by_draw, res0 = {}, None
        for d, idx in ctx["draws"].items():
            h1 = fit_head(ctx["feats"]["train"][0], ctx["splits"]["train"][2], idx)
            h2 = fit_head(tr_feats, ctx["splits"]["train"][2], idx)
            p1, p2 = probs(t["feats"], h1), probs(feats2[name], h2)
            a, b = w1(p1), w1(p2)
            a_hard, b_hard = a > 0.5, b > 0.5
            cues = cues_of(a_hard, -np.abs(a - 0.5), t["ok"], tile_phase=exp18.aligned_tile_phase(p1)[:, SL, SL], ndwi=t["ndwi"])
            res, mask = compare_pair(a_hard, b_hard, t["ok"], t["y"], t["groups"], cues)
            mask_by_draw[d] = mask
            if d == 0:
                res0 = res
                masks[f"{name}_v1"], masks[f"{name}_v1_2"] = a_hard, b_hard
                masks[f"{name}_ok"], masks[f"{name}_y"] = t["ok"], t["y"]
                cross[name]["backbones"] = mask
        res0 = with_draws(summary, "backbones", name, res0, mask_by_draw)
        summary["results"]["backbones"][name] = res0
        rows.append(row_of("backbones", name, res0))
        r = res0
        print(f"backbones/{name} (v1 vs {label}): disagree {fmt(r['disagreement_rate'], '.4f')} of {r['n_windows']} | boundary {enr(r, 'boundary')}x, "
              f"low-confidence {enr(r, 'low_confidence')}x, unstable {enr(r, 'unstable')}x, NDWI {enr(r, 'ndwi_ambiguous')}x | "
              f"a right {fmt(r['graded']['which_side']['share_a_right'], '.3f')}, b right {fmt(r['graded']['which_side']['share_b_right'], '.3f')}, errors phi {fmt(r['graded']['crosstab']['phi'], '.3f')} | "
              f"across draws phi {fmt((r.get('stability_across_draws') or {}).get('median_pairwise_phi'), '.3f')}", flush=True)


def pair_sensors(ctx, summary, rows, masks, cross):
    """The S2 head against the S1 head on v1 (exp46's A1 and A2), both as W1 decisions on identical windows."""
    t0 = time.time()
    model = hb.load_model()
    tr_s1 = encode_s1(model, ctx["splits"]["train"][0], shifts=(0,))[0]
    feats1 = {name: encode_s1(model, ctx["splits"][name][0]) for name in ctx["testbeds"]}
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    summary["results"]["sensors"]["encode_seconds"] = time.time() - t0
    ctx["frozen_s2_head"] = fit_head(ctx["feats"]["train"][0], ctx["splits"]["train"][2], ctx["draws"][0])
    for name in ctx["testbeds"]:
        t = ctx["inputs"][name]
        mask_by_draw, res0 = {}, None
        for d, idx in ctx["draws"].items():
            h2 = ctx["frozen_s2_head"] if d == 0 else fit_head(ctx["feats"]["train"][0], ctx["splits"]["train"][2], idx)
            h1 = fit_head(tr_s1, ctx["splits"]["train"][2], idx)
            p2, p1 = probs(t["feats"], h2), probs(feats1[name], h1)
            a, b = w1(p2), w1(p1)
            a_hard, b_hard = a > 0.5, b > 0.5
            cues = cues_of(a_hard, -np.abs(a - 0.5), t["ok"], tile_phase=exp18.aligned_tile_phase(p2)[:, SL, SL], ndwi=t["ndwi"])
            res, mask = compare_pair(a_hard, b_hard, t["ok"], t["y"], t["groups"], cues)
            mask_by_draw[d] = mask
            if d == 0:
                res0 = res
                masks[f"{name}_s2"], masks[f"{name}_s1"] = a_hard, b_hard
                masks[f"{name}_ok"], masks[f"{name}_y"] = t["ok"], t["y"]
                cross[name]["sensors"] = mask
                ctx["frozen"][name] = {"hard": a_hard, "cues": cues}
        res0 = with_draws(summary, "sensors", name, res0, mask_by_draw)
        summary["results"]["sensors"][name] = res0
        rows.append(row_of("sensors", name, res0))
        r = res0
        print(f"sensors/{name} (S2 head vs S1 head): disagree {fmt(r['disagreement_rate'], '.4f')} of {r['n_windows']} | boundary {enr(r, 'boundary')}x, "
              f"low-confidence {enr(r, 'low_confidence')}x, unstable {enr(r, 'unstable')}x, NDWI {enr(r, 'ndwi_ambiguous')}x | "
              f"a right {fmt(r['graded']['which_side']['share_a_right'], '.3f')}, b right {fmt(r['graded']['which_side']['share_b_right'], '.3f')}, errors phi {fmt(r['graded']['crosstab']['phi'], '.3f')} | "
              f"across draws phi {fmt((r.get('stability_across_draws') or {}).get('median_pairwise_phi'), '.3f')}", flush=True)


def pair_finetune(ctx, summary, rows, masks, cross, args, floods_dir):
    """The frozen S2 head against FT-S2 v1, fine-tuned once more with exp52's recipe."""
    import exp52_finetune as e52
    t0 = time.time()
    n = args.smoke_tiles if args.smoke else None
    tr_s1, tr_s2, tr_lab = e51.load_ours(floods_dir, "valid" if args.smoke else "train", n, seed=3)
    va_s1, va_s2, va_lab = e51.load_ours(floods_dir, "valid", n, seed=0)
    tr = (*e52.prep(tr_s2, tr_s1, 0), tr_lab[:, :CROP, :CROP])
    va = (*e52.prep(va_s2, va_s1, 0), va_lab[:, :CROP, :CROP])
    epochs = 1 if args.smoke else e52.EPOCHS
    model = hb.load_model()
    ft, history, best = e52.finetune(model, ["s2"], tr, va, epochs, log=lambda m: print(f"  FT-S2 v1{m}", flush=True))
    summary["results"]["finetune"]["train"] = {"train_tiles": int(len(tr_s2)), "valid_tiles": int(len(va_s2)), "epochs": epochs, "best_val_miou": best,
                                               "train_seconds": time.time() - t0, "history": history}
    if "frozen_s2_head" not in ctx:
        ctx["frozen_s2_head"] = fit_head(ctx["feats"]["train"][0], ctx["splits"]["train"][2], ctx["draws"][0])
    for name in ctx["testbeds"]:
        t = ctx["inputs"][name]
        if name in ctx["frozen"]:
            a_hard, cues = ctx["frozen"][name]["hard"], ctx["frozen"][name]["cues"]
        else:
            p2 = probs(t["feats"], ctx["frozen_s2_head"])
            a = w1(p2)
            a_hard = a > 0.5
            cues = cues_of(a_hard, -np.abs(a - 0.5), t["ok"], tile_phase=exp18.aligned_tile_phase(p2)[:, SL, SL], ndwi=t["ndwi"])
            ctx["frozen"][name] = {"hard": a_hard, "cues": cues}
        pshift = []
        for s in SHIFTS:
            x2, x1 = e52.prep(t["s2"], t["s1"], s)
            p = e52.predict(ft, x2, x1)
            pshift.append(p.reshape(len(p), G, PATCH, G, PATCH).mean(axis=(2, 4)))
        b_hard = w1(np.stack(pshift)) > 0.5
        res, mask = compare_pair(a_hard, b_hard, t["ok"], t["y"], t["groups"], cues)
        summary["results"]["finetune"][name] = res
        masks[f"{name}_frozen"], masks[f"{name}_ft"] = a_hard, b_hard
        masks[f"{name}_ok"], masks[f"{name}_y"] = t["ok"], t["y"]
        cross[name]["finetune"] = mask
        rows.append(row_of("finetune", name, res))
        r = res
        print(f"finetune/{name} (frozen S2 head vs FT-S2 v1): disagree {fmt(r['disagreement_rate'], '.4f')} of {r['n_windows']} | boundary {enr(r, 'boundary')}x, "
              f"low-confidence {enr(r, 'low_confidence')}x, unstable {enr(r, 'unstable')}x, NDWI {enr(r, 'ndwi_ambiguous')}x | "
              f"a right {fmt(r['graded']['which_side']['share_a_right'], '.3f')}, b right {fmt(r['graded']['which_side']['share_b_right'], '.3f')} | corrects {fmt(r['graded']['crosstab']['share_corrected'], '.3f')}, "
              f"breaks {r['graded']['crosstab']['broken']}, errors phi {fmt(r['graded']['crosstab']['phi'], '.3f')}", flush=True)
    del ft, model
    if DEV == "cuda":
        torch.cuda.empty_cache()


def their_decision(model, cache_dir, args):
    """exp51's arm D for one encoder: (window probability (N, 16, 16), majority label, validity, confidence, label keys)."""
    if args.smoke:
        import exp41_two_view_disagreement as exp41
        emb_tr, lab_tr = exp41.synthetic(model, "sen1floods11", "train")
        emb_te, lab_te = exp41.synthetic(model, "sen1floods11", "test")
    else:
        emb_tr, lab_tr = e51.load_theirs(model, "train", cache_dir)
        emb_te, lab_te = e51.load_theirs(model, "test", cache_dir)
    pp = lab_tr.shape[-1] // emb_tr.shape[1]
    probe = e51.train_probe(emb_tr, lab_tr, pp)
    p = e51.predict_pixels(probe, emb_te, pp, tuple(lab_te.shape[-2:]))
    lab_np = lab_te.numpy()
    wp, y, ok, _, conf = e51.window_view(p, lab_np, PATCH)
    return {"wp": wp, "y": y, "ok": ok, "conf": conf, "keys": e51.label_keys(lab_np), "token_grid": list(emb_tr.shape[1:3]), "n_test": int(len(emb_te))}


def pair_encoders(ctx, summary, rows, masks, args, floods_dir, others):
    """olmoearth_base against each other encoder on Ai2's embeddings and their probe (exp51's arm D), 16 x 16 windows."""
    cache_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "paper_embeddings")
    t0 = time.time()
    base = their_decision(e51.THEIRS, cache_dir, args)
    summary["results"]["encoders"]["olmoearth_base"] = {"seconds": time.time() - t0, "token_grid": base["token_grid"], "test_tiles": base["n_test"]}
    a_full = base["wp"] > 0.5
    ndwi = None
    if not args.smoke:
        try:
            _, s2_te, lab_te = e51.load_ours(floods_dir, "test", None, seed=2)
            pos = {k: i for i, k in enumerate(e51.label_keys(lab_te))}
            match = np.array([pos.get(k, -1) for k in base["keys"]])
            if (match >= 0).all():
                ndwi = ndwi_cue_input(s2_te[match], size=TILE_PX, crop=False)
            summary["results"]["encoders"]["olmoearth_base"]["ndwi_tiles_matched"] = int((match >= 0).sum())
            del s2_te
        except Exception as ex:  # noqa: BLE001
            fail(summary, "encoders NDWI cue", ex)
    cues = cues_of(a_full, base["conf"], base["ok"], ndwi=ndwi)
    masks["encoders_olmoearth_base"], masks["encoders_ok"], masks["encoders_y"] = a_full, base["ok"], base["y"]
    pos0 = {k: i for i, k in enumerate(base["keys"])}
    groups = np.arange(len(a_full))
    per_model_masks = {}
    for model in others:
        t1 = time.time()
        try:
            d = their_decision(model, cache_dir, args)
            m = np.array([pos0.get(k, -1) for k in d["keys"]])
            mi = np.flatnonzero(m >= 0)
            b_full = np.zeros_like(a_full)
            ok_b = np.zeros_like(base["ok"])
            b_full[m[mi]], ok_b[m[mi]] = d["wp"][mi] > 0.5, d["ok"][mi]
            ok = base["ok"] & ok_b
            res, mask = compare_pair(a_full, b_full, ok, base["y"], groups, cues)
            res["seconds"] = time.time() - t1
            res["aligned_tiles"] = int(len(mi))
            res["token_grid"] = d["token_grid"]
            summary["results"]["encoders"][model] = res
            masks[f"encoders_{model}"], masks[f"encoders_ok_{model}"] = b_full, ok
            per_model_masks[model] = mask
            rows.append(row_of(f"encoders olmoearth_base vs {model}", "their test split", res))
            print(f"encoders/{model} vs olmoearth_base: disagree {fmt(res['disagreement_rate'], '.4f')} of {res['n_windows']} ({len(mi)} tiles) | boundary {enr(res, 'boundary')}x, "
                  f"low-confidence {enr(res, 'low_confidence')}x" + (f", NDWI {enr(res, 'ndwi_ambiguous')}x" if "ndwi_ambiguous" in res["where"] else "") +
                  f" | a right {fmt(res['graded']['which_side']['share_a_right'], '.3f')}, b right {fmt(res['graded']['which_side']['share_b_right'], '.3f')}, errors phi {fmt(res['graded']['crosstab']['phi'], '.3f')}", flush=True)
        except Exception as ex:  # noqa: BLE001
            fail(summary, f"encoders {model}", ex)
    if len(per_model_masks) > 1:
        summary["results"]["encoders"]["stability_across_encoders"] = stability(per_model_masks)
        print(f"encoders: pairwise phi of the seven disagreement sets, median {fmt(summary['results']['encoders']['stability_across_encoders']['median_pairwise_phi'], '.3f')}", flush=True)


def geoid_tiles(args, data_dir):
    """exp55's tile tables: the extracted shard tree in the full run, the sample tree of the local Hub cache in the smoke."""
    import exp55_geoid_flood as e55
    if not args.smoke:
        root = os.path.join(data_dir, "tree")
        if not all(os.path.isdir(os.path.join(root, s)) for s in ("test", "val")):
            files = e55.hub_files()
            for split, n in (("test", 4), ("val", 2)):
                e55.fetch_split(files, split, n, data_dir, root)
        return {split: e55.scan_split(root, split) for split in ("test", "val")}
    import glob
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snaps = glob.glob(os.path.join(hf_home, "hub", "datasets--links-ads--geoid-flood", "snapshots", "*", "sample", "geoid-flood"))
    if not snaps:
        return None
    paths = {}
    for dirpath, _, fnames in os.walk(snaps[0]):
        for fn in fnames:
            if fn.endswith(".tif"):
                paths[fn] = os.path.join(dirpath, fn)
    table = e55.tile_table(paths)
    out = {}
    for aoi, split in e55.SMOKE_AOIS.items():
        cand = {k: v for k, v in table.items() if v["aoi"] == aoi}
        out[split] = dict(sorted(cand.items())[:args.smoke_tiles])
    return out


def pair_geoid(summary, rows, masks, args):
    """Per event: the pre-event S2 head against the post-event S1 head on the clear chips both predicted (exp55's stage 5)."""
    import exp55_geoid_flood as e55
    t0 = time.time()
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    data_dir = os.path.join(hf_home, "geoid_flood")
    tiles = geoid_tiles(args, data_dir)
    if tiles is None:
        summary["results"]["geoid"]["note"] = "smoke: GEOID sample tiles not in the local Hub cache; pair skipped"
        print("geoid: sample tiles not cached locally, skipped", flush=True)
        return
    data = {}
    for split in ("test", "val"):
        cand = e55.chip_candidates(tiles[split], split)
        sel = e55.select_chips(cand, 400, smoke_chips=4 if args.smoke else None)
        s2, s1, lab = e55.read_imagery(tiles[split], sel)
        data[split] = {"s2": s2, "s1": s1, "lab": lab, "aoi": np.array([x["aoi"] for x in sel]), "clear": np.array([x["clear"] >= e55.MIN_CLEAR for x in sel], bool)}
    val, te = data["val"], data["test"]
    summary["results"]["geoid"]["chips"] = {s: {"kept": int(len(d["lab"])), "clear": int(d["clear"].sum()), "events": int(len(set(d["aoi"].tolist())))} for s, d in data.items()}
    model = hb.load_model()
    heads, p_shift = {}, {}
    for task, sensor in (("A", "s2"), ("B", "s1")):
        sel_val = val["clear"] if sensor == "s2" else np.ones(len(val["lab"]), bool)
        heads[task] = e55.fit_head(e55.features(model, sensor, val[sensor][sel_val], 0), e55.task_labels(val["lab"][sel_val], task))
    idx_a = np.flatnonzero(te["clear"])                                       # task A predicts the clear chips only
    p_shift["A"] = np.stack([exp18.head_prob_logit(e55.features(model, "s2", te["s2"][idx_a], s), *heads["A"])[0] for s in SHIFTS])
    p_shift["B"] = np.stack([exp18.head_prob_logit(e55.features(model, "s1", te["s1"][idx_a], s), *heads["B"])[0] for s in SHIFTS])
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    summary["results"]["geoid"]["encode_seconds"] = time.time() - t0
    lab = te["lab"][idx_a]
    ya, oka = labels_of(e55.task_labels(lab, "A"))
    yb, okb = labels_of(e55.task_labels(lab, "B"))
    ok = oka & okb
    a, b = w1(p_shift["A"]), w1(p_shift["B"])
    a_hard, b_hard = a > 0.5, b > 0.5
    events = te["aoi"][idx_a]
    cues = cues_of(a_hard, -np.abs(a - 0.5), ok, tile_phase=exp18.aligned_tile_phase(p_shift["A"])[:, SL, SL], ndwi=ndwi_cue_input(te["s2"][idx_a]))
    out = compare_inferences(a_hard, b_hard, ok, groups=events, cues=cues)          # label-free, per event
    res = compact(out, keep_groups=True)
    mask = out["arrays"]["disagree"]
    # the label bridge, each error map against its own task label (exp55's cross-tab), and what the disagreement is
    err_a, err_b = a_hard != ya, b_hard != yb
    flooded = yb & ~ya                                                          # water after the event that is not permanent water
    res["graded"] = {"crosstab_own_labels": crosstab(err_a, err_b, ok),
                     "disagreement_is": {"n_disagree": int(mask.sum()),
                                         "share_flooded_by_label": float(flooded[mask].mean()) if mask.any() else None,
                                         "share_a_wrong_on_its_task": float(err_a[mask].mean()) if mask.any() else None,
                                         "share_b_wrong_on_its_task": float(err_b[mask].mean()) if mask.any() else None,
                                         "share_neither_wrong": float((~err_a & ~err_b)[mask].mean()) if mask.any() else None},
                     "accuracy_a_own_task": float((~err_a)[ok].mean()), "accuracy_b_own_task": float((~err_b)[ok].mean())}
    # per event: the boundary enrichment and the cross-tab, then their distributions
    per_event = {}
    for ev in sorted(set(events.tolist())):
        m = ok & (events == ev)[:, None, None]
        if m.sum() < 50:
            continue
        o = compare_inferences(a_hard, b_hard, m, cues={k: v for k, v in cues.items()})
        per_event[ev] = {"n_windows": o["n_windows"], "rate": o["disagreement_rate"],
                         "enrichment": {k: v["enrichment"] for k, v in o["where"].items()},
                         "crosstab_own_labels": crosstab(err_a, err_b, m),
                         "share_flooded_by_label": float(flooded[o["arrays"]["disagree"]].mean()) if o["n_disagree"] else None}
    res["per_event"] = per_event
    res["across_events"] = {"n_events": len(per_event),
                            "rate": quantiles([v["rate"] for v in per_event.values()]),
                            "boundary_enrichment": quantiles([v["enrichment"]["boundary"] for v in per_event.values()]),
                            "share_events_boundary_enrichment_above_2": float(np.mean([v["enrichment"]["boundary"] > MIN_ENRICHMENT for v in per_event.values() if np.isfinite(v["enrichment"]["boundary"])])) if per_event else None,
                            "errors_phi": quantiles([v["crosstab_own_labels"]["phi"] for v in per_event.values()]),
                            "share_flooded_by_label": quantiles([v["share_flooded_by_label"] for v in per_event.values()])}
    summary["results"]["geoid"].update(res)
    summary["results"]["geoid"]["seconds"] = time.time() - t0
    masks.update({"geoid_s2": a_hard, "geoid_s1": b_hard, "geoid_ok": ok, "geoid_y_permanent": ya, "geoid_y_after": yb, "geoid_event": events.astype("U")})
    rows.append(row_of("geoid S2 head vs S1 head", "GEOID-Flood pooled", res, extra={"errors_phi": res["graded"]["crosstab_own_labels"]["phi"], "share_a_right": None, "share_b_right": None}))
    ae = res["across_events"]
    di = res["graded"]["disagreement_is"]
    print(f"geoid (pre-event S2 head vs post-event S1 head, {len(idx_a)} clear chips, {ae['n_events']} events): disagree {fmt(res['disagreement_rate'], '.4f')} of {res['n_windows']} | "
          f"boundary {enr(res, 'boundary')}x pooled, median event {fmt(ae['boundary_enrichment'].get('q50'))}x, above 2 on {fmt(ae['share_events_boundary_enrichment_above_2'])} of events | "
          f"errors phi {fmt(res['graded']['crosstab_own_labels']['phi'], '.3f')}, median event {fmt(ae['errors_phi'].get('q50'), '.3f')} | "
          f"of the disagreement windows {fmt(di['share_flooded_by_label'], '.3f')} are flooded by the label, "
          f"{fmt(di['share_a_wrong_on_its_task'], '.3f')} S2 wrong, {fmt(di['share_b_wrong_on_its_task'], '.3f')} S1 wrong", flush=True)


# ----------------------------------------------------------------------------- cross-pair stability and prereg
def cross_pair_tables(summary, cross, masks):
    """Pairwise phi between the disagreement sets of the Sen1Floods11 pairs on each testbed."""
    for name, by_pair in cross.items():
        keep = {k: v for k, v in by_pair.items() if v is not None}
        if len(keep) > 1:
            summary["results"]["cross_pairs"][name] = stability(keep)
            print(f"cross-pair stability on {name}: " + ", ".join(f"{a} x {b} {phi(keep[a], keep[b]):.3f}" for i, a in enumerate(keep) for b in list(keep)[i + 1:]), flush=True)
        if "sensors" in keep and "backbones" in keep:
            ok = masks.get(f"{name}_ok")
            groups = np.arange(len(ok)) if ok is not None else None
            pooled = phi(keep["sensors"], keep["backbones"])
            per_tile = [phi(keep["sensors"][t], keep["backbones"][t]) for t in range(len(keep["sensors"]))]
            finite = [v for v in per_tile if np.isfinite(v)]
            w, l, t_ = wins_losses_ties([MAX_PHI - v for v in finite])
            summary["results"]["cross_pairs"].setdefault("sensors_vs_backbones", {})[name] = {
                "pooled_phi": pooled, "n_tiles_defined": len(finite), "tiles_below": w, "tiles_at_or_above": l, "ties": t_,
                "sign_p": sign_test(w, l, "greater"), "per_tile_phi": quantiles(finite), "groups": None if groups is None else int(len(groups))}


def prereg(summary):
    R = summary["results"]
    p1, p1_detail = [], {}
    for pair, key in (("offsets", OFFSET_PAIR), ("backbones", None), ("sensors", None), ("finetune", None)):
        for name in TESTBEDS:
            r = R.get(pair, {}).get(name)
            if r and key:
                r = r.get(key)
            if r and r.get("where"):
                v = r["where"]["boundary"]["enrichment"]
                p1_detail[f"{pair}/{name}"] = v
                p1.append(v is not None and np.isfinite(v) and v > MIN_ENRICHMENT)
    for model, r in R.get("encoders", {}).items():
        if isinstance(r, dict) and r.get("where"):
            v = r["where"]["boundary"]["enrichment"]
            p1_detail[f"encoders/{model}"] = v
            p1.append(v is not None and np.isfinite(v) and v > MIN_ENRICHMENT)
    ge = R.get("geoid", {}).get("across_events")
    if ge and ge.get("boundary_enrichment", {}).get("n"):
        v = ge["boundary_enrichment"]["q50"]
        p1_detail["geoid/median event"] = v
        p1.append(v > MIN_ENRICHMENT)
    expected = 2 * 4 + len(summary["config"]["others"]) + 1
    p2, p2_detail = [], {}
    for name in TESTBEDS:
        x = R.get("cross_pairs", {}).get("sensors_vs_backbones", {}).get(name)
        if x:
            p2_detail[name] = {"pooled_phi": x["pooled_phi"], "sign_p": x["sign_p"], "tiles_below": x["tiles_below"], "tiles_at_or_above": x["tiles_at_or_above"]}
            p2.append(np.isfinite(x["pooled_phi"]) and x["pooled_phi"] < MAX_PHI and x["sign_p"] < 0.05)
    complete = len(p1) == expected and len(p2) == 2 and not summary["failures"]
    summary["prereg"] = {"P1": (all(p1) if p1 else None), "P1_detail": p1_detail, "P1_n_tested": len(p1), "P1_n_expected": expected,
                         "P2": (all(p2) if len(p2) == 2 else None), "P2_detail": p2_detail,
                         "min_enrichment": MIN_ENRICHMENT, "max_phi": MAX_PHI, "complete": bool(complete)}
    summary["n_failures"] = len(summary["failures"])
    print(f"prereg: P1 {summary['prereg']['P1']} ({len(p1)}/{expected} enrichments tested, min {fmt(min([v for v in p1_detail.values() if v is not None and np.isfinite(v)], default=float('nan')))}) | "
          f"P2 {summary['prereg']['P2']} {p2_detail} | complete {complete}", flush=True)


# ----------------------------------------------------------------------------- main
def merge_previous(summary, masks, rows, suffix, pairs):
    """A run with a subset of the pairs keeps the other pairs' results, masks and rows from the existing artifact."""
    sp = os.path.join(hb.OUT, f"exp57_summary{suffix}.json")
    mp = os.path.join(hb.OUT, f"exp57_masks{suffix}.npz")
    cp = os.path.join(hb.OUT, f"exp57_difference_atlas{suffix}.csv")
    if set(pairs) == set(PAIRS) or not os.path.exists(sp):
        return
    old = json.load(open(sp))
    kept = [p for p in PAIRS if p not in pairs and p in old.get("results", {})]
    for p in kept:
        summary["results"][p] = old["results"][p]
    for k in ("cross_pairs",):
        if k in old.get("results", {}) and not any(p in pairs for p in ("offsets", "backbones", "sensors", "finetune")):
            summary["results"][k] = old["results"][k]
    summary["config"]["jobs"] = old.get("config", {}).get("jobs", []) + summary["config"]["jobs"]
    summary["failures"] = [f for f in old.get("failures", []) if not any(f["part"].startswith(p) for p in pairs)] + summary["failures"]
    if os.path.exists(mp):
        z = np.load(mp)
        prefixes = {"offsets": ("bolivia_offset", "test_offset"), "backbones": ("bolivia_v1", "test_v1"), "sensors": ("bolivia_s", "test_s"),
                    "finetune": ("bolivia_f", "test_f"), "encoders": ("encoders_",), "geoid": ("geoid_",)}
        drop = tuple(x for p in pairs for x in prefixes[p])
        for k in z.files:
            if not k.startswith(drop) and k not in masks:
                masks[k] = z[k]
    if os.path.exists(cp):
        with open(cp) as f:
            for r in csv.DictReader(f):
                if not any(r["pair"].startswith(p) for p in pairs):
                    rows.append(r)
    summary["config"]["merged_from"] = kept


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=PAIRS, help="pairs to run (a subset merges into the existing artifact)")
    ap.add_argument("--others", nargs="*", default=e51.OTHERS, help="the other encoders of the encoders pair")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp57 the difference atlas: oe_inferencex.compare on every pair of inferences", "smoke": args.smoke, "device": DEV,
               "config": {"pairs": list(args.pairs), "others": list(args.others), "draws": list(DRAWS), "draw_fraction": DRAW_FRACTION, "low_fraction": LOW_FRACTION,
                          "ndwi_ambiguous_above": NDWI_AMBIGUOUS, "offset_pair": OFFSET_PAIR, "window_grid": "exp47 W1: windows 1..14 of the offset-0 grid",
                          "cues": "boundary of a's hard map; least confident 20% of the testbed under a; 20% most unstable under a's tiling shift; |NDWI| < 0.1",
                          "prereg": "P1 boundary enrichment > 2 on the disagreement windows of every pair (offsets 0 vs 2, backbones, sensors, finetune on both testbeds; "
                                    "each encoder pair on the test split; the median GEOID event); P2 phi(sensors mask, backbones mask) < 0.5 pooled on both testbeds "
                                    "and tiles with phi < 0.5 a majority by a one-sided exact sign test (p < 0.05)",
                          "jobs": [{"pairs": list(args.pairs), "python": sys.executable, "device": DEV, "started": time.strftime("%Y-%m-%d %H:%M:%S")}]},
               "results": {p: {} for p in PAIRS}, "failures": []}
    summary["results"]["cross_pairs"] = {}
    rows, masks = [], {}
    cross = {name: {} for name in TESTBEDS}
    floods_pairs = [p for p in args.pairs if p in ("offsets", "backbones", "sensors", "finetune", "encoders")]
    ctx = None
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    if any(p in args.pairs for p in ("offsets", "backbones", "sensors", "finetune")):
        try:
            floods_dir, splits, feats = load_floods(args, summary)
            testbeds = [n for n in TESTBEDS if n in splits]
            ctx = {"splits": splits, "feats": feats, "testbeds": testbeds, "frozen": {},
                   "draws": draws_of(feats["train"][0], splits["train"][2], n_draws=len(DRAWS)),
                   "inputs": {n: testbed_inputs(splits, n, feats) for n in testbeds}}
            summary["config"]["draw_tiles"] = {int(d): int(len(idx)) for d, idx in ctx["draws"].items()}
        except Exception as ex:  # noqa: BLE001
            fail(summary, "floods inputs", ex)
    for pair in args.pairs:
        t0 = time.time()
        try:
            if pair == "offsets" and ctx:
                pair_offsets(ctx, summary, rows, masks, cross)
            elif pair == "backbones" and ctx:
                pair_backbones(ctx, summary, rows, masks, cross, args)
            elif pair == "sensors" and ctx:
                pair_sensors(ctx, summary, rows, masks, cross)
            elif pair == "finetune" and ctx:
                pair_finetune(ctx, summary, rows, masks, cross, args, floods_dir)
            elif pair == "encoders":
                pair_encoders(ctx, summary, rows, masks, args, floods_dir, args.others)
            elif pair == "geoid":
                pair_geoid(summary, rows, masks, args)
            summary["results"][pair]["seconds"] = time.time() - t0
        except Exception as ex:  # noqa: BLE001
            fail(summary, pair, ex)
        checkpoint(summary, masks, suffix)
    try:
        cross_pair_tables(summary, cross, masks)
    except Exception as ex:  # noqa: BLE001
        fail(summary, "cross_pairs", ex)
    merge_previous(summary, masks, rows, suffix, args.pairs)
    try:
        prereg(summary)
    except Exception as ex:  # noqa: BLE001
        fail(summary, "prereg", ex)
        summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp57_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    np.savez_compressed(os.path.join(hb.OUT, f"exp57_masks{suffix}.npz"), **masks)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(hb.OUT, f"exp57_difference_atlas{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    for stale in ("summary", "masks"):
        p = os.path.join(hb.OUT, f"exp57_{stale}{suffix}.partial." + ("json" if stale == "summary" else "npz"))
        if os.path.exists(p):
            os.remove(p)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
