"""Shared scaffolding for the exp28-pattern experiments (exp30, exp31).

Part A: the exp13 disagreement set on the 27 rule-selected exp11 scenes
(seed-0 Base head trained on katima, errors against WorldCover 2021, scenes
with fewer than 8 errors skipped). Part B: Sen1Floods11 Bolivia hand labels
exactly as exp18 (600 valid-split training tiles, 60x60 crops, same head,
same patch-label pooling, per-tile rule of at least three errors and three
correct patches). Reference signals: confidence (negative absolute logit),
aligned tile-phase, boundary indicator, NDWI-gradient control. Control set:
constant score, S2 within-patch variance, NDWI level. Preregistered
combination U+ = mean of the within-unit midrank percentiles of confidence
and of the experiment's primary score; inference = per-river mean gain,
one-sided exact sign test over the eight river clusters. All of this is
copied from exp/exp28_decoder_consistency.py, which stays as it ran; the
experiment scripts add their own signals and call `finish_part_a` /
`finish_part_b`. The signal, control and test arithmetic lives in
oe_inferencex.signals / oe_inferencex.stats; the wrappers here keep the
exp28 names and the exact exp13 / exp18 code paths on the real grids. The
per-reference Spearman columns use tie-averaged ranks
(oe_inferencex.stats.spearman) from exp36 on; exp28 to exp34 used exp14's
positional ranks, which order ties by raster position.

fp32 only: no autocast, no TF32, no compile.
"""
import csv
import json
import math
import os
import sys
import time
import traceback
import warnings
from collections import Counter

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402

from oe_inferencex.evidence import predict_head, predict_logit, train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected  # noqa: E402
from oe_inferencex import signals as sig_lib  # noqa: E402
from oe_inferencex import stats as stats_lib  # noqa: E402
import exp13_stat_corrections as exp13  # noqa: E402
import exp14_boundary_ablation as exp14  # noqa: E402
import exp18_sen1floods_expert as exp18  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

OUT = os.path.join(EXP_DIR, "out")
DEV = "cuda" if torch.cuda.is_available() and "--smoke" not in sys.argv else "cpu"   # --smoke runs on the CPU, as advertised
PATCH = 4
SHIFTS = exp13.SHIFTS
N_BANDSETS = Modality.SENTINEL2_L2A.num_band_sets  # 3
MIN_ERRORS_A = 8          # exp13 scene rule
N_PERM = 10000
FLOODS_URL = ("https://storage.googleapis.com/ai2-olmoearth-projects-public-data/"
              "research_benchmarks/floods/flood_{split}_data.pt")
FLOODS_SPLITS = ("valid", "bolivia")
RULE_SCENES = [
    "barotse", "cuando_20", "cuando_50", "cuando_80", "delta", "kafue_20", "kafue_50", "kafue_80",
    "kazungula", "luangwa_conf", "okavango_50", "okavango_80", "okavango_sep", "rovuma_20", "rovuma_50",
    "rovuma_80", "save_20", "save_50", "save_80", "shire_20", "shire_50", "shire_80", "shire_liwonde",
    "vicfalls_up", "zambezi_20", "zambezi_50", "zambezi_80",
]
SMOKE_SCENES = ("barotse", "okavango_80")
SMOKE_SIZE = 64
CONF, TILE, BOUND, CTRL = ("confidence (baseline)", "tile-phase (aligned)", "boundary indicator",
                            "control NDWI gradient")
REFERENCES = (CONF, TILE, BOUND, CTRL)
CONST, CTRL_VAR, CTRL_LVL = ("constant score", "control S2 patch variance", "control NDWI level")
EXTRA_CONTROLS = (CONST, CTRL_VAR, CTRL_LVL)
BASE_SIGNALS = REFERENCES + EXTRA_CONTROLS
RIVER = {"barotse": "Zambezi", "delta": "Zambezi", "kazungula": "Zambezi", "vicfalls_up": "Zambezi",
         "zambezi_20": "Zambezi", "zambezi_50": "Zambezi", "zambezi_80": "Zambezi",
         "cuando_20": "Cuando", "cuando_50": "Cuando", "cuando_80": "Cuando",
         "kafue_20": "Kafue", "kafue_50": "Kafue", "kafue_80": "Kafue", "luangwa_conf": "Luangwa",
         "okavango_50": "Okavango", "okavango_80": "Okavango", "okavango_sep": "Okavango",
         "rovuma_20": "Rovuma", "rovuma_50": "Rovuma", "rovuma_80": "Rovuma",
         "save_20": "Save", "save_50": "Save", "save_80": "Save",
         "shire_20": "Shire", "shire_50": "Shire", "shire_80": "Shire", "shire_liwonde": "Shire"}
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(nanvar|Degrees of freedom|Mean of empty slice|All-NaN).*")


# ----------------------------------------------------------------------------- features
def normalize_s2(x_dn):
    """(N, H, W, 12) DN -> normalized float32 tensor (N, H, W, 1, 12) on DEV (exp18/exp17 path)."""
    x = exp18._norm.normalize(Modality.SENTINEL2_L2A, x_dn.astype(np.float64)[:, :, :, None, :])
    return torch.tensor(x, dtype=torch.float32, device=DEV)


def scene_tensor(img, date, size, shift=0):
    """(12, H, W) DN scene -> (x (1, size, size, 1, 12), timestamps (1, 1, 3))."""
    crop = img[:, shift:shift + size, shift:shift + size].transpose(1, 2, 0)[None]
    d, m0, y = (int(v) for v in date)
    return normalize_s2(crop), torch.tensor([d, m0, y], device=DEV)[None, None, :]


@torch.no_grad()
def embed_pooled(model, x, ts):
    """Pooled (G, G, D) features exactly as exp11.embed_gpu (fast pass, mean over T and band sets)."""
    b = x.shape[0]
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=x,
        sentinel2_l2a_mask=torch.ones((b, x.shape[1], x.shape[2], 1, N_BANDSETS), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=ts)
    out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])[0].cpu().numpy()


def load_model():
    t0 = time.time()
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV).eval().float()
    print(f"loaded Base on {DEV} in {time.time() - t0:.1f}s", flush=True)
    return model


# ----------------------------------------------------------------------------- controls and references
def s2_patch_variance(img, size):
    """Observed-input control: mean over bands of the within-patch pixel std (raw DN)."""
    return sig_lib.s2_patch_variance(img, patch=PATCH, size=size)


def ndwi_level(img, size):
    """Observed-input control: -|patch-mean NDWI| (near zero = spectrally ambiguous water/land)."""
    return sig_lib.ndwi_level(img, patch=PATCH, size=size)


def midrank_pct(v):
    """Within-unit midrank percentile in [0, 1], ties averaged."""
    return sig_lib.midrank_pct(v)


def one_sided_sign_p(w, n):
    """P(X >= w), X ~ Binomial(n, 1/2); the preregistered one-sided exact sign test."""
    return stats_lib.sign_test(w, n - w, "greater")


def river_test(per_scene):
    """per_scene: {scene: gain}; average within river cluster, then a one-sided exact sign test (> 0 predicted)."""
    r = stats_lib.clustered_sign_test(per_scene, RIVER, aggregate="mean", alternative="greater")
    return {"per_river": r["per_cluster"], "w": r["w"], "l": r["l"], "t": r["t"], "one_sided_p": r["p"],
            "n_rivers": r["n_clusters"]}


def aligned_tile_phase_general(shift_probs, size, pad=exp13.PAD):
    """exp13.aligned_tile_phase for an arbitrary crop size (equal to exp13's at 128; checked in --smoke)."""
    return sig_lib.aligned_tile_phase(shift_probs, patch=PATCH)


def tile_phase_a(shift_probs, size):
    return exp13.aligned_tile_phase(shift_probs) if size == exp13.SIZE else aligned_tile_phase_general(shift_probs, size)


def ndwi_gradient_general(img, size):
    """exp13.ndwi_gradient for an arbitrary crop size (equal to exp13's at 128; checked in --smoke)."""
    return sig_lib.ndwi_gradient(img, patch=PATCH, size=size)


def ndwi_a(img, size):
    return exp13.ndwi_gradient(img) if size == exp13.SIZE else ndwi_gradient_general(img, size)


def boundary_indicator(p):
    """exp14's pred-boundary: fraction of a patch's 8 neighbours whose hard label differs (edge padding)."""
    return sig_lib.boundary_indicator(p, probabilities=True)


def paired_stats(diffs, rng):
    """diffs: per-unit (reference E-AURC - signal E-AURC); > 0 = signal better."""
    return stats_lib.paired_comparison(diffs, rng, N_PERM)


def json_ready(o):
    if isinstance(o, dict):
        return {str(k): json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_ready(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return json_ready(o.tolist())
    return o


def smoke_selfcheck():
    """The size-generic helpers must equal the exp13/exp18 originals on the real grids."""
    rng = np.random.default_rng(123)
    ps = [rng.random((exp13.GRID, exp13.GRID)) for _ in SHIFTS]
    assert np.allclose(aligned_tile_phase_general(ps, exp13.SIZE), exp13.aligned_tile_phase(ps)), "tile-phase mismatch"
    img = rng.integers(0, 6000, (12, exp13.SIZE + exp13.PAD, exp13.SIZE + exp13.PAD)).astype(np.int32)
    assert np.allclose(ndwi_gradient_general(img, exp13.SIZE), exp13.ndwi_gradient(img)), "ndwi mismatch"
    p = rng.random((1, exp18.G, exp18.G))
    assert np.allclose(boundary_indicator(p[0]), exp18.boundary(p)[0]), "boundary mismatch"
    ps60 = np.stack([rng.random((1, exp18.G, exp18.G)) for _ in SHIFTS])
    assert np.allclose(aligned_tile_phase_general([q[0] for q in ps60], exp18.CROP), exp18.aligned_tile_phase(ps60)[0]), "tile-phase (60) mismatch"
    v = np.array([3.0, 1.0, 3.0, 2.0])
    assert np.allclose(midrank_pct(v), np.array([2.5, 0, 2.5, 1]) / 3), "midrank mismatch"
    assert abs(one_sided_sign_p(7, 8) - 9 / 256) < 1e-12, "sign test mismatch"
    print("smoke self-check: size-generic helpers equal exp13/exp18 originals", flush=True)


# ----------------------------------------------------------------------------- part A
def load_part_a(model, args, summary):
    """Scenes, cached or recomputed pooled features, and the seed-0 Base head. Returns a context dict."""
    size = SMOKE_SIZE if args.smoke else exp13.SIZE
    G = size // PATCH
    scenes = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]
    feat_path = os.path.join(OUT, "exp11_feats.npz")
    feats, feats_source = {}, None
    if args.smoke:
        names = [n for n in SMOKE_SCENES if f"{n}_img" in scenes][:2]
        feats_source = f"recomputed on {size}-px crops (smoke)"
    elif os.path.exists(feat_path):
        feats = dict(np.load(feat_path, allow_pickle=True))
        names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
        names = [n for n in names if f"{n}_base0" in feats and n not in exp13.NON_RULE]   # exp13's scene set
        feats_source = "exp/out/exp11_feats.npz"
        if set(names) != set(RULE_SCENES):
            summary["part_a"]["scene_set_warning"] = {"missing": sorted(set(RULE_SCENES) - set(names)),
                                                       "extra": sorted(set(names) - set(RULE_SCENES))}
            print(f"WARNING: exp11_feats scene set differs from the 27 rule scenes: {summary['part_a']['scene_set_warning']}")
    else:
        names = [n for n in RULE_SCENES if f"{n}_img" in scenes]
        feats_source = "recomputed (exp11_feats.npz missing): same code path as exp11.embed_gpu, fp32"
        print("WARNING: exp11_feats.npz missing; recomputing pooled features", flush=True)
    summary["part_a"].update({"crop_size": size, "grid": G, "feats_source": feats_source, "scenes_requested": names})
    tr_lab = tr_labels[:G, :G]
    if "tr_base" not in feats:
        x, ts = scene_tensor(tr_img, tr_date, size)
        feats["tr_base"] = embed_pooled(model, x, ts)
    torch.manual_seed(0)   # exp13: seed 0, Base head trained first
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), tr_lab)
    return {"size": size, "G": G, "scenes": scenes, "feats": feats, "names": names, "hb": hb,
            "tr_lab": tr_lab, "tr_feats": np.asarray(feats["tr_base"], dtype=np.float32), "feats_source": feats_source}


def scene_unit(ctx, model, name, args, summary):
    """Per-scene inputs: image, label, shift-0 features, probabilities, logit, errors. None if the scene is skipped."""
    G, size, feats, scenes, hb = ctx["G"], ctx["size"], ctx["feats"], ctx["scenes"], ctx["hb"]
    img, date, lab = scenes[f"{name}_img"], scenes[f"{name}_date"], scenes[f"{name}_lab"][:G, :G]
    for s in SHIFTS:
        if f"{name}_base{s}" not in feats:
            x_s, ts_s = scene_tensor(img, date, size, shift=s)
            feats[f"{name}_base{s}"] = embed_pooled(model, x_s, ts_s)
    p_shift = [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in SHIFTS]
    p = p_shift[0]
    logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
    err = ((p > 0.5) != lab.astype(bool)).astype(np.float64)
    n_err = int(err.sum())
    if n_err < (1 if args.smoke else MIN_ERRORS_A):
        summary["part_a"]["skipped"].append({"scene": name, "reason": f"{n_err} errors"})
        print(f"{name}: {n_err} errors, skipped", flush=True)
        return None
    return {"name": name, "img": img, "date": date, "lab": lab, "feats0": np.asarray(feats[f"{name}_base0"], dtype=np.float32),
            "p_shift": p_shift, "p": p, "logit": logit, "err": err, "n_err": n_err}


def base_signals_a(unit, ctx):
    """The four references and the three controls for one scene, each (G, G)."""
    G, size = ctx["G"], ctx["size"]
    return {
        CONF: -np.abs(unit["logit"]),
        TILE: tile_phase_a(unit["p_shift"], size),
        BOUND: boundary_indicator(unit["p"]),
        CTRL: ndwi_a(unit["img"], size),
        CONST: np.zeros((G, G)),
        CTRL_VAR: s2_patch_variance(unit["img"], size),
        CTRL_LVL: ndwi_level(unit["img"], size),
    }


def combination(sigs, primary, mask=None):
    """U+ = mean of within-unit midrank percentiles of confidence and of the primary score."""
    if mask is None:
        return ((midrank_pct(sigs[CONF]) + midrank_pct(sigs[primary])) / 2).reshape(np.shape(sigs[CONF]))
    out = np.zeros(np.shape(sigs[CONF]))
    if mask.any():
        out[mask] = (midrank_pct(sigs[CONF][mask]) + midrank_pct(sigs[primary][mask])) / 2
    return out


def score_scene(unit, sigs, rows, per, rho_per, new_names, extra_cols=None):
    """E-AURC of every signal on the scene's errors; rows for the CSV; Spearman of new signals vs references."""
    name, err = unit["name"], unit["err"]
    ef = err.flatten()
    val = {k: exp13.eaurc(np.asarray(v, dtype=np.float64).flatten(), ef) for k, v in sigs.items()}
    per[name] = val
    rho_per[name] = {k: {r: stats_lib.spearman(np.asarray(sigs[k]).flatten(), np.asarray(sigs[r]).flatten()) for r in REFERENCES}
                     for k in new_names}
    for k in sigs:
        row = {"part": "A", "unit": name, "n_patches": int(ef.size), "n_errors": unit["n_err"], "signal": k,
               "aurc": aurc_expected(np.asarray(sigs[k]).flatten(), ef), "eaurc": val[k],
               "mean_value": float(np.mean(sigs[k]))}
        for r in REFERENCES:
            row[f"gain_vs_{r.split(' (')[0].split(' ')[0]}"] = val[r] - val[k]
            if k in new_names:
                row[f"spearman_vs_{r.split(' (')[0].split(' ')[0]}"] = rho_per[name][k][r]
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return val


def finish_part_a(summary, per, rho_per, new_names, primary, combo):
    """Cross-scene tests, river-level preregistered tests, tallies and Spearman summaries."""
    sn = sorted(per)
    summary["part_a"]["n_scenes"] = len(sn)
    summary["part_a"]["scenes"] = sn
    if not sn:
        return
    sig_names = list(per[sn[0]].keys())
    rng2 = np.random.default_rng(1)
    tests = {}
    for k in sig_names:
        tests[k] = {}
        refs = REFERENCES + (EXTRA_CONTROLS if k in new_names or k == combo else ())
        for r in refs:
            if r == k:
                continue
            d = [per[s][r] - per[s][k] for s in sn]
            tests[k][f"vs {r}"] = paired_stats(d, rng2)
            tests[k][f"vs {r}"]["river"] = river_test({s: per[s][r] - per[s][k] for s in sn})
    summary["part_a"]["tests"] = tests
    summary["part_a"]["prereg"] = {
        "primary_score": primary,
        "combination": "U+ = mean of within-scene midrank percentiles of confidence and of the primary score",
        "combination_gain_over_confidence": river_test({s: per[s][CONF] - per[s][combo] for s in sn}),
        "primary_vs_confidence": river_test({s: per[s][CONF] - per[s][primary] for s in sn}),
        "tile_phase_vs_confidence": river_test({s: per[s][CONF] - per[s][TILE] for s in sn}),
        "note": "river-level one-sided exact sign tests are the inference; scene-level W/L/T are descriptive",
    }
    pr = summary["part_a"]["prereg"]
    c, d = pr["combination_gain_over_confidence"], pr["primary_vs_confidence"]
    print(f"  prereg: U+ vs confidence rivers {c['w']}/{c['l']}/{c['t']} p={c['one_sided_p']:.3f}"
          f" | primary vs confidence rivers {d['w']}/{d['l']}/{d['t']} p={d['one_sided_p']:.3f}", flush=True)
    summary["part_a"]["best_tally"] = dict(Counter(min(per[s], key=per[s].get) for s in sn))
    summary["part_a"]["median_eaurc"] = {k: float(np.median([per[s][k] for s in sn])) for k in sig_names}
    summary["part_a"]["spearman"] = {
        k: {r: {"median": float(np.nanmedian([rho_per[s][k][r] for s in sn])),
                "min": float(np.nanmin([rho_per[s][k][r] for s in sn])), "max": float(np.nanmax([rho_per[s][k][r] for s in sn]))}
            for r in REFERENCES}
        for k in new_names if k in sig_names}
    print(f"\npart A: {len(sn)} scenes. E-AURC head-to-head (W/L/T, exact sign p, perm p, median gain):")
    for k in sig_names:
        for r, t in tests[k].items():
            print(f"  {k:<40} {r:<30} {t['w']:>2}/{t['l']:>2}/{t['t']:<2} sign p={t['sign_p']:.3g} perm p={t.get('perm_p', float('nan')):.3g} median {t['median_gain']:+.4f}")
    print("  best per scene:", summary["part_a"]["best_tally"])
    for k, v in summary["part_a"]["spearman"].items():
        print(f"  Spearman {k}: " + ", ".join(f"{r.split(' (')[0]} {vv['median']:+.2f} [{vv['min']:+.2f},{vv['max']:+.2f}]" for r, vv in v.items()))


# ----------------------------------------------------------------------------- part B
def ensure_floods(floods_dir, allow_download):
    paths = {s: os.path.join(floods_dir, f"flood_{s}_data.pt") for s in FLOODS_SPLITS}
    missing = [s for s, p in paths.items() if not os.path.exists(p)]
    if missing and not allow_download:
        return None, missing
    os.makedirs(floods_dir, exist_ok=True)
    import urllib.request
    for s in missing:
        url = FLOODS_URL.format(split=s)
        tmp = paths[s] + ".part"
        print(f"downloading {url} -> {paths[s]}", flush=True)
        last = [-1]

        def hook(blocks, bs, total, last=last):
            pct = int(100 * blocks * bs / max(total, 1))
            if pct // 10 != last[0]:
                last[0] = pct // 10
                print(f"  {min(pct, 100)}%", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        os.replace(tmp, paths[s])
    return paths, []


def load_floods_split(path, n=None, seed=0):
    """exp18.load_split with an absolute path (same band reorder, same subsampling)."""
    d = torch.load(path, weights_only=True)
    s2 = d["s2"].numpy().astype(np.float32)[:, exp18.BAND_IDX]
    lab = d["labels"].numpy()[:, 0]
    if n is not None and n < len(s2):
        idx = np.random.default_rng(seed).choice(len(s2), n, replace=False)
        s2, lab = s2[idx], lab[idx]
    return s2, lab


def load_part_b(model, args, summary, exp_tag, extra_keys=()):
    """Flood tiles, features (exp18 cache when present, else recomputed and cached as exp<tag>_floods_feats.npz),
    the exp18 head, Bolivia probabilities, logits and errors. Returns a context dict, or None when skipped."""
    floods_dir = args.floods_dir or os.path.join(ROOT, "data", "floods")
    paths, missing = ensure_floods(floods_dir, allow_download=not args.smoke)
    if paths is None:
        summary["part_b"]["skipped"] = f"flood files missing in smoke mode: {missing}"
        print(f"part B skipped: {summary['part_b']['skipped']}", flush=True)
        return None
    n_train = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES
    tr_s2, tr_lab = load_floods_split(paths["valid"], n_train)
    bo_s2, bo_lab = load_floods_split(paths["bolivia"])
    if args.smoke:
        bo_s2, bo_lab = bo_s2[:args.smoke_tiles], bo_lab[:args.smoke_tiles]
    CROP, G = exp18.CROP, exp18.G
    summary["part_b"].update({"floods_dir": floods_dir, "train_tiles": int(len(tr_s2)), "bolivia_tiles": int(len(bo_s2)), "crop": CROP})
    keys = ["tr_base"] + [f"bolivia_base{s}" for s in SHIFTS]
    z = {}
    exp18_cache = os.path.join(OUT, "exp18_feats.npz")
    own_cache = os.path.join(OUT, f"{exp_tag}_floods_feats{args.suffix}.npz")
    if not args.smoke and os.path.exists(exp18_cache):
        zz = np.load(exp18_cache)
        z = {k: zz[k] for k in keys}
        for k in extra_keys:
            if k in zz:
                z[k] = zz[k]
        summary["part_b"]["feats_source"] = exp18_cache
    elif os.path.exists(own_cache):
        zz = np.load(own_cache)
        z = {k: zz[k] for k in keys}
        summary["part_b"]["feats_source"] = own_cache
    else:
        t0 = time.time()
        z["tr_base"], _ = exp18.embed(model, tr_s2, 0)
        for s in SHIFTS:
            z[f"bolivia_base{s}"], _ = exp18.embed(model, bo_s2, s)
        np.savez(own_cache, **z)
        summary["part_b"]["feats_source"] = f"recomputed with exp18.embed in {time.time() - t0:.0f}s, cached {own_cache}"
    print(f"part B features: {summary['part_b']['feats_source']}", flush=True)

    tr_y, tr_ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
    sel = tr_ok.flatten()
    D = z["tr_base"].shape[-1]
    torch.manual_seed(0)
    hb = train_logistic_head(f32(z["tr_base"]).reshape(-1, D)[sel], tr_y.flatten()[sel])
    y, ok = exp18.patch_labels(bo_lab[:, :CROP, :CROP])
    p_shift = np.stack([exp18.head_prob_logit(z[f"bolivia_base{s}"], *hb)[0] for s in SHIFTS])
    p, logit = exp18.head_prob_logit(z["bolivia_base0"], *hb)
    err = ((p > 0.5) != (y > 0.5)).astype(np.float64)
    acc = 1 - err[ok].mean()
    summary["part_b"]["head_acc"] = float(acc)
    print(f"bolivia: tiles {len(y)}, valid patches {int(ok.sum())}, Base head accuracy {acc:.3f}", flush=True)
    return {"tr_s2": tr_s2, "tr_lab": tr_lab, "bo_s2": bo_s2, "bo_lab": bo_lab, "z": z, "hb": hb, "CROP": CROP, "G": G,
            "tr_feats": np.asarray(z["tr_base"], dtype=np.float32).reshape(-1, D)[sel], "tr_y": tr_y.flatten()[sel],
            "ev_feats": np.asarray(z["bolivia_base0"], dtype=np.float32), "y": y, "ok": ok, "p_shift": p_shift,
            "p": p, "logit": logit, "err": err, "acc": float(acc), "N": len(bo_s2)}


def base_signals_b(ctx):
    """The four references and the three controls for Bolivia, each (N, G, G)."""
    N, G, CROP, bo_s2 = ctx["N"], ctx["G"], ctx["CROP"], ctx["bo_s2"]
    return {
        CONF: -np.abs(ctx["logit"]),
        TILE: exp18.aligned_tile_phase(ctx["p_shift"]),
        BOUND: exp18.boundary(ctx["p"]),
        CTRL: exp18.ndwi_gradient(bo_s2),
        CONST: np.zeros((N, G, G)),
        CTRL_VAR: np.stack([s2_patch_variance(tile, CROP) for tile in bo_s2]),
        CTRL_LVL: np.stack([ndwi_level(tile, CROP) for tile in bo_s2]),
    }


def finish_part_b(summary, rows, ctx, sigs, new_names, primary, combo, ok=None, unit="bolivia"):
    """Pooled and per-tile E-AURC, W/L against references and controls, prereg combination, Spearman, CSV rows.

    `unit` names the evaluation split in the CSV rows (exp40 scores the test split with the same code)."""
    N, err = ctx["N"], ctx["err"]
    ok = ctx["ok"] if ok is None else ok
    ok = ok & np.all(np.isfinite(np.stack([np.asarray(sigs[k], dtype=np.float64) for k in new_names])), axis=0)
    combo_arr = np.zeros(np.shape(sigs[CONF]))
    for tt in range(N):
        m = ok[tt]
        if m.any():
            combo_arr[tt][m] = (midrank_pct(sigs[CONF][tt][m]) + midrank_pct(sigs[primary][tt][m])) / 2
    sigs[combo] = combo_arr
    pooled = {k: exp18.eaurc(np.asarray(v, dtype=np.float64)[ok], err[ok]) for k, v in sigs.items()}
    per = {k: [] for k in sigs}
    rho = {k: {r: [] for r in REFERENCES} for k in new_names}
    tiles_scored = []
    for t in range(N):
        m = ok[t]
        e = err[t][m]
        if e.sum() < 3 or e.sum() > len(e) - 3:   # exp18 rule
            continue
        tiles_scored.append(t)
        for k, v in sigs.items():
            per[k].append(exp18.eaurc(np.asarray(v[t], dtype=np.float64)[m], e))
        for k in new_names:
            for r in REFERENCES:
                rho[k][r].append(stats_lib.spearman(np.asarray(sigs[k][t])[m], np.asarray(sigs[r][t])[m]))
    for k in per:
        per[k] = np.array(per[k])
    n_tiles = len(tiles_scored)
    summary["part_b"].update({"valid_patches": int(ok.sum()), "n_tiles_scored": n_tiles, "pooled_eaurc": pooled})
    tests = {}
    for k in sigs:
        tests[k] = {}
        refs = REFERENCES + (EXTRA_CONTROLS if k in new_names or k == combo else ())
        for r in refs:
            if r == k:
                continue
            tests[k][f"vs {r}"] = paired_stats(per[r] - per[k], None)
    summary["part_b"]["tests"] = tests
    summary["part_b"]["prereg"] = {"primary_score": primary,
                                   "combination_gain_over_confidence_pooled": float(pooled[CONF] - pooled[combo]),
                                   "combination_tiles_better": int(((per[CONF] - per[combo]) > 1e-12).sum()),
                                   "combination_tiles_worse": int(((per[CONF] - per[combo]) < -1e-12).sum()),
                                   "primary_tiles_better": int(((per[CONF] - per[primary]) > 1e-12).sum()),
                                   "primary_tiles_worse": int(((per[CONF] - per[primary]) < -1e-12).sum()),
                                   "note": "Bolivia is one flood event; per-tile signs are descriptive, not an exact test"}
    summary["part_b"]["best_tally"] = dict(Counter(min(sigs, key=lambda k: per[k][i]) for i in range(n_tiles))) if n_tiles else {}
    summary["part_b"]["spearman"] = {k: {r: {"median": float(np.nanmedian(v)) if v else None, "min": float(np.nanmin(v)) if v else None,
                                            "max": float(np.nanmax(v)) if v else None} for r, v in rr.items()} for k, rr in rho.items()}
    on_b = sigs[BOUND] > 0
    summary["part_b"]["boundary_share"] = {"errors": float(on_b[ok & (err > 0)].mean()) if (ok & (err > 0)).any() else None,
                                           "correct": float(on_b[ok & (err == 0)].mean()) if (ok & (err == 0)).any() else None}
    print(f"  pooled E-AURC / per-tile (n={n_tiles}) W/L vs references:")
    for k in sigs:
        line = f"    {k:<40} pooled {pooled[k]:.4f}"
        for r, t in tests[k].items():
            line += f" | {r.split(' (')[0][:12]:<12} {t['w']:>3}/{t['l']:<3} p={t['sign_p']:.1e}"
        print(line)
    print("    best per tile:", summary["part_b"]["best_tally"])
    for k, rr in summary["part_b"]["spearman"].items():
        print(f"    Spearman {k}: " + ", ".join(f"{r.split(' (')[0]} {v['median']:+.2f}" for r, v in rr.items() if v["median"] is not None))
    rows.append({"part": "B", "unit": f"{unit} (pooled)", "n_patches": int(ok.sum()), "n_errors": int(err[ok].sum()), "signal": "",
                 "aurc": "", "eaurc": "", "mean_value": "", "head_acc": ctx["acc"], "n_tiles_scored": n_tiles})
    for k in sigs:
        v = np.asarray(sigs[k], dtype=np.float64)
        row = {"part": "B", "unit": f"{unit} (pooled)", "n_patches": int(ok.sum()), "n_errors": int(err[ok].sum()), "signal": k,
               "aurc": aurc_expected(v[ok], err[ok]), "eaurc": pooled[k], "mean_value": float(np.mean(v[ok])),
               "n_tiles_scored": n_tiles}
        for r in REFERENCES:
            tag = r.split(' (')[0].split(' ')[0]
            row[f"gain_vs_{tag}"] = pooled[r] - pooled[k]
            if k in new_names:
                row[f"W/L_vs_{tag}"] = f"{tests[k][f'vs {r}']['w']}/{tests[k][f'vs {r}']['l']}"
                row[f"sign_p_vs_{tag}"] = tests[k][f"vs {r}"]["sign_p"]
                row[f"spearman_vs_{tag}"] = summary["part_b"]["spearman"][k][r]["median"]
        rows.append(row)
    for i, t in enumerate(tiles_scored):
        m = ok[t]
        for k in sigs:
            v = np.asarray(sigs[k][t], dtype=np.float64)
            row = {"part": "B", "unit": f"{unit}/tile{t}", "n_patches": int(m.sum()), "n_errors": int(err[t][m].sum()), "signal": k,
                   "aurc": aurc_expected(v[m], err[t][m]), "eaurc": float(per[k][i]), "mean_value": float(np.mean(v[m]))}
            for r in REFERENCES:
                tag = r.split(' (')[0].split(' ')[0]
                row[f"gain_vs_{tag}"] = float(per[r][i] - per[k][i])
                if k in new_names:
                    row[f"spearman_vs_{tag}"] = rho[k][r][i]
            rows.append(row)
    return per, tiles_scored


# ----------------------------------------------------------------------------- main scaffolding
def make_parser(doc, extra=None):
    import argparse
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="CPU smoke: 2 scenes, 64-px crops, _smoke outputs")
    ap.add_argument("--floods-dir", default=None, help="directory holding flood_{valid,bolivia}_data.pt (default data/floods)")
    ap.add_argument("--smoke-tiles", type=int, default=4, help="tiles per split in smoke part B")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    if extra:
        extra(ap)
    return ap


def run(exp_tag, title, config, part_a_fn, part_b_fn, args, csv_name):
    """Runs both parts with per-part failure capture and always writes the CSV, the summary and the cache."""
    args.suffix = "_smoke" if args.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    summary = {"experiment": title, "device": DEV, "smoke": args.smoke,
               "config": {"patch": PATCH, "tf32": False, "autocast": False, "compile": False, **config},
               "part_a": {"skipped": []}, "part_b": {}, "failures": []}
    rows, cache = [], {}
    csv_path = os.path.join(OUT, f"{csv_name}{args.suffix}.csv")
    sum_path = os.path.join(OUT, f"{exp_tag}_summary{args.suffix}.json")
    cache_path = os.path.join(OUT, f"{exp_tag}_cache{args.suffix}.npz")
    try:
        if args.smoke:
            smoke_selfcheck()
        model = load_model()
        if not args.skip_a:
            try:
                part_a_fn(model, args, summary, rows, cache)
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": "A", "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
                print(f"part A FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
        if not args.skip_b:
            try:
                part_b_fn(model, args, summary, rows, cache)
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": "B", "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
                print(f"part B FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    except Exception as ex:  # noqa: BLE001
        summary["failures"].append({"part": "setup", "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
        print(f"setup FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    finally:
        if rows:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in keys})
            print(f"wrote {csv_path} ({len(rows)} rows)")
        if cache:
            np.savez(cache_path, **cache)
            print(f"wrote {cache_path}")
        summary["runtime_s"] = time.time() - t_start
        summary["n_failures"] = len(summary["failures"])
        with open(sum_path, "w") as f:
            json.dump(json_ready(summary), f, indent=1)
        print(f"wrote {sum_path}; failures: {len(summary['failures'])}; runtime {summary['runtime_s']:.0f}s")
    return summary
