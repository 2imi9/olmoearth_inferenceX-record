#!/usr/bin/env python
"""exp45: do the supported findings hold on OlmoEarth v1.2 Base, the encoder the served product uses?

Why. Every supported result in this repository was established on OlmoEarth v1 Base. The served land cover
change product uses v1.2 Base (its dataset card names the encoder), and v1.2 differs structurally: rotary
position encodings and one Sentinel-2 band-set token per patch instead of three (exp19). A result that does
not survive the backbone the product actually runs is not a result about the product. This run repeats the
four supported findings on v1.2, on the same tiles, with the same head protocol, and adds a cross-version
error-overlap measurement that bears on a separate open question.

Preregistered replications, each keeping the original's direction and unit. On Sen1Floods11 Bolivia (441
tiles) and the test split as exp18 sampled it (800 tiles, seed 1), hand labels, the exp18 head retrained on
the valid split's 600 tiles from v1.2 features (seed 0), four crop offsets:
  R1 (exp18)  confidence is the best ranker: its pooled E-AURC is below tile-phase, the boundary indicator
              and the NDWI-gradient control, and it wins the per-tile counts against each. Two-sided.
  R2 (exp36)  boundary first, then confidence, captures more errors than confidence at the 10% budget:
              per-tile one-sided exact sign test over tiles with 3 <= errors <= n - 3, plus a tile bootstrap
              of the pooled gain whose 95% interval must exclude zero.
  R3 (exp37)  the four label-free cues are enriched among error windows: boundary, least-confident 20%,
              tiling-unstable 20%, NDWI-ambiguous; each enrichment above 1 with a tile-clustered bootstrap
              interval above 1.
  R4 (exp42)  the shift-averaged decision beats the grid window on pixel accuracy per tile: one-sided,
              minimum worthwhile effect +0.002 mean tile gain (v1 gave +0.0102 and +0.0091).
Support for each requires the test to pass on BOTH testbeds. A failure is recorded as a failure to replicate
on v1.2, not as a retraction of the v1 result.

Descriptive alongside:
  D1  cross-version error overlap on identical windows: P(v1.2 probe wrong | v1 probe wrong), the converse,
      and phi. v1.2 changes the position encoding and the token structure, so this is a fourth architecture
      added to exp41's six, bearing on whether the errors follow the windows or the model.
  D2  mean tiling instability under each version (exp19 measured 0.046 against 0.032 on the WorldCover
      scenes and predicts v1.2 is the less stable; if so, R4's gain should be at least as large).

Inputs: data/floods/, exp/out/exp18_feats.npz (the cached v1 features, for D1 only). v1.2 features are
encoded here and not cached. Outputs: exp/out/exp45_summary.json, exp/out/exp45_v12_replication.csv.
Run with the v1.2 environment: ~/oe12/.venv/bin/python exp/exp45_v12_replication.py
--smoke: CPU, 4 tiles per split, _smoke outputs.
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
import harness_ab as hb  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.explain import cue_enrichment, top_fraction  # noqa: E402
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import ndwi_level, s2_patch_variance  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G, SHIFTS = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS
BUDGETS, PREREG_BUDGET, N_BOOT = (0.05, 0.10, 0.20), 0.10, 2000
MIN_EFFECT_R4, CUE_FRACTION, NDWI_TOL = 0.002, 0.2, 0.1
CONF, TILE, BOUND, CTRL = "confidence", "tile-phase", "boundary indicator", "control NDWI gradient"
CTRL_VAR, CTRL_LVL, LEX = "control S2 patch variance", "control NDWI level", "boundary first, then confidence"
REFS = [TILE, BOUND, CTRL]


def encode(model, tiles, shift, batch=32):
    return exp18.embed(model, tiles, shift)[0]


def build(model, splits, summary, tag):
    """Encode every split at the four offsets with `model`, train the head on the valid split, return per-testbed maps."""
    t0 = time.time()
    tr_s2, tr_lab = splits["train"]
    tr_feats = np.asarray(encode(model, tr_s2, 0), dtype=np.float32)
    tr_y, tr_ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    torch.manual_seed(0)
    head = train_logistic_head(torch.tensor(tr_feats.reshape(-1, tr_feats.shape[-1])[tr_ok.flatten()]), tr_y.flatten()[tr_ok.flatten()])
    out = {}
    for name in [k for k in splits if k != "train"]:
        s2, lab = splits[name]
        p_shift, logits = [], None
        for s in SHIFTS:
            f = np.asarray(encode(model, s2, s), dtype=np.float32)
            p, lg = exp18.head_prob_logit(f, *head)
            p_shift.append(p)
            if s == 0:
                logits = lg
        y, ok = exp18.patch_labels(lab[:, :CROP, :CROP])
        err = ((p_shift[0] > 0.5) != (y > 0.5)).astype(np.float64)
        out[name] = {"p_shift": np.stack(p_shift), "logit": logits, "y": y, "ok": ok, "err": err, "s2": s2, "lab": lab,
                     "acc": float(1 - err[ok].mean())}
        print(f"  {tag}/{name}: {len(s2)} tiles, valid windows {int(ok.sum())}, head accuracy {out[name]['acc']:.4f}", flush=True)
    summary["encoders"][tag] = {"seconds": time.time() - t0, "accuracy": {k: v["acc"] for k, v in out.items()}}
    return out


def signals(d):
    s2 = d["s2"]
    sig = {CONF: -np.abs(d["logit"]), TILE: exp18.aligned_tile_phase(d["p_shift"]), BOUND: exp18.boundary(d["p_shift"][0]),
           CTRL: exp18.ndwi_gradient(s2), CTRL_VAR: np.stack([s2_patch_variance(t, patch=PATCH, size=CROP) for t in s2]),
           CTRL_LVL: np.stack([ndwi_level(t, patch=PATCH, size=CROP) for t in s2])}
    sig[LEX] = np.stack([boundary_first_score(sig[CONF][t], sig[BOUND][t]) for t in range(len(s2))])
    return sig


def scored_tiles(err, ok):
    return [t for t in range(len(err)) if ok[t].any() and 3 <= err[t][ok[t]].sum() <= ok[t].sum() - 3]


def r1_best_ranker(sig, err, ok, tiles):
    pooled = {k: aurc_expected(np.asarray(v)[ok], err[ok]) - oracle_aurc(int(ok.sum()), int(err[ok].sum())) for k, v in sig.items()}
    per = {k: [aurc_expected(np.asarray(sig[k][t])[ok[t]], err[t][ok[t]]) - oracle_aurc(int(ok[t].sum()), int(err[t][ok[t]].sum())) for t in tiles] for k in sig}
    tests = {}
    for k in REFS + [CTRL_VAR, CTRL_LVL]:
        g = np.array(per[k]) - np.array(per[CONF])                    # positive: confidence better
        w, l, t_ = wins_losses_ties(g)
        tests[f"confidence vs {k}"] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l), "median_gain": float(np.median(g)),
                                       "pooled_confidence": pooled[CONF], "pooled_other": pooled[k]}
    passes = all(pooled[CONF] < pooled[k] and tests[f"confidence vs {k}"]["w"] > tests[f"confidence vs {k}"]["l"] for k in REFS)
    return {"pooled_eaurc": pooled, "tests": tests, "n_tiles": len(tiles), "replicates": bool(passes)}


def r2_boundary_first(sig, err, ok, tiles):
    cap = {t: {k: capture_at_budget_expected(np.asarray(sig[k][t])[ok[t]], err[t][ok[t]], BUDGETS) for k in (LEX, CONF)} for t in tiles}
    out = {}
    for b in BUDGETS:
        g = np.array([cap[t][LEX][b] - cap[t][CONF][b] for t in tiles])
        w, l, t_ = wins_losses_ties(g)
        one = b == PREREG_BUDGET
        out[str(b)] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one}
    idx = [t for t in range(len(err)) if ok[t].any()]
    flat = {k: np.concatenate([np.asarray(sig[k][t])[ok[t]] for t in idx]) for k in (LEX, CONF, BOUND)}
    e = np.concatenate([err[t][ok[t]] for t in idx])
    sizes = np.array([int(ok[t].sum()) for t in idx]); starts = np.r_[0, np.cumsum(sizes)[:-1]]

    def lex_of(sel):
        return boundary_first_score(flat[CONF][sel], flat[BOUND][sel])
    full = {b: capture_at_budget_expected(lex_of(np.arange(len(e))), e, BUDGETS)[b] for b in BUDGETS}
    fullc = capture_at_budget_expected(flat[CONF], e, BUDGETS)
    rng = np.random.default_rng(1)
    gains = {b: [] for b in BUDGETS}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([np.arange(starts[i], starts[i] + sizes[i]) for i in pick])
        ee = e[sel]
        if ee.sum() == 0:
            continue
        cl, cc = capture_at_budget_expected(lex_of(sel), ee, BUDGETS), capture_at_budget_expected(flat[CONF][sel], ee, BUDGETS)
        for b in BUDGETS:
            gains[b].append(cl[b] - cc[b])
    pooled = {str(b): {"lex": full[b], "confidence": fullc[b], "gain": full[b] - fullc[b],
                       "boot_lo": float(np.percentile(gains[b], 2.5)), "boot_hi": float(np.percentile(gains[b], 97.5))} for b in BUDGETS}
    p = out[str(PREREG_BUDGET)]; q = pooled[str(PREREG_BUDGET)]
    return {"per_tile": out, "pooled": pooled, "n_tiles": len(tiles), "replicates": bool(p["sign_p"] < 0.05 and q["boot_lo"] > 0)}


def r3_cues(sig, err, ok):
    idx = [t for t in range(len(err)) if ok[t].any()]
    e = np.concatenate([err[t][ok[t]] for t in idx])
    cl = np.concatenate([np.full(int(ok[t].sum()), t) for t in idx])
    cues = {"boundary": [], "low_confidence": [], "unstable": [], "ndwi_ambiguous": []}
    for t in idx:
        m = ok[t]
        cues["boundary"].append(np.asarray(sig[BOUND][t])[m] > 0)
        cues["low_confidence"].append(top_fraction(np.asarray(sig[CONF][t])[m], CUE_FRACTION))
        cues["unstable"].append(top_fraction(np.asarray(sig[TILE][t])[m], CUE_FRACTION))
        cues["ndwi_ambiguous"].append(np.asarray(sig[CTRL_LVL][t])[m] > -NDWI_TOL)
    res = {k: cue_enrichment(np.concatenate(v), e, clusters=cl, n_boot=N_BOOT) for k, v in cues.items()}
    return {"cues": res, "replicates": bool(all(r["enrichment"] > 1 and r["boot_lo"] > 1 for r in res.values()))}


def r4_window(d, sig):
    import exp42_window_design as e42
    gains, accs = [], {"W0 grid": [], "W1 shift-averaged": []}
    size = d["s2"].shape[-1]
    lo, hi = SHIFTS[-1], G * PATCH
    common = np.zeros((size, size), bool); common[lo:hi, lo:hi] = True
    for t in range(len(d["s2"])):
        pix = np.stack([e42.paint(d["p_shift"][s][t], s, size) for s in SHIFTS])
        valid = (d["lab"][t] >= 0) & common
        if not valid.any() or not np.isfinite(pix[:, common]).all():
            continue
        y = d["lab"][t] == 1
        w0, w1 = pix[0], pix.mean(0)
        a0 = float(((w0 > 0.5)[valid] == y[valid]).mean()); a1 = float(((w1 > 0.5)[valid] == y[valid]).mean())
        accs["W0 grid"].append(a0); accs["W1 shift-averaged"].append(a1); gains.append(a1 - a0)
    g = np.array(gains); w, l, t_ = wins_losses_ties(g)
    mean_gain = float(g.mean()) if len(g) else None
    return {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater"), "one_sided": True, "mean_gain": mean_gain,
            "median_gain": float(np.median(g)) if len(g) else None, "n_tiles": int(len(g)),
            "pixel_accuracy_mean": {k: float(np.mean(v)) for k, v in accs.items()},
            "replicates": bool(w > l and sign_test(w, l, "greater") < 0.05 and (mean_gain or 0) >= MIN_EFFECT_R4)}


def phi(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n11, n10, n01, n00 = (float((a & b).sum()), float((a & ~b).sum()), float((~a & b).sum()), float((~a & ~b).sum()))
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / den) if den > 0 else None


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp45 do the supported findings replicate on OlmoEarth v1.2 Base", "smoke": args.smoke,
               "config": {"encoder": "OlmoEarth-v1_2-Base, frozen, fp32", "reference_encoder": "OlmoEarth-v1-Base (cached features, D1 only)",
                          "shifts": list(SHIFTS), "budgets": list(BUDGETS), "prereg_budget": PREREG_BUDGET, "n_boot": N_BOOT,
                          "min_effect_R4": MIN_EFFECT_R4, "cue_fraction": CUE_FRACTION, "ndwi_tol": NDWI_TOL,
                          "prereg": "R1 confidence best ranker; R2 boundary-first at the 10% budget; R3 four cues enriched; R4 shift-averaged beats grid; each must pass on BOTH testbeds"},
               "encoders": {}, "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    paths, missing = hb.ensure_floods(floods_dir, allow_download=not args.smoke)
    n_tr = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES
    tr = hb.load_floods_split(paths["valid"], n_tr)
    bo = hb.load_floods_split(paths["bolivia"])
    te = hb.load_floods_split(os.path.join(floods_dir, "flood_test_data.pt"), n=exp18.N_TEST_TILES, seed=1)
    if args.smoke:
        bo = (bo[0][:args.smoke_tiles], bo[1][:args.smoke_tiles]); te = (te[0][:args.smoke_tiles], te[1][:args.smoke_tiles])
    splits = {"train": tr, "bolivia": bo, "test": te}
    t0 = time.time()
    model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(exp18.DEV).eval().float()
    print(f"loaded v1.2 Base on {exp18.DEV} in {time.time() - t0:.1f}s", flush=True)
    d12 = build(model, splits, summary, "v1_2")
    del model
    if exp18.DEV == "cuda":
        torch.cuda.empty_cache()

    for name in ("bolivia", "test"):
        try:
            d = d12[name]
            sig = signals(d)
            tiles = scored_tiles(d["err"], d["ok"])
            res = {"tiles": len(d["s2"]), "n_tiles_scored": len(tiles), "valid_windows": int(d["ok"].sum()), "head_accuracy": d["acc"],
                   "R1_best_ranker": r1_best_ranker(sig, d["err"], d["ok"], tiles),
                   "R2_boundary_first": r2_boundary_first(sig, d["err"], d["ok"], tiles),
                   "R3_cue_enrichment": r3_cues(sig, d["err"], d["ok"]),
                   "R4_shift_averaged_window": r4_window(d, sig),
                   "D2_mean_tile_phase": float(np.asarray(sig[TILE])[d["ok"]].mean())}
            res["replicates_all"] = bool(all(res[k]["replicates"] for k in ("R1_best_ranker", "R2_boundary_first", "R3_cue_enrichment", "R4_shift_averaged_window")))
            summary["results"][name] = res
            r1, r2, r3, r4 = res["R1_best_ranker"], res["R2_boundary_first"], res["R3_cue_enrichment"], res["R4_shift_averaged_window"]
            print(f"{name}: head acc {d['acc']:.4f} | R1 confidence pooled E-AURC {r1['pooled_eaurc'][CONF]:.4f} vs tile-phase {r1['pooled_eaurc'][TILE]:.4f} boundary {r1['pooled_eaurc'][BOUND]:.4f} -> {r1['replicates']} | "
                  f"R2 10% {r2['per_tile'][str(PREREG_BUDGET)]['w']}/{r2['per_tile'][str(PREREG_BUDGET)]['l']}/{r2['per_tile'][str(PREREG_BUDGET)]['t']} p={r2['per_tile'][str(PREREG_BUDGET)]['sign_p']:.2g} "
                  f"pooled {r2['pooled'][str(PREREG_BUDGET)]['lex']:.3f} vs {r2['pooled'][str(PREREG_BUDGET)]['confidence']:.3f} -> {r2['replicates']} | "
                  f"R3 " + ", ".join(f"{k} {v['enrichment']:.1f}x" for k, v in r3['cues'].items()) + f" -> {r3['replicates']} | "
                  f"R4 {r4['pixel_accuracy_mean']['W1 shift-averaged']:.4f} vs {r4['pixel_accuracy_mean']['W0 grid']:.4f} ({r4['w']}/{r4['l']}/{r4['t']}, mean {r4['mean_gain']:+.5f}) -> {r4['replicates']} | "
                  f"tile-phase mean {res['D2_mean_tile_phase']:.4f}", flush=True)
            rows.append({"testbed": name, "encoder": "v1_2", "head_accuracy": d["acc"], "n_tiles_scored": len(tiles),
                         **{f"pooled_eaurc {k}": v for k, v in r1["pooled_eaurc"].items()},
                         "R2_capture10_lex": r2["pooled"][str(PREREG_BUDGET)]["lex"], "R2_capture10_conf": r2["pooled"][str(PREREG_BUDGET)]["confidence"],
                         **{f"R3 {k}": v["enrichment"] for k, v in r3["cues"].items()},
                         "R4_acc_W0": r4["pixel_accuracy_mean"]["W0 grid"], "R4_acc_W1": r4["pixel_accuracy_mean"]["W1 shift-averaged"],
                         "D2_mean_tile_phase": res["D2_mean_tile_phase"]})
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    # ---------------- D1: cross-version error overlap on identical windows (v1 features from the cache)
    try:
        cache = os.path.join(hb.OUT, f"exp18_feats{suffix if args.smoke else ''}.npz")
        if args.smoke:
            cache = os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz")
        z = np.load(cache)
        trv1 = np.asarray(z["tr_base"], dtype=np.float32)
        tr_y, tr_ok = exp18.patch_labels(tr[1][:, :CROP, :CROP])
        torch.manual_seed(0)
        h1 = train_logistic_head(torch.tensor(trv1.reshape(-1, trv1.shape[-1])[tr_ok.flatten()]), tr_y.flatten()[tr_ok.flatten()])
        overlap = {}
        for name, key in (("bolivia", "bolivia_base0"), ("test", "test_base0")):
            if key not in z.files or name not in d12:
                continue
            f1 = np.asarray(z[key], dtype=np.float32)
            if args.smoke:
                f1 = f1[:args.smoke_tiles]
            p1, _ = exp18.head_prob_logit(f1, *h1)
            d = d12[name]
            ok = d["ok"]
            if p1.shape != d["y"].shape:
                overlap[name] = {"skipped": f"shape mismatch {p1.shape} vs {d['y'].shape}"}
                continue
            e1 = (((p1 > 0.5) != (d["y"] > 0.5))[ok])
            e2 = (d["err"][ok] > 0.5)
            overlap[name] = {"v1_accuracy": float(1 - e1.mean()), "v1_2_accuracy": float(1 - e2.mean()),
                             "p_v12_wrong_given_v1_wrong": float(e2[e1].mean()) if e1.any() else None,
                             "p_v12_wrong_given_v1_right": float(e2[~e1].mean()) if (~e1).any() else None,
                             "p_v1_wrong_given_v12_wrong": float(e1[e2].mean()) if e2.any() else None,
                             "phi": phi(e1, e2), "n_windows": int(ok.sum())}
            o = overlap[name]
            print(f"D1 {name}: v1 acc {o['v1_accuracy']:.4f}, v1.2 acc {o['v1_2_accuracy']:.4f}; "
                  f"P(v1.2 wrong | v1 wrong) {o['p_v12_wrong_given_v1_wrong']:.3f}, P(v1.2 wrong | v1 right) {o['p_v12_wrong_given_v1_right']:.3f}, phi {o['phi']:.3f}", flush=True)
        summary["D1_cross_version_error_overlap"] = overlap
    except Exception as ex:  # noqa: BLE001
        summary["failures"].append({"part": "D1", "error": repr(ex), "traceback": traceback.format_exc()})
        print(f"D1 FAILED: {ex!r}", flush=True)

    if len(summary["results"]) == 2:
        summary["prereg"] = {k: bool(all(summary["results"][n][k]["replicates"] for n in ("bolivia", "test")))
                             for k in ("R1_best_ranker", "R2_boundary_first", "R3_cue_enrichment", "R4_shift_averaged_window")}
        summary["prereg"]["complete"] = not summary["failures"]
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp45_v12_replication{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp45_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp45_summary{suffix}.json; replications: {summary.get('prereg')}; failures: {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
