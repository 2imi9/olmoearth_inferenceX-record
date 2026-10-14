#!/usr/bin/env python
"""exp47: on the decision the repository actually recommends, is the model's own confidence the best ranker?

Why. exp45 found that under OlmoEarth v1.2 Base, the encoder the served product uses, the model's confidence is
no longer the best error ranker on Sen1Floods11 Bolivia: tiling instability (pooled E-AURC 0.0138) and even the
no-model NDWI-level control (0.0119) beat it (0.0146). That measurement grades the GRID window: exp45's error
set is p_shift[0] > 0.5 and its confidence is the shift-0 logit margin, exactly as exp18, exp36 and exp37
defined them. But exp42 and exp45's own R4 recommend a different decision, the shift-averaged one, which wins on
pixel accuracy on both testbeds and now ships as signals.shift_averaged_probability. Its errors have never been
ranked by anything: exp42's selective-accuracy and capture columns also scored against the grid window's error
set (exp42_window_design.py, `sc, e = w_[key][m], w_["err0"][m]`). So the repository recommends one decision and
has only ever audited another. This run ranks the recommended decision's own errors.

Design. Both testbeds (Sen1Floods11 Bolivia, 441 tiles; the test split as exp18 sampled it, 800 tiles, seed 1),
both backbones (v1 Base from the cached exp/out/exp18_feats.npz, and v1.2 Base encoded here), the same head
protocol (the exp18 head trained on the valid split's 600 tiles, seed 0) and the same four crop offsets. Within
each arm the shift-averaged probability map is pooled back to the 4-px window grid, its hard decision defines
ONE error set, and every ranker in that arm is graded on it, so no comparison crosses error sets. Rankers:
the averaged decision's own confidence (the distance of the averaged probability from 0.5, fixed before the run),
the grid window's confidence (exp45's incumbent), aligned tile-phase, the boundary indicator of the averaged
hard map, the boundary-first order, and the three no-model controls (NDWI gradient, NDWI level, S2 patch
variance). Secondary: the U+ midrank combination of the averaged confidence with tile-phase and with NDWI level,
which exp45's result makes worth testing for the first time.

Preregistered, one-sided, on BOTH testbeds under v1.2; each must hold on both:
  P1  the averaged decision's confidence has a lower pooled excess AURC than aligned tile-phase, and wins the
      per-tile exact sign test against it (tiles with 3 <= errors <= n - 3).
  P2  the same against the no-model NDWI-level control.
  Minimum worthwhile effect: the pooled lead must be at least 0.001, since the grid window's deficit against the
  NDWI-level control under v1.2 is 0.0027 and a smaller lead is not a recovery.
Stated before the run: the averaged confidence and aligned tile-phase are both functions of the same four maps,
so a narrow lead is not a mechanism; and these numbers are never quoted against exp45's R1 line for line,
because the error sets differ. The v1 arm, computed identically here, is the only anchor. Falsification: if the
averaged confidence still loses to a no-model control on Bolivia, the ledger's confidence row and the Usage
guidance change to say so on the served backbone.

Inputs: data/floods/, exp/out/exp18_feats.npz. Outputs: exp/out/exp47_summary.json, exp/out/exp47_served_ranker.csv.
Run the v1.2 arm with ~/oe12/.venv. --smoke: CPU, 4 tiles per split, v1 only, _smoke outputs.
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
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import boundary_indicator, combine_midrank, ndwi_level, pool_to_windows, s2_patch_variance, shift_averaged_probability  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G, SHIFTS = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS
MIN_LEAD = 0.001
W1C, W0C, TILE, BOUND, LEX = "averaged confidence", "grid confidence", "tile-phase", "boundary indicator", "boundary first, then confidence"
CTRL_G, CTRL_L, CTRL_V = "control NDWI gradient", "control NDWI level", "control S2 patch variance"
COMBO_T, COMBO_N = "U+ averaged confidence + tile-phase", "U+ averaged confidence + NDWI level"
PRIMARY_AGAINST = [TILE, CTRL_L]


def head_from(tr_feats, tr_lab):
    y, ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    torch.manual_seed(0)
    return train_logistic_head(torch.tensor(tr_feats.reshape(-1, tr_feats.shape[-1])[ok.flatten()]), y.flatten()[ok.flatten()])


def arm(p_shift, logit0, s2, lab):
    """One backbone on one testbed: the averaged decision's error set and every ranker graded on it."""
    N = len(s2)
    y, ok = exp18.patch_labels(lab[:, :CROP, :CROP])
    # The averaged pixel map spans [0, 63) with every tiling covering only [3, 60). Pool at offset 0 so the windows
    # coincide with the shift-0 label grid, then keep the windows lying wholly inside the common region: window i
    # covers pixels [4i, 4i+4), which is inside [3, 60) exactly for i = 1..14.
    w1_full = np.stack([pool_to_windows(shift_averaged_probability(p_shift[:, t], patch=PATCH)[0], patch=PATCH, offset=0)
                        for t in range(N)])                               # (N, G, G) on the shift-0 grid
    bound_full = boundary_indicator(w1_full, probabilities=True)          # neighbours taken before the crop
    sl = slice(1, G)                                                       # windows 1..G-1, all inside the common region
    w1 = w1_full[:, sl, sl]
    yy, okk = y[:, sl, sl], ok[:, sl, sl]
    err = ((w1 > 0.5) != (yy > 0.5)).astype(np.float64)
    sig = {W1C: -np.abs(w1 - 0.5),
           W0C: -np.abs(logit0)[:, sl, sl],                              # exp45's incumbent exactly: the shift-0 logit margin
           TILE: exp18.aligned_tile_phase(p_shift)[:, sl, sl],
           BOUND: bound_full[:, sl, sl],
           CTRL_G: exp18.ndwi_gradient(s2)[:, sl, sl],
           CTRL_L: np.stack([ndwi_level(t, patch=PATCH, size=CROP) for t in s2])[:, sl, sl],
           CTRL_V: np.stack([s2_patch_variance(t, patch=PATCH, size=CROP) for t in s2])[:, sl, sl]}
    sig[LEX] = np.stack([boundary_first_score(sig[W1C][t], sig[BOUND][t]) for t in range(N)])
    for name, other in ((COMBO_T, TILE), (COMBO_N, CTRL_L)):
        c = np.full(w1.shape, np.nan)
        for t in range(N):
            m = okk[t]
            if m.any():
                c[t][m] = combine_midrank(sig[W1C][t][m], sig[other][t][m])
        sig[name] = c
    return sig, err, okk


def score(sig, err, ok):
    tiles = [t for t in range(len(err)) if ok[t].any() and 3 <= err[t][ok[t]].sum() <= ok[t].sum() - 3]
    pooled = {k: aurc_expected(np.asarray(v)[ok], err[ok]) - oracle_aurc(int(ok.sum()), int(err[ok].sum())) for k, v in sig.items()}
    per = {k: [aurc_expected(np.asarray(sig[k][t])[ok[t]], err[t][ok[t]]) - oracle_aurc(int(ok[t].sum()), int(err[t][ok[t]].sum())) for t in tiles] for k in sig}
    tests = {}
    for k in sig:
        if k == W1C:
            continue
        g = np.array(per[k]) - np.array(per[W1C])                          # positive: the averaged confidence is better
        w, l, t_ = wins_losses_ties(g)
        one = k in PRIMARY_AGAINST
        tests[k] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                    "median_gain": float(np.median(g)) if len(g) else None, "pooled_lead": pooled[k] - pooled[W1C]}
    passes = all(tests[k]["sign_p"] < 0.05 and tests[k]["pooled_lead"] >= MIN_LEAD for k in PRIMARY_AGAINST)
    return {"n_tiles_scored": len(tiles), "n_windows": int(ok.sum()), "n_errors": int(err[ok].sum()),
            "pooled_eaurc": pooled, "tests": tests, "prereg_passes": bool(passes)}


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp47 ranking the shift-averaged decision's own errors", "smoke": args.smoke,
               "config": {"rankers": [W1C, W0C, TILE, BOUND, LEX, CTRL_G, CTRL_L, CTRL_V, COMBO_T, COMBO_N],
                          "primary_against": PRIMARY_AGAINST, "min_lead": MIN_LEAD, "shifts": list(SHIFTS),
                          "prereg": "the averaged decision's own confidence beats aligned tile-phase and the NDWI-level control, pooled lead >= 0.001 and a one-sided per-tile sign test, on BOTH testbeds under v1.2"},
               "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    paths, _ = hb.ensure_floods(floods_dir, allow_download=not args.smoke)
    n_tr = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES
    tr_s2, tr_lab = hb.load_floods_split(paths["valid"], n_tr)
    bo = hb.load_floods_split(paths["bolivia"])
    te_path = os.path.join(floods_dir, "flood_test_data.pt")
    te = hb.load_floods_split(paths["valid"], args.smoke_tiles, seed=7) if (args.smoke and not os.path.exists(te_path)) else hb.load_floods_split(te_path, n=exp18.N_TEST_TILES, seed=1)
    if args.smoke:
        bo = (bo[0][:args.smoke_tiles], bo[1][:args.smoke_tiles]); te = (te[0][:args.smoke_tiles], te[1][:args.smoke_tiles])
    splits = {"bolivia": bo, "test": te}

    versions = {"v1": None} if args.smoke else {"v1": None, "v1_2": None}
    try:
        from olmoearth_pretrain.model_loader import ModelID
        has_v12 = hasattr(ModelID, "OLMOEARTH_V1_2_BASE")
    except Exception:
        has_v12 = False
    if not has_v12:
        if not args.smoke:
            raise SystemExit("OLMOEARTH_V1_2_BASE is unavailable: run the full experiment with ~/oe12/.venv, whose "
                             "olmoearth_pretrain is built from upstream main. The v1.2 arm is the preregistered one.")
        versions.pop("v1_2", None)
        summary["note"] = "v1.2 identifiers unavailable in this environment; only the v1 arm ran"

    cache = np.load(os.path.join(hb.OUT, "exp18_feats.npz")) if not args.smoke else np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz"))
    for version in list(versions):
        try:
            t0 = time.time()
            if version == "v1":
                trf = np.asarray(cache["tr_base"], dtype=np.float32)
                if len(trf) != len(tr_s2):                                # never slice a cache to fit: the seed-0 sample is not prefix-stable
                    model = hb.load_model(); trf = np.asarray(exp18.embed(model, tr_s2, 0)[0], dtype=np.float32); del model
                    print("  train cache length mismatch, re-encoded", flush=True)
                head = head_from(trf, tr_lab)
                pshift, logit0 = {}, {}
                for name, (s2, lab) in splits.items():
                    key = f"{'bolivia' if name == 'bolivia' else 'test'}_base"
                    if all(f"{key}{s}" in cache.files and len(cache[f"{key}{s}"]) == len(s2) for s in SHIFTS):
                        out = [exp18.head_prob_logit(np.asarray(cache[f"{key}{s}"], dtype=np.float32), *head) for s in SHIFTS]
                    else:
                        model = hb.load_model()
                        out = [exp18.head_prob_logit(np.asarray(exp18.embed(model, s2, s)[0], dtype=np.float32), *head) for s in SHIFTS]
                        del model
                    pshift[name], logit0[name] = np.stack([o[0] for o in out]), out[0][1]
            else:
                from olmoearth_pretrain.model_loader import load_model_from_id
                model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(exp18.DEV).eval().float()
                head = head_from(np.asarray(exp18.embed(model, tr_s2, 0)[0], dtype=np.float32), tr_lab)
                pshift, logit0 = {}, {}
                for name, (s2, lab) in splits.items():
                    out = [exp18.head_prob_logit(np.asarray(exp18.embed(model, s2, s)[0], dtype=np.float32), *head) for s in SHIFTS]
                    pshift[name], logit0[name] = np.stack([o[0] for o in out]), out[0][1]
                del model
                if exp18.DEV == "cuda":
                    torch.cuda.empty_cache()
            summary["results"][version] = {"encode_seconds": time.time() - t0}
            for name, (s2, lab) in splits.items():
                sig, err, ok = arm(pshift[name], logit0[name], s2, lab)
                r = score(sig, err, ok)
                summary["results"][version][name] = r
                lead = {k: r["tests"][k]["pooled_lead"] for k in PRIMARY_AGAINST}
                print(f"{version}/{name}: {r['n_windows']} windows, {r['n_errors']} errors of the averaged decision, {r['n_tiles_scored']} tiles | "
                      f"pooled E-AURC " + ", ".join(f"{k} {v:.4f}" for k, v in list(r["pooled_eaurc"].items())[:5]) +
                      f" | lead over tile-phase {lead[TILE]:+.4f} (p={r['tests'][TILE]['sign_p']:.2g}), over NDWI level {lead[CTRL_L]:+.4f} (p={r['tests'][CTRL_L]['sign_p']:.2g}) -> {r['prereg_passes']}", flush=True)
                rows.append({"version": version, "testbed": name, "n_windows": r["n_windows"], "n_errors": r["n_errors"],
                             **{f"eaurc {k}": v for k, v in r["pooled_eaurc"].items()}, "prereg_passes": r["prereg_passes"]})
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": version, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{version} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    if "v1_2" in summary["results"] and all(k in summary["results"]["v1_2"] for k in ("bolivia", "test")):
        summary["prereg"] = {"supported": bool(all(summary["results"]["v1_2"][n]["prereg_passes"] for n in ("bolivia", "test"))),
                             "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp47_served_ranker{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp47_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp47_summary{suffix}.json; prereg {summary.get('prereg')}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
