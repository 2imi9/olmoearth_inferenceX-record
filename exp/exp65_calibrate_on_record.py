#!/usr/bin/env python
"""exp65: the package's calibrate on the record: exp49's ranker fusion and exp59's side rule refitted held-out, per family.

Why. oe_inferencex.calibrate fuses the label-free readings with labels and reports the fit held-out, bound to a model
family. Its two claims come from experiments that fitted on the training split and graded on the testbeds (exp49's
linear fusion beat confidence on three of four arms; exp59's side rule gained 5 to 17 points over the raw margin on the
frozen pairs and lost 26 on the fine-tuned pair). Neither number is the package's: the module cross-fits by tile on the
labelled set itself, which is what a user with labels for one region would do. This experiment records what the
module gives on the same pairs and testbeds, and tests the family lock.

Design. exp57's loaders and exp59's inputs on Sen1Floods11 (Bolivia, the multi-region test split; the W1 grid): the
crop-offset pair (0 vs 2), v1 vs v1.2, the S2 head vs the S1 head, the frozen S2 head vs FT-S2 v1 (fine-tuned once
with exp52's recipe). Side rules: per side margin |p - 0.5|, midrank percentile of the margin, signed probability,
boundary indicator, aligned tile-phase; the NDWI level shared; fit_side cross-fitted by tile (five folds), the raw
margin rule as baseline. Three fits per pair and testbed: (a) cross-fitted on the testbed itself (the package's
number); (b) fitted on the 600 training tiles' disagreement windows and applied to the testbed (exp59's protocol,
restated with the module); (c) the family test: the rule of (b) pooled over the three frozen pairs applied to the
fine-tuned pair with force=True, against the fine-tuned pair's own cross-fit. Ranker fusion for the v1 S2 head:
readings confidence (-|p - 0.5|), aligned tile-phase, boundary indicator, NDWI level, S2 patch variance; fit_ranker
cross-fitted by tile with the per-tile sign test against the best single reading, on both testbeds.

Preregistered (one-sided):
  P1  the cross-fitted ranker fusion beats the model's own confidence on both testbeds: held-out excess AURC lower by
      at least 0.001 and lower on more tiles than not (p < 0.05).
  P2  the cross-fitted side rule beats the raw margin rule by at least 0.05 held-out on every pair and testbed,
      the fine-tuned pair included (8 tests): refitting per family recovers what exp59's transfer lost.
  P3  the frozen-fitted rule applied to the fine-tuned pair is below the raw margin rule on both testbeds (the loss
      exp59 recorded, reproduced through the module's force path).
  Falsification: P1 fails if the fusion does not beat confidence on a testbed (labels would then add nothing to the
  ranking that the model's own margin does not carry); P2 fails on any pair below 0.05 (the module's held-out gain
  would then be smaller than exp59's train-split gain, or the fine-tuned pair would not be recoverable per family);
  P3 fails if the transferred rule matches the margin rule (the family lock would be unnecessary).

Inputs as exp59 (exp/out/exp18_feats.npz, data/floods, the v1 and v1.2 checkpoints); ~/oe12/.venv. Outputs.
exp/out/exp65_summary.json, exp/out/exp65_calibrate.csv (one row per fit), exp/out/exp65_readings.npz (the side
readings on every disagreement window with tile ids and labels, and the ranker readings on every valid window, so the
fits reproduce on a laptop). Smoke as exp59's.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import exp59_fitted_resolution as e59  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.calibrate import fit_ranker, fit_side, side_features  # noqa: E402
from oe_inferencex.metrics import excess_aurc  # noqa: E402
from oe_inferencex.signals import boundary_indicator, midrank_pct, s2_patch_variance  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, SL, TESTBEDS = e57.PATCH, e57.CROP, e57.SL, e57.TESTBEDS
PAIRS = ("offsets", "backbones", "sensors", "finetune")
FROZEN = ("offsets", "backbones", "sensors")
MIN_LEAD, MIN_GAIN, FOLDS = 0.001, 0.05, 5
FAMILY = {"offsets": "OlmoEarth v1 frozen S2 head", "backbones": "OlmoEarth v1 / v1.2 frozen S2 heads", "sensors": "OlmoEarth v1 frozen S2 / S1 heads", "finetune": "OlmoEarth v1 frozen S2 head / FT-S2 v1"}
fmt = e57.fmt


def side_readings(p, tp, ok):
    """The per-side readings of a probability map: margin, its midrank percentile over the valid windows, the signed
    probability, the boundary indicator of the decision, the aligned tile-phase."""
    m = np.abs(p - 0.5)
    r = np.full(m.shape, np.nan)
    r[ok] = midrank_pct(m[ok])
    return {"margin": m, "rank": r, "signed": p - 0.5, "boundary": boundary_indicator(p > 0.5).astype(np.float64), "tile_phase": tp}


def side_fit(pa, pb, tpa, tpb, ok, y, ndwi, groups, family):
    fa, fb = side_readings(pa, tpa, ok), side_readings(pb, tpb, ok)
    fusion, rep = fit_side(fa, fb, pa > 0.5, pb > 0.5, ok, y, groups=groups, family=family, folds=FOLDS, baseline="margin", shared={"ndwi": ndwi})
    return fusion, rep, side_features(fa, fb, {"ndwi": ndwi})


def apply_rule(fusion, feats, pa, pb, ok, y, force=False, family=None):
    """Share right of a rule on the disagreement windows of another set."""
    d = ((pa > 0.5) != (pb > 0.5)) & ok
    believe_b = fusion.score(feats, family=family, force=force) > 0
    right = np.where(believe_b, pb > 0.5, pa > 0.5) == y
    margin_right = np.where(np.abs(pa - 0.5) >= np.abs(pb - 0.5), pa > 0.5, pb > 0.5) == y
    return {"n_disagree": int(d.sum()), "share_right": float(right[d].mean()) if d.any() else float("nan"), "margin_rule": float(margin_right[d].mean()) if d.any() else float("nan")}


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=PAIRS)
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp65 calibrate on the record: the ranker fusion and the side rule refitted held-out through the package", "smoke": args.smoke, "device": e57.DEV,
               "config": {"pairs": list(args.pairs), "folds": FOLDS, "min_lead": MIN_LEAD, "min_gain": MIN_GAIN, "families": FAMILY,
                          "prereg": "P1 cross-fitted ranker fusion beats confidence on both testbeds (held-out E-AURC lead >= 0.001, per-tile p < 0.05); "
                                    "P2 cross-fitted side rule beats the raw margin rule by >= 0.05 on all 8 pair-testbeds; "
                                    "P3 the frozen-fitted rule applied to the fine-tuned pair is below the margin rule on both testbeds"},
               "results": {"ranker": {}, "side": {p: {} for p in PAIRS}, "transfer": {}, "finetune": {}}, "failures": []}
    rows, store = [], {}
    try:
        floods_dir, splits, feats = e57.load_floods(args, summary)
        testbeds = [n for n in TESTBEDS if n in splits]
        ctx = {"splits": splits, "feats": feats, "testbeds": testbeds, "draws": e57.draws_of(feats["train"][0], splits["train"][2]),
               "inputs": {n: e57.testbed_inputs(splits, n, feats) for n in testbeds}}
        y_tr, ok_tr = e57.labels_of(splits["train"][2])
        ndwi_tr = e57.ndwi_cue_input(splits["train"][1])
        groups_tr = np.arange(len(y_tr))[:, None, None]
        inputs = e59.sen1floods_inputs(ctx, summary, args)
        # ---- ranker fusion for the v1 S2 head
        for name in testbeds:
            try:
                t = ctx["inputs"][name]
                pa, _, tpa, _ = inputs[[p for p in args.pairs if p in inputs and name in inputs[p]][0]][name]
                sig = {"confidence": -np.abs(pa - 0.5), "tile_phase": tpa, "boundary": boundary_indicator(pa > 0.5).astype(np.float64), "ndwi_level": t["ndwi"],
                       "s2_variance": np.stack([s2_patch_variance(x, patch=PATCH, size=CROP) for x in t["s2"]])[:, SL, SL]}
                err = ((pa > 0.5) != t["y"]).astype(np.float64)
                fusion, rep = fit_ranker(sig, err, t["ok"], groups=t["groups"], family=FAMILY["offsets"], folds=FOLDS)
                ok = t["ok"]
                conf_e = excess_aurc(sig["confidence"][ok], err[ok])
                held = rep["held_out"]["excess_aurc"]
                gains = {}
                g = np.broadcast_to(t["groups"][:, None, None], ok.shape)[ok]
                # per-tile: the held-out fusion against confidence, recomputed from the module's per-group test inputs is not exposed; refit held-out scores per tile
                from oe_inferencex.calibrate import _crossfit, _folds, _standardise
                names = list(sig)
                Xraw = np.nan_to_num(np.stack([sig[k][ok] for k in names], 1)); X, _, _ = _standardise(Xraw)
                held_scores = _crossfit(X, err[ok], _folds(g, int(ok.sum()), FOLDS), True, FOLDS)
                e = err[ok]
                for gid in np.unique(g):
                    m = g == gid
                    if 3 <= e[m].sum() <= m.sum() - 3:
                        gains[int(gid)] = excess_aurc(sig["confidence"][ok][m], e[m]) - excess_aurc(held_scores[m], e[m])
                w, l, tt = wins_losses_ties(np.array(list(gains.values())))
                r = {"n_windows": rep["n_windows"], "n_errors": rep["n_errors"], "weights": rep["weights"], "held_out_excess_aurc": held, "confidence_excess_aurc": conf_e, "lead_over_confidence": conf_e - held,
                     "best_single": rep["best_single"], "best_single_excess_aurc": rep["best_single_excess_aurc"], "singles": rep["singles_excess_aurc"],
                     "held_out_capture": rep["held_out"]["capture"],
                     # the recorded ECE is of the balanced fit's output; since 2026-10-06 ece_of_p_error adds the prior
                     # back, so the recorded quantity is read under its own name
                     "ece": rep["held_out"].get("ece_without_prior_correction", rep["held_out"]["ece_of_p_error"]),
                     "tiles_vs_confidence": {"w": w, "l": l, "t": tt, "sign_p": sign_test(w, l, "greater")}, "fusion": fusion.to_dict()}
                summary["results"]["ranker"][name] = r
                rows.append({"fit": "ranker", "pair": "v1 S2 head", "testbed": name, "held_out": held, "baseline": conf_e, "gain": conf_e - held, "tiles_w": w, "tiles_l": l, "sign_p": r["tiles_vs_confidence"]["sign_p"]})
                for k in names:
                    store[f"ranker/{name}/{k}"] = sig[k][ok].astype(np.float32)
                store[f"ranker/{name}/err"], store[f"ranker/{name}/tile"] = e.astype(np.int8), g.astype(np.int32)
                print(f"ranker {name}: held-out E-AURC {held:.4f} vs confidence {conf_e:.4f} (lead {conf_e - held:+.4f}; tiles {w}/{l}, p={r['tiles_vs_confidence']['sign_p']:.2g}); best single {rep['best_single']} {rep['best_single_excess_aurc']:.4f}; capture@5% {rep['held_out']['capture']['0.05']:.3f}", flush=True)
            except Exception as ex:  # noqa: BLE001
                e57.fail(summary, f"ranker {name}", ex)
        # ---- side rules
        frozen_train_feats, frozen_train_y = [], []
        for pair in [p for p in args.pairs if p in inputs]:
            try:
                if "train" in inputs[pair]:
                    pa, pb, tpa, tpb = inputs[pair]["train"]
                    fus_tr, rep_tr, _ = side_fit(pa, pb, tpa, tpb, ok_tr, y_tr, ndwi_tr, groups_tr, FAMILY[pair])
                    summary["results"]["side"][pair]["train_fit"] = {"n_disagree": rep_tr["n_disagree"], "weights": rep_tr["weights"]}
                    if pair in FROZEN:
                        fa, fb = side_readings(pa, tpa, ok_tr), side_readings(pb, tpb, ok_tr)
                        d = ((pa > 0.5) != (pb > 0.5)) & ok_tr
                        F = side_features(fa, fb, {"ndwi": ndwi_tr})
                        frozen_train_feats.append({k: v[d] for k, v in F.items()}); frozen_train_y.append(((pb > 0.5) == y_tr)[d])
                else:
                    fus_tr = None
                for name in testbeds:
                    t = ctx["inputs"][name]
                    pa, pb, tpa, tpb = inputs[pair][name]
                    fusion, rep, F = side_fit(pa, pb, tpa, tpb, t["ok"], t["y"], t["ndwi"], t["groups"], FAMILY[pair])
                    r = {"n_disagree": rep["n_disagree"], "cross_fit": rep["held_out"]["share_right"], "margin_rule": rep["baseline"]["share_right"], "always_a": rep["always_a"], "always_b": rep["always_b"],
                         "gain": rep["held_out"]["share_right"] - rep["baseline"]["share_right"], "tiles": {k: rep["over_groups_vs_baseline"][k] for k in ("w", "l", "t", "sign_p")}, "weights": rep["weights"]}
                    if fus_tr is not None:
                        r["train_fit_applied"] = apply_rule(fus_tr, F, pa, pb, t["ok"], t["y"], family=FAMILY[pair])
                    summary["results"]["side"][pair][name] = r
                    rows.append({"fit": "side", "pair": pair, "testbed": name, "held_out": r["cross_fit"], "baseline": r["margin_rule"], "gain": r["gain"], "tiles_w": r["tiles"]["w"], "tiles_l": r["tiles"]["l"], "sign_p": r["tiles"]["sign_p"],
                                 "train_fit_applied": r.get("train_fit_applied", {}).get("share_right")})
                    d = ((pa > 0.5) != (pb > 0.5)) & t["ok"]
                    for k, v in F.items():
                        store[f"side/{pair}/{name}/{k}"] = v[d].astype(np.float32)
                    store[f"side/{pair}/{name}/b_right"] = ((pb > 0.5) == t["y"])[d]
                    store[f"side/{pair}/{name}/a_right"] = ((pa > 0.5) == t["y"])[d]
                    store[f"side/{pair}/{name}/tile"] = np.broadcast_to(t["groups"][:, None, None], d.shape)[d].astype(np.int32)
                    print(f"side {pair}/{name}: cross-fit {r['cross_fit']:.3f} vs margin {r['margin_rule']:.3f} (gain {r['gain']:+.3f}; tiles {r['tiles']['w']}/{r['tiles']['l']}, p={r['tiles']['sign_p']:.2g}) | always a {r['always_a']:.3f}, b {r['always_b']:.3f}"
                          + (f" | train-fit applied {r['train_fit_applied']['share_right']:.3f}" if fus_tr is not None else ""), flush=True)
            except Exception as ex:  # noqa: BLE001
                e57.fail(summary, f"side {pair}", ex)
        # ---- the family test
        if "finetune" in inputs and frozen_train_feats:
            try:
                from oe_inferencex.calibrate import Fusion, _logistic, _standardise
                keys = list(frozen_train_feats[0])
                Xraw = np.nan_to_num(np.concatenate([np.stack([f[k] for k in keys], 1) for f in frozen_train_feats]))
                y = np.concatenate(frozen_train_y).astype(np.float64)
                X, mu, sd = _standardise(Xraw)
                w, b = _logistic(X, y)
                pooled = Fusion("side", keys, mu, sd, w, b, "OlmoEarth v1 frozen heads (pooled frozen pairs)", len(y), "margin")
                for name in testbeds:
                    t = ctx["inputs"][name]
                    pa, pb, tpa, tpb = inputs["finetune"][name]
                    F = side_features(side_readings(pa, tpa, t["ok"]), side_readings(pb, tpb, t["ok"]), {"ndwi": t["ndwi"]})
                    try:
                        pooled.score(F, family=FAMILY["finetune"])
                        locked = False
                    except ValueError:
                        locked = True
                    forced = apply_rule(pooled, F, pa, pb, t["ok"], t["y"], force=True)
                    summary["results"]["transfer"][name] = {"lock_raised": locked, "frozen_rule_forced": forced, "own_cross_fit": summary["results"]["side"]["finetune"][name]["cross_fit"]}
                    rows.append({"fit": "transfer", "pair": "frozen rule -> finetune", "testbed": name, "held_out": forced["share_right"], "baseline": forced["margin_rule"], "gain": forced["share_right"] - forced["margin_rule"]})
                    print(f"transfer {name}: lock raised {locked}; frozen-fitted rule forced onto the fine-tuned pair {forced['share_right']:.3f} vs margin {forced['margin_rule']:.3f}; the pair's own cross-fit {summary['results']['side']['finetune'][name]['cross_fit']:.3f}", flush=True)
            except Exception as ex:  # noqa: BLE001
                e57.fail(summary, "transfer", ex)
        np.savez_compressed(os.path.join(hb.OUT, f"exp65_readings{suffix}.npz"), **store)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "pipeline", ex)
    # ---- prereg
    try:
        R = summary["results"]
        p1 = [(r["lead_over_confidence"] >= MIN_LEAD and r["tiles_vs_confidence"]["sign_p"] < 0.05) for r in R["ranker"].values()]
        p2 = [(R["side"][p][n]["gain"] >= MIN_GAIN and R["side"][p][n]["tiles"]["sign_p"] < 0.05) for p in PAIRS for n in TESTBEDS if n in R["side"].get(p, {})]
        p3 = [(v["frozen_rule_forced"]["share_right"] < v["frozen_rule_forced"]["margin_rule"]) for v in R["transfer"].values()]
        summary["prereg"] = {"P1": all(p1) if len(p1) == 2 else None, "P2": all(p2) if len(p2) == 8 else None, "P3": all(p3) if len(p3) == 2 else None,
                             "n_tested": {"P1": len(p1), "P2": len(p2), "P3": len(p3)}, "complete": bool(len(p1) == 2 and len(p2) == 8 and len(p3) == 2 and not summary["failures"])}
        print(f"prereg: P1 {summary['prereg']['P1']} P2 {summary['prereg']['P2']} P3 {summary['prereg']['P3']} | complete {summary['prereg']['complete']}", flush=True)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["prereg"] = {"P1": None, "P2": None, "P3": None, "complete": False}
    summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp65_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    if rows:
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(os.path.join(hb.OUT, f"exp65_calibrate{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
