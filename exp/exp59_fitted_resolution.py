#!/usr/bin/env python
"""exp59: can a label-fitted rule say which side to believe where two inferences disagree? (the follow-up exp58 named)

Why. exp58 showed the label-free readings of a disagreement set barely beat the raw diff: believing the more confident
side is right on 51-70% of the disagreement windows, far below always choosing the side the labels prefer, and on
GEOID-Flood both-sides-confident separates change from error only pooled. The repository's one fusion that ever beat
confidence was label-fitted on the head's own training split (exp49, exp50). The same route for the comparison half:
fit the rule once on the tiles the heads were trained on, grade it on the testbeds, never touch their labels.

Design. exp57's pairs on the W1 grid (exp58's probabilities): crop offset 0 against 2, v1 against v1.2, the S2 head
against the S1 head, the frozen head against FT-S2 v1 (fine-tuned once more with exp52's recipe); the fitting set is
the 600 valid-split tiles the frozen heads were trained on, encoded at the four crop offsets so both sides have W1
maps there too. On every disagreement window eleven label-free features of the two sides: each side's margin
|p - 0.5|, its midrank percentile over the map's valid windows, its signed probability p - 0.5, its boundary indicator,
its aligned tile-phase, and the NDWI level of the window. A logistic rule (standardised features, full-batch Adam as
evidence.train_logistic_head, no class balancing) predicts "side B is right"; the decision is p > 0.5. Graded on
Bolivia and the multi-region test split against the raw margin rule (exp58's, the mean-probability decision), always
the first side, always the second, and the coin flip; per tile with compare.over_groups (tiles with at least 3
disagreement windows). GEOID-Flood: the same features for the pre-event S2 head against the post-event S1 head, the
rule fitted on the val events' disagreement windows (their labels already trained the heads) to predict "flooded by
the label", graded per test event against the raw diff (every disagreement is change, 27% precision) and exp58's
both-confident rule.

Preregistered (one-sided):
  P1  the fitted rule beats the raw margin rule on every Sen1Floods11 pair and testbed (8 tests): pooled share right
      higher by at least 0.05, and higher on more tiles than not (p < 0.05).
  P2  on GEOID-Flood the fitted change rule's pooled precision for flooded windows is at least 1.5 times the raw
      diff's, and it beats the raw diff on more test events than not (p < 0.05; events with at least 20 disagreement
      windows), the test exp58's both-confident rule failed.
  Falsification: P1 fails if any pair gains less than 0.05 or misses the per-tile test (a fitted rule would then add
  nothing a margin does not already say); P2 fails if the ratio is below 1.5 or the event test misses (the features
  would not separate change from error event by event). Stated prediction, not tested: the fitted rule stays below
  always-the-better-side on the sensor and fine-tune pairs.

Inputs as exp58. Outputs exp/out/exp59_summary.json, exp/out/exp59_fitted_resolution.csv. --pairs and the two-job
merge as exp58 (geoid under the pinned venv). Smoke as exp58's.
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
import exp51_their_probe as e51  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import exp58_tool_vs_diff as e58  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import over_groups  # noqa: E402
from oe_inferencex.signals import boundary_indicator, midrank_pct  # noqa: E402

PATCH, CROP, G, SHIFTS, DEV, SL = e57.PATCH, e57.CROP, e57.G, e57.SHIFTS, e57.DEV, e57.SL
PAIRS = ("offsets", "backbones", "sensors", "finetune", "geoid")
TESTBEDS = e57.TESTBEDS
MIN_GAIN, MIN_CHANGE_RATIO, MIN_WINDOWS, MIN_EVENT_WINDOWS = 0.05, 1.5, 3, 20
EPOCHS, LR = 300, 0.05
FEATURES = ["margin_a", "margin_b", "rank_a", "rank_b", "signed_a", "signed_b", "boundary_a", "boundary_b", "tile_phase_a", "tile_phase_b", "ndwi"]
fmt = e57.fmt


# ----------------------------------------------------------------------------- the rule
def features(pa, pb, ok, tp_a, tp_b, ndwi):
    """(N, H, W, 11) label-free features of the two sides; ranks are midrank percentiles over the map's valid windows."""
    ma, mb = np.abs(pa - 0.5), np.abs(pb - 0.5)
    ra, rb = np.full(ma.shape, np.nan), np.full(mb.shape, np.nan)
    ra[ok], rb[ok] = midrank_pct(ma[ok]), midrank_pct(mb[ok])
    return np.stack([ma, mb, ra, rb, pa - 0.5, pb - 0.5, boundary_indicator(pa > 0.5), boundary_indicator(pb > 0.5), tp_a, tp_b, ndwi], axis=-1)


class Rule:
    """Logistic regression on standardised features (evidence.train_logistic_head's optimiser, no balancing)."""

    def __init__(self, X, y, seed=0):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        torch.manual_seed(seed)
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        t = torch.tensor(np.asarray(y, dtype=np.float32))
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=LR)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, t)
            loss.backward()
            opt.step()
        self.w, self.b, self.n = w.detach().numpy(), float(b.detach()), int(len(y))

    def prob(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

    def weights(self):
        return {k: float(v) for k, v in zip(FEATURES, self.w)}


def disagreement_set(pa, pb, ok):
    return ((pa > 0.5) != (pb > 0.5)) & ok


def grade(rule, X, pa, pb, ok, y, groups, min_windows=MIN_WINDOWS):
    """The fitted rule against the raw margin rule on the disagreement windows, pooled and per group."""
    a, b = pa > 0.5, pb > 0.5
    d = disagreement_set(pa, pb, ok)
    believe_b = rule.prob(X) > 0.5
    fitted_right = np.where(believe_b, b, a) == y
    margin_right = np.where(np.abs(pa - 0.5) >= np.abs(pb - 0.5), a, b) == y
    gains = {}
    for gid in np.unique(groups):
        m = d & (groups == gid)[:, None, None]
        if m.sum() >= min_windows:
            gains[gid.item()] = float(fitted_right[m].mean() - margin_right[m].mean())
    return {"n_disagree": int(d.sum()), "fitted": e58._mean(fitted_right[d]), "margin": e58._mean(margin_right[d]),
            "always_a": e58._mean((a == y)[d]), "always_b": e58._mean((b == y)[d]), "coin": 0.5,
            "gain_over_margin": e58._mean(fitted_right[d]) - e58._mean(margin_right[d]), "share_believe_b": e58._mean(believe_b[d]),
            "over_groups": over_groups(gains)}


def grade_change(rule, X, pa, pb, ok, ya, yb, events, min_windows=MIN_EVENT_WINDOWS):
    """GEOID: the fitted change rule against the raw diff and exp58's both-confident rule, pooled and per event."""
    d = disagreement_set(pa, pb, ok)
    flooded = yb & ~ya
    change = rule.prob(X) > 0.5
    ma, mb = np.abs(pa - 0.5), np.abs(pb - 0.5)
    both = (ma >= e58.CONFIDENT_MARGIN) & (mb >= e58.CONFIDENT_MARGIN)
    per = {}
    for ev in np.unique(events):
        m = d & (events == ev)[:, None, None]
        if m.sum() < min_windows or not (m & change).any():
            continue
        per[ev.item()] = {"n_disagree": int(m.sum()), "n_change": int((m & change).sum()), "flooded_all": e58._mean(flooded[m]),
                          "flooded_rule": e58._mean(flooded[m & change]), "flooded_both_confident": e58._mean(flooded[m & both]) if (m & both).any() else None}
    base, tool = e58._mean(flooded[d]), e58._mean(flooded[d & change])
    return {"n_disagree": int(d.sum()), "n_change": int((d & change).sum()), "share_change": e58._mean(change[d]),
            "flooded_all": base, "flooded_rule": tool, "flooded_both_confident": e58._mean(flooded[d & both]),
            "ratio_to_raw_diff": (tool / base) if base > 0 else float("nan"), "recall_of_flooded": e58._mean(change[d & flooded]) if (d & flooded).any() else float("nan"),
            "per_event": per, "over_events_vs_raw_diff": over_groups({k: v["flooded_rule"] - v["flooded_all"] for k, v in per.items()}),
            "over_events_vs_both_confident": over_groups({k: v["flooded_rule"] - v["flooded_both_confident"] for k, v in per.items() if v["flooded_both_confident"] is not None})}


# ----------------------------------------------------------------------------- inputs: both sides' W1 probabilities on the fitting tiles and the testbeds
def sen1floods_inputs(ctx, summary, args):
    """{pair: {split: (pa, pb, tp_a, tp_b)}} for the fitting tiles ("train") and the testbeds, draw-0 heads."""
    tr_s1, tr_s2, tr_lab = ctx["splits"]["train"]
    head_s2 = e57.fit_head(ctx["feats"]["train"][0], tr_lab, ctx["draws"][0])
    out = {p: {} for p in ("offsets", "backbones", "sensors", "finetune")}

    def tp(p_shift):
        return exp18.aligned_tile_phase(p_shift)[:, SL, SL]

    model = hb.load_model()
    tr_feats = dict(ctx["feats"]["train"])
    for s in SHIFTS[1:]:
        tr_feats[s] = np.asarray(exp18.embed(model, tr_s2, s)[0], dtype=np.float32)
    feats_s2 = {"train": tr_feats, **{n: ctx["inputs"][n]["feats"] for n in ctx["testbeds"]}}
    p_s2 = {n: e57.probs(f, head_s2) for n, f in feats_s2.items()}
    if "offsets" in args.pairs:
        for n, p in p_s2.items():
            p0, p2 = e57.offset_windows(p[0], 0), e57.offset_windows(p[2], 2)
            out["offsets"][n] = (p0, p2, tp(p), tp(p))
    if "sensors" in args.pairs:
        f1 = {"train": e57.encode_s1(model, tr_s1), **{n: e57.encode_s1(model, ctx["splits"][n][0]) for n in ctx["testbeds"]}}
        head_s1 = e57.fit_head(f1["train"][0], tr_lab, ctx["draws"][0])
        for n in f1:
            p1 = e57.probs(f1[n], head_s1)
            out["sensors"][n] = (e57.w1(p_s2[n]), e57.w1(p1), tp(p_s2[n]), tp(p1))
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    if "backbones" in args.pairs:
        model2, label = e57.second_backbone(args)
        f2 = {"train": e57.encode_s2(model2, tr_s2), **{n: e57.encode_s2(model2, ctx["splits"][n][1]) for n in ctx["testbeds"]}}
        head2 = e57.fit_head(f2["train"][0], tr_lab, ctx["draws"][0])
        summary["config"]["second_backbone"] = label
        for n in f2:
            p2 = e57.probs(f2[n], head2)
            out["backbones"][n] = (e57.w1(p_s2[n]), e57.w1(p2), tp(p_s2[n]), tp(p2))
        del model2
        if DEV == "cuda":
            torch.cuda.empty_cache()
    if "finetune" in args.pairs:
        import exp52_finetune as e52
        t0 = time.time()
        floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
        n = args.smoke_tiles if args.smoke else None
        ft_s1, ft_s2, ft_l = e51.load_ours(floods_dir, "valid" if args.smoke else "train", n, seed=3)
        va_s1, va_s2, va_l = e51.load_ours(floods_dir, "valid", n, seed=0)
        tr = (*e52.prep(ft_s2, ft_s1, 0), ft_l[:, :CROP, :CROP])
        va = (*e52.prep(va_s2, va_s1, 0), va_l[:, :CROP, :CROP])
        model = hb.load_model()
        ft, history, best = e52.finetune(model, ["s2"], tr, va, 1 if args.smoke else e52.EPOCHS, log=lambda m: print(f"  FT-S2 v1{m}", flush=True))
        summary["results"]["finetune"]["train"] = {"train_tiles": int(len(ft_s2)), "best_val_miou": best, "train_seconds": time.time() - t0}
        sets = {"train": (tr_s1, tr_s2), **{n: (ctx["splits"][n][0], ctx["splits"][n][1]) for n in ctx["testbeds"]}}
        for n, (s1, s2) in sets.items():
            pshift = []
            for s in SHIFTS:
                x2, x1 = e52.prep(s2, s1, s)
                p = e52.predict(ft, x2, x1)
                pshift.append(p.reshape(len(p), G, PATCH, G, PATCH).mean(axis=(2, 4)))
            pshift = np.stack(pshift)
            out["finetune"][n] = (e57.w1(p_s2[n]), e57.w1(pshift), tp(p_s2[n]), tp(pshift))
        del ft, model
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return out


def geoid_inputs(args, summary):
    """Both heads' W1 probabilities on the val chips (fitting) and the clear test chips (grading), with events and labels."""
    import exp55_geoid_flood as e55
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    data_dir = os.path.join(hf_home, "geoid_flood")
    tiles = e57.geoid_tiles(args, data_dir)
    if tiles is None:
        return None
    if not args.smoke and any(not tiles[s] for s in ("test", "val")):
        import shutil
        root = os.path.join(data_dir, "tree")
        files = e55.hub_files()
        for split, n in (("test", 4), ("val", 2)):
            if not tiles[split]:
                shutil.rmtree(os.path.join(root, split, ".done"), ignore_errors=True)
                e55.fetch_split(files, split, n, data_dir, root)
                tiles[split] = e55.scan_split(root, split)
        summary["config"]["geoid_refetched"] = True
    data = {}
    for split in ("test", "val"):
        cand = e55.chip_candidates(tiles[split], split)
        sel = e55.select_chips(cand, 400, smoke_chips=4 if args.smoke else None)
        s2, s1, lab = e55.read_imagery(tiles[split], sel)
        data[split] = {"s2": s2, "s1": s1, "lab": lab, "aoi": np.array([x["aoi"] for x in sel]), "clear": np.array([x["clear"] >= e55.MIN_CLEAR for x in sel], bool)}
    model = hb.load_model()
    val = data["val"]
    heads = {"A": e55.fit_head(e55.features(model, "s2", val["s2"][val["clear"]], 0), e55.task_labels(val["lab"][val["clear"]], "A")),
             "B": e55.fit_head(e55.features(model, "s1", val["s1"], 0), e55.task_labels(val["lab"], "B"))}
    out = {}
    for split in ("val", "test"):
        d = data[split]
        idx = np.flatnonzero(d["clear"])
        p = {t: np.stack([exp18.head_prob_logit(e55.features(model, s, d[s][idx], sh), *heads[t])[0] for sh in SHIFTS]) for t, s in (("A", "s2"), ("B", "s1"))}
        lab = d["lab"][idx]
        ya, oka = e57.labels_of(e55.task_labels(lab, "A"))
        yb, okb = e57.labels_of(e55.task_labels(lab, "B"))
        out[split] = {"pa": e57.w1(p["A"]), "pb": e57.w1(p["B"]), "tp_a": exp18.aligned_tile_phase(p["A"])[:, SL, SL], "tp_b": exp18.aligned_tile_phase(p["B"])[:, SL, SL],
                      "ok": oka & okb, "ya": ya, "yb": yb, "events": d["aoi"][idx], "ndwi": e57.ndwi_cue_input(d["s2"][idx])}
        summary["results"]["geoid"][f"{split}_chips"] = int(len(idx))
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- prereg and main
def prereg(summary):
    R = summary["results"]
    p1, detail = [], {}
    for pair in ("offsets", "backbones", "sensors", "finetune"):
        for name in TESTBEDS:
            r = R.get(pair, {}).get(name)
            if r and "fitted" in r:
                detail[f"{pair}/{name}"] = {"fitted": r["fitted"], "margin": r["margin"], "gain": r["gain_over_margin"], "sign_p": r["over_groups"]["sign_p"],
                                            "w": r["over_groups"]["w"], "l": r["over_groups"]["l"], "always_better_side": max(r["always_a"], r["always_b"])}
                p1.append(e58._ge(r["gain_over_margin"], MIN_GAIN) and e58._lt(r["over_groups"]["sign_p"], 0.05))
    g = R.get("geoid", {}).get("change")
    p2 = None if not g else bool(e58._ge(g["ratio_to_raw_diff"], MIN_CHANGE_RATIO) and e58._lt(g["over_events_vs_raw_diff"]["sign_p"], 0.05))
    complete = len(p1) == 8 and p2 is not None and not summary["failures"]
    summary["prereg"] = {"P1": all(p1) if p1 else None, "P1_detail": detail, "P1_n_tested": len(p1), "P2": p2,
                         "P2_detail": None if not g else {k: g[k] for k in ("ratio_to_raw_diff", "flooded_all", "flooded_rule", "flooded_both_confident")} | {"events_w": g["over_events_vs_raw_diff"]["w"], "events_l": g["over_events_vs_raw_diff"]["l"], "sign_p": g["over_events_vs_raw_diff"]["sign_p"]},
                         "min_gain": MIN_GAIN, "min_change_ratio": MIN_CHANGE_RATIO, "complete": bool(complete)}
    summary["n_failures"] = len(summary["failures"])
    print(f"prereg: P1 {summary['prereg']['P1']} ({len(p1)}/8) P2 {p2} {summary['prereg']['P2_detail']} | complete {complete}", flush=True)


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=PAIRS)
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp59 a label-fitted rule for which side to believe where two inferences disagree", "smoke": args.smoke, "device": DEV,
               "config": {"pairs": list(args.pairs), "features": FEATURES, "epochs": EPOCHS, "lr": LR, "min_gain": MIN_GAIN, "min_change_ratio": MIN_CHANGE_RATIO,
                          "fitting_set": "the 600 valid-split tiles the frozen heads were trained on (exp18's sample), both sides at four crop offsets; GEOID: the val events' clear chips",
                          "prereg": "P1 fitted rule beats the raw margin rule by >= 0.05 pooled and on more tiles than not (p < 0.05) on all 8 Sen1Floods11 pair-testbeds; "
                                    "P2 GEOID fitted change rule >= 1.5x the raw diff's flooded precision pooled and better on more events than not (p < 0.05)",
                          "jobs": [{"pairs": list(args.pairs), "python": sys.executable, "device": DEV, "started": time.strftime("%Y-%m-%d %H:%M:%S")}]},
               "results": {p: {} for p in PAIRS}, "failures": []}
    rows = []
    floods_pairs = [p for p in args.pairs if p != "geoid"]
    if floods_pairs:
        try:
            floods_dir, splits, feats = e57.load_floods(args, summary)
            testbeds = [n for n in TESTBEDS if n in splits]
            ctx = {"splits": splits, "feats": feats, "testbeds": testbeds, "draws": e57.draws_of(feats["train"][0], splits["train"][2]),
                   "inputs": {n: e57.testbed_inputs(splits, n, feats) for n in testbeds}}
            y_tr, ok_tr = e57.labels_of(splits["train"][2])
            ndwi_tr = e57.ndwi_cue_input(splits["train"][1])
            inputs = sen1floods_inputs(ctx, summary, args)
            for pair in floods_pairs:
                try:
                    pa, pb, tpa, tpb = inputs[pair]["train"]
                    X = features(pa, pb, ok_tr, tpa, tpb, ndwi_tr)
                    d = disagreement_set(pa, pb, ok_tr)
                    rule = Rule(X[d], ((pb > 0.5) == y_tr)[d])
                    summary["results"][pair]["rule"] = {"n_fit_windows": rule.n, "weights": rule.weights(), "bias": rule.b}
                    for name in testbeds:
                        t = ctx["inputs"][name]
                        pa, pb, tpa, tpb = inputs[pair][name]
                        r = grade(rule, features(pa, pb, t["ok"], tpa, tpb, t["ndwi"]), pa, pb, t["ok"], t["y"], t["groups"])
                        summary["results"][pair][name] = r
                        rows.append({"pair": pair, "testbed": name, **{k: v for k, v in r.items() if k != "over_groups"}, "tiles_w": r["over_groups"]["w"], "tiles_l": r["over_groups"]["l"], "sign_p": r["over_groups"]["sign_p"]})
                        print(f"{pair}/{name}: {r['n_disagree']} disagreement windows | fitted rule right {fmt(r['fitted'], '.3f')} vs margin {fmt(r['margin'], '.3f')} "
                              f"(gain {fmt(r['gain_over_margin'], '+.3f')}; tiles {r['over_groups']['w']}/{r['over_groups']['l']}, p={r['over_groups']['sign_p']:.2g}) | always a {fmt(r['always_a'], '.3f')}, b {fmt(r['always_b'], '.3f')} | believes b on {fmt(r['share_believe_b'], '.3f')}", flush=True)
                except Exception as ex:  # noqa: BLE001
                    e57.fail(summary, pair, ex)
        except Exception as ex:  # noqa: BLE001
            e57.fail(summary, "floods pairs", ex)
    if "geoid" in args.pairs:
        try:
            gi = geoid_inputs(args, summary)
            if gi is None:
                summary["results"]["geoid"]["note"] = "smoke: GEOID sample tiles not in the local Hub cache; pair skipped"
            else:
                v = gi["val"]
                Xv = features(v["pa"], v["pb"], v["ok"], v["tp_a"], v["tp_b"], v["ndwi"])
                dv = disagreement_set(v["pa"], v["pb"], v["ok"])
                rule = Rule(Xv[dv], (v["yb"] & ~v["ya"])[dv])
                te = gi["test"]
                c = grade_change(rule, features(te["pa"], te["pb"], te["ok"], te["tp_a"], te["tp_b"], te["ndwi"]), te["pa"], te["pb"], te["ok"], te["ya"], te["yb"], te["events"])
                summary["results"]["geoid"].update({"rule": {"n_fit_windows": rule.n, "weights": rule.weights(), "bias": rule.b}, "change": c})
                rows.append({"pair": "geoid change", "testbed": "GEOID-Flood test events", "n_disagree": c["n_disagree"], "flooded_all": c["flooded_all"], "flooded_rule": c["flooded_rule"],
                             "flooded_both_confident": c["flooded_both_confident"], "ratio": c["ratio_to_raw_diff"], "events_w": c["over_events_vs_raw_diff"]["w"], "events_l": c["over_events_vs_raw_diff"]["l"], "sign_p": c["over_events_vs_raw_diff"]["sign_p"]})
                print(f"geoid change rule: calls {fmt(c['share_change'], '.3f')} of {c['n_disagree']} disagreement windows change; flooded {fmt(c['flooded_rule'], '.3f')} of those vs {fmt(c['flooded_all'], '.3f')} of all "
                      f"({fmt(c['ratio_to_raw_diff'])}x; recall {fmt(c['recall_of_flooded'], '.3f')}); both-confident rule {fmt(c['flooded_both_confident'], '.3f')} | events vs raw diff {c['over_events_vs_raw_diff']['w']}/{c['over_events_vs_raw_diff']['l']} p={c['over_events_vs_raw_diff']['sign_p']:.2g}, "
                      f"vs both-confident {c['over_events_vs_both_confident']['w']}/{c['over_events_vs_both_confident']['l']} p={c['over_events_vs_both_confident']['sign_p']:.2g}", flush=True)
        except Exception as ex:  # noqa: BLE001
            e57.fail(summary, "geoid", ex)
    # merge a subset run into the existing artifact (the two-venv split, as exp58)
    sp = os.path.join(hb.OUT, f"exp59_summary{suffix}.json")
    if set(args.pairs) != set(PAIRS) and os.path.exists(sp):
        old = json.load(open(sp))
        for p in [p for p in PAIRS if p not in args.pairs and p in old.get("results", {})]:
            summary["results"][p] = old["results"][p]
        summary["config"]["jobs"] = old.get("config", {}).get("jobs", []) + summary["config"]["jobs"]
        summary["failures"] = [f for f in old.get("failures", []) if not any(f["part"].startswith(p) for p in args.pairs)] + summary["failures"]
        cp = os.path.join(hb.OUT, f"exp59_fitted_resolution{suffix}.csv")
        if os.path.exists(cp):
            with open(cp) as f:
                rows.extend(r for r in csv.DictReader(f) if not any(r["pair"].startswith(p) for p in args.pairs))
    try:
        prereg(summary)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(hb.OUT, f"exp59_fitted_resolution{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
