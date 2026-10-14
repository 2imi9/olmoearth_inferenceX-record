#!/usr/bin/env python
"""exp58: is the comparison tool better than the raw diff? Three label-free readings of a disagreement set, graded.

Why. exp57 measured the difference between two inferences of the same scene and showed where it sits; the question a
user asks next is whether the package tells them anything a raw disagreement map does not. A raw diff says which windows
changed and nothing else: every disagreement window is equally suspect, no side is preferred, and on two dates every
change is change. The package reads the same disagreement set label-free in three ways, and each reading is a testable
prediction about the labels.

Design. The pairs and grids of exp57 (exp57's loaders): crop offset 0 against 2, v1 against v1.2, the S2 head against
the S1 head, the frozen head against FT-S2 v1 (fine-tuned once more with exp52's recipe), each on Bolivia and the
multi-region test split of Sen1Floods11 on the W1 grid; the seven encoders against OlmoEarth Base on Ai2's embeddings
and probe (their test split); the pre-event S2 head against the post-event S1 head per GEOID-Flood event. On every
pair the two inferences are probability maps, so each side has a margin |p - 0.5| on every window. Readings, on the
windows where the two decisions differ:
  resolution  the tool believes the side with the larger margin (on a binary task the mean-probability decision). Graded:
              the share of disagreement windows where the chosen side matches the label. Baselines: the raw diff, which
              prefers no side (a coin flip, 0.5 in expectation, since exactly one side is right on a binary window);
              always the first inference; always the second. Recorded alongside, not preregistered: the same rule on
              rank-normalised margins (each side's margin as its midrank percentile over the testbed's valid windows),
              which removes a scale difference between two probability maps.
  targeting   the tool orders the disagreement windows by the first inference's own confidence, least confident first;
              the raw diff has no order. Graded: the share of the first inference's errors inside the least confident
              half of the set (random order captures 0.5 in expectation), and at 20%.
  change      on GEOID-Flood the two heads predict different things (permanent water before, water after the event), so a
              disagreement is change or error. The tool reads a window where both sides are confident (each margin at
              least CONFIDENT_MARGIN = 0.25, that is both probabilities at least 0.75 from the other side's class) as
              change; the raw diff reads every disagreement as change.
              Graded: the share of flooded-by-label windows among the both-confident disagreements against the share
              among all disagreements, pooled and per event.
  stability   descriptive: with three head draws (exp57's), the resolution rule on the disagreement windows that persist
              across all draws against those that appear in draw 0 only.
Per-tile and per-event tests use compare.over_groups on the groups with at least MIN_WINDOWS disagreement windows: a
one-sided exact sign test that the reading beats its baseline on more groups than not.

Preregistered (one-sided; the thresholds are what exp42's averaging gain and exp57's cross-tabs imply):
  P1  resolution beats the coin flip on every pair: the chosen side is right on at least 55% of the disagreement windows
      pooled, and on more tiles than not (p < 0.05), for the four Sen1Floods11 pairs on both testbeds and the seven
      encoder pairs (15 tests).
  P2  targeting beats the unordered set on every pair: the least confident half of the disagreement windows holds at
      least 55% of the first inference's errors pooled, and more than half on more tiles than not (p < 0.05); 15 tests.
  P3  on GEOID-Flood the both-confident disagreement windows are flooded at least 1.5 times as often as the disagreement
      windows as a whole, pooled over the 55 events, and more often on more events than not (p < 0.05; events with at
      least MIN_EVENT_WINDOWS disagreement windows).
  Falsification: P1 fails if any pair resolves at or below 55% or the per-tile test misses 0.05 (a margin rule would then
  be no better than the raw diff's silence on which side to believe); P2 fails likewise (the order would add nothing to
  the set); P3 fails if the ratio is below 1.5 or the event test misses (confidence on both sides would not separate
  change from error). Stated prediction, not tested: the resolution rule is more accurate on persistent than on
  transient disagreement windows for the three pairs with head draws.

Inputs. As exp57 (exp/out/exp18_feats.npz, data/floods, the v1 and v1.2 checkpoints, $HF_HOME/paper_embeddings,
$HF_HOME/geoid_flood/tree). Outputs. exp/out/exp58_summary.json (per pair and testbed the three readings with their
baselines and per-group tests, the stability split, prereg), exp/out/exp58_tool_vs_diff.csv (one row per pair x
testbed). --pairs selects pairs; a subset merges into the existing summary (the GEOID pair runs under the pinned venv).
Smoke as exp57's (CPU, no downloads).
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
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import over_groups  # noqa: E402
from oe_inferencex.metrics import capture_at_budget_expected  # noqa: E402
from oe_inferencex.signals import midrank_pct  # noqa: E402

PATCH, CROP, G, SHIFTS, DEV, SL = e57.PATCH, e57.CROP, e57.G, e57.SHIFTS, e57.DEV, e57.SL
PAIRS, TESTBEDS, DRAWS = e57.PAIRS, e57.TESTBEDS, e57.DRAWS
MIN_WINDOWS, MIN_EVENT_WINDOWS = 3, 20
MIN_RESOLUTION, MIN_CAPTURE, MIN_CHANGE_RATIO = 0.55, 0.55, 1.5
CONFIDENT_MARGIN = 0.25
BUDGETS = (0.2, 0.5)
fmt = e57.fmt


# ----------------------------------------------------------------------------- the readings
def _mean(x):
    return float(np.mean(x)) if np.size(x) else float("nan")


def readings(pa, pb, ok, y, groups, min_windows=MIN_WINDOWS):
    """The three label-free readings of the disagreement set of two probability maps, graded on y."""
    a, b = pa > 0.5, pb > 0.5
    d = (a != b) & ok
    ma, mb = np.abs(pa - 0.5), np.abs(pb - 0.5)
    pick = np.where(ma >= mb, a, b)
    right = pick == y
    ra, rb = np.full(ma.shape, np.nan), np.full(mb.shape, np.nan)
    if ok.any():
        ra[ok], rb[ok] = midrank_pct(ma[ok]), midrank_pct(mb[ok])
    right_rank = np.where(ra >= rb, a, b) == y
    err_a = a != y
    ids = np.unique(groups)
    res_g, cap_g = {}, {}
    for gid in ids:
        m = d & (groups == gid)[:, None, None]
        if int(m.sum()) < min_windows:
            continue
        res_g[gid.item()] = _mean(right[m]) - 0.5
        cap_g[gid.item()] = capture_at_budget_expected(-ma[m], err_a[m], budgets=(0.5,))[0.5] - 0.5
    cap = capture_at_budget_expected(-ma[d], err_a[d], budgets=BUDGETS) if d.any() else {b_: float("nan") for b_ in BUDGETS}
    return {"n_disagree": int(d.sum()), "n_groups_tested": len(res_g),
            "resolution": {"share_right": _mean(right[d]), "always_a": _mean((a == y)[d]), "always_b": _mean((b == y)[d]), "coin": 0.5,
                           "share_right_rank_normalised": _mean(right_rank[d]), "over_groups": over_groups(res_g)},
            "targeting": {"errors_a_in_set": int(err_a[d].sum()), "share_errors_a_in_set": _mean(err_a[d]),
                          "capture": {str(k): v for k, v in cap.items()}, "random": {str(k): k for k in BUDGETS},
                          "over_groups": over_groups(cap_g)},
            "arrays": {"disagree": d, "right": right}}


def stability_split(pa, pb, y, ok, masks_by_draw):
    """The resolution rule on the disagreement windows of draw 0 that persist across every draw, against the rest."""
    d0 = masks_by_draw[0]
    persistent = d0.copy()
    for m in masks_by_draw.values():
        persistent &= m
    transient = d0 & ~persistent
    a, b = pa > 0.5, pb > 0.5
    pick = np.where(np.abs(pa - 0.5) >= np.abs(pb - 0.5), a, b)
    right = pick == y
    return {"n_draws": len(masks_by_draw), "n_persistent": int(persistent.sum()), "n_transient": int(transient.sum()),
            "share_persistent": _mean(persistent[d0]) if d0.any() else float("nan"),
            "resolution_persistent": _mean(right[persistent]), "resolution_transient": _mean(right[transient]),
            "always_a_persistent": _mean((a == y)[persistent]), "always_a_transient": _mean((a == y)[transient])}


def change_reading(pa, pb, ok, ya, yb, events, min_windows=MIN_EVENT_WINDOWS):
    """GEOID-Flood: both-confident disagreement windows as change, against every disagreement window as change."""
    a, b = pa > 0.5, pb > 0.5
    d = (a != b) & ok
    ma, mb = np.abs(pa - 0.5), np.abs(pb - 0.5)
    both = (ma >= CONFIDENT_MARGIN) & (mb >= CONFIDENT_MARGIN)
    flooded = yb & ~ya
    err_a, err_b = a != ya, b != yb
    unconf_side_wrong = np.where(ma < mb, err_a, err_b)              # the less confident side's own error
    per = {}
    for ev in np.unique(events):
        m = d & (events == ev)[:, None, None]
        if m.sum() < min_windows or not (m & both).any():
            continue
        base, tool = _mean(flooded[m]), _mean(flooded[m & both])
        per[ev.item()] = {"n_disagree": int(m.sum()), "n_both_confident": int((m & both).sum()), "flooded_all": base, "flooded_both_confident": tool}
    return {"n_disagree": int(d.sum()), "confident_margin": CONFIDENT_MARGIN, "n_both_confident": int((d & both).sum()), "share_both_confident": _mean(both[d]),
            "flooded_all": _mean(flooded[d]), "flooded_both_confident": _mean(flooded[d & both]), "flooded_not_both_confident": _mean(flooded[d & ~both]),
            "ratio": (_mean(flooded[d & both]) / _mean(flooded[d])) if d.any() and _mean(flooded[d]) > 0 else float("nan"),
            "error_of_the_less_confident_side": {"all": _mean(unconf_side_wrong[d]), "not_both_confident": _mean(unconf_side_wrong[d & ~both]), "both_confident": _mean(unconf_side_wrong[d & both])},
            "per_event": per, "over_events": over_groups({k: v["flooded_both_confident"] - v["flooded_all"] for k, v in per.items()})}


def row_of(pair, testbed, r, extra=None):
    res, tg = r["resolution"], r["targeting"]
    out = {"pair": pair, "testbed": testbed, "n_disagree": r["n_disagree"], "groups_tested": r["n_groups_tested"],
           "resolution_share_right": res["share_right"], "resolution_rank_normalised": res["share_right_rank_normalised"], "always_a": res["always_a"], "always_b": res["always_b"],
           "resolution_groups_w": res["over_groups"]["w"], "resolution_groups_l": res["over_groups"]["l"], "resolution_sign_p": res["over_groups"]["sign_p"],
           "capture_20": tg["capture"]["0.2"], "capture_50": tg["capture"]["0.5"], "share_errors_a_in_set": tg["share_errors_a_in_set"],
           "targeting_groups_w": tg["over_groups"]["w"], "targeting_groups_l": tg["over_groups"]["l"], "targeting_sign_p": tg["over_groups"]["sign_p"]}
    if extra:
        out.update(extra)
    return out


def report(pair, testbed, r):
    res, tg, st = r["resolution"], r["targeting"], r.get("stability")
    print(f"{pair}/{testbed}: {r['n_disagree']} disagreement windows | resolution right {fmt(res['share_right'], '.3f')} (rank-normalised {fmt(res['share_right_rank_normalised'], '.3f')}; always a {fmt(res['always_a'], '.3f')}, "
          f"always b {fmt(res['always_b'], '.3f')}; tiles {res['over_groups']['w']}/{res['over_groups']['l']}, p={res['over_groups']['sign_p']:.2g}) | "
          f"targeting: a's errors {fmt(tg['share_errors_a_in_set'], '.3f')} of the set, least-confident half holds {fmt(tg['capture']['0.5'], '.3f')}, "
          f"fifth holds {fmt(tg['capture']['0.2'], '.3f')} (tiles {tg['over_groups']['w']}/{tg['over_groups']['l']}, p={tg['over_groups']['sign_p']:.2g})"
          + (f" | persistent {st['n_persistent']} of {r['n_disagree']}: resolution {fmt(st['resolution_persistent'], '.3f')} vs transient {fmt(st['resolution_transient'], '.3f')}" if st else ""), flush=True)


# ----------------------------------------------------------------------------- the pairs, probabilities per draw
def probabilities_sen1floods(ctx, summary, args):
    """{pair: {testbed: {draw: (pa, pb)}}} on the W1 grid for the four Sen1Floods11 pairs."""
    out = {p: {n: {} for n in ctx["testbeds"]} for p in ("offsets", "backbones", "sensors", "finetune")}
    tr_lab = ctx["splits"]["train"][2]
    heads_s2 = {d: e57.fit_head(ctx["feats"]["train"][0], tr_lab, idx) for d, idx in ctx["draws"].items()}
    if "offsets" in args.pairs:
        for name in ctx["testbeds"]:
            for d, head in heads_s2.items():
                p = e57.probs(ctx["inputs"][name]["feats"], head)
                out["offsets"][name][d] = (e57.offset_windows(p[0], 0), e57.offset_windows(p[2], 2))
    if "backbones" in args.pairs:
        t0 = time.time()
        model, label = e57.second_backbone(args)
        tr2 = e57.encode_s2(model, ctx["splits"]["train"][1], shifts=(0,))[0]
        f2 = {name: e57.encode_s2(model, ctx["splits"][name][1]) for name in ctx["testbeds"]}
        del model
        summary["config"]["second_backbone"] = label
        summary["results"]["backbones"]["encode_seconds"] = time.time() - t0
        for name in ctx["testbeds"]:
            for d, idx in ctx["draws"].items():
                h2 = e57.fit_head(tr2, tr_lab, idx)
                out["backbones"][name][d] = (e57.w1(e57.probs(ctx["inputs"][name]["feats"], heads_s2[d])), e57.w1(e57.probs(f2[name], h2)))
    if "sensors" in args.pairs:
        t0 = time.time()
        model = hb.load_model()
        tr1 = e57.encode_s1(model, ctx["splits"]["train"][0], shifts=(0,))[0]
        f1 = {name: e57.encode_s1(model, ctx["splits"][name][0]) for name in ctx["testbeds"]}
        del model
        summary["results"]["sensors"]["encode_seconds"] = time.time() - t0
        for name in ctx["testbeds"]:
            for d, idx in ctx["draws"].items():
                h1 = e57.fit_head(tr1, tr_lab, idx)
                out["sensors"][name][d] = (e57.w1(e57.probs(ctx["inputs"][name]["feats"], heads_s2[d])), e57.w1(e57.probs(f1[name], h1)))
    if "finetune" in args.pairs:
        import exp52_finetune as e52
        t0 = time.time()
        floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
        n = args.smoke_tiles if args.smoke else None
        tr_s1, tr_s2, tr_l = e51.load_ours(floods_dir, "valid" if args.smoke else "train", n, seed=3)
        va_s1, va_s2, va_l = e51.load_ours(floods_dir, "valid", n, seed=0)
        tr = (*e52.prep(tr_s2, tr_s1, 0), tr_l[:, :CROP, :CROP])
        va = (*e52.prep(va_s2, va_s1, 0), va_l[:, :CROP, :CROP])
        epochs = 1 if args.smoke else e52.EPOCHS
        model = hb.load_model()
        ft, history, best = e52.finetune(model, ["s2"], tr, va, epochs, log=lambda m: print(f"  FT-S2 v1{m}", flush=True))
        summary["results"]["finetune"]["train"] = {"train_tiles": int(len(tr_s2)), "epochs": epochs, "best_val_miou": best, "train_seconds": time.time() - t0}
        for name in ctx["testbeds"]:
            t = ctx["inputs"][name]
            pshift = []
            for s in SHIFTS:
                x2, x1 = e52.prep(t["s2"], t["s1"], s)
                p = e52.predict(ft, x2, x1)
                pshift.append(p.reshape(len(p), G, PATCH, G, PATCH).mean(axis=(2, 4)))
            out["finetune"][name][0] = (e57.w1(e57.probs(t["feats"], heads_s2[0])), e57.w1(np.stack(pshift)))
        del ft, model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def geoid_probabilities(args, summary):
    """exp57's GEOID inputs: the two heads' W1 probabilities on the clear test chips, both task labels, events."""
    import exp55_geoid_flood as e55
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    data_dir = os.path.join(hf_home, "geoid_flood")
    tiles = e57.geoid_tiles(args, data_dir)
    if tiles is None:
        return None
    if not args.smoke and any(not tiles[split] for split in ("test", "val")):
        # the extracted tree was emptied after exp57 (the shard markers survived): drop the markers and refetch exp55's subset
        import shutil
        root = os.path.join(data_dir, "tree")
        files = e55.hub_files()
        refetched = {}
        for split, n in (("test", 4), ("val", 2)):
            if not tiles[split]:
                shutil.rmtree(os.path.join(root, split, ".done"), ignore_errors=True)
                refetched[split] = e55.fetch_split(files, split, n, data_dir, root)
                tiles[split] = e55.scan_split(root, split)
        summary["config"]["geoid_refetched"] = {k: len(v) for k, v in refetched.items()}
        print("geoid: tree was empty, refetched " + ", ".join(f"{k} ({len(v)} shards, {len(tiles[k])} tiles)" for k, v in refetched.items()), flush=True)
    data = {}
    for split in ("test", "val"):
        cand = e55.chip_candidates(tiles[split], split)
        sel = e55.select_chips(cand, 400, smoke_chips=4 if args.smoke else None)
        s2, s1, lab = e55.read_imagery(tiles[split], sel)
        data[split] = {"s2": s2, "s1": s1, "lab": lab, "aoi": np.array([x["aoi"] for x in sel]), "clear": np.array([x["clear"] >= e55.MIN_CLEAR for x in sel], bool)}
    val, te = data["val"], data["test"]
    model = hb.load_model()
    heads, p_shift = {}, {}
    for task, sensor in (("A", "s2"), ("B", "s1")):
        sel_val = val["clear"] if sensor == "s2" else np.ones(len(val["lab"]), bool)
        heads[task] = e55.fit_head(e55.features(model, sensor, val[sensor][sel_val], 0), e55.task_labels(val["lab"][sel_val], task))
    idx_a = np.flatnonzero(te["clear"])
    for task, sensor in (("A", "s2"), ("B", "s1")):
        p_shift[task] = np.stack([exp18.head_prob_logit(e55.features(model, sensor, te[sensor][idx_a], s), *heads[task])[0] for s in SHIFTS])
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    lab = te["lab"][idx_a]
    ya, oka = e57.labels_of(e55.task_labels(lab, "A"))
    yb, okb = e57.labels_of(e55.task_labels(lab, "B"))
    summary["results"]["geoid"]["chips"] = {"clear_test_chips": int(len(idx_a)), "events": int(len(set(te["aoi"][idx_a].tolist())))}
    return {"pa": e57.w1(p_shift["A"]), "pb": e57.w1(p_shift["B"]), "ok": oka & okb, "ya": ya, "yb": yb, "events": te["aoi"][idx_a]}


# ----------------------------------------------------------------------------- prereg and main
def _ge(v, t):
    """v >= t with an undefined v (None after a JSON round trip, NaN) counting as a failure."""
    return v is not None and np.isfinite(v) and v >= t


def _lt(v, t):
    return v is not None and np.isfinite(v) and v < t


def prereg(summary):
    R = summary["results"]
    p1, p2, detail = [], [], {}
    for pair in ("offsets", "backbones", "sensors", "finetune"):
        for name in TESTBEDS:
            r = R.get(pair, {}).get(name)
            if r and "resolution" in r:
                res, tg = r["resolution"], r["targeting"]
                detail[f"{pair}/{name}"] = {"share_right": res["share_right"], "resolution_p": res["over_groups"]["sign_p"], "capture_50": tg["capture"]["0.5"], "targeting_p": tg["over_groups"]["sign_p"]}
                p1.append(_ge(res["share_right"], MIN_RESOLUTION) and _lt(res["over_groups"]["sign_p"], 0.05))
                p2.append(_ge(tg["capture"]["0.5"], MIN_CAPTURE) and _lt(tg["over_groups"]["sign_p"], 0.05))
    for model, r in R.get("encoders", {}).items():
        if isinstance(r, dict) and "resolution" in r:
            res, tg = r["resolution"], r["targeting"]
            detail[f"encoders/{model}"] = {"share_right": res["share_right"], "resolution_p": res["over_groups"]["sign_p"], "capture_50": tg["capture"]["0.5"], "targeting_p": tg["over_groups"]["sign_p"]}
            p1.append(_ge(res["share_right"], MIN_RESOLUTION) and _lt(res["over_groups"]["sign_p"], 0.05))
            p2.append(_ge(tg["capture"]["0.5"], MIN_CAPTURE) and _lt(tg["over_groups"]["sign_p"], 0.05))
    expected = 8 + len(summary["config"]["others"])
    g = R.get("geoid", {}).get("change")
    p3 = None
    if g:
        p3 = bool(_ge(g["ratio"], MIN_CHANGE_RATIO) and _lt(g["over_events"]["sign_p"], 0.05))
    complete = len(p1) == expected and p3 is not None and not summary["failures"]
    summary["prereg"] = {"P1": all(p1) if p1 else None, "P2": all(p2) if p2 else None, "P3": p3, "detail": detail, "n_tested": len(p1), "n_expected": expected,
                         "P3_detail": None if not g else {"ratio": g["ratio"], "flooded_all": g["flooded_all"], "flooded_both_confident": g["flooded_both_confident"], "events_w": g["over_events"]["w"], "events_l": g["over_events"]["l"], "sign_p": g["over_events"]["sign_p"]},
                         "min_resolution": MIN_RESOLUTION, "min_capture": MIN_CAPTURE, "min_change_ratio": MIN_CHANGE_RATIO, "complete": bool(complete)}
    summary["n_failures"] = len(summary["failures"])
    print(f"prereg: P1 {summary['prereg']['P1']} P2 {summary['prereg']['P2']} ({len(p1)}/{expected} pairs tested) P3 {p3} {summary['prereg']['P3_detail']} | complete {complete}", flush=True)


def merge_previous(summary, rows, suffix, pairs):
    sp = os.path.join(hb.OUT, f"exp58_summary{suffix}.json")
    cp = os.path.join(hb.OUT, f"exp58_tool_vs_diff{suffix}.csv")
    if set(pairs) == set(PAIRS) or not os.path.exists(sp):
        return
    old = json.load(open(sp))
    kept = [p for p in PAIRS if p not in pairs and p in old.get("results", {})]
    for p in kept:
        summary["results"][p] = old["results"][p]
    summary["config"]["jobs"] = old.get("config", {}).get("jobs", []) + summary["config"]["jobs"]
    summary["failures"] = [f for f in old.get("failures", []) if not any(f["part"].startswith(p) for p in pairs)] + summary["failures"]
    if os.path.exists(cp):
        with open(cp) as f:
            rows.extend(r for r in csv.DictReader(f) if not any(r["pair"].startswith(p) for p in pairs))
    summary["config"]["merged_from"] = kept


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=PAIRS, help="pairs to run (a subset merges into the existing artifact)")
    ap.add_argument("--others", nargs="*", default=e51.OTHERS, help="the other encoders of the encoders pair")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp58 is the comparison tool better than the raw diff: three label-free readings of a disagreement set, graded", "smoke": args.smoke, "device": DEV,
               "config": {"pairs": list(args.pairs), "others": list(args.others), "draws": list(DRAWS), "min_windows": MIN_WINDOWS, "min_event_windows": MIN_EVENT_WINDOWS, "budgets": list(BUDGETS), "confident_margin": CONFIDENT_MARGIN,
                          "prereg": "P1 resolution by the larger margin right on >= 55% of the disagreement windows pooled and on more tiles than not (p < 0.05) for every pair; "
                                    "P2 the least confident half of the disagreement set holds >= 55% of the first inference's errors pooled and more than half on more tiles than not (p < 0.05) for every pair; "
                                    "P3 on GEOID-Flood the both-confident disagreement windows are flooded >= 1.5x as often as all disagreement windows, pooled, and more often on more events than not (p < 0.05)",
                          "jobs": [{"pairs": list(args.pairs), "python": sys.executable, "device": DEV, "started": time.strftime("%Y-%m-%d %H:%M:%S")}]},
               "results": {p: {} for p in PAIRS}, "failures": []}
    rows = []
    floods_pairs = [p for p in args.pairs if p in ("offsets", "backbones", "sensors", "finetune")]
    if floods_pairs:
        try:
            floods_dir, splits, feats = e57.load_floods(args, summary)
            testbeds = [n for n in TESTBEDS if n in splits]
            ctx = {"splits": splits, "feats": feats, "testbeds": testbeds, "draws": e57.draws_of(feats["train"][0], splits["train"][2]),
                   "inputs": {n: e57.testbed_inputs(splits, n, feats) for n in testbeds}}
            probs_by = probabilities_sen1floods(ctx, summary, args)
            for pair in floods_pairs:
                for name in testbeds:
                    try:
                        t = ctx["inputs"][name]
                        by_draw = probs_by[pair][name]
                        pa, pb = by_draw[0]
                        r = readings(pa, pb, t["ok"], t["y"], t["groups"])
                        if len(by_draw) > 1:
                            masks = {d: ((p_a > 0.5) != (p_b > 0.5)) & t["ok"] for d, (p_a, p_b) in by_draw.items()}
                            r["stability"] = stability_split(pa, pb, t["y"], t["ok"], masks)
                        r.pop("arrays")
                        summary["results"][pair][name] = r
                        rows.append(row_of(pair, name, r))
                        report(pair, name, r)
                    except Exception as ex:  # noqa: BLE001
                        e57.fail(summary, f"{pair} {name}", ex)
        except Exception as ex:  # noqa: BLE001
            e57.fail(summary, "floods pairs", ex)
    if "encoders" in args.pairs:
        try:
            cache_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "paper_embeddings")
            base = e57.their_decision(e51.THEIRS, cache_dir, args)
            pos0 = {k: i for i, k in enumerate(base["keys"])}
            groups = np.arange(len(base["wp"]))
            for model in args.others:
                try:
                    d = e57.their_decision(model, cache_dir, args)
                    m = np.array([pos0.get(k, -1) for k in d["keys"]])
                    mi = np.flatnonzero(m >= 0)
                    pb, ok_b = np.full(base["wp"].shape, 0.5), np.zeros(base["ok"].shape, bool)
                    pb[m[mi]], ok_b[m[mi]] = d["wp"][mi], d["ok"][mi]
                    r = readings(base["wp"], pb, base["ok"] & ok_b, base["y"], groups)
                    r.pop("arrays")
                    r["aligned_tiles"] = int(len(mi))
                    summary["results"]["encoders"][model] = r
                    rows.append(row_of(f"encoders olmoearth_base vs {model}", "their test split", r))
                    report(f"encoders olmoearth_base vs {model}", "their test split", r)
                except Exception as ex:  # noqa: BLE001
                    e57.fail(summary, f"encoders {model}", ex)
        except Exception as ex:  # noqa: BLE001
            e57.fail(summary, "encoders", ex)
    if "geoid" in args.pairs:
        try:
            t0 = time.time()
            gi = geoid_probabilities(args, summary)
            if gi is None:
                summary["results"]["geoid"]["note"] = "smoke: GEOID sample tiles not in the local Hub cache; pair skipped"
            else:
                r = readings(gi["pa"], gi["pb"], gi["ok"], gi["yb"], gi["events"], min_windows=MIN_EVENT_WINDOWS)   # resolution and targeting graded on water after the event
                r.pop("arrays")
                r["graded_against"] = "task B labels (water after the event)"
                r["change"] = change_reading(gi["pa"], gi["pb"], gi["ok"], gi["ya"], gi["yb"], gi["events"])
                r["seconds"] = time.time() - t0
                summary["results"]["geoid"].update(r)
                c = r["change"]
                rows.append(row_of("geoid S2 head vs S1 head", "GEOID-Flood pooled", r, extra={"flooded_all": c["flooded_all"], "flooded_both_confident": c["flooded_both_confident"], "change_ratio": c["ratio"],
                                                                                                 "change_events_w": c["over_events"]["w"], "change_events_l": c["over_events"]["l"], "change_sign_p": c["over_events"]["sign_p"]}))
                report("geoid S2 head vs S1 head", "GEOID-Flood (graded on water after the event)", r)
                print(f"  change: {c['n_both_confident']} of {c['n_disagree']} disagreement windows have both sides confident; flooded by the label {fmt(c['flooded_both_confident'], '.3f')} of them vs "
                      f"{fmt(c['flooded_all'], '.3f')} of all disagreements ({fmt(c['ratio'])}x; events {c['over_events']['w']}/{c['over_events']['l']}, p={c['over_events']['sign_p']:.2g}); "
                      f"the less confident side is wrong on {fmt(c['error_of_the_less_confident_side']['not_both_confident'], '.3f')} of the other windows", flush=True)
        except Exception as ex:  # noqa: BLE001
            e57.fail(summary, "geoid", ex)
    merge_previous(summary, rows, suffix, args.pairs)
    try:
        prereg(summary)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp58_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(hb.OUT, f"exp58_tool_vs_diff{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
