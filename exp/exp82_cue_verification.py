"""exp82: is the "why" verified where it is quoted? The explain module's cue enrichments, measured per task on the
suite's segmentation exports with tile-clustered intervals. Preregistered in docs/plan/cue_verification.md.

    python exp/exp82_cue_verification.py --stage estimate [--units DIR] [--tasks ...] [--n-boot 2000]
    python exp/exp82_cue_verification.py --stage grade
    python exp/exp82_cue_verification.py --stage smoke

Numpy only. The cues are computed exactly as the tool computes them on a raster: the boundary through
`assess._boundary_valid` on each tile's pooled decision map, the low-confidence cue as the least confident 20% of
the task's valid windows. The tile-clustered bootstrap of the enrichment resamples tiles and sums per-tile counts,
the same statistic `explain.cue_enrichment` computes by resampling windows within tiles; on MADOS the two agree
(tests/test_exp82.py). Errors are the export's; labels never define a cue.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)
from oe_inferencex import metrics                 # noqa: E402
from oe_inferencex.assess import _boundary_valid  # noqa: E402
from oe_inferencex.explain import CUES            # noqa: E402
import exp70_task_suite as e70                    # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SUMMARY = os.path.join(OUT, "exp82_summary.json")
N_BOOT, LOW_Q, BUDGETS = 2000, 0.2, (0.05, 0.10, 0.20)
# the preregistered bars, docs/plan/cue_verification.md
GATE_TOL = 2e-3                 # the low-confidence share among errors equals exp70's capture at 20%
P1_RATIO, P1_MAX_CLASSES = 2.0, 10
P2_MIN_TASKS = 4
LIBRARY_RATIO = CUES["boundary"].share_errors / CUES["boundary"].share_correct     # 0.750 / 0.214


# ----------------------------------------------------------------------------- units and cues
def load_export(task, units_dir):
    d = np.load(os.path.join(units_dir, f"{task}.npz"), allow_pickle=False)
    ok = np.unpackbits(d["ok_packed"])[:int(d["ok_len"][0])].astype(bool)
    N, hw, ww = (int(v) for v in d["grid"])
    valid = ok.reshape(N, hw, ww)
    hard = np.zeros((N, hw, ww), np.int64)
    hard[valid] = d["dec"].astype(np.int64)
    tile = np.broadcast_to(np.arange(N)[:, None, None], (N, hw, ww))[valid]
    return {"hard": hard, "valid": valid, "tile": tile, "margin": d["margin"].astype(np.float64),
            "err": d["err"].astype(np.float64), "y": d["y"].astype(np.int64), "dec": d["dec"].astype(np.int64),
            "n_classes": int(d["n_classes"][0]), "family": str(d["family"][0])}


def coarsen_export(u, k):
    """Addendum of 2026-09-23: the export at a k x k coarser window grid. Each block's decision and reference are
    the majority class over its valid fine windows (ties to the smallest class), its validity is any valid fine
    window, and its confidence the mean fine margin, a proxy used only for the low-confidence cue."""
    N, hw, ww = u["valid"].shape
    if hw % k or ww % k:
        raise ValueError(f"grid {hw}x{ww} does not divide by {k}")
    valid = u["valid"]
    full_y = np.zeros((N, hw, ww), np.int64); full_y[valid] = u["y"]
    full_m = np.zeros((N, hw, ww), np.float64); full_m[valid] = u["margin"]
    C = int(max(u["n_classes"], u["hard"].max() + 1, full_y.max() + 1))
    blocks = lambda a: a.reshape(N, hw // k, k, ww // k, k)

    def majority(full):
        counts = np.stack([blocks((full == c) & valid).sum(axis=(2, 4)) for c in range(C)], -1)   # (N, H, W, C)
        return counts.argmax(-1), counts.sum(-1)                                                  # argmax: first = smallest class
    hard_c, n_valid = majority(u["hard"])
    y_c, _ = majority(full_y)
    valid_c = n_valid > 0
    margin_c = np.where(valid_c, blocks(full_m).sum(axis=(2, 4)) / np.maximum(n_valid, 1), 0.0)
    hard_c = np.where(valid_c, hard_c, 0)
    tile = np.broadcast_to(np.arange(N)[:, None, None], valid_c.shape)[valid_c]
    return {"hard": hard_c, "valid": valid_c, "tile": tile, "margin": margin_c[valid_c],
            "err": (hard_c != y_c)[valid_c].astype(np.float64), "y": y_c[valid_c], "dec": hard_c[valid_c],
            "n_classes": u["n_classes"], "family": u["family"], "coarsen": k}


def boundary_cue(hard, valid):
    """The tool's boundary cue per valid window: `_boundary_valid` tile by tile, indicator > 0."""
    out = np.zeros(hard.shape, np.float64)
    for i in range(hard.shape[0]):
        out[i] = _boundary_valid(hard[i], valid[i])
    return out[valid] > 0


def low_confidence_cue(margin, q=LOW_Q):
    return margin <= np.quantile(margin, q)


def low_confidence_per_tile(margin, tile, q=LOW_Q):
    cue = np.zeros(margin.size, bool)
    o = np.argsort(tile, kind="stable")
    u, starts, counts = np.unique(tile[o], return_index=True, return_counts=True)
    for a, c in zip(starts, counts):
        idx = o[a:a + c]
        cue[idx] = margin[idx] <= np.quantile(margin[idx], q)
    return cue


def per_tile_counts(cue, err, tile):
    """Per tile: errors with the cue, errors, correct with the cue, correct: the sufficient statistics of the shares."""
    e = err > 0.5
    T = int(tile.max()) + 1
    return np.stack([np.bincount(tile, weights=(cue & e), minlength=T), np.bincount(tile, weights=e, minlength=T),
                     np.bincount(tile, weights=(cue & ~e), minlength=T), np.bincount(tile, weights=~e, minlength=T)], 1)


def enrichment_from_counts(counts, n_boot=N_BOOT, seed=0):
    """Shares, ratio, precision and a tile-clustered bootstrap interval of the ratio, from per-tile counts. Resampling
    tiles with replacement and summing counts is the statistic explain.cue_enrichment computes window by window."""
    tot = counts.sum(0)
    s_e, s_c = tot[0] / tot[1], tot[2] / tot[3]
    out = {"n": int(tot[1] + tot[3]), "n_errors": int(tot[1]), "n_with_cue": int(tot[0] + tot[2]), "n_tiles": int(counts.shape[0]),
           "share_errors": float(s_e), "share_correct": float(s_c), "enrichment": float(s_e / s_c) if s_c > 0 else float("nan"),
           "precision": float(tot[0] / (tot[0] + tot[2])) if tot[0] + tot[2] > 0 else float("nan")}
    rng = np.random.default_rng(seed)
    T = counts.shape[0]
    ratios, skipped = [], 0
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, T, T), minlength=T).astype(float)
        t = w @ counts
        if t[1] == 0 or t[3] == 0 or t[2] == 0:
            skipped += 1
            continue
        ratios.append((t[0] / t[1]) / (t[2] / t[3]))
    r = np.array(ratios)
    out.update({"boot_lo": float(np.percentile(r, 2.5)) if r.size else float("nan"),
                "boot_hi": float(np.percentile(r, 97.5)) if r.size else float("nan"),
                "n_boot": int(r.size), "boot_skipped": int(skipped), "clustered": True})
    return out


def captures(margin, boundary, err):
    """Error capture at the budgets for the confidence order and the boundary-first order (boundary windows by
    confidence, then the rest by confidence), tie-aware."""
    conf = -margin
    lexi = boundary.astype(np.float64) * (conf.max() - conf.min() + 1.0) + conf
    return {"confidence": metrics.capture_at_budget_expected(conf, err, BUDGETS),
            "boundary_first": metrics.capture_at_budget_expected(lexi, err, BUDGETS)}


def confusion_pairs(y, dec, top=3):
    e = dec != y
    if not e.any():
        return {"n_errors": 0, "top_pairs": [], "share_top": float("nan")}
    pairs, counts = np.unique(np.stack([dec[e], y[e]], 1), axis=0, return_counts=True)
    o = np.lexsort((pairs[:, 1], pairs[:, 0], -counts))[:top]     # ties broken by (predicted, reference), stably
    return {"n_errors": int(e.sum()), "n_pairs": int(counts.size),
            "top_pairs": [{"predicted": int(pairs[i, 0]), "reference": int(pairs[i, 1]), "share": float(counts[i] / e.sum())} for i in o],
            "share_top": float(counts[o].sum() / e.sum())}


# ----------------------------------------------------------------------------- stages
def estimate_task(task, units_dir, n_boot, record, coarsen=1):
    t0 = time.time()
    u = load_export(task, units_dir)
    if coarsen > 1:
        u = coarsen_export(u, coarsen)
    margin, err, tile = u["margin"], u["err"], u["tile"]
    bnd = boundary_cue(u["hard"], u["valid"])
    low = low_confidence_cue(margin)
    low_tile = low_confidence_per_tile(margin, tile)
    cues = {"boundary": bnd, "low_confidence": low, "boundary_and_low_confidence": bnd & low, "low_confidence_per_tile": low_tile}
    row = {"n_units": int(err.size), "n_tiles": int(tile.max()) + 1, "n_classes": u["n_classes"], "error_rate": float(err.mean()),
           "coarsen": coarsen, "boundary_prevalence": float(bnd.mean()),
           "recorded_capture_20": record.get(task, {}).get("signals", {}).get("margin", {}).get("capture", {}).get("0.2") if coarsen == 1 else None,
           "cues": {k: enrichment_from_counts(per_tile_counts(v, err, tile), n_boot) for k, v in cues.items()},
           "captures": captures(margin, bnd, err), "seconds": round(time.time() - t0, 1)}
    c = row["cues"]
    print(f"  {task:28s} C={u['n_classes']:2d} boundary {c['boundary']['enrichment']:.2f} [{c['boundary']['boot_lo']:.2f}, {c['boundary']['boot_hi']:.2f}] "
          f"low {c['low_confidence']['enrichment']:.2f} both prec {c['boundary_and_low_confidence']['precision']:.3f} vs "
          f"{c['boundary']['precision']:.3f}/{c['low_confidence']['precision']:.3f} {row['seconds']:.0f}s", flush=True)
    return row


def estimate_classification(task, units_dir):
    d = np.load(os.path.join(units_dir, f"{task}.npz"), allow_pickle=False)
    y, dec = d["y"].astype(np.int64), d["dec"].astype(np.int64)
    return {"n_units": int(y.size), "n_classes": int(max(y.max(), dec.max()) + 1), "confusion": confusion_pairs(y, dec)}


# ----------------------------------------------------------------------------- verdicts, pure functions of the rows
def gate(rows, tol=GATE_TOL):
    per = {t: {"share_errors_low_confidence": r["cues"]["low_confidence"]["share_errors"], "recorded_capture_20": r["recorded_capture_20"]}
           for t, r in rows.items()}
    bad = [t for t, v in per.items() if v["recorded_capture_20"] is None or abs(v["share_errors_low_confidence"] - v["recorded_capture_20"]) > tol]
    return {"holds": bool(per) and not bad, "failing": bad, "per_task": per, "tolerance": tol}


def grade_p1(rows, ratio=P1_RATIO, max_classes=P1_MAX_CLASSES):
    few = {t: r["cues"]["boundary"]["enrichment"] for t, r in rows.items() if r["n_classes"] <= max_classes}
    a = bool(few) and all(v >= ratio for v in few.values())
    two = [r["cues"]["boundary"]["enrichment"] for r in rows.values() if r["n_classes"] == 2]
    many = [r["cues"]["boundary"]["enrichment"] for r in rows.values() if r["n_classes"] == 19]
    b = bool(two) and bool(many) and float(np.median(many)) < min(two)
    return {"holds": a and b, "a_few_classes_at_least_2": a, "few_class_ratios": few,
            "b_19_class_below_2_class": b, "ratio_2_class": two, "ratio_19_class_median": float(np.median(many)) if many else None,
            "ratio_by_task": {t: (r["n_classes"], r["cues"]["boundary"]["enrichment"]) for t, r in rows.items()}}


def grade_p2(rows, library=LIBRARY_RATIO, min_tasks=P2_MIN_TASKS):
    outside = {t: not (r["cues"]["boundary"]["boot_lo"] <= library <= r["cues"]["boundary"]["boot_hi"]) for t, r in rows.items()}
    n = sum(outside.values())
    return {"holds": n >= min_tasks, "n_outside": n, "of": len(outside), "library_ratio": library, "outside_by_task": outside,
            "intervals": {t: (r["cues"]["boundary"]["boot_lo"], r["cues"]["boundary"]["boot_hi"]) for t, r in rows.items()}}


def grade_p3(rows):
    per = {}
    for t, r in rows.items():
        c = r["cues"]
        per[t] = {"both": c["boundary_and_low_confidence"]["precision"], "boundary": c["boundary"]["precision"],
                  "low_confidence": c["low_confidence"]["precision"]}
        per[t]["beats_boundary"] = per[t]["both"] > per[t]["boundary"]
        per[t]["beats_low_confidence"] = per[t]["both"] > per[t]["low_confidence"]
    return {"holds": bool(per) and all(v["beats_boundary"] and v["beats_low_confidence"] for v in per.values()),
            "n_beats_boundary": sum(v["beats_boundary"] for v in per.values()),
            "n_beats_low_confidence": sum(v["beats_low_confidence"] for v in per.values()), "of": len(per), "per_task": per}


def verdicts(rows):
    return {"gate_low_confidence_reproduces_exp70": gate(rows), "P1_boundary_real_where_classes_are_few": grade_p1(rows),
            "P2_library_value_outside_the_interval": grade_p2(rows), "P3_two_reasons_are_two_reasons": grade_p3(rows)}


def cmd_estimate(args):
    import exp78_error_rate_estimation as e78
    units_dir = args.units or e78.UNITS
    record = json.load(open(os.path.join(OUT, "exp70_summary.json")))["results"]["tasks"]
    if args.encoder != "olmoearth_base":
        record = json.load(open(os.path.join(OUT, "exp74_summary.json")))["results"]["tasks"].get(args.encoder, {})
    tasks = args.tasks or e70.TASKS_SEG
    rows, t0 = {}, time.time()
    for t in tasks:
        if not os.path.exists(os.path.join(units_dir, f"{t}.npz")):
            print(f"  {t}: no export", flush=True)
            continue
        rows[t] = estimate_task(t, units_dir, args.n_boot, record, args.coarsen)
    cls = {}
    for t in e70.TASKS_CLS:
        p = os.path.join(units_dir, f"{t}.npz")
        if os.path.exists(p):
            r = estimate_classification(t, units_dir)
            if r["n_classes"] >= 6:
                cls[t] = r
    summary = {"experiment": "exp82 is the why verified where it is quoted", "encoder": args.encoder, "units": os.path.relpath(units_dir, ROOT),
               "config": {"n_boot": args.n_boot, "low_confidence_quantile": LOW_Q, "budgets": BUDGETS, "library_ratio": LIBRARY_RATIO,
                          "seed": 0, "seconds": round(time.time() - t0)},
               "tasks": rows, "classification_confusion_pairs": cls, "prereg": verdicts(rows)}
    # the graded run is OlmoEarth Base's; another encoder's export (exp79) writes beside it, never over it
    path = SUMMARY if args.encoder == "olmoearth_base" else os.path.join(OUT, "exp82_cues", f"{args.encoder}.json")
    if args.coarsen > 1:   # the addendum's coarsened grids write beside the graded runs, never over them
        summary["prereg"] = {"note": "gate and P1-P3 are graded at the fine grain only (docs/plan/cue_verification.md, addendum)"}
        path = os.path.join(OUT, "exp82_cues", f"{args.encoder}_x{args.coarsen}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") if isinstance(v, dict) else v for k, v in summary["prereg"].items()}, indent=1))
    print(f"wrote {path}")
    return 0


def cmd_grade(args):
    d = json.load(open(SUMMARY))
    d["prereg"] = verdicts(d["tasks"])
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in d["prereg"].items()}, indent=1))
    return 0


GRAIN_FILE = os.path.join(OUT, "exp82_grain.json")
GRAINS = (1, 2, 4)


def risk_ratio(b):
    """Error rate inside the cue set over the rate outside, from the recorded counts."""
    inside = b["precision"]
    n_out = b["n"] - b["n_with_cue"]
    if n_out <= 0:
        return float("nan")
    outside = (b["n_errors"] - inside * b["n_with_cue"]) / n_out
    return float(inside / outside) if outside > 0 else float("inf")


def cmd_grain(args):
    """Addendum of 2026-09-23 (docs/plan/cue_verification.md): the boundary cue across window grains on Base's
    export, graded from the fine run and the coarsened runs; writes exp/out/exp82_grain.json."""
    from scipy.stats import spearmanr
    runs = {1: json.load(open(SUMMARY))}
    for k in GRAINS[1:]:
        runs[k] = json.load(open(os.path.join(OUT, "exp82_cues", f"olmoearth_base_x{k}.json")))
    cells = {}
    for k, d in runs.items():
        for t, v in d["tasks"].items():
            b = v["cues"]["boundary"]
            p, th = b["n_with_cue"] / v["n_units"], v["error_rate"]
            cells[f"{t}@x{k}"] = {"task": t, "k": k, "n_units": v["n_units"], "prevalence": float(p), "error_rate": float(th),
                                 "enrichment": b["enrichment"], "boot_lo": b["boot_lo"], "boot_hi": b["boot_hi"],
                                 "risk_ratio": risk_ratio(b), "ceiling": float((1 - th) / (p - th)) if p > th else float("inf")}
    tasks = list(runs[1]["tasks"])
    mono = {t: {"enrichment_falls": all(cells[f"{t}@x{a}"]["enrichment"] > cells[f"{t}@x{b}"]["enrichment"] for a, b in zip(GRAINS, GRAINS[1:])),
                "risk_ratio_falls": all(cells[f"{t}@x{a}"]["risk_ratio"] > cells[f"{t}@x{b}"]["risk_ratio"] for a, b in zip(GRAINS, GRAINS[1:])),
                "prevalence_rises": all(cells[f"{t}@x{a}"]["prevalence"] < cells[f"{t}@x{b}"]["prevalence"] for a, b in zip(GRAINS, GRAINS[1:]))}
            for t in tasks}
    cash = cells["m_cashew_plant@x4"]
    ceil = [c["ceiling"] if np.isfinite(c["ceiling"]) else 1e9 for c in cells.values()]   # an unbounded ceiling ranks highest
    rho = float(spearmanr([c["enrichment"] for c in cells.values()], ceil).correlation)
    sat = {n: c for n, c in cells.items() if c["prevalence"] >= 0.9}
    unsat_low = {n: c["risk_ratio"] for n, c in cells.items() if c["prevalence"] < 0.9 and c["risk_ratio"] <= 1.3}
    verdicts = {"G1_saturation_is_the_grains": {"holds": cash["prevalence"] >= 0.9 and cash["enrichment"] <= 1.1,
                                                 "prevalence": cash["prevalence"], "enrichment": cash["enrichment"], "bars": [0.9, 1.1]},
                "G2_ceiling_governs_across_grains": {"holds": rho >= 0.9, "spearman": rho, "n_cells": len(cells),
                                                      "unbounded_ceiling_cells": [n for n, c in cells.items() if not np.isfinite(c["ceiling"])]},
                "G3_saturation_threshold": {"holds": all(c["risk_ratio"] <= 1.3 for c in sat.values()) and not unsat_low,
                                            "saturated_cells": {n: c["risk_ratio"] for n, c in sat.items()}, "unsaturated_at_or_below_1.3": unsat_low},
                "descriptive_monotone": {"n_tasks": len(tasks),
                                         "enrichment_falls_on": sum(m["enrichment_falls"] for m in mono.values()),
                                         "risk_ratio_falls_on": sum(m["risk_ratio_falls"] for m in mono.values()),
                                         "prevalence_rises_on": sum(m["prevalence_rises"] for m in mono.values()), "per_task": mono}}
    out = {"experiment": "exp82 grain addendum: the boundary cue across window grains on OlmoEarth Base", "grains": GRAINS,
           "sources": {k: os.path.relpath(SUMMARY if k == 1 else os.path.join(OUT, "exp82_cues", f"olmoearth_base_x{k}.json"), ROOT) for k in GRAINS},
           "cells": cells, "verdicts": verdicts}
    with open(GRAIN_FILE, "w") as f:
        json.dump(out, f, indent=1, default=float)
    for t in tasks:
        print(f"  {t:28s} " + "  ".join(f"x{k}: p {cells[f'{t}@x{k}']['prevalence']:.3f} enr {cells[f'{t}@x{k}']['enrichment']:.2f} rr {cells[f'{t}@x{k}']['risk_ratio']:.2f}" for k in GRAINS))
    print(json.dumps({k: v.get("holds") for k, v in verdicts.items() if "holds" in v}, indent=1))
    print(json.dumps(verdicts["descriptive_monotone"] | {"per_task": None}))
    print(f"wrote {GRAIN_FILE}")
    return 0


def cmd_smoke(args):
    rng = np.random.default_rng(0)
    N, hw, ww = 40, 16, 16
    hard = (rng.random((N, hw, ww)) < 0.5).astype(np.int64)
    hard[:, :, :8] = 0                                            # a left half of one class: boundaries in the middle
    valid = rng.random((N, hw, ww)) > 0.05
    tile = np.broadcast_to(np.arange(N)[:, None, None], (N, hw, ww))[valid]
    bnd = boundary_cue(hard, valid)
    margin = rng.random(int(valid.sum())) * np.where(bnd, 0.6, 1.0)
    err = (rng.random(margin.size) < np.where(bnd, 0.4, 0.1)).astype(float)
    row = {"n_units": int(err.size), "n_tiles": N, "n_classes": 2, "error_rate": float(err.mean()),
           "recorded_capture_20": metrics.capture_at_budget_expected(-margin, err, (0.2,))[0.2],   # a second route, as the gate demands
           "cues": {k: enrichment_from_counts(per_tile_counts(v, err, tile), 200) for k, v in
                    {"boundary": bnd, "low_confidence": low_confidence_cue(margin), "boundary_and_low_confidence": bnd & low_confidence_cue(margin),
                     "low_confidence_per_tile": low_confidence_per_tile(margin, tile)}.items()},
           "captures": captures(margin, bnd, err)}
    v = verdicts({"synthetic2": row, "synthetic19": dict(row, n_classes=19)})
    # the synthetic map's boundary windows are wrong 4x as often as the rest; the cue's share ratio lands near 1.8
    ok = v["gate_low_confidence_reproduces_exp70"]["holds"] and row["cues"]["boundary"]["enrichment"] > 1.5
    print(json.dumps({k: vv.get("holds") for k, vv in v.items()}, indent=1))
    print("smoke", "ok" if ok else "FAILED", {k: round(c["enrichment"], 2) for k, c in row["cues"].items()})
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "grade", "smoke", "grain"), required=True)
    ap.add_argument("--units", default=None)
    ap.add_argument("--encoder", default="olmoearth_base", help="another encoder's exp79 export writes to exp/out/exp82_cues/<encoder>.json")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--coarsen", type=int, default=1, help="addendum: the export at a k x k coarser window grid, written to exp/out/exp82_cues/<encoder>_x<k>.json")
    args = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "grade": cmd_grade, "smoke": cmd_smoke, "grain": cmd_grain}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
