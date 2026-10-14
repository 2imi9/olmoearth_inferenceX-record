"""exp81, seventh amendment: does the rare-error warning mark the draws on which a per-class interval misses?

docs/plan/per_class_assessment.md, seventh amendment, has the predictions. Numpy only; reads the exp78 export for
Base's classification tasks and the exp79 export for MADOS, as exp81 did. Writes exp/out/exp81_rare_error_warning.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp81_per_class as e81                                           # noqa: E402
from oe_inferencex import estimate as est                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out", "exp81_rare_error_warning.json")
E78 = os.path.join(ROOT, "exp", "out", "exp78_units")
E79 = os.path.join(ROOT, "exp", "out", "exp79_units", "olmoearth_base")
Q = {"user_accuracy": "user's accuracy", "producer_accuracy": "producer's accuracy", "reference_share": "share"}
# the nine rare-error cells of exp81's record: (task, design, budget, class, quantity)
CELLS = [("m_eurosat", "random", 300, 0, "producer_accuracy"), ("m_eurosat", "random", 300, 3, "reference_share"),
         ("m_eurosat", "random", 300, 6, "producer_accuracy"), ("m_eurosat", "random", 300, 6, "reference_share"),
         ("m_eurosat", "confidence", 300, 6, "producer_accuracy"), ("mados", "confidence", 1000, 7, "producer_accuracy"),
         ("m_brick_kiln", "confidence", 300, 0, "producer_accuracy"), ("m_brick_kiln", "confidence", 300, 1, "user_accuracy"),
         ("nandi_landsat", "confidence", 300, 1, "user_accuracy")]


def warned(row, q):
    return "few errors" in row.get("warning_codes", []) and Q[q] in row.get("warning", "")


def cell(u, C, design, B, c, q, draws):
    y, dec, margin, p1 = u["y"], u["dec"], u["margin"], u["p1"]
    T = e81.truth_per_class(y, dec, C)
    truth = T[q][c]
    hit = {True: [0, 0], False: [0, 0]}                                  # warned -> [covered, draws]
    for r in range(draws):
        s = est.sample_for_estimation(margin, B, design=design, p1=p1, seed=r)
        row = est.estimate_per_class(s, y[s["indices"]], dec, n_classes=C)["per_class"][c]
        v = row[q]
        n_ok = row["n_labelled_map_class"] if q == "user_accuracy" else row["n_labelled_reference_class"]
        if v is None or (q != "reference_share" and n_ok < est.MIN_PER_CLASS):
            continue
        w = warned(row, q)
        hit[w][0] += v["low"] <= truth <= v["high"]
        hit[w][1] += 1
    miss_w, miss_n = hit[True][1] - hit[True][0], hit[False][1] - hit[False][0]
    return {"truth": float(truth), "n_graded": hit[True][1] + hit[False][1], "n_warned": hit[True][1],
            "coverage": (hit[True][0] + hit[False][0]) / max(hit[True][1] + hit[False][1], 1),
            "coverage_warned": hit[True][0] / hit[True][1] if hit[True][1] else None,
            "coverage_unwarned": hit[False][0] / hit[False][1] if hit[False][1] else None,
            "misses": miss_w + miss_n, "misses_warned": miss_w,
            "share_of_misses_warned": miss_w / (miss_w + miss_n) if miss_w + miss_n else None}


def firing_rate(tasks, draws):
    """Share of graded (draw, class, quantity) cells warned, 300 labels, both designs, the classification tasks."""
    n = w = 0
    for t in tasks:
        u = e81.load_units_with_reference(t, E78)
        C = int(max(u["y"].max(), u["dec"].max()) + 1)
        if u["y"].size <= 300:
            continue
        for design in ("random", "confidence"):
            for r in range(draws):
                s = est.sample_for_estimation(u["margin"], 300, design=design, p1=u["p1"], seed=r)
                for row in est.estimate_per_class(s, u["y"][s["indices"]], u["dec"], n_classes=C)["per_class"].values():
                    for q in Q:
                        n_ok = row["n_labelled_map_class"] if q == "user_accuracy" else row["n_labelled_reference_class"]
                        if row[q] is None or (q != "reference_share" and n_ok < est.MIN_PER_CLASS):
                            continue
                        if q == "user_accuracy" and design == "random":
                            continue                               # exact; never warned for this reason
                        n += 1
                        w += warned(row, q)
    return {"n_graded": n, "n_warned": w, "share_warned": w / max(n, 1)}


def main():
    t0 = time.time()
    rows = {}
    for task, design, B, c, q in CELLS:
        u = e81.load_units_with_reference(task, E79 if task == "mados" else E78)
        C = int(max(u["y"].max(), u["dec"].max()) + 1)
        rows[f"{task}/{design}/{B}/{c}/{q}"] = r = cell(u, C, design, B, c, q, 2000)
        print(f"  {task:14s} {design:10s} B={B:5d} class {c:2d} {q:17s} cov {r['coverage']:.3f} warned {r['n_warned']:4d}/"
              f"{r['n_graded']} misses {r['misses']:3d} of which warned {r['misses_warned']:3d}  cov unwarned {r['coverage_unwarned']}", flush=True)
    import exp70_task_suite as e70
    rate = firing_rate(e70.TASKS_CLS, 300)
    misses = sum(r["misses"] for r in rows.values())
    misses_w = sum(r["misses_warned"] for r in rows.values())
    unw = [r["coverage_unwarned"] for r in rows.values() if r["coverage_unwarned"] is not None]
    verdicts = {"a_warning_marks_the_misses": {"holds": bool(misses > 0 and misses_w / misses >= 0.8), "misses": misses,
                                               "misses_warned": misses_w, "share": misses_w / max(misses, 1), "bar": 0.8},
                "b_unwarned_draws_cover": {"holds": bool(unw) and bool(min(unw) >= 0.93), "min_coverage_unwarned": min(unw) if unw else None, "bar": 0.93},
                "c_warning_is_not_everywhere": {"holds": bool(rate["share_warned"] <= 0.2), **rate, "bar": 0.2}}
    out = {"experiment": "exp81 seventh amendment: the rare-error warning", "rare_errors": est.RARE_ERRORS,
           "cells": rows, "verdicts": verdicts, "seconds": round(time.time() - t0)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk in ("holds", "share", "min_coverage_unwarned", "share_warned")} for k, v in verdicts.items()}, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
