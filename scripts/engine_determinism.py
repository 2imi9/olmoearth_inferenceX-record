"""The same code, the same seeds, two GPU types: how far apart are the numbers? exp79's OlmoEarth Base export was
run twice from commit 15baa4d, once on an RTX PRO 6000 (job 1028965, kept as exp/out/exp79_engine/
olmoearth_base_rtx.json) and once on a B200 (job 1029369, the graded export exp/out/exp79_seeds/olmoearth_base.json).
Linear-probe training is not bit-identical across the two, and the record's gate (reproduce the recorded accuracy
to 1e-4) is met on one engine and not the other. This script writes the per-task differences so that the
engine's contribution is a measured quantity beside the seed's.

    python scripts/engine_determinism.py            # writes exp/out/exp79_engine/summary.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
RTX = os.path.join(OUT, "exp79_engine", "olmoearth_base_rtx.json")
B200 = os.path.join(OUT, "exp79_seeds", "olmoearth_base.json")
SUMMARY = os.path.join(OUT, "exp79_engine", "summary.json")


def lead_of(sig):
    return min(sig["ctl_embedding_distance"]["excess_aurc"], sig["ctl_class_rarity"]["excess_aurc"]) - sig["margin"]["excess_aurc"]


def compare(rtx, b200):
    rows = {}
    for t in rtx["tasks"]:
        if t not in b200["tasks"]:
            continue
        a, b = rtx["tasks"][t], b200["tasks"][t]
        n = min(len(a["seeds"]), len(b["seeds"]))
        acc_d = [a["seeds"][s]["test_accuracy"] - b["seeds"][s]["test_accuracy"] for s in range(n)]
        lead_d = [lead_of(a["seeds"][s]["signals"]) - lead_of(b["seeds"][s]["signals"]) for s in range(n)]
        leads_a = [lead_of(a["seeds"][s]["signals"]) for s in range(n)]
        leads_b = [lead_of(b["seeds"][s]["signals"]) for s in range(n)]
        rows[t] = {"family": a["family"], "n_units": a["n_units"], "n_seeds": n,
                   "recorded_accuracy": a.get("recorded_accuracy"),
                   "seed0_accuracy_rtx": a["seeds"][0]["test_accuracy"], "seed0_accuracy_b200": b["seeds"][0]["test_accuracy"],
                   "seed0_accuracy_minus_recorded_rtx": a.get("accuracy_minus_recorded"), "seed0_accuracy_minus_recorded_b200": b.get("accuracy_minus_recorded"),
                   "accuracy_difference_per_seed": acc_d, "max_abs_accuracy_difference": float(np.max(np.abs(acc_d))),
                   "net_predictions_moved_seed0": int(round(abs(acc_d[0]) * a["n_units"])),   # net: a swap of one right and one wrong prediction leaves accuracy unchanged
                   "lead_difference_per_seed": lead_d, "max_abs_lead_difference": float(np.max(np.abs(lead_d))),
                   "lead_seed_sd_rtx": float(np.std(leads_a, ddof=1)) if n > 1 else None,
                   "lead_seed_sd_b200": float(np.std(leads_b, ddof=1)) if n > 1 else None,
                   "engine_over_seed_lead_ratio": float(np.mean(np.abs(lead_d)) / np.std(leads_b, ddof=1)) if n > 1 and np.std(leads_b, ddof=1) > 0 else None,
                   "wins_rtx": int(sum(x > 0 for x in leads_a)), "wins_b200": int(sum(x > 0 for x in leads_b))}
    cls = [r for r in rows.values() if r["family"].startswith("cla")]
    seg = [r for r in rows.values() if r["family"].startswith("seg")]
    summary = {
        "what": "exp79 OlmoEarth Base export on two GPU types from the same commit and seeds: RTX PRO 6000 (job 1028965) vs B200 (job 1029369)",
        "n_tasks": len(rows), "tasks": rows,
        "classification": {"n": len(cls), "max_abs_accuracy_difference": max(r["max_abs_accuracy_difference"] for r in cls) if cls else None,
                           "median_net_predictions_moved_seed0": float(np.median([r["net_predictions_moved_seed0"] for r in cls])) if cls else None,
                           "median_engine_over_seed_lead_ratio": float(np.median([r["engine_over_seed_lead_ratio"] for r in cls if r["engine_over_seed_lead_ratio"] is not None])) if cls else None,
                           "median_engine_over_seed_lead_ratio_where_engine_moves": float(np.median([r["engine_over_seed_lead_ratio"] for r in cls if r["engine_over_seed_lead_ratio"] and r["max_abs_lead_difference"] > 1e-9])) if cls else None,
                           "tasks_gate_failed_rtx": sorted(t for t, r in rows.items() if r["family"].startswith("cla") and r["seed0_accuracy_minus_recorded_rtx"] is not None and abs(r["seed0_accuracy_minus_recorded_rtx"]) >= 1e-4),
                           "tasks_gate_failed_b200": sorted(t for t, r in rows.items() if r["family"].startswith("cla") and r["seed0_accuracy_minus_recorded_b200"] is not None and abs(r["seed0_accuracy_minus_recorded_b200"]) >= 1e-4)},
        "segmentation": {"n": len(seg), "max_abs_accuracy_difference": max(r["max_abs_accuracy_difference"] for r in seg) if seg else None,
                         "median_engine_over_seed_lead_ratio": float(np.median([r["engine_over_seed_lead_ratio"] for r in seg if r["engine_over_seed_lead_ratio"] is not None])) if seg else None,
                         "tasks_gate_failed_rtx": sorted(t for t, r in rows.items() if r["family"].startswith("seg") and r["seed0_accuracy_minus_recorded_rtx"] is not None and abs(r["seed0_accuracy_minus_recorded_rtx"]) >= 1e-4),
                         "tasks_gate_failed_b200": sorted(t for t, r in rows.items() if r["family"].startswith("seg") and r["seed0_accuracy_minus_recorded_b200"] is not None and abs(r["seed0_accuracy_minus_recorded_b200"]) >= 1e-4)},
        "lead": {"max_abs_lead_difference": max(r["max_abs_lead_difference"] for r in rows.values()),
                 "median_engine_over_seed_ratio_all_tasks": float(np.median([r["engine_over_seed_lead_ratio"] for r in rows.values() if r["engine_over_seed_lead_ratio"] is not None])),
                 "cells_with_identical_accuracy_but_different_lead": int(sum(1 for r in rows.values() for a, l in zip(r["accuracy_difference_per_seed"], r["lead_difference_per_seed"]) if a == 0 and abs(l) > 1e-12)),
                 "cells_with_identical_lead": int(sum(1 for r in rows.values() for l in r["lead_difference_per_seed"] if abs(l) <= 1e-12)),
                 "wins_differ_on": sorted(t for t, r in rows.items() if r["wins_rtx"] != r["wins_b200"])},
    }
    return summary


def main():
    if not (os.path.exists(RTX) and os.path.exists(B200)):
        print(f"need both {RTX} and {B200}")
        return 1
    s = compare(json.load(open(RTX)), json.load(open(B200)))
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w") as f:
        json.dump(s, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in s.items() if k != "tasks"}, indent=1, default=float))
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
