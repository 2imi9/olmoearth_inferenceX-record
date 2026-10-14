"""exp84: where the lead holds. The headline re-tabulated by group, with its accuracy confound, its multiplicity
correction and its spread across tile groups. Preregistered in docs/plan/where_the_lead_holds.md.

    python exp/exp84_where_the_lead_holds.py --stage estimate [--seeds-dir DIR] [--units DIR]
    python exp/exp84_where_the_lead_holds.py --stage grade
    python exp/exp84_where_the_lead_holds.py --stage smoke

Numpy only. Reads exp79's per-encoder exports (exp/out/exp79_seeds/<encoder>.json; ten seeds x tasks with the
signals exp70 scores), exp73/exp74's summaries for the sign tests, and exp78's per-unit export for the block
spread on OlmoEarth Base. Reruns as exports land; the verdicts count the encoders present and name the missing.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)
from oe_inferencex import metrics, stats     # noqa: E402
import exp70_task_suite as e70               # noqa: E402
import exp74_suite_encoders as e74           # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SUMMARY = os.path.join(OUT, "exp84_summary.json")
SEEDS_DIR = os.path.join(OUT, "exp79_seeds")
OTHERS = list(e74.ENCODERS)                       # the fifteen the predictions are on
N_BLOCKS = 50
# the preregistered bars, docs/plan/where_the_lead_holds.md
P1_RHO_AUROC, P1_RHO_LEAD = 0.4, 0.0
P2_MAX_UNITS, P2_MAX_ACC = 306, 0.40
P3_RANGE = 0.06
P4_ALPHA = 0.05


def sensor_group(task):
    if "sentinel1_sentinel2" in task or "sentinel2_sentinel1" in task or task == "m_so2sat":
        return "S1+S2"
    if "sentinel1" in task:
        return "S1"
    if "landsat" in task or task == "m_forestnet":
        return "Landsat"
    return "S2"


def size_group(n):
    return "<=306" if n <= 306 else ("<=1000" if n <= 1000 else ">1000")


def class_group(c):
    return "2" if c == 2 else ("6-10" if c <= 10 else ">=12")


def lead_of(sig):
    return min(sig["ctl_embedding_distance"]["excess_aurc"], sig["ctl_class_rarity"]["excess_aurc"]) - sig["margin"]["excess_aurc"]


# ----------------------------------------------------------------------------- per encoder, per seed
def encoder_rows(export):
    """Per seed: the cells (task, accuracy, AUROC, lead, n, classes) and the statistics the predictions grade."""
    n_seeds = export["n_seeds"]
    per_seed = []
    for s in range(n_seeds):
        cells = []
        for t, v in export["tasks"].items():
            if s >= len(v["seeds"]):
                continue
            r = v["seeds"][s]
            cells.append({"task": t, "accuracy": r["test_accuracy"], "auroc": r["signals"]["margin"]["auroc"],
                          "lead": lead_of(r["signals"]), "n_units": r["n_units"], "n_classes": r["n_classes"],
                          "sensor": sensor_group(t), "size": size_group(r["n_units"]), "classes": class_group(r["n_classes"])})
        acc = np.array([c["accuracy"] for c in cells]); au = np.array([c["auroc"] for c in cells]); ld = np.array([c["lead"] for c in cells])
        groups = {}
        for key in ("sensor", "size", "classes"):
            by = {}
            for c in cells:
                by.setdefault(c[key], []).append(c["lead"])
            groups[key] = {g: {"median_lead": float(np.median(v)), "n": len(v), "wins": int(sum(x > 0 for x in v))} for g, v in by.items()}
        wins = int((ld > 0).sum())
        per_seed.append({"seed": s, "n_tasks": len(cells), "rho_accuracy_auroc": float(stats.spearman(acc, au)),
                         "rho_accuracy_lead": float(stats.spearman(acc, ld)), "wins": wins, "losses": int(len(cells) - wins),
                         "sign_p": float(stats.sign_test(wins, len(cells) - wins, alternative="greater")),
                         "losses_on": [{"task": c["task"], "n_units": c["n_units"], "accuracy": c["accuracy"], "lead": c["lead"]} for c in cells if c["lead"] <= 0],
                         "sensor_range": float(max(v["median_lead"] for v in groups["sensor"].values()) - min(v["median_lead"] for v in groups["sensor"].values())),
                         "groups": groups, "min_lead": float(ld.min())})
    return per_seed


def holm(pvals, alpha=P4_ALPHA):
    """Holm's step-down: adjusted p per test and whether it survives."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, p * (m - i)))
        running = adj
        out[k] = {"p": p, "adjusted": adj, "survives": adj < alpha}
    return out


# ----------------------------------------------------------------------------- block spread on Base's export
def block_spread(units_dir, n_blocks=N_BLOCKS):
    import exp78_error_rate_estimation as e78
    rows = {}
    for t in e70.TASKS_SEG:
        path = os.path.join(units_dir, f"{t}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=False)
        u = e78.load_units(t, units_dir)
        err, margin, tile, dec = u["err"], u["margin"], u["tile"], d["dec"].astype(np.int64)
        C = int(d["n_classes"][0])
        rarity = 1.0 / np.bincount(dec, minlength=C)[dec].astype(float)          # exp70's class-rarity control: rare predicted class = suspect
        order = np.argsort(tile, kind="stable")
        cuts = np.linspace(0, err.size, n_blocks + 1).astype(int)
        leads = []
        for b in range(n_blocks):
            idx = order[cuts[b]:cuts[b + 1]]
            if err[idx].sum() == 0 or err[idx].sum() == idx.size:
                continue
            e_m = metrics.excess_aurc(-margin[idx], err[idx])
            e_c = metrics.excess_aurc(rarity[idx], err[idx])
            leads.append(float(e_c - e_m))
        leads = np.array(leads)
        rows[t] = {"n_blocks": int(leads.size), "share_positive": float((leads > 0).mean()), "p10_lead": float(np.percentile(leads, 10)),
                   "median_lead": float(np.median(leads)), "pooled_lead_rarity_control": float(
                       metrics.excess_aurc(rarity, err) - metrics.excess_aurc(-margin, err))}
        print(f"  block spread {t:28s} blocks {leads.size:3d} positive {rows[t]['share_positive']:.2f} p10 {rows[t]['p10_lead']:+.4f} pooled {rows[t]['pooled_lead_rarity_control']:+.4f}", flush=True)
    return rows


# ----------------------------------------------------------------------------- verdicts
def grade_p1(per_encoder):
    fails = [(e, r["seed"], round(r["rho_accuracy_auroc"], 3), round(r["rho_accuracy_lead"], 3)) for e, rows in per_encoder.items() if e in OTHERS
             for r in rows if not (r["rho_accuracy_auroc"] >= P1_RHO_AUROC and r["rho_accuracy_lead"] <= P1_RHO_LEAD)]
    have = sorted(e for e in per_encoder if e in OTHERS)
    return {"holds": len(have) == len(OTHERS) and not fails, "encoders_graded": have, "encoders_missing": sorted(set(OTHERS) - set(have)),
            "failing": fails, "min_rho_auroc": min((r["rho_accuracy_auroc"] for e in have for r in per_encoder[e]), default=None),
            "max_rho_lead": max((r["rho_accuracy_lead"] for e in have for r in per_encoder[e]), default=None)}


def grade_p2(per_encoder):
    bad = [(e, r["seed"], L["task"], L["n_units"], round(L["accuracy"], 3)) for e, rows in per_encoder.items() if e in OTHERS
           for r in rows for L in r["losses_on"] if not (L["n_units"] <= P2_MAX_UNITS or L["accuracy"] <= P2_MAX_ACC)]
    have = sorted(e for e in per_encoder if e in OTHERS)
    n_losses = sum(len(r["losses_on"]) for e in have for r in per_encoder[e])
    return {"holds": len(have) == len(OTHERS) and not bad, "encoders_graded": have, "n_losses": n_losses, "losses_elsewhere": bad,
            "loss_tasks": sorted({L["task"] for e in have for r in per_encoder[e] for L in r["losses_on"]})}


def grade_p3(per_encoder):
    fails = [(e, r["seed"], round(r["sensor_range"], 4)) for e, rows in per_encoder.items() if e in OTHERS for r in rows if r["sensor_range"] > P3_RANGE]
    have = sorted(e for e in per_encoder if e in OTHERS)
    return {"holds": len(have) == len(OTHERS) and not fails, "encoders_graded": have, "failing": fails,
            "max_range": max((r["sensor_range"] for e in have for r in per_encoder[e]), default=None)}


def grade_p4(per_encoder):
    have = sorted(e for e in per_encoder if e in OTHERS)
    per_seed = {}
    fails = []
    n_seeds = max((len(rows) for rows in per_encoder.values()), default=0)
    for s in range(n_seeds):
        ps = {e: per_encoder[e][s]["sign_p"] for e in have if s < len(per_encoder[e]) and per_encoder[e][s]["sign_p"] is not None}
        if not ps:
            continue
        h = holm(ps)
        per_seed[s] = h
        fails += [(e, s, v["adjusted"]) for e, v in h.items() if not v["survives"]]
    return {"holds": len(have) == len(OTHERS) and not fails, "encoders_graded": have, "failing": fails,
            "max_adjusted_p": max((v["adjusted"] for h in per_seed.values() for v in h.values()), default=None)}


def verdicts(per_encoder):
    return {"P1_confound_is_the_protocols": grade_p1(per_encoder), "P2_thin_cells_are_small_or_bad": grade_p2(per_encoder),
            "P3_no_sensor_is_a_hole": grade_p3(per_encoder), "P4_bar_survives_holm": grade_p4(per_encoder)}


# ----------------------------------------------------------------------------- stages
def cmd_estimate(args):
    import exp78_error_rate_estimation as e78
    t0 = time.time()
    seeds_dir = args.seeds_dir or SEEDS_DIR
    per_encoder, disclosed = {}, {}
    for path in sorted(glob.glob(os.path.join(seeds_dir, "*.json"))):
        enc = os.path.basename(path)[:-5]
        if enc not in OTHERS and enc != "olmoearth_base":
            continue
        rows = encoder_rows(json.load(open(path)))
        (per_encoder if enc in OTHERS else disclosed)[enc] = rows
        print(f"  {enc:18s} seeds {len(rows)} rho(acc,AUROC) {min(r['rho_accuracy_auroc'] for r in rows):+.3f}..{max(r['rho_accuracy_auroc'] for r in rows):+.3f} "
              f"rho(acc,lead) {min(r['rho_accuracy_lead'] for r in rows):+.3f}..{max(r['rho_accuracy_lead'] for r in rows):+.3f} "
              f"losses {sum(len(r['losses_on']) for r in rows)} sensor range max {max(r['sensor_range'] for r in rows):.3f}", flush=True)
    # seed-0 record: Holm over the sixteen encoders' recorded sign tests and exp73's three alternatives
    e74s = json.load(open(os.path.join(OUT, "exp74_summary.json")))["verdicts"]["P1"]
    e73s = json.load(open(os.path.join(OUT, "exp73_summary.json")))["verdicts"]
    ps = {e: v["p"] for e, v in e74s.get("per_encoder", {}).items()}
    ps.update({"exp70_olmoearth_base": json.load(open(os.path.join(OUT, "exp70_summary.json")))["verdicts"]["P1"]["p"],
               "exp73_ensemble": e73s["P1"]["p"], "exp73_knn_dist": e73s["P2"]["p"], "exp73_mahalanobis": e73s["P3"]["p"]})
    record_holm = holm(ps)
    summary = {"experiment": "exp84 where the lead holds", "config": {"n_blocks": N_BLOCKS, "encoders_predicted": OTHERS, "seconds": None},
               "per_encoder": per_encoder, "disclosed_olmoearth_base": disclosed,
               "record_holm": {"tests": record_holm, "all_survive": all(v["survives"] for v in record_holm.values()),
                               "max_adjusted_p": max(v["adjusted"] for v in record_holm.values())},
               "block_spread_olmoearth_base": block_spread(args.units or e78.UNITS), "prereg": verdicts(per_encoder)}
    summary["config"]["seconds"] = round(time.time() - t0)
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk in ("holds", "encoders_graded", "encoders_missing", "n_losses")} for k, v in summary["prereg"].items()}, indent=1))
    print(f"wrote {SUMMARY}")
    return 0


def cmd_grade(args):
    d = json.load(open(SUMMARY))
    d["prereg"] = verdicts(d["per_encoder"])
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in d["prereg"].items()}, indent=1))
    return 0


def cmd_smoke(args):
    rng = np.random.default_rng(0)
    tasks = e70.TASKS_CLS + e70.TASKS_SEG
    export = {"n_seeds": 3, "tasks": {}}
    for t in tasks:
        seeds = []
        for s in range(3):
            acc = float(rng.uniform(0.5, 0.99))
            m = float(rng.uniform(0.01, 0.1) * (1 - acc)); c = m + float(rng.uniform(0.01, 0.2))
            seeds.append({"test_accuracy": acc, "n_units": int(rng.integers(300, 5000)), "n_classes": 2,
                          "signals": {"margin": {"excess_aurc": m, "auroc": 0.6 + 0.35 * acc}, "ctl_class_rarity": {"excess_aurc": c},
                                      "ctl_embedding_distance": {"excess_aurc": c + 0.01}}})
        export["tasks"][t] = {"seeds": seeds}
    rows = encoder_rows(export)
    h = holm({"a": 0.001, "b": 0.02, "c": 0.06})     # adjusted 0.003, 0.04, 0.06: the last does not survive 0.05
    ok = len(rows) == 3 and all(r["losses"] == 0 for r in rows) and all(r["rho_accuracy_auroc"] > 0.9 for r in rows) \
        and h["a"]["survives"] and h["b"]["survives"] and not h["c"]["survives"] \
        and abs(h["c"]["adjusted"] - 0.06) < 1e-12 and abs(h["b"]["adjusted"] - 0.04) < 1e-12
    print("smoke", "ok" if ok else "FAILED", [round(r["rho_accuracy_auroc"], 3) for r in rows], h)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "grade", "smoke"), required=True)
    ap.add_argument("--seeds-dir", default=None)
    ap.add_argument("--units", default=None)
    args = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "grade": cmd_grade, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
