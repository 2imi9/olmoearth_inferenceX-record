"""exp83: can raters from different families estimate a map's accuracy without labels? Dawid-Skene and pairwise
disagreement over the suite's multi-encoder panels, labels held out to grade. Preregistered in
docs/plan/consensus_reliability.md.

    python exp/exp83_consensus.py --stage estimate
    python exp/exp83_consensus.py --stage grade
    python exp/exp83_consensus.py --stage smoke

Numpy only; reads exp/out/exp63_masks.npz (MADOS, PASTIS S2: six encoders) and exp/out/exp57_masks.npz
(Sen1Floods11: eight encoders). The estimator is oe_inferencex.reliability.dawid_skene, the one exp07 ran.
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
from oe_inferencex import stats                # noqa: E402
from oe_inferencex.reliability import dawid_skene  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SUMMARY = os.path.join(OUT, "exp83_summary.json")
SIX = ["olmoearth_base", "galileo_base", "croma_base", "terramind_base", "clay_large", "anysat"]
EIGHT = SIX[:5] + ["satlas_base", "anysat", "panopticon"]
# the preregistered bars, docs/plan/consensus_reliability.md
P1_TOL, P2_RHO, P3_RHO = 0.10, 0.8, 0.8
DIVERSE, UNANIMOUS = ("mados", "pastis_sentinel2"), ("sen1floods11",)


# ----------------------------------------------------------------------------- panels
def panel_exp63(task):
    z = np.load(os.path.join(OUT, "exp63_masks.npz"))
    ok = z[f"{task}/ok"]
    y = z[f"{task}/y"][ok].astype(np.int64)
    votes = np.stack([z[f"{task}/{e}/0/dec"][ok].astype(np.int64) for e in SIX], 1)
    draws = np.stack([z[f"{task}/olmoearth_base/{k}/dec"][ok].astype(np.int64) for k in range(3)], 1)
    return {"y": y, "votes": votes, "raters": SIX, "within_family": draws, "n_classes": int(max(votes.max(), draws.max()) + 1)}


def panel_exp57():
    z = np.load(os.path.join(OUT, "exp57_masks.npz"))
    common = z["encoders_ok"].copy()
    for e in EIGHT:
        k = f"encoders_ok_{e}"
        if k in z.files:
            common &= z[k]
    y = z["encoders_y"][common].astype(np.int64)
    votes = np.stack([z[f"encoders_{e}"][common].astype(np.int64) for e in EIGHT], 1)
    return {"y": y, "votes": votes, "raters": EIGHT, "within_family": None, "n_classes": 2}


def majority_shared_share(votes, y):
    """Per rater: the share of its errors on which a strict majority of the OTHER raters cast the same wrong vote."""
    N, R = votes.shape
    out = np.zeros(R)
    for j in range(R):
        err = votes[:, j] != y
        others = np.delete(votes, j, 1)
        same = (others == votes[:, j][:, None]).sum(1)
        out[j] = float(((same > (R - 1) / 2) & err).sum() / max(err.sum(), 1))
    return out


def pairwise_disagreement(votes):
    N, R = votes.shape
    d = np.zeros((R, R))
    for a in range(R):
        for b in range(R):
            if a != b:
                d[a, b] = float((votes[:, a] != votes[:, b]).mean())
    return d


def agreement_with_majority(votes, n_classes):
    """Per rater: the share of windows on which it agrees with the plurality vote of the others (ties count as
    disagreement), the naive label-free accuracy reading."""
    N, R = votes.shape
    out = np.zeros(R)
    for j in range(R):
        others = np.delete(votes, j, 1)
        counts = np.zeros((N, n_classes), np.int64)
        for c in range(others.shape[1]):
            np.add.at(counts, (np.arange(N), others[:, c]), 1)
        top = counts.max(1)
        plural = counts.argmax(1)
        unique = (counts == top[:, None]).sum(1) == 1
        out[j] = float((unique & (plural == votes[:, j])).mean())
    return out


def estimate_panel(task, P):
    t0 = time.time()
    y, votes, C = P["y"], P["votes"], P["n_classes"]
    acc = (votes == y[:, None]).mean(0)
    post, conf, rel, info = dawid_skene(votes, C, iters=2000, return_info=True)
    hidden = (rel - acc) / (1 - acc)
    m = majority_shared_share(votes, y)
    d = pairwise_disagreement(votes)
    dbar = d.sum(1) / (votes.shape[1] - 1)
    agree = agreement_with_majority(votes, C)
    rho_rank = float(stats.spearman(rel, acc))
    rho_dis = float(stats.spearman(dbar, 1 - acc))
    row = {"n_windows": int(y.size), "n_classes": C, "raters": P["raters"],
           "true_accuracy": acc.tolist(), "ds_estimate": rel.tolist(), "hidden_share": hidden.tolist(),
           "majority_shared_share": m.tolist(), "mean_pairwise_disagreement": dbar.tolist(),
           "agreement_with_majority": agree.tolist(),
           "disagreement_understates_error": [bool(dbar[j] < 1 - acc[j]) for j in range(len(acc))],
           "understatement_factor": [float((1 - acc[j]) / dbar[j]) if dbar[j] > 0 else None for j in range(len(acc))],
           "spearman_ds_vs_true": rho_rank, "spearman_disagreement_vs_error": rho_dis,
           "best_true": P["raters"][int(np.argmax(acc))], "best_estimated": P["raters"][int(np.argmax(rel))],
           "ds_argmax_label_accuracy": float((post.argmax(1) == y).mean()), "em": info, "seconds": round(time.time() - t0, 1)}
    if P["within_family"] is not None:
        w = P["within_family"]
        acc_w = (w == y[:, None]).mean(0)
        _, _, rel_w, info_w = dawid_skene(w, C, iters=2000, return_info=True)
        row["within_family"] = {"raters": "olmoearth_base draws 0-2", "true_accuracy": acc_w.tolist(), "ds_estimate": rel_w.tolist(),
                                "hidden_share": ((rel_w - acc_w) / (1 - acc_w)).tolist(),
                                "pairwise_disagreement": pairwise_disagreement(w)[np.triu_indices(3, 1)].tolist(), "em": info_w}
    print(f"  {task:18s} N={y.size:7d} C={C:2d} EM {info['iterations']} it {'ok' if info['converged'] else 'NOT CONVERGED'} rho(DS,true) {rho_rank:+.2f} best true {row['best_true']} est {row['best_estimated']} "
          f"hidden {np.round(hidden, 2).tolist()} m {np.round(m, 2).tolist()} {row['seconds']:.0f}s", flush=True)
    return row


# ----------------------------------------------------------------------------- verdicts, pure functions of the rows
def grade_p1(rows, tol=P1_TOL):
    per = {t: [abs(h - m) for h, m in zip(r["hidden_share"], r["majority_shared_share"])] for t, r in rows.items()}
    fails = [(t, r["raters"][j], round(r["hidden_share"][j], 3), round(r["majority_shared_share"][j], 3))
             for t, r in rows.items() for j, dlt in enumerate(per[t]) if dlt > tol]
    return {"holds": bool(per) and not fails, "max_abs_difference": max((max(v) for v in per.values()), default=None),
            "failing": fails, "per_task_max": {t: max(v) for t, v in per.items()}}


def grade_p2(rows, rho=P2_RHO):
    div = {t: rows[t] for t in DIVERSE if t in rows}
    una = {t: rows[t] for t in UNANIMOUS if t in rows}
    a = bool(div) and all(r["spearman_ds_vs_true"] >= rho and r["best_true"] == r["best_estimated"] for r in div.values())
    b = bool(una) and all(r["spearman_ds_vs_true"] < rho for r in una.values())
    return {"holds": a and b, "diverse_hold": a, "unanimous_below": b,
            "spearman": {t: r["spearman_ds_vs_true"] for t, r in rows.items()},
            "best": {t: (r["best_true"], r["best_estimated"]) for t, r in rows.items()}}


def grade_p3(rows, rho=P3_RHO):
    div = {t: rows[t] for t in DIVERSE if t in rows}
    a = bool(div) and all(r["spearman_disagreement_vs_error"] >= rho for r in div.values())
    under = all(all(r["disagreement_understates_error"]) for r in rows.values())
    factors = {t: float(np.median([f for f in r["understatement_factor"] if f])) for t, r in rows.items()}
    largest = max(factors, key=factors.get) if factors else None
    return {"holds": a and under and largest in UNANIMOUS, "diverse_rho_hold": a, "understates_everywhere": under,
            "median_understatement_factor": factors, "largest_on": largest,
            "spearman": {t: r["spearman_disagreement_vs_error"] for t, r in rows.items()}}


def verdicts(rows):
    return {"P1_hidden_share_is_the_majority_shared_share": grade_p1(rows),
            "P2_order_survives_where_the_panel_is_diverse": grade_p2(rows),
            "P3_disagreement_reads_error_with_the_same_blind_spot": grade_p3(rows)}


# ----------------------------------------------------------------------------- stages
def cmd_estimate(args):
    t0 = time.time()
    rows = {"mados": estimate_panel("mados", panel_exp63("mados")),
            "pastis_sentinel2": estimate_panel("pastis_sentinel2", panel_exp63("pastis_sentinel2")),
            "sen1floods11": estimate_panel("sen1floods11", panel_exp57())}
    summary = {"experiment": "exp83 can raters from different families estimate a map's accuracy without labels",
               "artifacts_read": ["exp/out/exp63_masks.npz", "exp/out/exp57_masks.npz"],
               "config": {"ds_iters_cap": 2000, "ds_tol": 1e-6, "seconds": round(time.time() - t0)}, "tasks": rows, "prereg": verdicts(rows)}
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in summary["prereg"].items()}, indent=1))
    print(f"wrote {SUMMARY}")
    return 0


def cmd_grade(args):
    d = json.load(open(SUMMARY))
    d["prereg"] = verdicts(d["tasks"])
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in d["prereg"].items()}, indent=1))
    return 0


def cmd_smoke(args):
    """Independent raters with known accuracies: Dawid-Skene recovers them, the hidden share is near zero, the
    majority-shared share is small, and the order is exact."""
    rng = np.random.default_rng(0)
    N, C, accs = 20000, 4, [0.95, 0.9, 0.85, 0.8, 0.7]
    y = rng.integers(0, C, N)
    votes = np.stack([np.where(rng.random(N) < a, y, (y + rng.integers(1, C, N)) % C) for a in accs], 1)
    P = {"y": y, "votes": votes, "raters": [f"r{i}" for i in range(5)], "within_family": None, "n_classes": C}
    row = estimate_panel("synthetic", P)
    ok = max(abs(e - a) for e, a in zip(row["ds_estimate"], accs)) < 0.02 and row["spearman_ds_vs_true"] > 0.99 \
        and max(abs(h) for h in row["hidden_share"]) < 0.15
    print("smoke", "ok" if ok else "FAILED", np.round(row["ds_estimate"], 3).tolist())
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "grade", "smoke"), required=True)
    args = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "grade": cmd_grade, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
