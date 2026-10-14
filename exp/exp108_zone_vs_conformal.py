"""exp108, report-only (not preregistered; the owner chose a comparison table on 12 October 2026): the package's
certified zone against conformal methods on the same maps and the same labels.

The 34 committed cells of exp95 (24 suite tasks, 10 products; errors in zone order, most confident first), budgets 300
and 1,000 where the budget is under half the units, alpha half the map's error rate, delta 0.1, 2,000 one-look draws
per cell, every arm on the same draws:

  certify    the package's prefix rule on its grid (fixed-sequence exact tests): P(zone error rate > alpha) <= delta
  crc_joint  conformal risk control (Angelopoulos et al. 2022) on the monotone loss 1[wrong and in the zone], unit grid:
             the largest prefix with (k + 1) / (n + 1) <= alpha, k the labelled errors inside it; guarantee: the
             expected share of the unlabelled windows that are wrong AND in the zone is at most alpha (not the zone's
             error rate)
  bh_select  conformal selection (Jin and Candes 2023) with Benjamini-Hochberg: unlabelled window j's p-value
             (1 + labelled errors ahead of it) / (n + 1), BH at alpha over the N - n unlabelled windows; guarantee:
             the expected share of wrong windows among the selected unlabelled ones is at most alpha

Each arm is graded by every arm's criterion: P(the zone's error rate > alpha); the mean zone error rate (no zone = 0);
the mean share of the unlabelled windows wrong and in the zone; the mean share of wrong windows among the selected
unlabelled ones; coverage. Numpy only.
"""
import argparse
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, ROOT)

from oe_inferencex import estimate as est  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
BUDGETS, DELTA, DRAWS, SEED = (300, 1000), 0.10, 2000, 108


def crc_joint_end(pos_sorted, wrong_sorted, n, N, alpha):
    """The zone CRC gives: the number of windows (a prefix of the zone order) before the (m + 1)-th labelled error,
    m = floor(alpha (n + 1) - 1); the whole map when there are at most m; empty when m < 0."""
    m = int(np.floor(alpha * (n + 1) - 1 + 1e-12))
    if m < 0:
        return 0
    errs = pos_sorted[wrong_sorted > 0]
    return N if errs.size <= m else int(errs[m])


def bh_select_end(pos_sorted, wrong_sorted, n, N, alpha):
    """The prefix BH selects among the unlabelled windows: the largest r with (1 + k_r)/(n + 1) <= alpha r / (N - n),
    k_r the labelled errors ahead of the r-th unlabelled window. Returns (zone end in the order, r selected)."""
    m_unl = N - n
    errs = pos_sorted[wrong_sorted > 0]
    labelled_before = np.searchsorted(pos_sorted, errs, side="left")      # labelled windows ahead of each labelled error
    # segment s (s labelled errors ahead) ends just before the (s+1)-th labelled error, or at the map's end
    ends = np.append(errs, N)                                              # zone end of each segment
    unl_at_end = np.append(errs - labelled_before, m_unl)                  # unlabelled windows ahead of that end
    best_r, best_end = 0, 0
    for s, (end, r) in enumerate(zip(ends, unl_at_end)):
        if r > 0 and (1 + s) / (n + 1) <= alpha * r / m_unl + 1e-15:
            best_r, best_end = int(r), int(end)
    return best_end, best_r


def run_cell(err_o, B, alpha, draws=DRAWS, seed=SEED):
    N = err_o.size
    cum = np.concatenate([[0], np.cumsum(err_o)])
    cov, sizes, _ = est.zone_levels(N, B, alpha, DELTA, cut="standard")
    rng = np.random.default_rng(seed)
    rec = {a: {"cov": np.zeros(draws), "risk": np.zeros(draws), "joint_unl": np.zeros(draws), "fdp_unl": np.zeros(draws),
               "zone": np.zeros(draws, bool)} for a in ("certify", "crc_joint", "bh_select")}
    memo = [dict() for _ in sizes]
    for r in range(draws):
        pos = np.sort(rng.choice(N, B, replace=False))
        w = err_o[pos]
        ends = {}
        if cov:
            b, k = est.zone_counts(pos, w, sizes)
            p = np.empty(len(sizes))
            for j, (bb, kk, nn) in enumerate(zip(b, k, sizes)):
                key = (int(bb), int(kk))
                if key not in memo[j]:
                    memo[j][key] = est.zone_pvalue(kk, bb, nn, alpha)
                p[j] = memo[j][key]
            _, best = est.apply_zone_rule(p, b, k, alpha, DELTA, "prefix")
            ends["certify"] = 0 if best is None else int(sizes[best])
        else:
            ends["certify"] = 0
        ends["crc_joint"] = crc_joint_end(pos, w, B, N, alpha)
        ends["bh_select"] = bh_select_end(pos, w, B, N, alpha)[0]
        for arm, e in ends.items():
            R = rec[arm]
            if e <= 0:
                continue
            lab_in = int(np.searchsorted(pos, e, side="left"))              # labelled windows inside the zone
            lab_err_in = int(w[:lab_in].sum())
            err_in = int(cum[e])
            unl_in, unl_err_in = e - lab_in, err_in - lab_err_in
            R["zone"][r] = True
            R["cov"][r] = e / N
            R["risk"][r] = err_in / e
            R["joint_unl"][r] = unl_err_in / (N - B)
            R["fdp_unl"][r] = unl_err_in / unl_in if unl_in > 0 else 0.0
    out = {"N": int(N), "budget": int(B), "alpha": float(alpha), "error_rate": float(err_o.mean()),
           "oracle_unit_coverage": float(max((e / N for e in range(1, N + 1) if cum[e] / e <= alpha), default=0.0))
           if N <= 50000 else None}
    for arm, R in rec.items():
        z = R["zone"]
        out[arm] = {"no_zone_rate": float((~z).mean()), "mean_coverage": float(R["cov"].mean()),
                    "median_coverage_when_zone": float(np.median(R["cov"][z])) if z.any() else None,
                    "p_zone_error_above_alpha": float((z & (R["risk"] > alpha + 1e-12)).mean()),
                    "mean_zone_error_rate": float(R["risk"].mean()),
                    "mean_unlabelled_wrong_and_in_zone": float(R["joint_unl"].mean()),
                    "mean_share_wrong_among_selected_unlabelled": float(R["fdp_unl"].mean())}
    return out


def run(cells, budgets=BUDGETS, draws=DRAWS, log=print):
    rows = {}
    for name, err_o in cells.items():
        N = err_o.size
        alpha = float(err_o.mean() / 2)
        rows[name] = {}
        for B in budgets:
            if not B < N / 2:
                continue
            c = run_cell(err_o, B, alpha, draws)
            rows[name][str(B)] = c
            log(f"[exp108] {name:44s} B {B:5d} " + " ".join(
                f"{a} cov {c[a]['mean_coverage']:.2f} P>a {c[a]['p_zone_error_above_alpha']:.3f}" for a in ("certify", "crc_joint", "bh_select")))
    return rows


def _dump(rec, path):
    import exp78_error_rate_estimation as e78
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e78.jsonable(rec), f, indent=1, allow_nan=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args(argv)
    if a.smoke:
        rng = np.random.default_rng(0)
        cells = {"synthetic": (rng.random(3000) < np.linspace(0.02, 0.5, 3000)).astype(float)}
        rows = run(cells, budgets=(300,), draws=min(a.draws, 100))
        rec = {"experiment": "exp108 smoke (synthetic)", "cells": rows}
        os.makedirs(a.out_dir, exist_ok=True)
        _dump(rec, os.path.join(a.out_dir, "exp108_summary_smoke.json"))
        return rec
    import exp95_level_cut as e95
    import exp89_finetuned_checkpoints as e89
    commit, dirty = e89.git_state()
    rows = run(e95.cells(), draws=a.draws)
    rec = {"experiment": "exp108, report-only: the certified zone against conformal risk control and conformal selection",
           "preregistered": False, "commit": commit, "dirty": dirty,
           "config": {"budgets": list(BUDGETS), "alpha": "half the map's error rate", "delta": DELTA, "draws": a.draws,
                      "seed": SEED, "cells": "exp95's 34 committed cells (seed-95 tie order for the products)",
                      "budget_rule": "B < N/2"},
           "cells": rows}
    os.makedirs(a.out_dir, exist_ok=True)
    _dump(rec, os.path.join(a.out_dir, "exp108_summary.json"))
    return rec


if __name__ == "__main__":
    main()
