"""exp107: which of two maps is more accurate - the tool's design against McNemar's test on a whole-map sample.

Preregistration: docs/plan/which_map_baselines.md (the page is the specification). Exact probabilities, numpy only.

For each of nine product pairs, from the recorded counts (N rows, D where the maps differ, K_A and K_B where only map A
or only map B agrees with the reference), and each budget n:
  A  practice: n labels from all N rows, McNemar's exact test (the sign test) on the one-right rows
  Az practice in Foody's uncorrected z form, z^2 = (b - c)^2 / (b + c) > 1.96^2 (graded with A)
  Acc McNemar's chi-square with continuity correction (report-only)
  B  the tool: n labels among the D differing rows, the package's union-bound interval (plan.which_map_probability)
  C  B's sample with the sign test
Each gives P(names the better-agreeing map), P(names the other) and P(names none). The grade is refused until the page
is frozen; --smoke runs on constructed counts anywhere.
"""
import argparse
import json
import os
import sys
from fractions import Fraction
from math import comb

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, ROOT)

from oe_inferencex import plan  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "which_map_baselines.md")
OUT = os.path.join(EXP_DIR, "out")
BUDGETS, HEADLINE, ALPHA = (50, 100, 200, 400, 800), 100, 0.05
P1_PAIRS, P2_PAIRS, P2_GAIN, WRONG_MAX = 7, 5, 0.05, 0.05                      # the page's bars
POWERS, POWER_LADDER = (0.8, 0.9), (50, 100, 200, 400, 800, 1600, 3200, 6400)   # report-only
Z2, CHI2 = 1.959963984540054 ** 2, 3.841458820694124
_Q = Fraction(ALPHA).limit_denominator(10 ** 6)
PAIRS = (                                     # (name, file, Q1 key) as exp92 and exp94 recorded them
    ("lcmap_vs_esri_2018", "exp92_summary.json", None),
    ("nlcd_vs_lcmap_A2001", "exp94_summary.json", "nlcd_A2001:nlcd_vs_lcmap"),
    ("nlcd_vs_lcmap_A2006", "exp94_summary.json", "nlcd_A2006:nlcd_vs_lcmap"),
    ("nlcd_vs_lcmap_A2011", "exp94_summary.json", "nlcd_A2011:nlcd_vs_lcmap"),
    ("nlcd_vs_lcmap_B2011", "exp94_summary.json", "nlcd_B2011:nlcd_vs_lcmap"),
    ("nlcd_vs_lcmap_B2016", "exp94_summary.json", "nlcd_B2016:nlcd_vs_lcmap"),
    ("gfc2020_vs_worldcover", "exp94_summary.json", "gfc2020:gfc_vs_worldcover"),
    ("cgls_vs_esri_eastafrica_2017", "exp94_summary.json", "eastafrica_2017:cgls_vs_esri"),
    ("odse_vs_esri_europe_2017", "exp94_summary.json", "europe_2017:odse_vs_esri"),
)
EXPECTED = {"lcmap_vs_esri_2018": (24968, 5387, 2728, 2062), "nlcd_vs_lcmap_A2001": (8000, 1467, 452, 782),
            "nlcd_vs_lcmap_A2006": (8000, 1812, 598, 976), "nlcd_vs_lcmap_A2011": (8000, 1733, 727, 817),
            "nlcd_vs_lcmap_B2011": (4629, 982, 454, 417), "nlcd_vs_lcmap_B2016": (4629, 1115, 636, 364),
            "gfc2020_vs_worldcover": (21612, 1770, 962, 808), "cgls_vs_esri_eastafrica_2017": (13975, 5479, 2388, 2297),
            "odse_vs_esri_europe_2017": (49897, 9274, 5553, 3181)}                  # the page's table


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        return "frozen" if "**Status: frozen" in f.read() else "draft"


def load_pairs():
    out = {}
    for name, fn, key in PAIRS:
        d = json.load(open(os.path.join(OUT, fn), encoding="utf-8"))["Q1"]
        q = d if key is None else d[key]
        out[name] = tuple(int(q[k]) for k in ("N", "D", "K_A", "K_B"))
    return out


_CRIT = {}


def critical(m):
    """The largest k with P(Bin(m, 1/2) <= k) < alpha / 2, -1 if none: the sign test on (a, m - a) rejects at alpha
    exactly when min(a, m - a) <= critical(m) (stats.sign_test's p = min(1, 2 P(X <= min)))."""
    if m not in _CRIT:
        total, cum, k = 2 ** m, 0, -1
        for i in range(m + 1):
            cum += comb(m, i)
            if 2 * cum * _Q.denominator < _Q.numerator * total:          # integer arithmetic: no overflow
                k = i
            else:
                break
        _CRIT[m] = k
    return _CRIT[m]


def rejects(a, m, form):
    """Boolean over the counts a right only in map A (m - a only in B): does the paired test reject equal accuracy?"""
    a = np.asarray(a, int)
    c = m - a
    if form == "exact":
        k = critical(m)
        return np.minimum(a, c) <= k
    if m == 0:
        return np.zeros(a.shape, bool)
    if form == "z":
        return (a - c) ** 2 > Z2 * m
    if form == "cc":
        return (np.abs(a - c) - 1) ** 2 > CHI2 * m
    raise ValueError(form)


def sign_arm(P, T, KA, n, form="exact"):
    """P(names A), P(names B) when n rows are drawn from P rows holding T one-right rows (KA right only in A): m one-right
    rows drawn, hypergeometric; a of them right in A, hypergeometric over T; the paired test in the given form."""
    pa = pb = 0.0
    ms, pm = plan._hyper_pmf(P, T, n)
    for m, wm in zip(ms, pm):
        m = int(m)
        if m == 0:
            continue
        xs, px = plan._hyper_pmf(T, KA, m)
        a = xs.astype(int)
        rej = rejects(a, m, form)
        pa += wm * float(px[rej & (a > m - a)].sum())
        pb += wm * float(px[rej & (a < m - a)].sum())
    return pa, pb


def arms(N, D, KA, KB, n):
    """(P right, P wrong, P none) of arms A, Az, Acc, B and C at budget n; right = the map with the larger K."""
    T = KA + KB
    got = {"A": sign_arm(N, T, KA, min(n, N)), "Az": sign_arm(N, T, KA, min(n, N), "z"),
           "Acc": sign_arm(N, T, KA, min(n, N), "cc")}
    if n >= D:
        got["B"] = got["C"] = (float(KA > KB), float(KB > KA))
    else:
        got["B"] = plan.which_map_probability(N, D, n, KA, KB, rule="interval")   # the interval, as run at 1daa6bb
        got["C"] = sign_arm(D, T, KA, n)
    out = {}
    for arm, (pa, pb) in got.items():
        right, wrong = (pa, pb) if KA > KB else (pb, pa)
        out[arm] = {"right": right, "wrong": wrong, "none": max(0.0, 1 - right - wrong)}
    return out


def labels_for_power(N, D, KA, KB, arm, power, ladder=POWER_LADDER):
    """The first budget on the ladder (capped at N for practice, at D for the tool's sample) at which the arm names the
    better-agreeing map with probability at least `power`; a string when the largest budget does not reach it."""
    cap = N if arm.startswith("A") else D
    last = None
    for n in ladder:
        n = min(n, cap)
        if n == last:
            break
        last = n
        if arms(N, D, KA, KB, n)[arm]["right"] >= power:
            return n
    return f"not reached by {last}"


def verdicts(rows, n=HEADLINE):
    r = {p: rows[p]["budgets"][str(n)] for p in rows}
    p1_pairs = [p for p in r if r[p]["B"]["right"] >= r[p]["A"]["right"]]
    p1z_pairs = [p for p in r if r[p]["B"]["right"] >= r[p]["Az"]["right"]]
    p1_wrong = all(r[p][a]["wrong"] <= WRONG_MAX for p in r for a in ("A", "Az", "B"))
    p2_pairs = [p for p in r if r[p]["C"]["right"] >= r[p]["B"]["right"] + P2_GAIN]
    p2_wrong = all(r[p]["C"]["wrong"] <= WRONG_MAX for p in r)
    return {"P1": {"bar": f"B right >= A right and >= Az right, each on >= {P1_PAIRS} of 9 at n={n}; A, Az and B wrong <= "
                          f"{WRONG_MAX} on every pair",
                   "pairs_meeting": p1_pairs, "pairs_meeting_z": p1z_pairs, "wrong_ok": p1_wrong,
                   "holds": len(p1_pairs) >= P1_PAIRS and len(p1z_pairs) >= P1_PAIRS and p1_wrong},
            "P2": {"bar": f"C right >= B right + {P2_GAIN} on >= {P2_PAIRS} of 9 at n={n}; C wrong <= {WRONG_MAX} on every pair",
                   "pairs_meeting": p2_pairs, "wrong_ok": p2_wrong,
                   "holds": len(p2_pairs) >= P2_PAIRS and p2_wrong,
                   "decision": "C's verdict replaces the interval's in the package" if len(p2_pairs) >= P2_PAIRS and p2_wrong
                   else "the package is unchanged"}}


def run(pairs, budgets=BUDGETS, log=print):
    rows = {}
    for name, (N, D, KA, KB) in pairs.items():
        rec = {"N": N, "D": D, "K_A": KA, "K_B": KB, "delta": (KA - KB) / N, "budgets": {}}
        for n in budgets:
            rec["budgets"][str(n)] = arms(N, D, KA, KB, n)
            a = rec["budgets"][str(n)]
            log(f"[exp107] {name:30s} n {n:4d} " + " ".join(f"{k} {a[k]['right']:.3f}/{a[k]['wrong']:.4f}"
                                                            for k in ("A", "Az", "B", "C")))
        rec["labels_for_power"] = {str(pw): {arm: labels_for_power(N, D, KA, KB, arm, pw) for arm in ("A", "Az", "B", "C")}
                                   for pw in POWERS}
        rec["largest_wrong_rate"] = {arm: max(rec["budgets"][str(n)][arm]["wrong"] for n in budgets)
                                     for arm in ("A", "Az", "Acc", "B", "C")}
        rows[name] = rec
    return rows


def recorded_b(pairs, rows):
    """Report-only second route on the real pairs: arm B's exact rates beside exp92's and exp94's recorded Monte Carlo
    rates of the same design (2,000 draws), oriented to the better-agreeing map."""
    out = {}
    for name, fn, key in PAIRS:
        d = json.load(open(os.path.join(OUT, fn), encoding="utf-8"))["Q1"]
        q = d if key is None else d[key]
        cells = q.get("cells", {})
        out[name] = {}
        for n in (50, 100, 200):
            c = cells.get(f"disagree_{n}")
            if not c or "decides_right" not in c:
                continue
            ex = rows[name]["budgets"][str(n)]["B"]
            se = (max(ex["right"] * (1 - ex["right"]), 1e-6) / 2000) ** 0.5
            out[name][str(n)] = {"recorded_right": c["decides_right"], "exact_right": ex["right"],
                                 "recorded_wrong": c.get("decides_wrong"), "exact_wrong": ex["wrong"],
                                 "difference_in_se": (c["decides_right"] - ex["right"]) / se}
    return out


def _dump(rec, path):
    import exp78_error_rate_estimation as e78
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e78.jsonable(rec), f, indent=1, allow_nan=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)
    if a.smoke:
        pairs = {"constructed_large_gap": (5000, 1000, 600, 300), "constructed_no_gap": (5000, 1000, 400, 400),
                 "constructed_small": (3000, 400, 150, 130)}
        rows = run(pairs, budgets=(50, 100, 400))
        rec = {"experiment": "exp107 smoke (constructed counts)", "pairs": rows}
        os.makedirs(a.out_dir, exist_ok=True)
        _dump(rec, os.path.join(a.out_dir, "exp107_summary_smoke.json"))
        return rec
    status = prereg_status()
    if status != "frozen":
        raise SystemExit(f"{os.path.relpath(PLAN, ROOT)} is {status}: no arm is computed on the pairs before it is frozen")
    pairs = load_pairs()
    if pairs != EXPECTED:
        raise SystemExit(f"the recorded counts {pairs} are not the page's table")
    import exp89_finetuned_checkpoints as e89
    commit, dirty = e89.git_state()
    rows = run(pairs)
    crosscheck = recorded_b(pairs, rows)
    rec = {"experiment": "exp107: which of two maps is more accurate, the tool's design against McNemar's test",
           "prereg_page": os.path.relpath(PLAN, ROOT), "prereg_status": status, "commit": commit, "dirty": dirty,
           "config": {"budgets": list(BUDGETS), "headline": HEADLINE, "alpha": ALPHA,
                      "arms": {"A": "simple random sample of the N rows, McNemar's exact (sign) test",
                               "B": "labels among the D differing rows, the package's interval (exact probabilities)",
                               "C": "B's sample, the sign test"}},
           "pairs": rows, "crosscheck_recorded_monte_carlo_b": crosscheck, "verdicts": verdicts(rows)}
    os.makedirs(a.out_dir, exist_ok=True)
    _dump(rec, os.path.join(a.out_dir, "exp107_summary.json"))
    print(json.dumps(rec["verdicts"]))
    return rec


if __name__ == "__main__":
    main()
