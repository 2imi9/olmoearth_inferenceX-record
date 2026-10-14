"""exp97 (sequential certify) by a second route: the harness against the package, draw by draw and budget by budget,
on synthetic cells no grading reads; and the analysis on hand-built rows."""
import math
import os
import sys

import numpy as np

from oe_inferencex import estimate as est
from oe_inferencex import sequential as sq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))


def _cell(N, seed):
    """Errors in zone order: a rate rising with the zone, and a block of wrong windows past the 25% zone, so that the
    prefix from an anchor below it stops after the anchor."""
    rng = np.random.default_rng(seed)
    u = (np.arange(N) + 0.5) / N
    err = (rng.random(N) < 0.02 + 0.3 * u ** 2).astype(float)
    err[int(0.3 * N):int(0.36 * N)] = 1.0
    return err


def test_the_harness_decides_as_the_package_at_every_budget():
    import exp97_sequential as e97
    grid = np.asarray(est.ZONE_GRID)
    for N, draws in ((306, 8), (4397, 4)):
        err_o = _cell(N, N)
        margin = np.linspace(1, 0, N)
        r = e97.run_cell(err_o, draws=draws, seed=e97.cell_seed(f"test{N}"), keep=True, check=draws)
        assert r["check"]["mismatches"] == [] and r["check"]["comparisons"] > 0
        certified = set()
        for d in range(draws):
            pos = r["_pos"][d]
            for j, s in enumerate(est.ZONE_GRID):                  # every level's pass time
                size = max(1, int(round(s * N)))
                inz = pos < size
                lp = sq.level_path(np.flatnonzero(inz) + 1, err_o[pos][inz].astype(int), size, r["alpha"])
                got = r["_cross"][d, j]
                assert (lp["passed_at"] or math.inf) == got
            for t, n in enumerate(r["budgets"]):                   # every budget, every arm
                p = pos[:n]
                f = est.certify_zone(margin, p, err_o[p], r["alpha"], cut="standard")
                b = r["_best"]["F"][d, t]
                assert (f["coverage"] or 0.0) == (grid[b] if b >= 0 else 0.0), ("F", N, d, n)
                for arm, c in r["anchors"].items():
                    z = sq.certify_zone_sequential(margin, p, err_o[p], r["alpha"], anchor=c)
                    b = r["_best"][arm][d, t]
                    assert (z["coverage"] or 0.0) == (grid[b] if b >= 0 else 0.0), (arm, N, d, n)
                    if b >= 0:
                        certified.add(arm)
        assert certified, N                                          # the comparison is not all empty zones


def _row(budgets, F, S, viol=0.0):
    arm = lambda C: {"mean_coverage": C, "area": float(np.mean(C)), "pathwise_violation": viol,
                     "full_claim_violation": viol, "never_falls": True}
    return {"budgets": budgets, "arm": {"F": {**arm(F), "full_claim_violation": None}, "S(n0=300)": arm(S)},
            "check": {"comparisons": 1, "mismatches": [], "refused": 0}}


def test_rho_and_the_analysis_on_hand_built_rows():
    import exp97_sequential as e97
    bs = [100, 200, 400, 800, 1600]
    rows = {"a": _row(bs, [0.0, 0.2, 0.4, 0.4, 0.4], [0.0, 0.0, 0.1, 0.2, 0.4]),     # nf 200, S at 800: rho 4
            "b": _row(bs, [0.2, 0.4, 0.4, 0.4, 0.4], [0.0, 0.2, 0.3, 0.3, 0.3]),     # nf 100, S at 200: rho 2
            "c": _row(bs, [0.0, 0.0, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0, 0.0]),     # nf 1600: ladder ends before 3200
            "d": _row(bs, [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0, 0.0])}     # never: rho inf
    assert e97._rho(rows["a"], "S(n0=300)") == (4.0, 200, True)
    assert e97._rho(rows["c"], "S(n0=300)")[2] is False
    assert e97._rho(rows["d"], "S(n0=300)")[0] == math.inf
    a = e97.analyze(rows, audit_cells=[])
    assert a["P4"]["cells"] == 3 and a["P4"]["median_rho"] == 4.0 and a["P4"]["cells_never_reaching"] == 1
    assert a["P4"]["rho_quartiles_inverted_cdf"] == [2.0, 4.0, None] and a["P4"]["holds"] is False
    assert a["P1"]["holds"] and a["P2"]["holds"] and a["decision"]["ships_as_option"]
    rows["b"]["arm"]["S(n0=300)"]["full_claim_violation"] = 0.2
    rows["a"]["check"]["mismatches"] = [["zone", 0, "F", 100, 0.1, 0.0]]
    a = e97.analyze(rows, audit_cells=[])
    assert not a["P1"]["holds"] and not a["P2"]["holds"] and not a["decision"]["ships_as_option"]


def test_the_default_anchor_is_capped_at_a_quarter():
    import exp97_sequential as e97
    assert e97.default_anchor(300, 0.05) == 0.25                 # 71 / 300 = 0.237, the next grid level
    assert e97.default_anchor(300, 0.02) == 0.25                 # 179 / 300 = 0.60, capped
    assert e97.default_anchor(1000, 0.05) == 0.1                 # 71 / 1000 = 0.071
    assert sq.anchor_coverage(100, 0.01) == 0.25                 # 360 labels at alpha 0.01: no zone at 100, capped
    assert sq.anchor_coverage(10_000, 0.05) == 0.05              # the smallest grid level
    assert e97.cell_seed("a").entropy == e97.cell_seed("a").entropy and \
        e97.cell_seed("a").generate_state(1)[0] != e97.cell_seed("b").generate_state(1)[0]


def test_exp97_verdicts_from_the_summary():
    """P1 to P4 recomputed from the committed per-cell curves with code written here, not exp97's analyze."""
    import json
    s = json.load(open(os.path.join(ROOT, "exp", "out", "exp97_summary.json")))
    cells = {k: v for k, v in s["cells"].items() if v["budgets"]}
    assert len(cells) == 366 and sum(k.startswith("exp96:") for k in cells) == 332
    se = math.sqrt(0.09 / 400)
    S = [(k, a) for k, v in cells.items() for a in v["arm"] if a != "F"]
    rates = [cells[k]["arm"][a][m] for k, a in S for m in ("pathwise_violation", "full_claim_violation")]
    assert len(rates) == 3122 and max(rates) == 0.1175 and max(rates) <= 0.1 + 5 * se
    assert not any(r > 0.1 + 3 * se for r in rates)
    assert all(cells[k]["arm"][a]["full_claim_violation"] >= cells[k]["arm"][a]["pathwise_violation"] for k, a in S)
    assert sum(v["check"]["comparisons"] for v in cells.values()) == 37764
    assert all(v["check"]["mismatches"] == [] and v["check"]["refused"] == 0 for v in cells.values())
    F = [v["arm"]["F"]["pathwise_violation"] for v in cells.values()]
    assert sum(f > 0.1 + 5 * se for f in F) == 132 and max(F) == 0.435
    assert max(v["arm"]["F"]["per_budget_max_violation"] for v in cells.values()) == 0.1225
    rho = []
    for v in cells.values():
        if "S(n0=300)" not in v["arm"]:
            continue
        cf, cs, bs = v["arm"]["F"]["mean_coverage"], v["arm"]["S(n0=300)"]["mean_coverage"], v["budgets"]
        if max(cf) < 0.1:
            continue
        target = max(cf) / 2
        nf = next(b for b, c in zip(bs, cf) if c >= target)
        if bs[-1] < 2 * nf:
            continue
        ns = next((b for b, c in zip(bs, cs) if c >= target), None)
        rho.append(ns / nf if ns else math.inf)
    rho.sort()
    assert len(rho) == 213 and sum(math.isinf(r) for r in rho) == 2
    assert round(float(np.median(rho)), 2) == 1.71 and np.median(rho) <= 2
    assert [round(rho[53], 2), round(rho[159], 2)] == [1.55, 2.19]
    assert s["prereg"]["decision"]["ships_as_option"] is True
    # the inputs are exp96's recorded files and exp95's cells
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp96_summary.json")))["input_sha256"]
    assert s["input_sha256"] == rec
    assert {k for k in cells if k.startswith("exp96:")} == {"exp96:" + k[:-4] for k in rec}
    # every sequential arm's mean curve never falls (a second route on never_falls)
    assert all(all(b >= a - 1e-12 for a, b in zip(c, c[1:])) for k, a_ in S
               for c in [cells[k]["arm"][a_]["mean_coverage"]])
    # P3: the six maps exp96's audit followed
    six = json.load(open(os.path.join(ROOT, "exp", "out", "exp96_audit.json")))["pathwise"]
    f6 = [cells["exp96:" + k]["arm"]["F"]["pathwise_violation"] for k in six]
    assert len(f6) == 6 and min(f6) > 0.1 + 5 * se and [round(min(f6), 2), round(max(f6), 2)] == [0.33, 0.38]
    # the area sentence of the record
    M = [k for k, v in cells.items() if "S(n0=300)" in v["arm"]]
    area = lambda k, a: cells[k]["arm"][a]["area"]
    assert len(M) == 279
    assert [round(float(np.median([area(k, a) for k in M])), 3) for a in ("S(n0=300)", "F")] == [0.146, 0.222]
    ratio = [area(k, "S(n0=300)") / area(k, "F") for k in M if area(k, "F") > 0]
    assert len(ratio) == 275 and round(float(np.median(ratio)), 2) == 0.70
    more = [k for k in M if area(k, "S(n0=300)") > area(k, "F")]
    flood = [k for k in M if k.endswith("sen1floods11")]
    assert len(more) == 17 and len(flood) == 16 and sum(k in more for k in flood) == 14
    assert all(area(k, "S(c=0.1)") < area(k, "F") for k in flood)
    # the audit: five seeds, each with P1 and P4 holding, the main one equal to this recomputation
    au = json.load(open(os.path.join(ROOT, "exp", "out", "exp97_audit.json")))["by_seed"]
    assert set(au) == {"97", "971", "972", "973", "974"}
    assert all(v["P1"]["holds"] and v["P4"]["holds"] for v in au.values())
    assert au["97"]["P4"]["cells"] == 213 and au["97"]["P4"]["median_rho"] == float(np.median(rho))
