"""exp95 and exp96 (certify's level cut) by a second route.

The ramp's construction is checked on its own terms; the preregistered verdicts are recomputed from the per-cell
curves in the summaries with code written here; and the harness both experiments share is checked against the
package's own certify_zone, draw by draw, on a committed cell."""
import json
import math
import os
import sys

import numpy as np
import pytest

from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
sys.path.insert(0, os.path.join(ROOT, "exp"))


def _summary(name):
    return json.load(open(os.path.join(OUT, name)))


@pytest.mark.parametrize("b", [7, 22, 45, 57, 114, 255])
def test_the_ramp_cut_is_continuous_never_rises_and_is_1_7_0_up_to_three_b_min(b):
    cuts = [est.zone_cut(n, b, "ramp") for n in range(1, 40 * b)]
    assert all(x >= y - 1e-15 for x, y in zip(cuts, cuts[1:]))                 # a level once tested stays tested
    assert all(est.zone_cut(n, b, "ramp") == est.zone_cut(n, b, "standard") == b / n for n in range(1, 3 * b + 1))
    assert est.zone_cut(3 * b, b, "ramp") == est.zone_cut(3 * b + 1, b, "ramp") == pytest.approx(1 / 3)
    assert est.zone_cut(9 * b, b, "ramp") == pytest.approx(1 / 3) and est.zone_cut(9 * b + 9, b, "ramp") < 1 / 3
    assert est.zone_cut(30 * b, b, "ramp") == pytest.approx(0.1)


def test_exp96_verdicts_from_the_curves():
    s = _summary("exp96_summary.json")
    held = {k: v for k, v in s["heldout"].items() if v["budgets"]}
    assert len(held) == 332
    se = math.sqrt(0.09 / s["draws"])
    viol = max(max(c[r]["violation"]) for v in held.values() for c in v["cut"].values() for r in ("prefix", "bonferroni"))
    assert viol <= 0.1 + 5 * se and round(viol, 4) == 0.1225

    def fall(curve):
        return max(curve[i] - min(curve[i + 1:]) for i in range(len(curve) - 1)) if len(curve) > 1 else 0.0
    big = [k for k, v in held.items() if fall(v["cut"]["k1"]["prefix"]["mean_coverage"]) > 0.05]
    halved = [k for k in big if fall(held[k]["cut"]["ramp"]["prefix"]["mean_coverage"])
              <= fall(held[k]["cut"]["k1"]["prefix"]["mean_coverage"]) / 2]
    losing = [k for k, v in held.items()
              if np.mean(v["cut"]["k1"]["prefix"]["mean_coverage"]) - np.mean(v["cut"]["ramp"]["prefix"]["mean_coverage"]) > 0.02]
    assert (len(big), len(halved), len(losing)) == (170, 131, 22)
    assert len(halved) >= 0.75 * len(big) and len(losing) <= 0.10 * len(held)
    assert all(v["ramp_equals_k1_small"] for v in held.values())
    assert all(s["prereg"][p]["holds"] for p in ("P1", "P2", "P3", "P4"))


def test_exp95_verdicts_from_the_curves():
    s = _summary("exp95_summary.json")
    cells = s["cells"]
    assert len(cells) == 34

    def fall(curve):
        return max(curve[i] - min(curve[i + 1:]) for i in range(len(curve) - 1))
    k1 = {k: fall(v["kappa"]["1"]["prefix"]["mean_coverage"]) for k, v in cells.items()}
    big = [k for k, f in k1.items() if f > 0.05]
    assert len(big) == 14                                                      # P2 fails: not 17
    k3 = [k for k in big if fall(cells[k]["kappa"]["3"]["prefix"]["mean_coverage"]) <= k1[k] / 2]
    loss = np.median([np.mean(v["kappa"]["1"]["prefix"]["mean_coverage"]) - np.mean(v["kappa"]["3"]["prefix"]["mean_coverage"])
                      for v in cells.values()])
    assert len(k3) == 12 and loss <= 0.02 and s["prereg"]["decision"]["default_kappa"] == 3.0


def test_the_shared_harness_decides_as_certify_zone():
    """exp95's run_cell, which exp96 reuses, against certify_zone on the same nested draws of a committed cell."""
    import exp95_level_cut as e95
    d = np.load(os.path.join(OUT, "exp78_units", "m_eurosat.npz"))
    order, _ = est.zone_order(d["margin"].astype(np.float64))
    err_o = d["err"].astype(np.float64)[order]
    margin = np.linspace(1, 0, err_o.size)
    for seed in range(3):
        r = e95.run_cell(err_o, draws=1, seed=seed, kappas=(1.0,))
        pos = np.random.default_rng(seed).choice(err_o.size, max(r["budgets"]), replace=False)
        for rule in ("prefix", "bonferroni"):
            for t, n in enumerate(r["budgets"]):
                z = est.certify_zone(margin, pos[:n], err_o[pos[:n]], r["alpha"], rule=rule, cut="standard")
                assert r["kappa"]["1"][rule]["mean_coverage"][t] == (z["coverage"] or 0.0), (seed, rule, n)


def test_the_pathwise_per_budget_rates_equal_exp96s():
    """exp96_audit.py's pathwise figures come through certify's own functions; at each single budget they must equal
    the violation rates exp96's vectorised harness recorded on the same draws."""
    a = _summary("exp96_audit.json")["pathwise"]
    s = _summary("exp96_summary.json")["heldout"]
    assert len(a) == 6
    for name, r in a.items():
        for cut in ("k1", "ramp"):
            assert r[cut]["per_budget_max"] == s[name]["cut"][cut]["prefix"]["max_violation"], (name, cut)
            assert r[cut]["some_doubling_budget"] >= r[cut]["per_budget_max"]          # a union is at least each part
