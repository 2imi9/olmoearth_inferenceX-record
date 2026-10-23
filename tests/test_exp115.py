"""exp115 (docs/plan/per_class_exact.md): an exact per-class report on 39 fully labelled maps. The run's outputs against
the independent second route, the outputs' privacy, the guarantee checked by enumeration on a small population, and
P4's post-hoc reading."""
import ast
import hashlib
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
SCRIPT = os.path.join(ROOT, "exp", "exp115_crosscheck.py")
RENAME = {"lcmap|exp92|lcmap": "lcmap|collection_1_3", "lcmap|exp92|esri": "lcmap|esri", "lcmap|exp112|oe13": "lcmap|olmoearth_v1_3"}


def _load():
    with open(os.path.join(OUT, "exp115_summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    with open(os.path.join(OUT, "exp115_crosscheck.json"), encoding="utf-8") as f:
        C = json.load(f)
    return S, C


def _name(n):
    n = n.replace("/", "|")
    return RENAME.get(n, n.replace("planteur|loio_", "planteur|loio:"))


def test_exp115_recorded_by_a_second_route():
    """Every class count is identical and every exact probability agrees to floating-point precision; the verdicts
    are the same."""
    src = open(SCRIPT, encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"oe_inferencex", "exp115_per_class_exact"}
    S, C = _load()
    assert C["script_sha256"] == hashlib.sha256(src.encode("utf-8")).hexdigest()
    CM = {_name(m["name"]): m for m in C["maps"]}
    assert set(CM) == set(S["maps"]) and len(S["maps"]) == 39
    cells = 0
    for name, rec in S["maps"].items():
        c = CM[name]
        off = c["legend"][0]
        assert (rec["N"], rec["C"], rec["unmapped"]) == (c["N"], c["C"], c["units_map_outside_legend"])
        for n, v in rec["joint_share_coverage"].items():
            if v != "census":
                assert abs(v - c["joint"][n]["all_share_intervals_cover"]) < 1e-9
        for k, cl in rec["classes"].items():
            x = c["classes"].get(str(int(k) + off))
            if x is None:
                assert cl["K"] == 0
                continue
            assert (cl["K"], cl["M"], cl["A"]) == (x["K"], x["M"], x["A"])
            for n, cell in cl["by_n"].items():
                y = x["by_budget"][n]
                cells += 1
                for a, b in ((cell["share"]["coverage"], y["share_coverage"]), (cell["share"]["flag_too_little"], y["flag_too_little"]),
                             (cell["share"]["flag_too_much"], y["flag_too_much"]), (cell["pa"]["coverage"], y["pa_coverage"]),
                             (cell["pa"]["upper_at_most_bar"], y["pa_upper_le_0.25"])):
                    assert abs(a - b) < 2e-9
    assert cells == 706
    for p, want in (("P1", "holds"), ("P2", "holds"), ("P3", "fails"), ("P4", "fails"), ("P5", "fails")):
        assert C["verdicts"][p]["verdict"] == want and S["verdicts"][p]["holds"] is (want == "holds")


def test_exp115_outputs_hold_aggregates_only():
    S, C = _load()

    def longest(o):
        if isinstance(o, dict):
            return max([longest(v) for v in o.values()] + [0])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0
    assert longest(S) <= 64


def test_exp115_share_and_pa_intervals_by_enumeration():
    """On a 12-unit, three-class population, every sample of 4 enumerated: the share interval covers each class's share
    with probability at least 1 - 0.05/3 and the producer's-accuracy interval [A_lo/K_hi, min(1, A_hi/K_lo, M/K_lo)]
    at least 1 - 0.05/3, using the package's interval."""
    from oe_inferencex import estimate as est
    m = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 2])
    r = np.array([0, 0, 1, 0, 1, 1, 2, 2, 2, 1, 1, 0])
    N, n, C = 12, 4, 3
    K, M = np.bincount(r, minlength=C), np.bincount(m, minlength=C)
    A = np.bincount(r[m == r], minlength=C)
    share, pa = np.zeros(C), np.zeros(C)
    total = 0
    for s in itertools.combinations(range(N), n):
        s = list(s)
        total += 1
        k = np.bincount(r[s], minlength=C)
        a = np.bincount(r[s][m[s] == r[s]], minlength=C)
        for c in range(C):
            lo, hi = est.hypergeom_interval(int(k[c]), n, N, conf=1 - 0.05 / C)
            share[c] += round(lo * N) <= K[c] <= round(hi * N)
            Alo, Ahi = (round(x * N) for x in est.hypergeom_interval(int(a[c]), n, N, conf=1 - 0.05 / (2 * C)))
            Klo, Khi = (round(x * N) for x in est.hypergeom_interval(int(k[c]), n, N, conf=1 - 0.05 / (2 * C)))
            lower = Alo / Khi if Khi else 0.0
            upper = min(1.0, Ahi / Klo, M[c] / Klo) if Klo else 1.0
            pa[c] += lower <= A[c] / K[c] <= upper
    assert (share / total >= 1 - 0.05 / C).all() and (pa / total >= 1 - 0.05 / C).all()


def test_exp115_p4_reading_by_relative_error():
    """P4's 69 flags above 0.10 among 164 qualifying classes: 62 on classes drawn at less than half or more than twice
    their share; among the 59 drawn within 25%, only Mayotte's Background (three maps)."""
    S, _ = _load()
    rows = [(cl["M"] / cl["K"], cl["by_n"]["400"]["share"]["flag_too_little"] + cl["by_n"]["400"]["share"]["flag_too_much"], m)
            for m, rec in S["maps"].items() if rec["N"] > 400 for cl in rec["classes"].values()
            if "400" in cl["by_n"] and 50 * abs(cl["M"] - cl["K"]) <= rec["N"]]
    assert len(rows) == 164 and sum(f > 0.1 for _, f, _ in rows) == 69
    assert sum(f > 0.1 and (q < 0.5 or q > 2) for q, f, _ in rows) == 62
    assert sorted(m for q, f, m in rows if 0.8 <= q <= 1.25 and f > 0.1) == [
        "planteur|loio:Mayotte|aef", "planteur|loio:Mayotte|oe13", "planteur|loio:Mayotte|tessera2"]
