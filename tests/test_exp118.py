"""exp118 (docs/plan/grade_second_route_and_coverage.md): grade's intervals by a second route written from the
specification alone, and their coverage at NLCD 2016's design shape. The second route's independence, its comparison
read back from the rows, and the summary against the raw outputs. Standard library only."""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")


def _json(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def test_the_second_route_imports_neither_the_package_nor_the_design_study():
    src = open(os.path.join(ROOT, "exp", "exp118_crosscheck", "second_route.py"), encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"csv", "itertools", "json", "math", "os", "sys", "time", "fractions", "numpy"}, imported


def test_the_second_route_agrees_with_grade_on_every_quantity():
    """Every interval of the five runs, read back from the second route's rows, within the plan's bars (P1)."""
    A, S = _json("exp118_second_route.json"), _json("exp118_summary.json")
    assert A["P1_bars"]["every_family_within_its_bar"] is True and S["second_route"]["P1_holds"] is True
    assert A["P1_bars"]["one_side_missing"] == A["P1_bars"]["method_disagreements"] == 0
    assert sorted(A["runs"]) == ["gfc2020", "nlcd2011_l2_alt", "nlcd2016_8_alt", "nlcd2016_l2_alt", "nlcd2016_l2_primary"]
    st = A["selftest"]
    assert st["order_average_vs_enumeration_max_abs_diff"] < 1e-12 and st["hypergeometric_intervals_differing_from_exact"] == []


def test_the_summary_against_the_raw_coverage():
    B, S = _json("exp118_coverage_B1.json"), _json("exp118_summary.json")["coverage_B1"]
    Q = B["quantities"]
    assert S["quantities"] == len(Q) and B["errors"] == 0
    assert all(q["draws"] == 5000 for q in Q.values())
    assert S["valid_least"]["coverage"] == min(q["valid_coverage"] for q in Q.values()) >= S["bar"]
    assert S["normal_below_bar"] == sum(q["normal_coverage"] < q["bar"] for q in Q.values())
    assert S["normal_least"]["coverage"] == min(q["normal_coverage"] for q in Q.values())
