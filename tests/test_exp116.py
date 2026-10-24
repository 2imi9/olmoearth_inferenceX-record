"""exp116 (docs/plan/target_class_certificate.md): the target rule, the class design and the certificate on fully
labelled real maps. The run's outputs against the independent second routes, the summary against the rows, and the
departures from the frozen plan."""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
RULES = ("prefix", "bonferroni", "target_0.25", "target_0.5", "target_0.75")


def _json(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def test_the_second_routes_import_neither_the_package_nor_the_run():
    for name in ("zone_audit.py", "audit_b.py"):
        src = open(os.path.join(ROOT, "exp", "exp116_crosscheck", name), encoding="utf-8").read()
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert not imported & {"oe_inferencex", "exp116_target_class_certificate"}, name


def test_arm_a_recorded_by_a_second_route():
    """Every cell of the planned files the second route ran (420, all of them) gives the same violation, certified
    and at-least shares for every rule, to the last digit, and the same mean coverage: the draws are the same."""
    run, cross = _json("exp116_ab.json"), _json("exp116_crosscheck_a.json")["results"]
    cells = {f"{r['map']}|{c['budget']}": c for r in run["arm_a"] for c in r.get("cells", [])}
    assert len(cross) == 420 and all(not k.startswith("olmoearth_base_rtx/") for k in cross)
    for key, c in cross.items():
        mine = cells[key]
        for rule in RULES:
            for field, value in c["rules"][rule].items():
                # the shares are counts over 400 draws, equal to the bit; the mean coverage is a float sum, equal up to
                # the order the two routes add it in
                ok = (abs(mine["rules"][rule][field] - value) < 1e-12 if field == "mean_coverage"
                      else mine["rules"][rule][field] == value)
                assert ok, (key, rule, field)


def test_arm_b_recorded_by_a_second_route():
    """Every map of the planned files the second route ran (269, all eligible ones) has the same allocation, user's
    accuracy coverage per class, joint coverage and overall coverage."""
    run = {r["map"]: r for r in _json("exp116_ab.json")["arm_b"] if "classes" in r}
    cross = _json("exp116_crosscheck_b.json")
    assert len(cross) == 269
    pairs = 0
    for m in cross:
        r = run[m["map"]]
        assert m["overall_coverage"] == r["overall_coverage"] and m["joint_coverage"]["L=n_classes"] == r["joint_coverage"]
        mine = {c["class"]: c for c in r["classes"]}
        for c in m["classes"]:
            assert (c["allocated"], c["ua_coverage"]) == (mine[c["class"]]["allocated"], mine[c["class"]]["ua_coverage"])
            pairs += 1
    assert pairs == 2402


def test_the_summary_is_the_rows_and_states_the_departures():
    S, run = _json("exp116_summary.json"), _json("exp116_ab.json")
    plan, ran = S["plan_population"], S["run_population"]
    assert plan["files"] == 356 and ran["files"] == 380 and ran["prereg"] == run["prereg"]
    p = plan["prereg"]
    assert p["P1"]["held"] and all(v["cells"] == 420 and v["cells_over_bar"] == 0 for v in p["P1"]["by_rule"].values())
    assert round(p["P1"]["by_rule"]["target_0.5"]["max_violation"], 4) == 0.0925
    assert p["P2"]["held"] and p["P2"]["cells_with_the_target_right"] == 771 and round(p["P2"]["share_target_at_least_bonferroni"], 4) == 0.965
    assert p["P3"]["held"] and (p["P3"]["pairs"], p["P3"]["maps"]) == (2402, 269) and round(p["P3"]["ua_share_at_bar"], 5) == 0.99958
    assert p["P4"]["held"] and p["P4"]["rare_pairs"] == 1148 and p["P4"]["share_class_design_narrower"] == 1.0
    assert any("380 unit files" in d for d in S["departures"])
    issued = plan["issued_certificates_descriptive"]["target_0.5"]
    assert (issued["cells_with_a_certificate"], issued["cells_over_delta"]) == (413, 46)
    tile = S["tile"]
    assert tile["P5"]["certified"] == 5 and tile["P5"]["all_verify"]
    assert [r["certified"] for r in tile["runs"]] == [True] * 5 + [False]
