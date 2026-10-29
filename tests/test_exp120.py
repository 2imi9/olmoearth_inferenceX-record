"""exp120 (docs/plan/valid_and_profiles_on_real_maps.md): estimate --valid and the per-profile verdicts on fully
labelled real maps. The run's counts against the second route's, cell by cell; the predictions recomputed from the
cells by a second traversal; the guards and the bars against the frozen plan. Standard library only."""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
PLAN = os.path.join(ROOT, "docs", "plan", "valid_and_profiles_on_real_maps.md")
VALID_CELLS = [("random", 300), ("random", 1000), ("confidence", 300), ("confidence", 1000),
               ("proportional", 300), ("proportional", 1000), ("class", 400), ("class", 1000)]
UNIT_FIELDS, TILE_FIELDS = ("confidence:4", "class"), ("confidence:4", "class", "grid:2,confidence:2")
UNIT_FACTORS, TILE_ALPHAS = (0.5, 1.0, 2.0), (0.1, 0.2, 0.3)


def _json(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


S, A, B = _json("exp120_summary.json"), _json("exp120_units.json"), _json("exp120_tile.json")


def test_the_second_route_imports_neither_the_run_nor_its_comparison():
    src = open(os.path.join(ROOT, "exp", "exp120_crosscheck.py"), encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"exp120_valid_profiles", "exp120_compare"}


def _first_cells():
    """The run's cells under the plan's cell numbers: {map: {cell: counts}}."""
    def cells(valid, profiles, fields, alpha_index):
        out = {}
        for c in valid:
            out[str(VALID_CELLS.index((c["design"], c["n"])))] = c
        for c in profiles:
            out[str(100 + 10 * fields.index(c["spec"]) + alpha_index(c))] = c
        return out
    first = {m["map"]: cells(m["valid"], m["profiles"], UNIT_FIELDS, lambda c: UNIT_FACTORS.index(c["factor"]))
             for m in A["maps"]}
    first["dynamic_world_tile"] = cells(B["valid"], B["profiles"], TILE_FIELDS, lambda c: TILE_ALPHAS.index(c["alpha"]))
    return first


def test_every_cell_recorded_by_a_second_route():
    """The second route (written from the plan and the package's source, without the first script) draws the same
    cells with the same seeds and grades them with its own code: every count both report is equal, on every map and
    cell of both arms."""
    first = _first_cells()
    second = {**_json("exp120_crosscheck_units.json")["maps"], **_json("exp120_crosscheck_tile.json")["maps"]}
    assert set(first) == set(second) and len(first) == 357
    n = 0
    for name, cells in first.items():
        assert set(cells) == set(second[name]["cells"]), name
        for cell, c in cells.items():
            s = second[name]["cells"][cell]
            keys = (("draws", "refused", "valid_covers", "valid_not_computed", "plain_covers") if int(cell) < 100
                    else ("draws", "refused", "any_wrong", "any_miss"))
            assert all(c[k] == s[k] for k in keys), (name, cell, {k: (c[k], s[k]) for k in keys})
            n += 1
    compare = _json("exp120_compare.json")
    assert compare["identical"] is True and compare["differences"] == 0 and compare["cells_compared"] == n


def test_the_predictions_by_a_second_traversal():
    cover = [(c["valid_covers"] / c["draws"]) for m in A["maps"] for c in m["valid"] if c["draws"]]
    prof = [c for m in A["maps"] for c in m["profiles"] if c["draws"]]
    tile_cover = [c["valid_covers"] / c["draws"] for c in B["valid"] if c["draws"]]
    tile_prof = [c for c in B["profiles"] if c["draws"]]
    p1, p2, p3 = S["P1"], S["P2"], S["P3"]
    assert p1["cells_a"] == len(cover) and p1["cells_b"] == len(tile_cover) == 8
    assert p1["share_a_at_bar"] == sum(x >= 0.917 for x in cover) / len(cover)
    assert p1["min_coverage_a"] == min(cover) and p1["min_coverage_b"] == min(tile_cover)
    assert p1["all_b_at_bar"] == all(x >= 0.917 for x in tile_cover)
    wrong = [c["any_wrong"] / c["draws"] for c in prof]
    miss = [c["any_miss"] / c["draws"] for c in prof]
    assert p2["cells_a"] == p3["cells_a"] == len(prof) and p2["cells_b"] == p3["cells_b"] == len(tile_prof) == 9
    assert p2["share_a_at_bar"] == sum(x <= 0.083 for x in wrong) / len(wrong)
    assert p3["share_a_at_bar"] == sum(x <= 0.145 for x in miss) / len(miss)
    assert p2["max_share_a"] == max(wrong) and p3["max_share_a"] == max(miss)
    assert p2["max_share_b"] == max(c["any_wrong"] / c["draws"] for c in tile_prof)
    assert p3["max_share_b"] == max(c["any_miss"] / c["draws"] for c in tile_prof)
    for p in (p1, p2, p3):
        assert p["held"] == (p["share_a_at_bar"] >= 0.99 and p["all_b_at_bar"])


def test_the_guards_and_the_draws():
    """Every cell has 400 draws taken (graded or refused); arm B's map class is the script's own argmax and its
    population the expert-labelled windows, on every draw."""
    for c in [c for m in A["maps"] for c in m["valid"] + m["profiles"]] + B["valid"] + B["profiles"]:
        assert c["draws"] + c["refused"] == 400
    assert S["guards"] == {"b_map_class_mismatch": 0, "b_population_mismatch": 0}
    assert A["draws"] == B["draws"] == 400 and len(A["maps"]) == 356


def test_the_bars_and_the_package_are_the_plans():
    plan = open(PLAN, encoding="utf-8").read()
    assert S["bars"] == {"cover": 0.917, "wrong": 0.083, "miss": 0.145, "share": 0.99}
    for text in ("0.917", "0.083", "0.145", "99%", "1.20.3"):
        assert text in plan, text
    assert A["package_version"] == B["package_version"] == "1.20.3"


def test_the_widths_by_a_second_traversal():
    """The median, over a design's cells, of the valid interval's mean width over the interval given without it."""
    import statistics
    for design, n in VALID_CELLS:
        cells = [c for m in A["maps"] for c in m["valid"]
                 if c["design"] == design and c["n"] == n and c["draws"] > c["plain_none"] and c["plain_width_sum"] > 0]
        ratios = [(c["valid_width_sum"] / c["draws"]) / (c["plain_width_sum"] / (c["draws"] - c["plain_none"]))
                  for c in cells]
        got = S["valid_by_design_a"][f"{design}_{n}"]["median_width_valid_over_plain"]
        assert abs(statistics.median(ratios) - got) < 1e-12, (design, n)


def test_the_descriptive_counts_by_a_second_traversal():
    D = _json("exp120_describe.json")
    av = [c for m in A["maps"] for c in m["valid"] if c["draws"]]
    ap_ = [c for m in A["maps"] for c in m["profiles"] if c["draws"]]
    assert D["a_valid_cells_below_0.95"] == sum(c["valid_covers"] < 0.95 * c["draws"] for c in av) == 85
    assert D["a_profile_cells_miss_above_0.10"] == sum(c["any_miss"] > 0.10 * c["draws"] for c in ap_) == 32
    assert round(sum(c["valid_covers"] for c in av) / sum(c["draws"] for c in av), 3) == 0.988
    conf300 = [c["plain_covers"] / (c["draws"] - c["plain_none"]) for c in av
               if (c["design"], c["n"]) == ("confidence", 300)]
    assert sum(x < 0.917 for x in conf300) == 10 and min(conf300) == 0.91


def test_where_the_cells_fall_and_on_how_many_maps():
    """Every --valid cell below 0.95 is a simple random sample, where the valid interval is estimate's own exact one;
    the stratified designs cover on at least 0.99 of draws in every cell; and the map counts the claims give."""
    graded = [(m, c) for m in A["maps"] for c in m["valid"] if c["draws"]]
    assert all(c["design"] == "random" for _, c in graded if c["valid_covers"] < 0.95 * c["draws"])
    assert all(c["valid_covers"] == c["plain_covers"] and c["valid_width_sum"] == c["plain_width_sum"]
               for _, c in graded if c["design"] == "random")
    assert min(c["valid_covers"] / c["draws"] for _, c in graded if c["design"] != "random") >= 0.99
    assert len({m["map"] for m, _ in graded}) == 269
    assert len({m["map"] for m, c in graded if (c["design"], c["n"]) == ("class", 400)}) == 152
    prof = [(m, c) for m in A["maps"] for c in m["profiles"] if c["draws"]]
    assert len({m["map"] for m, _ in prof}) == 174
    assert len({m["map"] for m, c in prof if c["spec"] == "class"}) == 127
