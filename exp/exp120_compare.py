#!/usr/bin/env python
"""exp120: the first route's counts against the second route's, cell by cell.

    python exp/exp120_compare.py --units A1.json --tile B1.json --second-units A2.json --second-tile B2.json --out FILE

The first route (exp/exp120_valid_profiles.py) lists cells by design and n or by field and alpha factor; the second
(exp/exp120_crosscheck.py) by the plan's cell numbers. This maps the first onto the numbers and compares every count
both report: for a --valid cell draws, refused, valid_covers, valid_not_computed and plain_covers; for a verdict cell
draws, refused, any_wrong and any_miss.
"""
import argparse
import json
import sys

VALID_CELLS = [("random", 300), ("random", 1000), ("confidence", 300), ("confidence", 1000),
               ("proportional", 300), ("proportional", 1000), ("class", 400), ("class", 1000)]
UNIT_FIELDS = ("confidence:4", "class")
TILE_FIELDS = ("confidence:4", "class", "grid:2,confidence:2")
UNIT_FACTORS = (0.5, 1.0, 2.0)
TILE_ALPHAS = (0.1, 0.2, 0.3)
VALID_KEYS = ("draws", "refused", "valid_covers", "valid_not_computed", "plain_covers")
PROFILE_KEYS = ("draws", "refused", "any_wrong", "any_miss")


def _first_cells(valid, profiles, fields, alpha_index):
    cells = {}
    for c in valid:
        cells[str(VALID_CELLS.index((c["design"], c["n"])))] = {k: c[k] for k in VALID_KEYS}
    for c in profiles:
        cells[str(100 + 10 * fields.index(c["spec"]) + alpha_index(c))] = {k: c[k] for k in PROFILE_KEYS}
    return cells


def main():
    ap = argparse.ArgumentParser()
    for name in ("--units", "--tile", "--second-units", "--second-tile", "--out"):
        ap.add_argument(name, required=True)
    a = ap.parse_args()
    u1, t1 = json.load(open(a.units)), json.load(open(a.tile))
    u2, t2 = json.load(open(a.second_units)), json.load(open(a.second_tile))
    first = {m["map"]: _first_cells(m["valid"], m["profiles"], UNIT_FIELDS,
                                    lambda c: UNIT_FACTORS.index(c["factor"])) for m in u1["maps"]}
    first["dynamic_world_tile"] = _first_cells(t1["valid"], t1["profiles"], TILE_FIELDS,
                                               lambda c: TILE_ALPHAS.index(c["alpha"]))
    second = {name: m["cells"] for name, m in u2["maps"].items()}
    second.update({name: m["cells"] for name, m in t2["maps"].items()})
    diffs, cells = [], 0
    for name in sorted(set(first) | set(second)):
        f, s = first.get(name), second.get(name)
        if f is None or s is None:
            diffs.append({"map": name, "only_in": "first" if s is None else "second"})
            continue
        for cell in sorted(set(f) | set(s), key=int):
            cells += 1
            if cell not in f or cell not in s:
                diffs.append({"map": name, "cell": cell, "only_in": "first" if cell not in s else "second"})
                continue
            bad = {k: [f[cell][k], s[cell].get(k)] for k in f[cell] if f[cell][k] != s[cell].get(k)}
            if bad:
                diffs.append({"map": name, "cell": cell, "differ": bad})
    out = {"maps_first": len(first), "maps_second": len(second), "cells_compared": cells, "differences": len(diffs),
           "first_differences": diffs[:50], "identical": not diffs,
           "versions": [u1["package_version"], t1["package_version"], u2["package_version"], t2["package_version"]]}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("maps_first", "maps_second", "cells_compared", "differences",
                                         "identical", "versions")}))
    print(json.dumps(diffs[:10], indent=1))
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
