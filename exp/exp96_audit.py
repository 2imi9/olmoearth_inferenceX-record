#!/usr/bin/env python
"""exp96's pre-record audit, made reproducible (8 October 2026; after the result, not preregistered).

    python exp/exp96_audit.py digest DIR      # P3 at each reseeded run in DIR (seed*.json, exp96's run_cell output)
    python exp/exp96_audit.py pathwise        # a reviewer who adds labels and reruns certify, on six held-out maps

**digest.** The audit reran exp96's held-out grading with exp96's own run_cell at seeds 1 to 8 (400 draws each, all
else as preregistered; the files stay local and their sha256 is recorded). P3 asks that ramp halve kappa 1's largest
prefix fall on at least 75% of the maps where it exceeds 0.05: this records, per seed, the maps counted, the halvings
and the area condition, and P3's verdict, beside the preregistered seed 96.

**pathwise.** certify's guarantee holds at a budget fixed before any label. exp95 and exp96 grade it so, budget by
budget. A reviewer who labels, certifies, adds labels and certifies again runs several tests on nested samples; this
measures, on the six held-out maps with the largest per-budget violation rate in exp96, the share of nested draws on
which some budget of the doubling ladder 50, 100, ..., 3,200 (and some budget of exp96's whole ladder) certifies a zone
wrong more than alpha, under kappa 1 and ramp. The draws, cuts and rules are exp96's (seed 96, 400 draws).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex import estimate as est  # noqa: E402
import exp95_level_cut as e95  # noqa: E402
import exp96_level_cut_heldout as e96  # noqa: E402

OUT = os.path.join(ROOT, "exp", "out")
AUDIT = os.path.join(OUT, "exp96_audit.json")


def _p3(held):
    rows = {k: v for k, v in held.items() if v["budgets"]}
    big = [k for k, r in rows.items() if r["cut"]["k1"]["prefix"]["largest_fall"] > 0.05]
    halved = [k for k in big if rows[k]["cut"]["ramp"]["prefix"]["largest_fall"]
              <= rows[k]["cut"]["k1"]["prefix"]["largest_fall"] / 2]
    losing = [k for k, r in rows.items() if r["cut"]["k1"]["prefix"]["area"] - r["cut"]["ramp"]["prefix"]["area"] > 0.02]
    viol = max(r["cut"][c][rule]["max_violation"] for r in rows.values() for c in r["cut"] for rule in e96.RULES)
    return {"big": len(big), "halved": len(halved), "share": len(halved) / len(big), "losing": len(losing),
            "of": len(rows), "holds": len(halved) >= 0.75 * len(big) and len(losing) <= 0.10 * len(rows),
            "max_violation": viol}


def digest(directory):
    out = json.load(open(AUDIT)) if os.path.exists(AUDIT) else {}
    seeds = {"96": {**_p3(json.load(open(os.path.join(OUT, "exp96_summary.json")))["heldout"]), "file": "exp96_summary.json"}}
    for f in sorted(glob.glob(os.path.join(directory, "seed*.json"))):
        s = os.path.basename(f)[4:-5]
        seeds[s] = {**_p3(json.load(open(f))["heldout"]), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()}
    out["p3_by_seed"] = seeds
    out["p3_holds_on"] = f"{sum(v['holds'] for v in seeds.values())} of {len(seeds)} seeds"
    json.dump(out, open(AUDIT, "w"), indent=1)
    for k, v in seeds.items():
        print(k, {x: v[x] for x in ("big", "halved", "share", "losing", "holds", "max_violation")})


def _violations(err_o, cut, rule, draws=e96.DRAWS, seed=e96.SEED):
    """(budgets, draws x budgets bool): the certified zone wrong more than alpha, on exp96's nested draws."""
    N = err_o.size
    alpha = float(err_o.mean() / 2)
    b_min = est.min_labels_to_certify(alpha, e96.DELTA)
    grid = est.ZONE_GRID
    sizes = np.array([max(1, int(round(c * N))) for c in grid])
    risk = np.cumsum(err_o)[sizes - 1] / sizes
    bs = e95.budgets(N)
    rng = np.random.default_rng(seed)
    out = np.zeros((draws, len(bs)), bool)
    for d in range(draws):
        pos = rng.choice(N, max(bs), replace=False)
        for t, n in enumerate(bs):
            lv = [j for j, c in enumerate(grid) if c >= cut(n, b_min) - 1e-12]
            if not lv:
                continue
            b, k = est.zone_counts(pos[:n], err_o[pos[:n]], sizes[lv])
            p = [est.zone_pvalue(int(kk), int(bb), int(sz), alpha) for kk, bb, sz in zip(k, b, sizes[lv])]
            level = e96.DELTA if rule == "prefix" else est._level(e96.DELTA) / len(lv)
            _, best = est.apply_zone_rule(np.array(p), b, k, alpha, level if rule == "prefix" else e96.DELTA, rule,
                                          n=[int(x) for x in sizes[lv]])
            if best is not None:
                out[d, t] = risk[lv[best]] > alpha
    return bs, out


def pathwise(n_cells=6):
    s = json.load(open(os.path.join(OUT, "exp96_summary.json")))["heldout"]
    worst = sorted((k for k, v in s.items() if v["budgets"]),
                   key=lambda k: -max(s[k]["cut"][c]["prefix"]["max_violation"] for c in ("k1", "ramp")))[:n_cells]
    cells = dict(e96.heldout_cells()[0])
    res = {}
    for name in worst:
        res[name] = {}
        for cut in ("k1", "ramp"):
            bs, v = _violations(cells[name], e96.CUTS[cut], "prefix")
            doubling = [i for i, b in enumerate(bs) if b in {min(bs, key=lambda x: abs(x - 50 * 2 ** j)) for j in range(7)}]
            res[name][cut] = {"per_budget_max": float(v.mean(axis=0).max()),
                              "some_doubling_budget": float(v[:, doubling].any(axis=1).mean()),
                              "some_budget": float(v.any(axis=1).mean()),
                              "doubling_budgets": [bs[i] for i in doubling]}
        print(name, json.dumps({c: {k: round(x, 4) if isinstance(x, float) else x for k, x in r.items() if k != "doubling_budgets"}
                                 for c, r in res[name].items()}), flush=True)
    out = json.load(open(AUDIT)) if os.path.exists(AUDIT) else {}
    out["pathwise"] = res
    json.dump(out, open(AUDIT, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["digest"]:
        digest(sys.argv[2])
    elif sys.argv[1:2] == ["pathwise"]:
        pathwise()
    else:
        raise SystemExit(__doc__)
