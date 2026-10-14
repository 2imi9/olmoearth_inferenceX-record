#!/usr/bin/env python
"""exp38: spectral ambiguity first, then boundary, then confidence, at fixed review budgets.

Why. exp36 showed that reviewing boundary windows first, then by confidence, captures more errors than confidence
alone at the 5% and 10% budgets on hand labels. exp37 measured five label-free cues on identical windows and found
that spectral ambiguity (|patch-mean NDWI| < 0.1) is the most error-enriched cue (7.2x on Bolivia, 3.8x on the
WorldCover scenes), the one least covered by confidence's review set (27% of the 5% set) and the one that separates
error rates inside it (0.51 with the cue against 0.33 without). The natural next order puts that cue ahead.

Preregistered (before running). Primary: on Sen1Floods11 Bolivia, the order "NDWI-ambiguous windows first (by
confidence), then boundary windows (by confidence), then the rest (by confidence)" against exp36's order "boundary
first, then confidence", capture at the 5% and 10% budgets, per-tile one-sided exact sign tests over the tiles with
3 <= errors <= n - 3 (the exp18 rule, 351 tiles), predicted direction: the NDWI-first order captures more; and a
tile bootstrap (2000 resamples over every tile with valid windows, shared by all comparisons) of the pooled gain,
with the lexicographic scores rebuilt from one global confidence midrank per resample (exp36's fix). Secondary, reported alongside: against
confidence; the 20% budget; the order "NDWI-ambiguous first, then confidence" without the boundary level; and the
27 WorldCover rule scenes with one vote per river (8 rivers, one-sided exact sign test on the mean per-river gain).
Caveat stated up front: the cue was chosen after seeing exp37's Bolivia enrichment, on these same tiles; its
threshold (0.1) was fixed before exp37 ran and the cue is enriched on both testbeds, but the WorldCover river vote
is the only check that does not share windows with the cue's selection. The cue is task-specific (water against
land). The fine-tuned AWF model is not tested: its per-window table (exp21) carries no NDWI.

Inputs: the per-window tables of exp37 (exp/out/exp37_patches_bolivia.npz, exp/out/exp37_patches_scenes.npz: valid
windows only, with confidence -|logit|, boundary indicator, -|NDWI| and the error of the same heads exp36 scored).
Consistency check recorded: exp36's per-tile counts of the boundary-first order against confidence (85/31/235 at
5%, 112/48/191 at 10%) must reproduce from the table. CPU only, seconds plus the bootstrap. Outputs:
exp/out/exp38_summary.json, exp/out/exp38_ndwi_first.csv (per-unit captures).
"""
import csv
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
from oe_inferencex.metrics import capture_at_budget_expected  # noqa: E402
from oe_inferencex.signals import midrank_pct  # noqa: E402
from oe_inferencex.stats import clustered_sign_test, sign_test, wins_losses_ties  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
BUDGETS = (0.05, 0.10, 0.20)
PREREG_BUDGETS = (0.05, 0.10)
NDWI_TOL = 0.1
N_BOOT = 2000
SEED = 1
CONF, BND, NDWI_BND, NDWI = ("confidence", "boundary, then confidence", "NDWI-ambiguous, then boundary, then confidence",
                             "NDWI-ambiguous, then confidence")
ORDERS = (CONF, BND, NDWI_BND, NDWI)
PRIMARY = {"order": NDWI_BND, "against": BND, "budgets": PREREG_BUDGETS}
EXP36_CHECK = {"0.05": (85, 31, 235), "0.1": (112, 48, 191)}


def levels(order, boundary, ndwi_level):
    """Integer review level per window for an order; higher is reviewed first, confidence breaks ties within a level."""
    b, a = np.asarray(boundary) > 0, np.asarray(ndwi_level) > -NDWI_TOL
    if order == CONF:
        return np.zeros(b.shape, int)
    if order == BND:
        return b.astype(int)
    if order == NDWI_BND:
        return np.where(a, 2, np.where(b, 1, 0))
    if order == NDWI:
        return a.astype(int)
    raise ValueError(order)


def score(order, conf, boundary, ndwi_level):
    """Lexicographic score: 2 * level + the confidence midrank percentile (exp36's construction, one more level)."""
    return 2.0 * levels(order, boundary, ndwi_level) + midrank_pct(conf)


def per_unit(table, unit_col):
    units = table[unit_col]
    err = table["err"] > 0.5
    ids = np.unique(units)
    rows, caps = [], {}
    for u in ids:
        m = units == u
        e = err[m].astype(float)
        n = int(m.sum())
        if e.sum() < 3 or e.sum() > n - 3:                       # exp18 rule, as exp36 scored the tiles
            continue
        c = {k: capture_at_budget_expected(score(k, table["conf"][m], table["boundary"][m], table["ndwi_level"][m]), e, BUDGETS) for k in ORDERS}
        caps[u] = c
        row = {"unit": str(u), "n_windows": n, "n_errors": int(e.sum())}
        for k in ORDERS:
            for b in BUDGETS:
                row[f"{k} @{b}"] = c[k][b]
        rows.append(row)
    return caps, rows


def paired(caps, a, b, budgets, one_sided_budgets=()):
    """Per-unit capture gains of order a over order b: wins/losses/ties and the exact sign test per budget."""
    out = {}
    units = list(caps)
    for bud in budgets:
        g = np.array([caps[u][a][bud] - caps[u][b][bud] for u in units])
        w, l, t = wins_losses_ties(g)
        one = bud in one_sided_budgets
        out[str(bud)] = {"w": w, "l": l, "t": t, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                         "median_gain": float(np.median(g)), "mean_gain": float(g.mean()), "n_units": len(units)}
    return out


def pooled_bootstrap(table, unit_col, pairs, rng):
    """Pooled capture of every order over every unit with valid windows, and one unit bootstrap shared by all the
    comparisons in `pairs`: each resample rebuilds every order's score from the pooled confidence with one global
    midrank, so the compared orders always see the same resampled windows."""
    units = table[unit_col]
    ids = np.unique(units)
    by = {u: np.flatnonzero(units == u) for u in ids}
    err = (table["err"] > 0.5).astype(float)

    def captures(sel):
        e = err[sel]
        return {k: capture_at_budget_expected(score(k, table["conf"][sel], table["boundary"][sel], table["ndwi_level"][sel]), e, BUDGETS) for k in ORDERS}

    full = captures(np.arange(len(err)))
    gains = {pair: {bud: [] for bud in BUDGETS} for pair in pairs}
    for _ in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([by[u] for u in pick])
        if err[sel].sum() == 0:
            continue
        c = captures(sel)
        for a, b in pairs:
            for bud in BUDGETS:
                gains[(a, b)][bud].append(c[a][bud] - c[b][bud])
    res = {}
    for a, b in pairs:
        res[(a, b)] = {}
        for bud in BUDGETS:
            g = np.array(gains[(a, b)][bud])
            res[(a, b)][str(bud)] = {"pooled": full[a][bud], "pooled_against": full[b][bud], "gain": full[a][bud] - full[b][bud],
                                     "boot_lo": float(np.percentile(g, 2.5)), "boot_hi": float(np.percentile(g, 97.5)),
                                     "p_better": float((g > 0).mean()), "n_boot": int(len(g))}
    return res


def river_votes(caps, rivers_of, a, b):
    out = {}
    for bud in BUDGETS:
        gains = {u: caps[u][a][bud] - caps[u][b][bud] for u in caps}
        r = clustered_sign_test(gains, rivers_of, aggregate="mean", alternative="greater")
        out[str(bud)] = {"w": r["w"], "l": r["l"], "t": r["t"], "one_sided_p": r["p"], "per_river": r["per_cluster"], "n_rivers": r["n_clusters"]}
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    summary = {"experiment": "exp38 NDWI-ambiguous first, then boundary, then confidence, at fixed budgets",
               "config": {"orders": list(ORDERS), "ndwi_tol": NDWI_TOL, "budgets": list(BUDGETS), "n_boot": N_BOOT, "seed": SEED,
                          "primary": PRIMARY, "unit_rule": "units with 3 <= errors <= n - 3 (exp18)", "inputs": ["exp37_patches_bolivia.npz", "exp37_patches_scenes.npz"]},
               "part_b": {}, "part_a": {}}
    rows = []
    # ------------------------------------------------------------ part B: Bolivia hand labels
    zb = np.load(os.path.join(OUT, "exp37_patches_bolivia.npz"))
    tb = {k: zb[k] for k in zb.files}
    caps_b, rows_b = per_unit(tb, "tile")
    for r in rows_b:
        r["part"] = "B"
    rows += rows_b
    check = paired(caps_b, BND, CONF, BUDGETS, PREREG_BUDGETS)
    summary["part_b"]["exp36_consistency"] = {b: {"recomputed": (check[b]["w"], check[b]["l"], check[b]["t"]), "exp36": EXP36_CHECK[b],
                                                  "match": (check[b]["w"], check[b]["l"], check[b]["t"]) == EXP36_CHECK[b]} for b in EXP36_CHECK}
    print("exp36 consistency:", summary["part_b"]["exp36_consistency"], flush=True)
    summary["part_b"].update({"n_windows": int(len(tb["err"])), "n_tiles_with_valid_windows": int(len(np.unique(tb["tile"]))),
                              "n_tiles_scored": len(caps_b), "n_errors": int((tb["err"] > 0.5).sum())})
    tests = {}
    pairs = ((NDWI_BND, BND), (NDWI_BND, CONF), (NDWI, CONF), (NDWI, BND), (BND, CONF))
    pooled = pooled_bootstrap(tb, "tile", pairs, rng)
    for a, b in pairs:
        one = PREREG_BUDGETS if (a, b) == (NDWI_BND, BND) else ()
        tests[f"{a} vs {b}"] = {"per_tile": paired(caps_b, a, b, BUDGETS, one), "pooled": pooled[(a, b)]}
        pt, po = tests[f"{a} vs {b}"]["per_tile"], tests[f"{a} vs {b}"]["pooled"]
        print(f"B {a} vs {b}: " + " | ".join(f"{bud}: {pt[str(bud)]['w']}/{pt[str(bud)]['l']}/{pt[str(bud)]['t']} p={pt[str(bud)]['sign_p']:.2g} "
                                          f"pooled {po[str(bud)]['pooled']:.3f} vs {po[str(bud)]['pooled_against']:.3f} CI [{po[str(bud)]['boot_lo']:+.3f}, {po[str(bud)]['boot_hi']:+.3f}]"
                                          for bud in BUDGETS), flush=True)
    summary["part_b"]["tests"] = tests
    summary["part_b"]["prereg"] = {str(b): {"per_tile": tests[f"{NDWI_BND} vs {BND}"]["per_tile"][str(b)], "pooled": tests[f"{NDWI_BND} vs {BND}"]["pooled"][str(b)]}
                                   for b in PREREG_BUDGETS}
    summary["part_b"]["median_capture"] = {k: {str(b): float(np.median([caps_b[u][k][b] for u in caps_b])) for b in BUDGETS} for k in ORDERS}
    # ------------------------------------------------------------ part A: WorldCover rule scenes, one vote per river
    za = np.load(os.path.join(OUT, "exp37_patches_scenes.npz"))
    ta = {k: za[k] for k in za.files}
    caps_a, rows_a = per_unit(ta, "scene")
    for r in rows_a:
        r["part"] = "A"
    rows += rows_a
    rivers_of = {s: str(ta["river"][ta["scene"] == s][0]) for s in caps_a}
    summary["part_a"].update({"n_windows": int(len(ta["err"])), "n_scenes_scored": len(caps_a), "n_errors": int((ta["err"] > 0.5).sum()),
                              "river_votes": {f"{a} vs {b}": river_votes(caps_a, rivers_of, a, b) for a, b in ((NDWI_BND, BND), (NDWI_BND, CONF), (NDWI, CONF), (BND, CONF))},
                              "per_scene": {f"{a} vs {b}": paired(caps_a, a, b, BUDGETS) for a, b in ((NDWI_BND, BND), (NDWI_BND, CONF), (NDWI, CONF), (BND, CONF))},
                              "median_capture": {k: {str(b): float(np.median([caps_a[u][k][b] for u in caps_a])) for b in BUDGETS} for k in ORDERS}})
    for key, rv in summary["part_a"]["river_votes"].items():
        ps = summary["part_a"]["per_scene"][key]
        print(f"A {key}: " + " | ".join(f"{b}: rivers {rv[str(b)]['w']}/{rv[str(b)]['l']} p={rv[str(b)]['one_sided_p']:.3f}, scenes {ps[str(b)]['w']}/{ps[str(b)]['l']}/{ps[str(b)]['t']}" for b in BUDGETS), flush=True)
    summary["runtime_s"] = time.time() - t0
    csv_path = os.path.join(OUT, "exp38_ndwi_first.csv")
    keys = ["part", "unit", "n_windows", "n_errors"] + [f"{k} @{b}" for k in ORDERS for b in BUDGETS]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "exp38_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {csv_path} ({len(rows)} rows) and exp38_summary.json; runtime {summary['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
