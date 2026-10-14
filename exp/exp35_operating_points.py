"""exp35: operating points. Does any supported signal help confidence at a fixed review budget?

Every comparison in the repository ranks by the area under the risk-coverage
curve. A reviewer with a budget cares about something else: the fraction of
the errors that sit inside the windows they will look at. exp21 hinted the two
can disagree, tiling instability capturing 0.71 of the fine-tuned model's
errors at a 20% budget against 0.63 for confidence while losing on AURC.
Roadmap item 3, issue #9.

Metric: error capture at budget b, the expected fraction of a unit's errors
inside its round(b n) most suspect windows under random tie-breaking
(oe_inferencex.metrics.capture_at_budget_expected), so the nine-level
boundary indicator is neither credited nor penalised for raster order; the
stable-sort form exp21 used is reported alongside. Budgets 5, 10 and 20%.

Testbeds and inference, preregistered.
  Sen1Floods11 Bolivia (hand labels; primary). Signals: confidence, aligned
    tile-phase, boundary indicator, NDWI-gradient control, S2 patch variance,
    NDWI level, constant. Units: the tiles with at least three errors and
    three correct patches (the exp18 rule). Primary test: tiling instability
    against confidence at the 20% budget, per-tile one-sided exact sign test
    (the exp21 hint is directional). Pooled capture over all valid patches
    with a bootstrap over every tile that has valid patches (the same
    population as the pooled estimate) for every signal and budget.
  AWF fine-tuned model (expert points; exp21's per-window table, 16-px
    crops primary, 32-px secondary). Signals: confidence, boundary, tiling
    instability, probe disagreement, NDVI temporal-std control. Capture on
    the 344 windows, bootstrap over the 30 task clusters (exp21's design) of
    capture(signal) - capture(confidence): CI and P(signal better).
  WorldCover scenes (weak reference; secondary). Per-scene capture, one vote
    per river on the gain over confidence, one-sided exact sign test.
For every signal and budget the sign of the capture gain is set beside the
sign of the AURC gain on the same units, which is the question the issue
asks: does a signal help at a budget where it does not help on AURC?

Inputs: exp/out/exp21_finetuned_awf.csv (committed), the exp18 and exp11
feature caches on the cluster (recomputed when missing), exp/out/
exp11_scenes.npz, exp/out/exp03_cache.npz, data/floods/. Outputs:
exp/out/exp35_operating_points.csv, exp35_summary.json. --smoke: CPU, 2
scenes at 64 px, 4 flood tiles, the AWF table in full. fp32.
"""
import csv
import os
import traceback

import numpy as np

import harness_ab as hb
from harness_ab import CONF, TILE, BOUND, CTRL, CONST, CTRL_VAR, CTRL_LVL, OUT, RIVER  # noqa: F401
from oe_inferencex.metrics import aurc_expected, capture_at_budget, capture_at_budget_expected
from oe_inferencex.stats import sign_test, wins_losses_ties

BUDGETS = (0.05, 0.10, 0.20)
PRIMARY_BUDGET = 0.20
N_BOOT = 2000
AWF_SIGNALS = {"confidence (baseline)": ("logit_margin", -1.0), "boundary indicator": ("boundary", 1.0),
               "tile-phase (aligned)": ("tile_phase", 1.0), "probe disagreement": ("probe_disagree", 1.0),
               "control NDVI temporal std": ("control_ndvi_tstd", 1.0)}


def capture_all(sigs, err, mask=None):
    """{signal: {budget: expected capture}} and the stable-sort form, on the masked patches of one unit."""
    m = np.ones(np.shape(err), bool) if mask is None else mask
    exp_ = {k: capture_at_budget_expected(np.asarray(v)[m], err[m], BUDGETS) for k, v in sigs.items()}
    plain = {k: capture_at_budget(np.asarray(v)[m], err[m], BUDGETS) for k, v in sigs.items()}
    return exp_, plain


def bootstrap_pooled_gain(per_unit_sig, per_unit_err, per_unit_conf, budget, rng, n_boot=N_BOOT):
    """Bootstrap over units of pooled capture(signal) - pooled capture(confidence) at one budget.

    per_unit_*: lists of flat arrays per unit, one entry per tile with valid patches (the population of the pooled
    estimate), not only the tiles that pass the per-tile scoring rule. Returns (mean gain, lo, hi, P(gain > 0))."""
    n = len(per_unit_err)
    gains = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        e = np.concatenate([per_unit_err[i] for i in pick])
        if e.sum() == 0:
            continue
        s = np.concatenate([per_unit_sig[i] for i in pick])
        c = np.concatenate([per_unit_conf[i] for i in pick])
        gains.append(capture_at_budget_expected(s, e, (budget,))[budget] - capture_at_budget_expected(c, e, (budget,))[budget])
    g = np.array(gains)
    return {"mean": float(g.mean()), "lo": float(np.percentile(g, 2.5)), "hi": float(np.percentile(g, 97.5)),
            "p_better": float((g > 0).mean()), "n_boot": int(len(g))}


def sign_summary(gains, one_sided=False):
    w, l, t = wins_losses_ties(gains)
    return {"w": w, "l": l, "t": t, "sign_p": sign_test(w, l, "greater" if one_sided else "two-sided"),
            "median_gain": float(np.median(gains)) if len(gains) else None}


# ----------------------------------------------------------------------------- AWF fine-tuned model (exp21 table)
def part_awf(summary, rows):
    path = os.path.join(OUT, "exp21_finetuned_awf.csv")
    table = list(csv.DictReader(open(path)))
    summary["part_awf"] = {"source": path, "crops": {}}
    rng = np.random.default_rng(0)
    for crop in ("16", "32"):
        R = [r for r in table if r["crop"] == crop]
        err = np.array([float(r["error"]) for r in R])
        clusters = np.array([r["task"] for r in R])
        sigs = {k: sgn * np.array([float(r[col]) for r in R]) for k, (col, sgn) in AWF_SIGNALS.items()}
        exp_, plain = capture_all(sigs, err)
        aurc = {k: aurc_expected(v, err) for k, v in sigs.items()}
        ids = np.unique(clusters)
        by = {c: np.flatnonzero(clusters == c) for c in ids}
        boots = {}
        for k in sigs:
            if k == CONF:
                continue
            boots[k] = {}
            for b in BUDGETS:
                gains = []
                for _ in range(N_BOOT):
                    pick = rng.choice(ids, size=len(ids), replace=True)
                    sel = np.concatenate([by[c] for c in pick])
                    if err[sel].sum() == 0:
                        continue
                    gains.append(capture_at_budget_expected(sigs[k][sel], err[sel], (b,))[b] - capture_at_budget_expected(sigs[CONF][sel], err[sel], (b,))[b])
                g = np.array(gains)
                boots[k][str(b)] = {"gain": float(exp_[k][b] - exp_[CONF][b]), "lo": float(np.percentile(g, 2.5)), "hi": float(np.percentile(g, 97.5)),
                                    "p_better": float((g > 0).mean()), "aurc_gain": float(aurc[CONF] - aurc[k])}
        summary["part_awf"]["crops"][crop] = {"n_windows": len(R), "n_errors": int(err.sum()), "n_tasks": int(len(ids)),
                                             "capture_expected": {k: {str(b): v[b] for b in BUDGETS} for k, v in exp_.items()},
                                             "capture_stable_sort": {k: {str(b): v[b] for b in BUDGETS} for k, v in plain.items()},
                                             "aurc": aurc, "bootstrap_vs_confidence": boots}
        for k in sigs:
            row = {"part": "AWF", "unit": f"crop{crop}", "n_patches": len(R), "n_errors": int(err.sum()), "signal": k, "aurc": aurc[k]}
            for b in BUDGETS:
                row[f"capture@{b}"] = exp_[k][b]
                row[f"capture_stable@{b}"] = plain[k][b]
                row[f"gain_vs_confidence@{b}"] = exp_[k][b] - exp_[CONF][b]
            rows.append(row)
        print(f"AWF crop {crop}: capture at 20% " + ", ".join(f"{k.split(' (')[0]} {exp_[k][0.2]:.3f}" for k in sigs)
              + " | " + ", ".join(f"{k.split(' (')[0]} P(better) {boots[k]['0.2']['p_better']:.2f}" for k in boots), flush=True)


# ----------------------------------------------------------------------------- Sen1Floods11 Bolivia
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp35")
    if ctx is None:
        return
    N, err, ok = ctx["N"], ctx["err"], ctx["ok"]
    sigs = hb.base_signals_b(ctx)
    names = list(sigs)
    pooled_exp, pooled_plain = capture_all({k: np.asarray(v) for k, v in sigs.items()}, err, ok)
    aurc_pooled = {k: aurc_expected(np.asarray(v)[ok], err[ok]) for k, v in sigs.items()}
    per = {k: [] for k in names}
    per_aurc = {k: [] for k in names}
    all_sig, all_err = {k: [] for k in names}, []           # every tile with valid patches: the pooled estimate's population
    tiles = []
    for t in range(N):
        m = ok[t]
        if not m.any():
            continue
        e = err[t][m]
        all_err.append(e)
        for k in names:
            all_sig[k].append(np.asarray(sigs[k][t])[m])
        if e.sum() < 3 or e.sum() > len(e) - 3:              # the exp18 rule applies to the per-tile tests only
            continue
        tiles.append(t)
        cap, _ = capture_all({k: np.asarray(v[t]) for k, v in sigs.items()}, err[t], m)
        for k in names:
            per[k].append(cap[k])
            per_aurc[k].append(aurc_expected(np.asarray(sigs[k][t])[m], e))
    rng = np.random.default_rng(1)
    tests, boots = {}, {}
    for k in names:
        if k == CONF:
            continue
        tests[k], boots[k] = {}, {}
        for b in BUDGETS:
            gains = np.array([per[k][i][b] - per[CONF][i][b] for i in range(len(tiles))])
            tests[k][str(b)] = sign_summary(gains, one_sided=(k == TILE and b == PRIMARY_BUDGET))
            tests[k][str(b)]["aurc_w_l"] = list(wins_losses_ties(np.array(per_aurc[CONF]) - np.array(per_aurc[k]))[:2])
            boots[k][str(b)] = bootstrap_pooled_gain(all_sig[k], all_err, all_sig[CONF], b, rng)
    summary["part_b"].update({"n_tiles_scored": len(tiles), "n_tiles_pooled": len(all_err), "budgets": list(BUDGETS),
                              "pooled_capture_expected": {k: {str(b): v[b] for b in BUDGETS} for k, v in pooled_exp.items()},
                              "pooled_capture_stable_sort": {k: {str(b): v[b] for b in BUDGETS} for k, v in pooled_plain.items()},
                              "pooled_aurc": aurc_pooled, "per_tile_tests_vs_confidence": tests, "bootstrap_pooled_gain_vs_confidence": boots})
    pt = tests[TILE][str(PRIMARY_BUDGET)]
    summary["part_b"]["prereg"] = {"test": "tile-phase vs confidence, capture at the 20% budget, per-tile one-sided exact sign test",
                                   "w": pt["w"], "l": pt["l"], "t": pt["t"], "one_sided_p": pt["sign_p"], "median_gain": pt["median_gain"],
                                   "pooled_gain": pooled_exp[TILE][PRIMARY_BUDGET] - pooled_exp[CONF][PRIMARY_BUDGET],
                                   "pooled_bootstrap": boots[TILE][str(PRIMARY_BUDGET)]}
    print(f"bolivia: {len(tiles)} tiles; prereg tile-phase vs confidence at 20%: {pt['w']}/{pt['l']}/{pt['t']} one-sided p={pt['sign_p']:.3g}, "
          f"pooled capture {pooled_exp[TILE][0.2]:.3f} vs {pooled_exp[CONF][0.2]:.3f}", flush=True)
    for k in names:
        line = f"  {k:<28} pooled capture " + " ".join(f"{b:.2f}:{pooled_exp[k][b]:.3f}" for b in BUDGETS)
        if k in tests:
            line += " | per-tile W/L " + " ".join(f"{b:.2f}:{tests[k][str(b)]['w']}/{tests[k][str(b)]['l']}" for b in BUDGETS)
        print(line, flush=True)
    for k in names:
        row = {"part": "B", "unit": "bolivia (pooled)", "n_patches": int(ok.sum()), "n_errors": int(err[ok].sum()), "signal": k, "aurc": aurc_pooled[k]}
        for b in BUDGETS:
            row[f"capture@{b}"] = pooled_exp[k][b]
            row[f"capture_stable@{b}"] = pooled_plain[k][b]
            row[f"gain_vs_confidence@{b}"] = pooled_exp[k][b] - pooled_exp[CONF][b]
            if k in tests:
                row[f"W/L@{b}"] = f"{tests[k][str(b)]['w']}/{tests[k][str(b)]['l']}"
        rows.append(row)
    for i, t in enumerate(tiles):
        for k in names:
            row = {"part": "B", "unit": f"bolivia/tile{t}", "n_patches": int(ok[t].sum()), "n_errors": int(err[t][ok[t]].sum()), "signal": k,
                   "aurc": per_aurc[k][i]}
            for b in BUDGETS:
                row[f"capture@{b}"] = per[k][i][b]
                row[f"gain_vs_confidence@{b}"] = per[k][i][b] - per[CONF][i][b]
            rows.append(row)


# ----------------------------------------------------------------------------- WorldCover scenes (secondary)
def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    per, per_aurc = {}, {}
    for name in ctx["names"]:
        try:
            unit = hb.scene_unit(ctx, model, name, args, summary)
            if unit is None:
                continue
            sigs = hb.base_signals_a(unit, ctx)
            cap, _ = capture_all(sigs, unit["err"])
            per[name] = cap
            per_aurc[name] = {k: aurc_expected(np.asarray(v), unit["err"]) for k, v in sigs.items()}
            for k in sigs:
                row = {"part": "A", "unit": name, "n_patches": int(unit["err"].size), "n_errors": unit["n_err"], "signal": k, "aurc": per_aurc[name][k]}
                for b in BUDGETS:
                    row[f"capture@{b}"] = cap[k][b]
                    row[f"gain_vs_confidence@{b}"] = cap[k][b] - cap[CONF][b]
                rows.append(row)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    sn = sorted(per)
    summary["part_a"]["n_scenes"] = len(sn)
    if not sn:
        return
    names = list(per[sn[0]])
    tests = {}
    for k in names:
        if k == CONF:
            continue
        tests[k] = {}
        for b in BUDGETS:
            gains = {s: per[s][k][b] - per[s][CONF][b] for s in sn}
            r = hb.river_test(gains)
            tests[k][str(b)] = {"scene_w_l_t": list(wins_losses_ties(list(gains.values()))), "river": r,
                                "aurc_river": hb.river_test({s: per_aurc[s][CONF] - per_aurc[s][k] for s in sn})}
    summary["part_a"]["tests_vs_confidence"] = tests
    summary["part_a"]["median_capture"] = {k: {str(b): float(np.median([per[s][k][b] for s in sn])) for b in BUDGETS} for k in names}
    print(f"part A: {len(sn)} scenes; capture gain vs confidence, rivers W/L at 20%: "
          + ", ".join(f"{k.split(' (')[0]} {tests[k]['0.2']['river']['w']}/{tests[k]['0.2']['river']['l']}" for k in tests), flush=True)


def main():
    args = hb.make_parser(__doc__).parse_args()
    config = {"budgets": list(BUDGETS), "metric": "expected error capture at budget under random tie-breaking (stable-sort form reported alongside)",
              "prereg": "tile-phase vs confidence at the 20% budget on Sen1Floods11 Bolivia, per-tile one-sided exact sign test; "
                        "AWF: bootstrap over the 30 task clusters; WorldCover: one vote per river (secondary)", "n_boot": N_BOOT}

    def part_a_then_awf(model, args, summary, rows, cache):
        part_awf(summary, rows)
        part_a(model, args, summary, rows, cache)
    hb.run("exp35", "exp35 operating points at fixed review budgets", config, part_a_then_awf, part_b, args, "exp35_operating_points")


if __name__ == "__main__":
    main()
