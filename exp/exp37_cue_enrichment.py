#!/usr/bin/env python
"""exp37: cue enrichment on identical windows, for the explanation layer.

Why. The explanation layer (oe_inferencex.explain; docs/plan/roadmap.md) attaches to each review window the
label-free cues that fire, each with its measured share among error windows against correct windows. The shares
recorded so far come from different experiments on different windows and error definitions (boundary from exp18,
seasonal water from exp25 on WorldCover disagreements, ...). This experiment measures every label-free cue the
package can derive on identical windows with one error definition per testbed: the share of each cue among error
and among correct windows with a cluster bootstrap of the ratio, the cues' co-occurrence, the share of errors no
cue explains, and how much of confidence's own review set each cue explains at the 5, 10 and 20% budgets (with the
error rate inside the set among windows with and without the cue). Descriptive, no preregistered test: the
selective questions were settled in exp35 and exp36; the numbers here feed the cue library.

Cues (per 4-px window, all label-free): boundary (indicator > 0); low confidence (the 20% least confident windows of
the unit, ties included); unstable (tile-phase in the unit's top 20%); NDWI-ambiguous (|patch-mean NDWI| < 0.1);
dihedral disagreement (exp36's std over the 8 flips and rotations in the unit's top 20%, when exp36's cache is
present for every unit of the part). Part A: the 27 WorldCover rule scenes (exp13 error set, seed-0 Base head),
scene-clustered bootstrap, per-river shares alongside. Part B: Sen1Floods11 Bolivia hand labels (exp18 head), every
valid patch, tile-clustered bootstrap. Inputs: exp/out/exp11_scenes.npz, exp/out/exp11_feats.npz,
exp/out/exp18_feats.npz, data/floods/, exp/out/exp36_cache.npz (optional). Outputs: exp/out/exp37_summary.json,
exp/out/exp37_cue_enrichment.csv, and the per-window tables exp/out/exp37_patches_scenes.npz and
exp/out/exp37_patches_bolivia.npz (compressed) from which the package tests recompute the shares.
--smoke: CPU, 2 scenes at 64 px, 4 tiles, _smoke outputs.
"""
import os
import sys
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import harness_ab as hb  # noqa: E402
from oe_inferencex.assess import review_mask  # noqa: E402
from oe_inferencex.explain import cooccurrence, cue_enrichment, top_fraction  # noqa: E402

CONF, TILE, BOUND, CTRL_LVL = hb.CONF, hb.TILE, hb.BOUND, hb.CTRL_LVL
BUDGETS = (0.05, 0.10, 0.20)
FRACTION = 0.2          # "top 20% of the unit" cues
NDWI_TOL = 0.1          # |NDWI| below this = spectrally ambiguous
N_BOOT = 2000
CUE_NAMES = ["boundary", "low_confidence", "unstable", "ndwi_ambiguous", "dihedral_disagree"]
TABLE_COLS = ("row", "col", "err", "conf", "tile_phase", "boundary", "ndwi_level", "dihedral")


def unit_cues(sigs, valid, dihedral_std=None):
    """Boolean cue maps for one unit (scene or tile) from the harness's base signals; invalid windows carry none."""
    cues = {"boundary": (np.asarray(sigs[BOUND]) > 0) & valid,
            "low_confidence": top_fraction(sigs[CONF], FRACTION, valid),       # CONF = -|logit|: high = least confident
            "unstable": top_fraction(sigs[TILE], FRACTION, valid),
            "ndwi_ambiguous": (np.asarray(sigs[CTRL_LVL]) > -NDWI_TOL) & valid}  # CTRL_LVL = -|NDWI|
    if dihedral_std is not None:
        cues["dihedral_disagree"] = top_fraction(dihedral_std, FRACTION, valid)
    return cues


def review_masks(conf, valid):
    """Confidence's review set at each budget within one unit, exactly as the assessor builds it (assess.review_mask:
    k = max(1, round(b n)) most suspect valid windows, ties by descending raster position)."""
    return {b: review_mask(conf, valid, b) for b in BUDGETS}


def load_dihedral_std(args, key):
    """Std over exp36's eight transformed probability maps for `key`, or None when the cache or key is absent."""
    path = os.path.join(hb.OUT, f"exp36_cache{args.suffix}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return np.asarray(z[key], dtype=np.float64).std(0) if key in z.files else None


class Collector:
    """Accumulates per-window rows (valid windows only) for one part."""

    def __init__(self, unit_col):
        self.unit_col = unit_col
        self.cols = {unit_col: []}
        self.cols.update({k: [] for k in TABLE_COLS})
        self.cues = {n: [] for n in CUE_NAMES}
        self.review = {b: [] for b in BUDGETS}
        self.units, self.dihedral_missing = [], 0

    def add(self, unit_id, sigs, err, valid, dihedral_std):
        cues = unit_cues(sigs, valid, dihedral_std)
        rev = review_masks(sigs[CONF], valid)
        G0, G1 = valid.shape
        rr, cc = np.meshgrid(np.arange(G0), np.arange(G1), indexing="ij")
        sel = valid.ravel()
        vals = {"row": rr, "col": cc, "err": err, "conf": sigs[CONF], "tile_phase": sigs[TILE], "boundary": sigs[BOUND],
                "ndwi_level": sigs[CTRL_LVL], "dihedral": dihedral_std if dihedral_std is not None else np.full(valid.shape, np.nan)}
        self.cols[self.unit_col].append(np.full(int(sel.sum()), unit_id))
        for k in TABLE_COLS:
            self.cols[k].append(np.asarray(vals[k], dtype=np.float64).ravel()[sel])
        for n in CUE_NAMES:
            self.cues[n].append(cues[n].ravel()[sel] if n in cues else np.zeros(int(sel.sum()), bool))
        if dihedral_std is None:
            self.dihedral_missing += 1
        for b in BUDGETS:
            self.review[b].append(rev[b].ravel()[sel])
        self.units.append(unit_id)
        return cues, rev

    def table(self):
        return {k: np.concatenate(v) for k, v in self.cols.items()}

    def analysis(self, clusters, rivers=None):
        err = np.concatenate(self.cols["err"]) > 0.5
        names = [n for n in CUE_NAMES if not (n == "dihedral_disagree" and self.dihedral_missing)]
        cues = {n: np.concatenate(self.cues[n]) for n in names}
        review = {b: np.concatenate(self.review[b]) for b in BUDGETS}
        return analyse(err, cues, review, clusters, rivers) | {"n_units": len(self.units), "dihedral_missing_units": self.dihedral_missing}


def analyse(err, cues, review, clusters, rivers=None):
    """Pooled shares, enrichment ratios with a cluster bootstrap, co-occurrence, unexplained shares, review-set coverage."""
    names = list(cues)
    res = {"n_windows": int(len(err)), "n_errors": int(err.sum()), "error_rate": float(err.mean()), "cues": {}}
    for n in names:
        r = cue_enrichment(cues[n], err, clusters=clusters, n_boot=N_BOOT)
        if rivers is not None:
            r["per_river"] = {}
            for rv in sorted(set(rivers.tolist())):
                m = rivers == rv
                e_, c_ = err[m], cues[n][m]
                r["per_river"][rv] = {"share_errors": float(c_[e_].mean()) if e_.any() else None,
                                      "share_correct": float(c_[~e_].mean()) if (~e_).any() else None}
        res["cues"][n] = r
    stack = np.stack([cues[n] for n in names]) if names else np.zeros((0, len(err)), bool)
    n_cues = stack.sum(0)
    res["unexplained"] = {"share_of_errors_without_cue": float((n_cues[err] == 0).mean()) if err.any() else None,
                          "share_of_correct_without_cue": float((n_cues[~err] == 0).mean()) if (~err).any() else None,
                          "share_of_errors_with_two_or_more": float((n_cues[err] >= 2).mean()) if err.any() else None,
                          "mean_cues_per_error": float(n_cues[err].mean()) if err.any() else None,
                          "mean_cues_per_correct": float(n_cues[~err].mean()) if (~err).any() else None}
    _, m_err = cooccurrence(cues, err)
    _, m_ok = cooccurrence(cues, ~err)
    res["cooccurrence"] = {"names": names, "among_errors": m_err.tolist(), "among_correct": m_ok.tolist()}
    res["review_set"] = {}
    for b, m in review.items():
        k = int(m.sum())
        row = {"n_windows": k, "error_rate_in_set": float(err[m].mean()) if k else None,
               "errors_captured": float(err[m].sum() / max(err.sum(), 1)),
               "share_with_any_cue": float((n_cues[m] > 0).mean()) if k else None, "cues": {}}
        for n in names:
            c, nc = cues[n] & m, (~cues[n]) & m
            row["cues"][n] = {"share_in_set": float(c.sum() / max(k, 1)),
                              "error_rate_with_cue": float(err[c].mean()) if c.any() else None,
                              "error_rate_without_cue": float(err[nc].mean()) if nc.any() else None}
        res["review_set"][str(b)] = row
    return res


def csv_rows(part, res, rows):
    for n, r in res["cues"].items():
        rows.append({"part": part, "scope": "pooled", "cue": n, "n_windows": r["n"], "n_errors": r["n_errors"], "n_with_cue": r["n_with_cue"],
                     "share_errors": r["share_errors"], "share_correct": r["share_correct"], "enrichment": r["enrichment"],
                     "boot_lo": r.get("boot_lo"), "boot_hi": r.get("boot_hi"), "precision": r["precision"]})
        for rv, pr in r.get("per_river", {}).items():
            rows.append({"part": part, "scope": rv, "cue": n, "share_errors": pr["share_errors"], "share_correct": pr["share_correct"]})


def report(tag, res):
    print(f"{tag}: {res['n_windows']} windows, {res['n_errors']} errors (rate {res['error_rate']:.3f}); "
          f"errors without any cue {res['unexplained']['share_of_errors_without_cue']:.3f}", flush=True)
    for n, r in res["cues"].items():
        ci = f" CI [{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]" if "boot_lo" in r else ""
        print(f"  {n:18s} errors {r['share_errors']:.3f} correct {r['share_correct']:.3f} x{r['enrichment']:.2f}{ci} precision {r['precision']:.3f}", flush=True)
    for b, rs in res["review_set"].items():
        print(f"  review {b}: {rs['n_windows']} windows, error rate {rs['error_rate_in_set']:.3f}, any cue {rs['share_with_any_cue']:.3f}; "
              + ", ".join(f"{n} {c['share_in_set']:.2f}" for n, c in rs["cues"].items()), flush=True)


# ----------------------------------------------------------------------------- part A
def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    col = Collector("scene")
    rivers = []
    for name in ctx["names"]:
        try:
            unit = hb.scene_unit(ctx, model, name, args, summary)
            if unit is None:
                continue
            sigs = hb.base_signals_a(unit, ctx)
            valid = np.ones(unit["err"].shape, bool)
            dih = load_dihedral_std(args, f"{name}_dihedral")
            cues, _ = col.add(name, sigs, unit["err"], valid, dih)
            rivers.append(np.full(int(valid.sum()), hb.RIVER.get(name, name)))
            e = unit["err"] > 0.5
            print(f"{name}: {unit['n_err']} errors; " + ", ".join(f"{n} {c[e].mean():.2f}/{c[~e].mean():.2f}" for n, c in cues.items()), flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    if not col.units:
        return
    table = col.table()
    table["river"] = np.concatenate(rivers)
    np.savez_compressed(os.path.join(hb.OUT, f"exp37_patches_scenes{args.suffix}.npz"), **table)
    _, scene_idx = np.unique(table["scene"], return_inverse=True)
    res = col.analysis(scene_idx, rivers=table["river"])
    res["cluster"] = "scene"
    summary["part_a"].update({"n_scenes": len(col.units), "scenes": col.units, "analysis": res})
    csv_rows("A", res, rows)
    report("part A (WorldCover reference)", res)


# ----------------------------------------------------------------------------- part B
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp37")
    if ctx is None:
        return
    N, ok, err = ctx["N"], ctx["ok"], ctx["err"]
    sigs = hb.base_signals_b(ctx)
    dih_all = load_dihedral_std(args, "bolivia_dihedral")
    if dih_all is not None and dih_all.shape[0] != N:
        summary["part_b"]["dihedral_cache_mismatch"] = [int(dih_all.shape[0]), int(N)]
        dih_all = None
    col = Collector("tile")
    for t in range(N):
        valid = ok[t]
        if not valid.any():
            continue
        col.add(t, {k: np.asarray(v[t]) for k, v in sigs.items()}, err[t], valid, None if dih_all is None else dih_all[t])
    table = col.table()
    np.savez_compressed(os.path.join(hb.OUT, f"exp37_patches_bolivia{args.suffix}.npz"), **table)
    res = col.analysis(table["tile"].astype(int))
    res["cluster"] = "tile"
    summary["part_b"].update({"n_tiles_with_valid_patches": len(col.units), "analysis": res})
    csv_rows("B", res, rows)
    report("part B (Bolivia hand labels)", res)


def main():
    args = hb.make_parser(__doc__).parse_args()
    config = {"cues": {"boundary": "indicator > 0", "low_confidence": f"top {FRACTION:.0%} of -|logit| within the unit, ties included",
                       "unstable": f"top {FRACTION:.0%} of aligned tile-phase within the unit", "ndwi_ambiguous": f"|patch-mean NDWI| < {NDWI_TOL}",
                       "dihedral_disagree": f"top {FRACTION:.0%} of exp36's std over 8 transforms within the unit (when cached for every unit)"},
              "budgets": list(BUDGETS), "n_boot": N_BOOT, "bootstrap": "cluster bootstrap of the enrichment ratio (scene in A, tile in B)",
              "review_set": "confidence's k = max(1, round(b n)) most suspect valid windows per unit, ties by descending raster position (assess.review_mask)",
              "prereg": "none; descriptive measurement for the cue library"}
    hb.run("exp37", "exp37 cue enrichment on identical windows", config, part_a, part_b, args, "exp37_cue_enrichment")


if __name__ == "__main__":
    main()
