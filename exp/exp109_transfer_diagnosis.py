"""exp109, report-only (not preregistered; exploratory, on plots already examined): why the AWF fine-tunes' confidence
misses the 0.70 ranking bar under transfer (exp99, v1: 0.638; exp103, v1.2: 0.642, STRICT, at the 271 East Africa
TimeSync 2017 plots with input).

The diagnosis was done on 12 and 13 October 2026, after exp99 and exp103 were recorded, on the same plots and labels;
this script records the part of it that the committed units files can reproduce. Numpy only; it reads
exp/out/exp99_units.npz and exp/out/exp103/transfer2017/exp99_units.npz (the 309 plots; the 271 with input are graded)
and exp/out/exp98_units.npz and exp/out/exp103/awf2023/exp98_units.npz (Ai2's 259 covered validation points), and
writes aggregates only (no identifier, no position) to exp/out/exp109_summary.json.

  confident third   the plots whose top-1 probability p1 is at or above the 2/3 quantile of p1 over the 271 plots,
                    an unweighted count (91 plots); likewise over Ai2's 259 validation points (87)
  STRICT-only       wrong under STRICT and right under LENIENT (exp99's crosswalk): a disagreement the lenient reading
                    of the two legends removes
  AUROCs            design-weighted (exp99's frozen weights) for the suspicion -log p1: STRICT and LENIENT errors
                    against the rest; LENIENT errors, and STRICT-only errors, each against the STRICT-correct plots;
                    90% intervals by resampling plots within each country, 2,000 resamples, seed 0 (the diagnosis's
                    seed, kept so the audited numbers reproduce)
  in distribution   at Ai2's validation points (unweighted, as exp98 graded them): the AUROC over all points and on the
                    grass/shrub boundary (points the map calls and Ai2 labels shrubland_savanna or grassland_barren),
                    and the direction of the errors there
  losses            (Ai2's points' AUROC) - (LENIENT errors against STRICT-correct plots), read as transfer and point
                    placement; (LENIENT errors against STRICT-correct plots) - (STRICT AUROC), read as the legend's.
                    A descriptive split of two populations (unweighted expert-placed points against a design-weighted
                    random sample), not an identified decomposition
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, ROOT)

from oe_inferencex import metrics  # noqa: E402
import timesync_awf_crosswalk as tsx  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
TRANSFER = {"v1": "exp/out/exp99_units.npz", "v1.2": "exp/out/exp103/transfer2017/exp99_units.npz"}
AI2_POINTS = {"v1": "exp/out/exp98_units.npz", "v1.2": "exp/out/exp103/awf2023/exp98_units.npz"}
SHRUB, GRASS = tsx.AWF_INDEX["shrubland_savanna"], tsx.AWF_INDEX["grassland_barren"]
R_BOOT, SEED, LEVEL = 2000, 0, 0.90
CUTS = (0.5, 0.6, 2 / 3, 0.75, 0.8, 0.9)               # report-only: the confident share at other cuts


def sha256(rel):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, rel), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(rel):
    d = np.load(os.path.join(ROOT, rel), allow_pickle=False)
    return {k: d[k] for k in d.files}


def confident_third(p1):
    """The confident third: p1 at or above the 2/3 quantile of p1 over the set (unweighted)."""
    cut = float(np.quantile(p1, 2 / 3))
    return p1 >= cut, cut


def wauroc(u, e, w):
    return float(metrics.weighted_auroc(u, np.asarray(e, float), w))


def wmean(x, w):
    return float((np.asarray(x, float) * w).sum() / w.sum())


def pairs(pred, ref, mask, ref_names, top=6):
    """Counts of (map class -> reference class) over the masked units, commonest first."""
    c = Counter(f"{tsx.AWF_CLASSES.get(int(p), f'channel {int(p)}')} -> {ref_names[int(r)]}"
                for p, r, m in zip(pred, ref, mask) if m)
    return [{"pair": k, "n": v} for k, v in c.most_common(top)]


def transfer_units(T):
    """The graded plots of both models, checked to be the same plots in the same order, with errors recomputed from
    the crosswalk."""
    a, b = T["v1"], T["v1.2"]
    for k in ("label", "label_names", "country", "country_names", "covered", "weight", "inside_awf"):
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"the two transfer units files differ in {k!r}")
    K = a["covered"].astype(bool)
    names = [str(x) for x in a["label_names"]]
    base = {"label": a["label"][K], "names": names, "weight": a["weight"][K].astype(float),
            "country": a["country"][K]}
    out = {}
    for m, d in T.items():
        pred = d["pred"][K]
        es, el = d["wrong_strict"][K] > 0, d["wrong_lenient"][K] > 0
        labs = [names[i] for i in base["label"]]
        if not (np.array_equal(es, tsx.wrong(labs, pred, "strict") > 0)
                and np.array_equal(el, tsx.wrong(labs, pred, "lenient") > 0)):
            raise SystemExit(f"{m}: the units file's errors are not the crosswalk's")
        out[m] = dict(base, pred=pred, p1=d["p1"][K].astype(float), u=d["suspicion"][K].astype(float), es=es, el=el)
    return out


def ai2_points(D):
    a, b = D["v1"], D["v1.2"]
    for k in ("names", "label", "split", "covered"):
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"the two Ai2-point units files differ in {k!r}")
    out = {}
    for m, d in D.items():
        k = d["covered"].astype(bool) & (d["split"] == "val")
        out[m] = {"pred": d["pred_deployed"][k], "label": d["label"][k], "p1": d["p1"][k].astype(float),
                  "u": d["suspicion"][k].astype(float)}
    return out


def auroc_set(t, idx=None):
    """The four design-weighted AUROCs on the plots idx (all when None)."""
    i = np.arange(t["u"].size) if idx is None else idx
    u, w, s, l = t["u"][i], t["weight"][i], t["es"][i], t["el"][i]
    a, b = l | ~s, ~l                                   # LENIENT errors or STRICT-correct; not a LENIENT error
    return {"strict": wauroc(u, s, w), "lenient": wauroc(u, l, w),
            "lenient_errors_vs_strict_correct": wauroc(u[a], l[a], w[a]),
            "strict_only_errors_vs_strict_correct": wauroc(u[b], s[b], w[b])}


def bootstrap(t, R=R_BOOT, seed=SEED):
    """Resample plots within each country; the 90% interval of each AUROC and of the LENIENT-minus-STRICT-only gap."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(t["country"] == g) for g in np.unique(t["country"])]
    keys = ("lenient_errors_vs_strict_correct", "strict_only_errors_vs_strict_correct", "strict", "lenient")
    draws = []
    for _ in range(R):
        ix = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        a = auroc_set(t, ix)
        draws.append([a[k] for k in keys])
    draws = np.array(draws, float)
    q = ((1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2)
    out = {k: [float(np.nanquantile(draws[:, j], q[0])), float(np.nanquantile(draws[:, j], q[1]))]
           for j, k in enumerate(keys)}
    diff = draws[:, 0] - draws[:, 1]
    out["lenient_minus_strict_only"] = [float(np.nanquantile(diff, q[0])), float(np.nanquantile(diff, q[1]))]
    out["resamples"] = int(R)
    return out


def transfer_block(t, R=R_BOOT):
    names, lab, w = t["names"], t["label"], t["weight"]
    es, el, pred = t["es"], t["el"], t["pred"]
    so = es & ~el
    conf, cut = confident_third(t["p1"])
    og, wg = lab == names.index("Open Grassland"), lab == names.index("Wooded Grassland")
    shrub_on_og, grass_on_wg = so & og & (pred == SHRUB), so & wg & (pred == GRASS)
    point = auroc_set(t)
    point["lenient_minus_strict_only"] = (point["lenient_errors_vs_strict_correct"]
                                          - point["strict_only_errors_vs_strict_correct"])
    by_cut = {}
    for q in CUTS:
        c = t["p1"] >= np.quantile(t["p1"], q)
        by_cut[f"top_{1 - q:.2f}"] = {"plots": int(c.sum()), "strict_errors": int((c & es).sum()),
                                      "strict_only": int((c & so).sum())}
    return {
        "n": int(es.size), "strict_errors": int(es.sum()), "lenient_errors": int(el.sum()), "strict_only": int(so.sum()),
        "strict_only_pairs": pairs(pred, lab, so, names),
        "grass_shrub_direction": {"shrubland_savanna_on_open_grassland": int(shrub_on_og.sum()),
                                  "grassland_barren_on_wooded_grassland": int(grass_on_wg.sum())},
        "confident_third": {
            "definition": "p1 at or above the 2/3 quantile of p1 over the plots with input (unweighted count)",
            "cut_p1": cut, "plots": int(conf.sum()),
            "weighted_error_rate": {"strict": wmean(es[conf], w[conf]), "lenient": wmean(el[conf], w[conf])},
            "strict_errors": int((conf & es).sum()), "lenient_errors": int((conf & el).sum()),
            "strict_only": int((conf & so).sum()),
            "share_of_confident_strict_errors_strict_only": float((conf & so).sum() / (conf & es).sum()),
            "shrubland_savanna_on_open_grassland": int((conf & shrub_on_og).sum()),
            "grassland_barren_on_wooded_grassland": int((conf & grass_on_wg).sum()),
            "pairs_strict_errors": pairs(pred, lab, conf & es, names)},
        "strict_only_by_cut": by_cut,
        "auroc_weighted": point, "bootstrap": bootstrap(t, R)}


def ai2_block(v, transfer_cut):
    e = v["pred"] != v["label"]
    conf, cut = confident_third(v["p1"])
    gs = np.isin(v["pred"], [SHRUB, GRASS]) & np.isin(v["label"], [SHRUB, GRASS])
    ones = np.ones(e.size)
    return {
        "n": int(e.size), "errors": int(e.sum()), "auroc_unweighted": wauroc(v["u"], e, ones),
        "confident_third": {"cut_p1": cut, "points": int(conf.sum()), "errors": int((conf & e).sum())},
        "grass_shrub": {
            "points": int(gs.sum()), "errors": int((gs & e).sum()),
            "auroc_unweighted": wauroc(v["u"][gs], e[gs], ones[gs]),
            "shrubland_savanna_where_ai2_labels_grassland_barren": int((gs & e & (v["pred"] == SHRUB)).sum()),
            "grassland_barren_where_ai2_labels_shrubland_savanna": int((gs & e & (v["pred"] == GRASS)).sum()),
            "median_p1_errors": float(np.median(v["p1"][gs & e])),
            "median_p1_correct": float(np.median(v["p1"][gs & ~e])),
            "errors_at_or_above_the_transfer_confident_cut": int((gs & e & (v["p1"] >= transfer_cut)).sum())}}


def run(R=R_BOOT, log=print):
    T = transfer_units({m: load(p) for m, p in TRANSFER.items()})
    V = ai2_points({m: load(p) for m, p in AI2_POINTS.items()})
    models = {}
    for m in TRANSFER:
        tr = transfer_block(T[m], R)
        ai = ai2_block(V[m], tr["confident_third"]["cut_p1"])
        a = tr["auroc_weighted"]
        losses = {"transfer_and_point_placement": ai["auroc_unweighted"] - a["lenient_errors_vs_strict_correct"],
                  "legend": a["lenient_errors_vs_strict_correct"] - a["strict"]}
        models[m] = {"transfer_2017": tr, "ai2_validation_points": ai, "losses": losses}
        c = tr["confident_third"]
        log(f"[exp109] {m:4s} confident third: STRICT {c['weighted_error_rate']['strict']:.3f} LENIENT "
            f"{c['weighted_error_rate']['lenient']:.3f}; STRICT-only {c['strict_only']}/{c['strict_errors']}, "
            f"shrub on OG {c['shrubland_savanna_on_open_grassland']}; AUROC STRICT-only "
            f"{a['strict_only_errors_vs_strict_correct']:.3f} LENIENT {a['lenient_errors_vs_strict_correct']:.3f}; "
            f"Ai2 grass/shrub {ai['grass_shrub']['auroc_unweighted']:.3f}; losses "
            f"{losses['transfer_and_point_placement']:.3f} / {losses['legend']:.3f}")
    return models


def _dump(rec, path):
    import exp78_error_rate_estimation as e78
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e78.jsonable(rec), f, indent=1, allow_nan=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--draws", type=int, default=R_BOOT)
    a = ap.parse_args(argv)
    models = run(R=a.draws)
    rec = {"experiment": "exp109, report-only: why the AWF fine-tunes' confidence misses the ranking bar under transfer",
           "preregistered": False,
           "reading": "exploratory, on plots and labels already examined (exp99, exp101, exp103); descriptive",
           "inputs": {p: sha256(p) for p in (*TRANSFER.values(), *AI2_POINTS.values())},
           "config": {"bootstrap": {"R": a.draws, "seed": SEED, "level": LEVEL, "resampling": "plots within each country"},
                      "weights": "exp99's frozen design weights at the TimeSync plots; Ai2's points unweighted",
                      "suspicion": "-log p1 (the units files' suspicion)",
                      "crosswalk_sha256": tsx.crosswalk_record()["sha256"]},
           "models": models}
    os.makedirs(a.out_dir, exist_ok=True)
    _dump(rec, os.path.join(a.out_dir, "exp109_summary.json"))
    return rec


if __name__ == "__main__":
    main()
