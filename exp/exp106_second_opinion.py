"""exp106: does disagreement with an independent map product find a product's confident errors?

Preregistration: docs/plan/second_opinion.md (the page is the specification). Numpy and csv only; reads the point
files exp92 and exp94 committed (nothing fetched).

Cells (every row weighted equally):
  1 CGLS-LC100 2017 (cgls_proba) against Esri 2017 at the East Africa TimeSync 2017 plots (exp94's 7-class legend)
  2 ODSE-LULC 2017 (odse_prob) against Esri 2017 at the S2GLC points' random 5,000 (exp94's 8-class legend)
  3 LCMAP 2018 (lcpconf) against Esri 2018 at LCMAP's reference plots' random 5,000 (LCMAP's legend)
  control: LCMAP (lcpconf) against NLCD at NLCD's assessment points, five set-years (report-only; related products)

Scores: u = 101 - confidence; d = 1 where the product and the second product differ; c = d + u / 102. P1: AUROC(c) -
AUROC(u) >= 0.02 on at least 2 of the 3 cells. P2: among rows at or above the median confidence of the cell's correct
rows, the share of errors with d = 1 is at least twice that of correct rows, on all 3 cells (floor: 20 confident errors).
The grade is refused until the page is frozen; --smoke runs on synthetic rows anywhere.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, ROOT)

from oe_inferencex import metrics  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "second_opinion.md")
OUT = os.path.join(EXP_DIR, "out")
EA = os.path.join(OUT, "exp94_eastafrica_points.csv")
EU = os.path.join(OUT, "exp94_europe_points.csv")
NLCD = os.path.join(OUT, "exp94_nlcd_points.csv")
LCMAP = os.path.join(OUT, "exp92_plots.csv")
CELLS = ("east_africa_cgls", "europe_odse", "conus_lcmap")
P1_GAIN, P1_CELLS, P2_RATIO, P2_FLOOR = 0.02, 2, 2.0, 20      # the page's bars
R_BOOT, SEED, LEVEL = 2000, 106, 0.90
EXPECTED_ROWS = {"east_africa_cgls": 12574, "europe_odse": 4807, "conus_lcmap": 4794}   # the page's inventory
EXPECTED_CONTROL = {"nlcd_A2001": 7337, "nlcd_A2006": 7193, "nlcd_A2011": 7201, "nlcd_B2011": 4217, "nlcd_B2016": 4240}
BLIND = ("east_africa_cgls", "europe_odse")       # cell 3 is not blind for P1 (exp92 graded d and u on its rows)


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        return "frozen" if "**Status: frozen" in f.read() else "draft"


def _int(v, missing=0):
    v = (v or "").strip()
    return int(float(v)) if v else missing


def load_cells():
    """The three graded cells as the page defines them: per row the product's class, its confidence, the second
    product's class, the reference class and a group (country or tile) for the bootstrap; plus the control."""
    cells = {}
    rows = [r for r in csv.DictReader(open(EA, encoding="utf-8")) if _int(r["year"]) == 2017]
    ref, prod, conf, sec = (np.array([_int(r[k], m) for r in rows]) for k, m in
                            (("ref", 0), ("cgls", 0), ("cgls_proba", 255), ("esri", 0)))
    grp = np.array([r["country"] for r in rows])
    keep = (ref > 0) & (prod > 0) & (conf <= 100) & (sec > 0)
    cells["east_africa_cgls"] = dict(ref=ref[keep], prod=prod[keep], conf=conf[keep].astype(float), sec=sec[keep],
                                     group=grp[keep], boot="within each country")
    rows = list(csv.DictReader(open(EU, encoding="utf-8")))
    sub = np.array([_int(r["in_subset"]) == 1 for r in rows])
    ref, prod, sec = (np.array([_int(r[k]) for r in rows]) for k in ("ref", "odse", "esri"))
    has = np.array([(r["odse_prob"] or "").strip() != "" for r in rows])
    conf = np.array([float(r["odse_prob"]) if h else -1.0 for r, h in zip(rows, has)])
    grp = np.array([r["tile"] for r in rows])
    keep = sub & (ref > 0) & (prod > 0) & has & (conf >= 0) & (conf <= 100) & (sec > 0)
    cells["europe_odse"] = dict(ref=ref[keep], prod=prod[keep], conf=conf[keep], sec=sec[keep], group=grp[keep],
                                boot="by tile")
    rows = list(csv.DictReader(open(LCMAP, encoding="utf-8")))
    sub = np.array([_int(r["in_conf_subset"]) == 1 for r in rows])
    ref, prod, conf, sec = (np.array([_int(r[k]) for r in rows]) for k in ("ref", "lcmap", "lcpconf", "esri"))
    keep = sub & (ref > 0) & (prod > 0) & (conf >= 1) & (conf <= 100) & (sec > 0)
    cells["conus_lcmap"] = dict(ref=ref[keep], prod=prod[keep], conf=conf[keep].astype(float), sec=sec[keep],
                                group=np.zeros(int(keep.sum()), int), boot="simple")
    return cells


def load_control():
    import exp94_breadth as e94
    rows = list(csv.DictReader(open(NLCD, encoding="utf-8")))
    out = {}
    for s, year in sorted({(r["set"], r["year"]) for r in rows}):
        rs = [r for r in rows if (r["set"], r["year"]) == (s, year)]
        ref8 = np.array([e94.NLCD_TO_LCMAP.get(_int(r["ref_nlcd"]), 0) for r in rs])
        nl8 = np.array([e94.NLCD_TO_LCMAP.get(_int(r["nlcd"]), 0) for r in rs])
        lc, conf = np.array([_int(r["lcmap"]) for r in rs]), np.array([_int(r["lcpconf"]) for r in rs])
        keep = (ref8 > 0) & (lc > 0) & (nl8 > 0) & (conf >= 1) & (conf <= 100)
        out[f"nlcd_{s}{year}"] = dict(ref=ref8[keep], prod=lc[keep], conf=conf[keep].astype(float), sec=nl8[keep],
                                      group=np.zeros(int(keep.sum()), int), boot="simple")
    return out


def scores(cell):
    u = 101.0 - cell["conf"]
    d = (cell["prod"] != cell["sec"]).astype(float)
    return u, d, d + u / 102.0


def auroc(score, err):
    return float(metrics.weighted_auroc(score, err.astype(float), np.ones(err.size)))


def confident(cell, err):
    """The page's confident rows: confidence at or above the median confidence of the cell's correct rows."""
    m = float(np.median(cell["conf"][~err])) if (~err).any() else float("inf")
    return m, cell["conf"] >= m


def flag_shares(cell, err, d):
    m, hi = confident(cell, err)
    ce, cok, fh = hi & err, hi & ~err, hi & (d == 1)
    f_err = float(d[ce].mean()) if ce.any() else float("nan")
    f_ok = float(d[cok].mean()) if cok.any() else float("nan")
    ratio = (f_err / f_ok) if f_ok > 0 else (float("inf") if f_err > 0 else float("nan"))
    return {"median_correct_confidence": m, "confident_errors": int(ce.sum()), "confident_correct": int(cok.sum()),
            "f_err": f_err, "f_ok": f_ok, "ratio": ratio, "recall_of_d_for_confident_errors": f_err,
            "precision_of_d_among_confident": float(err[fh].mean()) if fh.any() else float("nan")}


def decomposition(u, d, err):
    """Report-only. AUROC(c) - AUROC(u) = r(1 - q)(1 - A10) - (1 - r) q A01, with r and q the flag's shares among errors
    and correct rows, A10 u's AUROC for disputed errors against undisputed correct rows and A01 for undisputed errors
    against disputed correct rows; and the part of the gain from error/correct pairs that u ties (where c decides by d
    alone): sum over tied values of n_err(v) n_ok(v) (r(v) - q(v)) / 2, over n_err n_ok."""
    r, q = float(d[err].mean()), float(d[~err].mean())

    def a(e, k):
        sel = e | k
        return auroc(u[sel], e[sel]) if e.any() and k.any() else float("nan")

    A10, A01 = a(err & (d == 1), ~err & (d == 0)), a(err & (d == 0), ~err & (d == 1))
    tied = 0.0
    for v in np.unique(u[err]):
        e_v, k_v = err & (u == v), ~err & (u == v)
        if k_v.any():
            tied += e_v.sum() * k_v.sum() * (d[e_v].mean() - d[k_v].mean()) / 2
    tied /= float(err.sum() * (~err).sum())
    return {"r": r, "q": q, "A10": A10, "A01": A01, "identity_gain": r * (1 - q) * (1 - A10) - (1 - r) * q * A01,
            "gain_from_pairs_u_ties": tied}


def stats(cell):
    err = cell["prod"] != cell["ref"]
    u, d, c = scores(cell)
    fs = flag_shares(cell, err, d)
    cap = {k: float(metrics.capture_at_budget_expected(s, err.astype(int), (0.10,))[0.10]) for k, s in
           (("own", u), ("disagreement", d), ("combined", c))}
    prec = float(err[d == 1].mean()) if (d == 1).any() else float("nan")
    rec = float(d[err].mean()) if err.any() else float("nan")
    return {"n": int(err.size), "errors": int(err.sum()), "error_rate": float(err.mean()),
            "disagreement_share": float(d.mean()), "auroc_own": auroc(u, err), "auroc_disagreement": auroc(d, err),
            "auroc_combined": auroc(c, err), "gain": auroc(c, err) - auroc(u, err), "capture_10pct": cap,
            "precision_of_d_for_errors": prec, "recall_of_d_for_errors": rec, "confident": fs,
            "decomposition": decomposition(u, d, err)}


def boot_index(group, rng, how):
    """One bootstrap resample by the page's scheme for the cell: simple, within each country, or by tile."""
    if how == "simple":
        return rng.integers(0, group.size, group.size)
    labels = np.unique(group)
    by = {g: np.flatnonzero(group == g) for g in labels}
    if how == "within each country":
        return np.concatenate([rng.choice(by[g], by[g].size, replace=True) for g in labels])
    if how == "by tile":
        return np.concatenate([by[g] for g in rng.choice(labels, labels.size, replace=True)])
    raise ValueError(f"unknown bootstrap scheme {how!r}")


def bootstrap(cell, rng, R=R_BOOT):
    gains, ratios = [], []
    for _ in range(R):
        i = boot_index(cell["group"], rng, cell["boot"])
        sub = {k: (v[i] if isinstance(v, np.ndarray) else v) for k, v in cell.items()}
        err = sub["prod"] != sub["ref"]
        if not (0 < err.sum() < err.size):
            continue
        u, d, c = scores(sub)
        gains.append(auroc(c, err) - auroc(u, err))
        ratios.append(flag_shares(sub, err, d)["ratio"])
    q = ((1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2)
    fin = [r for r in ratios if np.isfinite(r)]
    return {"resamples": len(gains), "ratio_resamples": len(fin), "resampling": cell["boot"],
            "gain_interval": [float(np.quantile(gains, q[0])), float(np.quantile(gains, q[1]))],
            "ratio_interval": [float(np.quantile(fin, q[0])), float(np.quantile(fin, q[1]))] if fin else None}


def verdicts(rows):
    p1_cells = [c for c in CELLS if rows[c]["gain"] >= P1_GAIN]
    p2 = {}
    for c in CELLS:
        fs = rows[c]["confident"]
        p2[c] = (None if fs["confident_errors"] < P2_FLOOR
                 else bool(fs["f_err"] > 0 and fs["f_err"] >= P2_RATIO * fs["f_ok"]))
    return {"P1": {"bar": f"AUROC(c) - AUROC(u) >= {P1_GAIN} on at least {P1_CELLS} of 3",
                   "cells_meeting": p1_cells, "holds": len(p1_cells) >= P1_CELLS,
                   "blind_cells_meeting": [c for c in p1_cells if c in BLIND],
                   "reading": "cell 3 is not blind for P1 (exp92 graded d and u on its rows); reported, not a bar"},
            "P2": {"bar": f"f_err >= {P2_RATIO} f_ok on all 3 (floor {P2_FLOOR} confident errors)",
                   "by_cell": p2, "holds": all(v is True for v in p2.values()),
                   "determined": all(v is not None for v in p2.values())}}


def run(cells, control=None, R=R_BOOT, seed=SEED, log=print):
    rows = {}
    for name, cell in cells.items():
        rng = np.random.default_rng(seed)
        rows[name] = stats(cell)
        rows[name]["bootstrap"] = bootstrap(cell, rng, R)
        if name == "east_africa_cgls":
            rows[name]["by_country"] = {g: {k: v for k, v in stats({k2: (v2[cell["group"] == g] if isinstance(v2, np.ndarray)
                                                                        else v2) for k2, v2 in cell.items()}).items()
                                            if k in ("n", "errors", "auroc_own", "auroc_combined", "gain")}
                                        for g in sorted(np.unique(cell["group"]))}
        r = rows[name]
        log(f"[exp106] {name:18s} n {r['n']:6d} err {r['error_rate']:.3f} d {r['disagreement_share']:.3f} "
            f"AUROC own {r['auroc_own']:.3f} d {r['auroc_disagreement']:.3f} comb {r['auroc_combined']:.3f} "
            f"gain {r['gain']:+.3f} f_err/f_ok {r['confident']['f_err']:.3f}/{r['confident']['f_ok']:.3f}")
    ctrl = {k: stats(v) for k, v in (control or {}).items()}
    return rows, ctrl


def synthetic_cell(seed=0, n=3000, informative=True, boot="within each country"):
    rng = np.random.default_rng(seed)
    ref = rng.integers(1, 5, n)
    err = rng.random(n) < 0.25
    prod = np.where(err, (ref % 4) + 1, ref)
    conf = np.clip(np.where(err, rng.normal(55, 20, n), rng.normal(75, 15, n)), 1, 100).round()
    agree_ok = rng.random(n) < 0.8                           # the second product agrees with the product when right
    sec = np.where(err, np.where(rng.random(n) < (0.7 if informative else 0.2), ref, prod), np.where(agree_ok, prod, ((prod) % 4) + 1))
    group = {"within each country": rng.integers(0, 7, n).astype(str), "by tile": rng.integers(0, 55, n).astype(str),
             "simple": np.zeros(n, int)}[boot]
    return dict(ref=ref, prod=prod, conf=conf, sec=sec, group=group, boot=boot)


def _dump(rec, path):
    import exp78_error_rate_estimation as e78
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e78.jsonable(rec), f, indent=1, allow_nan=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--draws", type=int, default=R_BOOT)
    a = ap.parse_args(argv)
    if a.smoke:
        cells = {c: synthetic_cell(seed=i, informative=i != 2, boot=b) for i, (c, b) in
                 enumerate(zip(CELLS, ("within each country", "by tile", "simple")))}
        rows, _ = run(cells, R=min(a.draws, 50))
        rec = {"experiment": "exp106 smoke (synthetic rows)", "cells": rows, "verdicts": verdicts(rows)}
        os.makedirs(a.out_dir, exist_ok=True)
        _dump(rec, os.path.join(a.out_dir, "exp106_summary_smoke.json"))
        print(json.dumps(rec["verdicts"]))
        return rec
    status = prereg_status()
    if status != "frozen":
        raise SystemExit(f"{os.path.relpath(PLAN, ROOT)} is {status}: nothing using the labels is graded before it is frozen")
    import exp89_finetuned_checkpoints as e89
    commit, dirty = e89.git_state()
    cells = load_cells()
    counts = {k: int(v["ref"].size) for k, v in cells.items()}
    if counts != EXPECTED_ROWS:
        raise SystemExit(f"the cells' rows {counts} are not the page's inventory {EXPECTED_ROWS}")
    control = load_control()
    ccounts = {k: int(v["ref"].size) for k, v in control.items()}
    if ccounts != EXPECTED_CONTROL:
        raise SystemExit(f"the control's rows {ccounts} are not the page's {EXPECTED_CONTROL}")
    rows, ctrl = run(cells, control, R=a.draws)
    rec = {"experiment": "exp106: does disagreement with an independent product find a product's confident errors?",
           "prereg_page": os.path.relpath(PLAN, ROOT), "prereg_status": status, "commit": commit, "dirty": dirty,
           "config": {"bootstrap": {"R": a.draws, "seed": SEED, "level": LEVEL}, "scores": "u = 101 - confidence; "
                      "d = product != second product; c = d + u / 102", "weights": "every row equal"},
           "cells": rows, "control_related_products": ctrl, "verdicts": verdicts(rows)}
    os.makedirs(a.out_dir, exist_ok=True)
    _dump(rec, os.path.join(a.out_dir, "exp106_summary.json"))
    print(json.dumps(rec["verdicts"]))
    return rec


if __name__ == "__main__":
    main()
