#!/usr/bin/env python
"""exp112: a map made from OlmoEarth v1.3 embeddings, graded at LCMAP's probability sample (docs/plan/v13_embeddings.md;
the page is the specification).

Modes:
    draw      write the plots of S (seed 112) and the fit order (seed 113) to the private directory; no network
    extract   read the 2018 embedding at each plot of S from Ai2's store; resumable, one checkpoint line per plot
    analyze   fit, predict and grade (refused unless the page is frozen); writes exp/out/exp112_summary.json and
              exp/out/exp112_plots.csv
    smoke     extract a handful of plots of S into a separate checkpoint and print their norms; nothing is graded

Positions (plot coordinates, rows and columns) and the embeddings stay in --private-dir, never in the repository.

    uv run --with "zarr>=3.1,<4" --with obstore --with pyproj --with scikit-learn python exp/exp112_v13_lcmap.py \
        extract --private-dir DIR --workers 16
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from oe_inferencex import estimate as est, metrics, plan  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "v13_embeddings.md")
REF = os.path.join(ROOT, "data", "lcmap", "LCMAP_CU_20211117_V01_REF", "LCMAP_CU_20211117_V01_REF.csv")
PRJ = os.path.join(ROOT, "data", "lcmap", "LCMAP_CU_20211117_V01_REF", "LCMAP_CU_20200414_V01_REF.prj")
REF_SHA256_PREFIX = "eb935d96"                        # of the zip, as exp92 records; the csv's own sha is recorded
PLOTS92 = os.path.join(EXP_DIR, "out", "exp92_plots.csv")
OUT = os.path.join(EXP_DIR, "out")
STORE = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
ENDPOINT = "https://data.source.coop"
CLASSES = {"Developed": 1, "Cropland": 2, "Grass/Shrub": 3, "Tree Cover": 4, "Water": 5, "Wetland": 6, "Snow/Ice": 7,
           "Barren": 8}                               # exp92's
YEAR = 2018
N_S, SEED_S, SEED_F = 12000, 112, 113
FIT, FIT_REPORTED = 1000, (300, 3000)
DIMS, DIMS_REPORTED = 128, 64
DRAWS, SEED, BOOT, VERIFY = 2000, 112, 2000, 20
BUDGETS, CERTIFY_BUDGET = (100, 300), 300
ALPHA, ALPHA_REPORTED, DELTA = 0.05, 0.10, 0.10
CAPTURE_BUDGET = 0.10
# bars (exp110's)
P1_AUROC, P1_CAPTURE, P1_LEAD, P1_ERROR_FLOOR = 0.70, 0.25, 0.05, 200
P2_COVERAGE, P2_MC_SE = 0.95, 3.0
P3_VIOLATION, P3_SHARE = 0.115, 0.25
P4_WRONG = 0.025
MAX_CONSECUTIVE_FAILURES = 20


def log(*a):
    print(*a, flush=True)


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        return "frozen" if "**Status: frozen" in f.read() else "draft"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- the plots
def reference():
    """{plot id: LCMAP class} for 2018, classes 1-8 only."""
    out = {}
    with open(REF, newline="") as f:
        for r in csv.DictReader(f):
            if r["image_year"] == str(YEAR) and r["LCMAP"] in CLASSES:
                out[int(r["plotid"])] = CLASSES[r["LCMAP"]]
    return out


def positions():
    """{plot id: (x, y)} in LCMAP's Albers grid (private: never written to the repository)."""
    out = {}
    with open(REF, newline="") as f:
        for r in csv.DictReader(f):
            if r["image_year"] == str(YEAR):
                out[int(r["plotid"])] = (float(r["x"]), float(r["y"]))
    return out


def draw(ref):
    """S (seed 112, from the sorted ids with a 2018 class) and the fit order (a permutation of S, seed 113)."""
    ids = np.array(sorted(ref), dtype=np.int64)
    s = np.sort(np.random.default_rng(SEED_S).choice(ids, N_S, replace=False))
    order = s[np.random.default_rng(SEED_F).permutation(s.size)]
    return s, order


# ----------------------------------------------------------------------------- reading the store
def open_store():
    import zarr
    from obstore.store import S3Store
    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store.from_url(STORE, endpoint=ENDPOINT, skip_signature=True, client_options={"timeout": "120s"})
    return zarr.open_group(zarr.storage.ObjectStore(s3, read_only=True), mode="r")


def dequantize(q, quant):
    """The store's own formula: (|q| / 127.5) ** power * sign(q), NaN at no data."""
    if quant.get("method") != "signed_power":
        raise SystemExit(f"exp112: quantization {quant.get('method')!r}, not signed_power")
    q = np.asarray(q)
    v = q.astype(np.float64)
    v[q == quant["nodata"]] = np.nan
    return (np.abs(v) / (float(quant["valid_range"][1]) + 0.5)) ** float(quant["power"]) * np.sign(v)


def locate(pos, plot_ids):
    """{plot id: (zone, row, col)} of the 10 m pixel holding each plot's centre."""
    import pyproj
    wkt = open(PRJ).read()
    to_ll = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)
    xs, ys = zip(*(pos[p] for p in plot_ids))
    lon, lat = to_ll.transform(xs, ys)
    out = {}
    by_zone = {}
    for p, lo, la in zip(plot_ids, lon, lat):
        by_zone.setdefault(int((lo + 180) // 6) % 60 + 1, []).append((p, lo, la))
    for z, rows in by_zone.items():
        tr = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:326{z:02d}", always_xy=True)
        e, n = tr.transform([r[1] for r in rows], [r[2] for r in rows])
        for (p, _, _), ee, nn in zip(rows, e, n):
            out[p] = (z, ee, nn)
    return out


def extract(priv, plot_ids, checkpoint, workers):
    """Read each plot's 2018 vector; append {plotid, zone, q} lines to the checkpoint (positions stay private)."""
    done = set()
    if os.path.exists(checkpoint):
        with open(checkpoint) as f:
            done = {json.loads(line)["plotid"] for line in f if line.strip()}
    todo = [int(p) for p in plot_ids if int(p) not in done]
    log(f"[exp112] extract: {len(done)} plots done, {len(todo)} to read, {workers} workers")
    if not todo:
        return
    pos = positions()
    loc = locate(pos, todo)
    root = open_store()
    groups = {}

    def group(z):
        if z not in groups:
            g = root[f"utm{z:02d}"]
            a, b, c, d, e, f = (float(t) for t in g.attrs["spatial:transform"])
            if b or d:
                raise SystemExit("exp112: a rotated grid")
            years = [int(t) for t in g["time"][:]]
            groups[z] = (g["embeddings"], years.index(YEAR), a, c, e, f, g.attrs["geoemb:quantization"])
        return groups[z]

    for z in sorted({loc[p][0] for p in todo}):                # open each zone once, before the threads
        group(z)

    def read(p):
        z, x, y = loc[p]
        arr, t, a, c, e, f, _ = groups[z]
        col, row = int(math.floor((x - c) / a)), int(math.floor((y - f) / e))
        return p, z, arr[t, :, row, col]

    t0, n_ok, fails = time.time(), 0, 0
    with open(checkpoint, "a") as out, ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(read, p): p for p in todo}
        for fu in as_completed(futs):
            try:
                p, z, q = fu.result()
            except Exception as ex:                             # noqa: BLE001 - counted, and a run of them stops the job
                fails += 1
                log(f"[exp112] read failed ({type(ex).__name__}: {str(ex)[:160]})")
                if fails >= MAX_CONSECUTIVE_FAILURES:
                    raise SystemExit(f"exp112: {fails} reads failed in a row; stopping (the checkpoint keeps the rest)")
                continue
            fails = 0
            out.write(json.dumps({"plotid": int(p), "zone": int(z), "q": [int(v) for v in np.asarray(q).ravel()]}) + "\n")
            out.flush()
            n_ok += 1
            if n_ok % 250 == 0:
                rate = n_ok / (time.time() - t0)
                log(f"[exp112] {n_ok} of {len(todo)} read, {rate:.1f} plots/s, about "
                    f"{(len(todo) - n_ok) / max(rate, 1e-9) / 60:.0f} min left")
    log(f"[exp112] extract: {n_ok} plots read in {time.time() - t0:.0f} s")


def load(checkpoint, quant):
    """{plot id: float vector (128,), NaN where the store has no data}."""
    out = {}
    with open(checkpoint) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["plotid"]] = dequantize(np.array(r["q"], dtype=np.int16), quant)
    return out


# ----------------------------------------------------------------------------- grading
def auroc(u, e):
    return float(metrics.weighted_auroc(np.asarray(u, float), np.asarray(e, bool), np.ones(len(e))))


def rank_stats(u, e):
    e = np.asarray(e, bool)
    return {"auroc": auroc(u, e),
            "capture_10": float(metrics.capture_at_budget_expected(np.asarray(u, float), e, (CAPTURE_BUDGET,))[CAPTURE_BUDGET]),
            "excess_aurc": float(metrics.excess_aurc(np.asarray(u, float), e.astype(float)))}


def rarity(pred):
    out = np.empty(pred.shape)
    for k in np.unique(pred):
        out[pred == k] = -float((pred == k).mean())
    return out


def fit_predict(X_fit, y_fit, X):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(X_fit, y_fit)
    p = clf.predict_proba(X)
    return clf.classes_[p.argmax(1)], p.max(1)


def arm1(conf, err, pred, dist, rng):
    u = -conf
    own = rank_stats(u, err)
    ctl = {"rarity": rank_stats(rarity(pred), err), "distance": rank_stats(dist, err)}
    best = max(ctl, key=lambda k: ctl[k]["auroc"])
    boot_a, boot_l = [], []
    n = err.size
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        if 0 < err[i].sum() < n:
            a = auroc(u[i], err[i])
            boot_a.append(a)
            boot_l.append(a - auroc((rarity(pred[i]) if best == "rarity" else dist[i]), err[i]))
    q = lambda v: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]       # noqa: E731
    correct = ~err
    conf_err = float((conf[err] >= np.median(conf[correct])).mean()) if correct.any() and err.any() else None
    return {"own": own, "controls": ctl, "best_control": best, "lead": own["auroc"] - ctl[best]["auroc"],
            "boot90": {"auroc": q(boot_a), "lead": q(boot_l)}, "errors": int(err.sum()), "n": int(n),
            "confident_error_share": conf_err,
            "ceiling_10": float(metrics.attainable_ceiling(CAPTURE_BUDGET, float(err.mean()), int(n)))}


def zone_setup(margin, err):
    margin = np.asarray(margin, np.float64)
    N = margin.size
    order, pos = est.zone_order(margin)
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(err, np.float64)[order])])
    o = np.lexsort((np.random.default_rng(0).random(N), -margin))
    ranks = np.empty(N)
    ranks[o] = np.arange(N)
    return {"N": N, "pos": pos, "cum": cum, "pct": (ranks + 0.5) / N}


def certify_steps(Z, idx, wrong, alpha, delta=DELTA):
    """exp110's certify_steps: certify_zone's prefix rule from Z, its review-set guard included."""
    idx = np.asarray(idx, int)
    wrong = np.asarray(wrong, np.float64)
    n = idx.size
    if float(np.mean(Z["pct"][idx])) > est.review_set_threshold(n, Z["N"]) + 1e-12:
        return {"coverage": None, "refused": True}
    cov, sizes, _ = est.zone_levels(Z["N"], n, alpha, delta, est.ZONE_GRID, None)
    if not cov:
        return {"coverage": None}
    b, k = est.zone_counts(Z["pos"][idx], wrong, sizes)
    p = np.array([est.zone_pvalue(kk, bb, nn, alpha) for kk, bb, nn in zip(k, b, sizes)])
    _, best = est.apply_zone_rule(p, b, k, alpha, delta, "prefix", n=sizes)
    if best is None:
        return {"coverage": None}
    nz = int(sizes[best])
    return {"coverage": float(cov[best]), "n_zone": nz, "zone_error": float(Z["cum"][nz] / nz)}


def arms23(conf, err):
    margin = np.asarray(conf, np.float64)
    w = np.asarray(err, np.float64)
    N, K = margin.size, int(w.sum())
    theta = K / N
    Z = zone_setup(margin, w)
    cov = {B: np.zeros(DRAWS, bool) for B in BUDGETS}
    width = {B: np.zeros(DRAWS) for B in BUDGETS}
    zones = {a: [] for a in (ALPHA, ALPHA_REPORTED)}
    checked = 0
    for r in range(DRAWS):
        for B in BUDGETS:
            s = est.sample_for_estimation(margin, B, design="random", seed=SEED + r)
            idx = s["indices"]
            res = est.estimate_error_rate(s, w[idx])
            cov[B][r] = res["low"] <= theta <= res["high"]
            width[B][r] = res["high"] - res["low"]
            if B == CERTIFY_BUDGET:
                for a in zones:
                    z = certify_steps(Z, idx, w[idx], a)
                    if r < VERIFY and a == ALPHA:
                        try:
                            ref = est.certify_zone(margin, idx, w[idx], a, DELTA)
                            got = (ref["coverage"], ref["n_zone"])
                        except ValueError:
                            got = ("refused", None)
                        mine = ("refused", None) if z.get("refused") else (z["coverage"], z.get("n_zone"))
                        if got != mine:
                            raise SystemExit(f"exp112: certify steps {mine} differ from certify_zone {got} at draw {r}")
                        checked += 1
                    zones[a].append(z)
    arm2 = {"N": N, "errors": K, "theta": theta, "cells": {}}
    for B in BUDGETS:
        exact = est.exact_coverage_srs(N, K, B, interval=est.hypergeom_interval)
        mc = float(cov[B].mean())
        se = float(np.sqrt(max(exact * (1 - exact), 1e-12) / DRAWS))
        arm2["cells"][str(B)] = {"exact_coverage": exact, "mc_coverage": mc, "mc_se": se,
                                 "mc_minus_exact_in_se": (mc - exact) / se, "median_width": float(np.median(width[B]))}
    arm3 = {"budget": CERTIFY_BUDGET, "delta": DELTA, "rule": "prefix", "certify_zone_checked_on_draws": checked,
            "by_alpha": {}}
    for a, zs in zones.items():
        c = np.array([z["coverage"] if z["coverage"] is not None else 0.0 for z in zs])
        viol = np.array([z["coverage"] is not None and z["zone_error"] > a for z in zs])
        zerr = [z["zone_error"] for z in zs if z["coverage"] is not None]
        arm3["by_alpha"][f"{a:g}"] = {
            "p_certify_any": float(np.mean([z["coverage"] is not None for z in zs])),
            "refused_as_review_set": int(sum(bool(z.get("refused")) for z in zs)),
            "violation_share": float(viol.mean()), "median_certified_share": float(np.median(c)),
            "certified_share_quantiles": {q: float(np.quantile(c, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "zone_true_error_median": float(np.median(zerr)) if zerr else None}
    return arm2, arm3


def arm4(pred, ref, other, name):
    """The v1.3 map (A) against a second map (B) on the plots where B has a class."""
    m = other > 0
    a, b, t = pred[m], other[m], ref[m]
    N = int(m.sum())
    D = int((a != b).sum())
    KA = int(((a == t) & (b != t)).sum())
    KB = int(((b == t) & (a != t)).sum())
    out = {"second_map": name, "N": N, "D": D, "K_A": KA, "K_B": KB,
           "acc_A": float((a == t).mean()) if N else None, "acc_B": float((b == t).mean()) if N else None, "n": {}}
    for n in BUDGETS:
        pa, pb = plan.which_map_probability(N, D, min(n, D), KA, KB) if D else (0.0, 0.0)
        better = "A" if KA > KB else "B" if KB > KA else None
        out["n"][str(n)] = {"p_names_A": float(pa), "p_names_B": float(pb),
                            "p_names_worse": float(pb if better == "A" else pa if better == "B" else pa + pb),
                            "p_names_better": None if better is None else float(pa if better == "A" else pb)}
    return out


def analyze(priv, checkpoint):
    if prereg_status() != "frozen":
        raise SystemExit("exp112: docs/plan/v13_embeddings.md is not frozen; no statistic of the map before it is")
    import zarr  # noqa: F401  (the environment the extract ran in)
    ref = reference()
    s, order = draw(ref)
    root = open_store()
    quant = root.attrs["geoemb:quantization"]
    vec = load(checkpoint, quant)
    missing = [int(p) for p in s if int(p) not in vec]
    if missing:
        raise SystemExit(f"exp112: {len(missing)} plots of S were not read; rerun extract")
    has = {p: bool(np.isfinite(vec[p]).all()) for p in vec}
    p92 = {}
    with open(PLOTS92, newline="") as f:
        for r in csv.DictReader(f):
            p92[int(r["plotid"])] = (int(r["lcmap"] or 0), int(r["esri"] or 0))
    fit_order = [int(p) for p in order if has[int(p)]]
    result = {"page": "docs/plan/v13_embeddings.md", "year": YEAR, "n_S": int(s.size),
              "no_embedding": int(sum(not has[int(p)] for p in s)), "reference_csv_sha256": sha256_file(REF),
              "by_fit": {}}
    rows_out = []
    rng = np.random.default_rng(SEED)
    for nfit in (FIT,) + FIT_REPORTED:
        F = fit_order[:nfit]
        Fs = set(F)
        T = [p for p in fit_order if p not in Fs]
        y_fit = np.array([ref[p] for p in F])
        y = np.array([ref[p] for p in T])
        cell = {}
        for dims in (DIMS, DIMS_REPORTED):
            Xf = np.stack([vec[p][:dims] for p in F])
            X = np.stack([vec[p][:dims] for p in T])
            pred, conf = fit_predict(Xf, y_fit, X)
            err = pred != y
            dist = np.linalg.norm(X - Xf.mean(0), axis=1)
            c = {"n_fit": nfit, "n_T": len(T), "theta": float(err.mean()), "fit_classes": sorted(int(k) for k in set(y_fit))}
            if nfit == FIT and dims == DIMS:
                c["arm1"] = arm1(conf, err, pred, dist, rng)
                c["arm2"], c["arm3"] = arms23(conf, err)
                other = np.array([p92.get(p, (0, 0)) for p in T])
                c["arm4"] = [arm4(pred, y, other[:, 0], "LCMAP CONUS Collection 1.3"),
                             arm4(pred, y, other[:, 1], "Esri 10 m LULC, crosswalked")]
                per = {}
                for k in sorted(set(y) | set(pred)):
                    mp, tp = pred == k, y == k
                    per[str(int(k))] = {"mapped": int(mp.sum()), "true": int(tp.sum()),
                                        "users": float((y[mp] == k).mean()) if mp.any() else None,
                                        "producers": float((pred[tp] == k).mean()) if tp.any() else None}
                c["per_class"] = per
                for p, pr, cf in zip(T, pred, conf):
                    rows_out.append((p, 0, ref[p], 1, int(pr), float(cf)))
                for p in F:
                    rows_out.append((p, 1, ref[p], 1, "", ""))
            else:
                c["ranking"] = rank_stats(-conf, err)
            cell[str(dims)] = c
        result["by_fit"][str(nfit)] = cell
    norms = np.array([np.linalg.norm(vec[int(p)]) for p in s if has[int(p)]])
    result["norms"] = {q: float(np.quantile(norms, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)}
    for p in s:
        if not has[int(p)]:
            rows_out.append((int(p), 0, ref[int(p)], 0, "", ""))
    result["verdicts"] = verdicts(result["by_fit"][str(FIT)][str(DIMS)])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp112_summary.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(OUT, "exp112_plots.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plotid", "in_fit_1000", "ref", "has_embedding", "pred", "top_probability"])
        for r in sorted(rows_out):
            w.writerow(r)
    log(json.dumps(result["verdicts"], indent=1))


def verdicts(c):
    a1, a2, a3 = c["arm1"], c["arm2"], c["arm3"]
    out = {}
    if a1["errors"] < P1_ERROR_FLOOR:
        out["P1"] = "not determined"
    else:
        pa = a1["own"]["auroc"] >= P1_AUROC and a1["own"]["capture_10"] >= P1_CAPTURE
        pb = a1["lead"] >= P1_LEAD
        out["P1"] = {"a": pa, "b": pb, "holds": pa and pb}
    out["P2"] = all(v["exact_coverage"] >= P2_COVERAGE and abs(v["mc_minus_exact_in_se"]) <= P2_MC_SE
                    for v in a2["cells"].values())
    z = a3["by_alpha"][f"{ALPHA:g}"]
    out["P3"] = {"a": z["violation_share"] <= P3_VIOLATION, "b": z["median_certified_share"] >= P3_SHARE}
    out["P4"] = all(v["p_names_worse"] <= P4_WRONG for m in c["arm4"] for v in m["n"].values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("draw", "extract", "analyze", "smoke"))
    ap.add_argument("--private-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke-plots", type=int, default=4)
    args = ap.parse_args(argv)
    priv = args.private_dir
    os.makedirs(priv, exist_ok=True)
    ref = reference()
    s, order = draw(ref)
    if args.mode == "draw":
        with open(os.path.join(priv, "exp112_draw.json"), "w") as f:
            json.dump({"S": [int(p) for p in s], "fit_order": [int(p) for p in order]}, f)
        log(f"[exp112] S: {s.size} plots of {len(ref)} with a 2018 class; written to the private directory")
    elif args.mode == "extract":
        extract(priv, s, os.path.join(priv, "exp112_vectors.jsonl"), args.workers)
    elif args.mode == "smoke":
        ck = os.path.join(priv, "exp112_smoke.jsonl")
        extract(priv, s[: args.smoke_plots], ck, min(args.workers, args.smoke_plots))
        quant = open_store().attrs["geoemb:quantization"]
        for p, v in load(ck, quant).items():
            log(f"[exp112] smoke plot: norm {np.linalg.norm(v):.3f}, nan {int(np.isnan(v).sum())}")
    else:
        analyze(priv, os.path.join(priv, "exp112_vectors.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
