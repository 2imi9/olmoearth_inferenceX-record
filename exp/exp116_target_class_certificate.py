#!/usr/bin/env python
"""exp116: the target rule, the class design and the certificate on fully labelled real maps (preregistered 24
October 2026, docs/plan/target_class_certificate.md; the predictions are there and were frozen before this ran).

    python exp/exp116_target_class_certificate.py run --units DIR --out FILE [--workers K]   # arms A and B
    python exp/exp116_target_class_certificate.py tile --out FILE --work DIR                 # arm C (P5)
    python exp/exp116_target_class_certificate.py smoke                                      # synthetic units

Runs against the installed package (1.17.0 or later: `estimate.class_allocation`, `_class_design_exact`, the target
rule, `certificate`), whose version and, when given in the environment as PACKAGE_SHA, commit are recorded.

Arm A grades the target rule beside prefix and Bonferroni: every rule's zone is computed from the package's own
functions in certify_zone's order (zone_order, zone_levels, zone_counts, zone_pvalue, apply_zone_rule,
target_weights), checked against certify_zone on the first draws. Arm B grades the class design's exact intervals and
its widths against a simple random sample. Arm C writes six certificates on the package's Dynamic World tile and
verifies them. Every number is a count or share of draws; there is no fit and no Monte Carlo beyond the draws.
"""
import argparse
import contextlib
import functools
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zlib
from multiprocessing import Pool

import numpy as np

from oe_inferencex import __version__
from oe_inferencex import estimate as est

D = 400                      # draws per map and budget
DELTA = 0.1
BUDGETS = (300, 1000)
TARGETS = (0.25, 0.5, 0.75)
P1_BAR = 0.145               # delta + 3 binomial standard errors at 400 draws
COVER_BAR = 0.917            # 0.95 - 3 binomial standard errors at 400 draws
P2_SHARE, P3_SHARE, P4_SHARE = 0.80, 0.99, 0.80
B_BUDGET = 400
RARE = 0.05


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_hypergeom():
    """est.hypergeom_interval with a cache: arm B asks the same (k, n, N, level) many times."""
    orig = est.hypergeom_interval
    if getattr(orig, "_exp116", False):
        return
    cached = functools.lru_cache(maxsize=1 << 20)(lambda k, n, N, conf: orig(int(k), int(n), int(N), float(conf)))
    wrapper = lambda k, n, N, conf=0.95: cached(k, n, N, conf)
    wrapper._exp116 = True
    est.hypergeom_interval = wrapper


# ----------------------------------------------------------------------------- arm A
def _zones(margin_order_pos, idx, err, alpha, n, rules):
    """{rule name: (best level index or None, coverage list, sizes list, refused)} for one draw."""
    pos = margin_order_pos
    out = {}
    for name, (cov, sizes, rule, shares) in rules.items():
        if not cov:
            out[name] = (None, cov, sizes)
            continue
        b, k = est.zone_counts(pos[idx], err[idx].astype(float), sizes)
        p = np.array([est.zone_pvalue(kk, bb, s, alpha) for kk, bb, s in zip(k, b, sizes)])
        _, best = est.apply_zone_rule(p, b, k, alpha, DELTA, rule, n=sizes, weights=shares)
        out[name] = (best, cov, sizes)
    return out


def arm_a(path, name):
    z = np.load(path, allow_pickle=False)
    margin, err = z["margin"].astype(np.float64), z["err"].astype(np.int64)
    N = margin.size
    e = float(err.mean())
    if not 0.02 <= e <= 0.5:
        return {"map": name, "skipped": f"error rate {e:.4f} outside [0.02, 0.5]"}
    alpha = e / 2
    order, pos = est.zone_order(margin)
    cum = np.cumsum(err[order])
    true_err = lambda size: cum[size - 1] / size
    cells = []
    for n in BUDGETS:
        if n > N // 2:
            continue
        cov, sizes, _ = est.zone_levels(N, n, alpha, DELTA, est.ZONE_GRID, None)
        rules = {"prefix": (list(cov), list(sizes), "prefix", None), "bonferroni": (list(cov), list(sizes), "bonferroni", None)}
        for t in TARGETS:
            tg = next(g for g in est.ZONE_GRID if abs(t - g) < 1e-9)
            ct, st = list(cov), list(sizes)
            if tg not in ct:
                ct = sorted(ct + [tg])
                st = [max(1, int(round(c * N))) for c in ct]
            rules[f"target_{t:g}"] = (ct, st, "target", est.target_weights(len(ct), ct.index(tg), DELTA, est.TARGET_WEIGHT))
        rng = np.random.default_rng(np.random.SeedSequence([116, zlib.crc32(name.encode()), n]))
        stats = {r: {"violations": 0, "certified": 0, "coverage_sum": 0.0,
                     **{f"at_least_{t:g}": 0 for t in TARGETS}} for r in rules}
        refused = 0
        checked = []
        for d in range(D):
            idx = rng.choice(N, n, replace=False)
            chk = est.review_set_check(idx, margin)
            if chk["looks_like_a_review_set"] or chk["looks_drawn_from_the_confident_end"]:
                refused += 1                            # certify refuses such a draw: no zone, no violation
                continue
            zs = _zones(pos, idx, err, alpha, n, rules)
            if d < 2 and len(checked) < 2 * len(rules):    # the package's own certify_zone gives the same zones
                for r, (best, cv, _) in zs.items():
                    rule = rules[r][2]
                    kw = {"target": float(r.split("_")[1])} if rule == "target" else {}
                    ref = est.certify_zone(margin, idx, err[idx], alpha, DELTA, rule=rule, **kw)
                    mine = None if best is None else cv[best]
                    checked.append(ref["coverage"] == mine)
            for r, (best, cv, sz) in zs.items():
                s = stats[r]
                if best is None:
                    continue
                s["certified"] += 1
                s["coverage_sum"] += cv[best]
                s["violations"] += int(true_err(sz[best]) > alpha)
                for t in TARGETS:
                    s[f"at_least_{t:g}"] += int(cv[best] >= t - 1e-9)
        if not all(checked):
            raise RuntimeError(f"{name}: the fast path's zones differ from certify_zone's")
        target_right = {f"{t:g}": bool(true_err(max(1, int(round(t * N)))) <= alpha) for t in TARGETS}
        cells.append({"budget": n, "alpha": alpha, "n_units": N, "refused": refused, "checked_against_certify_zone":
                      len(checked), "target_right": target_right,
                      "rules": {r: {"violation_rate": s["violations"] / D, "certified_rate": s["certified"] / D,
                                    "mean_coverage": s["coverage_sum"] / D,
                                    **{k: v / D for k, v in s.items() if k.startswith("at_least_")}}
                                for r, s in stats.items()}})
    return {"map": name, "error_rate": e, "cells": cells}


# ----------------------------------------------------------------------------- arm B
def arm_b(path, name):
    _cached_hypergeom()
    z = np.load(path, allow_pickle=False)
    dec, y = z["dec"].astype(np.int64), z["y"].astype(np.int64)
    C = int(z["n_classes"][0])
    N = dec.size
    present = np.unique(dec)
    K = present.size
    if K < 2 or N < 2 * B_BUDGET or y.max() >= C or dec.max() >= C:
        return {"map": name, "skipped": f"{K} map classes, {N} units"}
    floor = min(50, B_BUDGET // (2 * K))
    members = [np.flatnonzero(dec == c) for c in present]
    sizes = np.array([m.size for m in members])
    alloc = est.class_allocation(sizes, B_BUDGET, floor)
    ua_true = {int(c): float((y[m] == c).mean()) for c, m in zip(present, members)}
    K_ref = np.bincount(y, minlength=C)
    share_true = K_ref / N
    pa_true = [float(((dec == j) & (y == j)).sum() / K_ref[j]) if K_ref[j] else None for j in range(C)]
    oa = float((dec == y).mean())
    rng = np.random.default_rng(np.random.SeedSequence([116, zlib.crc32(name.encode()), 1]))
    rng2 = np.random.default_rng(np.random.SeedSequence([116, zlib.crc32(name.encode()), 2]))
    ua_hit = {int(c): 0 for c in present}
    ua_width = {int(c): 0.0 for c in present}
    joint_hit = oa_hit = 0
    for _ in range(D):
        picks = [rng.choice(m, int(a), replace=False) for m, a in zip(members, alloc)]
        strata = np.concatenate([np.full(len(p), h) for h, p in enumerate(picks)])
        units = np.concatenate(picks)
        refs = y[units]
        per = {c: {} for c in range(C)}
        est._class_design_exact(per, {"condition": {"values": [int(c) for c in present]}}, strata,
                                np.arange(units.size), [int(s) for s in sizes], None, refs, C, C, N)
        ok = True
        for j in range(C):
            se, pe = per[j]["reference_share_exact"], per[j]["producer_accuracy_exact"]
            ok &= se["low"] - 1e-12 <= share_true[j] <= se["high"] + 1e-12
            if pa_true[j] is not None:
                ok &= pe["low"] - 1e-12 <= pa_true[j] <= pe["high"] + 1e-12
        joint_hit += ok
        for c in present:
            u = per[int(c)]["user_accuracy"]
            ua_hit[int(c)] += u["low"] - 1e-12 <= ua_true[int(c)] <= u["high"] + 1e-12
            ua_width[int(c)] += u["high"] - u["low"]
        n_h = np.array([len(p) for p in picks])
        k_h = np.array([int((y[p] != dec[p]).sum()) for p in picks])
        _, lo, hi, _ = est._union_interval(k_h, n_h, sizes)
        oa_hit += 1 - hi - 1e-12 <= oa <= 1 - lo + 1e-12
    srs_width = {int(c): 0.0 for c in present}
    for _ in range(D):
        idx = rng2.choice(N, B_BUDGET, replace=False)
        for c, m in zip(present, members):
            sel = idx[dec[idx] == c]
            if sel.size == 0:
                srs_width[int(c)] += 1.0
                continue
            lo, hi = est.hypergeom_interval(int((y[sel] == c).sum()), int(sel.size), int(m.size))
            srs_width[int(c)] += hi - lo
    classes = [{"class": int(c), "share_of_map": float(s / N), "n_units": int(s), "allocated": int(a),
                "ua_coverage": ua_hit[int(c)] / D, "ua_width_class": ua_width[int(c)] / D,
                "ua_width_srs": srs_width[int(c)] / D} for c, s, a in zip(present, sizes, alloc)]
    return {"map": name, "n_units": N, "n_map_classes": int(K), "floor": int(floor), "joint_coverage": joint_hit / D,
            "overall_coverage": oa_hit / D, "classes": classes}


def _work(args):
    path, name, arm = args
    return (arm_a if arm == "a" else arm_b)(path, name)


def analyze(a_rows, b_rows):
    cells = [(r["map"], c) for r in a_rows if "cells" in r for c in r["cells"]]
    worst = {}
    for rule in ("prefix", "bonferroni", *(f"target_{t:g}" for t in TARGETS)):
        v = [c["rules"][rule]["violation_rate"] for _, c in cells]
        worst[rule] = {"max_violation": max(v) if v else None,
                       "cells_over_bar": sum(x > P1_BAR for x in v), "cells": len(v)}
    p1 = all(worst[f"target_{t:g}"]["cells_over_bar"] == 0 for t in TARGETS)
    right = [(c, t) for _, c in cells for t in TARGETS if c["target_right"][f"{t:g}"]]
    wins = [c["rules"][f"target_{t:g}"][f"at_least_{t:g}"] >= c["rules"]["bonferroni"][f"at_least_{t:g}"]
            for c, t in right]
    p2_share = (sum(wins) / len(wins)) if wins else None
    pairs = [(r, c) for r in b_rows if "classes" in r for c in r["classes"]]
    maps = [r for r in b_rows if "classes" in r]
    ua_ok = sum(c["ua_coverage"] >= COVER_BAR for _, c in pairs) / len(pairs) if pairs else None
    joint_ok = sum(r["joint_coverage"] >= COVER_BAR for r in maps) / len(maps) if maps else None
    oa_ok = sum(r["overall_coverage"] >= COVER_BAR for r in maps) / len(maps) if maps else None
    rare = [c for r, c in pairs if c["share_of_map"] < RARE and c["n_units"] >= r["floor"]]
    narrower = sum(c["ua_width_class"] < c["ua_width_srs"] for c in rare) / len(rare) if rare else None
    return {"P1": {"bar": P1_BAR, "by_rule": worst, "held": p1},
            "P2": {"cells_with_the_target_right": len(right), "share_target_at_least_bonferroni": p2_share,
                   "bar": P2_SHARE, "held": p2_share is not None and p2_share >= P2_SHARE},
            "P3": {"pairs": len(pairs), "maps": len(maps), "ua_share_at_bar": ua_ok, "joint_share_at_bar": joint_ok,
                   "overall_share_at_bar": oa_ok, "coverage_bar": COVER_BAR, "share_bar": P3_SHARE,
                   "held": all(x is not None and x >= P3_SHARE for x in (ua_ok, joint_ok, oa_ok))},
            "P4": {"rare_pairs": len(rare), "share_class_design_narrower": narrower, "bar": P4_SHARE,
                   "held": narrower is not None and narrower >= P4_SHARE}}


def run(units, out, workers):
    files = sorted(glob.glob(os.path.join(units, "*", "*.npz")))
    names = [os.path.relpath(f, units) for f in files]
    shas = {n: _sha256(f) for n, f in zip(names, files)}
    jobs = [(f, n, arm) for f, n in zip(files, names) for arm in ("a", "b")]
    with Pool(workers) as pool:
        res = pool.map(_work, jobs, chunksize=1)
    a_rows = [r for (f, n, arm), r in zip(jobs, res) if arm == "a"]
    b_rows = [r for (f, n, arm), r in zip(jobs, res) if arm == "b"]
    result = {"experiment": "exp116", "package_version": __version__, "package_sha": os.environ.get("PACKAGE_SHA"),
              "python": sys.version.split()[0], "numpy": np.__version__, "draws": D, "delta": DELTA,
              "input_sha256": shas, "arm_a": a_rows, "arm_b": b_rows, "prereg": analyze(a_rows, b_rows)}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["prereg"], indent=1))


# ----------------------------------------------------------------------------- arm C
def tile(out, work):
    from oe_inferencex.cli import main
    from oe_inferencex import certificate as cert
    import oe_inferencex
    z = np.load(os.path.join(os.path.dirname(oe_inferencex.__file__), "sample", "dynamic_world_tile.npz"))
    probs, expert = z["probs"].astype(np.float32), z["expert"].astype(np.int64)
    probs[:, expert < 0] = np.nan                       # no expert label: no-data
    os.makedirs(work, exist_ok=True)
    np.save(os.path.join(work, "dw.npy"), probs)
    dec = np.nanargmax(np.where(np.isnan(probs), -np.inf, probs), axis=0)
    runs = []
    plans = [("random", n, ["--design", "random", "--target", "0.5"]) for n in (200, 300, 400, 500, 600)]
    plans.append(("class", 600, ["--design", "class"]))
    for kind, n, extra in plans:
        tag = f"{kind}_{n}"
        csv_path = os.path.join(work, f"{tag}.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["sample", os.path.join(work, "dw.npy"), "--budget", str(n), "--patch", "1", "--seed-from-map",
                  "--alpha", "0.1", "--out", csv_path, *extra])
        import csv
        rows = list(csv.DictReader(open(csv_path)))
        for r in rows:
            r["wrong"] = int(int(r["map_class"]) != int(expert[int(r["pixel_row"]), int(r["pixel_col"])]))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["certify", csv_path])
        zone = json.load(open(csv_path[:-4] + "_zone.json"))
        certified = zone.get("coverage") is not None or bool(zone.get("certified_share_of_map"))
        row = {"sample": tag, "budget": n, "certified": certified}
        if certified:
            mask = np.load(zone["zone_mask"])
            wrong = (dec != expert)
            row.update({"zone_share": float(mask.mean()), "zone_true_error": float(wrong[mask.astype(bool)].mean()),
                        "alpha": 0.1})
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["certificate", csv_path[:-4] + "_zone.json", "--date", "2021-01-01/2021-12-31",
                             "--out", os.path.join(work, f"cert_{tag}")])
            res = cert.verify(os.path.join(work, f"cert_{tag}", "certificate.json"),
                              scores=os.path.join(work, "dw.npy"))
            row.update({"certificate_written": code == 0, "verified": res["verified"],
                        "statement": res["statement"]})
        runs.append(row)
    written = [r for r in runs if r["certified"]]
    result = {"experiment": "exp116 arm C", "package_version": __version__, "package_sha": os.environ.get("PACKAGE_SHA"),
              "runs": runs, "P5": {"certified": len(written), "all_verify": all(r["verified"] for r in written),
                                   "held": bool(written) and all(r["verified"] for r in written)}}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["P5"]))


def smoke():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(d, "enc"))
    for t in range(2):
        N = 3000
        dec = rng.integers(0, 3, N)
        margin = rng.random(N)
        err = (rng.random(N) < 0.25 * (1 - margin)).astype(np.uint8)
        y = np.where(err == 1, (dec + 1) % 3, dec)
        np.savez(os.path.join(d, "enc", f"task{t}.npz"), margin=margin.astype(np.float32), dec=dec.astype(np.uint8),
                 y=y.astype(np.uint8), err=err, n_classes=np.array([3]))
    run(d, os.path.join(d, "out.json"), 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--units", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    t = sp.add_parser("tile")
    t.add_argument("--out", required=True)
    t.add_argument("--work", required=True)
    sp.add_parser("smoke")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.units, a.out, a.workers)
    elif a.cmd == "tile":
        tile(a.out, a.work)
    else:
        smoke()
