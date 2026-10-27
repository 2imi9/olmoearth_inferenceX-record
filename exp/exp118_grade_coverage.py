"""exp118 arm B: the coverage of grade's intervals (1.18.1) and of normal theory at NLCD 2016's and GFC2020's design
shapes (docs/plan/grade_second_route_and_coverage.md, frozen 000bdf7). Run on aicr, never on the laptop.

    python exp118_grade_coverage.py RECORD_ROOT B1|B2 DRAWS WORKERS > exp118_coverage_B1.json

B1, NLCD 2016 set B (finite): each stratum a population of ten copies of each of its labelled points; n_h drawn
without replacement; the design has the published shares W_h and N_h = 10 n_h. Quantities: grade's level-II run with
the primary-or-alternate rule and the 15 no-change strata declared pure (seed block 0), and its eight-class run of NLCD
against LCMAP (seed block 1, an independent draw).
B2, GFC2020 (point frame): n_h points drawn with replacement from each stratum's labelled points; the design as exp117
read it (areas, unit area 1 ha). Quantities: grade's run of GFC2020 V2 and WorldCover (seed block 0).
Each population's own values are its truth: grade's point estimates on the population (ten copies, or the labelled
points themselves, hold the same shares). Draws from SeedSequence([118, arm, block, draw])."""
import csv
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np

from oe_inferencex import __version__, agency
from oe_inferencex.stratified import Design

ALPHA = 0.05
T8 = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6, 12: 7, 31: 8}
PURE = {str(h): str(c) for h, c in zip(range(1, 16), (11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95))}
PURE8 = {str(h): str(c) for h, c in zip(range(1, 16), (5, 1, 1, 1, 1, 8, 4, 4, 4, 3, 3, 2, 2, 6, 6))}
_S = {}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def quantities(res):
    """Every number of a grading with an interval: (key, estimate, valid interval, normal-theory interval)."""
    out = []
    for m, q in res["overall_accuracy"].items():
        out.append((f"overall/{m}", q))
    for k, q in res["differences"].items():
        out.append((f"difference/{k}", q))
    for m, per in res["per_class"].items():
        for c, row in per.items():
            for k in ("user", "producer", "reference_share"):
                out.append((f"{k}/{m}/{c}", row[k]))
    return out


def load_b1(rec):
    rows = [r for r in csv.DictReader(open(os.path.join(rec, "exp", "out", "exp94_nlcd_points.csv"), newline=""))
            if r["set"] == "B" and r["year"] == "2016"]
    c = lambda r, k: int(float(r[k]))
    ids = sorted({r["stratum"] for r in rows}, key=int)
    pos = {s: h for h, s in enumerate(ids)}
    hidx = np.array([pos[r["stratum"]] for r in rows])
    wsum = np.bincount(hidx, weights=np.array([float(r["weight"]) for r in rows]))
    Nw = np.round(wsum)
    n = np.bincount(hidx)
    pts = {"hidx": hidx, "ids": ids, "W": Nw / Nw.sum(), "n": n,
           "ref": [str(c(r, "ref_nlcd")) for r in rows], "alt": [str(c(r, "ref_nlcd_alternate")) for r in rows],
           "nlcd": [str(c(r, "nlcd")) for r in rows],
           "ref8": [str(T8[c(r, "ref_nlcd")]) for r in rows],
           "alt8": [str(T8[c(r, "ref_nlcd_alternate")]) if c(r, "ref_nlcd_alternate") in T8 else None for r in rows],
           "nlcd8": [str(T8[c(r, "nlcd")]) for r in rows], "lcmap": [str(c(r, "lcmap")) for r in rows]}
    return pts


def grade_b1(P, idx, block):
    """grade on the points idx (a sample, or every point for the truth) with N_h = 10 n_h."""
    d = Design(P["hidx"][idx], P["W"], 10.0 * P["n"], names=P["ids"])
    take = lambda key: [P[key][i] for i in idx]
    if block == 0:
        return agency.grade(d, take("ref"), {"nlcd": take("nlcd")}, alternate=take("alt"), pure=PURE, alpha=ALPHA)
    return agency.grade(d, take("ref8"), {"nlcd8": take("nlcd8"), "lcmap": take("lcmap")}, alternate=take("alt8"),
                        pure=PURE8, alpha=ALPHA)


def draw_b1(P, rng):
    idx = []
    for h in range(len(P["ids"])):
        members = np.flatnonzero(P["hidx"] == h)
        pick = rng.choice(10 * members.size, members.size, replace=False) // 10
        idx.extend(members[pick].tolist())
    return np.array(idx)


def load_b2(rec):
    area = {r["Strata"]: float(r["strata_ha"]) for r in csv.DictReader(
        open(os.path.join(rec, "exp", "out", "exp117_gfc2020_strata_size.csv"), newline=""))}
    rows = list(csv.DictReader(open(os.path.join(rec, "exp", "out", "exp94_gfc_points.csv"), newline="")))
    st = [r["stratum"] for r in rows]
    members = {}
    for i, s in enumerate(st):
        members.setdefault(s, []).append(i)
    return {"area": area, "st": st, "members": {s: np.array(v) for s, v in members.items()},
            "ref": [r["ref_forest"] for r in rows], "gfc": [r["gfc_v2"] for r in rows],
            "wc": [r["worldcover_forest"] for r in rows]}


def grade_b2(P, idx):
    d = agency.design_from_sizes([P["st"][i] for i in idx], P["area"], kind="area", unit_area=1.0)
    take = lambda key: [agency._code(P[key][i], ()) for i in idx]
    return agency.grade(d, take("ref"), {"gfc_v2": take("gfc"), "worldcover_forest": take("wc")}, alpha=ALPHA)


def draw_b2(P, rng):
    idx = []
    for s in sorted(P["members"]):
        m = P["members"][s]
        idx.extend(m[rng.integers(0, m.size, m.size)].tolist())
    return np.array(idx)


def job(args):
    arm, block, d = args
    P = _S["P"]
    rng = np.random.default_rng(np.random.SeedSequence([118, 1 if arm == "B1" else 2, block, d]))
    try:
        if arm == "B1":
            res = grade_b1(P, draw_b1(P, rng), block)
        else:
            res = grade_b2(P, draw_b2(P, rng))
        return arm, block, d, [(k, q["interval"], q["normal_theory"], q["estimate"]) for k, q in quantities(res)], None
    except Exception as exc:
        return arm, block, d, None, f"{type(exc).__name__}: {exc}"


def main(rec, arm, draws, workers):
    t0 = time.time()
    P = load_b1(rec) if arm == "B1" else load_b2(rec)
    _S["P"] = P
    blocks = (0, 1) if arm == "B1" else (0,)
    truth = {}
    for b in blocks:
        res = grade_b1(P, np.arange(P["hidx"].size), b) if arm == "B1" else grade_b2(P, np.arange(len(P["st"])))
        for k, q in quantities(res):
            truth[(b, k)] = q["estimate"]
    say(f"{arm}: {len(truth)} quantities, {draws} draws per block, {workers} workers")
    acc = {key: {"valid": 0, "normal": 0, "n": 0, "valid_width": 0.0, "normal_width": 0.0, "not_computed": 0}
           for key in truth}
    errors = []
    tasks = [(arm, b, d) for b in blocks for d in range(draws)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, (a, b, d, qs, err) in enumerate(pool.imap_unordered(job, tasks, chunksize=4)):
            if err:
                errors.append({"block": b, "draw": d, "error": err})
                continue
            for k, iv, nt, _ in qs:
                t = truth.get((b, k))
                if t is None or isinstance(t, list):
                    continue
                s = acc[(b, k)]
                s["n"] += 1
                if iv is None:
                    s["not_computed"] += 1
                else:
                    s["valid"] += int(iv[0] - 1e-12 <= t <= iv[1] + 1e-12)
                    s["valid_width"] += iv[1] - iv[0]
                if nt is not None:
                    s["normal"] += int(nt[0] - 1e-12 <= t <= nt[1] + 1e-12)
                    s["normal_width"] += nt[1] - nt[0]
            if (i + 1) % 500 == 0:
                say(f"{i + 1} of {len(tasks)} ({time.time() - t0:.0f} s)")
    out = {"arm": arm, "draws_per_block": draws, "package": __version__, "alpha": ALPHA, "quantities": {},
           "errors": len(errors), "first_errors": errors[:5]}
    for (b, k), s in acc.items():
        if not s["n"]:
            continue
        R = s["n"]
        bar = 0.95 - 3 * math.sqrt(0.95 * 0.05 / R)
        valid = s["valid"] / max(R - s["not_computed"], 1)
        out["quantities"][f"{b}/{k}"] = {
            "truth": truth[(b, k)], "draws": R, "bar": bar, "valid_coverage": valid, "normal_coverage": s["normal"] / R,
            "valid_mean_width": s["valid_width"] / max(R - s["not_computed"], 1), "normal_mean_width": s["normal_width"] / R,
            "not_computed": s["not_computed"], "valid_holds": valid >= bar, "normal_below_bar": s["normal"] / R < bar}
    Q = out["quantities"].values()
    out["P2_every_valid_interval_holds"] = all(q["valid_holds"] for q in Q) and not errors
    out["P3_normal_below_bar_somewhere"] = any(q["normal_below_bar"] for q in Q)
    out["valid_coverage_least"] = min(q["valid_coverage"] for q in Q)
    out["normal_coverage_least"] = min(q["normal_coverage"] for q in Q)
    out["seconds"] = time.time() - t0
    json.dump(out, sys.stdout, indent=1, default=float)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
