#!/usr/bin/env python
"""exp97: a trusted zone that stays certified however often the reviewer looks (preregistered 8 October 2026).

    python exp/exp97_sequential.py run       # reads exp95's committed cells and exp/out/exp79_units (local)
    python exp/exp97_sequential.py smoke     # synthetic units

**Why.** certify's guarantee holds for one run at a budget fixed before any label (exp96's audit, e85d202): a
reviewer who labels, certifies, adds labels and certifies again gave a wrong zone on 21% to 30% of draws on six maps
where one run is wrong at most 12.25%. The user asked for the real fix (8 October 2026): a certificate that is valid
at every look.

**The rule tested ('sequential', oe_inferencex/sequential.py).** Each grid zone is tested by an e-process on the
labels inside it, read in the order they were drawn: the likelihood ratio of a zone with M wrong windows against one
with M0 = floor(alpha n) + 1, averaged with equal weights over M = 0 .. M0 - 1 (Robbins's method of mixtures), in
closed form through one hypergeometric tail. It is a supermartingale under every count M >= M0 (tests/test_sequential.py
checks it in exact arithmetic, and Ville's bound by dynamic programming over every path on small zones). A zone passes
once its e-process has reached 1/delta and stays passed. Zones are tested in a fixed sequence from an anchor, fixed
before any label, outward; the certified zone is the largest one that, with every zone between it and the anchor, has
passed. So the chance that any look ever certifies a zone wrong more than alpha is at most delta, and the certified
zone never shrinks as labels are added. The default anchor, fixed before this experiment on synthetic maps only (no
cell below was read for it): the smaller of 0.25 and the smallest grid coverage whose zone expects
`min_labels_sequential` labels at the sample's first budget n0 (sequential.anchor_coverage; `sample --design
sequential --alpha` records it in the sidecar). A zone with no wrong label needs `min_labels_sequential` labels, 71
at alpha 0.05 and delta 0.1, where certify's one-look test needs 45: the price of the equal weights, which keep a zone
certifiable after wrong labels, not of reading at every label (an e-process on M = 0 alone passes a clean zone at 45
and never after a wrong label). What exp97 measures as cost is the cost of this rule.

**Cells.** exp95's 34 cells (committed) and exp96's 332 held-out cells (exp/out/exp79_units, local; sha256 recorded),
366 maps, in exp80's zone order; alpha half the cell's error rate; delta 0.1; the package's grid. Budgets exp95's
ladder: round(50 x 1.05^i) from 50 to 3,000 and below half the cell's units. 400 nested draws per cell (a draw is one
random order of the units; the sample at budget n is its first n, as a reviewer who adds labels), from a seed per
cell, SeedSequence([97, crc32(cell name)]), so that cells of the same size do not share their draws.

**Arms.** F: certify as shipped (prefix rule, standard cut), rerun at every budget of the ladder on the same draws.
S(n0): sequential with the default anchor at first budget n0 = 300 and n0 = 1,000 (where n0 is below half the
units). S(c): sequential with the anchor fixed at coverage c = 0.1, 0.25, 0.5. Every arm reads the same draws and is
checked at every ladder budget; P1 grades every S arm, the rest of each S(c) arm is descriptive.

**Measured, per cell and arm.** The pathwise violation: the share of draws on which the zone certified at some
ladder budget is wrong more than alpha; for S also the full claim, the share on which any zone of the certified
sequence at the last budget is wrong more than alpha (it contains the pathwise event, since the certified zone never
shrinks). C(n), the mean certified coverage (0 when no zone); the area, the mean of C over the ladder. The label ratio
rho: with target half of F's largest C, the first ladder budget at which S's C reaches the target over the first at
which F's does, nf (infinite if S's never does within the ladder). The implementation check: on the first two draws of
every cell, each level's pass time against `sequential.level_path`, and each arm's zone at six budgets spread over the
ladder against the package's `certify_zone` (F, standard cut) and `certify_zone_sequential` (S).

**Predictions** (SE = sqrt(0.09 / 400) = 0.015).
- P1 (validity): every S arm's full-claim violation, and so its pathwise violation, is at most 0.1 + 5 SE = 0.175 on
  every cell (the guarantee; about 3,100 rates are read, 1,560 of them distinct full-claim rates; the exact binomial
  tail at 0.1 makes the chance of any rate above 0.175 about 0.005 if all hold at delta). Rates above 0.1 + 3 SE are
  counted, as exp96 did.
- P2 (the harness is the package): the implementation check finds no mismatch on any cell. It can fail; the
  certified zone not shrinking with labels holds by construction in the harness and is recorded, not predicted.
- P3 (replication of exp96's audit, not a new prediction): on the six maps exp96's audit followed, F's pathwise
  violation exceeds 0.175. The share of the 366 cells where it does is reported.
- P4 (cost): on the cells where n0 = 300 is checked, F's largest C is at least 0.1 and the ladder reaches 2 nf (so that
  rho <= 2 can be decided), S(300)'s median rho (np.median of finite and infinite values) is at most 2.

**Decision, fixed now.** If P1 and P2 hold, sequential certify ships as an option (`sample --design sequential`, read
from the top; certify then uses this rule); certify's default stays the one-look rule, and the record states P4's
value as the cost of this rule whether P4 holds or not. If P1 or P2 fails, it does not ship and the failure is
investigated. P1 restates a theorem checked in exact arithmetic, so the decision is expected to be to ship; P4 and the
areas are the informative results.

**Disclosure, before this commit.** To time the harness, run_cell was run at seed 97 (one seed for every cell, the
version before per-cell seeds) on two of the 366 cells, MADOS and Sen1Floods11 (exp95's), and every arm's area and
pathwise violation on them were printed. Nothing above was changed because of them; the two cells stay in the grading
(with their own seeds now, so with other draws). The independent review of this preregistration (three lenses) read
only F curves already recorded by exp95 and exp96.

**What it does not show.** Power on maps other than these; budgets above 3,000; alpha other than half the error rate;
a reviewer whose labels are wrong; anchors other than those listed; other weights for the mixture. The held-out maps
share the 24 tasks' labels. The anchor cannot move to smaller zones as labels accrue without splitting delta (a
weighted fallback could hold part of delta for a smaller zone); that is not tested.
"""
import json
import math
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex import estimate as est  # noqa: E402
from oe_inferencex import sequential as sq  # noqa: E402
import exp95_level_cut as e95  # noqa: E402
import exp96_level_cut_heldout as e96  # noqa: E402

OUT = os.path.join(ROOT, "exp", "out")
SUMMARY = os.path.join(OUT, "exp97_summary.json")
SEED, DRAWS, DELTA = 97, 400, est.ZONE_DELTA
SE = math.sqrt(DELTA * (1 - DELTA) / DRAWS)
BOUND = DELTA + 5 * SE
N0S = (300, 1000)
ANCHORS = (0.1, 0.25, 0.5)
CHECK_DRAWS, CHECK_BUDGETS = 2, 6
AUDIT = os.path.join(OUT, "exp96_audit.json")


def cell_seed(name):
    """The cell's own seed: cells of the same size must not share their draws (review of the preregistration)."""
    return np.random.SeedSequence([SEED, zlib.crc32(name.encode())])


def default_anchor(n0, alpha, delta=DELTA, grid=est.ZONE_GRID):
    """The default anchor fixed above, the package's own: sequential.anchor_coverage(n0, alpha, delta)."""
    return sq.anchor_coverage(n0, alpha, delta, grid)


def kstar(n, alpha, T, delta=DELTA):
    """For t = 0 .. T labels inside a zone of n windows, the most wrong labels with which e(t, k) reaches 1/delta
    (-1 if none), from the package's own decision `sequential.reaches`. e falls with k and rises with t, so the set
    is k <= kstar(t) and kstar never falls: one sweep."""
    out = np.full(T + 1, -1, dtype=np.int64)
    k = -1
    for t in range(1, T + 1):
        while k + 1 <= t and sq.reaches(t, k + 1, n, alpha, delta):
            k += 1
        out[t] = k
    return out


def _fixed_arm(B, K, sizes, alpha, bs, grid, draws, b_min):
    """F: certify's prefix rule with the standard cut at every budget; the certified level per draw and budget
    (-1 for none), decided as certify decides (exp96's harness)."""
    G = len(grid)
    pv = np.ones(B.shape)
    for g in range(G):
        bg, kg, size = B[:, :, g], K[:, :, g], int(sizes[g])
        K0 = math.floor(alpha * size) + 1
        for b in np.unique(bg):
            if b == 0:
                continue
            m = bg == b
            pv[:, :, g][m] = 0.0 if K0 > size else e95._cdf(size, K0, int(b), int(kg[m].max()))[kg[m]]
    best = np.full((draws, len(bs)), -1, dtype=np.int64)
    lv_level = float(est._level(DELTA))
    for t, n in enumerate(bs):
        lv = [j for j, c in enumerate(grid) if c >= est.zone_cut(n, b_min, "standard") - 1e-12]
        if not lv:
            continue
        p = pv[:, t, lv]
        acc = p <= lv_level
        for d, i in np.argwhere(np.abs(p - lv_level) <= 1e-6 * lv_level):
            g = lv[i]
            b, k, size = int(B[d, t, g]), int(K[d, t, g]), int(sizes[g])
            acc[d, i] = est._at_most(est.zone_pvalue(k, b, size, alpha), DELTA,
                                     lambda: est.zone_pvalue_exact(k, b, size, alpha), size)
        acc = np.cumprod(acc, axis=1).astype(bool)
        last = np.where(acc, np.arange(len(lv))[None, :], -1).max(axis=1)
        best[:, t] = np.where(last >= 0, np.array(lv)[np.maximum(last, 0)], -1)
    return best


def _sequential_best(cross, j0, bs):
    """The certified level per draw and budget for the anchor level j0: the largest j >= j0 whose prefix j0..j has
    all passed by the budget (-1 for none). cross[d, j] is the label count at which level j passed (inf if not)."""
    pm = np.maximum.accumulate(cross[:, j0:], axis=1)            # the prefix has passed by the largest of its times
    best = np.full((cross.shape[0], len(bs)), -1, dtype=np.int64)
    for t, n in enumerate(bs):
        cnt = (pm <= n).sum(axis=1)                              # pm rises with j, so the passed prefix is its head
        best[:, t] = np.where(cnt > 0, j0 + cnt - 1, -1)
    return best


def _grade(best, risk, alpha, grid, bs, j0=None):
    cov = np.where(best >= 0, np.asarray(grid)[np.maximum(best, 0)], 0.0)
    wrong = (best >= 0) & (risk[np.maximum(best, 0)] > alpha)
    C = cov.mean(axis=0)
    out = {"mean_coverage": C.tolist(), "area": float(C.mean()), "pathwise_violation": float(wrong.any(axis=1).mean()),
           "per_budget_max_violation": float(wrong.mean(axis=0).max()),
           "never_falls": bool((np.diff(cov, axis=1) >= -1e-12).all())}
    if j0 is not None:                                           # the full claim: every zone j0 .. best at the last budget
        last = best[:, -1]
        bad = np.array([(b >= j0) and bool((risk[j0:b + 1] > alpha).any()) for b in last])
        out["full_claim_violation"] = float(bad.mean())
    return out


def run_cell(err_o, draws=DRAWS, seed=SEED, grid=est.ZONE_GRID, keep=False, check=CHECK_DRAWS):
    """One cell, every arm. With `keep`, the draws and each arm's certified level per draw and budget are returned
    too. `check`: on that many first draws, the implementation check against the package (P2)."""
    N = err_o.size
    alpha = float(err_o.mean() / 2)
    b_min = est.min_labels_to_certify(alpha, DELTA)
    sizes = np.array([max(1, int(round(c * N))) for c in grid])
    risk = np.cumsum(err_o)[sizes - 1] / sizes
    bs = e95.budgets(N)
    out = {"N": int(N), "error_rate": float(err_o.mean()), "alpha": alpha, "b_min": b_min,
           "b_seq": sq.min_labels_sequential(alpha, DELTA), "budgets": bs, "risk_by_level": risk.tolist(), "arm": {}}
    if not bs:
        return out
    nmax, G = max(bs), len(grid)
    rng = np.random.default_rng(seed)
    pos = np.stack([rng.choice(N, nmax, replace=False) for _ in range(draws)])     # draws x nmax, in draw order
    err = err_o[pos]
    idx = np.array(bs) - 1
    B = np.zeros((draws, len(bs), G), np.int32)
    K = np.zeros((draws, len(bs), G), np.int32)
    for d in range(draws):
        inside = pos[d][:, None] < sizes[None, :]
        B[d] = np.cumsum(inside, axis=0)[idx]
        K[d] = np.cumsum(err[d][:, None] * inside, axis=0)[idx]
    kept = {"F": _fixed_arm(B, K, sizes, alpha, bs, grid, draws, b_min)}
    out["arm"]["F"] = _grade(kept["F"], risk, alpha, grid, bs)
    # per level, the label count at which its e-process first reached 1/delta, on every draw
    cross = np.full((draws, G), np.inf)
    for j, s in enumerate(sizes):
        T = int(B[:, -1, j].max())
        ks = kstar(int(s), alpha, T)
        for d in range(draws):
            inz = pos[d] < s
            m = np.flatnonzero(inz) + 1
            if m.size == 0:
                continue
            k = np.cumsum(err[d][inz]).astype(np.int64)
            hit = np.flatnonzero(k <= ks[1:m.size + 1])
            if hit.size:
                cross[d, j] = m[hit[0]]
    anchors = {f"S(c={c:g})": c for c in ANCHORS}
    anchors.update({f"S(n0={n0})": default_anchor(n0, alpha) for n0 in N0S if n0 < N // 2})
    out["anchors"] = anchors
    for name, c in anchors.items():
        j0 = int(np.flatnonzero(np.asarray(grid) >= c - 1e-12)[0])
        kept[name] = _sequential_best(cross, j0, bs)
        out["arm"][name] = _grade(kept[name], risk, alpha, grid, bs, j0)
    if check:
        out["check"] = _check(err_o, pos, cross, kept, bs, alpha, sizes, anchors, grid, check)
    if keep:
        out["_pos"], out["_best"], out["_cross"] = pos, kept, cross
    return out


def _check(err_o, pos, cross, kept, bs, alpha, sizes, anchors, grid, draws):
    """P2: on the first `draws` draws, each level's pass time against sequential.level_path, and each arm's zone at
    CHECK_BUDGETS budgets spread over the ladder against certify_zone (F, standard cut) and certify_zone_sequential
    (S), called on the same labels. err_o is in zone order, so a decreasing margin gives that order."""
    N = err_o.size
    margin = np.linspace(1.0, 0.0, N)
    tb = sorted(set(np.linspace(0, len(bs) - 1, CHECK_BUDGETS).round().astype(int).tolist()))
    grid = np.asarray(grid)
    mism, refused, n = [], 0, 0
    for d in range(min(draws, pos.shape[0])):
        for j, s in enumerate(sizes):
            inz = pos[d] < s
            r = sq.level_path(np.flatnonzero(inz) + 1, err_o[pos[d]][inz].astype(int), int(s), alpha, DELTA)
            got = cross[d, j]
            n += 1
            if (r["passed_at"] is None) != (not np.isfinite(got)) or (r["passed_at"] is not None and r["passed_at"] != got):
                mism.append(["cross", d, j, r["passed_at"], None if not np.isfinite(got) else int(got)])
        for t in tb:
            p = pos[d][:bs[t]]
            calls = [("F", lambda: est.certify_zone(margin, p, err_o[p], alpha, cut="standard"))]
            calls += [(a, (lambda c=c: sq.certify_zone_sequential(margin, p, err_o[p], alpha, anchor=c)))
                      for a, c in anchors.items()]
            for arm, call in calls:
                n += 1
                try:
                    got = call()["coverage"] or 0.0
                except ValueError:
                    refused += 1                                 # the package refuses a draw like a review set
                    continue
                b = kept[arm][d, t]
                want = float(grid[b]) if b >= 0 else 0.0
                if got != want:
                    mism.append(["zone", d, arm, bs[t], got, want])
    return {"comparisons": n, "mismatches": mism, "refused": refused}


def _rho(cell, arm):
    """(rho, nf, decidable): rho is inf when S never reaches the target within the ladder; decidable when the ladder
    reaches 2 nf, so that rho <= 2 can be told from rho > 2."""
    C_f = np.asarray(cell["arm"]["F"]["mean_coverage"])
    target = C_f.max() / 2
    bs = cell["budgets"]
    nf = bs[int(np.flatnonzero(C_f >= target)[0])]
    C_s = np.asarray(cell["arm"][arm]["mean_coverage"])
    hit = np.flatnonzero(C_s >= target)
    return (bs[int(hit[0])] / nf if hit.size else math.inf), nf, max(bs) >= 2 * nf


def _finite(x):
    return None if x is None or not math.isfinite(x) else float(x)


def analyze(rows, audit_cells=None):
    rows = {k: v for k, v in rows.items() if v["budgets"]}
    s_arms = [(k, a) for k, v in rows.items() for a in v["arm"] if a != "F"]
    full = [rows[k]["arm"][a]["full_claim_violation"] for k, a in s_arms]
    path = [rows[k]["arm"][a]["pathwise_violation"] for k, a in s_arms]
    worst = max(max(full), max(path))
    falls = [(k, a) for k, a in s_arms if not rows[k]["arm"][a]["never_falls"]]
    mism = {k: v["check"]["mismatches"] for k, v in rows.items() if v.get("check") and v["check"]["mismatches"]}
    f_over = [k for k, v in rows.items() if v["arm"]["F"]["pathwise_violation"] > BOUND]
    if audit_cells is None and os.path.exists(AUDIT):
        audit_cells = [f"exp96:{k}" for k in json.load(open(AUDIT))["pathwise"]]
    rep = {k: rows[k]["arm"]["F"]["pathwise_violation"] for k in (audit_cells or []) if k in rows}
    elig, rho = [], []
    for k, v in rows.items():
        if "S(n0=300)" not in v["arm"] or max(v["arm"]["F"]["mean_coverage"]) < 0.1:
            continue
        r, nf, ok = _rho(v, "S(n0=300)")
        if ok:
            elig.append(k)
            rho.append(r)
    med = float(np.median(rho)) if rho else None
    q = np.percentile(rho, [25, 50, 75], method="inverted_cdf").tolist() if rho else None
    desc = {}
    for a in sorted({a for v in rows.values() for a in v["arm"]}):
        have = [v for v in rows.values() if a in v["arm"]]
        desc[a] = {"cells": len(have), "median_area": float(np.median([v["arm"][a]["area"] for v in have])),
                   "max_pathwise_violation": float(max(v["arm"][a]["pathwise_violation"] for v in have)),
                   "cells_pathwise_above_bound": int(sum(v["arm"][a]["pathwise_violation"] > BOUND for v in have)),
                   "cells_never_falling": int(sum(v["arm"][a]["never_falls"] for v in have))}
    return {"P1": {"holds": worst <= BOUND, "max_violation": worst, "bound": BOUND, "rates": len(full) + len(path),
                   "rates_above_3se": int(sum(x > DELTA + 3 * SE for x in full + path))},
            "P2": {"holds": not mism, "comparisons": int(sum(v["check"]["comparisons"] for v in rows.values() if v.get("check"))),
                   "refused": int(sum(v["check"]["refused"] for v in rows.values() if v.get("check"))),
                   "cells_with_mismatch": {k: m[:5] for k, m in list(mism.items())[:20]},
                   "never_falls_by_construction": not falls},
            "P3": {"replication": True, "holds": bool(rep) and all(x > BOUND for x in rep.values()),
                   "audit_cells": rep, "cells_F_pathwise_above_bound": len(f_over), "of": len(rows)},
            "P4": {"holds": med is not None and med <= 2, "median_rho": _finite(med), "median_is_infinite":
                   med is not None and math.isinf(med), "cells": len(elig),
                   "rho_quartiles_inverted_cdf": [_finite(x) for x in q] if q else None,
                   "cells_never_reaching": int(sum(math.isinf(r) for r in rho))},
            "decision": {"ships_as_option": worst <= BOUND and not mism},
            "descriptive": desc}


def heldout_as_recorded():
    """exp96's held-out cells, exactly the files exp96 recorded: a units directory may hold more (the cluster's also
    holds exp79's olmoearth_base_rtx run, which heldout_cells would read as a 16th encoder), so only the recorded
    files are kept, and each must have the sha256 exp96 recorded."""
    held, shas = e96.heldout_cells()
    recorded = json.load(open(os.path.join(OUT, "exp96_summary.json")))["input_sha256"]
    missing = sorted(set(recorded) - set(shas))
    changed = sorted(k for k in recorded if k in shas and shas[k] != recorded[k])
    if missing or changed:
        raise SystemExit(f"exp96's held-out files differ here: {len(missing)} missing, {len(changed)} changed "
                         f"(first: {(missing + changed)[:3]})")
    keep = {k[:-4] for k in recorded}
    return {k: v for k, v in held.items() if k in keep}, dict(recorded)


def _job(item):
    name, err_o = item
    return name, run_cell(err_o, seed=cell_seed(name))


def main(argv):
    if argv[:1] == ["smoke"]:
        rng = np.random.default_rng(0)
        err = (rng.random(6000) < np.linspace(0.005, 0.4, 6000)).astype(float)
        r = run_cell(err, draws=40, seed=cell_seed("smoke"))
        print({a: (round(v["area"], 3), round(v["pathwise_violation"], 3), v["never_falls"]) for a, v in r["arm"].items()})
        print("check:", r["check"]["comparisons"], "comparisons,", len(r["check"]["mismatches"]), "mismatches,",
              r["check"]["refused"], "refused")
        print("smoke ok")
        return
    held, shas = heldout_as_recorded()
    seen = e95.cells()
    items = [(f"exp95:{k}", v) for k, v in seen.items()] + [(f"exp96:{k}", v) for k, v in held.items()]
    rows = {}
    with Pool(int(os.environ.get("EXP97_WORKERS", "6"))) as pool:
        for name, r in pool.imap_unordered(_job, items):
            rows[name] = r
            if r["budgets"]:
                print(f"{name:60s} N={r['N']:7d} " + " ".join(
                    f"{a} {v['area']:.3f}/{v['pathwise_violation']:.3f}" for a, v in r["arm"].items()), flush=True)
    rows = dict(sorted(rows.items()))
    out = {"experiment": "exp97", "preregistered": True, "draws": DRAWS, "seed": "SeedSequence([97, crc32(cell name)])",
           "delta": DELTA,
           "input_sha256": shas, "cells": rows, "prereg": analyze(rows)}
    for v in rows.values():                                  # JSON has no infinity: a never-passed level is null
        if v.get("check"):
            v["check"]["mismatches"] = [[None if isinstance(x, float) and not math.isfinite(x) else x for x in m]
                                        for m in v["check"]["mismatches"]]
    json.dump(out, open(SUMMARY, "w"), default=float, allow_nan=False)
    print(json.dumps({k: v for k, v in out["prereg"].items() if k != "descriptive"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
