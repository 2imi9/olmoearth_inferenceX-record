# Written on 24 October 2026 by an independent agent (workflow wf_8dbe7de7-5db) from the frozen plan
# (docs/plan/target_class_certificate.md), the report and the package's Usage alone, without the run's script or the
# package code; paths made relative to the record when it was committed, nothing else changed.
"""Independent second route for exp116 arm A (certified-zone rules).

Written from the frozen plan, the report's 'The certified zone' section and Usage.md's certify section only.
Does not import oe_inferencex.
"""
import json
import math
import os
import sys
import zlib
from fractions import Fraction

import numpy as np
from scipy.stats import hypergeom

ROOT = os.environ.get('EXP79_UNITS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'exp79_units'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'exp116_ab.json')
DELTA = 0.1
DRAWS = 400
GRID = [k / 20 for k in range(1, 21)]  # 0.05 .. 1.00
TARGETS = [0.25, 0.5, 0.75]
REL_TIE = 1e-9
TARGET_ALWAYS = os.environ.get("TARGET_ALWAYS", "1") == "1"
REFUSE_ON = os.environ.get("REFUSE_ON", "1") == "1"
REFUSE_Z = -4.0

near_ties = []  # (map, n, draw, level, p, thr)
ambiguous = []


def b_min(alpha, delta=DELTA):
    """Large-zone limit, report: ceil(ln delta / ln(1 - alpha))."""
    return math.ceil(math.log(delta) / math.log(1.0 - alpha))


def exact_cdf(k, n_c, K, b):
    """P(X <= k), X ~ Hypergeom(population n_c, K successes, b draws), exact rational."""
    tot = math.comb(n_c, b)
    s = 0
    for x in range(0, min(k, K, b) + 1):
        if b - x <= n_c - K:
            s += math.comb(K, x) * math.comb(n_c - K, b - x)
    return Fraction(s, tot)


def passes(p, thr, k, n_c, K, b, tag):
    if thr <= 0:
        return False
    if abs(p - thr) <= REL_TIE * thr:
        near_ties.append((tag, float(p), thr))
        return exact_cdf(k, n_c, K, b) <= Fraction(thr)
    return p <= thr


def target_weights(tested_cov, target, delta=DELTA):
    """Weight per tested level (ascending): target 0.8 delta, smaller tested levels share 0.2 delta equally,
    larger levels 0. With no smaller tested level the target holds all of delta."""
    w = np.zeros(len(tested_cov))
    smaller = [i for i, c in enumerate(tested_cov) if c < target - 1e-12]
    ti = [i for i, c in enumerate(tested_cov) if abs(c - target) < 1e-12]
    if not ti:
        return w, False
    if smaller:
        w[ti[0]] = 0.8 * delta
        for i in smaller:
            w[i] = 0.2 * delta / len(smaller)
    else:
        w[ti[0]] = delta
    return w, True


def run_cell(name, n, write_detail=False):
    z = np.load(os.path.join(ROOT, name))
    margin = z['margin']
    err = z['err'].astype(bool)
    N = len(margin)
    rate = err.mean()
    alpha = rate / 2
    idx = np.arange(N)
    order = np.lexsort((idx, -margin.astype(np.float64)))  # decreasing margin, ties by index
    rank = np.empty(N, dtype=np.int64)
    rank[order] = idx
    err_sorted = err[order]
    cum_err = np.concatenate([[0], np.cumsum(err_sorted)])
    n_cs = [int(round(c * N)) for c in GRID]
    # exact check of half ties in round(c*N)
    for k, nc in zip(range(1, 21), n_cs):
        num = k * N
        if num % 20 == 10:
            print('  half tie in round(cN)', name, k, N, nc)
    Ks = [math.floor(alpha * nc) + 1 for nc in n_cs]
    true_err = [int(cum_err[nc]) for nc in n_cs]
    zone_wrong = [true_err[i] > alpha * n_cs[i] for i in range(20)]  # error rate > alpha over all units
    zone_wrong2 = [true_err[i] >= Ks[i] for i in range(20)]
    assert zone_wrong == zone_wrong2
    bm = b_min(alpha)
    tested = [i for i, c in enumerate(GRID) if (i + 1) * n >= 20 * bm]  # c*n >= b_min exactly
    tested_f = [i for i, c in enumerate(GRID) if c * n >= bm]
    if tested != tested_f:
        print('  level-cut float/exact differ', name, n, bm)
    J = len(tested)
    tcov = [GRID[i] for i in tested]
    # target rule: the cut-passing levels plus the target itself (fixed before labelling, tested at any budget)
    ttest = {}
    tw = {}
    for t in TARGETS:
        ti = GRID.index(t)
        lv = sorted(set(tested) | ({ti} if TARGET_ALWAYS else set()))
        ttest[t] = lv
        tw[t] = target_weights([GRID[i] for i in lv], t)

    ss = np.random.SeedSequence([116, zlib.crc32(name.encode()), n])
    rng = np.random.default_rng(ss)
    rules = ['prefix', 'bonferroni'] + [f'target_{t}' for t in TARGETS]
    zmin_all = []
    zmax_all = []
    nref = [0]
    cov = {r: np.zeros(DRAWS) for r in rules}
    viol = {r: np.zeros(DRAWS, dtype=bool) for r in rules}
    for d in range(DRAWS):
        s = rng.choice(N, n, replace=False)
        pos = np.sort(rank[s])
        e = err_sorted[pos]
        ce = np.concatenate([[0], np.cumsum(e)])
        p = {}
        bk = {}
        alltest = sorted(set(tested).union(*[set(v) for v in ttest.values()]))
        # review-set check, both ends: mean suspicion percentile more than 4 SD from 0.5 (above: a review set;
        # below: the confident end). The exact percentile/SD convention is not documented: three are computed and
        # a draw whose decision differs between them is flagged.
        zs = []
        for pct, sd in ((pos / N, math.sqrt((N * N - 1) / (12 * N * N) / n * (N - n) / (N - 1))),
                        (pos / (N - 1), math.sqrt((N - n) / (N - 1) / (12 * n))),
                        ((pos + 1) / N, math.sqrt((N * N - 1) / (12 * N * N) / n * (N - n) / (N - 1)))):
            zs.append((pct.mean() - 0.5) / sd)
        zmin_all.append(min(zs)); zmax_all.append(max(zs))
        dec = [abs(zz) > 4.0 for zz in zs]
        refused_draw = dec[0] if REFUSE_ON else False
        if len(set(dec)) > 1:
            ambiguous.append((name, n, d, [float(x) for x in zs]))
            print('  refusal ambiguous across percentile conventions', name, n, d, zs)
        if refused_draw:
            nref[0] += 1
        if alltest:
            ncs = np.array([n_cs[i] for i in alltest])
            bs = np.searchsorted(pos, ncs)  # sampled units with rank < n_c
            ks = ce[bs]
            pv = hypergeom.cdf(ks, ncs, np.array([Ks[i] for i in alltest]), bs)
            for j, i in enumerate(alltest):
                b = int(bs[j]); k = int(ks[j])
                p[i] = float(pv[j]) if b > 0 else 1.0
                bk[i] = (k, n_cs[i], Ks[i], b)
        # prefix
        cert = None
        for i in tested:
            if passes(p[i], DELTA, *bk[i], (name, n, d, i, 'prefix')):
                cert = i
            else:
                break
        res = {'prefix': cert}
        # bonferroni
        cert = None
        for i in tested:
            if passes(p[i], DELTA / J, *bk[i], (name, n, d, i, 'bonf')):
                cert = i
        res['bonferroni'] = cert
        # target (fallback, Wiens 2003)
        for t in TARGETS:
            w, ok = tw[t]
            cert = None
            carry = 0.0
            for j, i in enumerate(ttest[t]):
                lvl = w[j] + carry
                if passes(p[i], lvl, *bk[i], (name, n, d, i, f't{t}')):
                    cert = i
                    carry = lvl
                else:
                    carry = 0.0
            res[f'target_{t}'] = cert
        for r in rules:
            i = res[r]
            if i is not None and not refused_draw:
                cov[r][d] = GRID[i]
                viol[r][d] = zone_wrong[i]
    out = {}
    for r in rules:
        c = cov[r]
        out[r] = {
            'violation_rate': float(viol[r].mean()),
            'certified_rate': float((c > 0).mean()),
            'mean_coverage': float(c.mean()),
            **{f'at_least_{t}': float((c >= t - 1e-12).mean()) for t in TARGETS},
        }
    target_right = {str(t): (not zone_wrong[GRID.index(t)]) for t in TARGETS}
    info = {'refused': nref[0], 'min_z': float(min(zmin_all)), 'max_z': float(max(zmax_all)), 'N': N, 'alpha': alpha, 'b_min': bm, 'tested_levels': J, 'smallest_tested': tcov[0] if tcov else None,
            'target_passes_cut': {str(t): GRID.index(t) in tested for t in TARGETS}, 'target_right': target_right}
    return out, info


if __name__ == '__main__':
    cells = json.loads(sys.argv[1])
    res = {}
    for name, n in cells:
        out, info = run_cell(name, n)
        res[f'{name}|{n}'] = {'rules': out, 'info': info}
        print(name, n, json.dumps(info), flush=True)
    json.dump({'results': res, 'near_ties': near_ties, 'ambiguous_refusals': ambiguous}, open(sys.argv[2], 'w'), indent=1)
