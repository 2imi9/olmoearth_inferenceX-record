"""exp112's second route (docs/plan/v13_embeddings.md, Independent check before anything is recorded): every graded
number of the run, and the reported ones the inputs allow, recomputed from the extracted vectors, the LCMAP reference
file and exp92's per-plot table with numpy, scipy and scikit-learn only.

Written for the independent audit of 22 October 2026 by an agent that did not write the run script. It does not import
exp/exp112_v13_lcmap.py or oe_inferencex, and it does not read the run's summary or per-plot table. scikit-learn is
used because the map IS logistic regression; everything after the fit is this file's own. Re-derived from the page:
  S: 12,000 of the sorted plot ids with a 2018 class, numpy.random.default_rng(112).choice without replacement, sorted;
  the fit order: S permuted by default_rng(113); F_k: its first k plots with an embedding (k = 300, 1,000, 3,000);
  T_k: the rest of the fit order with an embedding; the map: LogisticRegression(max_iter=2000) on F_k's vectors
  (dequantized (|q|/127.5)^2 * sign(q), NaN at -128) and LCMAP classes; confidence the top probability; error the
  predicted class differing from the reference. The graded map is k = 1,000 on all 128 dimensions.
Its own implementations (exp110_crosscheck's): AUROC by average ranks, capture and excess AURC in expectation over
ties, the bootstrap by weighted ranks, the exact coverage by scipy's hypergeometric tails, certify's prefix rule and
standard level cut by hypergeometric p-values with the review-set guard, and the which-map sign test by enumeration
over the draw (here over the pair of counts, since with eight classes two differing maps can both be wrong).
Its draws are its own (seeds in `config`), not the run's: arm 2's coverage and arm 3's shares agree with the run's
to Monte Carlo error only. Beside P3's 2,000 draws it certifies 100,000 more per alpha, to read how far a 2,000-draw
median can move. About 25 seconds on a laptop CPU.

    uv run --with scikit-learn python -I exp/exp112_crosscheck.py --vectors <exp112_vectors.jsonl> \
        --reference data/lcmap/LCMAP_CU_20211117_V01_REF/LCMAP_CU_20211117_V01_REF.csv --p92 exp/out/exp92_plots.csv

Writes exp/out/exp112_crosscheck.json: aggregates only and the inputs' sha256. No plot id, coordinate, row or
embedding is written. The vectors file is private (the OlmoEarth Artifact License): it is read, never copied.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out", "exp112_crosscheck.json")
CLASSES = {"Developed": 1, "Cropland": 2, "Grass/Shrub": 3, "Tree Cover": 4, "Water": 5, "Wetland": 6, "Snow/Ice": 7,
           "Barren": 8}
YEAR = "2018"
N_S, S_SEED, FIT_SEED = 12000, 112, 113
FIT, FIT_SIZES, DIMS = 1000, (300, 1000, 3000), (128, 64)
ALPHAS, ALPHA, DELTA = (0.05, 0.10), 0.05, 0.10
GRID = tuple(round(j / 20, 2) for j in range(1, 21))      # the package's default certify grid, 0.05 to 1.0
DRAWS = 2000
DRAW_SEED = {100: 2026102201, 300: 2026102203}            # this audit's own simple random samples (arms 2 and 3)
A4_MC_SEED = 2026102204                                   # a Monte Carlo check of the arm-4 enumeration
BOOT, BOOT_SEED = 2000, 2026102205
LARGE_MC, LARGE_MC_SEED, MEDIAN_SETS = 100000, 2026102300, 20000   # certify's distribution, beyond 2,000 draws
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ inputs
def read_reference(path):
    """plotid -> (class or 0, x, y) for the 2018 rows."""
    ref = {}
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image_year"] != YEAR:
                continue
            pid = int(r["plotid"])
            assert pid not in ref, "two 2018 rows for one plot"
            ref[pid] = (CLASSES.get(r["LCMAP"], 0), float(r["x"]), float(r["y"]))
            labels[r["LCMAP"]] = labels.get(r["LCMAP"], 0) + 1
    return ref, labels


def read_vectors(path):
    """plotid -> (zone, dequantized float64 vector, or None when any dimension is no data)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pid = int(d["plotid"])
            assert pid not in out, "a plot twice in the vectors file"
            q = d.get("q")
            if q is None:
                out[pid] = (d.get("zone"), None)
                continue
            q = np.asarray(q, np.int16)
            assert q.shape == (128,) and q.min() >= -128 and q.max() <= 127
            v = (np.abs(q) / 127.5) ** 2 * np.sign(q)
            v = np.where(q == -128, np.nan, v)
            out[pid] = (d.get("zone"), None if np.isnan(v).any() else v)
    return out


def read_p92(path):
    """plotid -> (ref, lcmap, esri) with 0 for no class."""
    def c(s):
        s = (s or "").strip()
        return int(float(s)) if s else 0
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["plotid"])] = (c(r["ref"]), c(r["lcmap"]), c(r["esri"]))
    return out


def albers_lon(x, y):
    """Longitude on LCMAP's Albers (WGS84; parallels 29.5, 45.5; origin 23 N, 96 W). The longitude is exact given
    the ellipsoidal cone constant n; only the zone check uses it."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    e = math.sqrt(e2)

    def m(p):
        return math.cos(p) / math.sqrt(1 - e2 * math.sin(p) ** 2)

    def q(p):
        s = math.sin(p)
        return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * math.log((1 - e * s) / (1 + e * s)))

    p1, p2, p0 = map(math.radians, (29.5, 45.5, 23.0))
    n = (m(p1) ** 2 - m(p2) ** 2) / (q(p2) - q(p1))
    C = m(p1) ** 2 + n * q(p1)
    rho0 = a * math.sqrt(C - n * q(p0)) / n
    theta = np.arctan2(np.asarray(x), rho0 - np.asarray(y))
    return -96.0 + np.degrees(theta / n)


# ------------------------------------------------------------------ ranking statistics (exp110_crosscheck's)
def auroc(u, e):
    u = np.asarray(u, np.float64)
    e = np.asarray(e, bool)
    r = stats.rankdata(u)                     # average ranks: ties counted half
    ne, nc = int(e.sum()), int((~e).sum())
    return float((r[e].sum() - ne * (ne + 1) / 2.0) / (ne * nc))


def groups_desc(u, e):
    vals, inv = np.unique(-np.asarray(u, np.float64), return_inverse=True)   # most suspect first
    tot = np.bincount(inv, minlength=vals.size).astype(np.float64)
    err = np.bincount(inv, weights=np.asarray(e, np.float64), minlength=vals.size)
    return tot, err


def capture(u, e, budget=0.10):
    e = np.asarray(e, bool)
    k = max(1, int(round(budget * e.size)))
    tot, err = groups_desc(u, e)
    start = np.concatenate([[0.0], np.cumsum(tot)[:-1]])
    full = start + tot <= k
    cap = err[full].sum()
    part = np.flatnonzero((start < k) & ~full)
    if part.size:
        g = part[0]
        cap += err[g] * (k - start[g]) / tot[g]
    return float(cap / e.sum())


def aurc_expected(u, e):
    """Mean over ranks i (most confident first) of the expected cumulative errors / i, ties spread uniformly."""
    e = np.asarray(e, np.float64)
    n = e.size
    tot, err = groups_desc(u, e)
    tot, err = tot[::-1], err[::-1]
    start = np.concatenate([[0.0], np.cumsum(tot)[:-1]])
    ebefore = np.concatenate([[0.0], np.cumsum(err)[:-1]])
    gi = np.repeat(np.arange(tot.size), tot.astype(np.int64))
    i = np.arange(1, n + 1, dtype=np.float64)
    j = i - start[gi]
    return float(((ebefore[gi] + err[gi] * j / tot[gi]) / i).mean())


def excess_aurc(u, e):
    e = np.asarray(e, np.float64)
    n, k = e.size, int(e.sum())
    i = np.arange(1, n + 1)
    return aurc_expected(u, e) - float((np.maximum(0, i - (n - k)) / i).mean())


def boot_auroc_prep(u, e):
    order = np.argsort(u, kind="stable")
    s = u[order]
    newg = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(newg)
    gid = np.cumsum(newg) - 1
    return order, starts, gid, e[order]


def boot_auroc(prep, w_full):
    order, starts, gid, es = prep
    w = w_full[order]
    cw = np.cumsum(w)
    wb = np.r_[0.0, cw][starts]
    wg = np.add.reduceat(w, starts)
    mid = wb[gid] + 0.5 * wg[gid]
    wp, wn = w[es].sum(), w[~es].sum()
    return ((w[es] * mid[es]).sum() - wp * wp / 2.0) / (wp * wn)


# ------------------------------------------------------------------ the map
def fit_map(perm, emb, ref, k, dims):
    """F = the first k of the fit order with an embedding; T = the rest with one, in plot-id order."""
    have = [p for p in perm if emb.get(p) is not None]
    F, T = have[:k], sorted(have[k:])
    XF = np.stack([emb[p][:dims] for p in F])
    yF = np.array([ref[p] for p in F])
    XT = np.stack([emb[p][:dims] for p in T])
    yT = np.array([ref[p] for p in T])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(max_iter=2000).fit(XF, yF)
    conv = [str(x.message) for x in w if issubclass(x.category, ConvergenceWarning)]
    other = sorted({x.category.__name__ for x in w if not issubclass(x.category, ConvergenceWarning)})
    pr = clf.predict_proba(XT)
    conf = pr.max(1)
    pred = clf.classes_[pr.argmax(1)]
    info = {"F": len(F), "T": len(T), "classes_in_F": {int(c): int((yF == c).sum()) for c in clf.classes_},
            "n_iter": [int(x) for x in np.ravel(clf.n_iter_)], "convergence_warnings": len(conv),
            "other_warning_types": other,
            "predict_disagrees_with_argmax": int((clf.predict(XT) != pred).sum())}
    dist = np.linalg.norm(XT - XF.mean(0), axis=1)
    return {"T": np.array(T), "y": yT, "pred": pred, "conf": conf, "err": pred != yT, "dist": dist, "info": info}


def ranking(conf, err, pred, dist):
    shares = {c: float((pred == c).mean()) for c in np.unique(pred)}
    rare = -np.array([shares[c] for c in pred])
    out = {"N": int(err.size), "errors": int(err.sum()), "theta": float(err.mean()),
           "auroc": auroc(-conf, err), "capture_10": capture(-conf, err), "excess_aurc": excess_aurc(-conf, err)}
    ctl = {"ctl_class_rarity": rare, "ctl_distance_from_F_mean": dist}
    out["controls"] = {k: {"auroc": auroc(v, err), "capture_10": capture(v, err)} for k, v in ctl.items()}
    best = max(out["controls"], key=lambda k: out["controls"][k]["auroc"])
    out["best_control"] = best
    out["lead_over_best_control"] = out["auroc"] - out["controls"][best]["auroc"]
    k = max(1, int(round(0.1 * err.size)))
    out["oracle_capture_10"] = min(1.0, k / err.sum())
    m = float(np.median(conf[~err]))
    out["median_confidence_correct"] = m
    out["confident_error_share"] = float((conf[err] >= m).mean())
    return out, ctl[best]


# ------------------------------------------------------------------ arm 2: the exact hypergeometric interval
def covered_table(N, K, B, a=0.025):
    ks = np.arange(max(0, B - (N - K)), min(B, K) + 1)
    up = stats.hypergeom.sf(ks - 1, N, K, B)        # P(X >= k | K)
    lo = stats.hypergeom.cdf(ks, N, K, B)           # P(X <= k | K)
    pmf = stats.hypergeom.pmf(ks, N, K, B)
    cov = (up > a) & (lo > a)
    return ks, pmf, cov


# ------------------------------------------------------------------ arm 3: certify (prefix rule, standard cut)
def b_min(alpha, delta=DELTA):
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))


def levels(N, budget, alpha):
    cmin = b_min(alpha) / budget
    cov = [c for c in GRID if c >= cmin - 1e-12]
    sizes = [max(1, int(round(c * N))) for c in cov]
    K0 = [math.floor(alpha * s) + 1 for s in sizes]
    return np.array(cov), np.array(sizes), np.array(K0)


class Zone:
    def __init__(self, conf, err):
        N = conf.size
        self.N = N
        idx = np.arange(N)
        self.order = np.lexsort((idx, -conf))              # descending confidence, ties by index (plot-id order)
        self.pos = np.empty(N, np.int64)
        self.pos[self.order] = idx
        self.cum = np.concatenate([[0], np.cumsum(err[self.order])])
        o = np.lexsort((np.random.default_rng(0).random(N), -conf))   # the review-set guard's tie order
        r = np.empty(N)
        r[o] = idx
        self.pct = (r + 0.5) / N
        self.err = err


def certify_many(Z, samples, alpha, budget=300):
    cov, sizes, K0 = levels(Z.N, budget, alpha)
    S = np.sort(Z.pos[samples], axis=1)
    W = Z.err[Z.order][S]
    cw = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(W, 1)], 1)
    b = np.stack([np.searchsorted(row, sizes, side="left") for row in S])
    k = np.take_along_axis(cw, b, 1).astype(int)
    p = np.where(b == 0, 1.0, stats.hypergeom.cdf(k, sizes[None, :], K0[None, :], b))
    p = np.where(K0[None, :] > sizes[None, :], 0.0, p)
    acc = p <= DELTA
    first_fail = np.where(acc.all(1), acc.shape[1], np.argmin(acc, axis=1))
    n = budget
    fpc = max(Z.N - n, 0) / (Z.N - 1)
    thr = min(0.95, 0.5 + 4.0 * math.sqrt(1 / 12) * math.sqrt(fpc / n))
    refused = Z.pct[samples].mean(1) > thr + 1e-12
    share = np.zeros(S.shape[0])
    zerr = np.full(S.shape[0], np.nan)
    ok = (first_fail > 0) & ~refused
    best = first_fail - 1
    share[ok] = cov[best[ok]]
    nz = sizes[best[ok]]
    zerr[ok] = Z.cum[nz] / nz
    viol = ok & (np.nan_to_num(zerr) > alpha)
    return {"share": share, "certified": ok, "refused": refused, "viol": viol, "zerr": zerr,
            "first_level_passed": acc[:, 0], "levels": {"coverage": cov.tolist(), "sizes": sizes.tolist(),
                                                        "K0": K0.tolist(), "b_min": b_min(alpha)}}


def summarize_certify(res):
    sh = res["share"]
    vals, cnts = np.unique(sh, return_counts=True)
    zr = res["zerr"][res["certified"]]
    return {"p_certify_any": float(res["certified"].mean()),
            "p_first_level_passes_guard_aside": float(res["first_level_passed"].mean()),
            "refused_as_review_set": int(res["refused"].sum()),
            "violations": int(res["viol"].sum()), "violation_share": float(res["viol"].mean()),
            "median_certified_share": float(np.median(sh)), "mean_certified_share": float(sh.mean()),
            "certified_share_counts": {f"{v:g}": int(c) for v, c in zip(vals, cnts)},
            "zone_true_error_median": float(np.median(zr)) if zr.size else None,
            "levels": res["levels"]}


def exact_p_first_level(Z, alpha, budget=300):
    """P(the smallest testable level passes) = P(certify any zone), the review-set guard aside."""
    cov, sizes, K0 = levels(Z.N, budget, alpha)
    n0, k0 = int(sizes[0]), int(K0[0])
    Kz = int(Z.cum[n0])
    bs = np.arange(0, budget + 1)
    pb = stats.hypergeom.pmf(bs, Z.N, n0, budget)
    tot = 0.0
    for bb, w in zip(bs, pb):
        if w < 1e-300 or bb == 0:
            continue
        ks = np.arange(0, bb + 1)
        pk = stats.hypergeom.pmf(ks, n0, Kz, bb)
        pv = stats.hypergeom.cdf(ks, n0, k0, bb)
        tot += w * pk[pv <= DELTA].sum()
    return float(tot), {"n_zone": n0, "zone_true_error": Kz / n0, "level": float(cov[0])}


def large_mc(Z, alpha, budget=300, draws=LARGE_MC, chunk=10000):
    """certify on many more simple random samples, to read how far a 2,000-draw median can move: the per-draw
    distribution of the certified share, and the probability that the median of 2,000 draws reaches P3b's bar,
    by resampling 2,000-draw sets from that distribution."""
    rng = np.random.default_rng(LARGE_MC_SEED + int(round(alpha * 100)))
    share, viol, cert = [], 0, 0
    for s in range(0, draws, chunk):
        smp = np.stack([rng.choice(Z.N, budget, replace=False) for _ in range(min(chunk, draws - s))])
        r = certify_many(Z, smp, alpha, budget)
        share.append(r["share"])
        viol += int(r["viol"].sum())
        cert += int(r["certified"].sum())
    share = np.concatenate(share)
    vals, cnts = np.unique(share, return_counts=True)
    p = cnts / cnts.sum()
    rr = np.random.default_rng(LARGE_MC_SEED + 7)
    meds = np.empty(MEDIAN_SETS)
    for i in range(MEDIAN_SETS):
        c = np.cumsum(rr.multinomial(DRAWS, p))
        lo, hi = vals[np.searchsorted(c, DRAWS // 2 - 1, side="right")], vals[np.searchsorted(c, DRAWS // 2, side="right")]
        meds[i] = (lo + hi) / 2
    return {"draws": draws, "seed": LARGE_MC_SEED + int(round(alpha * 100)), "p_certify_any": cert / draws,
            "violation_share": viol / draws, "median_certified_share": float(np.median(share)),
            "p_share_at_least_0.25": float((share >= 0.25 - 1e-12).mean()),
            "median_of_2000_draws": {"sets": MEDIAN_SETS, "p_at_least_0.25": float((meds >= 0.25 - 1e-12).mean()),
                                     "quantiles_5_50_95": [float(np.quantile(meds, q)) for q in (0.05, 0.5, 0.95)]}}


def zone_true_errors(Z, alpha, budget=300):
    """The true error rate of every testable level's zone (the most confident share), for reading P3."""
    cov, sizes, _ = levels(Z.N, budget, alpha)
    return {f"{c:g}": float(Z.cum[s] / s) for c, s in zip(cov, sizes)}


# ------------------------------------------------------------------ arm 4: which map, by enumeration
def exact_bounds_table(D, n, a=0.0125):
    """For every count x of n drawn from D: the smallest and largest K' whose exact interval at 97.5% holds x."""
    Ks = np.arange(0, D + 1)
    lo, hi = np.empty(n + 1, np.int64), np.empty(n + 1, np.int64)
    for x in range(n + 1):
        ok = (stats.hypergeom.sf(x - 1, D, Ks, n) > a) & (stats.hypergeom.cdf(x, D, Ks, n) > a)
        lo[x], hi[x] = Ks[ok].min(), Ks[ok].max()
    return lo, hi


def which_map(D, KA, KB, n, alpha=0.05):
    """P(names A), P(names B) for n labels drawn without replacement among the D differing plots, of which KA have
    only A right, KB only B right and the rest both wrong. While n <= D/5: the exact two-sided sign test at alpha on
    the labelled plots where one map is right; above: the union-bound interval on the difference (each share at
    97.5%) names a map when it excludes 0."""
    rule = "sign_test" if n <= D / 5 else "interval"
    KN = D - KA - KB
    if rule == "interval":
        lo, hi = exact_bounds_table(D, n)
    pa_tot = pb_tot = 0.0
    xas = np.arange(max(0, n - (D - KA)), min(n, KA) + 1)
    for xa, w in zip(xas, stats.hypergeom.pmf(xas, D, KA, n)):
        if w < 1e-300:
            continue
        rem = n - xa
        xb = np.arange(max(0, rem - KN), min(rem, KB) + 1)
        pxb = stats.hypergeom.pmf(xb, D - KA, KB, rem)
        if rule == "sign_test":
            m = xa + xb
            p = np.where(m > 0, np.minimum(1.0, 2.0 * stats.binom.cdf(np.minimum(xa, xb), np.maximum(m, 1), 0.5)), 1.0)
            na, nb = (p < alpha) & (xa > xb), (p < alpha) & (xb > xa)
        else:
            na, nb = lo[xa] > hi[xb], lo[xb] > hi[xa]
        pa_tot += w * pxb[na].sum()
        pb_tot += w * pxb[nb].sum()
    return float(pa_tot), float(pb_tot), rule


def which_map_mc(a_right, b_right, n, rng, alpha=0.05):
    """The same verdict on DRAWS simulated draws (a check of the enumeration; sign test only)."""
    D = a_right.size
    na = nb = 0
    for _ in range(DRAWS):
        i = rng.choice(D, n, replace=False)
        xa, xb = int(a_right[i].sum()), int(b_right[i].sum())
        m = xa + xb
        p = min(1.0, 2.0 * stats.binom.cdf(min(xa, xb), m, 0.5)) if m else 1.0
        na += p < alpha and xa > xb
        nb += p < alpha and xb > xa
    return na / DRAWS, nb / DRAWS


def arm4(pred, sec, y, rng):
    keep = sec > 0
    a, b, t = pred[keep], sec[keep], y[keep]
    diff = a != b
    D = int(diff.sum())
    KA, KB = int((diff & (a == t)).sum()), int((diff & (b == t)).sum())
    out = {"N": int(keep.sum()), "D": D, "K_A": KA, "K_B": KB, "both_wrong_where_differ": D - KA - KB,
           "acc_A": float((a == t).mean()), "acc_B": float((b == t).mean()),
           "better_agreeing": "A" if KA > KB else ("B" if KB > KA else "tie"), "budgets": {}}
    ar, br = (a == t)[diff], (b == t)[diff]
    for n in (100, 300):
        pa, pb, rule = which_map(D, KA, KB, n)
        right, wrong = (pa, pb) if KA > KB else ((pb, pa) if KB > KA else (None, max(pa, pb)))
        cell = {"p_names_A": pa, "p_names_B": pb, "p_right": right, "p_wrong": wrong, "rule": rule,
                "labels_over_D": n / D}
        if rule == "sign_test":
            ma, mb = which_map_mc(ar, br, n, rng)
            cell["mc_check"] = {"draws": DRAWS, "p_names_A": ma, "p_names_B": mb}
        out["budgets"][str(n)] = cell
    return out


def per_class(y, pred):
    out = {}
    for name, c in CLASSES.items():
        nr, npd = int((y == c).sum()), int((pred == c).sum())
        hit = int(((y == c) & (pred == c)).sum())
        out[name] = {"reference": nr, "predicted": npd, "both": hit,
                     "users_accuracy": hit / npd if npd else None, "producers_accuracy": hit / nr if nr else None}
    return out


def git_commit():
    try:
        c = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "exp/exp112_crosscheck.py"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return c, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vectors", required=True, help="the run's extracted vectors (JSON lines: plotid, zone, q)")
    ap.add_argument("--reference", required=True, help="LCMAP_CU_20211117_V01_REF.csv")
    ap.add_argument("--p92", required=True, help="exp/out/exp92_plots.csv")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    out = {"experiment": "exp112 second route: the graded and reported numbers recomputed from the vectors, the LCMAP "
                         "reference and exp92's table with numpy, scipy and scikit-learn, without "
                         "exp/exp112_v13_lcmap.py or oe_inferencex, and without reading the run's outputs",
           "prereg_page": "docs/plan/v13_embeddings.md", "script": "exp/exp112_crosscheck.py",
           "script_sha256": sha256(os.path.abspath(__file__))}
    commit, dirty = git_commit()
    out["commit"], out["script_uncommitted"] = commit, dirty
    out["environment"] = {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                          "python": platform.python_version(), "machine": platform.machine()}
    out["inputs"] = {"vectors": {"file": os.path.basename(a.vectors), "sha256": sha256(a.vectors)},
                     "reference": {"file": os.path.relpath(os.path.abspath(a.reference), ROOT), "sha256": sha256(a.reference)},
                     "p92": {"file": os.path.relpath(os.path.abspath(a.p92), ROOT), "sha256": sha256(a.p92)}}
    zp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.reference))), "LCMAP_CU_20211117_V01_REF.zip")
    if os.path.isfile(zp):                       # the page's sha256 (eb935d96...b4c) is of this zip, as exp92 records
        import zipfile
        with zipfile.ZipFile(zp) as z:
            member = [m for m in z.namelist() if m.endswith(os.path.basename(a.reference))]
            msha = hashlib.sha256(z.read(member[0])).hexdigest() if len(member) == 1 else None
        out["inputs"]["reference"]["zip"] = {"file": os.path.relpath(zp, ROOT), "sha256": sha256(zp),
                                             "csv_member_matches": msha == out["inputs"]["reference"]["sha256"]}
    out["config"] = {"S": {"size": N_S, "seed": S_SEED}, "fit_order_seed": FIT_SEED, "fit": FIT, "fit_sizes": FIT_SIZES,
                     "alphas": ALPHAS, "delta": DELTA, "grid": GRID, "draws": DRAWS,
                     "draw_seeds": {str(k): v for k, v in DRAW_SEED.items()}, "arm4_mc_seed": A4_MC_SEED,
                     "bootstrap": {"resamples": BOOT, "seed": BOOT_SEED, "level": 0.9, "by": "plot of T"},
                     "certify_large_mc": {"draws": LARGE_MC, "seed": f"{LARGE_MC_SEED} + 100 alpha",
                                          "median_sets": MEDIAN_SETS, "median_sets_seed": LARGE_MC_SEED + 7},
                     "classifier": "sklearn LogisticRegression(max_iter=2000), all else default, float64 inputs",
                     "tie_order_in_certify": "plots of T tied in confidence enter the zone in plot-id order"}

    # ---- inputs and S
    refx, labels = read_reference(a.reference)
    ref = {p: v[0] for p, v in refx.items()}
    out["reference_2018"] = {"plots": len(ref), "labels": labels, "with_class": int(sum(c > 0 for c in ref.values()))}
    ids = np.array(sorted(p for p, c in ref.items() if c > 0))
    S = np.sort(np.random.default_rng(S_SEED).choice(ids, N_S, replace=False))
    vec = read_vectors(a.vectors)
    emb = {p: vec[p][1] for p in S if p in vec}
    sset = set(S.tolist())
    no_emb = int(sum(emb.get(p) is None for p in S))
    out["S"] = {"size": int(S.size), "vectors_file_plots": len(vec),
                "vectors_file_equals_S": bool(set(vec) == sset),
                "S_not_in_vectors_file": int(sum(p not in vec for p in S)),
                "vectors_file_not_in_S": int(sum(p not in sset for p in vec)),
                "without_embedding": no_emb, "class_counts": {k: int(sum(ref[p] == c for p in S)) for k, c in CLASSES.items()}}
    # zone check: the vectors' UTM zone against the plot centre's longitude
    lon = albers_lon([refx[p][1] for p in S], [refx[p][2] for p in S])
    zone = np.floor((lon + 180.0) / 6.0).astype(int) + 1
    vz = np.array([vec[p][0] if p in vec else -1 for p in S])
    edge = np.abs(((lon + 180.0) / 6.0) - np.round((lon + 180.0) / 6.0)) * 6.0 < 0.01
    out["S"]["zone_check"] = {"mismatches": int((vz != zone).sum()), "mismatches_within_0.01_deg_of_a_zone_edge":
                              int(((vz != zone) & edge).sum())}
    # exp92's own reference column, as a check of the class mapping
    p92 = read_p92(a.p92)
    out["S"]["p92_ref_disagrees_with_reference"] = int(sum(p92[p][0] != ref[p] for p in S))
    norms = np.array([np.linalg.norm(emb[p]) for p in S if emb.get(p) is not None])
    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    out["norms_2018_S"] = {"n": int(norms.size), "mean": float(norms.mean()),
                           "quantiles": {f"{q:g}": float(np.quantile(norms, q)) for q in qs}}
    log(f"S {S.size}, vectors equal S: {out['S']['vectors_file_equals_S']}, no embedding {no_emb}, "
        f"zone mismatches {out['S']['zone_check']['mismatches']}, p92 ref disagreements "
        f"{out['S']['p92_ref_disagrees_with_reference']}, median norm {out['norms_2018_S']['quantiles']['0.5']:.4f}")

    perm = np.random.default_rng(FIT_SEED).permutation(S)

    # ---- the graded map: k = 1,000, 128 dimensions
    M = fit_map(perm, emb, ref, FIT, 128)
    out["map"] = M["info"]
    ties = int(M["conf"].size - np.unique(M["conf"]).size)
    out["map"]["confidence_ties"] = {"tied_plots_beyond_first": ties, "confidence_eq_1": int((M["conf"] == 1.0).sum()),
                                     "min": float(M["conf"].min()), "median": float(np.median(M["conf"]))}
    a1, best_ctl = ranking(M["conf"], M["err"], M["pred"], M["dist"])
    # bootstrap: plots of T resampled with replacement, by weighted ranks
    rng = np.random.default_rng(BOOT_SEED)
    pc, pb = boot_auroc_prep(-M["conf"], M["err"]), boot_auroc_prep(best_ctl, M["err"])
    N = M["err"].size
    au, ld = np.empty(BOOT), np.empty(BOOT)
    for r in range(BOOT):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
        au[r] = boot_auroc(pc, w)
        ld[r] = au[r] - boot_auroc(pb, w)
    a1["bootstrap"] = {"resamples": BOOT, "seed": BOOT_SEED, "level": 0.9,
                       "auroc": [float(np.quantile(au, 0.05)), float(np.quantile(au, 0.95))],
                       "lead_over_best_control": [float(np.quantile(ld, 0.05)), float(np.quantile(ld, 0.95))]}
    a1["per_class"] = per_class(M["y"], M["pred"])
    out["arm1"] = a1
    log(f"map: T {N}, errors {a1['errors']}, theta {a1['theta']:.6f}, AUROC {a1['auroc']:.6f}, capture "
        f"{a1['capture_10']:.6f}, best control {a1['best_control']} lead {a1['lead_over_best_control']:.6f}")

    # ---- arm 2
    K = a1["errors"]
    err = M["err"].astype(np.int64)
    a2, samples = {}, {}
    for B in (100, 300):
        ks, pmf, cv = covered_table(N, K, B)
        exact = float(pmf[cv].sum())
        covk = dict(zip(ks.tolist(), cv.tolist()))
        drng = np.random.default_rng(DRAW_SEED[B])
        smp = np.stack([drng.choice(N, B, replace=False) for _ in range(DRAWS)])
        samples[B] = smp
        rep = np.array([covk.get(int(x), False) for x in err[smp].sum(1)])
        se = math.sqrt(exact * (1 - exact) / DRAWS)
        a2[str(B)] = {"exact_coverage": exact, "mc_coverage": float(rep.mean()),
                      "mc_minus_exact_in_se": (float(rep.mean()) - exact) / se}
    out["arm2"] = a2
    log("arm 2", json.dumps(a2))

    # ---- arm 3, on arm 2's B = 300 draws
    Z = Zone(M["conf"], err)
    a3 = {}
    for alpha in ALPHAS:
        s = summarize_certify(certify_many(Z, samples[300], alpha))
        pex, info = exact_p_first_level(Z, alpha)
        s["exact_p_certify_any_guard_aside"] = pex
        s["first_level"] = info
        s["zone_true_error_by_level"] = zone_true_errors(Z, alpha)
        s["p_more_than_half_of_2000_draws_certify"] = float(stats.binom.sf(DRAWS // 2, DRAWS, pex))
        s["large_mc"] = large_mc(Z, alpha)
        a3[f"{alpha:g}"] = s
    out["arm3"] = a3
    log("arm 3", json.dumps({k: {x: v[x] for x in ("violation_share", "median_certified_share", "p_certify_any",
                                                   "exact_p_certify_any_guard_aside")} for k, v in a3.items()}))

    # ---- arm 4
    lc = np.array([p92[p][1] if p in p92 else 0 for p in M["T"]])
    es = np.array([p92[p][2] if p in p92 else 0 for p in M["T"]])
    arng = np.random.default_rng(A4_MC_SEED)
    out["arm4"] = {"lcmap_collection_1_3": arm4(M["pred"], lc, M["y"], arng), "esri": arm4(M["pred"], es, M["y"], arng)}
    for k, v in out["arm4"].items():
        log(f"arm 4 {k}: N {v['N']} D {v['D']} K_A {v['K_A']} K_B {v['K_B']} acc A {v['acc_A']:.4f} B {v['acc_B']:.4f}",
            json.dumps(v["budgets"]))

    # ---- reported: fit sizes and the first 64 dimensions; T_k = the rest of the fit order, and a common set
    common = set(perm[3000:].tolist()) if no_emb == 0 else None
    rep = {}
    for dims in DIMS:
        for k in FIT_SIZES:
            Mk = M if (k == FIT and dims == 128) else fit_map(perm, emb, ref, k, dims)
            r, _ = ranking(Mk["conf"], Mk["err"], Mk["pred"], Mk["dist"])
            r["fit"] = Mk["info"]
            if common is not None:
                c = np.array([p in common for p in Mk["T"].tolist()])
                rc, _ = ranking(Mk["conf"][c], Mk["err"][c], Mk["pred"][c], Mk["dist"][c])
                r["on_common_set"] = {x: rc[x] for x in ("N", "errors", "theta", "auroc", "capture_10", "excess_aurc",
                                                         "lead_over_best_control", "best_control")}
            rep[f"dims{dims}_fit{k}"] = r
            log(f"dims {dims} fit {k}: T {r['N']} theta {r['theta']:.6f} AUROC {r['auroc']:.6f} capture "
                f"{r['capture_10']:.6f} lead {r['lead_over_best_control']:.6f}")
    out["reported_fit_sizes_and_dims"] = rep

    # ---- verdicts by the page's bars, on this route's numbers
    v = {}
    p1a = a1["auroc"] >= 0.70 and a1["capture_10"] >= 0.25
    p1b = a1["lead_over_best_control"] >= 0.05
    v["P1"] = {"errors": a1["errors"], "determined": a1["errors"] >= 200, "a": p1a, "b": p1b,
               "holds": (p1a and p1b) if a1["errors"] >= 200 else None}
    v["P2"] = {"min_exact_coverage": min(x["exact_coverage"] for x in a2.values()),
               "max_abs_mc_minus_exact_in_se": max(abs(x["mc_minus_exact_in_se"]) for x in a2.values()),
               "holds": all(x["exact_coverage"] >= 0.95 and abs(x["mc_minus_exact_in_se"]) <= 3 for x in a2.values())}
    v["P3a"] = {"violation_share": a3["0.05"]["violation_share"], "holds": a3["0.05"]["violation_share"] <= 0.115}
    v["P3b"] = {"median_certified_share": a3["0.05"]["median_certified_share"],
                "holds": a3["0.05"]["median_certified_share"] >= 0.25}
    wrong = {f"{m}:{n}": out["arm4"][m]["budgets"][n]["p_wrong"] for m in out["arm4"] for n in ("100", "300")}
    v["P4"] = {"p_wrong": wrong, "max_p_wrong": max(wrong.values()), "holds": all(x <= 0.025 for x in wrong.values())}
    out["verdicts"] = v
    out["not_computed"] = [
        "the share of plots in 2018 windows computed without the 48-pixel margin (needs the store's provenance "
        "parquet, not an input here)",
        "the run's own Monte Carlo draws (seed 112 through the package's sampler): this route draws its own"]
    out["seconds"] = round(time.time() - T0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=1)
        f.write("\n")
    log("verdicts", json.dumps({k: x["holds"] for k, x in v.items()}))
    log("wrote", os.path.relpath(a.out, ROOT))
    return out


if __name__ == "__main__":
    main()
