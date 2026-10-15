"""exp110's second route (docs/plan/ftw_fields.md, Independent check before anything is recorded): every graded number
of exp/out/exp110_summary.json recomputed from the local FTW data with numpy, scipy and rasterio only.

Built from the independent audit of 14 October 2026, written by an agent that did not write the run script. It does not
import exp/exp110_ftw_field_map.py or oe_inferencex, and it does not read the run's summary: tests/test_exp110.py
(test_exp110_recorded_by_a_second_route) compares the two files. Every reading is re-derived from the page:
  window 8 x 8 mask px per chip; the map's extent p = field + boundary; pixel class p > 0.5; pixel confidence
  2|p - 0.5|; window confidence the mean; window class the majority, a 32/32 tie to the class with the larger summed
  confidence (equal sums: background, as an argmax would); truth the 2-class mask's majority, a 32/32 window left out.
Its own implementations: AUROC by average ranks (scipy.stats.rankdata), capture and excess AURC in expectation over
ties, the chip bootstrap by weighted ranks, the exact coverage by scipy's hypergeometric tails, certify's prefix rule
and standard level cut by hypergeometric p-values with the review-set guard, the which-map sign test by enumeration.

Beside the graded reading it computes the field + boundary reading (truth = the majority of mask3 in {1, 2}), which
the owner ruled on 14 October 2026 is reported beside the graded result, never in place of it: the 2-class mask
turned out to be field only (mask2 == (mask3 == 1)), a difference found after the labels were read with the maps.

    python -I exp/exp110_crosscheck.py --data <the local FTW data directory>

Writes exp/out/exp110_crosscheck.json: aggregates only (per country and pooled), the inputs' sha256 and whether they
match the acquisition manifest's file list. No chip identifier, tile name or coordinate is written. About seven
minutes on a laptop CPU; peak memory about 1.3 GB.
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

import numpy as np
import scipy
from scipy import ndimage, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out", "exp110_crosscheck.json")
COUNTRIES = ("belgium", "croatia", "slovenia", "sweden")
POPS = COUNTRIES + ("pooled",)
MAP_YEAR = {"belgium": 2021, "croatia": 2023, "slovenia": 2021, "sweden": 2021}
PATCH = 8
G = 256 // PATCH
ALPHA, DELTA = 0.05, 0.10
GRID = tuple(round(j / 20, 2) for j in range(1, 21))      # the package's default certify grid, 0.05 to 1.0
DRAWS = 2000
SEED = 110                                                # draw r uses seed 110 + r, as the run's draws did
INDEPENDENT_SEED = 20261014                               # a second, independent set of 2,000 certify draws
FB_INDEPENDENT_SEED = 777                                 # the audit's draws for the field + boundary reading
BOOT_SEED = 424242
BOOT = {"country": 400, "pooled": 150}                    # fewer resamples than the run's 2,000 (Monte Carlo)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ readings
def win(a, patch=PATCH):
    """(n, 256, 256) -> (n, g, g, patch * patch)"""
    n, H, W = a.shape
    g = H // patch
    return a.reshape(n, g, patch, g, patch).transpose(0, 1, 3, 2, 4).reshape(n, g, g, patch * patch)


def extent(f, b, patch=PATCH, chunk=48):
    n = f.shape[0]
    g = 256 // patch
    conf = np.empty((n, g, g))
    cls = np.empty((n, g, g), np.int8)
    ties = [0, 0]
    for s in range(0, n, chunk):
        p = f[s:s + chunk].astype(np.float64) + b[s:s + chunk].astype(np.float64)
        hard = win(p > 0.5, patch)
        mar = win(np.abs(p - 0.5) * 2.0, patch)
        conf[s:s + len(p)] = mar.mean(axis=-1)
        c1 = hard.sum(-1)
        npx = patch * patch
        s1 = np.where(hard, mar, 0.0).sum(-1)
        s0 = np.where(~hard, mar, 0.0).sum(-1)
        tie = 2 * c1 == npx
        cl = np.where(2 * c1 > npx, 1, 0)
        cl = np.where(tie, (s1 > s0).astype(int), cl)
        cls[s:s + len(p)] = cl
        ties[0] += int(tie.sum())
        ties[1] += int((tie & (s1 > s0)).sum())
    return conf, cls, ties


def three_class(f, b, patch=PATCH, chunk=32):
    n = f.shape[0]
    g = 256 // patch
    conf = np.empty((n, g, g))
    cls = np.empty((n, g, g), np.int8)
    for s in range(0, n, chunk):
        ff = f[s:s + chunk].astype(np.float64)
        bb = b[s:s + chunk].astype(np.float64)
        pr = np.stack([1.0 - ff - bb, ff, bb], axis=1)
        hard = pr.argmax(1)
        top = pr.max(1)
        hw, tw = win(hard, patch), win(top, patch)
        conf[s:s + len(ff)] = tw.mean(-1)
        cnt = np.stack([(hw == k).sum(-1) for k in range(3)], -1)
        wsum = np.stack([np.where(hw == k, tw, 0.0).sum(-1) for k in range(3)], -1)
        topc = cnt.max(-1, keepdims=True)
        tied = (cnt == topc).sum(-1) > 1
        res = cnt.argmax(-1)
        res = np.where(tied, np.where(cnt == topc, wsum, -np.inf).argmax(-1), res)
        cls[s:s + len(ff)] = res
    return conf, cls


def truth(mask, n_classes, patch=PATCH):
    """The majority of each window; -1 where the top count is tied."""
    w = win(np.asarray(mask).astype(np.int16), patch)
    cnt = np.stack([(w == k).sum(-1) for k in range(n_classes)], -1)
    topc = cnt.max(-1, keepdims=True)
    tied = (cnt == topc).sum(-1) > 1
    return np.where(tied, -1, cnt.argmax(-1)).astype(np.int8)


def flat(a):
    """Per-chip window grid (n, g, g) -> the one-row mosaic's raster order."""
    return np.ascontiguousarray(np.asarray(a).transpose(1, 0, 2)).ravel()


# ------------------------------------------------------------------ ranking statistics
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


def arm1_stats(conf, err, ctl):
    out = {"N": int(err.size), "errors": int(err.sum()), "theta": float(err.mean()),
           "auroc": auroc(-conf, err), "capture_10": capture(-conf, err)}
    out["controls"] = {k: {"auroc": auroc(v, err), "capture_10": capture(v, err)} for k, v in ctl.items()}
    best = max(out["controls"], key=lambda k: out["controls"][k]["auroc"])
    out["best_control"] = best
    out["lead_over_best_control"] = out["auroc"] - out["controls"][best]["auroc"]
    k = max(1, int(round(0.1 * err.size)))
    out["oracle_capture_10"] = min(1.0, k / err.sum())
    m = float(np.median(conf[~err]))
    out["confident_error_share"] = float((conf[err] >= m).mean())
    return out


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


# ------------------------------------------------------------------ arm 2: the exact hypergeometric interval
def covered_table(N, K, B, a=0.025):
    """Error counts k of a sample of B, their probabilities, and whether the exact interval at k contains K/N."""
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
        self.order = np.lexsort((idx, -conf))              # descending confidence, ties by index (raster order)
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
    S = np.sort(Z.pos[samples], axis=1)                    # (draws, budget) positions in zone order
    W = Z.err[Z.order][S]
    cw = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(W, 1)], 1)
    b = np.stack([np.searchsorted(row, sizes, side="left") for row in S])        # (draws, levels)
    k = np.take_along_axis(cw, b, 1).astype(int)
    p = np.where(b == 0, 1.0, stats.hypergeom.cdf(k, sizes[None, :], K0[None, :], b))
    p = np.where(K0[None, :] > sizes[None, :], 0.0, p)
    acc = p <= DELTA
    first_fail = np.where(acc.all(1), acc.shape[1], np.argmin(acc, axis=1))   # prefix: levels before the first failure
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
    return {"share": share, "certified": ok, "refused": refused, "viol": viol, "zerr": zerr}


def summarize_certify(res):
    sh = res["share"]
    vals, cnts = np.unique(sh, return_counts=True)
    zr = res["zerr"][res["certified"]]
    return {"p_certify_any": float(res["certified"].mean()), "refused_as_review_set": int(res["refused"].sum()),
            "violations": int(res["viol"].sum()), "violation_share": float(res["viol"].mean()),
            "median_certified_share": float(np.median(sh)),
            "certified_share_counts": {f"{v:g}": int(c) for v, c in zip(vals, cnts)},
            "zone_true_error_median": float(np.median(zr)) if zr.size else None}


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


def draws(N, budget, seed0=None, rng=None):
    if rng is None:
        return np.stack([np.random.default_rng(seed0 + r).choice(N, budget, replace=False) for r in range(DRAWS)])
    return np.stack([rng.choice(N, budget, replace=False) for _ in range(DRAWS)])


# ------------------------------------------------------------------ arm 4
def sign_crit(m, alpha=0.05):
    ks = np.arange(0, m // 2 + 1)
    ok = 2 * stats.binom.cdf(ks, m, 0.5) < alpha
    return int(ks[ok].max()) if ok.any() else -1


def which_map_exact(D, KA, KB, n):
    """(P(names map a = 2e), P(names map b = B3), critical count) for the sign test on n of the D differing windows."""
    assert KA + KB == D
    kc = sign_crit(n)
    xs = np.arange(max(0, n - (D - KA)), min(n, KA) + 1)
    px = stats.hypergeom.pmf(xs, D, KA, n)
    return float(px[(n - xs) <= kc].sum()), float(px[xs <= kc].sum()), kc


def arm4_counts(a, b, t):
    diff = a != b
    D, KA, KB = int(diff.sum()), int((diff & (a == t)).sum()), int((diff & (b == t)).sum())
    return {"N": int(a.size), "D": D, "K_A": KA, "K_B": KB, "better_agreeing": "2e" if KA > KB else "b3"}


# ------------------------------------------------------------------ data
def manifest(data):
    m = {}
    with open(os.path.join(data, "manifest_files.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[row["path"]] = row["sha256"]
    return m


def read_s2(data, c, ids):
    import rasterio
    out = np.empty((len(ids), 8, 256, 256), np.uint16)
    hashes = []
    for i, cid in enumerate(ids):
        for j, w in enumerate(("window_a", "window_b")):
            rel = f"v1/{c}/s2_images/{w}/{cid}.tif"
            hashes.append((rel, sha256(os.path.join(data, rel))))
            with rasterio.open(os.path.join(data, rel)) as src:
                out[i, 4 * j:4 * j + 4] = src.read()
    return out, hashes


def variance_control(s2, keep):
    n = s2.shape[0]
    sds = []
    acc = np.zeros((n, G, G))
    kw = win(keep).astype(np.float64)
    cnt = kw.sum(-1)
    for j in range(8):
        v = s2[:, j]
        nn = s1 = s2s = 0
        for s in range(0, n, 64):
            vals = v[s:s + 64][keep[s:s + 64]].astype(np.int64)
            nn, s1, s2s = nn + vals.size, s1 + int(vals.sum()), s2s + int((vals * vals).sum())
        sd = math.sqrt((nn * s2s - s1 * s1) / (nn * nn))       # exact integers until the division
        sds.append(sd)
        for s in range(0, n, 64):
            x = win(v[s:s + 64].astype(np.float64) / sd)
            k = kw[s:s + 64]
            c = cnt[s:s + 64]
            m = (x * k).sum(-1) / np.maximum(c, 1)
            d = (((x - m[..., None]) ** 2) * k).sum(-1) / np.maximum(c, 1)
            acc[s:s + 64] += np.where(c > 0, d, 0.0)
    return acc / 8, int((cnt == 0).sum()), sds


def country(data, c, all_hashes):
    t = time.time()
    rec = {}
    mp = f"v1/{c}/map2e_{MAP_YEAR[c]}_on_mask_grid_{c}.npz"
    b3p, swp = f"v1/{c}/b3_on_mask_grid_{c}.npz", f"v1/{c}/b3swap_on_mask_grid_{c}.npz"
    for rel in (mp, b3p, swp):
        all_hashes[rel] = sha256(os.path.join(data, rel))
    with np.load(os.path.join(data, mp), allow_pickle=False) as z:
        ids = z["chip_ids"].astype(str)
        seam = np.array(["+" in t_ for t_ in z["map_tiles"].astype(str)])
        assert bool(z["map_valid"].all())
        f, b = z["map_field"], z["map_boundary"]
        m2, m3 = z["mask2"], z["mask3"]
    n = len(ids)
    rec["chips"], rec["seam_chips"], rec["windows"] = n, int(seam.sum()), n * G * G
    rec["sorted_ids"] = bool(list(ids) == sorted(ids))
    # label checks
    rec["mask2_values"] = sorted(int(v) for v in np.unique(m2))
    rec["mask3_values"] = sorted(int(v) for v in np.unique(m3))
    rec["mask2_eq_mask3_field"] = bool(np.array_equal(m2 == 1, m3 == 1))
    rec["mask2_eq_mask3_field_or_boundary"] = bool(np.array_equal(m2 == 1, np.isin(m3, (1, 2))))
    rec["boundary_px_coded_background_in_mask2"] = int(((m3 == 2) & (m2 == 0)).sum())
    rec["mask3_px_counts"] = [int((m3 == k).sum()) for k in range(4)]
    # the 2e map
    conf8, cls8, ties8 = extent(f, b)
    conf16, cls16, _ = extent(f, b, 16)
    conf3, cls3 = three_class(f, b)
    rec["map_window_ties"] = {"tied_32_32": ties8[0], "tie_to_field": ties8[1]}
    rec["map_class_share_field"] = float((cls8 == 1).mean())
    rec["map_windows_conf_1"] = int((conf8 == 1.0).sum())
    rec["map_conf_median"] = float(np.median(conf8))
    del f, b
    # truths: the graded one (the 2-class mask), the field + boundary one, the reported ones
    t8 = truth(m2 == 1, 2)
    t16 = truth(m2 == 1, 2, 16)
    t3 = truth(m3, 3)
    tfb = truth(np.isin(m3, (1, 2)), 2)
    dil = ndimage.binary_dilation(m2 == 1, structure=np.ones((1, 3, 5), bool))
    tlen = truth(dil, 2)
    del m2, m3, dil
    rec["truth_tied"], rec["truth_tied_16"], rec["truth3_tied"] = int((t8 < 0).sum()), int((t16 < 0).sum()), int((t3 < 0).sum())
    rec["truth_field_share"] = float((t8[t8 >= 0] == 1).mean())
    rec["field_plus_boundary_truth_tied"] = int((tfb < 0).sum())
    # S2: zero imagery and the variance control
    s2, hashes = read_s2(data, c, ids)
    for rel, h in hashes:
        all_hashes[rel] = h
    za, zb = (s2[:, :4] == 0).all(1), (s2[:, 4:] == 0).all(1)
    zero_either = za | zb
    keep8 = (s2 != 0).any(1)                     # the run's reading: a pixel zero in all 8 bands is left out
    var8, var_empty, sds = variance_control(s2, keep8)
    rec["px_zero_one_window_only"], rec["px_zero_all8"] = int((za ^ zb).sum()), int((za & zb).sum())
    var_alt = variance_control(s2, ~zero_either)[0] if (za ^ zb).any() else None   # the other zero reading
    del s2
    zero_win = win(zero_either).any(-1)
    rec["zero_imagery_windows"] = int(zero_win.sum())
    rec["variance_windows_no_pixel"], rec["band_sd"] = var_empty, sds
    rare = np.where(cls8 == 1, -float((cls8 == 1).mean()), -float((cls8 == 0).mean()))
    # PRUE B3, in the stated band order and reversed
    with np.load(os.path.join(data, b3p), allow_pickle=False) as z:
        assert list(z["class_order"].astype(str)) == ["background", "field", "boundary"]
        assert np.array_equal(z["chip_ids"].astype(str), ids)
        rec["b3_zero_flag_matches_s2"] = bool(np.array_equal(z["s2_allzero_either"], zero_either))
        _, b3cls, _ = extent(z["b3_field"], z["b3_boundary"])
    with np.load(os.path.join(data, swp), allow_pickle=False) as z:
        assert np.array_equal(z["chip_ids"].astype(str), ids)
        rec["swap_band_order"] = [str(x) for x in z["band_order_in"]]
        _, swcls, _ = extent(z["b3swap_field"], z["b3swap_boundary"])
    rec["b3_agree_stated"] = float((b3cls == cls8).mean())
    rec["b3_agree_reversed"] = float((swcls == cls8).mean())
    rec["D_all_windows"] = int((b3cls != cls8).sum())
    # populations in the one-row mosaic's flat order
    chip = np.broadcast_to(np.arange(n)[:, None, None], t8.shape)
    seamw = np.broadcast_to(seam[:, None, None], t8.shape)
    P = {"n_chips": n, "rep": {}}
    for name, tt in (("graded", t8), ("fb", tfb)):
        pop = tt >= 0
        fp = flat(pop)
        k4 = flat(pop & ~zero_win)
        P[name] = {"conf": flat(conf8)[fp], "err": flat(cls8 != tt)[fp], "var": flat(var8)[fp],
                   "rare": flat(rare)[fp], "chip": flat(chip)[fp], "seam": flat(seamw)[fp],
                   "a4": {"a": flat(cls8)[k4], "b": flat(b3cls)[k4], "t": flat(tt)[k4]}}
    if var_alt is not None:
        P["graded"]["var_alt"] = flat(var_alt)[flat(t8 >= 0)]
    for name, cf, cl, tt in (("windows_16px", conf16, cls16, t16), ("three_class_8px", conf3, cls3, t3),
                             ("lenient_extent_8px", conf8, cls8, tlen)):
        q = tt >= 0
        e = (cl != tt)[q]
        P["rep"][name] = {"N": int(q.sum()), "errors": int(e.sum()), "theta": float(e.mean()), "auroc": auroc(-cf[q], e)}
    log(f"{c}: read and windowed in {time.time() - t:.0f} s")
    return P, rec


def pool(Ps):
    """The four countries' populations concatenated; chips renumbered; each window keeps its country's controls."""
    out = {"n_chips": sum(Ps[c]["n_chips"] for c in COUNTRIES)}
    for name in ("graded", "fb"):
        off, parts = 0, {k: [] for k in ("conf", "err", "var", "rare", "chip", "seam")}
        a4 = {k: [] for k in "abt"}
        for c in COUNTRIES:
            P = Ps[c][name]
            for k in parts:
                parts[k].append(P[k] + off if k == "chip" else P[k])
            for k in "abt":
                a4[k].append(P["a4"][k])
            off += Ps[c]["n_chips"]
        out[name] = {k: np.concatenate(v) for k, v in parts.items()}
        out[name]["a4"] = {k: np.concatenate(v) for k, v in a4.items()}
    out["rep"] = {}
    return out


def bootstrap(P, n_chips, best_control, nb):
    rng = np.random.default_rng(BOOT_SEED)
    prep_c = boot_auroc_prep(-P["conf"], P["err"])
    prep_b = boot_auroc_prep(P[{"ctl_pixel_variance": "var", "ctl_class_rarity": "rare"}[best_control]], P["err"])
    au, ld = [], []
    for _ in range(nb):
        mult = np.bincount(rng.integers(0, n_chips, n_chips), minlength=n_chips).astype(np.float64)
        w = mult[P["chip"]]
        x = boot_auroc(prep_c, w)
        au.append(x)
        ld.append(x - boot_auroc(prep_b, w))
    return {"resamples": nb, "seed": BOOT_SEED, "level": 0.9, "by": "chip",
            "auroc": [float(np.quantile(au, 0.05)), float(np.quantile(au, 0.95))],
            "lead_over_best_control": [float(np.quantile(ld, 0.05)), float(np.quantile(ld, 0.95))]}


def graded_population(name, P, n_chips):
    res = {}
    t = time.time()
    ctl = {"ctl_pixel_variance": P["var"], "ctl_class_rarity": P["rare"]}
    a1 = arm1_stats(P["conf"], P["err"], ctl)
    a1["excess_aurc"] = excess_aurc(-P["conf"], P["err"])
    if "var_alt" in P:
        a1["variance_auroc_zero_in_either_image_reading"] = auroc(P["var_alt"], P["err"])
    a1["bootstrap"] = bootstrap(P, n_chips, a1["best_control"], BOOT["pooled" if name == "pooled" else "country"])
    res["arm1"] = a1
    log(f"{name}: arm 1 {time.time() - t:.0f} s  AUROC {a1['auroc']:.6f} capture {a1['capture_10']:.6f} "
        f"lead {a1['lead_over_best_control']:.6f}")
    # arm 2: the exact coverage, and the run's 2,000 draws replicated from their seeds
    N, K = a1["N"], a1["errors"]
    err = P["err"].astype(np.int64)
    a2 = {}
    for B in (100, 300):
        ks, pmf, cv = covered_table(N, K, B)
        exact = float(pmf[cv].sum())
        covk = dict(zip(ks.tolist(), cv.tolist()))
        rep = np.empty(DRAWS, bool)
        for r in range(DRAWS):
            idx = np.random.default_rng(SEED + r).choice(N, B, replace=False)
            rep[r] = covk.get(int(err[idx].sum()), False)
        se = math.sqrt(exact * (1 - exact) / DRAWS)
        a2[str(B)] = {"exact_coverage": exact, "mc_coverage_same_seeds": float(rep.mean()),
                      "mc_minus_exact_in_se": (float(rep.mean()) - exact) / se}
    res["arm2"] = a2
    # arm 3: certify on the run's draws, on an independent set, and exactly for the first level
    t = time.time()
    Z = Zone(P["conf"], err)
    same = draws(N, 300, seed0=SEED)
    indep = draws(N, 300, rng=np.random.default_rng(INDEPENDENT_SEED))
    a3 = {}
    for alpha in (0.05, 0.10):
        a3[f"{alpha:g}"] = {"same_seeds": summarize_certify(certify_many(Z, same, alpha)),
                            "independent_seeds": summarize_certify(certify_many(Z, indep, alpha))}
    pex, info = exact_p_first_level(Z, ALPHA)
    a3["exact_p_certify_any_0.05"] = pex
    a3["p_at_most_half_of_2000_draws_certify"] = float(stats.binom.cdf(DRAWS // 2, DRAWS, pex))
    a3["first_level"] = info
    res["arm3"] = a3
    log(f"{name}: arm 3 {time.time() - t:.0f} s  exact P(certify any) {pex:.4f}; same seeds "
        f"{a3['0.05']['same_seeds']['violation_share']:.4f} / {a3['0.05']['same_seeds']['median_certified_share']}; "
        f"independent {a3['0.05']['independent_seeds']['violation_share']:.4f} / "
        f"{a3['0.05']['independent_seeds']['median_certified_share']}")
    # arm 4: the counts, the exact probabilities and the run's 2,000 draws per cell replicated from their seeds
    a, b_, tt = P["a4"]["a"], P["a4"]["b"], P["a4"]["t"]
    a4 = arm4_counts(a, b_, tt)
    pop4 = np.flatnonzero(a != b_)
    aright = a == tt
    a4["budgets"] = {}
    for n in (100, 300):
        pa, pb, kc = which_map_exact(a4["D"], a4["K_A"], a4["K_B"], n)
        right, wrong = (pa, pb) if a4["better_agreeing"] == "2e" else (pb, pa)
        cnt = {"a": 0, "b": 0, None: 0}
        for r in range(DRAWS):
            i = pop4[np.random.default_rng(SEED + r).choice(pop4.size, n, replace=False)]
            x = int(aright[i].sum())
            cnt["a" if (n - x) <= kc else ("b" if x <= kc else None)] += 1
        mr, mw = (cnt["a"], cnt["b"]) if a4["better_agreeing"] == "2e" else (cnt["b"], cnt["a"])
        a4["budgets"][str(n)] = {"p_right": right, "p_wrong": wrong, "sign_critical": kc,
                                 "mc_same_seeds": {"p_right": mr / DRAWS, "p_wrong": mw / DRAWS}}
    res["arm4"] = a4
    log(f"{name}: arm 4  D {a4['D']} K_A {a4['K_A']} K_B {a4['K_B']}")
    return res


def field_plus_boundary_population(name, P):
    """The other truth reading, reported beside the graded one: arms 1 to 4 on the same windows and draws' rules."""
    ctl = {"ctl_pixel_variance": P["var"], "ctl_class_rarity": P["rare"]}
    a1 = arm1_stats(P["conf"], P["err"], ctl)
    N, K = a1["N"], a1["errors"]
    a2 = {}
    for B in (100, 300):
        ks, pmf, cv = covered_table(N, K, B)
        a2[str(B)] = {"exact_coverage": float(pmf[cv].sum())}
    err = P["err"].astype(np.int64)
    Z = Zone(P["conf"], err)
    a3 = {"same_seeds": summarize_certify(certify_many(Z, draws(N, 300, seed0=SEED), ALPHA)),
          "independent_seeds": summarize_certify(certify_many(Z, draws(N, 300, rng=np.random.default_rng(FB_INDEPENDENT_SEED)), ALPHA))}
    pex, info = exact_p_first_level(Z, ALPHA)
    a3["exact_p_certify_any_0.05"], a3["first_level"] = pex, info
    a4 = arm4_counts(P["a4"]["a"], P["a4"]["b"], P["a4"]["t"])
    a4["budgets"] = {}
    for n in (100, 300):
        pa, pb, kc = which_map_exact(a4["D"], a4["K_A"], a4["K_B"], n)
        right, wrong = (pa, pb) if a4["better_agreeing"] == "2e" else (pb, pa)
        a4["budgets"][str(n)] = {"p_names_2e": pa, "p_names_b3": pb, "p_right": right, "p_wrong": wrong}
    log(f"{name}: field + boundary theta {a1['theta']:.4f} AUROC {a1['auroc']:.4f} lead {a1['lead_over_best_control']:.4f}; "
        f"certify median {a3['same_seeds']['median_certified_share']} (independent "
        f"{a3['independent_seeds']['median_certified_share']}); D {a4['D']} K_A {a4['K_A']} K_B {a4['K_B']}")
    return {"arm1": a1, "arm2": a2, "arm3": a3, "arm4": a4}


def verdicts(pr, draws_key="same_seeds"):
    """The page's bars applied to this route's own numbers (P2's Monte Carlo is the replicated draws)."""
    v = {}
    meet = [c for c in COUNTRIES if pr[c]["arm1"]["errors"] >= 200 and pr[c]["arm1"]["auroc"] >= 0.70
            and pr[c]["arm1"]["capture_10"] >= 0.25 and pr[c]["arm1"]["lead_over_best_control"] >= 0.05]
    v["P1"] = {"countries_meeting": meet, "holds": len(meet) >= 3}
    cells = {f"{p}:{B}": pr[p]["arm2"][B] for p in POPS for B in ("100", "300")}
    if all("mc_coverage_same_seeds" in x for x in cells.values()):
        ok = all(x["exact_coverage"] >= 0.95 and abs(x["mc_minus_exact_in_se"]) <= 3 for x in cells.values())
    else:
        ok = all(x["exact_coverage"] >= 0.95 for x in cells.values())
    v["P2"] = {"min_exact_coverage": min(x["exact_coverage"] for x in cells.values()), "holds": ok}
    a3 = {p: (pr[p]["arm3"]["0.05"] if "0.05" in pr[p]["arm3"] else pr[p]["arm3"])[draws_key] for p in POPS}
    viol = {p: a3[p]["violation_share"] for p in POPS}
    v["P3a"] = {"violation_share": viol, "holds": all(x <= 0.115 for x in viol.values())}
    med = {c: a3[c]["median_certified_share"] for c in COUNTRIES}
    m3 = [c for c in COUNTRIES if med[c] >= 0.25]
    v["P3b"] = {"median_certified_share": med, "countries_meeting": m3, "holds": len(m3) >= 2}
    wrong = {f"{p}:{n}": pr[p]["arm4"]["budgets"][n]["p_wrong"] for p in POPS for n in ("100", "300")}
    v["P4a"] = {"max_p_wrong": max(wrong.values()), "holds": all(x <= 0.025 for x in wrong.values())}
    right = {c: pr[c]["arm4"]["budgets"]["100"]["p_right"] for c in COUNTRIES}
    m4 = [c for c in COUNTRIES if right[c] >= 0.8]
    v["P4b"] = {"p_right": right, "countries_meeting": m4, "holds": len(m4) >= 3}
    return v


def git_commit():
    try:
        c = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "exp/exp110_crosscheck.py"],
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
    ap.add_argument("--data", default=os.environ.get("FTW_DATA"), help="the local FTW data directory")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if not a.data or not os.path.isdir(a.data):
        sys.exit("--data: the local FTW data directory (its manifest_files.csv lists every input's sha256)")
    M = manifest(a.data)
    hashes, Ps, recs = {}, {}, {}
    for c in COUNTRIES:
        Ps[c], recs[c] = country(a.data, c, hashes)
    Ps["pooled"] = pool(Ps)
    hashes["ftwq/index/index.parquet"] = sha256(os.path.join(a.data, "ftwq/index/index.parquet"))
    npz = {rel: h for rel, h in sorted(hashes.items()) if "/s2_images/" not in rel}
    s2h = sorted((rel, h) for rel, h in hashes.items() if "/s2_images/" in rel)
    commit, dirty = git_commit()
    out = {
        "experiment": "exp110 second route: the graded numbers recomputed from the local FTW data with numpy, scipy "
                      "and rasterio, without exp/exp110_ftw_field_map.py or oe_inferencex",
        "prereg_page": "docs/plan/ftw_fields.md",
        "script": "exp/exp110_crosscheck.py",
        "script_sha256": sha256(os.path.abspath(__file__)),
        "commit": commit, "script_uncommitted": dirty,
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
                        "machine": platform.machine()},
        "config": {"patch": PATCH, "alpha": ALPHA, "delta": DELTA, "draws": DRAWS,
                   "draw_seed": "seed 110 + r for draw r, as the run's draws", "independent_certify_seed": INDEPENDENT_SEED,
                   "field_plus_boundary_independent_seed": FB_INDEPENDENT_SEED,
                   "bootstrap": {"resamples": BOOT, "seed": BOOT_SEED, "level": 0.9, "by": "chip"}},
        "inputs": {
            "npz_and_index_sha256": npz,
            "npz_and_index_match_manifest": all(M.get(rel) == h for rel, h in npz.items()),
            "s2_files": {"count": len(s2h), "all_match_manifest": all(M.get(r) == h for r, h in s2h),
                         "sha256_of_their_sha256_in_sorted_path_order":
                             hashlib.sha256("".join(h for _, h in s2h).encode()).hexdigest()},
            "manifest_files_csv_sha256": sha256(os.path.join(a.data, "manifest_files.csv"))},
        "countries": recs, "populations": {}, "field_plus_boundary": {}}
    log("inputs match the manifest:", out["inputs"]["npz_and_index_match_manifest"], out["inputs"]["s2_files"]["all_match_manifest"])
    for name in POPS:
        P = Ps[name]
        res = graded_population(name, P["graded"], P["n_chips"])
        rep = dict(P["rep"])
        keep = ~P["graded"]["seam"]
        e_ns = P["graded"]["err"][keep]
        rep["without_seam_chips"] = {"N": int(keep.sum()), "errors": int(e_ns.sum()), "theta": float(e_ns.mean()),
                                     "auroc": auroc(-P["graded"]["conf"][keep], e_ns)}
        res["reported"] = rep
        out["populations"][name] = res
        out["field_plus_boundary"][name] = field_plus_boundary_population(name, P["fb"])
    out["verdicts"] = verdicts(out["populations"])
    out["verdicts_independent_certify_draws"] = {k: verdicts(out["populations"], "independent_seeds")[k] for k in ("P3a", "P3b")}
    out["field_plus_boundary_verdicts"] = verdicts(out["field_plus_boundary"])
    out["field_plus_boundary_verdicts_independent_certify_draws"] = {
        k: verdicts(out["field_plus_boundary"], "independent_seeds")[k] for k in ("P3a", "P3b")}
    out["notes"] = [
        "Graded reading: truth = the majority of the 2-class mask, which is field only (mask2 == (mask3 == 1) in every "
        "country). The field + boundary reading (truth = the majority of mask3 in {1, 2}) is reported beside it, as the "
        "owner ruled on 14 October 2026; it was computed after the labels had been read with the maps.",
        "The bootstrap here uses fewer resamples than the run (400 per country, 150 pooled, against 2,000), so its "
        "interval ends agree with the run's to Monte Carlo error only.",
        "The tie order in certify is the package's: windows tied in confidence enter the zone by index, the one-row "
        "mosaic's raster order."]
    out["seconds"] = round(time.time() - T0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=1)
        f.write("\n")
    log("verdicts", json.dumps({k: v["holds"] for k, v in out["verdicts"].items()}))
    log("field + boundary verdicts", json.dumps({k: v["holds"] for k, v in out["field_plus_boundary_verdicts"].items()}))
    log("wrote", os.path.relpath(a.out, ROOT))
    return out


if __name__ == "__main__":
    main()
