"""exp113's second route (docs/plan/planteur_transfer.md, Independent check before anything is recorded): every graded
number of the run, and the reported ones, recomputed from the PLANTEUR labels and embeddings with numpy, scipy and
scikit-learn only.

Written for the independent audit of 22 October 2026 by an agent that did not write the run script. It does not import
exp/exp113_planteur_transfer.py or oe_inferencex, and it was written without reading the run's outputs. scikit-learn
is used because the map IS StandardScaler + LogisticRegression; everything after the fit is this file's own.
Re-derived from the page:
  configurations: loio:<territory> for Guadeloupe, Martinique, Guyane, Réunion and Mayotte (splits/loio.json:
    folds[territory]["train"] as the fit patches in file order, folds[territory]["test"] as the test patches in file
    order) and default (metadata.geojson: Fold 1 the fit patches, Fold 5 the test patches, ascending ID_PATCH);
  the fit set: each fit patch's non-void pixels row-major, the patches in that order, numbered globally;
    numpy.random.default_rng(113).choice(total, 100000, replace=False) per configuration, the same pixels for the three
    embeddings, sorted ascending (the fit rows are in that order);
  the map: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) on the float32 vectors as stored; every
    test pixel's eight class probabilities (a class absent from the fit set has probability 0);
  windows: 4 x 4 px, 32 x 32 per patch; confidence the mean of the 16 pixels' top probability; class the majority of
    the pixels' argmax, a tied majority to the tied class whose voting pixels have the larger summed top probability
    (equal sums: the lower class index); truth the majority of the non-void labels, a window with more than 8 void
    pixels or a tied label majority left out and counted; error: window class != truth;
  population: the scored windows of the test patches, in test-list order, row-major inside each patch (the zone's
    tie order); the controls: minus the population's share of the window's predicted class, the Euclidean distance of
    the window's mean vector (its 16 pixels', raw as stored) from the fit set's mean vector, and the boundary
    indicator (share of the 8 neighbouring windows of the same patch with another class, edge padding);
  edge band: a pixel is in the band when a pixel of another predicted class lies within one row and one column inside
    its patch; a window takes its pixels' majority (8 of 16 against 8: no condition, counted).
Its own implementations (exp110/exp112 crosscheck's): AUROC by average ranks, capture in expectation over ties, the
patch bootstrap by weighted ranks, the exact coverage by scipy's hypergeometric tails, certify's prefix rule and
standard level cut at 400 labels by hypergeometric p-values with the review-set guard, the exact probability that the
first testable level passes, and the which-map test by enumeration (the sign test while 5 n <= D, else the union of
two exact 97.5% intervals, each widened to hold its sample share; a census when n = D), with integer sign-test
critical values.
Its draws are its own (seeds in `config`), not the run's: arm 2's Monte Carlo, arm 3's shares and the bootstrap agree
with the run's to Monte Carlo error only.

    uv run --with scikit-learn==1.9.1 python -I exp/exp113_crosscheck.py --data <PLANTEUR directory> [--out JSON]
    ... --smoke    loio:Guadeloupe and default only, their first 3 fit and first 2 test patches, 2,000 fit pixels,
                   5 draws (a code check: no number from it means anything)

Streams patch by patch: each embedding file is read twice (once to gather fit pixels, once to predict), with its
sha256 taken on the first read; about 110 GB read in all. Expected on one 32-CPU node: 30 to 60 minutes and a peak
resident memory of 2 to 3 GB (estimated from timings on the local smoke data, before any full run). Writes
aggregates only (per configuration, embedding and territory), the inputs' digests and the fit sets' sha256; no
patch, window, pixel or coordinate list.

Checks beside the numbers: the split files against metadata.geojson; label values; array shapes and dtypes; finite
fit vectors; each fitted map's in-sample agreement on crop pixels with the labels as given and rotated 180 degrees
(the embedding and label grids are aligned when the first is far above the second); probabilities summing to 1; a
Monte Carlo of arm 2's coverage and of arm 4's sign test beside their enumerations.
"""
import argparse
import hashlib
import io
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy
import sklearn
from scipy import ndimage, stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out", "exp113_crosscheck.json")
TERRITORIES = ("Guadeloupe", "Martinique", "Guyane", "Réunion", "Mayotte")
CONFIGS = tuple(f"loio:{t}" for t in TERRITORIES) + ("default",)
EMBS = {"v1.3": ("EMB_OLMOEARTH_V1_3", "OE13", 128),
        "tessera_v2": ("EMB_TESSERA_V2", "TESSERA", 128),
        "alphaearth": ("EMB_AEF", "AEF", 64)}
GRADED = "v1.3"
PAIRS = (("v1.3", "tessera_v2"), ("v1.3", "alphaearth"), ("tessera_v2", "alphaearth"))
N_CLASSES, VOID, SIDE, PATCH = 8, 255, 128, 4
G = SIDE // PATCH
CLASS_NAMES = ("Background", "Meadow", "Fruits, vegetables, flowers", "Orchard", "Sugarcane", "Banana", "Pineapple",
               "Tropical tuber")
FIT_SIZE, FIT_SEED = 100000, 113
ALPHAS, DELTA = (0.05, 0.10), 0.10
GRID = tuple(round(j / 20, 2) for j in range(1, 21))      # the package's default certify grid, 0.05 to 1.0
CERT_BUDGET = 400
COVER_B = (100, 400)
WHICH_N = (100, 400)
DRAWS = 2000
BOOT = 2000
# this audit's own draws: numpy default_rng([seed, configuration index (CONFIGS order), ...])
SEED_COVER100, SEED_CERT, SEED_ARM4_MC, SEED_BOOT = 2026102231, 2026102232, 2026102233, 2026102234
REVIEW_SET_SIGMAS = 4.0
SIGN_SHARE = 5
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20      # bytes on macOS, KiB on Linux


class Reader:
    """np.load of a file's bytes, with each file's sha256 taken the first time it is read."""

    def __init__(self, data):
        self.data = data
        self.digests = {}
        self.bytes_read = 0

    def load(self, rel):
        with open(os.path.join(self.data, rel), "rb") as f:
            buf = f.read()
        self.bytes_read += len(buf)
        if rel not in self.digests:
            self.digests[rel] = hashlib.sha256(buf).hexdigest()
        return np.load(io.BytesIO(buf), allow_pickle=False)

    def directory_digest(self, prefix):
        """One sha256 over the sorted 'name sha256' lines of the files read under `prefix`, and their count."""
        items = sorted((k, v) for k, v in self.digests.items() if k.startswith(prefix + "/"))
        h = hashlib.sha256("".join(f"{k} {v}\n" for k, v in items).encode()).hexdigest()
        return {"files": len(items), "sha256_of_sorted_name_sha256_lines": h}


# ------------------------------------------------------------------ splits and configurations
def read_configs(data, smoke):
    with open(os.path.join(data, "splits", "loio.json"), encoding="utf-8") as f:
        loio = json.load(f)
    with open(os.path.join(data, "metadata.geojson"), encoding="utf-8") as f:
        meta = json.load(f)
    props = [ft["properties"] for ft in meta["features"]]
    ids = [int(p["ID_PATCH"]) for p in props]
    island = {int(p["ID_PATCH"]): p.get("Island") for p in props}
    fold = {int(p["ID_PATCH"]): int(p["Fold"]) for p in props}
    chk = {"metadata_patches": len(ids), "metadata_ids_unique": len(set(ids)) == len(ids),
           "loio_rotation": loio.get("rotation"), "loio_rotation_matches_page": list(loio.get("rotation", [])) == list(TERRITORIES),
           "loio_excluded_from_rotation": loio.get("excluded_from_rotation"),
           "fold_counts": {str(k): int(v) for k, v in zip(*np.unique(list(fold.values()), return_counts=True))},
           "island_counts": {str(k): int(v) for k, v in zip(*np.unique([str(x) for x in island.values()], return_counts=True))},
           "per_configuration": {}}
    cfgs = []
    for t in TERRITORIES:
        fd = loio["folds"][t]
        tr, va, te = [int(x) for x in fd["train"]], [int(x) for x in fd.get("val", [])], [int(x) for x in fd["test"]]
        c = {"name": f"loio:{t}", "territory": t, "train": tr, "test": te}
        chk["per_configuration"][c["name"]] = {
            "train": len(tr), "val": len(va), "test": len(te),
            "test_equals_metadata_island": set(te) == {i for i in ids if island[i] == t},
            "train_from_held_out_territory": sum(island.get(i) == t for i in tr),
            "train_test_overlap": len(set(tr) & set(te)), "train_val_overlap": len(set(tr) & set(va)),
            "val_test_overlap": len(set(va) & set(te)), "train_unknown_ids": sum(i not in island for i in tr),
            "train_duplicates": len(tr) - len(set(tr)), "test_duplicates": len(te) - len(set(te)),
            "saint_martin_in_train": sum(island.get(i) == "Saint-Martin" for i in tr)}
        cfgs.append(c)
    tr = sorted(i for i in ids if fold[i] == 1)
    te = sorted(i for i in ids if fold[i] == 5)
    cfgs.append({"name": "default", "territory": None, "train": tr, "test": te})
    chk["per_configuration"]["default"] = {"train": len(tr), "test": len(te), "train_test_overlap": len(set(tr) & set(te)),
                                           "test_by_island": {str(k): int(v) for k, v in zip(*np.unique(
                                               [str(island[i]) for i in te], return_counts=True))}}
    if smoke:
        cfgs = [c for c in cfgs if c["name"] in ("loio:Guadeloupe", "default")]
        for c in cfgs:
            c["train"], c["test"] = c["train"][:3], c["test"][:2]
    return cfgs, chk


# ------------------------------------------------------------------ windows
def blocks(a):
    """(128, 128) -> (32, 32, 16): each 4 x 4 window's pixels."""
    return a.reshape(G, PATCH, G, PATCH).transpose(0, 2, 1, 3).reshape(G, G, PATCH * PATCH)


def window_truth(lab):
    """-1: more than half void; -2: tied label majority; else the majority class."""
    lw = blocks(lab)
    void = (lw == VOID).sum(-1)
    cnt = np.stack([(lw == c).sum(-1) for c in range(N_CLASSES)], -1)
    top = cnt.max(-1, keepdims=True)
    tied = (cnt == top).sum(-1) > 1
    t = cnt.argmax(-1).astype(np.int8)
    t = np.where(tied, np.int8(-2), t)
    return np.where(void > PATCH * PATCH // 2, np.int8(-1), t).astype(np.int8)


def window_map(P8):
    """Per-pixel probabilities (16384, 8) -> window confidence, class, tie count; pixel hard class (128, 128)."""
    top = P8.max(1).reshape(SIDE, SIDE)
    hard = P8.argmax(1).reshape(SIDE, SIDE)
    hw, tw = blocks(hard), blocks(top)
    conf = tw.mean(-1)
    cnt = np.stack([(hw == c).sum(-1) for c in range(N_CLASSES)], -1)
    wsum = np.stack([np.where(hw == c, tw, 0.0).sum(-1) for c in range(N_CLASSES)], -1)
    topc = cnt.max(-1, keepdims=True)
    tied = (cnt == topc).sum(-1) > 1
    cls = np.where(tied, np.where(cnt == topc, wsum, -np.inf).argmax(-1), cnt.argmax(-1))
    return conf, cls.astype(np.int8), int(tied.sum()), hard


def boundary_indicator(cls):
    pad = np.pad(cls, 1, mode="edge")
    nb = np.zeros(cls.shape)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += pad[1 + di:1 + di + G, 1 + dj:1 + dj + G] != cls
    return nb / 8.0


def band_condition(hard):
    """The 1-px class-edge band inside the patch, and each window's majority of it (-1: 8 against 8)."""
    mx = ndimage.maximum_filter(hard, size=3, mode="nearest")
    mn = ndimage.minimum_filter(hard, size=3, mode="nearest")
    band = mx != mn
    s = blocks(band).sum(-1)
    cond = np.where(s > 8, 1, np.where(s < 8, 0, -1)).astype(np.int8)
    return int(band.sum()), cond


# ------------------------------------------------------------------ ranking statistics (exp110/exp112 crosscheck's)
def auroc(u, e):
    u = np.asarray(u, np.float64)
    e = np.asarray(e, bool)
    ne, nc = int(e.sum()), int((~e).sum())
    if ne == 0 or nc == 0:
        return None
    r = stats.rankdata(u)                     # average ranks: ties counted half
    return float((r[e].sum() - ne * (ne + 1) / 2.0) / (ne * nc))


def groups_desc(u, e):
    vals, inv = np.unique(-np.asarray(u, np.float64), return_inverse=True)   # most suspect first
    tot = np.bincount(inv, minlength=vals.size).astype(np.float64)
    err = np.bincount(inv, weights=np.asarray(e, np.float64), minlength=vals.size)
    return tot, err


def capture(u, e, budget=0.10):
    e = np.asarray(e, bool)
    if not e.any():
        return None
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
    if wp == 0 or wn == 0:
        return np.nan
    return ((w[es] * mid[es]).sum() - wp * wp / 2.0) / (wp * wn)


def ranking(conf, err, controls):
    out = {"N": int(err.size), "errors": int(err.sum()), "theta": float(err.mean()) if err.size else None,
           "auroc": auroc(-conf, err), "capture_10": capture(-conf, err)}
    out["controls"] = {k: {"auroc": auroc(v, err), "capture_10": capture(v, err)} for k, v in controls.items()}
    have = {k: v["auroc"] for k, v in out["controls"].items() if v["auroc"] is not None}
    best = max(have, key=have.get) if have else None
    out["best_control"] = best
    out["lead_over_best_control"] = (out["auroc"] - have[best]) if (best and out["auroc"] is not None) else None
    k = max(1, int(round(0.1 * err.size)))
    out["oracle_capture_10"] = min(1.0, k / err.sum()) if err.any() else None
    if err.any() and (~err).any():
        m = float(np.median(conf[~err]))
        out["median_confidence_correct"] = m
        out["confident_errors"] = int((conf[err] >= m).sum())
        out["confident_error_share"] = float((conf[err] >= m).mean())
    out["confidence_ties_beyond_first"] = int(conf.size - np.unique(conf).size)
    return out


# ------------------------------------------------------------------ arm 2: the exact hypergeometric interval
def covered_table(N, K, B, a=0.025):
    """Error counts k of a simple random sample of B, their probabilities, and whether the exact equal-tailed 95%
    interval at k (every K' whose two tails exceed 0.025) holds K. The package widens its interval to hold k/B; that
    can add K/N only when K is excluded on the side k/B lies on, which a tail of 0.025 rules out short of a census."""
    ks = np.arange(max(0, B - (N - K)), min(B, K) + 1)
    up = stats.hypergeom.sf(ks - 1, N, K, B)        # P(X >= k | K)
    lo = stats.hypergeom.cdf(ks, N, K, B)           # P(X <= k | K)
    pmf = stats.hypergeom.pmf(ks, N, K, B)
    cov = (up > a) & (lo > a)
    return ks, pmf, cov


Z95 = 1.959963984540054


def wilson_coverage(N, K, B):
    """The exact coverage of the finite-population Wilson interval (the score form at n_eff = B (N - 1) / (N - B)),
    estimate.exact_coverage_srs's default interval, reported beside the exact interval's."""
    if B >= N:
        return 1.0
    ks = np.arange(max(0, B - (N - K)), min(B, K) + 1)
    pmf = stats.hypergeom.pmf(ks, N, K, B)
    p = ks / B
    ne = B * (N - 1) / (N - B) if N > 1 else float(B)
    d = 1 + Z95 ** 2 / ne
    centre = (p + Z95 ** 2 / (2 * ne)) / d
    half = Z95 * np.sqrt(p * (1 - p) / ne + Z95 ** 2 / (4 * ne ** 2)) / d
    lo, hi = centre - half, centre + half
    lo, hi = np.where(lo < 1e-12, 0.0, lo), np.where(hi > 1 - 1e-12, 1.0, hi)
    th = K / N
    return float(pmf[(lo <= th) & (th <= hi)].sum())


# ------------------------------------------------------------------ arm 3: certify (prefix rule, standard cut)
def b_min(alpha, delta=DELTA):
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))


def levels(N, budget, alpha):
    cmin = b_min(alpha) / budget
    cov = [c for c in GRID if c >= cmin - 1e-12]
    sizes = [max(1, int(round(c * N))) for c in cov]
    K0 = [math.floor(alpha * s) + 1 for s in sizes]
    return np.array(cov), np.array(sizes, np.int64), np.array(K0, np.int64)


class Zone:
    def __init__(self, conf, err):
        N = conf.size
        self.N = N
        idx = np.arange(N)
        self.order = np.lexsort((idx, -conf))              # descending confidence, ties by population index
        self.pos = np.empty(N, np.int64)
        self.pos[self.order] = idx
        self.cum = np.concatenate([[0], np.cumsum(err[self.order])])
        o = np.lexsort((np.random.default_rng(0).random(N), -conf))   # the review-set guard's tie order
        r = np.empty(N)
        r[o] = idx
        self.pct = (r + 0.5) / N
        self.err = err


def certify_many(Z, samples, alpha, budget):
    cov, sizes, K0 = levels(Z.N, budget, alpha)
    ndraw = samples.shape[0]
    if cov.size == 0:
        z = np.zeros(ndraw, bool)
        return {"share": np.zeros(ndraw), "certified": z, "refused": z, "viol": z, "zerr": np.full(ndraw, np.nan),
                "first_level_passed": z, "levels": {"coverage": [], "sizes": [], "K0": [], "b_min": b_min(alpha)}}
    S = np.sort(Z.pos[samples], axis=1)
    W = Z.err[Z.order][S]
    cw = np.concatenate([np.zeros((ndraw, 1)), np.cumsum(W, 1)], 1)
    b = np.stack([np.searchsorted(row, sizes, side="left") for row in S])
    k = np.take_along_axis(cw, b, 1).astype(np.int64)
    p = np.where(b == 0, 1.0, stats.hypergeom.cdf(k, sizes[None, :], K0[None, :], b))
    p = np.where(K0[None, :] > sizes[None, :], 0.0, p)
    acc = p <= DELTA
    first_fail = np.where(acc.all(1), acc.shape[1], np.argmin(acc, axis=1))
    n = budget
    fpc = max(Z.N - n, 0) / (Z.N - 1)
    thr = min(0.95, 0.5 + REVIEW_SET_SIGMAS * math.sqrt(1 / 12) * math.sqrt(fpc / n))
    refused = Z.pct[samples].mean(1) > thr + 1e-12
    share = np.zeros(ndraw)
    zerr = np.full(ndraw, np.nan)
    ok = (first_fail > 0) & ~refused
    best = first_fail - 1
    share[ok] = cov[best[ok]]
    nz = sizes[best[ok]]
    zerr[ok] = Z.cum[nz] / nz
    viol = ok & (np.nan_to_num(zerr) > alpha)
    return {"share": share, "certified": ok, "refused": refused, "viol": viol, "zerr": zerr,
            "first_level_passed": acc[:, 0],
            "levels": {"coverage": cov.tolist(), "sizes": sizes.tolist(), "K0": K0.tolist(), "b_min": b_min(alpha)}}


def summarize_certify(res):
    sh = res["share"]
    vals, cnts = np.unique(sh, return_counts=True)
    zr = res["zerr"][res["certified"]]
    return {"draws": int(sh.size), "p_certify_any": float(res["certified"].mean()),
            "p_first_level_passes_guard_aside": float(res["first_level_passed"].mean()),
            "refused_as_review_set": int(res["refused"].sum()),
            "violations": int(res["viol"].sum()), "violation_share": float(res["viol"].mean()),
            "median_certified_share": float(np.median(sh)), "mean_certified_share": float(sh.mean()),
            "certified_share_counts": {f"{v:g}": int(c) for v, c in zip(vals, cnts)},
            "zone_true_error_median": float(np.median(zr)) if zr.size else None,
            "levels": res["levels"]}


def exact_p_first_level(Z, alpha, budget):
    """P(the smallest testable level passes) = P(certify any zone), the review-set guard aside."""
    cov, sizes, K0 = levels(Z.N, budget, alpha)
    if cov.size == 0:
        return 0.0, None
    n0, k0 = int(sizes[0]), int(K0[0])
    Kz = int(Z.cum[n0])
    if k0 > n0:
        return None, {"note": "null empty at the first level"}
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


def zone_true_errors(Z, alpha, budget):
    cov, sizes, _ = levels(Z.N, budget, alpha)
    return {f"{c:g}": float(Z.cum[s] / s) for c, s in zip(cov, sizes)}


# ------------------------------------------------------------------ arm 4: which map, by enumeration
_SIGN = {}


def sign_crit(m):
    """The largest k with 2 P(Bin(m, 1/2) <= k) < 0.05, or -1; integer arithmetic (0.05 = 1/20)."""
    if m not in _SIGN:
        total, cum, c, k = 1 << m, 0, 1, -1
        for i in range(m // 2 + 1):
            cum += c
            if 40 * cum < total:
                k = i
            else:
                break
            c = c * (m - i) // (i + 1)
        _SIGN[m] = k
    return _SIGN[m]


def interval_ends(D, n, a=0.0125):
    """For every count x of n drawn from D: the exact interval's ends at 97.5% (each tail 0.0125) over K'/D,
    widened to hold x/n."""
    Ks = np.arange(0, D + 1)
    lo, hi = np.empty(n + 1), np.empty(n + 1)
    for x in range(n + 1):
        ok = (stats.hypergeom.sf(x - 1, D, Ks, n) > a) & (stats.hypergeom.cdf(x, D, Ks, n) > a)
        lo[x], hi[x] = min(Ks[ok].min() / D, x / n), max(Ks[ok].max() / D, x / n)
    return lo, hi


def which_map(D, KA, KB, n):
    """P(names A), P(names B), the rule: n labels drawn without replacement among the D differing windows, KA of
    which only A has right, KB only B, the rest neither."""
    if n >= D:
        return float(KA > KB), float(KB > KA), "census"
    rule = "sign" if SIGN_SHARE * n <= D else "interval"
    KN = D - KA - KB
    if rule == "interval":
        lo, hi = interval_ends(D, n)
    pa = pb = 0.0
    xas = np.arange(max(0, n - (D - KA)), min(n, KA) + 1)
    for xa, w in zip(xas, stats.hypergeom.pmf(xas, D, KA, n)):
        if w < 1e-300:
            continue
        rem = n - int(xa)
        xb = np.arange(max(0, rem - KN), min(rem, KB) + 1)
        pxb = stats.hypergeom.pmf(xb, D - KA, KB, rem)
        if rule == "sign":
            crit = np.array([sign_crit(int(xa) + int(y)) for y in xb])
            na, nb = xb <= crit, xa <= crit                 # crit < m / 2: the smaller count names the other map
        else:
            na, nb = lo[xa] > hi[xb], lo[xb] > hi[xa]
        pa += w * pxb[na].sum()
        pb += w * pxb[nb].sum()
    return float(pa), float(pb), rule


def which_map_mc(a_right, b_right, n, rng, draws):
    """The sign-test verdict on simulated draws (a check of the enumeration)."""
    D = a_right.size
    na = nb = 0
    for _ in range(draws):
        i = rng.choice(D, n, replace=False)
        xa, xb = int(a_right[i].sum()), int(b_right[i].sum())
        c = sign_crit(xa + xb)
        na += xb <= c
        nb += xa <= c
    return na / draws, nb / draws


def arm4(a, b, t, rng, draws):
    diff = a != b
    D = int(diff.sum())
    KA, KB = int((diff & (a == t)).sum()), int((diff & (b == t)).sum())
    out = {"N": int(a.size), "D": D, "K_A": KA, "K_B": KB, "both_wrong_where_differ": D - KA - KB,
           "agreement_A": float((a == t).mean()), "agreement_B": float((b == t).mean()),
           "better_agreeing": "A" if KA > KB else ("B" if KB > KA else "tie"), "budgets": {}}
    ar, br = (a == t)[diff], (b == t)[diff]
    for n in WHICH_N:
        if D == 0:
            out["budgets"][str(n)] = None
            continue
        nn = min(n, D)
        pa, pb, rule = which_map(D, KA, KB, nn)
        right, wrong = (pa, pb) if KA > KB else ((pb, pa) if KB > KA else (None, max(pa, pb)))
        cell = {"labels": nn, "p_names_A": pa, "p_names_B": pb, "p_right": right, "p_wrong": wrong, "rule": rule}
        if rule == "sign":
            ma, mb = which_map_mc(ar, br, nn, rng, draws)
            cell["mc_check"] = {"draws": draws, "p_names_A": ma, "p_names_B": mb}
        out["budgets"][str(n)] = cell
    return out


def miou(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    present = union > 0
    return {"miou_8_absent_0": float(iou.mean()),
            "miou_present_classes": float(iou[present].mean()) if present.any() else None,
            "classes_present": int(present.sum()), "iou": {CLASS_NAMES[c]: float(iou[c]) for c in range(N_CLASSES)},
            "accuracy": float(inter.sum() / cm.sum()) if cm.sum() else None}


# ------------------------------------------------------------------ the fit set
def draw_fit(cfg, labels, size):
    """The configuration's fit pixels: (patch ids, flat row-major pixel indices, non-void ranks), in ascending order
    of the global index, and the draw's hashes."""
    tr = cfg["train"]
    nv = [np.flatnonzero(labels[p].ravel() != VOID) for p in tr]
    cnt = np.array([x.size for x in nv], np.int64)
    end = np.cumsum(cnt)
    start = end - cnt
    total = int(end[-1])
    g = np.random.default_rng(FIT_SEED).choice(total, size, replace=False)
    gs = np.sort(g)
    pos = np.searchsorted(end, gs, side="right")
    rank = gs - start[pos]
    pid = np.array(tr, np.int64)[pos]
    flat = np.array([nv[i][r] for i, r in zip(pos.tolist(), rank.tolist())], np.int64)
    # the draw order too (before sorting), for the alternative reading of the hash
    pos_u = np.searchsorted(end, g, side="right")
    flat_u = np.array([nv[i][r] for i, r in zip(pos_u.tolist(), (g - start[pos_u]).tolist())], np.int64)
    h = lambda a, b: hashlib.sha256(np.asarray(a, "<i8").tobytes() + np.asarray(b, "<i8").tobytes()).hexdigest()
    info = {"total_nonvoid_pixels": total, "size": size,
            "sha256_patch_ids_then_flat_pixel_index_sorted": h(pid, flat),
            "sha256_patch_ids_then_nonvoid_rank_sorted": h(pid, rank),
            "sha256_patch_ids_then_flat_pixel_index_draw_order": h(np.array(tr, np.int64)[pos_u], flat_u),
            "sha256_sorted_global_indices": hashlib.sha256(np.asarray(gs, "<i8").tobytes()).hexdigest(),
            "patches_drawn_from": int(np.unique(pid).size)}
    return pos, flat, info


def fit_map(X, y):
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    lr = pipe[-1]
    conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    info = {"classes": [int(c) for c in lr.classes_],
            "class_counts": {CLASS_NAMES[int(c)]: int((y == c).sum()) for c in lr.classes_},
            "n_iter": [int(x) for x in np.ravel(lr.n_iter_)], "convergence_warnings": len(conv),
            "other_warning_types": sorted({x.category.__name__ for x in w if not issubclass(x.category, ConvergenceWarning)}),
            "seconds": round(time.time() - t, 1), "X_dtype": str(X.dtype)}
    return pipe, info


# ------------------------------------------------------------------ one population's arms
def population_arms(name, store, emb, draws_cover, draws_cert):
    P = store["pop"]
    sw = store["w"][emb]
    conf, cls, truth = sw["conf"][P], sw["cls"][P], store["truth"][P]
    err = cls != truth
    shares = np.bincount(cls, minlength=N_CLASSES) / cls.size
    controls = {"ctl_class_rarity": -shares[cls], "ctl_distance_from_fit_mean": sw["dist_raw"][P],
                "ctl_boundary": sw["bnd"][P]}
    a1 = ranking(conf, err, controls)
    a1["reported_distance_in_standardized_space"] = {"auroc": auroc(sw["dist_std"][P], err),
                                                     "capture_10": capture(sw["dist_std"][P], err)}
    a1["predicted_window_majority_ties"] = int(sw["tied"])
    a1["window_class_counts_predicted"] = {CLASS_NAMES[c]: int(x) for c, x in enumerate(np.bincount(cls, minlength=N_CLASSES))}
    cmw = np.bincount(truth.astype(np.int64) * N_CLASSES + cls, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    res = {"arm1": a1, "miou_pixels_nonvoid": miou(sw["cm_px"]), "miou_windows": miou(cmw)}
    # edge band
    cond = sw["cond"][P]
    eb = {"pixel_share_in_band": sw["band_px"] / sw["px"]}
    for lab, v in (("band", 1), ("interior", 0), ("no_condition_8_of_16", -1)):
        m = cond == v
        eb[lab] = {"windows": int(m.sum()), "errors": int(err[m].sum()),
                   "error_rate": float(err[m].mean()) if m.any() else None}
    eb["window_share_in_band"] = float((cond == 1).mean())
    res["edge_band"] = eb
    # arm 2
    N, K = int(err.size), int(err.sum())
    e64 = err.astype(np.int64)
    a2 = {}
    for B in COVER_B:
        BB = min(B, N)
        ks, pmf, cv = covered_table(N, K, BB)
        exact = float(pmf[cv].sum())
        covk = dict(zip(ks.tolist(), cv.tolist()))
        smp = draws_cover if B == 100 else draws_cert
        rep = np.array([covk.get(int(x), False) for x in e64[smp].sum(1)])
        se = math.sqrt(exact * (1 - exact) / smp.shape[0]) if 0 < exact < 1 else float("nan")
        a2[str(B)] = {"labels": BB, "exact_coverage": exact, "mc_coverage": float(rep.mean()), "mc_draws": int(smp.shape[0]),
                      "mc_minus_exact_in_se": (float(rep.mean()) - exact) / se if se == se and se > 0 else None,
                      "reported_wilson_fpc_exact_coverage": wilson_coverage(N, K, BB)}
    res["arm2"] = a2
    # arm 3
    Z = Zone(conf, e64)
    a3 = {}
    budget = min(CERT_BUDGET, N)
    for alpha in ALPHAS:
        s = summarize_certify(certify_many(Z, draws_cert, alpha, budget))
        pex, info = exact_p_first_level(Z, alpha, budget)
        s["exact_p_certify_any_guard_aside"] = pex
        s["first_level"] = info
        s["zone_true_error_by_level"] = zone_true_errors(Z, alpha, budget)
        a3[f"{alpha:g}"] = s
    res["arm3"] = a3
    log(f"{name} {emb}: N {N} errors {K} theta {a1['theta']:.4f} AUROC {a1['auroc']} capture {a1['capture_10']} "
        f"best {a1['best_control']} lead {a1['lead_over_best_control']}; coverage "
        f"{[round(a2[str(B)]['exact_coverage'], 4) for B in COVER_B]}; certify 0.05 viol {a3['0.05']['violation_share']:.4f} "
        f"any {a3['0.05']['p_certify_any']:.3f} exact {a3['0.05']['exact_p_certify_any_guard_aside']}")
    return res, controls


def bootstrap(store, emb, best_control, controls, seed, nb):
    P = store["pop"]
    sw = store["w"][emb]
    conf, err = sw["conf"][P], sw["cls"][P] != store["truth"][P]
    patch = store["patch"][P]
    n_p = int(store["n_patches"])
    rng = np.random.default_rng(seed)
    pc, pb = boot_auroc_prep(-conf, err), boot_auroc_prep(controls[best_control], err)
    au, ld = np.empty(nb), np.empty(nb)
    for r in range(nb):
        w = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(np.float64)[patch]
        au[r] = boot_auroc(pc, w)
        ld[r] = au[r] - boot_auroc(pb, w)
    return {"resamples": nb, "seed": f"default_rng([{SEED_BOOT}, configuration index])", "level": 0.9, "by": "patch",
            "best_control_fixed": best_control,
            "auroc": [float(np.nanquantile(au, 0.05)), float(np.nanquantile(au, 0.95))],
            "lead_over_best_control": [float(np.nanquantile(ld, 0.05)), float(np.nanquantile(ld, 0.95))]}


# ------------------------------------------------------------------ verdicts
def verdicts(pops, a4, smoke):
    v = {}
    terr = [f"loio:{t}" for t in TERRITORIES if f"loio:{t}" in pops]
    rows, n_und, n_meet, n_a, n_b = {}, 0, 0, 0, 0
    for c in terr:
        r = pops[c][GRADED]["arm1"]
        det = r["errors"] >= 200
        a = bool(det and r["auroc"] is not None and r["capture_10"] is not None
                 and r["auroc"] >= 0.70 and r["capture_10"] >= 0.25)
        b = bool(det and r["lead_over_best_control"] is not None and r["lead_over_best_control"] >= 0.05)
        rows[c] = {"errors": r["errors"], "determined": det, "auroc": r["auroc"], "capture_10": r["capture_10"],
                   "lead": r["lead_over_best_control"], "best_control": r["best_control"], "a": a, "b": b}
        n_und += not det
        n_meet += a and b
        n_a += a
        n_b += b
    v["P1"] = {"territories": rows, "undetermined": n_und, "meeting_a_and_b": n_meet, "meeting_a": n_a, "meeting_b": n_b,
               "holds": bool(n_und < 2 and n_meet >= 4),
               "holds_reading_a_and_b_counted_separately": bool(n_und < 2 and n_a >= 4 and n_b >= 4),
               "note": "holds when at least 4 of the 5 territories are determined (>= 200 window errors) and meet "
                       "(a) and (b) together; two or more undetermined: cannot hold"}
    lo = [pops[c][GRADED]["arm1"]["auroc"] for c in terr]
    if "default" in pops and lo and all(x is not None for x in lo):
        dflt = pops["default"][GRADED]["arm1"]["auroc"]
        m = float(np.mean(lo))
        v["P2"] = {"default_auroc": dflt, "mean_held_out_auroc": m, "default_minus_mean": dflt - m,
                   "territories_in_mean": len(lo), "holds": bool(dflt - m >= 0.05)}
    cells = {f"{c}|{e}|{B}": pops[c][e]["arm2"][B]["exact_coverage"] for c in pops for e in pops[c]
             for B in pops[c][e]["arm2"]}
    wil = {f"{c}|{e}|{B}": pops[c][e]["arm2"][B]["reported_wilson_fpc_exact_coverage"] for c in pops for e in pops[c]
           for B in pops[c][e]["arm2"]}
    v["P3"] = {"cells": len(cells), "min_exact_coverage": min(cells.values()), "argmin": min(cells, key=cells.get),
               "holds": bool(all(x >= 0.95 for x in cells.values())),
               "interval": "the exact hypergeometric interval, the package's interval for a simple random sample",
               "reading_wilson_fpc": {"min_exact_coverage": min(wil.values()), "argmin": min(wil, key=wil.get),
                                      "holds": bool(all(x >= 0.95 for x in wil.values()))}}
    viol = {f"{c}|{e}": pops[c][e]["arm3"]["0.05"]["violation_share"] for c in terr for e in pops[c]}
    v["P4"] = {"populations": len(viol), "violation_share": viol, "max_violation_share": max(viol.values()) if viol else None,
               "holds": bool(viol and all(x <= 0.115 for x in viol.values()))}
    wrong = {}
    for c in terr:
        for pair, r in a4[c].items():
            if r["K_A"] == r["K_B"]:
                continue
            for n, cell in r["budgets"].items():
                if cell is not None:
                    wrong[f"{c}|{pair}|{n}"] = cell["p_wrong"]
    v["P5"] = {"cells": len(wrong), "p_wrong": wrong, "max_p_wrong": max(wrong.values()) if wrong else None,
               "holds": bool(all(x <= 0.025 for x in wrong.values()))}
    if smoke:
        v["note"] = "SMOKE: 2 configurations, 3 fit and 2 test patches, 2,000 fit pixels, 5 draws; not a result"
    return v


def git_commit():
    try:
        c = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "exp/exp113_crosscheck.py"],
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
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="the PLANTEUR directory (ANNOTATIONS, EMB_*, metadata.geojson, splits)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true", help="2 configurations, 3 + 2 patches, 2,000 fit pixels, 5 draws")
    ap.add_argument("--report-first", choices=sorted(EMBS), default=None,
                    help="the embedding the v1.3 report ranks first overall, if known (not on the page)")
    a = ap.parse_args(argv)
    smoke = a.smoke
    fit_size = 2000 if smoke else FIT_SIZE
    ndraw = 5 if smoke else DRAWS
    nboot = 5 if smoke else BOOT
    data = os.path.abspath(a.data)
    if os.path.isfile(os.path.join(ROOT, "docs", "plan", "planteur_transfer.md")):
        with open(os.path.join(ROOT, "docs", "plan", "planteur_transfer.md"), encoding="utf-8") as f:
            page = f.read()
        if "**Status: frozen" not in page:
            sys.exit("the page is not frozen; this route grades only on the frozen page")
    commit, dirty = git_commit()
    out = {"experiment": "exp113 second route: the graded and reported numbers recomputed from the PLANTEUR labels and "
                         "embeddings with numpy, scipy and scikit-learn, without exp/exp113_planteur_transfer.py or "
                         "oe_inferencex, and written without reading the run's outputs",
           "prereg_page": "docs/plan/planteur_transfer.md", "script": "exp/exp113_crosscheck.py",
           "script_sha256": sha256_file(os.path.abspath(__file__)), "commit": commit, "script_uncommitted": dirty,
           "smoke": smoke,
           "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                           "python": platform.python_version(), "machine": platform.machine()},
           "config": {"fit": {"size": fit_size, "seed": FIT_SEED, "order": "fit rows in ascending global index"},
                      "classifier": "make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), defaults, "
                                    "float32 vectors as stored",
                      "patch": PATCH, "alphas": ALPHAS, "delta": DELTA, "grid": GRID, "certify_budget": CERT_BUDGET,
                      "level_cut": "standard (b_min / budget)", "rule": "prefix", "coverage_B": COVER_B,
                      "which_map_n": WHICH_N, "draws": ndraw,
                      "draw_seeds": {"arm2_B100": f"default_rng([{SEED_COVER100}, configuration index])",
                                     "arm3_and_arm2_B400": f"default_rng([{SEED_CERT}, configuration index]), "
                                                           "one draw set per configuration shared by its three maps",
                                     "arm4_mc": f"default_rng([{SEED_ARM4_MC}, configuration index, pair index])"},
                      "bootstrap": {"resamples": nboot, "seed": f"default_rng([{SEED_BOOT}, configuration index])",
                                    "by": "patch", "level": 0.9},
                      "configuration_index": {c: i for i, c in enumerate(CONFIGS)},
                      "tie_order_in_certify": "windows tied in confidence enter the zone in population order (test "
                                              "list order, row-major windows)"}}
    cfgs, split_chk = read_configs(data, smoke)
    out["splits"] = split_chk
    out["inputs"] = {"splits/loio.json": sha256_file(os.path.join(data, "splits", "loio.json")),
                     "metadata.geojson": sha256_file(os.path.join(data, "metadata.geojson"))}
    R = Reader(data)
    # ---- labels
    need = sorted({p for c in cfgs for p in c["train"] + c["test"]})
    labels, lab_vals = {}, np.zeros(256, np.int64)
    bad_shape = 0
    for p in need:
        t = R.load(f"ANNOTATIONS/TARGET_{p}.npy")
        bad_shape += t.shape != (1, SIDE, SIDE) or t.dtype != np.uint8
        t = np.asarray(t).reshape(SIDE, SIDE)
        lab_vals += np.bincount(t.ravel(), minlength=256)
        labels[p] = t
    out["labels"] = {"patches": len(need), "bad_shape_or_dtype": int(bad_shape),
                     "pixel_values": {str(v): int(c) for v, c in enumerate(lab_vals) if c},
                     "values_outside_classes_and_void": int(lab_vals[N_CLASSES:VOID].sum() + lab_vals[VOID + 1:].sum())}
    log(f"labels: {len(need)} patches, values {out['labels']['pixel_values']}")
    # ---- fit sets, truths
    stores, fit_rows = {}, {}
    for ci, c in enumerate(cfgs):
        pos, flat, info = draw_fit(c, labels, fit_size)
        tr = np.array(c["train"], np.int64)
        y = np.array([labels[int(tr[i])].ravel()[f] for i, f in zip(pos.tolist(), flat.tolist())], np.int64)
        # the label at the pixel rotated 180 degrees in its patch, for the in-sample alignment check
        y_rot = np.array([labels[int(tr[i])].ravel()[SIDE * SIDE - 1 - f] for i, f in zip(pos.tolist(), flat.tolist())],
                         np.int64)
        info["class_counts"] = {CLASS_NAMES[k]: int(v) for k, v in enumerate(np.bincount(y, minlength=N_CLASSES))}
        fit_rows[c["name"]] = (tr[pos], flat, y, y_rot)
        truth = np.stack([window_truth(labels[p]) for p in c["test"]]).reshape(len(c["test"]), G * G)
        st = {"fit": info, "n_patches": len(c["test"]), "truth": truth.ravel(),
              "patch": np.repeat(np.arange(len(c["test"])), G * G), "w": {}}
        st["pop"] = st["truth"] >= 0
        st["windows"] = {"all": int(truth.size), "scored": int(st["pop"].sum()),
                         "left_out_more_than_half_void": int((truth == -1).sum()),
                         "left_out_tied_label_majority": int((truth == -2).sum())}
        st["truth_class_counts"] = {CLASS_NAMES[k]: int(v) for k, v in
                                    enumerate(np.bincount(st["truth"][st["pop"]], minlength=N_CLASSES))}
        stores[c["name"]] = st
        log(f"{c['name']}: fit {len(c['train'])} patches, {info['total_nonvoid_pixels']} non-void px, draw "
            f"{info['sha256_patch_ids_then_flat_pixel_index_sorted'][:16]}; test {len(c['test'])} patches, windows "
            f"{st['windows']}")
    # patch -> what each pass needs
    by_patch_fit = {}
    for c in cfgs:
        pid, flat = fit_rows[c["name"]][:2]
        for p in np.unique(pid):
            rows = np.flatnonzero(pid == p)
            by_patch_fit.setdefault(int(p), []).append((c["name"], rows, flat[rows]))
    by_patch_test = {}
    for c in cfgs:
        for i, p in enumerate(c["test"]):
            by_patch_test.setdefault(int(p), []).append((c["name"], i))
    fits = {}
    # ---- per embedding: gather, fit, predict
    for emb, (d, pre, dims) in EMBS.items():
        te = time.time()
        X = {c["name"]: np.empty((fit_size, dims), np.float32) for c in cfgs}
        bad = 0
        for p in sorted(by_patch_fit):
            e = R.load(f"{d}/{pre}_{p}.npy")
            bad += e.shape != (dims, SIDE, SIDE) or e.dtype != np.float32
            ef = e.reshape(dims, SIDE * SIDE)
            for name, rows, fl in by_patch_fit[p]:
                X[name][rows] = ef[:, fl].T
        fin = {name: bool(np.isfinite(x).all()) for name, x in X.items()}
        log(f"{emb}: gathered fit vectors in {time.time() - te:.0f} s (finite {fin})")
        models = {}
        fits[emb] = {}
        for c in cfgs:
            name = c["name"]
            pipe, info = fit_map(X[name], fit_rows[name][2])
            info["fit_vectors_finite"] = fin[name]
            y, y_rot = fit_rows[name][2], fit_rows[name][3]
            pf = pipe.predict(X[name])
            crop, crop_r = y != 0, (y_rot != 0) & (y_rot != VOID)
            info["alignment_in_sample"] = {
                "crop_pixels": int(crop.sum()),
                "crop_agreement_label_as_given": float((pf[crop] == y[crop]).mean()) if crop.any() else None,
                "crop_agreement_label_rotated_180": float((pf[crop_r] == y_rot[crop_r]).mean()) if crop_r.any() else None,
                "accuracy": float((pf == y).mean())}
            mu_raw = X[name].mean(0, dtype=np.float64)
            models[name] = (pipe, mu_raw)
            fits[emb][name] = info
            log(f"{emb} {name}: fitted in {info['seconds']} s, n_iter {info['n_iter']}, convergence warnings "
                f"{info['convergence_warnings']}")
        del X
        for c in cfgs:
            n = len(c["test"]) * G * G
            stores[c["name"]]["w"][emb] = {"conf": np.empty(n), "cls": np.empty(n, np.int8), "dist_raw": np.empty(n),
                                           "dist_std": np.empty(n), "bnd": np.empty(n), "cond": np.empty(n, np.int8),
                                           "cm_px": np.zeros((N_CLASSES, N_CLASSES), np.int64), "tied": 0,
                                           "band_px": 0, "px": 0, "prob_sum_max_dev": 0.0}
        al = np.zeros((2, 2))        # [as given, rotated] x [crop pixels the map gets right, crop pixels]
        bad_t = 0
        tp = time.time()
        for j, p in enumerate(sorted(by_patch_test)):
            e = R.load(f"{d}/{pre}_{p}.npy")
            bad_t += e.shape != (dims, SIDE, SIDE) or e.dtype != np.float32
            Xp = np.ascontiguousarray(e.reshape(dims, SIDE * SIDE).T)
            wm = e.reshape(dims, G, PATCH, G, PATCH).mean(axis=(2, 4), dtype=np.float64).reshape(dims, G * G).T
            lab = labels[p]
            for name, i in by_patch_test[p]:
                pipe, mu = models[name]
                pr = pipe.predict_proba(Xp)
                P8 = np.zeros((SIDE * SIDE, N_CLASSES))
                P8[:, pipe[-1].classes_] = pr
                conf, cls, ntied, hard = window_map(P8)
                sw = stores[name]["w"][emb]
                sl = slice(i * G * G, (i + 1) * G * G)
                sw["prob_sum_max_dev"] = max(sw["prob_sum_max_dev"], float(np.abs(P8.sum(1) - 1).max()))
                sw["conf"][sl] = conf.ravel()
                sw["cls"][sl] = cls.ravel()
                sw["tied"] += ntied
                sw["bnd"][sl] = boundary_indicator(cls).ravel()
                nb, cond = band_condition(hard)
                sw["band_px"] += nb
                sw["px"] += SIDE * SIDE
                sw["cond"][sl] = cond.ravel()
                sw["dist_raw"][sl] = np.linalg.norm(wm - mu, axis=1)
                sc = pipe[0]
                sw["dist_std"][sl] = np.linalg.norm((wm - sc.mean_) / sc.scale_, axis=1)
                lf = lab.ravel()
                nv = lf != VOID
                sw["cm_px"] += np.bincount(lf[nv].astype(np.int64) * N_CLASSES + hard.ravel()[nv],
                                           minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
                # alignment: the map's agreement on crop pixels (non-void, not Background) with the label as given,
                # and with the label rotated 180 degrees; aligned grids agree far more with the first
                for k, L in ((0, lab), (1, lab[::-1, ::-1])):
                    crop = (L != VOID) & (L != 0)
                    al[k] += (int((hard[crop] == L[crop]).sum()), int(crop.sum()))
            if (j + 1) % 250 == 0:
                log(f"{emb}: predicted {j + 1}/{len(by_patch_test)} test patches ({time.time() - tp:.0f} s), "
                    f"peak RSS {peak_rss_gb():.2f} GB")
        fits[emb]["_shape_or_dtype_mismatches"] = int(bad + bad_t)
        ratio = [float(al[k, 0] / al[k, 1]) if al[k, 1] else None for k in (0, 1)]
        fits[emb]["_alignment"] = {
            "crop_pixel_agreement_label_as_given": ratio[0], "crop_pixel_agreement_label_rotated_180": ratio[1],
            "crop_pixels": int(al[0, 1]),
            "note": "the maps' pixel agreement with the labels on crop pixels (non-void, not Background) of the test "
                    "patches, all configurations together; with the embedding and label grids aligned the first is "
                    "far above the second"}
        log(f"{emb}: predicted in {time.time() - tp:.0f} s; crop agreement {ratio[0]} (label rotated {ratio[1]}); "
            f"total {time.time() - te:.0f} s, peak RSS {peak_rss_gb():.2f} GB")
    out["fits"] = fits
    out["inputs"]["labels"] = R.directory_digest("ANNOTATIONS")
    for emb, (d, _, _) in EMBS.items():
        out["inputs"][d] = R.directory_digest(d)
    out["inputs"]["bytes_read"] = R.bytes_read
    # ---- arms
    pops, a4, cfg_out = {}, {}, {}
    for c in cfgs:
        name = c["name"]
        ci = CONFIGS.index(name)
        st = stores[name]
        N = int(st["pop"].sum())
        cfg_out[name] = {"fit_set": st["fit"], "fit_patches": len(c["train"]), "test_patches": len(c["test"]),
                         "windows": st["windows"], "truth_class_counts": st["truth_class_counts"]}
        r100 = np.random.default_rng([SEED_COVER100, ci])
        draws_cover = np.stack([r100.choice(N, min(100, N), replace=False) for _ in range(ndraw)])
        rc = np.random.default_rng([SEED_CERT, ci])
        draws_cert = np.stack([rc.choice(N, min(CERT_BUDGET, N), replace=False) for _ in range(ndraw)])
        pops[name] = {}
        for emb in EMBS:
            res, controls = population_arms(name, st, emb, draws_cover, draws_cert)
            res["prob_sum_max_dev"] = st["w"][emb]["prob_sum_max_dev"]
            if emb == GRADED and name.startswith("loio:") and res["arm1"]["best_control"]:
                res["arm1"]["bootstrap"] = bootstrap(st, emb, res["arm1"]["best_control"], controls, [SEED_BOOT, ci], nboot)
                log(f"{name} bootstrap: {res['arm1']['bootstrap']['auroc']} lead {res['arm1']['bootstrap']['lead_over_best_control']}")
            pops[name][emb] = res
        P = st["pop"]
        a4[name] = {}
        for pi, (ea, eb) in enumerate(PAIRS):
            rng = np.random.default_rng([SEED_ARM4_MC, ci, pi])
            r = arm4(st["w"][ea]["cls"][P], st["w"][eb]["cls"][P], st["truth"][P], rng, ndraw)
            r["A"], r["B"] = ea, eb
            a4[name][f"{ea}~{eb}"] = r
            log(f"arm 4 {name} {ea} vs {eb}: N {r['N']} D {r['D']} K_A {r['K_A']} K_B {r['K_B']} "
                + json.dumps({n: (None if x is None else [round(x['p_names_A'], 4), round(x['p_names_B'], 4), x['rule']])
                              for n, x in r["budgets"].items()}))
        acc = {e: float((st["w"][e]["cls"][P] == st["truth"][P]).mean()) for e in EMBS}
        best = max(acc, key=acc.get)
        a4[name]["_best_agreeing_map"] = {"agreement": acc, "best": best,
                                          "is_report_first": (best == a.report_first) if a.report_first else None,
                                          "report_first_given": a.report_first}
    out["configurations"] = cfg_out
    out["populations"] = pops
    out["arm4"] = {k: {kk: vv for kk, vv in v.items()} for k, v in a4.items()}
    a4_pairs = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in a4.items()}
    out["verdicts"] = verdicts(pops, a4_pairs, smoke)
    out["not_computed"] = [
        "the run's own Monte Carlo draws (seed 113 + r): this route draws its own (config.draw_seeds), so arm 2's "
        "Monte Carlo, arm 3's shares and the bootstrap agree with the run's to Monte Carlo error only",
        "whether the better map is the one the v1.3 report ranks first overall, unless --report-first names it: the "
        "report's ranking is not on the page",
        "the exact chance that certify violates (only the first level's exact pass probability, which is the chance "
        "of certifying any zone with the review-set guard aside)"]
    out["seconds"] = round(time.time() - T0, 1)
    out["peak_rss_gb"] = round(peak_rss_gb(), 3)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=1)
        f.write("\n")
    log("verdicts", json.dumps({k: (x.get("holds") if isinstance(x, dict) else x) for k, x in out["verdicts"].items()}))
    log("wrote", a.out, f"peak RSS {peak_rss_gb():.2f} GB")
    return out


if __name__ == "__main__":
    main()
