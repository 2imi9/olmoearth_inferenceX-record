"""exp114's second route (docs/plan/prior_shift_pathway.md, "Independent check before anything is recorded"): the graded
numbers of the run, and the reported numbers the page names, recomputed from the PLANTEUR labels and embeddings and
exp113's saved fit selections with numpy, scipy and scikit-learn only.

Written on 23 October 2026 by an agent that did not open the run's script (exp/exp114_prior_shift.py), its job file or
its outputs, and that imports neither it, exp/exp113_planteur_transfer.py nor oe_inferencex. It follows the frozen page
and exp113's page; exp/exp113_crosscheck.py was read as a model of how to read the data. scikit-learn is used because
the map IS StandardScaler + LogisticRegression; everything after the fit is this file's own.

Re-derived from the pages (R marks a reading of something the page leaves open):
  configurations: loio_<territory> for Guadeloupe, Martinique, Guyane, Réunion and Mayotte (splits/loio.json:
    folds[territory] train, val and test, in file order) and default (metadata.geojson: Fold 1 train, Fold 4 val, Fold 5
    test, ascending ID_PATCH);
  the fit set: exp113_fitsel_<configuration>.npz, arrays 'patch' and 'pixel'. R: 'patch' holds patch ids and 'pixel'
    the flat row-major index inside the 128 x 128 patch. The reading is checked against this route's own redraw of
    exp113's rule (the non-void pixels of the train patches, list order, row-major; numpy.random.default_rng(113)
    .choice(total, 100000, replace=False)); patch positions, (row, column) pairs and non-void ranks are tried too, and
    the first reading that equals the redraw as a set is used. The fit rows keep the file's order;
  the map: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), defaults, on float64 vectors, the label
    at each fit pixel as its class; eight probabilities per pixel (a class absent from the fit set: 0);
  units: target = every non-void pixel of the test patches; source for BBSE = every non-void pixel of the validation
    patches; pi_s = the fit pixels' class shares; pi_t = the target's label shares; pi_v = the validation shares;
  arm 1: oracle weights w = pi_t / pi_s (R: 0 where pi_s = 0, a class the map never predicts), the corrected posterior
    p w / sum(p w); pixel accuracy, 8-class pixel mIoU (absent classes IoU 0, as exp113 recorded it), predicted (argmax)
    shares, producer's accuracies, the window error rate (exp113's window rules, the tie rule of exp113's second
    route), the main crop's recovered share (predicted share over true share) and producer's accuracy (Guyane Meadow,
    Réunion Sugarcane, Mayotte classes 1-7 together; R: for Mayotte a crop pixel counts as found when it is predicted
    any crop class, the exact-class reading reported beside it); the pathway gap TV(mean corrected posterior, pi_t),
    and the uncorrected gap TV(mean posterior, pi_t);
  arm 2: EM (Saerens et al. 2002): pi <- mean over the target of the per-pixel normalised p pi / pi_s, from pi_s, until
    the largest change < 1e-10 or 10,000 updates (the count of updates reported); BBSE (Lipton et al. 2018):
    C[i, j] = the share of validation pixels predicted i AND labelled j (R: the joint share, which is what makes
    C w = mu hold under prior shift), mu = the target's predicted shares, w = lstsq(C, mu) clipped at 0,
    pi proportional to w pi_v; their TV to pi_t beside the predicted shares' and pi_s's; the EM-corrected map's
    measures as in arm 1;
  arm 3: 1,000 simple random samples of n = 100 and 400 target pixels per map (this route's own seeds,
    numpy.random.default_rng([2026102314, map index, n])); per class the exact equal-tailed hypergeometric interval at
    level 1 - 0.05/8: every K whose P(X >= k | K) and P(X <= k | K) both exceed (1 - level) / 2, as K / N (found by
    bisection, both tails being monotone in K; checked against brute force on a small population); reject when EM's
    share of any class is outside its interval. Beside the Monte Carlo share, the exact rejection probability by a
    multivariate-hypergeometric recursion over the classes. The test's size: the sum over classes of the exact
    probability that the class's interval misses its true share (the Bonferroni bound P4 grades), and the exact
    probability that the test rejects the true pi_t (the same recursion);
  grades: P1 on v1.3 (oracle gap >= 0.05 and >= 3 x the default split's, in >= 2 of Guyane, Réunion, Mayotte); P2 (EM's
    TV < the predicted shares' TV in >= 10 of the 15 held-out maps); P3 (every held-out v1.3 map with EM's TV >= 0.05
    rejected with probability >= 0.8 at n = 400, the Monte Carlo share; none such: not determined); P4 (the size
    <= 0.05 at n = 100 and 400 on all 18 maps).

    uv run --frozen --with "scikit-learn==1.9.1" python -I exp/exp114_crosscheck.py --data <PLANTEUR directory> \
        --e113-private <directory of exp113_fitsel_*.npz> --out <JSON> [--smoke]
    --smoke: the first 3 patches of every train, val and test list, the fit pixels inside those train patches, 20
             draws (a code check: no number from it means anything)

Memory: the fit vectors of the six configurations of one embedding (float32 as stored, about 0.3 GB at 128 channels),
then one configuration and one embedding at a time: its test pixels' eight probabilities in float64 (about 0.85 GB for
Réunion's 813 patches) and labels. EM costs about two passes over that array per update. Writes aggregates only: no
patch, window, pixel or coordinate list.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import unicodedata
import warnings

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "plan", "prior_shift_pathway.md")
TERRITORIES = ("Guadeloupe", "Martinique", "Guyane", "Réunion", "Mayotte")
CONFIGS = tuple(f"loio_{t}" for t in TERRITORIES) + ("default",)
EMBS = (("v1.3", "EMB_OLMOEARTH_V1_3", "OE13", 128),
        ("tessera_v2", "EMB_TESSERA_V2", "TESSERA", 128),
        ("alphaearth", "EMB_AEF", "AEF", 64))
GRADED = "v1.3"
NC, VOID, SIDE, WIN = 8, 255, 128, 4
PX = SIDE * SIDE
G = SIDE // WIN
CLASS_NAMES = ("Background", "Meadow", "Fruits, vegetables, flowers", "Orchard", "Sugarcane", "Banana", "Pineapple",
               "Tropical tuber")
CLOSEST = ("Guyane", "Réunion", "Mayotte")
MAIN_CROP = {"Guyane": (1,), "Réunion": (4,), "Mayotte": (1, 2, 3, 4, 5, 6, 7)}
FIT_SIZE, FIT_SEED = 100000, 113
EM_TOL, EM_MAX = 1e-10, 10000
LEVEL = 1 - 0.05 / 8
NS = (100, 400)
DRAWS, SMOKE_DRAWS, SMOKE_PATCHES = 1000, 20, 3
SEED_LABELS = 2026102314
# exp113's recorded numbers (docs/plan/planteur_transfer.md, Result), for a reproduction check of the refitted maps
E113_WINDOW_ERROR = {"loio_Guadeloupe": (0.108, 0.117, 0.127), "loio_Martinique": (0.112, 0.116, 0.124),
                     "loio_Guyane": (0.172, 0.171, 0.172), "loio_Réunion": (0.251, 0.252, 0.358),
                     "loio_Mayotte": (0.0052, 0.0055, 0.0058), "default": (0.088, 0.088, 0.099)}
E113_MIOU_V13 = {"loio_Guadeloupe": 0.308, "loio_Martinique": 0.319, "loio_Guyane": 0.120, "loio_Réunion": 0.127,
                 "loio_Mayotte": 0.125, "default": 0.356}
E113_MAIN_TRUE_SHARE = {"Guyane": 0.143, "Réunion": 0.160, "Mayotte": 0.006}
E113_MAIN_IOU_V13 = {"Guyane": 0.007, "Réunion": 0.013}
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


def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum())


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------ splits
def read_configs(data, smoke):
    with open(os.path.join(data, "splits", "loio.json"), encoding="utf-8") as f:
        loio = json.load(f)
    with open(os.path.join(data, "metadata.geojson"), encoding="utf-8") as f:
        meta = json.load(f)
    props = [ft["properties"] for ft in meta["features"]]
    fold = {int(p["ID_PATCH"]): int(p["Fold"]) for p in props}
    island = {int(p["ID_PATCH"]): p.get("Island") for p in props}
    folds = {unicodedata.normalize("NFC", k): v for k, v in loio["folds"].items()}
    cfgs, chk = [], {"metadata_patches": len(props), "per_configuration": {}}
    for t in TERRITORIES:
        fd = folds[t]
        c = {"name": f"loio_{t}", "territory": t, "train": [int(x) for x in fd["train"]],
             "val": [int(x) for x in fd.get("val", [])], "test": [int(x) for x in fd["test"]]}
        cfgs.append(c)
    ids = sorted(fold)
    cfgs.append({"name": "default", "territory": None, "train": [i for i in ids if fold[i] == 1],
                 "val": [i for i in ids if fold[i] == 4], "test": [i for i in ids if fold[i] == 5]})
    for c in cfgs:
        tr, va, te = set(c["train"]), set(c["val"]), set(c["test"])
        d = {"train": len(c["train"]), "val": len(c["val"]), "test": len(c["test"]),
             "train_val_overlap": len(tr & va), "train_test_overlap": len(tr & te), "val_test_overlap": len(va & te),
             "unknown_ids": len((tr | va | te) - set(fold))}
        if c["territory"]:
            d["test_equals_metadata_island"] = te == {i for i in ids if unicodedata.normalize(
                "NFC", str(island[i])) == c["territory"]}
            d["val_from_held_out_territory"] = sum(unicodedata.normalize("NFC", str(island.get(i))) == c["territory"]
                                                   for i in va)
        chk["per_configuration"][c["name"]] = d
        c["train_full"] = list(c["train"])
        if smoke:
            for k in ("train", "val", "test"):
                c[k] = c[k][:SMOKE_PATCHES]
    return cfgs, chk


class Labels:
    def __init__(self, data):
        self.data, self.cache, self.bad = data, {}, 0

    def __call__(self, p):
        if p not in self.cache:
            a = np.load(os.path.join(self.data, "ANNOTATIONS", f"TARGET_{p}.npy"), allow_pickle=False)
            self.bad += a.shape != (1, SIDE, SIDE) or a.dtype != np.uint8
            a = np.asarray(a).reshape(SIDE, SIDE)
            vals = np.unique(a)
            if ((vals >= NC) & (vals != VOID)).any():
                raise SystemExit(f"label values outside 0-7 and 255 in patch {p}: {vals.tolist()}")
            self.cache[p] = a
        return self.cache[p]


def load_emb(data, d, pre, p):
    e = np.load(os.path.join(data, d, f"{pre}_{p}.npy"), allow_pickle=False)
    if e.ndim != 3 or e.shape[1:] != (SIDE, SIDE) or not np.issubdtype(e.dtype, np.floating):
        raise SystemExit(f"{d}/{pre}_{p}.npy: shape {e.shape} dtype {e.dtype}")
    return e


# ------------------------------------------------------------------ the fit selection
def find_npz(dirp, name):
    for form in ("NFC", "NFD"):
        p = os.path.join(dirp, unicodedata.normalize(form, f"exp113_fitsel_{name}.npz"))
        if os.path.isfile(p):
            return p
    want = unicodedata.normalize("NFC", f"exp113_fitsel_{name}.npz")
    for f in os.listdir(dirp):
        if unicodedata.normalize("NFC", f) == want:
            return os.path.join(dirp, f)
    raise SystemExit(f"no exp113_fitsel_{name}.npz in {dirp}")


def nonvoid_flat(lab):
    return np.flatnonzero(lab.ravel() != VOID)


def redraw(train, labels, size):
    """This route's redraw of exp113's fit pixels: (patch ids, flat indices, non-void ranks), ascending global index."""
    nv = [nonvoid_flat(labels(p)) for p in train]
    cnt = np.array([x.size for x in nv], np.int64)
    end = np.cumsum(cnt)
    start = end - cnt
    total = int(end[-1]) if end.size else 0
    if total < size:
        return None, total
    gs = np.sort(np.random.default_rng(FIT_SEED).choice(total, size, replace=False))
    pos = np.searchsorted(end, gs, side="right")
    rank = gs - start[pos]
    flat = np.empty(size, np.int64)
    for i in np.unique(pos):
        m = pos == i
        flat[m] = nv[i][rank[m]]
    return (np.asarray(train, np.int64)[pos], flat, rank), total


def keys(pid, flat):
    return np.sort(np.asarray(pid, np.int64) * PX + np.asarray(flat, np.int64))


def resolve_fitsel(path, cfg, labels, smoke):
    z = np.load(path, allow_pickle=False)
    praw, xraw = np.asarray(z["patch"]), np.asarray(z["pixel"])
    info = {"file": os.path.basename(path), "file_sha256": sha256_file(path),
            "patch_array": {"dtype": str(praw.dtype), "shape": list(praw.shape)},
            "pixel_array": {"dtype": str(xraw.dtype), "shape": list(xraw.shape)},
            "sha256_arrays_as_stored": hashlib.sha256(np.ascontiguousarray(praw).tobytes()
                                                      + np.ascontiguousarray(xraw).tobytes()).hexdigest()}
    train = cfg["train_full"]
    tr_arr = np.asarray(train, np.int64)
    prw = praw.astype(np.int64).ravel()
    patch_reads = []
    if np.isin(prw, tr_arr).all():
        patch_reads.append(("patch_id", prw))
    if prw.size and prw.min() >= 0 and prw.max() < len(train):
        patch_reads.append(("position_in_train_list", tr_arr[prw]))
    try:
        rd, total = redraw(train, labels, FIT_SIZE)
        info["redraw_total_nonvoid_pixels"] = total
        info["redraw_available"] = rd is not None
    except FileNotFoundError as ex:
        rd = None
        info["redraw_available"] = False
        info["redraw_note"] = f"labels missing for the redraw: {ex.filename}"
    chosen, matched = None, (False if rd is not None else None)
    for pname, pid in patch_reads:
        reads = []
        if xraw.ndim == 2 and xraw.shape[1] == 2:
            rc = xraw.astype(np.int64)
            if rc.min() >= 0 and rc.max() < SIDE:
                reads.append(("row_col", rc[:, 0] * SIDE + rc[:, 1]))
        elif xraw.ndim == 1:
            x = xraw.astype(np.int64)
            if x.min() >= 0 and x.max() < PX:
                reads.append(("flat_row_major", x))
            ok, fl = True, np.empty_like(x)
            for p in np.unique(pid):
                m = pid == p
                nv = nonvoid_flat(labels(int(p)))
                if x[m].max() >= nv.size:
                    ok = False
                    break
                fl[m] = nv[x[m]]
            if ok:
                reads.append(("nonvoid_rank", fl))
        for xname, flat in reads:
            if chosen is None:
                chosen = (pname, xname, pid, flat)
            if rd is not None and np.array_equal(keys(pid, flat), keys(rd[0], rd[1])):
                chosen, matched = (pname, xname, pid, flat), True
                break
        if matched:
            break
    if chosen is None:
        raise SystemExit(f"{path}: no reading of 'patch' and 'pixel' fits the configuration's train patches")
    pname, xname, pid, flat = chosen
    info.update({"reading": {"patch": pname, "pixel": xname}, "matches_independent_redraw": matched,
                 "size": int(pid.size), "duplicates": int(pid.size - np.unique(keys(pid, flat)).size)})
    posmap = {p: i for i, p in enumerate(train)}
    gpos = np.array([posmap[int(p)] for p in pid], np.int64)
    o = np.lexsort((flat, gpos))
    info["sha256_patch_ids_then_flat_pixel_index_sorted"] = hashlib.sha256(
        pid[o].astype("<i8").tobytes() + flat[o].astype("<i8").tobytes()).hexdigest()
    info["sha256_patch_ids_then_flat_pixel_index_as_stored"] = hashlib.sha256(
        pid.astype("<i8").tobytes() + flat.astype("<i8").tobytes()).hexdigest()
    if rd is not None:
        info["sha256_redraw_patch_ids_then_flat_pixel_index_sorted"] = hashlib.sha256(
            rd[0].astype("<i8").tobytes() + rd[1].astype("<i8").tobytes()).hexdigest()
    if smoke:
        keep = np.isin(pid, np.asarray(cfg["train"], np.int64))
        pid, flat = pid[keep], flat[keep]
        info["smoke_kept"] = int(keep.sum())
    y = np.empty(pid.size, np.int64)
    for p in np.unique(pid):
        m = pid == p
        y[m] = labels(int(p)).ravel()[flat[m]]
    void = y == VOID
    info["void_fit_pixels_dropped"] = int(void.sum())
    pid, flat, y = pid[~void], flat[~void], y[~void]
    info["class_counts"] = {CLASS_NAMES[c]: int(v) for c, v in enumerate(np.bincount(y, minlength=NC))}
    return pid, flat, y, info


# ------------------------------------------------------------------ windows (exp113's rules)
def blocks(a):
    """(n, 128, 128) -> (n, 32, 32, 16)."""
    n = a.shape[0]
    return a.reshape(n, G, WIN, G, WIN).transpose(0, 1, 3, 2, 4).reshape(n, G, G, WIN * WIN)


def window_truth(lab):
    """(n, 128, 128) labels -> (n, 1024): -1 more than half void, -2 tied label majority, else the majority class."""
    lw = blocks(lab)
    void = (lw == VOID).sum(-1)
    cnt = np.stack([(lw == c).sum(-1) for c in range(NC)], -1)
    top = cnt.max(-1, keepdims=True)
    tied = (cnt == top).sum(-1) > 1
    t = np.where(tied, -2, cnt.argmax(-1))
    t = np.where(void > WIN * WIN // 2, -1, t)
    return t.reshape(lab.shape[0], G * G).astype(np.int8)


def window_classes(hard, top):
    """Majority of the 16 pixels' classes; a tie to the tied class with the larger summed top probability (then the
    lower index)."""
    hb, tb = blocks(hard), blocks(top)
    cnt = np.stack([(hb == c).sum(-1) for c in range(NC)], -1)
    wsum = np.stack([np.where(hb == c, tb, 0.0).sum(-1) for c in range(NC)], -1)
    mx = cnt.max(-1, keepdims=True)
    tied = (cnt == mx).sum(-1) > 1
    cls = np.where(tied, np.where(cnt == mx, wsum, -np.inf).argmax(-1), cnt.argmax(-1))
    return cls.reshape(hard.shape[0], G * G), int(tied.sum())


# ------------------------------------------------------------------ one map's measures under a prior correction
def measures(P, lab, wtruth, w, pi_t, crop, chunk=32):
    """P (pixels, 8) posteriors of every pixel of the test patches, lab their labels, wtruth the windows' truth;
    w = None: the map as fitted; else the corrected posterior p w / sum(p w)."""
    npat = wtruth.shape[0]
    cm = np.zeros((NC, NC), np.int64)            # [truth, predicted], non-void pixels
    post = np.zeros(NC)
    fallback, werr, wsc, wties = 0, 0, 0, 0
    for a in range(0, npat, chunk):
        b = min(npat, a + chunk)
        sl = slice(a * PX, b * PX)
        Pc = P[sl]
        L = lab[sl]
        nv = L != VOID
        if w is None:
            S = Pc
        else:
            S = Pc * w
            den = S.sum(1)
            fb = ~(den > 0)
            S /= np.where(fb, 1.0, den)[:, None]
            if fb.any():
                S[fb] = Pc[fb]
                fallback += int((fb & nv).sum())
        hard = S.argmax(1)
        top = S.max(1)
        cm += np.bincount(L[nv].astype(np.int64) * NC + hard[nv], minlength=NC * NC).reshape(NC, NC)
        post += S[nv].sum(0)
        cls, nt = window_classes(hard.reshape(b - a, SIDE, SIDE), top.reshape(b - a, SIDE, SIDE))
        t = wtruth[a:b]
        sc = t >= 0
        werr += int((cls[sc] != t[sc]).sum())
        wsc += int(sc.sum())
        wties += nt
    N = int(cm.sum())
    diag = np.diag(cm).astype(np.float64)
    rows, colsum = cm.sum(1), cm.sum(0)
    union = rows + colsum - np.diag(cm)
    iou = np.where(union > 0, diag / np.maximum(union, 1), 0.0)
    pred = colsum / N
    mean_post = post / N
    out = {"pixel_accuracy": float(diag.sum() / N), "miou_8": float(iou.mean()),
           "miou_present_classes": float(iou[union > 0].mean()) if (union > 0).any() else None,
           "iou": [float(x) for x in iou], "predicted_shares": [float(x) for x in pred],
           "producers_accuracy": [float(diag[c] / rows[c]) if rows[c] else None for c in range(NC)],
           "tv_predicted_shares_to_pi_t": tv(pred, pi_t), "mean_posterior": [float(x) for x in mean_post],
           "gap_tv_mean_posterior_to_pi_t": tv(mean_post, pi_t),
           "window_error_rate": werr / wsc if wsc else None, "windows_scored": wsc, "window_errors": werr,
           "predicted_window_majority_ties": wties, "nonvoid_pixels_zero_corrected_mass_left_uncorrected": fallback}
    if crop:
        cr = list(crop)
        true_n = int(cm[cr, :].sum())
        pred_n = int(cm[:, cr].sum())
        out["main_crop"] = {"classes": cr, "true_share": true_n / N, "predicted_share": pred_n / N,
                            "recovered_share": (pred_n / true_n) if true_n else None,
                            "producers_accuracy_any_crop_class": float(cm[np.ix_(cr, cr)].sum() / true_n) if true_n else None,
                            "producers_accuracy_exact_class": float(sum(cm[c, c] for c in cr) / true_n) if true_n else None,
                            "iou_merged": float(cm[np.ix_(cr, cr)].sum() / (true_n + pred_n - cm[np.ix_(cr, cr)].sum()))
                            if (true_n + pred_n) else None}
    return out


def weights(target, pi_s):
    ok = pi_s > 0
    return np.where(ok, target / np.where(ok, pi_s, 1.0), 0.0)


def em(P, nvf, pi_s, label):
    """Saerens, Latinne and Decaestecker (2002): pi <- mean_i p_i(pi / pi_s) / sum_k p_ik pi_k / pi_s,k, from pi_s."""
    N = float(nvf.sum())
    pi = pi_s.astype(np.float64).copy()
    it, delta, t = 0, np.inf, time.time()
    zero_q = 0
    while it < EM_MAX:
        w = weights(pi, pi_s)
        q = P @ w
        r = np.divide(nvf, q, out=np.zeros_like(q), where=q > 0)
        if it == 0:
            zero_q = int(((q <= 0) & (nvf > 0)).sum())
        new = w * (P.T @ r) / N
        new /= new.sum()
        delta = float(np.abs(new - pi).max())
        pi = new
        it += 1
        if delta < EM_TOL:
            break
        if it % 1000 == 0:
            log(f"  {label} EM update {it}: largest change {delta:.3e} ({time.time() - t:.0f} s)")
    return pi, {"iterations": it, "converged": bool(delta < EM_TOL), "last_largest_change": delta,
                "nonvoid_pixels_with_zero_mass_at_start": zero_q, "seconds": round(time.time() - t, 1)}


def bbse(Cv, mu):
    """Lipton, Wang and Smola (2018): C w = mu by least squares, w clipped at 0, pi proportional to w pi_v."""
    tot = Cv.sum()
    if tot == 0:
        return None
    C = Cv / tot                                   # [predicted i, label j], joint shares
    w, _, rank, sv = np.linalg.lstsq(C, mu, rcond=None)
    pi_v = C.sum(0)
    wc = np.clip(w, 0.0, None)
    est = wc * pi_v
    s = est.sum()
    return {"w": [float(x) for x in w], "w_clipped": [float(x) for x in wc], "pi_v": [float(x) for x in pi_v],
            "pi": [float(x) for x in est / s] if s > 0 else None, "rank_C": int(rank),
            "singular_values_C": [float(x) for x in sv],
            "note": "a rank-deficient C (a class never predicted) has many least-squares solutions; numpy's lstsq "
                    "returns the one of minimum norm",
            "condition_number_C": float(sv[0] / sv[-1]) if sv[-1] > 0 else None}


# ------------------------------------------------------------------ arm 3: exact intervals, the test, its size
_TABLES = {}
TABLE_CHECKS = {}        # per (N, n): counts k whose interval ends fail their own check (0 expected)


def interval_table(N, n, level=LEVEL):
    """For every count k = 0..n of a simple random sample of n from N: the smallest and largest K whose two one-sided
    tails P(X >= k | K) and P(X <= k | K) each exceed (1 - level) / 2. Bisection: the first tail rises with K, the
    second falls."""
    key = (N, n, level)
    if key in _TABLES:
        return _TABLES[key]
    a = (1 - level) / 2
    k = np.arange(n + 1)
    L, R = np.zeros(n + 1, np.int64), np.full(n + 1, N, np.int64)
    while (L < R).any():
        act = L < R
        mid = (L + R) // 2
        ok = stats.hypergeom.sf(k - 1, N, mid, n) > a
        R = np.where(act & ok, mid, R)
        L = np.where(act & ~ok, mid + 1, L)
    klo = L
    L, R = np.zeros(n + 1, np.int64), np.full(n + 1, N, np.int64)
    while (L < R).any():
        act = L < R
        mid = (L + R + 1) // 2
        ok = stats.hypergeom.cdf(k, N, mid, n) > a
        L = np.where(act & ok, mid, L)
        R = np.where(act & ~ok, mid - 1, R)
    khi = L
    # the ends' own check: each end passes and its outer neighbour fails, and the interval is not empty
    bad = ~(stats.hypergeom.sf(k - 1, N, klo, n) > a)
    bad |= (klo > 0) & (stats.hypergeom.sf(k - 1, N, np.maximum(klo - 1, 0), n) > a)
    bad |= ~(stats.hypergeom.cdf(k, N, khi, n) > a)
    bad |= (khi < N) & (stats.hypergeom.cdf(k, N, np.minimum(khi + 1, N), n) > a)
    bad |= klo > khi
    TABLE_CHECKS[f"N{N}_n{n}"] = int(bad.sum())
    _TABLES[key] = (klo, khi)
    return klo, khi


def interval_table_brute(N, n, level=LEVEL):
    a = (1 - level) / 2
    Ks = np.arange(N + 1)
    lo, hi = np.empty(n + 1, np.int64), np.empty(n + 1, np.int64)
    for k in range(n + 1):
        ok = (stats.hypergeom.sf(k - 1, N, Ks, n) > a) & (stats.hypergeom.cdf(k, N, Ks, n) > a)
        lo[k], hi[k] = Ks[ok].min(), Ks[ok].max()
    return lo, hi


def joint_accept(N, K, n, acc):
    """P(every class's count k_c lies in its acceptance set acc[c]) under the multivariate hypergeometric of a simple
    random sample of n from N with class counts K, class by class: k_c ~ HG(N - K_1 - ... - K_{c-1}, K_c, n - s)."""
    prob = np.zeros(n + 1)
    prob[0] = 1.0
    M = int(N)
    for c in range(NC):
        Kc = int(K[c])
        new = np.zeros(n + 1)
        for s in np.flatnonzero(prob > 0):
            r = n - s
            if Kc == 0:
                pk = np.zeros(r + 1)
                pk[0] = 1.0
            elif Kc == M:
                pk = np.zeros(r + 1)
                pk[r] = 1.0
            else:
                pk = stats.hypergeom.pmf(np.arange(r + 1), M, Kc, r)
            new[s:s + r + 1] += prob[s] * pk * acc[c, :r + 1]
        prob = new
        M -= Kc
    return float(prob[n])


def random_labels(y_t, K, em_pi, map_index, draws):
    N = int(y_t.size)
    out = {}
    cls = np.arange(NC)
    for n0 in NS:
        n = min(n0, N)
        klo, khi = interval_table(N, n)
        lo, hi = klo / N, khi / N
        acc_true = (klo[None, :] <= K[:, None]) & (K[:, None] <= khi[None, :])
        acc_em = (lo[None, :] <= em_pi[:, None]) & (em_pi[:, None] <= hi[None, :])
        miss = []
        for c in range(NC):
            pmf = stats.hypergeom.pmf(np.arange(n + 1), N, int(K[c]), n)
            miss.append(float(pmf[~acc_true[c]].sum()))
        rng = np.random.default_rng([SEED_LABELS, map_index, n0])
        rej_em = rej_true = khat_out = 0
        by_class = np.zeros(NC, np.int64)
        for _ in range(draws):
            k = np.bincount(y_t[rng.choice(N, n, replace=False)], minlength=NC)
            re = ~acc_em[cls, k]
            rej_em += bool(re.any())
            by_class += re
            rej_true += bool((~acc_true[cls, k]).any())
            khat_out += int(((klo[k] * n > k * N) | (khi[k] * n < k * N)).sum())
        p_rej = rej_em / draws
        out[str(n0)] = {"labels": n, "draws": draws, "seed": f"default_rng([{SEED_LABELS}, {map_index}, {n0}])",
                        "em_rejection_share": p_rej,
                        "em_rejection_share_mc_se": math.sqrt(p_rej * (1 - p_rej) / draws),
                        "em_rejection_exact": 1 - joint_accept(N, K, n, acc_em),
                        "em_rejections_by_class": [int(x) for x in by_class],
                        "size_bonferroni_sum_exact": float(sum(miss)), "size_per_class_exact": miss,
                        "size_joint_exact": 1 - joint_accept(N, K, n, acc_true),
                        "true_pi_t_rejection_share_mc": rej_true / draws,
                        "class_intervals_not_holding_sample_share": khat_out}
    return out


# ------------------------------------------------------------------ fitting
def fit_map(X, y):
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    lr = pipe[-1]
    return pipe, {"classes": [int(c) for c in lr.classes_], "rows": int(X.shape[0]), "X_dtype": str(X.dtype),
                  "n_iter": [int(x) for x in np.ravel(lr.n_iter_)],
                  "convergence_warnings": sum(issubclass(x.category, ConvergenceWarning) for x in w),
                  "other_warning_types": sorted({x.category.__name__ for x in w
                                                 if not issubclass(x.category, ConvergenceWarning)}),
                  "seconds": round(time.time() - t, 1)}


def proba8(pipe, e):
    C = e.shape[0]
    X = np.ascontiguousarray(e.reshape(C, PX).T, dtype=np.float64)
    P8 = np.zeros((PX, NC))
    P8[:, pipe[-1].classes_] = pipe.predict_proba(X)
    return P8


# ------------------------------------------------------------------ verdicts
def verdicts(maps, smoke):
    v = {}
    g = {c: maps[c][GRADED] for c in maps if GRADED in maps[c]}
    dflt = g.get("default", {}).get("oracle", {}).get("gap_tv_mean_posterior_to_pi_t")
    rows, n_ok = {}, 0
    for t in CLOSEST:
        r = g.get(f"loio_{t}")
        if r is None or dflt is None:
            rows[t] = None
            continue
        gap = r["oracle"]["gap_tv_mean_posterior_to_pi_t"]
        a, b = gap >= 0.05, gap >= 3 * dflt
        rows[t] = {"oracle_gap": gap, "uncorrected_gap": r["uncorrected"]["gap_tv_mean_posterior_to_pi_t"],
                   "ratio_to_default": gap / dflt if dflt > 0 else None, "gap_at_least_0.05": bool(a),
                   "at_least_3x_default": bool(b), "meets": bool(a and b)}
        n_ok += a and b
    v["P1"] = {"embedding": GRADED, "default_oracle_gap": dflt, "territories": rows, "meeting": int(n_ok),
               "holds": bool(n_ok >= 2) if dflt is not None else None, "bar": ">= 2 of Guyane, Réunion, Mayotte"}
    held = {f"{c}|{e}": maps[c][e] for c in maps if c != "default" for e in maps[c]}
    rows, n_ok = {}, 0
    for k, r in held.items():
        a, b = r["tv"]["em_to_pi_t"], r["tv"]["predicted_shares_to_pi_t"]
        rows[k] = {"em_tv": a, "predicted_shares_tv": b, "em_closer": bool(a < b)}
        n_ok += a < b
    v["P2"] = {"maps": len(held), "em_closer": int(n_ok), "rows": rows, "holds": bool(n_ok >= 10),
               "bar": ">= 10 of the 15 held-out maps"}
    rows = {}
    for c in maps:
        if c == "default" or GRADED not in maps[c]:
            continue
        r = maps[c][GRADED]
        cell = r["random_labels"]["400"]
        rows[c] = {"em_tv": r["tv"]["em_to_pi_t"], "far": bool(r["tv"]["em_to_pi_t"] >= 0.05),
                   "rejection_share_400": cell["em_rejection_share"], "rejection_exact_400": cell["em_rejection_exact"]}
    far = {c: x for c, x in rows.items() if x["far"]}
    v["P3"] = {"embedding": GRADED, "rows": rows, "maps_far": len(far),
               "holds": (bool(all(x["rejection_share_400"] >= 0.8 for x in far.values())) if far else None),
               "holds_on_exact_probability": (bool(all(x["rejection_exact_400"] >= 0.8 for x in far.values()))
                                              if far else None),
               "note": "graded on the Monte Carlo share (1,000 draws); None: no held-out v1.3 map has EM at least 0.05 "
                       "from pi_t, not determined"}
    sz = {f"{c}|{e}|{n}": maps[c][e]["random_labels"][n]["size_bonferroni_sum_exact"]
          for c in maps for e in maps[c] for n in maps[c][e]["random_labels"]}
    szj = {f"{c}|{e}|{n}": maps[c][e]["random_labels"][n]["size_joint_exact"]
           for c in maps for e in maps[c] for n in maps[c][e]["random_labels"]}
    v["P4"] = {"cells": len(sz), "max_size_bonferroni_sum": max(sz.values()) if sz else None,
               "argmax": max(sz, key=sz.get) if sz else None, "holds": bool(sz and all(x <= 0.05 for x in sz.values())),
               "max_size_joint_exact": max(szj.values()) if szj else None,
               "holds_on_joint_exact": bool(szj and all(x <= 0.05 for x in szj.values())),
               "note": "the graded number is the sum over classes of the exact per-class non-coverage (an upper bound "
                       "on the exact rejection probability); the joint exact probability is beside it"}
    if smoke:
        v["note"] = "SMOKE: the first 3 patches of each list, 20 draws; not a result"
    return v


def reproduction(maps):
    out = {}
    for c in maps:
        for ei, (emb, _, _, _) in enumerate(EMBS):
            if emb not in maps[c]:
                continue
            r = maps[c][emb]["uncorrected"]
            rec = E113_WINDOW_ERROR.get(c, (None,) * 3)[ei]
            cell = {"window_error_rate": r["window_error_rate"], "exp113_recorded": rec,
                    "difference": (r["window_error_rate"] - rec) if rec is not None and r["window_error_rate"] is not None else None}
            if emb == GRADED:
                cell["miou_8"] = r["miou_8"]
                cell["exp113_recorded_miou_8"] = E113_MIOU_V13.get(c)
                t = c[5:] if c.startswith("loio_") else None
                if t in MAIN_CROP:
                    mc = r["main_crop"]
                    cell["main_crop_true_share"] = mc["true_share"]
                    cell["exp113_recorded_main_crop_true_share"] = E113_MAIN_TRUE_SHARE[t]
                    if t in E113_MAIN_IOU_V13:
                        cell["main_crop_iou"] = r["iou"][MAIN_CROP[t][0]]
                        cell["exp113_recorded_main_crop_iou"] = E113_MAIN_IOU_V13[t]
            out[f"{c}|{emb}"] = cell
    return out


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="the PLANTEUR directory (ANNOTATIONS, EMB_*, metadata.geojson, splits)")
    ap.add_argument("--e113-private", required=True, help="the directory of exp113_fitsel_<configuration>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="the first 3 patches of each list, 20 draws")
    a = ap.parse_args(argv)
    smoke = a.smoke
    draws = SMOKE_DRAWS if smoke else DRAWS
    data, priv = os.path.abspath(a.data), os.path.abspath(a.e113_private)
    if os.path.isfile(PAGE):
        with open(PAGE, encoding="utf-8") as f:
            if "**Status: frozen" not in f.read():
                sys.exit("the page is not frozen; this route grades only on the frozen page")
        page_sha = sha256_file(PAGE)
    else:
        page_sha = None
    out = {"experiment": "exp114 second route: the graded and reported numbers recomputed from the PLANTEUR labels and "
                         "embeddings and exp113's fit selections with numpy, scipy and scikit-learn, without "
                         "exp/exp114_prior_shift.py, exp/exp113_planteur_transfer.py or oe_inferencex, and written "
                         "without reading the run's script or outputs",
           "prereg_page": "docs/plan/prior_shift_pathway.md", "prereg_page_sha256": page_sha,
           "script": "exp/exp114_crosscheck.py", "script_sha256": sha256_file(os.path.abspath(__file__)),
           "smoke": smoke,
           "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                           "python": platform.python_version(), "machine": platform.machine(),
                           "platform": platform.platform()},
           "config": {"fit": {"size": FIT_SIZE, "seed_of_redraw": FIT_SEED, "rows": "as stored in the npz"},
                      "classifier": "make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), defaults, "
                                    "float64 vectors",
                      "em": {"start": "pi_s", "tolerance_largest_change": EM_TOL, "max_updates": EM_MAX},
                      "bbse": "C[i, j] = joint share of validation pixels predicted i and labelled j; lstsq; clip at 0; "
                              "pi proportional to w pi_v",
                      "intervals": {"level": LEVEL, "each_tail": (1 - LEVEL) / 2},
                      "labels_n": NS, "draws": draws,
                      "draw_seed": f"numpy default_rng([{SEED_LABELS}, map index, n]); map index = 3 x configuration "
                                   "index + embedding index",
                      "configuration_index": {c: i for i, c in enumerate(CONFIGS)},
                      "embedding_index": {e[0]: i for i, e in enumerate(EMBS)},
                      "main_crop": {t: [CLASS_NAMES[c] for c in v] for t, v in MAIN_CROP.items()},
                      "class_names": CLASS_NAMES}}
    # self-check of the interval table against brute force
    chk = {}
    for n in (20, 100):
        b, f = interval_table_brute(500, n), interval_table(500, n)
        chk[f"N500_n{n}"] = bool(np.array_equal(b[0], f[0]) and np.array_equal(b[1], f[1]))
    _TABLES.clear()
    TABLE_CHECKS.clear()
    out["self_checks"] = {"interval_bisection_equals_brute_force": chk}
    if not all(chk.values()):
        sys.exit(f"interval table self-check failed: {chk}")
    cfgs, split_chk = read_configs(data, smoke)
    out["splits"] = split_chk
    out["inputs"] = {"splits/loio.json": sha256_file(os.path.join(data, "splits", "loio.json")),
                     "metadata.geojson": sha256_file(os.path.join(data, "metadata.geojson"))}
    labels = Labels(data)
    # ---- the fit selections
    fits = {}
    out["fit_selection"] = {}
    for c in cfgs:
        pid, flat, y, info = resolve_fitsel(find_npz(priv, c["name"]), c, labels, smoke)
        fits[c["name"]] = (pid, flat, y)
        out["fit_selection"][c["name"]] = info
        log(f"{c['name']}: fit selection {info['size']} px, reading {info['reading']}, matches redraw "
            f"{info['matches_independent_redraw']}, used {y.size}, sha256 {info['sha256_patch_ids_then_flat_pixel_index_sorted'][:16]}")
        if info["matches_independent_redraw"] is False and not smoke:
            log(f"WARNING {c['name']}: the npz does not equal this route's redraw of exp113's rule")
    by_patch_fit = {}
    for c in cfgs:
        pid, flat, _ = fits[c["name"]]
        for p in np.unique(pid):
            rows = np.flatnonzero(pid == p)
            by_patch_fit.setdefault(int(p), []).append((c["name"], rows, flat[rows]))
    maps, fit_info = {c["name"]: {} for c in cfgs}, {}
    for ei, (emb, d, pre, dims_expected) in enumerate(EMBS):
        te = time.time()
        X, dims = {}, None
        for p in sorted(by_patch_fit):
            e = load_emb(data, d, pre, p)
            if dims is None:
                dims = e.shape[0]
                dt = np.result_type(e.dtype, np.float32)
                X = {c["name"]: np.empty((fits[c["name"]][2].size, dims), dt) for c in cfgs}
            if e.shape[0] != dims:
                sys.exit(f"{d}: channel count {e.shape[0]} in patch {p}, {dims} before")
            ef = e.reshape(dims, PX)
            for name, rows, fl in by_patch_fit[p]:
                X[name][rows] = ef[:, fl].T
        if dims is None:
            log(f"{emb}: no fit pixels in any configuration; skipped")
            continue
        log(f"{emb}: {dims} channels (page: {dims_expected}); fit vectors gathered in {time.time() - te:.0f} s")
        models = {}
        for c in cfgs:
            name = c["name"]
            y = fits[name][2]
            Xf = X[name].astype(np.float64)
            fin = bool(np.isfinite(Xf).all())
            if np.unique(y).size < 2 or not fin:
                fit_info[f"{name}|{emb}"] = {"fitted": False, "classes_in_fit": int(np.unique(y).size), "finite": fin}
                log(f"{emb} {name}: not fitted (classes {np.unique(y).tolist()}, finite {fin})")
                continue
            pipe, info = fit_map(Xf, y)
            info["channels"], info["channels_expected"], info["fitted"] = dims, dims_expected, True
            models[name] = pipe
            fit_info[f"{name}|{emb}"] = info
            log(f"{emb} {name}: fitted in {info['seconds']} s, n_iter {info['n_iter']}, convergence warnings "
                f"{info['convergence_warnings']}")
        del X
        for ci, c in enumerate(cfgs):
            name = c["name"]
            if name not in models:
                continue
            pipe = models[name]
            mi = 3 * CONFIGS.index(name) + ei
            tc = time.time()
            # validation pixels: the source of BBSE
            Cv = np.zeros((NC, NC), np.int64)        # [predicted, label]
            for p in c["val"]:
                hard = proba8(pipe, load_emb(data, d, pre, p)).argmax(1)
                L = labels(p).ravel()
                nv = L != VOID
                Cv += np.bincount(hard[nv] * NC + L[nv].astype(np.int64), minlength=NC * NC).reshape(NC, NC)
            # the target: every pixel of the test patches (void kept for the windows)
            npat = len(c["test"])
            P = np.empty((npat * PX, NC))
            lab = np.empty(npat * PX, np.uint8)
            for i, p in enumerate(c["test"]):
                P[i * PX:(i + 1) * PX] = proba8(pipe, load_emb(data, d, pre, p))
                lab[i * PX:(i + 1) * PX] = labels(p).ravel()
            wtruth = window_truth(lab.reshape(npat, SIDE, SIDE))
            psum_dev = float(np.abs(P.sum(1) - 1).max())
            nvm = lab != VOID
            y_t = lab[nvm]
            K = np.bincount(y_t, minlength=NC).astype(np.int64)
            N = int(K.sum())
            pi_t = K / N
            pi_s = np.bincount(fits[name][2], minlength=NC) / fits[name][2].size
            crop = MAIN_CROP.get(c["territory"])
            log(f"{emb} {name}: predicted {npat} test and {len(c['val'])} val patches in {time.time() - tc:.0f} s "
                f"({N} target px), peak RSS {peak_rss_gb():.2f} GB")
            r = {"map_index": mi, "target_nonvoid_pixels": N, "target_pixels": int(lab.size), "test_patches": npat,
                 "val_patches": len(c["val"]), "val_nonvoid_pixels": int(Cv.sum()), "prob_sum_max_dev": psum_dev,
                 "pi_s": pi_s, "pi_t": pi_t, "target_class_counts": K}
            r["uncorrected"] = measures(P, lab, wtruth, None, pi_t, crop)
            wo = weights(pi_t, pi_s)
            r["oracle"] = measures(P, lab, wtruth, wo, pi_t, crop)
            r["oracle"]["weights"] = wo
            r["oracle"]["pathway_gap"] = r["oracle"]["gap_tv_mean_posterior_to_pi_t"]
            r["oracle"]["target_share_in_classes_absent_from_fit"] = float(pi_t[pi_s == 0].sum())
            r["oracle_raises_pixel_accuracy"] = bool(r["oracle"]["pixel_accuracy"] > r["uncorrected"]["pixel_accuracy"])
            pi_em, em_info = em(P, nvm.astype(np.float64), pi_s, f"{emb} {name}")
            r["em"] = {"pi": pi_em, **em_info}
            r["em"]["measures"] = measures(P, lab, wtruth, weights(pi_em, pi_s), pi_t, crop)
            r["em"]["fixed_point_check_tv_mean_corrected_posterior_to_pi_em"] = tv(
                r["em"]["measures"]["mean_posterior"], pi_em)
            del P
            mu = np.array(r["uncorrected"]["predicted_shares"])
            r["bbse"] = bbse(Cv, mu)
            r["tv"] = {"predicted_shares_to_pi_t": tv(mu, pi_t), "pi_s_to_pi_t": tv(pi_s, pi_t),
                       "em_to_pi_t": tv(pi_em, pi_t),
                       "bbse_to_pi_t": tv(r["bbse"]["pi"], pi_t) if r["bbse"] and r["bbse"]["pi"] else None}
            tr = time.time()
            r["random_labels"] = random_labels(y_t, K, pi_em, mi, draws)
            del lab, y_t
            maps[name][emb] = r
            rl = r["random_labels"]
            log(f"{emb} {name}: acc {r['uncorrected']['pixel_accuracy']:.4f} -> oracle {r['oracle']['pixel_accuracy']:.4f}; "
                f"mIoU {r['uncorrected']['miou_8']:.4f} -> {r['oracle']['miou_8']:.4f}; window err "
                f"{r['uncorrected']['window_error_rate']}; gap {r['uncorrected']['gap_tv_mean_posterior_to_pi_t']:.4f} -> "
                f"oracle {r['oracle']['pathway_gap']:.4f}; TV pred {r['tv']['predicted_shares_to_pi_t']:.4f} EM "
                f"{r['tv']['em_to_pi_t']:.4f} ({em_info['iterations']} updates) BBSE {r['tv']['bbse_to_pi_t']}; reject "
                f"{rl['100']['em_rejection_share']:.3f}/{rl['400']['em_rejection_share']:.3f}, size "
                f"{rl['100']['size_bonferroni_sum_exact']:.4f}/{rl['400']['size_bonferroni_sum_exact']:.4f} "
                f"(labels arm {time.time() - tr:.0f} s; map {time.time() - tc:.0f} s, peak RSS {peak_rss_gb():.2f} GB)")
    out["self_checks"]["interval_ends_failing_their_own_check"] = dict(TABLE_CHECKS)
    out["fits"] = fit_info
    out["labels"] = {"patches_read": len(labels.cache), "bad_shape_or_dtype": int(labels.bad)}
    out["maps"] = maps
    out["reproduction_against_exp113_recorded"] = reproduction(maps)
    out["verdicts"] = verdicts(maps, smoke)
    out["not_computed"] = [
        "the reproduction check against exp113's stored window classes (99.9% of windows): they are not an input of "
        "this route; the refitted maps' window error rate, v1.3 pixel mIoU and main-crop truth share are compared "
        "with exp113's recorded numbers instead (reproduction_against_exp113_recorded)",
        "the comparison with exp113's own fit-selection sha256: its value is not an input; this route reports the "
        "selection's hashes in the forms exp113's second route used and checks the selection against its own redraw "
        "of exp113's rule (matches_independent_redraw)",
        "the run's own draws (seed 114 + r): this route draws its own, so the rejection shares agree with the run's to "
        "Monte Carlo error only; the exact rejection probability is reported beside each share",
        "the reported-only map corrected with the label-estimated priors (k_c + 1) / (n + 8), mean over the first 100 "
        "draws",
        "the BBSE-corrected map's measures (the page reports BBSE's distance to pi_t, beside EM's)",
        "digests of the embedding files"]
    out["seconds"] = round(time.time() - T0, 1)
    out["peak_rss_gb"] = round(peak_rss_gb(), 3)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=1, ensure_ascii=False)
        f.write("\n")
    log("verdicts", json.dumps({k: (x.get("holds") if isinstance(x, dict) else x) for k, x in out["verdicts"].items()}))
    log("wrote", a.out, f"peak RSS {peak_rss_gb():.2f} GB")
    return out


if __name__ == "__main__":
    main()
