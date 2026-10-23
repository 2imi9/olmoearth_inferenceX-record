#!/usr/bin/env python
"""exp113: train on five territories, map the sixth (docs/plan/planteur_transfer.md; the page is the specification).

PLANTEUR (Ai2, ODC-BY) leave-one-island-out and its default split, with three 2019 embeddings on the label grid
(OlmoEarth v1.3, TESSERA v2, AlphaEarth). Per configuration and embedding, StandardScaler + LogisticRegression on
100,000 random non-void train pixels, applied to every pixel of the test patches; the package's assess_prediction reads
each test patch's probabilities into 4 x 4 windows (confidence, class, boundary indicator), exactly as the tool does.

Modes:
    fetch    download the labels, the three embeddings, metadata.geojson and the splits (about 58 GB) to --data
    run      fit, predict and grade (refused unless the page is frozen); writes exp/out/exp113_summary.json and the
             private per-window arrays to --private-dir
    smoke    the same code on a few patches per configuration and a small fit set; nothing is graded or written to
             exp/out

    uv run --with scikit-learn --with huggingface_hub python exp/exp113_planteur_transfer.py run \
        --data DIR --private-dir DIR
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import oe_inferencex  # noqa: E402
from oe_inferencex import assess as A, estimate as est, metrics, plan  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "planteur_transfer.md")
OUT = os.path.join(EXP_DIR, "out")
REPO = "allenai/planteur-benchmark"
TERRITORIES = ("Guadeloupe", "Martinique", "Guyane", "Réunion", "Mayotte")
EMBS = {"oe13": ("EMB_OLMOEARTH_V1_3", "OE13_{}.npy", 128),
        "tessera2": ("EMB_TESSERA_V2", "TESSERA_{}.npy", 128),
        "aef": ("EMB_AEF", "AEF_{}.npy", 64)}
GRADED = "oe13"
PAIRS = (("oe13", "tessera2"), ("oe13", "aef"), ("tessera2", "aef"))
N_CLASSES, VOID, PATCH, SIZE = 8, 255, 4, 128
FIT_PIXELS, SEED = 100_000, 113
DRAWS, CERTIFY_BUDGET, VERIFY = 2000, 400, 20
BUDGETS = (100, 400)
ALPHA, ALPHA_REPORTED, DELTA = 0.05, 0.10, 0.10
CAPTURE_BUDGET = 0.10
BOOT = 2000
# bars (the page's)
P1_AUROC, P1_CAPTURE, P1_LEAD, P1_TERRITORIES, P1_ERROR_FLOOR = 0.70, 0.25, 0.05, 4, 200
P2_DROP = 0.05
P3_COVERAGE = 0.95
P4_VIOLATION = 0.115
P5_WRONG = 0.025
DEFAULT_TRAIN, DEFAULT_TEST, DEFAULT_SIZES = 1, 5, (1896, 443)


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


# ----------------------------------------------------------------------------- data
def fetch(data):
    from huggingface_hub import snapshot_download
    pats = ["ANNOTATIONS/*", "metadata.geojson", "splits/*", "EMB_MISSING.json", "README.md"]
    pats += [f"{d}/*" for d, _, _ in EMBS.values()]
    snapshot_download(REPO, repo_type="dataset", local_dir=data, allow_patterns=pats, max_workers=16)
    counts = {d: len(os.listdir(os.path.join(data, d))) for d in ["ANNOTATIONS"] + [d for d, _, _ in EMBS.values()]}
    log(f"[exp113] fetched: {counts}")
    bad = {d: n for d, n in counts.items() if n < 2750}
    if bad:
        raise SystemExit(f"exp113: incomplete download {bad}")


def splits(data):
    """{config: (train ids, test ids)}: the five leave-one-island-out folds and the default split."""
    with open(os.path.join(data, "splits", "loio.json"), encoding="utf-8") as f:
        L = json.load(f)
    out = {}
    for t in TERRITORIES:
        fold = L["folds"][t]
        out[f"loio:{t}"] = ([int(i) for i in fold["train"]], [int(i) for i in fold["test"]])
    with open(os.path.join(data, "metadata.geojson"), encoding="utf-8") as f:
        G = json.load(f)
    tr, te = [], []
    for feat in G["features"]:
        pr = feat["properties"]
        pid = int(pr.get("ID_PATCH", pr.get("id_patch", pr.get("id"))))
        fold = int(pr.get("Fold", pr.get("fold")))
        if fold == DEFAULT_TRAIN:
            tr.append(pid)
        elif fold == DEFAULT_TEST:
            te.append(pid)
    if (len(tr), len(te)) != DEFAULT_SIZES:
        raise SystemExit(f"exp113: default split has {len(tr)} train and {len(te)} test patches, not {DEFAULT_SIZES}")
    out["default"] = (sorted(tr), sorted(te))
    island = {int(f["properties"]["ID_PATCH"]): f["properties"].get("Island") for f in G["features"]}
    for t in TERRITORIES:
        tr_t, te_t = out[f"loio:{t}"]
        if set(te_t) != {p for p, i in island.items() if i == t} or any(island.get(p) == t for p in tr_t):
            raise SystemExit(f"exp113: the leave-one-island-out fold of {t} does not hold out exactly its patches")
    return out


def label(data, pid):
    a = np.load(os.path.join(data, "ANNOTATIONS", f"TARGET_{pid}.npy"))
    if a.shape != (1, SIZE, SIZE) or a.dtype != np.uint8 or not np.isin(a, list(range(N_CLASSES)) + [VOID]).all():
        raise SystemExit(f"exp113: label {pid} has shape {a.shape}, dtype {a.dtype} or values outside 0-7 and 255")
    return a[0]


def embedding(data, key, pid):
    d, name, c = EMBS[key]
    e = np.load(os.path.join(data, d, name.format(pid)))
    if e.shape != (c, SIZE, SIZE) or not np.isfinite(e).all():
        raise SystemExit(f"exp113: {key} patch {pid} has shape {e.shape} or non-finite values")
    return e


# ----------------------------------------------------------------------------- windows
def truth_windows(lab):
    """(class per window, valid): the majority of its non-void pixels; more than half void, or a tied majority, left out."""
    w = lab.reshape(SIZE // PATCH, PATCH, SIZE // PATCH, PATCH).transpose(0, 2, 1, 3).reshape(SIZE // PATCH, SIZE // PATCH, -1)
    counts = np.stack([(w == k).sum(-1) for k in range(N_CLASSES)], -1)
    nonvoid = counts.sum(-1)
    top = counts.max(-1)
    tied = (counts == top[..., None]).sum(-1) > 1
    valid = (nonvoid >= (PATCH * PATCH) / 2) & ~tied
    return counts.argmax(-1), valid


def window_mean(e):
    c = e.shape[0]
    return e.reshape(c, SIZE // PATCH, PATCH, SIZE // PATCH, PATCH).mean(axis=(2, 4))


def band_windows(pix_class):
    """Windows held mostly by the one-pixel class-edge band of the predicted map (1), the interior (0), or split (-1),
    by the package's band definition (a pixel with a pixel of another class among its eight neighbours) and its
    condition pooling (the majority of the window's pixels; a tie unrecorded)."""
    p = np.pad(pix_class, 1, mode="edge")
    edge = np.zeros(pix_class.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                edge |= p[1 + dy:1 + dy + SIZE, 1 + dx:1 + dx + SIZE] != pix_class
    s = edge.reshape(SIZE // PATCH, PATCH, SIZE // PATCH, PATCH).sum(axis=(1, 3))
    half = PATCH * PATCH / 2
    return np.where(s > half, 1, np.where(s < half, 0, -1))


# ----------------------------------------------------------------------------- the maps
def fit_selection(data, train_ids, n_pixels, seed):
    """{patch id: flat pixel indices}: n_pixels drawn uniformly without replacement from the non-void train pixels."""
    nonvoid = {pid: np.flatnonzero(label(data, pid).ravel() != VOID) for pid in train_ids}
    sizes = np.array([nonvoid[p].size for p in train_ids])
    total = int(sizes.sum())
    pick = np.sort(np.random.default_rng(seed).choice(total, min(n_pixels, total), replace=False))
    edges = np.concatenate([[0], np.cumsum(sizes)])
    owner = np.searchsorted(edges, pick, side="right") - 1
    out = {}
    for k in np.unique(owner):
        pid = train_ids[k]
        out[pid] = nonvoid[pid][pick[owner == k] - edges[k]]
    return out, total


def fit_maps(data, configs, n_pixels, private=None):
    """{config: {emb: (pipeline, fit mean vector)}} and the fit sets' sizes; each train patch is read once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sel, info = {}, {}
    for c, (tr, _) in configs.items():
        sel[c], total = fit_selection(data, tr, n_pixels, SEED)
        info[c] = {"train_patches": len(tr), "nonvoid_train_pixels": total, "fit_pixels": int(sum(v.size for v in sel[c].values())),
                   "draw_order": "the configuration's train list in its file order (loio.json; the default split ascending), "
                                 "each patch's non-void pixels row-major; choice(total, n, replace=False) from default_rng(113)"}
        pt = np.repeat(np.array(list(sel[c]), np.int64), [v.size for v in sel[c].values()])
        px = np.concatenate(list(sel[c].values())).astype(np.int64)
        info[c]["selection_sha256"] = hashlib.sha256(pt.tobytes() + px.tobytes()).hexdigest()
        if private:
            np.savez_compressed(os.path.join(private, f"exp113_fitsel_{c.replace(':', '_')}.npz"), patch=pt, pixel=px)
    X = {(c, k): [] for c in configs for k in EMBS}
    Y = {c: [] for c in configs}
    for pid in sorted({p for s in sel.values() for p in s}):
        lab = label(data, pid).ravel()
        es = {k: embedding(data, k, pid).reshape(EMBS[k][2], -1) for k in EMBS}
        for c in configs:
            if pid in sel[c]:
                idx = sel[c][pid]
                Y[c].append(lab[idx])
                for k in EMBS:
                    X[(c, k)].append(es[k][:, idx].T)
    models = {c: {} for c in configs}
    for c in configs:
        y = np.concatenate(Y[c])
        info[c]["fit_classes"] = {str(int(k)): int((y == k).sum()) for k in np.unique(y)}
        for k in EMBS:
            Xk = np.concatenate(X[(c, k)]).astype(np.float64)
            t0 = time.time()
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xk, y)
            info[c][f"fit_seconds_{k}"] = round(time.time() - t0, 1)
            info[c][f"n_iter_{k}"] = int(np.max(m[-1].n_iter_))
            info[c][f"converged_{k}"] = bool(np.max(m[-1].n_iter_) < 2000)
            models[c][k] = (m, Xk.mean(0))
        log(f"[exp113] {c}: fitted three maps on {y.size} pixels")
    return models, info


def predict_probs(model, e):
    """(N_CLASSES, H, W) probabilities; a class the fit set lacked gets probability 0."""
    m, _ = model
    c = e.shape[0]
    p = m.predict_proba(e.reshape(c, -1).T.astype(np.float64))
    full = np.zeros((p.shape[0], N_CLASSES))
    full[:, m[-1].classes_.astype(int)] = p
    return full.T.reshape(N_CLASSES, SIZE, SIZE)


def predict_all(data, configs, models, private):
    """Per (config, emb), the windows of the test patches: confidence, predicted class, true class, patch, the three
    controls' inputs, the band; and each map's pixel confusion. Each test patch is read once."""
    tests = {c: set(te) for c, (_, te) in configs.items()}
    pops = {(c, k): {f: [] for f in ("conf", "pred", "true", "patch", "win", "bnd", "dist", "band")} for c in configs for k in EMBS}
    conf_px = {(c, k): np.zeros((N_CLASSES, N_CLASSES), np.int64) for c in configs for k in EMBS}
    left_out = {c: 0 for c in configs}
    for pid in sorted(set().union(*tests.values())):
        lab = label(data, pid)
        tcls, tvalid = truth_windows(lab)
        es = {k: embedding(data, k, pid) for k in EMBS}
        for c in configs:
            if pid not in tests[c]:
                continue
            for k in EMBS:
                probs = predict_probs(models[c][k], es[k])
                out = A.assess_prediction(probs, is_logit=False, patch=PATCH, budgets=(CAPTURE_BUDGET,))
                arr = out["arrays"]
                ok = arr["valid"] & tvalid
                if k == GRADED:
                    left_out[c] += int((arr["valid"] & ~tvalid).sum())
                wm = window_mean(es[k])
                dist = np.linalg.norm(wm - models[c][k][1][:, None, None], axis=0)
                pix = probs.argmax(0)
                P = pops[(c, k)]
                P["conf"].append(arr["confidence"][ok])
                P["pred"].append(arr["pooled_argmax"][ok])
                P["true"].append(tcls[ok])
                P["patch"].append(np.full(int(ok.sum()), pid))
                P["win"].append(np.flatnonzero(ok.ravel()))
                P["bnd"].append(arr["boundary"][ok])
                P["dist"].append(dist[ok])
                P["band"].append(band_windows(pix)[ok])
                nv = lab != VOID
                np.add.at(conf_px[(c, k)], (lab[nv].astype(int), pix[nv]), 1)
    for key, P in pops.items():
        for f in P:
            P[f] = np.concatenate(P[f])
        if private:
            np.savez_compressed(os.path.join(private, f"exp113_windows_{key[0].replace(':', '_')}_{key[1]}.npz"), **P)
    return pops, conf_px, left_out


# ----------------------------------------------------------------------------- grading
def auroc(u, e):
    return float(metrics.weighted_auroc(np.asarray(u, float), np.asarray(e, bool), np.ones(len(e))))


def rarity(pred):
    out = np.empty(pred.shape)
    for k in np.unique(pred):
        out[pred == k] = -float((pred == k).mean())
    return out


def pair_matrix(u, err, patch, pids):
    """M[p, q]: pairs (error in patch p, correct in patch q) with the error ranked above, ties half; for the patch
    bootstrap of the AUROC, which is w'Mw / ((w'E)(w'C)) for patch multiplicities w."""
    index = {p: i for i, p in enumerate(pids)}
    pi = np.array([index[p] for p in patch])
    P = len(pids)
    M = np.zeros((P, P))
    ue, pe = u[err], pi[err]
    for q in range(P):
        cq = np.sort(u[(~err) & (pi == q)])
        if cq.size == 0:
            continue
        below = np.searchsorted(cq, ue, side="left")
        equal = np.searchsorted(cq, ue, side="right") - below
        M[:, q] = np.bincount(pe, weights=below + 0.5 * equal, minlength=P)
    E = np.bincount(pi[err], minlength=P).astype(float)
    C = np.bincount(pi[~err], minlength=P).astype(float)
    return M, E, C


def boot_auroc(M, E, C, W):
    return ((W @ M) * W).sum(1) / ((W @ E) * (W @ C))


def arm1(P, boot):
    err = P["pred"] != P["true"]
    u = -P["conf"]
    own = {"auroc": auroc(u, err),
           "capture_10": float(metrics.capture_at_budget_expected(u, err, (CAPTURE_BUDGET,))[CAPTURE_BUDGET]),
           "excess_aurc": float(metrics.excess_aurc(u, err.astype(float)))}
    ctl = {"rarity": rarity(P["pred"]), "distance": P["dist"].astype(float), "boundary": P["bnd"].astype(float)}
    ctl_stats = {n: {"auroc": auroc(v, err),
                     "capture_10": float(metrics.capture_at_budget_expected(v, err, (CAPTURE_BUDGET,))[CAPTURE_BUDGET])}
                 for n, v in ctl.items()}
    best = max(ctl_stats, key=lambda n: ctl_stats[n]["auroc"])
    correct = ~err
    out = {"n": int(err.size), "errors": int(err.sum()), "theta": float(err.mean()), "own": own,
           "controls": ctl_stats, "best_control": best, "lead": own["auroc"] - ctl_stats[best]["auroc"],
           "ceiling_10": float(metrics.attainable_ceiling(CAPTURE_BUDGET, float(err.mean()), int(err.size))) if err.any() else None,
           "confident_error_share": float((P["conf"][err] >= np.median(P["conf"][correct])).mean())
           if err.any() and correct.any() else None,
           "band": {name: {"windows": int((P["band"] == v).sum()),
                           "error_rate": float(err[P["band"] == v].mean()) if (P["band"] == v).any() else None}
                    for name, v in (("interior", 0), ("edge_band", 1), ("split", -1))}}
    if boot:
        pids = sorted(set(P["patch"].tolist()))
        Mo, E, C = pair_matrix(u, err, P["patch"], pids)
        Mc, _, _ = pair_matrix(ctl[best], err, P["patch"], pids)
        W = np.random.default_rng(SEED).multinomial(len(pids), np.full(len(pids), 1 / len(pids)), size=BOOT).astype(float)
        a, b = boot_auroc(Mo, E, C, W), boot_auroc(Mc, E, C, W)
        q = lambda v: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]          # noqa: E731
        out["boot90"] = {"auroc": q(a), "lead": q(a - b), "patches": len(pids), "resamples": BOOT, "seed": SEED,
                         "full_matches": bool(abs(float(boot_auroc(Mo, E, C, np.ones((1, len(pids))))[0]) - own["auroc"]) < 1e-9)}
        if not out["boot90"]["full_matches"]:
            raise SystemExit("exp113: the patch pair matrix does not reproduce the AUROC")
    return out


def arm2(P):
    err = P["pred"] != P["true"]
    N, K = int(err.size), int(err.sum())
    return {str(B): est.exact_coverage_srs(N, K, B, interval=est.hypergeom_interval) for B in BUDGETS}


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
    """exp110's and exp112's certify_steps: certify_zone's prefix rule from Z, the review-set guard included."""
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


def arm3(P, draws, name):
    margin = P["conf"].astype(np.float64)
    w = (P["pred"] != P["true"]).astype(np.float64)
    Z = zone_setup(margin, w)
    N = margin.size
    zones = {a: [] for a in (ALPHA, ALPHA_REPORTED)}
    checked = 0
    for r in range(draws):
        idx = np.random.default_rng(SEED + r).choice(N, CERTIFY_BUDGET, replace=False)
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
                    raise SystemExit(f"exp113 {name}: certify steps {mine} differ from certify_zone {got} at draw {r}")
                checked += 1
            zones[a].append(z)
    out = {"budget": CERTIFY_BUDGET, "draws": draws, "certify_zone_checked_on_draws": checked, "by_alpha": {}}
    for a, zs in zones.items():
        c = np.array([z["coverage"] if z["coverage"] is not None else 0.0 for z in zs])
        viol = np.array([z["coverage"] is not None and z["zone_error"] > a for z in zs])
        zerr = [z["zone_error"] for z in zs if z["coverage"] is not None]
        out["by_alpha"][f"{a:g}"] = {
            "p_certify_any": float(np.mean([z["coverage"] is not None for z in zs])),
            "refused_as_review_set": int(sum(bool(z.get("refused")) for z in zs)),
            "violation_share": float(viol.mean()), "median_certified_share": float(np.median(c)),
            "certified_share_quantiles": {q: float(np.quantile(c, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "zone_true_error_median": float(np.median(zerr)) if zerr else None}
    return out


def arm4(pa, pb):
    """Two maps' windows of one territory, aligned (the same windows, as both exclude only by the truth)."""
    if not all(np.array_equal(pa[f], pb[f]) for f in ("patch", "win", "true")):
        raise SystemExit("exp113: two maps' windows are not aligned")
    a, b, t = pa["pred"], pb["pred"], pa["true"]
    N, D = int(a.size), int((a != b).sum())
    KA, KB = int(((a == t) & (b != t)).sum()), int(((b == t) & (a != t)).sum())
    out = {"N": N, "D": D, "K_A": KA, "K_B": KB, "acc_A": float((a == t).mean()), "acc_B": float((b == t).mean()), "n": {}}
    better = "A" if KA > KB else "B" if KB > KA else None
    for n in BUDGETS:
        pA, pB = plan.which_map_probability(N, D, min(n, D), KA, KB) if D else (0.0, 0.0)
        out["n"][str(n)] = {"p_names_A": float(pA), "p_names_B": float(pB),
                            "p_names_worse": None if better is None else float(pB if better == "A" else pA),
                            "p_names_better": None if better is None else float(pA if better == "A" else pB)}
    out["better"] = better
    return out


def miou(cm):
    ious = []
    for k in range(N_CLASSES):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k, :].sum() - cm[k, k]
        ious.append(float(tp / (tp + fp + fn)) if cm[k, :].sum() > 0 and (tp + fp + fn) > 0 else 0.0)
    return float(np.mean(ious)), ious


def verdicts(S, territories=TERRITORIES):
    out = {}
    per, below = {}, []
    for t in territories:
        a = S["maps"][f"loio:{t}"][GRADED]["arm1"]
        if a["errors"] < P1_ERROR_FLOOR:
            below.append(t)
            continue
        per[t] = {"a": a["own"]["auroc"] >= P1_AUROC and a["own"]["capture_10"] >= P1_CAPTURE, "b": a["lead"] >= P1_LEAD}
    met = [t for t, v in per.items() if v["a"] and v["b"]]
    out["P1"] = {"per_territory": per, "below_floor": below, "territories_meeting": met,
                 "territories_meeting_a": [t for t, v in per.items() if v["a"]],
                 "a_holds": len(below) < 2 and sum(v["a"] for v in per.values()) >= P1_TERRITORIES,
                 "holds": len(below) < 2 and len(met) >= P1_TERRITORIES}
    loio = float(np.mean([S["maps"][f"loio:{t}"][GRADED]["arm1"]["own"]["auroc"] for t in territories]))
    dflt = S["maps"]["default"][GRADED]["arm1"]["own"]["auroc"]
    out["P2"] = {"loio_mean_auroc": loio, "default_auroc": dflt, "drop": dflt - loio, "holds": dflt - loio >= P2_DROP}
    cov = [v for c in S["maps"].values() for m in c.values() for v in m["arm2"].values()]
    out["P3"] = {"min_exact_coverage": min(cov), "holds": min(cov) >= P3_COVERAGE}
    viol = [S["maps"][f"loio:{t}"][k]["arm3"]["by_alpha"][f"{ALPHA:g}"]["violation_share"] for t in territories for k in EMBS]
    out["P4"] = {"max_violation_share": max(viol), "holds": max(viol) <= P4_VIOLATION}
    worse = [v["p_names_worse"] for t in territories for pr in S["which_map"][t].values() for v in pr["n"].values()
             if v["p_names_worse"] is not None]
    out["P5"] = {"max_p_names_worse": max(worse) if worse else None, "holds": all(x <= P5_WRONG for x in worse)}
    return out


def run(data, private, smoke):
    if not smoke and prereg_status() != "frozen":
        raise SystemExit("exp113: docs/plan/planteur_transfer.md is not frozen; no map before it is")
    t0 = time.time()
    configs = splits(data)
    n_pixels, draws = FIT_PIXELS, DRAWS
    if smoke:
        configs = {c: (configs[c][0][:3], configs[c][1][:2]) for c in (f"loio:{TERRITORIES[0]}", "default")}
        n_pixels, draws = 2000, 5
    models, info = fit_maps(data, configs, n_pixels, None if smoke else private)
    if not smoke and any(v["fit_pixels"] != FIT_PIXELS for v in info.values()):
        raise SystemExit("exp113: a fit set is smaller than the page's 100,000 pixels")
    pops, conf_px, left_out = predict_all(data, configs, models, None if smoke else private)
    import platform
    import sklearn
    S = {"page": "docs/plan/planteur_transfer.md", "package": oe_inferencex.__version__, "smoke": smoke,
         "environment": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "fit": info, "windows_left_out_by_the_truth": left_out, "maps": {}, "which_map": {}}
    for c in configs:
        S["maps"][c] = {}
        for k in EMBS:
            P = pops[(c, k)]
            m = {"arm1": arm1(P, boot=(k == GRADED and c.startswith("loio:"))), "arm2": arm2(P)}
            m["miou"], m["iou_per_class"] = miou(conf_px[(c, k)])
            m["truth_pixels_per_class"] = conf_px[(c, k)].sum(1).tolist()
            present = [v for v, n in zip(m["iou_per_class"], m["truth_pixels_per_class"]) if n > 0]
            m["miou_classes_present"] = float(np.mean(present)) if present else None
            m["arm3"] = arm3(P, draws, f"{c}/{k}")
            S["maps"][c][k] = m
            log(f"[exp113] {c} {k}: theta {m['arm1']['theta']:.4f}, AUROC {m['arm1']['own']['auroc']:.4f}, "
                f"mIoU {m['miou']:.3f}")
        if c.startswith("loio:"):
            S["which_map"][c.split(":", 1)[1]] = {f"{a}_vs_{b}": arm4(pops[(c, a)], pops[(c, b)]) for a, b in PAIRS}
    S["best_agreeing_map"] = {t: max(EMBS, key=lambda k: 1 - S["maps"][f"loio:{t}"][k]["arm1"]["theta"])
                              for t in TERRITORIES if f"loio:{t}" in configs}
    if smoke:
        # the first run (job 1467772) failed here, in verdicts, on a string stored among the pairs; the smoke now runs
        # verdicts on the territories it has
        verdicts(S, [t for t in TERRITORIES if f"loio:{t}" in configs])
        json.dumps(S)
    else:
        S["verdicts"] = verdicts(S)
        if not S["verdicts"]["P3"]["holds"]:
            log("[exp113] P3 FAILED: a defect; the summary is written for diagnosis and must not be recorded")
        S["inputs"] = {"loio_json_sha256": sha256_file(os.path.join(data, "splits", "loio.json")),
                       "metadata_sha256": sha256_file(os.path.join(data, "metadata.geojson"))}
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "exp113_summary.json"), "w") as f:
            json.dump(S, f, indent=1)
        log(json.dumps(S["verdicts"], indent=1))
    log(f"[exp113] done in {time.time() - t0:.0f} s")
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("fetch", "run", "smoke"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--private-dir", default=None)
    args = ap.parse_args(argv)
    if args.mode == "fetch":
        fetch(args.data)
    elif args.mode == "run":
        if not args.private_dir:
            raise SystemExit("exp113 run: --private-dir is required")
        os.makedirs(args.private_dir, exist_ok=True)
        run(args.data, args.private_dir, smoke=False)
    else:
        run(args.data, None, smoke=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
