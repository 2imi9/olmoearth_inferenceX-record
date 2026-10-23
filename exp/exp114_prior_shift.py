#!/usr/bin/env python
"""exp114: prior shift or feature shift? (docs/plan/prior_shift_pathway.md; the page is the specification).

exp113's 18 maps are refitted with exp113's own code (same fit pixels, same pipeline) and checked against its stored
window classes. Per map, on the non-void pixels of the test patches: the oracle prior correction p_s * pi_t / pi_s and
its pathway gap (the mean corrected posterior against the true class shares), EM and BBSE estimates of the class
shares without labels, and random pixel labels with exact Bonferroni intervals testing EM's estimate.

Modes:
    run      refit, correct, estimate and grade (refused unless the page is frozen); writes exp/out/exp114_summary.json
    smoke    the same code on a few patches per configuration and a small fit set; nothing is graded or written to
             exp/out

    uv run --with scikit-learn python exp/exp114_prior_shift.py run --data DIR --e113-private DIR
"""
import argparse
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
from oe_inferencex import assess as A, estimate as est  # noqa: E402
import exp113_planteur_transfer as E  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "prior_shift_pathway.md")
OUT = os.path.join(EXP_DIR, "out")
C = E.N_CLASSES
SEED = 114
BUDGETS = (100, 400)
DRAWS, REPORT_DRAWS = 1000, 100
LEVEL = 1 - 0.05 / C                     # Bonferroni over the eight classes
EM_TOL, EM_MAX = 1e-10, 10_000
GRADED = "oe13"
COLLAPSED = ("Guyane", "Réunion", "Mayotte")
MAIN = {"loio:Guyane": (1,), "loio:Réunion": (4,), "loio:Mayotte": (1, 2, 3, 4, 5, 6, 7)}
REPRO_AGREEMENT = 0.999
# bars (the page's)
P1_GAP, P1_RATIO, P1_TERRITORIES = 0.05, 3.0, 2
P2_MAPS = 10
P3_TV, P3_POWER = 0.05, 0.8
P4_SIZE = 0.05


def log(*a):
    print(*a, flush=True)


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        return "frozen" if "**Status: frozen" in f.read() else "draft"


def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())


def shares(counts):
    counts = np.asarray(counts, float)
    return counts / counts.sum()


def val_ids(data):
    """{config: validation patch ids}: loio.json's val per fold; the default split's Fold 4."""
    with open(os.path.join(data, "splits", "loio.json"), encoding="utf-8") as f:
        L = json.load(f)
    out = {f"loio:{t}": sorted(int(i) for i in L["folds"][t]["val"]) for t in E.TERRITORIES}
    with open(os.path.join(data, "metadata.geojson"), encoding="utf-8") as f:
        G = json.load(f)
    out["default"] = sorted(int(ft["properties"]["ID_PATCH"]) for ft in G["features"] if int(ft["properties"]["Fold"]) == 4)
    return out


# ----------------------------------------------------------------------------- estimators
def correct(P, w):
    """Posteriors P (N, C) reweighted by w (C,) and renormalised, in float64."""
    Q = P.astype(np.float64) * w
    Q /= Q.sum(1, keepdims=True)
    return Q


def em_priors(P, pi_s):
    """Saerens et al. (2002): pi <- mean over pixels of p_s * pi / pi_s, renormalised, from pi_s. Written with two
    matrix-vector products per step: with q_i = p_i / pi_s, the new pi_c is pi_c * mean_i q_ic / (q_i . pi)."""
    Q = np.divide(P.astype(np.float64), pi_s, out=np.zeros(P.shape), where=pi_s > 0)   # a class the fit lacked stays 0
    pi = pi_s.astype(np.float64).copy()
    for it in range(1, EM_MAX + 1):
        r = Q @ pi
        new = pi * (Q.T @ (1.0 / r)) / Q.shape[0]
        new /= new.sum()
        change = float(np.max(np.abs(new - pi)))
        pi = new
        if change < EM_TOL:
            return pi, it, True, change
    return pi, EM_MAX, False, change


def bbse_priors(conf_joint, mu_t, pi_v):
    """Lipton et al. (2018): C w = mu by least squares, w clipped at 0, pi_t(j) proportional to w_j pi_v(j)."""
    w, *_ = np.linalg.lstsq(conf_joint, mu_t, rcond=None)
    clipped = int((w < 0).sum())
    w = np.clip(w, 0, None)
    pi = w * pi_v
    cond = float(np.linalg.cond(conf_joint)) if np.isfinite(conf_joint).all() else float("inf")
    return pi / pi.sum(), clipped, cond


def pixel_measures(P, y, w, main=None):
    """The map with posteriors P reweighted by w (w = 1: the map itself), against labels y."""
    Q = correct(P, w)
    pred = Q.argmax(1)
    cm = np.bincount(y.astype(np.int64) * C + pred, minlength=C * C).reshape(C, C)      # rows truth, columns predicted
    tp = np.diag(cm).astype(float)
    truth, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    union = truth + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(C), where=union > 0)
    pa = np.divide(tp, truth, out=np.full(C, np.nan), where=truth > 0)
    out = {"pixel_accuracy": float(tp.sum() / cm.sum()), "miou": float(iou.mean()), "iou": iou.tolist(),
           "producer_accuracy": [None if np.isnan(x) else float(x) for x in pa],
           "predicted_share": (predicted / predicted.sum()).tolist(), "soft_share": Q.mean(0).tolist()}
    if main is not None:
        m = list(main)
        out["main_recovered_share"] = float(predicted[m].sum() / truth[m].sum())
        out["main_producer_accuracy"] = float(tp[m].sum() / truth[m].sum()) if len(m) == 1 else \
            float(cm[np.ix_(m, m)].sum() / truth[m].sum())       # several crops together: any crop called any crop
    return out


def window_theta(patches, tvalid, tcls, w):
    """exp113's window error rate of the map reweighted by w: the package reads each patch's probabilities."""
    err = n = 0
    preds = []
    for probs, ok_t, cls in zip(patches, tvalid, tcls):
        q = probs.astype(np.float64) * w[:, None, None]
        q /= q.sum(0, keepdims=True)
        arr = A.assess_prediction(q, is_logit=False, patch=E.PATCH, budgets=(E.CAPTURE_BUDGET,))["arrays"]
        ok = arr["valid"] & ok_t
        p = arr["pooled_argmax"][ok]
        preds.append(p)
        err += int((p != cls[ok]).sum())
        n += int(ok.sum())
    return err / n, np.concatenate(preds), n


def interval_table(n, N, level=LEVEL):
    """The exact interval for every count k of n, for one population size."""
    return np.array([est.hypergeom_interval(k, n, N, conf=level) for k in range(n + 1)])


def exact_size(K, N, n, table):
    """The exact probability that the per-class Bonferroni test rejects the TRUE shares: by the union bound at most
    the sum over classes of each interval's non-coverage, which is what is computed (an upper bound on the joint
    rejection probability; the page's bar applies to it)."""
    from scipy.stats import hypergeom
    total = 0.0
    k = np.arange(n + 1)
    for Kc in K:
        pmf = hypergeom.pmf(k, N, int(Kc), n)
        cover = (table[:, 0] <= Kc / N) & (Kc / N <= table[:, 1])
        total += float(pmf[~cover].sum())
    return total


# ----------------------------------------------------------------------------- one configuration
def configuration(data, cfg, test_ids, v_ids, models, info, e113_private, draws, report_draws, smoke):
    """Every measure of the page for the three maps of one configuration."""
    res = {}
    labels = {pid: E.label(data, pid) for pid in test_ids}
    y_all = np.concatenate([labels[p][labels[p] != E.VOID].ravel() for p in test_ids]).astype(np.int8)
    K = np.bincount(y_all, minlength=C)
    N = int(y_all.size)
    pi_t = shares(K)
    pi_s = shares([info["fit_classes"].get(str(k), 0) for k in range(C)])
    tw = [E.truth_windows(labels[p]) for p in test_ids]
    tcls, tvalid = [t[0] for t in tw], [t[1] for t in tw]
    main = MAIN.get(cfg)
    tables = {n: interval_table(n, N) for n in BUDGETS}
    for k in E.EMBS:
        t0 = time.time()
        patches = [E.predict_probs(models[k], E.embedding(data, k, pid)) for pid in test_ids]   # float64, as exp113
        P = np.concatenate([pp[:, labels[pid] != E.VOID].T for pp, pid in zip(patches, test_ids)])
        r = {"n_pixels": N, "pi_s": pi_s.tolist(), "pi_t": pi_t.tolist()}
        # the map itself, and the reproduction check against exp113's stored window classes
        ones = np.ones(C)
        r["uncorrected"] = pixel_measures(P, y_all, ones, main)
        theta, wpred, nwin = window_theta(patches, tvalid, tcls, ones)
        r["uncorrected"]["window_error_rate"] = theta
        r["n_windows"] = nwin
        if not smoke:
            ref = np.load(os.path.join(e113_private, f"exp113_windows_{cfg.replace(':', '_')}_{k}.npz"))
            if ref["pred"].size != wpred.size:
                raise SystemExit(f"exp114: {cfg} {k}: {wpred.size} windows against exp113's {ref['pred'].size}")
            agree = float((ref["pred"] == wpred).mean())
            r["reproduction_window_agreement"] = agree
            if agree < REPRO_AGREEMENT:
                raise SystemExit(f"exp114: {cfg} {k}: the refitted map agrees with exp113's on {agree:.5f} of windows")
        r["pathway_gap_uncorrected"] = tv(r["uncorrected"]["soft_share"], pi_t)
        r["tv_predicted_shares"] = tv(r["uncorrected"]["predicted_share"], pi_t)
        r["tv_fit_shares"] = tv(pi_s, pi_t)
        # arm 1: the oracle correction
        w_or = np.divide(pi_t, pi_s, out=np.zeros(C), where=pi_s > 0)
        r["oracle"] = pixel_measures(P, y_all, w_or, main)
        r["oracle"]["window_error_rate"] = window_theta(patches, tvalid, tcls, w_or)[0]
        r["pathway_gap"] = tv(r["oracle"]["soft_share"], pi_t)
        # arm 2: label-free priors
        pi_em, it, conv, change = em_priors(P, pi_s)
        r["em"] = {"pi": pi_em.tolist(), "iterations": it, "converged": conv, "last_change": change,
                   "tv_to_pi_t": tv(pi_em, pi_t)}
        w_em = np.divide(pi_em, pi_s, out=np.zeros(C), where=pi_s > 0)
        r["em"]["corrected"] = pixel_measures(P, y_all, w_em, main)
        r["em"]["corrected"]["window_error_rate"] = window_theta(patches, tvalid, tcls, w_em)[0]
        cj = np.zeros((C, C))
        pi_v_counts = np.zeros(C)
        for pid in v_ids:
            lab = E.label(data, pid)
            nv = lab != E.VOID
            pv = E.predict_probs(models[k], E.embedding(data, k, pid)).argmax(0)
            np.add.at(cj, (pv[nv], lab[nv].astype(int)), 1)
            pi_v_counts += np.bincount(lab[nv].ravel().astype(int), minlength=C)
        n_v = cj.sum()
        mu_t = np.asarray(r["uncorrected"]["predicted_share"])
        pi_b, clipped, cond = bbse_priors(cj / n_v, mu_t, pi_v_counts / pi_v_counts.sum())
        r["bbse"] = {"pi": pi_b.tolist(), "tv_to_pi_t": tv(pi_b, pi_t), "weights_clipped": clipped,
                     "condition_number": cond, "validation_pixels": int(n_v)}
        # arm 3: random labels and the pathway test; P4's exact size
        r["random"] = {}
        for n in BUDGETS:
            tab = tables[n]
            rejections = 0
            corr = []
            for d in range(draws):
                pick = np.random.default_rng(SEED + d).choice(N, n, replace=False)
                kc = np.bincount(y_all[pick], minlength=C)
                lo, hi = tab[kc, 0], tab[kc, 1]
                rejections += int(((pi_em < lo) | (pi_em > hi)).any())
                if d < report_draws:
                    pl = (kc + 1.0) / (n + C)
                    m = pixel_measures(P, y_all, np.divide(pl, pi_s, out=np.zeros(C), where=pi_s > 0), main)
                    corr.append(m)
            r["random"][str(n)] = {
                "draws": draws, "reject_em": rejections / draws, "size_union_bound": exact_size(K, N, n, tab),
                "label_corrected_mean": {f: float(np.mean([m[f] for m in corr])) for f in ("pixel_accuracy", "miou")}
                | ({"main_recovered_share": float(np.mean([m["main_recovered_share"] for m in corr])),
                    "main_producer_accuracy": float(np.mean([m["main_producer_accuracy"] for m in corr]))} if main else {})}
        r["seconds"] = round(time.time() - t0, 1)
        res[k] = r
        log(f"[exp114] {cfg} {k}: N {N} gap {r['pathway_gap']:.4f} (uncorrected {r['pathway_gap_uncorrected']:.4f}) "
            f"acc {r['uncorrected']['pixel_accuracy']:.4f}->{r['oracle']['pixel_accuracy']:.4f} "
            f"EM tv {r['em']['tv_to_pi_t']:.4f} ({it} it) pred tv {r['tv_predicted_shares']:.4f} "
            f"BBSE tv {r['bbse']['tv_to_pi_t']:.4f} reject400 {r['random']['400']['reject_em']:.3f} "
            f"size400 {r['random']['400']['size_union_bound']:.4f} {r['seconds']} s")
        del P, patches
    return res


def verdicts(S):
    m = S["maps"]
    out = {}
    dflt = m["default"][GRADED]["pathway_gap"]
    per = {t: {"gap": m[f"loio:{t}"][GRADED]["pathway_gap"]} for t in COLLAPSED}
    for t, v in per.items():
        v["meets"] = v["gap"] >= P1_GAP and v["gap"] >= P1_RATIO * dflt
    out["P1"] = {"default_gap": dflt, "territories": per, "meeting": sum(v["meets"] for v in per.values()),
                 "holds": sum(v["meets"] for v in per.values()) >= P1_TERRITORIES}
    held = [(t, k) for t in E.TERRITORIES for k in E.EMBS]
    better = [f"{t}|{k}" for t, k in held if m[f"loio:{t}"][k]["em"]["tv_to_pi_t"] < m[f"loio:{t}"][k]["tv_predicted_shares"]]
    out["P2"] = {"em_closer": better, "count": len(better), "holds": len(better) >= P2_MAPS}
    far = {t: m[f"loio:{t}"][GRADED]["random"]["400"]["reject_em"] for t in E.TERRITORIES
           if m[f"loio:{t}"][GRADED]["em"]["tv_to_pi_t"] >= P3_TV}
    out["P3"] = {"maps": far, "determined": bool(far), "holds": (all(v >= P3_POWER for v in far.values()) if far else None)}
    sizes = [r["random"][str(n)]["size_union_bound"] for c in m.values() for r in c.values() for n in BUDGETS]
    out["P4"] = {"max_size_union_bound": max(sizes), "cells": len(sizes), "holds": max(sizes) <= P4_SIZE}
    return out


def run(data, e113_private, smoke):
    import sklearn
    status = prereg_status()
    if not smoke and status != "frozen":
        raise SystemExit("exp114: the preregistration page is not frozen; refusing to grade")
    configs = E.splits(data)
    vals = val_ids(data)
    if smoke:
        configs = {c: (tr[:60], te[:6]) for c, (tr, te) in configs.items()}
        vals = {c: v[:4] for c, v in vals.items()}
    n_fit = 5_000 if smoke else E.FIT_PIXELS
    models, info = E.fit_maps(data, configs, n_fit)
    S = {"page": os.path.relpath(PLAN, ROOT), "package": oe_inferencex.__version__, "smoke": smoke,
         "environment": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
         "fit": {}, "maps": {}}
    if not smoke:
        with open(os.path.join(OUT, "exp113_summary.json"), encoding="utf-8") as f:
            S113 = json.load(f)
        for c in configs:
            if info[c]["selection_sha256"] != S113["fit"][c]["selection_sha256"]:
                raise SystemExit(f"exp114: {c}: fit selection differs from exp113's")
    for c in configs:
        S["fit"][c] = {k: info[c][k] for k in ("fit_pixels", "selection_sha256", "fit_classes")}
        S["maps"][c] = configuration(data, c, sorted(configs[c][1]), vals[c], models[c], info[c], e113_private,
                                     20 if smoke else DRAWS, 5 if smoke else REPORT_DRAWS, smoke)
    if not smoke:
        S["verdicts"] = verdicts(S)
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run", "smoke"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--e113-private", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = run(a.data, a.e113_private, a.mode == "smoke")
    if a.mode == "smoke":
        json.dumps(S)
        log(f"[exp114] smoke done in {time.time() - t0:.0f} s")
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp114_summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1, ensure_ascii=False)
    log(json.dumps(S["verdicts"], indent=1, ensure_ascii=False))
    log(f"[exp114] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
