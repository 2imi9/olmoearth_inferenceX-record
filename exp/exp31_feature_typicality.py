"""exp31: feature-space typicality as a label-free error signal.

exp13's E_dist (mean cosine distance from a patch's pooled Base feature to
its five nearest patches of the head's training scene) found 13/27 against
confidence, sign p = 1.00, and on hand labels was the worst signal tested
(22/329, exp18). It conflates two choices: the reference set (the head's 1024
training patches at Katima, a proxy for neither the pretraining distribution
nor the evaluated scene) and the density estimator (a raw k-NN distance).
Roadmap item 9 asks for a formalisation (AOA / dissimilarity index). This
experiment separates the two: does the pooled 768-d Base feature of a 4-px
patch being atypical, relative to a stated reference set, rank the probe's
errors? Three reference sets, three density scores each, plus two
labelled-reference variants. No evaluation label is ever an input.

Reference sets. R1 "training patches of the head": part A the katima
training scene's 1024 pooled patches; part B the valid patches of the 600
valid-split training tiles (~135k) with their training labels. R2 "same
scene": the evaluated unit's own patches (a 32x32 scene in A, a 15x15 tile in
B), cross-fitted over 5 random folds (numpy seed 0 per unit): the reference
is fitted on 4/5 of the unit's patches and scores the held-out fifth, so no
patch is scored against itself; the same code path serves all three
densities. That reference is small (819 / 180 patches, fewer than the 768
feature dimensions in B), so the Gaussian scores need shrinkage and the PCA
dimension is capped. R3 "cross-testbed pool", a proxy for a
pretraining-distribution reference: part A all Sen1Floods11 tiles cached by
exp18 (exp/out/exp18_feats.npz keys tr_base, bolivia_base0, test_base0;
~394k patches from 11 countries); part B the 27 rule scenes' shift-0
features plus katima from exp/out/exp11_feats.npz (~28.7k patches). The real
thing would be a sample of olmoearth_pretrain_dataset; that is the upgrade
path. When a cache is missing (always in --smoke) R3 falls back, with a
WARNING, to the other scenes' shift-0 pooled features plus the training
scene (part A, per scene, excluding the scene itself) or to the part-A
features (part B; the training scene alone if part A was skipped); the
summary records what R3 was and its size.

Scores per reference set R, all higher = more atypical. (1) kNN: mean cosine
distance to the k = 5 nearest reference patches (L2-normalised features,
distance 1 - cosine); for R1 in part A this is exp13's E_dist exactly, and a
real run compares its per-scene E-AURC with exp/out/exp13_corrected_stats.csv.
(2) Mahalanobis: (x - mu)^T S_a^-1 (x - mu) with mu and S from the reference
and S_a = (1 - a) S + a tr(S)/D I, a the closed-form Ledoit-Wolf (2004)
coefficient (own implementation; no sklearn). (3) PCA residual (the residual
term of ViM, Wang et al. 2022): norm of the component of x - mu orthogonal to
the top-d principal directions of the reference, d = min(192, n_ref // 2).
One eigendecomposition of S serves (2) and (3), since S_a shares S's
eigenvectors. Labelled-reference variants on R1 only, where the head's own
training labels are legitimately available: (4) class-conditional
Mahalanobis (Lee et al. 2018): per-class means, shared shrunk covariance,
min over the two classes. (5) ViM (Wang et al. 2022) adapted to the
one-logit binary head: origin o = -pinv(W) b = -b w / ||w||^2, principal
subspace fitted on X_train - o, residual r(x) = ||P_perp (x - o)||, alpha =
sum_train max(0, logit) / sum_train r, virtual logit alpha r, class logits
[0, logit] (a sigmoid is a two-way softmax), score = the log-odds
alpha r - log(1 + exp(logit)) of the virtual class, monotone in the ViM
softmax probability exp(alpha r) / (1 + exp(logit) + exp(alpha r)) without
its saturation at 1.0. ViM fuses confidence and
typicality by construction and is not a pure typicality score. One
diagnostic signal, ||x||_2 of the pooled feature, shows whether any density
is just feature norm. The encoder runs in fp32 (harness); the density fits
and scores are float64 on the same device.

Controls and references (exp/harness_ab.py): confidence (negative absolute
logit), aligned tile-phase, boundary indicator and the NDWI-gradient control;
constant score, S2 within-patch variance and NDWI level as observed-input
controls.

Preregistered, fixed before running. Primary score: "kNN-5 cosine [R2
scene]", the only member of the family that is label-free AND needs no
external data at inference. Combination U+ = "combination conf+typicality
(prereg)" = mean of the within-unit midrank percentiles of confidence and of
the primary score. Inference: per-river mean gain of U+ over confidence,
one-sided exact sign test over the 8 river clusters (7/8 -> p = 0.035).
Everything else is secondary and reported with the same tests.

Evaluation. (A) The 27 rule-selected exp11 scenes against the exp13
disagreement set (seed-0 Base head trained on katima, errors against
WorldCover 2021; scenes with fewer than 8 errors skipped). (B) Sen1Floods11
Bolivia hand labels exactly as exp18 (600 valid-split training tiles, 60x60
crops, same head, same patch-label pooling, per-tile rule of at least three
errors and three correct patches). Both parts compare every signal with
confidence, tile-phase, boundary and the NDWI-gradient control on identical
errors: W/L/T, exact sign tests, a sign-flip permutation test on mean
E-AURC differences (A), and per-unit Spearman of the new signals with the
references (is it new information?).

Inputs. exp/out/exp11_scenes.npz and exp/out/exp03_cache.npz (committed),
exp/out/exp11_feats.npz and exp/out/exp18_feats.npz (gitignored cluster
caches; the harness recomputes pooled features when the first is missing,
part B recomputes with exp18.embed when the second is missing), the Base
checkpoint from the HF cache, data/floods/flood_{valid,bolivia}_data.pt
(downloaded when missing outside --smoke), exp/out/exp13_corrected_stats.csv
for the R1 kNN self-check.

Outputs. exp/out/exp31_feature_typicality.csv (one row per scene x signal
for part A, pooled and per tile x signal for part B), exp31_summary.json
(tests, preregistered river test, diagnostics: shrinkage, PCA dimension,
reference sizes, mean kNN distances, R3 provenance, exp13 self-check,
failures), exp31_cache.npz (per-unit score arrays and errors), and
exp31_floods_feats.npz when part-B features are recomputed. --smoke runs on
CPU with 2 scenes at 64-px crops and --smoke-tiles flood tiles per split and
writes the same files with a _smoke suffix.

fp32 only: no autocast, no TF32, no compile.
"""
import csv
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
sys.path.insert(0, EXP_DIR)

from harness_ab import (  # noqa: E402
    CONF, TILE, BOUND, CTRL, CONST, CTRL_VAR, CTRL_LVL, REFERENCES, BASE_SIGNALS, DEV, OUT, SHIFTS,  # noqa: F401
    RULE_SCENES, SMOKE_SIZE, load_part_a, scene_unit, base_signals_a, score_scene, combination, finish_part_a,
    load_part_b, base_signals_b, finish_part_b, make_parser, run, scene_tensor, embed_pooled)
import exp13_stat_corrections as exp13  # noqa: E402

K_NN = 5
PCA_CAP = 192
R2_FOLDS, R2_SEED = 5, 0
SHORT = {  # signal name -> cache key
    "kNN-5 cosine [R1 train]": "knn_r1", "Mahalanobis [R1 train]": "maha_r1", "PCA residual [R1 train]": "pca_r1",
    "Mahalanobis class-cond. [R1 train]": "mahacc_r1", "ViM [R1 train]": "vim_r1",
    "kNN-5 cosine [R2 scene]": "knn_r2", "Mahalanobis [R2 scene]": "maha_r2", "PCA residual [R2 scene]": "pca_r2",
    "kNN-5 cosine [R3 pool]": "knn_r3", "Mahalanobis [R3 pool]": "maha_r3", "PCA residual [R3 pool]": "pca_r3",
    "diagnostic feature norm": "norm",
}
NEW_NAMES = list(SHORT)
KNN_R1 = "kNN-5 cosine [R1 train]"
PRIMARY = "kNN-5 cosine [R2 scene]"
COMBO = "combination conf+typicality (prereg)"
CONFIG = {
    "k_nn": K_NN, "pca_dim": f"min({PCA_CAP}, n_ref // 2)", "r2_folds": R2_FOLDS, "r2_fold_seed": R2_SEED,
    "shrinkage": "Ledoit-Wolf (2004) closed form toward tr(S)/D I, own implementation, float64",
    "density_dtype": "float64 on the run device (fits and scores); encoder fp32",
    "reference_sets": {
        "R1": "the head's training patches (A: katima scene, 1024; B: valid patches of the 600 valid-split tiles, with training labels)",
        "R2": f"the unit's own patches, cross-fitted over {R2_FOLDS} random folds (seed {R2_SEED} per unit); no patch scored against itself",
        "R3": "cross-testbed pool as a proxy for the pretraining distribution (A: exp18 Sen1Floods11 features; B: exp11 rule scenes + katima); "
              "upgrade path: a sample of olmoearth_pretrain_dataset"},
    "signals": NEW_NAMES,
    "prereg": {"primary_score": PRIMARY, "combination": COMBO,
               "combination_definition": "mean of within-unit midrank percentiles of confidence and of the primary score",
               "inference": "per-river mean gain of U+ over confidence, one-sided exact sign test over the 8 rivers (7/8 -> p = 0.035); "
                            "every other signal is secondary and gets the same tests"},
}
_A = {}   # part-A features and scene names, reused by part B's R3 fallback


# ----------------------------------------------------------------------------- density estimators
def pca_dim(n_ref, D):
    return int(min(PCA_CAP, n_ref // 2, D - 1))


def f64(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32)).to(DEV, torch.float64)


def lw_fit(X, center=None, labels=None, chunk=16384):
    """Gaussian reference in float64 on DEV from X (n, D) float32, streamed in chunks.

    Centre: the mean (default), a fixed vector `center` (the ViM origin), or per-class means from `labels`
    (tied covariance). Returns the centre(s), the eigendecomposition of the biased covariance S (descending)
    and the closed-form Ledoit-Wolf (2004) shrinkage a of S toward mu I, mu = tr(S)/D:
    delta = ||S - mu I||_F^2 / D, beta = min(delta, ((1/n) sum_k ||x_k||^4 - ||S||_F^2) / (n D)) with x_k the
    centred rows, a = beta / delta. S_a = (1 - a) S + a mu I shares S's eigenvectors."""
    X = np.asarray(X, dtype=np.float32)
    n, D = X.shape
    mean, means, cls = None, None, None
    if labels is not None:
        lab = np.asarray(labels).astype(int).ravel()
        cls = np.unique(lab)
        idx = torch.as_tensor(np.searchsorted(cls, lab), device=DEV)
        sums = torch.zeros(len(cls), D, dtype=torch.float64, device=DEV)
        for i in range(0, n, chunk):
            sums.index_add_(0, idx[i:i + chunk], f64(X[i:i + chunk]))
        means = sums / torch.as_tensor(np.bincount(np.searchsorted(cls, lab)), dtype=torch.float64, device=DEV)[:, None]
        centre = lambda i: means[idx[i:i + chunk]]  # noqa: E731
    else:
        if center is not None:
            mean = torch.as_tensor(center, dtype=torch.float64).to(DEV)   # numpy array or tensor on DEV
        else:
            s = torch.zeros(D, dtype=torch.float64, device=DEV)
            for i in range(0, n, chunk):
                s += f64(X[i:i + chunk]).sum(0)
            mean = s / n
        centre = lambda i: mean  # noqa: E731
    M = torch.zeros(D, D, dtype=torch.float64, device=DEV)
    q4 = 0.0
    for i in range(0, n, chunk):
        xc = f64(X[i:i + chunk]) - centre(i)
        M += xc.T @ xc
        q4 += float(xc.pow(2).sum(1).pow(2).sum())
    S = M / n
    tr = float(torch.trace(S))
    mu = tr / D
    fro2 = float((S * S).sum())
    delta = (fro2 - 2 * mu * tr + D * mu * mu) / D
    beta = max(0.0, (q4 / n - fro2) / (n * D))
    a = float(min(beta, delta) / delta) if delta > 0 and beta > 0 else 0.0
    evals, evecs = torch.linalg.eigh(S)
    return {"n": n, "mean": mean, "class_means": means, "classes": None if cls is None else cls.tolist(),
            "evals": evals.flip(0).clamp_min(0.0), "evecs": evecs.flip(1), "a": a, "mu": mu}


def gauss_scores(Q, fit, d, center=None, chunk=8192):
    """Q (m, D) -> (Mahalanobis under S_a, PCA residual = norm outside the top-d eigenvectors), float64 (m,) each."""
    c = fit["mean"] if center is None else center
    denom = ((1 - fit["a"]) * fit["evals"] + fit["a"] * fit["mu"]).clamp_min(max(1e-12 * fit["mu"], 1e-300))
    maha, resid = [], []
    for i in range(0, len(Q), chunk):
        z = (f64(Q[i:i + chunk]) - c) @ fit["evecs"]
        maha.append((z.pow(2) / denom).sum(1).cpu())
        resid.append(z[:, d:].pow(2).sum(1).sqrt().cpu())
    return torch.cat(maha).numpy(), torch.cat(resid).numpy()


def knn_cos(Q, rn, k=K_NN):
    """Mean cosine distance from each row of Q to its k nearest rows of the L2-normalised reference rn (on DEV).

    The op sequence is exp13's E_dist: topk(1 - q @ r.T, k, largest=False).values.mean(1) in fp32; the query
    set is chunked so that a chunk holds at most 2^27 fp32 distances (512 MiB; with the 1 - sim temporary under 1 GiB)."""
    q = F.normalize(torch.as_tensor(np.asarray(Q, dtype=np.float32)).to(DEV), dim=-1)
    step = max(16, min(4096, (1 << 27) // rn.shape[0]))
    out = [torch.topk(1 - q[i:i + step] @ rn.T, k=k, largest=False).values.mean(1).cpu() for i in range(0, len(q), step)]
    return torch.cat(out).numpy().astype(np.float64)


def build_ref(X, tag):
    """One reference set: normalised rows for kNN, the shrunk Gaussian, and the PCA dimension."""
    X = np.asarray(X, dtype=np.float32)
    return {"tag": tag, "n": len(X), "d": pca_dim(*X.shape), "fit": lw_fit(X),
            "rn": F.normalize(torch.as_tensor(X).to(DEV), dim=-1)}


def score_against(Q, ref):
    """kNN, Mahalanobis and PCA residual of Q (m, D) against one reference set."""
    maha, pca = gauss_scores(Q, ref["fit"], ref["d"])
    return {"knn": knn_cos(Q, ref["rn"]), "maha": maha, "pca": pca}


def crossfit_r2(X, seed=R2_SEED, folds=R2_FOLDS):
    """R2: each of `folds` random folds of the unit's own patches is scored against a reference built on the others."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    out = {k: np.full(n, np.nan) for k in ("knn", "maha", "pca")}
    shr, ds, ns = [], [], []
    for held in np.array_split(perm, folds):
        m = np.ones(n, dtype=bool)
        m[held] = False
        ref = build_ref(X[m], "R2")
        s = score_against(X[held], ref)
        for k in out:
            out[k][held] = s[k]
        shr.append(ref["fit"]["a"])
        ds.append(ref["d"])
        ns.append(int(m.sum()))
    return out, {"n_ref_per_fold": int(min(ns)), "shrinkage_mean_over_folds": float(np.mean(shr)), "d": int(min(ds)),
                 "folds": folds, "mean_knn": float(np.mean(out["knn"]))}


def class_cond_maha(Q, cc, d):
    """Lee et al. 2018: min over classes of the Mahalanobis distance to the class mean under the tied shrunk covariance."""
    return np.min([gauss_scores(Q, cc, d, center=cc["class_means"][j])[0] for j in range(len(cc["classes"]))], axis=0)


def head_logit64(X, w64, b64, chunk=16384):
    return np.concatenate([(f64(X[i:i + chunk]) @ w64 + b64).cpu().numpy() for i in range(0, len(X), chunk)])


def vim_fit(X, w, b):
    """ViM (Wang et al. 2022) for a one-logit head: origin o = -pinv(W) b = -b w / ||w||^2, principal subspace of the
    training features about o (d = min(192, n // 2)), alpha = sum_train max(0, logit) / sum_train residual."""
    w64 = torch.as_tensor(w).to(DEV, torch.float64).ravel()
    b64 = float(torch.as_tensor(b).ravel()[0])
    o = -b64 * w64 / (w64 @ w64)
    fit = lw_fit(X, center=o)
    d = pca_dim(*np.shape(X))
    r_tr = gauss_scores(X, fit, d)[1]
    logit_tr = head_logit64(X, w64, b64)
    return {"fit": fit, "d": d, "alpha": float(np.clip(logit_tr, 0, None).sum() / max(r_tr.sum(), 1e-300)), "origin_norm": float(o.norm()),
            "mean_train_residual": float(r_tr.mean())}


def vim_score(Q, logit, vim):
    """Log-odds of the virtual logit alpha r against the class logits [0, logit]: vl - logaddexp(0, l).

    Monotone in the ViM softmax probability exp(vl) / (1 + exp(l) + exp(vl)) but free of its saturation at 1.0,
    which tied every patch with a large virtual logit (Codex review of the first run)."""
    r = gauss_scores(Q, vim["fit"], vim["d"])[1]
    vl = vim["alpha"] * r
    l = np.asarray(logit, dtype=np.float64).ravel()
    return vl - np.logaddexp(0.0, l)


def fixed_ref_scores(Q, logit, r1, cc, vim, r3):
    """Every signal against the fixed references (R1 and its labelled variants, R3) plus the norm diagnostic; (m,) each."""
    s1, s3 = score_against(Q, r1), score_against(Q, r3)
    return {"knn_r1": s1["knn"], "maha_r1": s1["maha"], "pca_r1": s1["pca"],
            "mahacc_r1": class_cond_maha(Q, cc, r1["d"]), "vim_r1": vim_score(Q, logit, vim),
            "knn_r3": s3["knn"], "maha_r3": s3["maha"], "pca_r3": s3["pca"],
            "norm": np.linalg.norm(np.asarray(Q, dtype=np.float64), axis=1)}


def ref_diag(ref, knn):
    return {"n_ref": ref["n"], "shrinkage": ref["fit"]["a"], "d": ref["d"], "mean_knn": float(np.mean(knn))}


def fixed_refs(tr, labels, hb):
    """R1 with its labelled variants from the head's training patches (n, D) and labels (n,)."""
    t0 = time.time()
    r1 = build_ref(tr, "R1")
    cc = lw_fit(tr, labels=labels)
    vim = vim_fit(tr, *hb)
    print(f"  R1: {r1['n']} patches, shrinkage {r1['fit']['a']:.4f}, d {r1['d']}; class-cond shrinkage {cc['a']:.4f}; "
          f"ViM alpha {vim['alpha']:.4f} (origin norm {vim['origin_norm']:.2f}); {time.time() - t0:.1f}s", flush=True)
    return r1, cc, vim


# ----------------------------------------------------------------------------- part A
def r3_pool_a(args, summary):
    """R3 for part A: the exp18 Sen1Floods11 feature cache, fitted once. None -> per-scene fallback."""
    path = os.path.join(OUT, "exp18_feats.npz")
    keys = ("tr_base", "bolivia_base0", "test_base0")
    if not args.smoke and os.path.exists(path):
        t0 = time.time()
        z = np.load(path)
        X = np.concatenate([np.asarray(z[k], dtype=np.float32).reshape(-1, z[k].shape[-1]) for k in keys])
        ref = build_ref(X, "R3")
        src = f"{path} keys {list(keys)}: {len(X)} patches (shrinkage {ref['fit']['a']:.4f}, d {ref['d']})"
        summary["part_a"]["r3_source"] = src
        print(f"  R3: {src}, {time.time() - t0:.1f}s", flush=True)
        return ref
    src = ("smoke: " if args.smoke else f"WARNING {path} missing: ") + \
        "per scene, the other scenes' shift-0 pooled features plus the training scene (fitted per scene)"
    summary["part_a"]["r3_source"] = src
    print(("  " if args.smoke else "WARNING: ") + f"part A R3 fallback: {src}", flush=True)
    return None


def r3_fallback_a(ctx, name):
    D = ctx["tr_feats"].shape[-1]
    others = [np.asarray(ctx["feats"][f"{n}_base0"], dtype=np.float32).reshape(-1, D)
              for n in ctx["names"] if n != name and f"{n}_base0" in ctx["feats"]]
    return np.concatenate(others + [ctx["tr_feats"].reshape(-1, D)])


def check_vs_exp13(summary, per):
    """Per-scene E-AURC of the R1 kNN score against exp13's E_dist column (same features, head, errors)."""
    path = os.path.join(OUT, "exp13_corrected_stats.csv")
    if not os.path.exists(path):
        summary["part_a"]["r1_knn_vs_exp13_max_abs_diff"] = None
        print("  exp13 self-check skipped: exp13_corrected_stats.csv missing", flush=True)
        return
    with open(path) as f:
        ref = {r["scene"]: float(r["eaurc"]) for r in csv.DictReader(f) if r["signal"] == "E_dist"}
    diffs = {s: abs(per[s][KNN_R1] - ref[s]) for s in per if s in ref}
    mx = max(diffs.values()) if diffs else None
    summary["part_a"]["r1_knn_vs_exp13_max_abs_diff"] = mx
    summary["part_a"]["r1_knn_vs_exp13_n_scenes"] = len(diffs)
    summary["part_a"]["r1_knn_vs_exp13_feats_source"] = summary["part_a"].get("feats_source")
    if mx is None:
        print("  exp13 self-check: no scene in common", flush=True)
    elif mx > 1e-6:
        print(f"WARNING: R1 kNN E-AURC differs from exp13 E_dist by up to {mx:.3g} over {len(diffs)} scenes", flush=True)
    else:
        print(f"  exp13 self-check: R1 kNN E-AURC equals exp13 E_dist (max abs diff {mx:.3g}, {len(diffs)} scenes)", flush=True)


def part_a(model, args, summary, rows, cache):
    ctx = load_part_a(model, args, summary)
    _A.update(feats=ctx["feats"], names=ctx["names"], size=ctx["size"])
    G = ctx["G"]
    D = ctx["tr_feats"].shape[-1]
    tr = ctx["tr_feats"].reshape(-1, D)
    print("part A references:", flush=True)
    r1, cc, vim = fixed_refs(tr, ctx["tr_lab"].reshape(-1), ctx["hb"])
    r3 = r3_pool_a(args, summary)
    units = {}
    for name in ctx["names"]:            # features first: the fallback R3 needs every scene's shift-0 features
        try:
            units[name] = scene_unit(ctx, model, name, args, summary)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    per, rho_per, diag = {}, {}, {}
    for name, unit in units.items():
        if unit is None:
            continue
        t0 = time.time()
        try:
            X = unit["feats0"].reshape(-1, D)
            r3s = r3 if r3 is not None else build_ref(r3_fallback_a(ctx, name), "R3")
            out = fixed_ref_scores(X, unit["logit"], r1, cc, vim, r3s)
            s2, d2 = crossfit_r2(X)
            out.update({"knn_r2": s2["knn"], "maha_r2": s2["maha"], "pca_r2": s2["pca"]})
            sigs = base_signals_a(unit, ctx)
            for nm, key in SHORT.items():
                sigs[nm] = out[key].reshape(G, G)
                cache[f"{name}_{key}"] = out[key].reshape(G, G).astype(np.float32)
            cache[f"{name}_err"] = unit["err"].astype(np.float32)
            sigs[COMBO] = combination(sigs, PRIMARY)
            val = score_scene(unit, sigs, rows, per, rho_per, NEW_NAMES)
            diag[name] = {"R1": ref_diag(r1, out["knn_r1"]), "R1 class-cond": {"shrinkage": cc["a"], "d": r1["d"]},
                          "ViM": {"alpha": vim["alpha"], "d": vim["d"]}, "R2": d2, "R3": ref_diag(r3s, out["knn_r3"])}
            print(f"{name}: {unit['n_err']} errors, {time.time() - t0:.1f}s, E-AURC conf {val[CONF]:.4f} tile {val[TILE]:.4f} "
                  f"primary {val[PRIMARY]:.4f} U+ {val[COMBO]:.4f} | R1 kNN {val[KNN_R1]:.4f}, R3 n={r3s['n']}", flush=True)
        except Exception as ex:  # noqa: BLE001 - record and continue; the summary must always be written
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    summary["part_a"]["diagnostics"] = diag
    finish_part_a(summary, per, rho_per, NEW_NAMES, PRIMARY, COMBO)
    if args.smoke:
        summary["part_a"]["r1_knn_vs_exp13_max_abs_diff"] = "skipped in smoke (crop size differs)"
    elif per:
        check_vs_exp13(summary, per)


# ----------------------------------------------------------------------------- part B
def r3_pool_b(model, args, summary, D):
    """R3 for part B: the 27 rule scenes' shift-0 features plus katima from exp11_feats.npz; fallbacks as documented."""
    path = os.path.join(OUT, "exp11_feats.npz")
    warn = ""
    if not args.smoke and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        keys = [f"{s}_base0" for s in RULE_SCENES if f"{s}_base0" in z] + ["tr_base"]
        X = np.concatenate([np.asarray(z[k], dtype=np.float32).reshape(-1, D) for k in keys])
        src = f"{path}: shift-0 pooled features of {len(keys) - 1} rule scenes + katima training scene, {len(X)} patches"
    elif _A.get("feats"):
        keys = [f"{n}_base0" for n in _A["names"] if f"{n}_base0" in _A["feats"]] + ["tr_base"]
        X = np.concatenate([np.asarray(_A["feats"][k], dtype=np.float32).reshape(-1, D) for k in keys])
        warn = "smoke: " if args.smoke else f"WARNING {path} missing: "
        src = warn + f"part-A pooled features at {_A['size']} px ({', '.join(keys)}), {len(X)} patches"
    else:
        size = SMOKE_SIZE if args.smoke else exp13.SIZE
        z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
        x, ts = scene_tensor(z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), size)
        X = embed_pooled(model, x, ts).reshape(-1, D)
        warn = "smoke: " if args.smoke else f"WARNING {path} missing: "
        src = warn + f"training scene only (part A skipped), {size} px, {len(X)} patches"
    ref = build_ref(X, "R3")
    src += f" (shrinkage {ref['fit']['a']:.4f}, d {ref['d']})"
    summary["part_b"]["r3_source"] = src
    print(("WARNING: " if warn.startswith("WARNING") else "  ") + f"part B R3: {src}", flush=True)
    return ref


def part_b(model, args, summary, rows, cache):
    ctx = load_part_b(model, args, summary, "exp31", extra_keys=("test_base0",))
    if ctx is None:
        return
    sigs = base_signals_b(ctx)
    N, G = ctx["N"], ctx["G"]
    ev = ctx["ev_feats"]
    D = ev.shape[-1]
    Q = ev.reshape(-1, D)
    print("part B references:", flush=True)
    r1, cc, vim = fixed_refs(ctx["tr_feats"], ctx["tr_y"], ctx["hb"])
    r3 = r3_pool_b(model, args, summary, D)
    t0 = time.time()
    out = fixed_ref_scores(Q, ctx["logit"], r1, cc, vim, r3)
    print(f"  fixed-reference scores for {len(Q)} patches in {time.time() - t0:.1f}s", flush=True)
    for key in ("knn_r2", "maha_r2", "pca_r2"):
        out[key] = np.full(len(Q), np.nan)
    r2_diag, failed = [], []
    t0 = time.time()
    for t in range(N):
        try:
            s2, d2 = crossfit_r2(ev[t].reshape(-1, D))
            for k in ("knn", "maha", "pca"):
                out[f"{k}_r2"][t * G * G:(t + 1) * G * G] = s2[k]
            r2_diag.append(d2)
        except Exception as ex:  # noqa: BLE001 - the tile's R2 scores stay NaN and finish_part_b drops its patches
            failed.append(t)
            summary["failures"].append({"part": "B", "unit": f"bolivia/tile{t}", "error": repr(ex), "traceback": traceback.format_exc()})
        if t % 50 == 0 or t == N - 1:
            print(f"  R2 cross-fit: tile {t + 1}/{N}, {time.time() - t0:.0f}s", flush=True)
    for nm, key in SHORT.items():
        sigs[nm] = out[key].reshape(N, G, G)
        cache[f"bolivia_{key}"] = out[key].reshape(N, G, G).astype(np.float32)
    cache["bolivia_err"] = ctx["err"].astype(np.float32)
    cache["bolivia_ok"] = ctx["ok"]
    summary["part_b"]["tiles_failed"] = failed
    summary["part_b"]["diagnostics"] = {
        "R1": ref_diag(r1, out["knn_r1"]), "R1 class-cond": {"shrinkage": cc["a"], "d": r1["d"], "classes": cc["classes"]},
        "ViM": {"alpha": vim["alpha"], "d": vim["d"], "origin_norm": vim["origin_norm"], "mean_train_residual": vim["mean_train_residual"]},
        "R2": {"n_tiles": len(r2_diag), "folds": R2_FOLDS,
               **{f"{k}_{st}": float(getattr(np, st)([d[k] for d in r2_diag])) if r2_diag else None
                  for k in ("shrinkage_mean_over_folds", "d", "n_ref_per_fold")
                  for st in ("mean", "median")},
               "mean_knn": float(np.nanmean(out["knn_r2"]))},
        "R3": ref_diag(r3, out["knn_r3"]),
    }
    finish_part_b(summary, rows, ctx, sigs, NEW_NAMES, PRIMARY, COMBO)


# ----------------------------------------------------------------------------- main
def main():
    ap = make_parser(__doc__)
    args = ap.parse_args()
    run("exp31", "exp31 feature-space typicality", CONFIG, part_a, part_b, args, "exp31_feature_typicality")


if __name__ == "__main__":
    main()
