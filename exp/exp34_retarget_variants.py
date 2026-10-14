"""exp34: three further readings of the re-targeted latent-MIM objective at inference.

exp33 showed that a PCA-whitened target makes OlmoEarth's masked-token
objective predictable from context (60-70% of the whitened variance) and
that the residual of that prediction ranks input texture, not the probe's
errors. This experiment tests the three readings of the same proposal that
ask a different question from "how large is the residual":

  (a) discrete target, HuBERT-style. K-means over the whitened training
      tokens (K clusters, fitted label-free on the training pool); the
      predictor outputs a distribution over clusters for every hidden
      patch. Scores: the negative log-probability of the patch's true
      cluster, and the predictive entropy of that distribution, which
      needs no true token and is the model's own uncertainty about a patch
      from its context.
  (b) gap masking, Latent MIM-style. A 3x3 hole is hidden and only its
      centre is scored, at every position once (36 lattice phases at
      stride 6), so the predictor cannot copy texture from immediate
      neighbours; the residual measures longer-range predictability.
  (c) residual along the decision direction. The whitened residual of the
      exp33 predictor projected onto the water head's weight direction as
      it appears inside the retained whitened subspace (the component of
      the weight outside the top-d directions is discarded; the fraction of
      the training logit variance the subspace keeps is recorded), so only
      the unpredictability that matters for the class counts; the head is
      the same one that defines confidence.

The exp33 residual (whitened MSE, random 25% masks, K = 8 quarter masks) is
recomputed in the same run as the reference variant. Whitening (top 64
directions), the predictor (two pre-norm transformer layers, width 128,
learned mask token, sinusoidal 2-d positions, Adam 1e-3, 400 steps),
river-disjoint folds in part A (katima, on the Zambezi, only in the pools
that do not hold the Zambezi out), the 600 valid tiles in part B, controls,
U+ and the river test are exp33's. The Spearman diagnostics against the S2
variance and boundary controls use tie-averaged ranks. No evaluation label
enters any fit; the k-means and the whitener see the training pool only.

Preregistered: primary score = the discrete predictive entropy; U+ = mean
of the within-unit midrank percentiles of confidence and of that score;
inference = per-river mean gain of U+ over confidence, one-sided exact sign
test over the 8 rivers (7/8 gives p = 0.035). Null hypothesis, from exp33:
every variant tracks input texture (S2 patch variance) and the boundary
indicator; both Spearman correlations are recorded per unit.

Outputs. exp/out/exp34_retarget_variants.csv, exp34_summary.json,
exp34_cache.npz. --smoke: CPU, 2 scenes at 64 px, 4 flood tiles, 60 steps,
K = 16, _smoke files. fp32 for the encoder; float64 whitening.
"""
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

import harness_ab as hb
import exp33_context_predictor as e33
from oe_inferencex import stats as stats_lib
from harness_ab import CONF, TILE, BOUND, DEV, RIVER  # noqa: F401

K_CLUSTERS, K_SMOKE = 128, 16
HOLE, STRIDE = 3, 6                        # gap masking: 3x3 hole, centres on a stride-6 lattice per phase
RES33, DIR, GAP, DNLL, DENT = ("whitened residual (exp33 form)", "residual along the decision direction",
                               "gap-masked residual (3x3 hole, centre)", "discrete target NLL (k-means)",
                               "discrete predictive entropy (k-means)")
NEW_NAMES = [DENT, DNLL, GAP, DIR, RES33]
PRIMARY = DENT
COMBO = "combination conf+entropy (prereg)"


# ----------------------------------------------------------------------------- k-means (label-free, on DEV)
def kmeans(Z, k, seed=0, iters=30):
    """Plain k-means on (n, d) float32 rows; k-means++ seeding; returns centroids (k, d)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.tensor(Z, device=DEV)
    n = len(X)
    idx = [int(torch.randint(n, (1,), generator=g))]
    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        p = (d2 / d2.sum()).cpu()
        nxt = int(torch.multinomial(p, 1, generator=g))
        idx.append(nxt)
        d2 = torch.minimum(d2, ((X - X[nxt]) ** 2).sum(1))
    C = X[idx].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return C.cpu().numpy()


def assign(C, Z, chunk=8192):
    Ct = torch.tensor(C, device=DEV)
    out = [torch.cdist(torch.tensor(Z[i:i + chunk], device=DEV), Ct).argmin(1).cpu() for i in range(0, len(Z), chunk)]
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- predictors
class ClusterPredictor(e33.Predictor):
    def __init__(self, d, G, k):
        super().__init__(d, G)
        self.out = nn.Sequential(nn.LayerNorm(e33.WIDTH), nn.Linear(e33.WIDTH, k))


def train_cluster_predictor(Z, labels, G, k, steps, seed=0, log=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ClusterPredictor(Z.shape[-1], G, k).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=e33.LR)
    Zt, Lt = torch.tensor(Z, device=DEV), torch.tensor(labels, device=DEV, dtype=torch.long)
    n, trace, t0 = len(Z), [], time.time()
    for step in range(steps):
        idx = rng.choice(n, min(e33.BATCH_UNITS, n), replace=False)
        hidden = torch.tensor(rng.random((len(idx), Z.shape[1])) < e33.MASK_FRAC, device=DEV)
        logits = model(Zt[idx], hidden)
        loss = nn.functional.cross_entropy(logits[hidden], Lt[idx][hidden])
        opt.zero_grad(); loss.backward(); opt.step()
        trace.append(float(loss))
        if log and (step % 100 == 0 or step == steps - 1):
            print(f"  {log} step {step}: CE {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model.eval(), trace


def hole_masks(G, phase_rng=None):
    """Gap masking: for each of the STRIDE*STRIDE lattice phases, hide HOLE x HOLE blocks centred on the lattice;
    returns (n_phases, G*G) hidden masks and (n_phases, G*G) centre masks (the scored positions)."""
    hid, cen = [], []
    r = HOLE // 2
    for pi in range(STRIDE):
        for pj in range(STRIDE):
            h = np.zeros((G, G), bool); c = np.zeros((G, G), bool)
            for i in range(pi, G, STRIDE):
                for j in range(pj, G, STRIDE):
                    c[i, j] = True
                    h[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = True
            hid.append(h.ravel()); cen.append(c.ravel())
    return np.stack(hid), np.stack(cen)


def train_gap_predictor(Z, G, steps, seed=0, log=""):
    """exp33's predictor trained with hole masks (loss on the centres only), so it learns to predict from a distance."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = e33.Predictor(Z.shape[-1], G).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=e33.LR)
    Zt = torch.tensor(Z, device=DEV)
    hid_all, cen_all = hole_masks(G)
    n, trace, t0 = len(Z), [], time.time()
    for step in range(steps):
        idx = rng.choice(n, min(e33.BATCH_UNITS, n), replace=False)
        ph = rng.integers(0, len(hid_all), len(idx))
        hidden = torch.tensor(hid_all[ph], device=DEV); centre = torch.tensor(cen_all[ph], device=DEV)
        pred = model(Zt[idx], hidden)
        loss = ((pred - Zt[idx]) ** 2).mean(-1)[centre].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        trace.append(float(loss))
        if log and (step % 100 == 0 or step == steps - 1):
            print(f"  {log} step {step}: loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model.eval(), trace


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_variants(Z, G, labels, w_dir, res_model, gap_model, clu_model, rng):
    """Z (n_units, G*G, d) whitened tokens; labels (n_units, G*G) cluster ids; w_dir (d,) unit decision direction.
    Returns per-patch arrays (n_units, G, G) for every signal and a diagnostics dict."""
    n_units, n_tok, d = Z.shape
    Zt = torch.tensor(Z, device=DEV)
    wd = torch.tensor(w_dir, device=DEV, dtype=torch.float32)
    out = {k: np.zeros((n_units, n_tok)) for k in (RES33, DIR, GAP, DNLL, DENT)}
    hid_all, cen_all = hole_masks(G)
    hid_t, cen_t = torch.tensor(hid_all, device=DEV), torch.tensor(cen_all, device=DEV)
    for u in range(n_units):
        masks = e33.quarter_masks(n_tok, e33.K_ROUNDS, rng)
        hidden = torch.tensor(masks, device=DEV)
        z = Zt[u][None].expand(len(masks), -1, -1)
        pred = res_model(z, hidden)
        e = ((pred - z) ** 2).mean(-1)
        proj = ((pred - z) @ wd).abs()
        cnt = torch.tensor(masks, device=DEV, dtype=torch.float32).sum(0)
        out[RES33][u] = ((e * hidden).sum(0) / cnt).cpu().numpy()
        out[DIR][u] = ((proj * hidden).sum(0) / cnt).cpu().numpy()
        # gap masking: every patch is a hole centre in exactly one phase
        zg = Zt[u][None].expand(len(hid_all), -1, -1)
        pg = gap_model(zg, hid_t)
        eg = ((pg - zg) ** 2).mean(-1)
        out[GAP][u] = (eg * cen_t).sum(0).cpu().numpy()
        # discrete target: NLL of the true cluster and predictive entropy at hidden positions
        logits = clu_model(z, hidden)
        logp = torch.log_softmax(logits, -1)
        lab = torch.tensor(labels[u], device=DEV, dtype=torch.long)[None].expand(len(masks), -1)
        nll = -logp.gather(-1, lab[..., None])[..., 0]
        ent = -(logp.exp() * logp).sum(-1)
        out[DNLL][u] = ((nll * hidden).sum(0) / cnt).cpu().numpy()
        out[DENT][u] = ((ent * hidden).sum(0) / cnt).cpu().numpy()
    return {k: v.reshape(n_units, G, G) for k, v in out.items()}


def decision_direction(w, whitener, X_train=None):
    """The head's weight projected into the whitened subspace: within the retained top-d directions x.w = z.((P^T w) /
    scale) + const; the component of w outside that subspace is discarded. Returns the unit direction and a
    diagnostic with the fraction of ||w||^2 retained and, given training tokens, the fraction of the training logit
    variance the retained subspace carries."""
    w = np.asarray(w, dtype=np.float64)
    P = whitener["P"]
    v = (P.T @ w) / whitener["scale"]
    diag = {"weight_norm_fraction_retained": float(np.linalg.norm(P.T @ w) ** 2 / max(w @ w, 1e-300))}
    if X_train is not None:
        Xc = np.asarray(X_train, dtype=np.float64) - whitener["mu"]
        full, kept = Xc @ w, (Xc @ P) @ (P.T @ w)
        diag["logit_variance_fraction_retained"] = float(kept.var() / max(full.var(), 1e-300))
    return v / max(np.linalg.norm(v), 1e-12), diag


def fit_all(Ztr, G, k, steps, seed, log):
    """Train the three predictors on one training pool; returns models, k-means centroids and training diagnostics."""
    C = kmeans(Ztr.reshape(-1, Ztr.shape[-1]), k, seed=seed)
    lab_tr = assign(C, Ztr.reshape(-1, Ztr.shape[-1])).reshape(Ztr.shape[:2])
    res_model, tr_res = e33.train_predictor(Ztr, G, steps, seed=seed, log=f"{log} residual")
    gap_model, tr_gap = train_gap_predictor(Ztr, G, steps, seed=seed, log=f"{log} gap")
    clu_model, tr_clu = train_cluster_predictor(Ztr, lab_tr, G, k, steps, seed=seed, log=f"{log} clusters")
    occ = np.bincount(lab_tr.ravel(), minlength=k) / lab_tr.size
    diag = {"k": k, "cluster_entropy_train_nats": float(-(occ[occ > 0] * np.log(occ[occ > 0])).sum()),
            "log_k": float(np.log(k)), "loss_first_last": {"residual": [tr_res[0], tr_res[-1]], "gap": [tr_gap[0], tr_gap[-1]],
                                                            "clusters_ce": [tr_clu[0], tr_clu[-1]]}}
    return res_model, gap_model, clu_model, C, diag


def unit_diag(sigs, unit_ok=None):
    """Tie-aware Spearman (oe_inferencex.stats.spearman) with the two controls the null names; the boundary indicator
    has nine levels, so exp14's positional ranks would let it correlate with raster order."""
    ctrl, bnd = np.asarray(sigs[hb.CTRL_VAR]), np.asarray(sigs[BOUND])
    m = np.ones(ctrl.shape, bool) if unit_ok is None else unit_ok
    return {k: {"spearman_vs_s2_variance": stats_lib.spearman(np.asarray(sigs[k])[m], ctrl[m]),
                "spearman_vs_boundary": stats_lib.spearman(np.asarray(sigs[k])[m], bnd[m])} for k in NEW_NAMES}


# ----------------------------------------------------------------------------- part A
def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    G, D = ctx["G"], ctx["tr_feats"].shape[-1]
    w_head = ctx["hb"][0].numpy()
    units = {}
    for name in ctx["names"]:
        try:
            units[name] = hb.scene_unit(ctx, model, name, args, summary)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
    all_names = [n for n in ctx["names"] if n in hb.RULE_SCENES and f"{n}_base0" in ctx["feats"]]
    names = [n for n in all_names if units.get(n) is not None]
    feats = {n: np.asarray(ctx["feats"][f"{n}_base0"], dtype=np.float32).reshape(G * G, D) for n in all_names}
    feats["__katima__"] = ctx["tr_feats"].reshape(G * G, D)
    folds = e33.FOLDS_A if not args.smoke else tuple({RIVER[n]} for n in all_names)
    rng = np.random.default_rng(0)
    scores, diag = {}, {}
    for f, rivers in enumerate(folds):
        held = [n for n in names if RIVER[n] in rivers]
        train = [n for n in all_names if RIVER[n] not in rivers]
        if "Zambezi" not in rivers:                                                    # katima lies on the Zambezi
            train.append("__katima__")
        if not held:
            continue
        try:
            w = e33.fit_whitener(np.concatenate([feats[n] for n in train]), args.d)
            Ztr = np.stack([e33.whiten(w, feats[n]) for n in train])
            res_m, gap_m, clu_m, C, fd = fit_all(Ztr, G, args.k, args.steps, f, f"fold {f} ({len(train)} units)")
            Zev = np.stack([e33.whiten(w, feats[n]) for n in held])
            lab_ev = assign(C, Zev.reshape(-1, Zev.shape[-1])).reshape(Zev.shape[:2])
            wdir, wdiag = decision_direction(w_head, w, np.concatenate([feats[n] for n in train]))
            out = score_variants(Zev, G, lab_ev, wdir, res_m, gap_m, clu_m, rng)
            for i, n in enumerate(held):
                scores[n] = {k: out[k][i] for k in NEW_NAMES}
            summary["part_a"].setdefault("folds", {})[str(f)] = {"held": held, "n_train_units": len(train), "seed": f,
                                                                  "katima_in_pool": "__katima__" in train,
                                                                  "var_kept_by_whitening": w["var_kept"], "decision_direction": wdiag, **fd}
            print(f"fold {f}: whitening {w['var_kept']:.3f}; cluster CE {fd['loss_first_last']['clusters_ce'][0]:.2f} -> "
                  f"{fd['loss_first_last']['clusters_ce'][1]:.2f} (log K {fd['log_k']:.2f}); gap loss -> {fd['loss_first_last']['gap'][1]:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": f"fold{f}", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"fold {f}: FAILED {ex!r}", flush=True)
    per, rho_per = {}, {}
    for name in names:
        if name not in scores:
            continue
        try:
            unit = units[name]
            sigs = hb.base_signals_a(unit, ctx)
            for k in NEW_NAMES:
                sigs[k] = scores[name][k]
                cache[f"{name}_{NEW_NAMES.index(k)}"] = np.asarray(sigs[k], dtype=np.float32)
            cache[f"{name}_err"] = unit["err"].astype(np.float32)
            sigs[COMBO] = hb.combination(sigs, PRIMARY)
            val = hb.score_scene(unit, sigs, rows, per, rho_per, NEW_NAMES)
            diag[name] = unit_diag(sigs)
            print(f"{name}: {unit['n_err']} errors, conf {val[CONF]:.4f} tile {val[TILE]:.4f} entropy {val[DENT]:.4f} "
                  f"nll {val[DNLL]:.4f} gap {val[GAP]:.4f} dir {val[DIR]:.4f} res33 {val[RES33]:.4f} U+ {val[COMBO]:.4f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    summary["part_a"]["diagnostics"] = diag
    if diag:
        summary["part_a"]["spearman_vs_controls_median"] = {
            k: {c: float(np.nanmedian([diag[n][k][c] for n in diag])) for c in ("spearman_vs_s2_variance", "spearman_vs_boundary")} for k in NEW_NAMES}
    hb.finish_part_a(summary, per, rho_per, NEW_NAMES, PRIMARY, COMBO)


# ----------------------------------------------------------------------------- part B
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp34")
    if ctx is None:
        return
    N, G, D = ctx["N"], ctx["G"], ctx["ev_feats"].shape[-1]
    w_head = ctx["hb"][0].numpy()
    tr_units = np.asarray(ctx["z"]["tr_base"], dtype=np.float32).reshape(-1, G * G, D)
    w = e33.fit_whitener(tr_units.reshape(-1, D), args.d)
    Ztr = np.stack([e33.whiten(w, u) for u in tr_units])
    res_m, gap_m, clu_m, C, fd = fit_all(Ztr, G, args.k, args.steps, 0, f"part B ({len(Ztr)} tiles)")
    Zev = np.stack([e33.whiten(w, u) for u in ctx["ev_feats"].reshape(N, G * G, D)])
    lab_ev = assign(C, Zev.reshape(-1, Zev.shape[-1])).reshape(Zev.shape[:2])
    wdir, wdiag = decision_direction(w_head, w, tr_units.reshape(-1, D))
    out = score_variants(Zev, G, lab_ev, wdir, res_m, gap_m, clu_m, np.random.default_rng(0))
    summary["part_b"]["fit"] = {"n_train_tiles": int(len(Ztr)), "seed": 0, "d": args.d, "var_kept_by_whitening": w["var_kept"],
                                "decision_direction": wdiag, **fd}
    print(f"part B: whitening {w['var_kept']:.3f}; cluster CE {fd['loss_first_last']['clusters_ce'][0]:.2f} -> "
          f"{fd['loss_first_last']['clusters_ce'][1]:.2f} (log K {fd['log_k']:.2f}); gap loss -> {fd['loss_first_last']['gap'][1]:.3f}", flush=True)
    sigs = hb.base_signals_b(ctx)
    for k in NEW_NAMES:
        sigs[k] = out[k]
        cache[f"bolivia_{NEW_NAMES.index(k)}"] = np.asarray(sigs[k], dtype=np.float32)
    cache["bolivia_err"] = ctx["err"].astype(np.float32)
    cache["bolivia_ok"] = ctx["ok"]
    summary["part_b"]["diagnostics"] = unit_diag(sigs, ctx["ok"])
    hb.finish_part_b(summary, rows, ctx, sigs, NEW_NAMES, PRIMARY, COMBO)


def main():
    def extra(ap):
        ap.add_argument("--steps", type=int, default=None, help=f"training steps per predictor (default {e33.STEPS}; smoke 60)")
        ap.add_argument("--d", type=int, default=e33.D_WHITE, help=f"whitened dimensions (default {e33.D_WHITE})")
        ap.add_argument("--k", type=int, default=None, help=f"k-means clusters (default {K_CLUSTERS}; smoke {K_SMOKE})")
    args = hb.make_parser(__doc__, extra).parse_args()
    args.steps = args.steps or (60 if args.smoke else e33.STEPS)
    args.k = args.k or (K_SMOKE if args.smoke else K_CLUSTERS)
    config = {"whitening": f"PCA whitening in the top {args.d} directions, fitted on the training pool",
              "k_means": f"{args.k} clusters, k-means++ seeding, 30 iterations, fitted on the whitened training pool",
              "gap_masking": f"{HOLE}x{HOLE} holes centred on a stride-{STRIDE} lattice; {STRIDE * STRIDE} phases at scoring, centre scored",
              "decision_direction": "the water head's weight in whitened coordinates, unit norm; |residual . direction|",
              "predictors": "exp33's architecture; residual (random 25% masks), gap (hole masks, centre loss), cluster (cross-entropy)",
              "steps": args.steps, "folds_a": [sorted(f) for f in e33.FOLDS_A], "primary_score": PRIMARY, "combination": COMBO,
              "null": "every variant tracks S2 patch variance and the boundary indicator, as exp33's residual did"}
    hb.run("exp34", "exp34 re-targeted latent-MIM variants", config, part_a, part_b, args, "exp34_retarget_variants")


if __name__ == "__main__":
    main()
