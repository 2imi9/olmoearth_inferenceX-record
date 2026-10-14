#!/usr/bin/env python
"""exp41: two-view disagreement from Ai2's paper embeddings, no forward pass.

Why. Same-family models err together (exp07, exp10), and an outside embedding added nothing as a neighbourhood
space (exp39). The remaining cross-model question is disagreement between two *heads* on two *views* of the same
windows: does an outside representation err where OlmoEarth does not? Ai2's allenai/olmoearth-paper-embeddings
(June 2026) holds row-aligned embeddings of 26 models on the same samples with labels, so every head can be fitted
and compared on identical windows without running an encoder. Accuracy parity is not required: what matters is
error independence, which is measured here directly (exp10's lesson).

Testbeds (the dump's own splits). Sen1Floods11 (Sentinel-1 input in this dump; hand labels; embeddings (N, 16, 16,
D) for 64-px chips, labels at 64 px, nodata -1): windows are 4-px blocks labelled by majority as in exp18, chips are
the clusters. AWF Sentinel-2 (the expert points of the paper's real-world task, one embedding per sample): pooled
tests with a sample bootstrap. Heads: one multinomial logistic probe per model (L-BFGS, standardised features),
fitted on 80% of the valid chips (samples for AWF); the other 20% ("held-out valid") selects the partner; test is
scored once.

Preregistered, before the run. Partner: the outside model whose probe errors are least correlated (phi) with
OlmoEarth Base's on held-out valid; same-family partners (OlmoEarth nano/tiny/large) are references, never
selected. Primary signal: confidence-weighted disagreement, D = 1[partner prediction != OlmoEarth prediction] times
the partner's max softmax probability. Primary tests, one-sided, against OlmoEarth's confidence (top-1 minus top-2
logit): Sen1Floods11, capture at the 10% budget per chip (chips with 3 <= errors <= n - 3) and a chip bootstrap of
the pooled gain, and E-AURC per chip; AWF, pooled capture at 10% with a sample bootstrap. Secondary: unweighted
disagreement, the accuracy-weighted vote over all outside partners, the same-family references, U+ (midrank mean of
confidence and the primary), 5% and 20%, E-AURC. Falsification: the outside partners' error correlation with
OlmoEarth is not below the same-family references', or no gain over confidence. Limitation stated up front: the
dump has no imagery, so the no-model pixel control cannot run; a pass here means "beats confidence", and the
protocol's full "supported" needs a repeat with imagery (Bolivia, partner encoder run by us).

Inputs: hf://datasets/allenai/olmoearth-paper-embeddings/{model}/{task}/{valid,test}.pt via huggingface_hub
(HF_HOME cache). Outputs: exp/out/exp41_summary.json, exp/out/exp41_two_view.csv, exp/out/exp41_cache.npz.
--smoke: synthetic embeddings and labels for three fake models (no download), _smoke outputs.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import midrank_pct  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
REPO = "allenai/olmoearth-paper-embeddings"
DEV = "cuda" if torch.cuda.is_available() and "--smoke" not in sys.argv else "cpu"
BUDGETS = (0.05, 0.10, 0.20)
PREREG_BUDGET = 0.10
N_BOOT = 2000
PATCH = 4
OE = "olmoearth_base"
FAMILY = ["olmoearth_nano", "olmoearth_tiny", "olmoearth_large"]
TASKS = {
    "sen1floods11": {"kind": "seg", "partners": ["croma_base", "galileo_base", "terramind_base", "clay_large", "copernicusfm", "panopticon", "satlas_base"]},
    "awf_sentinel2": {"kind": "cls", "partners": ["croma_base", "galileo_base", "terramind_base", "dino_v3_dinov3_vitl16_sat", "dino_v3_dinov3_vit7b16_sat",
                                                   "anysat", "clay_large", "copernicusfm", "panopticon", "satlas_base", "prithvi_v2_Prithvi-EO-2.0-300M"]},
}
CONF, PRIMARY, DISAGREE, VOTE, COMBO, CONST = ("confidence (OlmoEarth)", "weighted disagreement (selected partner)", "disagreement (selected partner)",
                                               "accuracy-weighted vote (all outside partners)", "combination conf+weighted disagreement (U+)", "constant score")


# ----------------------------------------------------------------------------- data
def fetch(model, task, split, smoke):
    if smoke:
        return synthetic(model, task, split)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, f"{model}/{task}/{split}.pt", repo_type="dataset")
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["embeddings"], d["labels"]


_SYN = {}


def synthetic(model, task, split):
    """Three fake models sharing one latent truth: their embeddings are noisy views of the class signal."""
    rng = np.random.default_rng(abs(hash((task, split))) % 2 ** 32)
    if task == "sen1floods11":
        N, G, D = (12, 16, 32) if split != "train" else (12, 16, 32)
        key = (task, split)
        if key not in _SYN:
            truth = (rng.random((N, G * PATCH, G * PATCH)) < 0.3).astype(np.int64)
            truth[rng.random((N, G * PATCH, G * PATCH)) < 0.05] = -1
            _SYN[key] = truth
        lab = torch.tensor(_SYN[key])
        sig = torch.tensor(_SYN[key]).float().clamp(min=0).reshape(N, G, PATCH, G, PATCH).mean(axis=(2, 4))
        seed = abs(hash((model, task, split))) % 2 ** 32
        g = torch.Generator().manual_seed(seed)
        emb = sig[..., None] * torch.randn(D, generator=g) * 1.2 + torch.randn(N, G, G, D, generator=g)
        return emb.to(torch.bfloat16), lab
    N, D, C = (300 if split == "train" else 120), 32, 5
    key = (task, split)
    if key not in _SYN:
        _SYN[key] = rng.integers(0, C, N)
    lab = torch.tensor(_SYN[key])
    g = torch.Generator().manual_seed(abs(hash((model, task, split))) % 2 ** 32)
    proto = torch.randn(C, D, generator=g)
    emb = proto[lab] * 0.6 + torch.randn(N, D, generator=g)
    return emb.to(torch.bfloat16), lab


def window_labels(lab, G):
    """(N, H, W) labels with -1 nodata -> water label (N, G, G) and valid mask, as exp18.patch_labels."""
    N = lab.shape[0]
    l = lab.reshape(N, G, PATCH, G, PATCH)
    valid = l >= 0
    n_valid = valid.sum(axis=(2, 4))
    n_water = (l == 1).sum(axis=(2, 4))
    ok = n_valid >= PATCH * PATCH / 2
    water = np.where(ok, n_water > n_valid / 2, False)
    return water.astype(np.int64), ok


def flatten(emb, lab, kind):
    """-> X (M, D) float32, y (M,), unit id per row, ok mask (M,)."""
    if kind == "seg":
        emb = emb.float().numpy()
        N, G = emb.shape[0], emb.shape[1]
        y, ok = window_labels(lab.numpy(), G)
        unit = np.repeat(np.arange(N), G * G)
        return emb.reshape(N * G * G, -1), y.reshape(-1), unit, ok.reshape(-1)
    emb = emb.float().numpy()
    y = lab.numpy().astype(np.int64)
    return emb.reshape(len(emb), -1), y, np.arange(len(emb)), np.ones(len(emb), bool)


# ----------------------------------------------------------------------------- probes
class Probe:
    """Multinomial logistic regression on standardised features, L-BFGS with a small ridge, on DEV."""

    def __init__(self, X, y, n_classes, ridge=1e-3, iters=150):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=DEV)
        yt = torch.tensor(y, dtype=torch.long, device=DEV)
        self.C = n_classes
        self.W = torch.zeros(X.shape[1], n_classes, device=DEV, requires_grad=True)
        self.b = torch.zeros(n_classes, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([self.W, self.b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ self.W + self.b, yt) + ridge * (self.W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = self.W.detach(), self.b.detach()

    @torch.no_grad()
    def logits(self, X, chunk=200000):
        out = []
        for i in range(0, len(X), chunk):
            Xt = torch.tensor((X[i:i + chunk] - self.mu) / self.sd, dtype=torch.float32, device=DEV)
            out.append((Xt @ self.W + self.b).cpu().numpy())
        return np.concatenate(out)


def head_outputs(logits):
    """prediction, top-1 minus top-2 logit margin, max softmax probability."""
    pred = logits.argmax(1)
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2] if logits.shape[1] > 1 else np.abs(logits[:, 0])
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return pred, margin, p.max(1)


def phi(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n11, n10, n01, n00 = (float((a & b).sum()), float((a & ~b).sum()), float((~a & b).sum()), float((~a & ~b).sum()))
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))     # Python floats: no int64 overflow
    return float((n11 * n00 - n10 * n01) / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------------- one task
def run_task(task, cfg, args, summary):
    t0 = time.time()
    kind = cfg["kind"]
    rng = np.random.default_rng(0)
    def load(m):
        ev, lv = fetch(m, task, "valid", args.smoke)
        et, lt = fetch(m, task, "test", args.smoke)
        return {"valid": flatten(ev, lv, kind), "test": flatten(et, lt, kind)}

    def meta(d):
        return {sp: (d[sp][1], d[sp][2], d[sp][3]) for sp in ("valid", "test")}

    def aligned(md, ref_md):
        for sp in ("valid", "test"):
            y, u, ok = md[sp]
            ry, ru, rok = ref_md[sp]
            if not (len(y) == len(ry) and np.array_equal(y, ry) and np.array_equal(u, ru) and np.array_equal(ok, rok)):
                return False
        return True

    # OlmoEarth first: it defines the labels, units, masks and the valid split; embeddings are released after fitting
    ref = load(OE)
    ref_md = meta(ref)
    y_ref, unit_ref, ok_ref = ref_md["test"]
    yv_ref, v_unit, okv_ref = ref_md["valid"]
    n_classes = int(max(y_ref[ok_ref].max(), yv_ref[okv_ref].max())) + 1
    v_units = np.unique(v_unit)
    held = set(rng.choice(v_units, size=max(1, int(round(0.2 * len(v_units)))), replace=False).tolist())
    fit_m = np.array([u not in held for u in v_unit]) & okv_ref
    held_m = np.array([u in held for u in v_unit]) & okv_ref
    print(f"  {task}/{OE}: valid {ref['valid'][0].shape}, test {ref['test'][0].shape}, classes {n_classes}", flush=True)

    def fit_and_predict(m, d):
        probe = Probe(d["valid"][0][fit_m], d["valid"][1][fit_m], n_classes)
        pred_h, marg_h, pmax_h = head_outputs(probe.logits(d["valid"][0][held_m]))
        pred_t, marg_t, pmax_t = head_outputs(probe.logits(d["test"][0]))
        out = {"held": {"pred": pred_h, "err": pred_h != d["valid"][1][held_m]}, "test": {"pred": pred_t, "margin": marg_t, "pmax": pmax_t, "err": pred_t != y_ref},
               "acc_held": float((pred_h == d["valid"][1][held_m]).mean()), "acc_test": float((pred_t == y_ref)[ok_ref].mean())}
        print(f"  head {m}: held-out acc {out['acc_held']:.3f}, test acc {out['acc_test']:.3f}", flush=True)
        return out

    heads = {OE: fit_and_predict(OE, ref)}
    del ref
    partners, family = [], []
    for m in cfg["partners"] + FAMILY:
        d = None
        try:
            d = load(m)
            if not aligned(meta(d), ref_md):
                summary["failures"].append({"part": task, "unit": m, "error": "rows not aligned with OlmoEarth (labels, units or masks differ on valid or test)"})
                print(f"  {task}/{m}: not aligned, dropped", flush=True)
                continue
            heads[m] = fit_and_predict(m, d)
            (family if m in FAMILY else partners).append(m)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": task, "unit": m, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"  {task}/{m}: FAILED {ex!r}", flush=True)
        finally:
            d = None                                   # release this model's embeddings before the next load
    # error correlation with OlmoEarth
    corr = {}
    for m in partners + family:
        e_oe_h, e_m_h = heads[OE]["held"]["err"], heads[m]["held"]["err"]
        e_oe_t, e_m_t = heads[OE]["test"]["err"][ok_ref], heads[m]["test"]["err"][ok_ref]
        corr[m] = {"family": m in family, "phi_held": phi(e_m_h, e_oe_h), "phi_test": phi(e_m_t, e_oe_t),
                   "p_err_given_oe_err_test": float(e_m_t[e_oe_t].mean()) if e_oe_t.any() else None,
                   "p_err_given_oe_ok_test": float(e_m_t[~e_oe_t].mean()) if (~e_oe_t).any() else None,
                   "acc_held": heads[m]["acc_held"], "acc_test": heads[m]["acc_test"]}
    outside = list(partners)
    finite = [m for m in outside if np.isfinite(corr[m]["phi_held"])]
    if finite:
        selected = min(finite, key=lambda m: corr[m]["phi_held"])
        selection = "lowest held-out-valid phi with OlmoEarth among outside partners"
    else:                                                                    # no held-out errors to correlate: fall back to accuracy
        selected = max(outside, key=lambda m: heads[m]["acc_held"])
        selection = "fallback: highest held-out accuracy (no finite correlation)"

    fam_phi = [corr[m]["phi_held"] for m in family if np.isfinite(corr[m]["phi_held"])]
    # signals on test (higher = more suspect)
    oe_t = heads[OE]["test"]
    sig = {CONF: -oe_t["margin"], CONST: np.zeros(len(y_ref))}
    dis = (heads[selected]["test"]["pred"] != oe_t["pred"]).astype(np.float64)
    sig[DISAGREE] = dis
    sig[PRIMARY] = dis * heads[selected]["test"]["pmax"]
    vote = np.zeros(len(y_ref))
    for m in outside:
        vote += heads[m]["acc_held"] * (heads[m]["test"]["pred"] != oe_t["pred"]) * heads[m]["test"]["pmax"]
    sig[VOTE] = vote / max(sum(heads[m]["acc_held"] for m in outside), 1e-9)
    for m in family:
        sig[f"weighted disagreement ({m}, same family)"] = (heads[m]["test"]["pred"] != oe_t["pred"]) * heads[m]["test"]["pmax"]
    for m in outside:
        if m != selected:
            sig[f"weighted disagreement ({m})"] = (heads[m]["test"]["pred"] != oe_t["pred"]) * heads[m]["test"]["pmax"]
    err = oe_t["err"].astype(np.float64)
    res = {"kind": kind, "n_classes": n_classes, "models": {"olmoearth": OE, "partners": partners, "family": family, "outside_with_correlation": outside},
           "held_out_valid_units": len(held), "selected_partner": selected, "selection_rule": selection,
           "error_correlation": corr, "falsification_correlation": {"selected_phi_held": corr[selected]["phi_held"], "family_phi_held_min": min(fam_phi) if fam_phi else None,
                                                                   "outside_below_family": bool(fam_phi and corr[selected]["phi_held"] < min(fam_phi))},
           "heads": {m: {"acc_held": heads[m]["acc_held"], "acc_test": heads[m]["acc_test"]} for m in heads},
           "n_test_windows": int(ok_ref.sum()), "n_test_errors": int(err[ok_ref].sum())}
    if kind == "seg":
        # U+ within chip, per-chip tests, chip bootstrap
        units = np.unique(unit_ref)
        combo = np.full(len(y_ref), np.nan)
        for u in units:
            m = (unit_ref == u) & ok_ref
            if m.any():
                combo[m] = (midrank_pct(sig[CONF][m]) + midrank_pct(sig[PRIMARY][m])) / 2
        sig[COMBO] = combo
        names = [k for k in sig]
        scored = []
        per = {k: [] for k in names}
        cap = {}
        rows = []
        for u in units:
            m = (unit_ref == u) & ok_ref
            e = err[m]
            if m.sum() == 0 or e.sum() < 3 or e.sum() > m.sum() - 3:
                continue
            scored.append(u)
            cap[u] = {k: capture_at_budget_expected(sig[k][m], e, BUDGETS) for k in names}
            for k in names:
                ea = aurc_expected(sig[k][m], e) - oracle_aurc(int(m.sum()), int(e.sum()))
                per[k].append(ea)
                rows.append({"task": task, "unit": f"chip{u}", "n_windows": int(m.sum()), "n_errors": int(e.sum()), "signal": k, "eaurc": ea,
                             **{f"capture@{b}": cap[u][k][b] for b in BUDGETS}})
        res["n_units_scored"] = len(scored)
        tests = {}
        for k in names:
            if k == CONF:
                continue
            g = np.array(per[CONF]) - np.array(per[k])
            w, l, t = wins_losses_ties(g)
            one = k == PRIMARY
            tests[k] = {"eaurc_vs_confidence": {"w": w, "l": l, "t": t, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                                                "median_gain": float(np.median(g)) if len(g) else None}, "capture_vs_confidence": {}}
            for b in BUDGETS:
                gc = np.array([cap[u][k][b] - cap[u][CONF][b] for u in scored])
                w, l, t = wins_losses_ties(gc)
                oneb = k == PRIMARY and b == PREREG_BUDGET
                tests[k]["capture_vs_confidence"][str(b)] = {"w": w, "l": l, "t": t, "sign_p": sign_test(w, l, "greater" if oneb else "two-sided"), "one_sided": oneb}
        res["per_unit_tests"] = tests
        res["median_eaurc"] = {k: float(np.median(per[k])) if per[k] else None for k in names}
        # pooled over valid windows, chip bootstrap shared across signals
        all_units = [u for u in units if ((unit_ref == u) & ok_ref).any()]
        idx_by = {u: np.flatnonzero((unit_ref == u) & ok_ref) for u in all_units}
        pooled_idx = np.concatenate([idx_by[u] for u in all_units])
        full = {k: capture_at_budget_expected(sig[k][pooled_idx], err[pooled_idx], BUDGETS) for k in names}
        full_eaurc = {k: aurc_expected(sig[k][pooled_idx], err[pooled_idx]) - oracle_aurc(len(pooled_idx), int(err[pooled_idx].sum())) for k in names}
        gains = {k: {b: [] for b in BUDGETS} for k in names if k != CONF}
        brng = np.random.default_rng(1)
        for _ in range(N_BOOT):
            pick = brng.integers(0, len(all_units), len(all_units))
            sel = np.concatenate([idx_by[all_units[i]] for i in pick])
            e = err[sel]
            if e.sum() == 0:
                continue
            cc = {k: capture_at_budget_expected(sig[k][sel], e, BUDGETS) for k in names}
            for k in gains:
                for b in BUDGETS:
                    gains[k][b].append(cc[k][b] - cc[CONF][b])
        res["pooled_capture"] = {k: {str(b): full[k][b] for b in BUDGETS} for k in names}
        res["pooled_eaurc"] = full_eaurc
        res["pooled_capture_gain_vs_confidence"] = {k: {str(b): {"gain": full[k][b] - full[CONF][b], "boot_lo": float(np.percentile(gains[k][b], 2.5)) if gains[k][b] else None,
                                                                 "boot_hi": float(np.percentile(gains[k][b], 97.5)) if gains[k][b] else None,
                                                                 "p_better": float((np.array(gains[k][b]) > 0).mean()) if gains[k][b] else None} for b in BUDGETS} for k in gains}
        pt, pg, pe = tests[PRIMARY]["capture_vs_confidence"][str(PREREG_BUDGET)], res["pooled_capture_gain_vs_confidence"][PRIMARY][str(PREREG_BUDGET)], tests[PRIMARY]["eaurc_vs_confidence"]
        res["prereg"] = {"capture_10_per_unit": pt, "capture_10_pooled": pg, "eaurc_per_unit": pe,
                         "supported_vs_confidence": bool(pt["sign_p"] < 0.05 and (pg["boot_lo"] or 0) > 0 and pe["sign_p"] < 0.05 and res["falsification_correlation"]["outside_below_family"])}
        print(f"  {task}: selected partner {selected} (phi held {corr[selected]['phi_held']:.3f}; family min {min(fam_phi) if fam_phi else float('nan'):.3f}); "
              f"primary vs confidence: capture 10% {pt['w']}/{pt['l']}/{pt['t']} p={pt['sign_p']:.2g}, pooled {full[PRIMARY][PREREG_BUDGET]:.3f} vs {full[CONF][PREREG_BUDGET]:.3f} "
              + (f"CI [{pg['boot_lo']:+.3f}, {pg['boot_hi']:+.3f}]" if pg['boot_lo'] is not None else "CI n/a") + f"; E-AURC per chip {pe['w']}/{pe['l']}/{pe['t']} p={pe['sign_p']:.2g}; pooled E-AURC {full_eaurc[PRIMARY]:.4f} vs {full_eaurc[CONF]:.4f}", flush=True)
        return res, rows, {f"{task}_{k}": np.asarray(v, dtype=np.float32) for k, v in sig.items() if k in (CONF, PRIMARY, DISAGREE, VOTE)}
    # classification task: pooled, sample bootstrap
    sig[COMBO] = (midrank_pct(sig[CONF]) + midrank_pct(sig[PRIMARY])) / 2
    names = list(sig)
    m = ok_ref
    full = {k: capture_at_budget_expected(sig[k][m], err[m], BUDGETS) for k in names}
    full_eaurc = {k: aurc_expected(sig[k][m], err[m]) - oracle_aurc(int(m.sum()), int(err[m].sum())) for k in names}
    idx = np.flatnonzero(m)
    gains = {k: {b: [] for b in BUDGETS} for k in names if k != CONF}
    egains = {k: [] for k in names if k != CONF}
    brng = np.random.default_rng(1)
    for _ in range(N_BOOT):
        sel = idx[brng.integers(0, len(idx), len(idx))]
        e = err[sel]
        if e.sum() == 0:
            continue
        cc = {k: capture_at_budget_expected(sig[k][sel], e, BUDGETS) for k in names}
        for k in gains:
            for b in BUDGETS:
                gains[k][b].append(cc[k][b] - cc[CONF][b])
            egains[k].append(aurc_expected(sig[CONF][sel], e) - aurc_expected(sig[k][sel], e))
    res["pooled_capture"] = {k: {str(b): full[k][b] for b in BUDGETS} for k in names}
    res["pooled_eaurc"] = full_eaurc
    res["pooled_capture_gain_vs_confidence"] = {k: {str(b): {"gain": full[k][b] - full[CONF][b], "boot_lo": float(np.percentile(gains[k][b], 2.5)) if gains[k][b] else None,
                                                             "boot_hi": float(np.percentile(gains[k][b], 97.5)) if gains[k][b] else None,
                                                             "p_better": float((np.array(gains[k][b]) > 0).mean()) if gains[k][b] else None} for b in BUDGETS} for k in gains}
    res["pooled_eaurc_gain_vs_confidence"] = {k: {"gain": full_eaurc[CONF] - full_eaurc[k], "boot_lo": float(np.percentile(egains[k], 2.5)) if egains[k] else None,
                                                  "boot_hi": float(np.percentile(egains[k], 97.5)) if egains[k] else None} for k in egains}
    pg = res["pooled_capture_gain_vs_confidence"][PRIMARY][str(PREREG_BUDGET)]
    res["prereg"] = {"capture_10_pooled": pg, "one_sided_criterion": "5th percentile of the sample-bootstrap gain above zero",
                     "boot_p05": float(np.percentile(gains[PRIMARY][PREREG_BUDGET], 5)) if gains[PRIMARY][PREREG_BUDGET] else None}
    res["prereg"]["supported_vs_confidence"] = bool((res["prereg"]["boot_p05"] or 0) > 0 and res["falsification_correlation"]["outside_below_family"])
    rows = [{"task": task, "unit": "pooled", "n_windows": int(m.sum()), "n_errors": int(err[m].sum()), "signal": k, "eaurc": full_eaurc[k],
             **{f"capture@{b}": full[k][b] for b in BUDGETS}} for k in names]
    print(f"  {task}: selected partner {selected} (phi held {corr[selected]['phi_held']:.3f}); primary vs confidence pooled capture 10% "
          f"{full[PRIMARY][PREREG_BUDGET]:.3f} vs {full[CONF][PREREG_BUDGET]:.3f} (5th pct of gain " + (f"{res['prereg']['boot_p05']:+.3f}" if res['prereg']['boot_p05'] is not None else "n/a") + "); "
          f"E-AURC {full_eaurc[PRIMARY]:.4f} vs {full_eaurc[CONF]:.4f}; test acc OlmoEarth {heads[OE]['acc_test']:.3f}, {selected} {heads[selected]['acc_test']:.3f}", flush=True)
    res["runtime_s"] = time.time() - t0
    return res, rows, {f"{task}_{k}": np.asarray(v, dtype=np.float32) for k, v in sig.items() if k in (CONF, PRIMARY, DISAGREE, VOTE)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    summary = {"experiment": "exp41 two-view disagreement from the paper embeddings", "device": DEV, "smoke": args.smoke, "repo": REPO,
               "config": {"budgets": list(BUDGETS), "prereg_budget": PREREG_BUDGET, "n_boot": N_BOOT, "probe": "multinomial logistic, L-BFGS, ridge 1e-3, standardised features, fitted on 80% of valid units",
                          "partner_selection": "lowest held-out-valid phi with OlmoEarth among outside partners", "primary": PRIMARY, "family_references": FAMILY,
                          "limitation": "no imagery in the dump: no pixel control; a pass means beats confidence"},
               "tasks": {}, "failures": []}
    rows, cache = [], {}
    t0 = time.time()
    for task in args.tasks:
        try:
            print(f"== {task}", flush=True)
            res, r, c = run_task(task, TASKS[task], args, summary)
            summary["tasks"][task] = res
            rows += r
            cache.update(c)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": task, "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{task} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    summary["runtime_s"] = time.time() - t0
    summary["n_failures"] = len(summary["failures"])
    import csv
    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(OUT, f"exp41_two_view{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if cache:
        np.savez_compressed(os.path.join(OUT, f"exp41_cache{suffix}.npz"), **cache)

    def ready(o):
        if isinstance(o, dict):
            return {str(k): ready(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ready(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(OUT, f"exp41_summary{suffix}.json"), "w") as f:
        json.dump(ready(summary), f, indent=1)
    print(f"wrote exp41_summary{suffix}.json; failures: {summary['n_failures']}; runtime {summary['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
