#!/usr/bin/env python
"""exp42: is the 4-px grid window the right unit? Four alternative window designs against it, judged at pixel level.

Why. exp41 showed that no frozen encoder read by a linear probe escapes these errors, which makes the window worth testing as a design. The window itself,
a fixed 4-px block on a fixed grid, was never tested as a design choice, and the cue results say where it fails:
errors sit on boundaries and mixed surfaces, where a grid block straddles two classes. Four designs that keep the
model frozen and change only how a decision is placed on the ground, all computable from cached features:

  W0  grid: the shift-0 window decision broadcast to its 16 pixels (the current design).
  W1  shift-averaged: the four tilings (crop offsets 0-3 px, exp18's cache) each give a window map; painted to pixels
      and averaged, the decision at every pixel is the mean of four windows that cover it. A window that follows the
      content by averaging over its own placement.
  W2  scale-adaptive: a window's decision defers to the 8-px block it belongs to (mean of the four fine logits) when
      the block's margin exceeds its own. Coarser where the fine window is unsure.
  W3  spectral split: inside each window the pixels are split by NDWI (> 0 against <= 0) and each part takes the mean
      shift-averaged probability of its own pixels; a window that follows the spectral edge instead of the grid.
  W4  purity abstention: windows whose four tilings disagree (tile-phase above the tile's 80th percentile) abstain;
      compared with confidence abstention at equal coverage within each tile.

Testbeds and judge. Sen1Floods11 Bolivia (441 tiles) and the test split as exp18 sampled it (800 tiles), hand labels
at pixel level (-1 ignored), the exp18 head on cached OlmoEarth Base features at four crop offsets. Pixel accuracy
is measured on the 57 x 57 region every tiling covers; window accuracy on the windows inside it.

Preregistered, before the run. Primary: W1 against W0, pixel accuracy per tile (paired, exact sign test over tiles
with at least one labelled pixel, one-sided: W1 better), on both testbeds; support needs p < 0.05 on both.
Secondary, two-sided or descriptive: W2 against W0 on window accuracy; W3 against W1 on pixel accuracy inside
boundary windows and overall; W4 against confidence abstention, selective accuracy at 80% and 90% coverage; the
share of errors on impure windows (pixel-label water fraction between 0.1 and 0.9) and their error rate against
pure windows; error capture of the shift-averaged confidence against the shift-0 confidence at 5/10/20% budgets.
Falsification: W1 not better than W0 on either testbed.

Inputs: exp/out/exp18_feats.npz, data/floods/. Outputs: exp/out/exp42_summary.json, exp/out/exp42_window_design.csv.
--smoke: CPU, 4 Bolivia tiles from the harness smoke cache, _smoke outputs.
"""
import os
import sys
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import harness_ab as hb  # noqa: E402
import exp18_sen1floods_expert as exp18  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import boundary_indicator, ndwi  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH = 4
SHIFTS = (0, 1, 2, 3)
BUDGETS = (0.05, 0.10, 0.20)
COVERAGES = (0.8, 0.9)
PURE = 0.1                 # a window is pure when its labelled water fraction is <= 0.1 or >= 0.9


def paint(win_map, shift, size):
    """(G, G) window values at crop offset `shift` -> (size, size) pixel map over the tile; NaN outside the crop."""
    G = win_map.shape[0]
    out = np.full((size, size), np.nan)
    px = np.repeat(np.repeat(win_map, PATCH, 0), PATCH, 1)
    out[shift:shift + G * PATCH, shift:shift + G * PATCH] = px
    return out


def tile_designs(logit_shift, s2, lab, size, G):
    """All designs for one tile. logit_shift (4, G, G); s2 (12, size, size) DN; lab (size, size) in {-1, 0, 1}."""
    lo, hi = SHIFTS[-1], SHIFTS[0] + G * PATCH             # pixels covered by every tiling: [3, 60)
    common = np.zeros((size, size), bool)
    common[lo:hi, lo:hi] = True
    valid = (lab >= 0) & common
    p_shift = [1 / (1 + np.exp(-logit_shift[s])) for s in SHIFTS]
    pix = [paint(p_shift[s], s, size) for s in SHIFTS]
    w0_pix = pix[0]                                           # grid decision broadcast to pixels
    w1_pix = np.nanmean(np.stack(pix), 0)                     # shift-averaged
    # W3: spectral split inside each shift-0 window, decision per part from the shift-averaged map
    nd = ndwi(s2[:, :size, :size])
    w3_pix = w1_pix.copy()
    for i in range(G):
        for j in range(G):
            r0, c0 = i * PATCH, j * PATCH
            blk = w1_pix[r0:r0 + PATCH, c0:c0 + PATCH]
            wet = nd[r0:r0 + PATCH, c0:c0 + PATCH] > 0
            for part in (wet, ~wet):
                if part.any() and np.isfinite(blk[part]).any():
                    w3_pix[r0:r0 + PATCH, c0:c0 + PATCH][part] = np.nanmean(blk[part])
    y = lab
    acc = {}
    for name, m in (("W0 grid", w0_pix), ("W1 shift-averaged", w1_pix), ("W3 spectral split", w3_pix)):
        pred = m > 0.5
        acc[name] = float((pred[valid] == (y[valid] == 1)).mean()) if valid.any() else np.nan
    # boundary windows (shift-0 indicator) for the W3-vs-W1 comparison inside them
    hard0 = (p_shift[0] > 0.5).astype(int)
    bnd = boundary_indicator(hard0) > 0
    bnd_pix = paint(bnd.astype(float), 0, size) > 0
    vb = valid & bnd_pix
    acc_bnd = {name: (float(((m > 0.5)[vb] == (y[vb] == 1)).mean()) if vb.sum() >= 16 else np.nan)
               for name, m in (("W1 shift-averaged", w1_pix), ("W3 spectral split", w3_pix))}
    # window level (shift-0 windows fully inside the common region: i, j in 1..G-2 ... window i covers [4i, 4i+4): inside iff 4i >= 3 -> i >= 1)
    y_win, ok_win = exp18.patch_labels(lab[None, :G * PATCH, :G * PATCH])
    y_win, ok_win = y_win[0], ok_win[0]
    inside = np.zeros((G, G), bool)
    inside[1:, 1:] = True
    okw = ok_win & inside
    fine = logit_shift[0]
    err0 = ((fine > 0) != (y_win > 0.5))
    # W2 scale-adaptive: 8-px block mean logit; defer when the block margin exceeds the fine margin
    Gb = (G // 2) * 2
    coarse = fine[:Gb, :Gb].reshape(Gb // 2, 2, Gb // 2, 2).mean(axis=(1, 3))
    coarse_up = np.repeat(np.repeat(coarse, 2, 0), 2, 1)
    w2 = fine.copy()
    defer = np.zeros((G, G), bool)
    defer[:Gb, :Gb] = np.abs(coarse_up) > np.abs(fine[:Gb, :Gb])
    w2[:Gb, :Gb] = np.where(defer[:Gb, :Gb], coarse_up, fine[:Gb, :Gb])
    err2 = ((w2 > 0) != (y_win > 0.5))
    # purity from pixel labels
    l = lab[:G * PATCH, :G * PATCH].reshape(G, PATCH, G, PATCH)
    nv = (l >= 0).sum(axis=(1, 3))
    frac = np.where(nv > 0, (l == 1).sum(axis=(1, 3)) / np.maximum(nv, 1), np.nan)
    impure = (frac > PURE) & (frac < 1 - PURE)
    # W4 abstention signals on windows: confidence and tile-phase (std of the four window maps aligned by exp18)
    conf = -np.abs(fine)
    tile_phase = exp18.aligned_tile_phase(np.stack(p_shift)[:, None])[0] if np.stack(p_shift).ndim == 3 else None
    w1_win = np.nanmean(w1_pix[:G * PATCH, :G * PATCH].reshape(G, PATCH, G, PATCH), axis=(1, 3))         # W1 probability per shift-0 window
    conf_w1 = -np.abs(w1_win - 0.5)                                                                     # confidence of the W1 decision (higher = suspect)
    conf_ml = -np.abs(np.nanmean(np.stack([paint(logit_shift[s], s, size) for s in SHIFTS]), 0))        # mean-logit score, a separate reading
    conf_ml_win = conf_ml[:G * PATCH, :G * PATCH].reshape(G, PATCH, G, PATCH).mean(axis=(1, 3))
    return {"acc_pix": acc, "acc_pix_boundary": acc_bnd, "n_valid_pix": int(valid.sum()), "n_boundary_pix": int(vb.sum()),
            "win": {"ok": okw, "y": y_win, "err0": err0, "err2": err2, "defer": defer, "impure": impure, "frac": frac,
                    "conf": conf, "conf_w1": conf_w1, "conf_ml": conf_ml_win, "tile_phase": tile_phase, "n_win": int(okw.sum())}}


def load_testbed(name, args, summary):
    """Returns logits at the four shifts (N, 4, G, G), S2 tiles (N, 12, 64, 64), pixel labels (N, 64, 64)."""
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    CROP, G = exp18.CROP, exp18.G
    if args.smoke:
        z = np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz"))
        paths, _ = hb.ensure_floods(floods_dir, allow_download=False)
        tr_s2, tr_lab = hb.load_floods_split(paths["valid"], args.smoke_tiles)
        s2, lab = hb.load_floods_split(paths["bolivia"])
        s2, lab = s2[:args.smoke_tiles], lab[:args.smoke_tiles]
        feats = {s: np.asarray(z[f"bolivia_base{s}"], dtype=np.float32) for s in SHIFTS}
        tr_feats = np.asarray(z["tr_base"], dtype=np.float32)
    else:
        paths, _ = hb.ensure_floods(floods_dir, allow_download=True)
        z = np.load(os.path.join(hb.OUT, "exp18_feats.npz"))
        tr_s2, tr_lab = hb.load_floods_split(paths["valid"], exp18.N_TRAIN_TILES)
        tr_feats = np.asarray(z["tr_base"], dtype=np.float32)
        if name == "bolivia":
            s2, lab = hb.load_floods_split(paths["bolivia"])
            feats = {s: np.asarray(z[f"bolivia_base{s}"], dtype=np.float32) for s in SHIFTS}
        else:
            s2, lab = hb.load_floods_split(os.path.join(floods_dir, "flood_test_data.pt"), n=exp18.N_TEST_TILES, seed=1)
            feats = {s: np.asarray(z[f"test_base{s}"], dtype=np.float32) for s in SHIFTS}
    if any(len(feats[s]) != len(s2) for s in SHIFTS):
        raise RuntimeError(f"{name}: feature/image counts differ")
    tr_y, tr_ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    sel = tr_ok.flatten()
    D = tr_feats.shape[-1]
    torch.manual_seed(0)
    head = train_logistic_head(torch.tensor(tr_feats.reshape(-1, D)[sel]), tr_y.flatten()[sel])
    logits = np.stack([exp18.head_prob_logit(feats[s], *head)[1] for s in SHIFTS], 1)      # (N, 4, G, G)
    summary["testbeds"][name] = {"tiles": int(len(s2)), "crop": CROP, "grid": G, "head": "exp18 head retrained from the valid split (seed 0)"}
    return logits, s2, lab, G


def analyse(name, logits, s2, lab, G, summary, rows):
    N = len(s2)
    size = s2.shape[-1]
    per = []
    for t in range(N):
        try:
            per.append(tile_designs(logits[t], s2[t], lab[t], size, G))
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "unit": int(t), "error": repr(ex), "traceback": traceback.format_exc()})
    res = {"tiles": N, "tiles_analysed": len(per)}
    # primary: W1 vs W0 pixel accuracy per tile
    def paired(a, b, one_sided):
        d = np.array([p["acc_pix"][a] - p["acc_pix"][b] for p in per if p["n_valid_pix"] > 0 and np.isfinite(p["acc_pix"][a]) and np.isfinite(p["acc_pix"][b])])
        w, l, t_ = wins_losses_ties(d)
        return {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one_sided else "two-sided"), "one_sided": one_sided,
                "mean_gain": float(d.mean()) if len(d) else None, "median_gain": float(np.median(d)) if len(d) else None, "n_tiles": int(len(d))}
    res["pixel_accuracy_mean"] = {k: float(np.nanmean([p["acc_pix"][k] for p in per])) for k in ("W0 grid", "W1 shift-averaged", "W3 spectral split")}
    res["prereg_W1_vs_W0_pixel"] = paired("W1 shift-averaged", "W0 grid", True)
    res["W3_vs_W1_pixel"] = paired("W3 spectral split", "W1 shift-averaged", False)
    db = np.array([p["acc_pix_boundary"]["W3 spectral split"] - p["acc_pix_boundary"]["W1 shift-averaged"] for p in per
                   if np.isfinite(p["acc_pix_boundary"]["W3 spectral split"]) and np.isfinite(p["acc_pix_boundary"]["W1 shift-averaged"])])
    w, l, t_ = wins_losses_ties(db)
    res["W3_vs_W1_pixel_in_boundary_windows"] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l), "mean_gain": float(db.mean()) if len(db) else None, "n_tiles": int(len(db)),
                                                "acc_W1": float(np.nanmean([p["acc_pix_boundary"]["W1 shift-averaged"] for p in per])),
                                                "acc_W3": float(np.nanmean([p["acc_pix_boundary"]["W3 spectral split"] for p in per]))}
    # W2 window accuracy vs W0
    d2, a0, a2, nd = [], [], [], []
    for p in per:
        w_ = p["win"]
        m = w_["ok"]
        if m.sum() == 0:
            continue
        acc0, acc2 = 1 - w_["err0"][m].mean(), 1 - w_["err2"][m].mean()
        a0.append(acc0); a2.append(acc2); d2.append(acc2 - acc0); nd.append(w_["defer"][m].mean())
    w, l, t_ = wins_losses_ties(np.array(d2))
    res["W2_vs_W0_window"] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l), "acc_W0": float(np.mean(a0)), "acc_W2": float(np.mean(a2)),
                              "share_of_windows_deferred": float(np.mean(nd)), "n_tiles": int(len(d2))}
    # purity: error share and error rate on impure windows (pooled)
    ok_all = np.concatenate([p["win"]["ok"].ravel() for p in per])
    err_all = np.concatenate([p["win"]["err0"].ravel() for p in per])[ok_all]
    imp_all = np.concatenate([p["win"]["impure"].ravel() for p in per])[ok_all]
    res["purity"] = {"share_of_windows_impure": float(imp_all.mean()), "share_of_errors_on_impure_windows": float(imp_all[err_all].mean()) if err_all.any() else None,
                     "error_rate_impure": float(err_all[imp_all].mean()) if imp_all.any() else None, "error_rate_pure": float(err_all[~imp_all].mean()) if (~imp_all).any() else None}
    # W4 abstention: selective accuracy at matched coverage within each tile (the policy of the docstring), then pooled
    def selective(key):
        out = {}
        for c in COVERAGES:
            kept_err, kept_n = 0.0, 0
            for p in per:
                w_ = p["win"]
                m = w_["ok"]
                if m.sum() == 0:
                    continue
                sc, e = w_[key][m], w_["err0"][m]
                order = np.argsort(sc, kind="stable")                   # least suspect first
                k = max(1, int(round(c * len(order))))
                kept_err += e[order[:k]].sum()
                kept_n += k
            out[str(c)] = float(1 - kept_err / max(kept_n, 1))
        return out
    sel = {"confidence": selective("conf"), "tile-phase (W4 purity proxy)": selective("tile_phase"), "W1 confidence": selective("conf_w1"), "mean-logit confidence": selective("conf_ml")}
    res["selective_accuracy"] = sel
    conf_all = np.concatenate([p["win"]["conf"].ravel() for p in per])[ok_all]
    tp_all = np.concatenate([p["win"]["tile_phase"].ravel() for p in per])[ok_all]
    cw1_all = np.concatenate([p["win"]["conf_w1"].ravel() for p in per])[ok_all]
    ca_all = np.concatenate([p["win"]["conf_ml"].ravel() for p in per])[ok_all]
    # error capture: shift-averaged confidence vs shift-0 confidence, pooled and per tile
    scores = (("confidence", conf_all), ("W1 confidence", cw1_all), ("mean-logit confidence", ca_all), ("tile-phase", tp_all))
    res["capture_pooled"] = {sname: {str(b): v for b, v in capture_at_budget_expected(s, err_all.astype(float), BUDGETS).items()} for sname, s in scores}
    res["eaurc_pooled"] = {sname: aurc_expected(s, err_all.astype(float)) - oracle_aurc(len(err_all), int(err_all.sum())) for sname, s in scores}
    for p_i, p in enumerate(per):
        rows.append({"testbed": name, "tile": p_i, "n_valid_pix": p["n_valid_pix"], **{f"pix_acc {k}": v for k, v in p["acc_pix"].items()},
                     "win_acc_W0": float(1 - p["win"]["err0"][p["win"]["ok"]].mean()) if p["win"]["ok"].any() else None,
                     "win_acc_W2": float(1 - p["win"]["err2"][p["win"]["ok"]].mean()) if p["win"]["ok"].any() else None,
                     "share_impure": float(p["win"]["impure"][p["win"]["ok"]].mean()) if p["win"]["ok"].any() else None})
    summary["results"][name] = res
    pr = res["prereg_W1_vs_W0_pixel"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    print(f"{name}: pixel acc W0 {f(res['pixel_accuracy_mean']['W0 grid'])} W1 {f(res['pixel_accuracy_mean']['W1 shift-averaged'])} W3 {f(res['pixel_accuracy_mean']['W3 spectral split'])} | "
          f"W1 vs W0 per tile {pr['w']}/{pr['l']}/{pr['t']} one-sided p={pr['sign_p']:.2g} mean gain {f(pr['mean_gain'])} | "
          f"W3 vs W1 in boundary windows {f(res['W3_vs_W1_pixel_in_boundary_windows']['acc_W3'])} vs {f(res['W3_vs_W1_pixel_in_boundary_windows']['acc_W1'])} | "
          f"W2 window acc {f(res['W2_vs_W0_window']['acc_W2'])} vs {f(res['W2_vs_W0_window']['acc_W0'])} (deferred {f(res['W2_vs_W0_window']['share_of_windows_deferred'])}) | "
          f"impure windows {f(res['purity']['share_of_windows_impure'])} carry {f(res['purity']['share_of_errors_on_impure_windows'])} of errors | "
          f"selective acc @0.8: conf {f(sel['confidence']['0.8'])} tile-phase {f(sel['tile-phase (W4 purity proxy)']['0.8'])} W1-conf {f(sel['W1 confidence']['0.8'])}", flush=True)


def main():
    ap = hb.make_parser(__doc__)
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp42 window design: four alternatives to the 4-px grid window", "smoke": args.smoke, "testbeds": {}, "results": {}, "failures": [],
               "config": {"shifts": list(SHIFTS), "patch": PATCH, "pure_threshold": PURE, "coverages": list(COVERAGES), "budgets": list(BUDGETS),
                          "prereg": "W1 (shift-averaged) beats W0 (grid) on pixel accuracy per tile, one-sided exact sign test, on both testbeds; all else two-sided or descriptive"}}
    rows = []
    for name in (["bolivia"] if args.smoke else ["bolivia", "test"]):
        try:
            logits, s2, lab, G = load_testbed(name, args, summary)
            analyse(name, logits, s2, lab, G, summary, rows)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    if len(summary["results"]) == 2:
        summary["prereg"] = {"supported": all(summary["results"][n]["prereg_W1_vs_W0_pixel"]["sign_p"] < 0.05 for n in ("bolivia", "test"))}
    import csv, json
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp42_window_design{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp42_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp42_summary{suffix}.json; failures: {len(summary['failures'])}", flush=True)


if __name__ == "__main__":
    main()
