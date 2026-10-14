"""exp75: experts that see different inputs. Does disagreement between sensors find the errors the margin misses?

Everything label-free this project tested was a function of one head's features, and every such function lost to
the head's own margin (exp73). The margin misses roughly four in ten of the errors a perfect ranking would put in
a 10% review set (exp70), and exp46 says the misses are errors the model makes confidently, shared across models,
from the input itself. No reading of the same features can see them. A different input might. Ai2 published the
same tasks embedded from different sensors, with the same units in the same order (checked on 2026-09-17: AWF and
Nandi from Landsat, Sentinel-1 and Sentinel-2; CropHarvest Togo and China and PASTIS from one sensor, the other,
and both), so a head fitted on another sensor is an expert on a different input, and its disagreement with the
reference head is a per-unit reading that costs no encoder pass and no label.

Design. Five groups. The reference head is the most accurate head of the group in exp70's record (Sentinel-2 for
AWF, Nandi, Togo and PASTIS; Sentinel-1 plus Sentinel-2 for China); the other heads are views. Each head is
Ai2's linear probe with exp70's recipe, seed 0, so the reference head's errors are exp70's errors. On the
reference head's test units: its margin and entropy; the share of views whose decision differs (vote disagreement)
and the mean KL from the reference's probabilities to each view's (view KL), also per view; a label-free
combination, the interleaving of the margin's ranking and the view KL's ranking (a unit's rank is the better of
its two ranks, so the top of the list alternates between the two); and exp70's two no-model controls. PASTIS
runs on the 4-px window path with tiles as bootstrap clusters; the classification groups treat units as
independent.

Preregistered, before the run.
  P1  disagreement is a signal: view KL beats the best no-model control on the reference head's errors (excess
      AURC) on at least 4 of the 5 groups.
  P2  it sees what the margin misses: among the reference head's errors outside the margin's 10% review set,
      the share inside view KL's own 10% set exceeds the share inside the best control's 10% set, with a 95%
      bootstrap interval (clusters: tiles on PASTIS, units elsewhere) that excludes zero, on at least 4 of 5.
  P3  the combination pays at the budgets that matter: the interleaved ranking captures more of the reference
      head's errors than the margin alone at both 5% and 10% on at least 4 of 5 groups.
  P4  descriptive: on AWF and Nandi, the share of missed errors found by the Landsat view, by the Sentinel-1
      view, and by both together, so whether a third sensor adds to a second is on the record.
  Also descriptive, added on 2026-09-17 before the run started: the accuracy of a late fusion at the head, the
  mean of the reference's and the views' probabilities, beside the reference's and each view's accuracy, so
  whether fusing sensors at the head buys accuracy is on the record next to Ai2's early fusion in the encoder.
Falsification. If P2 fails on 3 or more groups, then even a different sensor does not see the confident errors,
and the record states that what remains needs labels. If P3 holds, the interleaving enters oe_inferencex as a
label-free set rule for scenes with a second sensor, with the family lock exp65 measured.
Outputs: exp/out/exp75_summary.json, exp75_groups.csv; per-group checkpoints under exp75_parts/ (gitignored).
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp70_task_suite as e70                    # noqa: E402
from oe_inferencex import metrics                 # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PARTS = os.path.join(OUT, "exp75_parts")
# reference (the most accurate head of the group in exp70's record) and its views
GROUPS = {"awf": ("awf_sentinel2", ["awf_landsat", "awf_sentinel1"]),
          "nandi": ("nandi_sentinel2", ["nandi_landsat", "nandi_sentinel1"]),
          "togo": ("cropharvest_Togo_12_sentinel2", ["cropharvest_Togo_12_sentinel1", "cropharvest_Togo_12_sentinel2_sentinel1"]),
          "china": ("cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2",
                    ["cropharvest_Peoples_Republic_of_China_6", "cropharvest_Peoples_Republic_of_China_6_sentinel1"]),
          "pastis": ("pastis_sentinel2", ["pastis_sentinel1", "pastis_sentinel1_sentinel2"])}
VIEW_LABEL = {"awf_landsat": "landsat", "awf_sentinel1": "s1", "nandi_landsat": "landsat", "nandi_sentinel1": "s1",
              "cropharvest_Togo_12_sentinel1": "s1", "cropharvest_Togo_12_sentinel2_sentinel1": "s2s1",
              "cropharvest_Peoples_Republic_of_China_6": "plain", "cropharvest_Peoples_Republic_of_China_6_sentinel1": "s1",
              "pastis_sentinel1": "s1", "pastis_sentinel1_sentinel2": "s1s2"}
MISS_BUDGET, GROUP_BAR, N_BOOT = 0.10, 4, 2000


# ----------------------------------------------------------------------------- readings
def _kl(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1); q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def view_readings(p_ref, p_views):
    """Vote disagreement and view KL against the reference, pooled over views and per view; higher = more suspect."""
    dec = np.asarray(p_ref).argmax(1)
    out, dis, kl = {}, [], []
    for name, p in p_views.items():
        d = (np.asarray(p).argmax(1) != dec).astype(np.float64)
        k = _kl(p_ref, p)
        out[f"dis_{name}"], out[f"kl_{name}"] = d, k
        dis.append(d); kl.append(k)
    out["view_vote_dis"] = np.mean(dis, axis=0)
    out["view_kl"] = np.mean(kl, axis=0)
    return out


def ranks(u, rng):
    """0 = most suspect; ties broken by a fixed random permutation so set membership is reproducible."""
    u = np.asarray(u, dtype=np.float64)
    tie = rng.permutation(len(u))
    order = np.lexsort((tie, -u))
    r = np.empty(len(u), dtype=np.int64); r[order] = np.arange(len(u))
    return r


def interleave(u_a, u_b, rng):
    """A ranking whose top-k is, up to ties, the union of the two top-k/2 sets: suspicion = -(min rank), the other
    rank breaking ties."""
    ra, rb = ranks(u_a, rng), ranks(u_b, rng)
    lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
    return -(lo.astype(np.float64) + hi / (4.0 * len(ra)))


def missed_shares(sig, err, budget, rng, clusters):
    """Errors outside the margin's review set at `budget`, and the share of them inside each other signal's set of the
    same size; a cluster bootstrap on the difference between view_kl and the best control."""
    n = len(err); k = max(1, int(round(budget * n)))
    r = {name: ranks(u, np.random.default_rng(rng.integers(2**31))) for name, u in sig.items()}
    inside = {name: r[name] < k for name in sig}
    missed = (err > 0.5) & ~inside["margin"]
    m = int(missed.sum())
    share = {name: float(inside[name][missed].mean()) if m else float("nan") for name in sig if name != "margin"}
    ctl = min((c for c in e70.CONTROLS if c in sig), key=lambda c: -share[c]) if m else e70.CONTROLS[0]
    out = {"n_units": n, "review_k": k, "errors": int((err > 0.5).sum()), "missed_by_margin": m,
           "share_found": share, "best_control_for_misses": ctl}
    if m:
        cl = np.asarray(clusters); ids = np.unique(cl)
        idx_by = {c: np.flatnonzero(cl == c) for c in ids}
        a, b = inside["view_kl"], inside[ctl]
        diffs = []
        brng = np.random.default_rng(0)
        for _ in range(N_BOOT):
            sel = np.concatenate([idx_by[c] for c in brng.choice(ids, size=len(ids), replace=True)])
            mm = missed[sel]
            if mm.sum() == 0:
                continue
            diffs.append(a[sel][mm].mean() - b[sel][mm].mean())
        out["view_kl_minus_control"] = {"point": share["view_kl"] - share[ctl],
                                        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else [float("nan")] * 2,
                                        "n_boot": len(diffs)}
    return out


# ----------------------------------------------------------------------------- heads, exp70's recipe
def seg_window_probs(probe, emb, lab, pp, C):
    """exp54.window_quantities' window probabilities, decisions, majority labels and validity, returned in full."""
    import torch
    import exp54_multiclass_embeddings as e54
    N, H, W = lab.shape
    hw, ww = H // e54.WIN, W // e54.WIN
    wp = np.zeros((N, C, hw, ww), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, 128):
            x = emb[i:i + 128].to(e54.DEV, dtype=torch.float32)
            n, h, w = x.shape[:3]
            lg = probe(x)["logits"].reshape(n, h, w, C, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(n, C, h * pp, w * pp)
            if lg.shape[-2:] != (H, W):
                lg = torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=True)
            p = torch.softmax(lg, dim=1)
            wp[i:i + n] = p[:, :, :hw * e54.WIN, :ww * e54.WIN].reshape(n, C, hw, e54.WIN, ww, e54.WIN).mean(dim=(3, 5)).cpu().numpy()
    l = lab[:, :hw * e54.WIN, :ww * e54.WIN].numpy().reshape(N, hw, e54.WIN, ww, e54.WIN).transpose(0, 1, 3, 2, 4).reshape(N, hw, ww, e54.WIN * e54.WIN)
    valid = l >= 0
    ok = valid.sum(-1) >= (e54.WIN * e54.WIN) // 2
    counts = np.zeros((N, hw, ww, C), dtype=np.int32)
    for c in range(C):
        counts[..., c] = ((l == c) & valid).sum(-1)
    y_win = counts.argmax(-1)
    P = wp.transpose(0, 2, 3, 1).reshape(-1, C)                     # (N*hw*ww, C), tile-major
    return {"P": P, "ok": ok.reshape(-1), "y": y_win.reshape(-1), "hw": (hw, ww),
            "tile": np.repeat(np.arange(N), hw * ww)}


def fit_group_cls(ref, views, cache):
    import torch
    import exp54_multiclass_embeddings as e54
    import exp73_alternatives_suite as e73
    xtr, ytr = e54.load_theirs(e70.MODEL, ref, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, ref, "test", cache)
    C = int(max(int(ytr.max()), int(yte.max())) + 1)
    p_ref = e73._train_cls_probe(xtr, ytr, C, 0)(xte)
    p_views = {}
    for v in views:
        vtr, vytr = e54.load_theirs(e70.MODEL, v, "train", cache)
        vte, vyte = e54.load_theirs(e70.MODEL, v, "test", cache)
        assert torch.equal(vytr, ytr) and torch.equal(vyte, yte), f"{v}: units differ from {ref}"
        p_views[VIEW_LABEL[v]] = e73._train_cls_probe(vtr, vytr, C, 0)(vte)
    y = yte.numpy()
    etr, ete = xtr.to(torch.float32).numpy(), xte.to(torch.float32).numpy()
    return p_ref, p_views, y, etr, ete, np.arange(len(y)), C, {"family": "classification"}


def fit_group_seg(ref, views, cache):
    import torch
    import exp54_multiclass_embeddings as e54
    import exp74_suite_encoders as e74
    xtr, ytr = e54.load_theirs(e70.MODEL, ref, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, ref, "test", cache)
    C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
    pp = int(round(ytr.shape[-1] / xtr.shape[1]))
    probe = e54.train_probe(xtr, ytr, pp, C, e54.TASK_LR.get(ref, 0.1), seed=0)
    q = seg_window_probs(probe, xte, yte, pp, C)
    ok = q["ok"]
    p_views = {}
    for v in views:
        vtr, vytr = e54.load_theirs(e70.MODEL, v, "train", cache)
        vte, vyte = e54.load_theirs(e70.MODEL, v, "test", cache)
        assert torch.equal(vytr, ytr) and torch.equal(vyte, yte), f"{v}: units differ from {ref}"
        vp = e54.train_probe(vtr, vytr, pp, C, e54.TASK_LR.get(v, 0.1), seed=0)
        p_views[VIEW_LABEL[v]] = seg_window_probs(vp, vte, vyte, pp, C)["P"][ok]
    hw, ww = q["hw"]
    H, W = yte.shape[1], yte.shape[2]
    ete = e74.window_embeddings(xte.to(torch.float32).numpy(), hw, ww, H, W, e54.WIN).reshape(-1, xte.shape[-1])[ok]
    etr = xtr.to(torch.float32).numpy().reshape(-1, xtr.shape[-1])
    return q["P"][ok], p_views, q["y"][ok], etr, ete, q["tile"][ok], C, {"family": "segmentation", "patch_px": pp}


def run_group(name, cache):
    import exp73_alternatives_suite as e73
    ref, views = GROUPS[name]
    fit = fit_group_seg if ref in e70.TASKS_SEG else fit_group_cls
    p_ref, p_views, y, etr, ete, clusters, C, extra = fit(ref, views, cache)
    sig, dec = e70.readings_from_probs(p_ref)
    err = (dec != y).astype(np.float64)
    sig.update(e73.controls_chunked(ete, etr, dec))
    sig.update(view_readings(p_ref, p_views))
    rng = np.random.default_rng(0)
    sig["interleave"] = interleave(sig["margin"], sig["view_kl"], rng)
    sc = e70.score_task(sig, err, C)
    cname, cval = e70.best_control(sc)
    miss = missed_shares(sig, err, MISS_BUDGET, np.random.default_rng(1), clusters)
    rec = {"reference": ref, "views": {VIEW_LABEL[v]: v for v in views}, "n_units": int(len(err)), "n_classes": C,
           "test_accuracy": float(1 - err.mean()), "error_rate": float(err.mean()), "signals": sc,
           "best_control": cname, "lead_over_control": {s: float(cval - sc[s]["excess_aurc"]) for s in sc if s not in e70.CONTROLS},
           "capture_gain_interleave": {b: float(sc["interleave"]["capture"][b] - sc["margin"]["capture"][b]) for b in sc["margin"]["capture"]},
           "misses": miss, "view_accuracy": {k: float((np.asarray(p).argmax(1) == y).mean()) for k, p in p_views.items()},
           "late_fusion_accuracy": float((np.mean([np.asarray(p_ref)] + [np.asarray(p) for p in p_views.values()], axis=0).argmax(1) == y).mean())}
    rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- verdicts
def verdicts(results):
    groups = sorted(results)
    p1 = [g for g in groups if results[g]["lead_over_control"]["view_kl"] > 0]
    p2 = [g for g in groups if results[g]["misses"].get("view_kl_minus_control", {}).get("ci95", [0, 0])[0] > 0]
    p3 = [g for g in groups if results[g]["capture_gain_interleave"]["0.05"] > 0 and results[g]["capture_gain_interleave"]["0.1"] > 0]
    v = {"P1": {"holds": len(p1) >= GROUP_BAR, "groups": p1, "of": len(groups)},
         "P2": {"holds": len(p2) >= GROUP_BAR, "groups": p2, "of": len(groups),
                "point": {g: results[g]["misses"].get("view_kl_minus_control", {}).get("point") for g in groups}},
         "P3": {"holds": len(p3) >= GROUP_BAR, "groups": p3, "of": len(groups),
                "gain_at_5": {g: results[g]["capture_gain_interleave"]["0.05"] for g in groups},
                "gain_at_10": {g: results[g]["capture_gain_interleave"]["0.1"] for g in groups}},
         "P4": {"note": "descriptive: share of the margin's missed errors found by each view's KL and by the pooled view KL",
                "per_view": {g: {k: s for k, s in results[g]["misses"]["share_found"].items() if k.startswith("kl_") or k == "view_kl"}
                             for g in groups}},
         "descriptive": {"view_kl_vs_margin_eaurc": {g: float(results[g]["signals"]["margin"]["excess_aurc"] - results[g]["signals"]["view_kl"]["excess_aurc"]) for g in groups},
                         "missed_by_margin": {g: results[g]["misses"]["missed_by_margin"] for g in groups},
                         "accuracy": {g: {"reference": results[g].get("test_accuracy"), "late_fusion": results[g].get("late_fusion_accuracy"),
                                          **results[g].get("view_accuracy", {})} for g in groups}}}
    return v


# ----------------------------------------------------------------------------- driver
def cmd_all(args):
    cache = os.environ.get("HF_HOME")
    os.makedirs(PARTS, exist_ok=True)
    results, rows, failures = {}, [], {}
    for g in GROUPS:
        if args.only and g not in args.only:
            continue
        part = os.path.join(PARTS, f"{g}.json")
        if os.path.exists(part) and not args.redo:
            with open(part) as fh:
                r = json.load(fh)
        else:
            t0 = time.time()
            try:
                r = run_group(g, cache)
            except Exception as exc:
                failures[g] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  {g:8s} FAILED {failures[g][:100]}", flush=True)
                continue
            r["seconds"] = round(time.time() - t0, 1)
            with open(part, "w") as fh:
                json.dump(r, fh, default=float)
        results[g] = r
        m = r["misses"]
        rows.append({"group": g, "reference": r["reference"], "family": r["family"], "n_units": r["n_units"],
                     "test_accuracy": r["test_accuracy"], "margin_eaurc": r["signals"]["margin"]["excess_aurc"],
                     "view_kl_eaurc": r["signals"]["view_kl"]["excess_aurc"], "view_kl_lead": r["lead_over_control"]["view_kl"],
                     "missed_by_margin": m["missed_by_margin"], "found_by_view_kl": m["share_found"]["view_kl"],
                     "found_by_control": m["share_found"][m["best_control_for_misses"]],
                     "ci_lo": m.get("view_kl_minus_control", {}).get("ci95", [None, None])[0],
                     "gain_interleave_5": r["capture_gain_interleave"]["0.05"], "gain_interleave_10": r["capture_gain_interleave"]["0.1"],
                     "seconds": r["seconds"]})
        print(f"  {g:8s} {r['family'][:3]} n={r['n_units']:7d} acc {r['test_accuracy']:.3f} | view_kl lead {r['lead_over_control']['view_kl']:+.4f} "
              f"| misses {m['missed_by_margin']:5d} found kl {m['share_found']['view_kl']:.3f} ctl {m['share_found'][m['best_control_for_misses']]:.3f} "
              f"ci {m.get('view_kl_minus_control', {}).get('ci95')} | interleave gain @5 {r['capture_gain_interleave']['0.05']:+.3f} "
              f"@10 {r['capture_gain_interleave']['0.1']:+.3f} ({r['seconds']:.0f}s)", flush=True)
    summary = {"experiment": "exp75 experts that see different inputs: sensor disagreement on Ai2's published suite",
               "config": {"model": e70.MODEL, "groups": {g: {"reference": r, "views": v} for g, (r, v) in GROUPS.items()},
                          "miss_budget": MISS_BUDGET, "group_bar": GROUP_BAR, "n_boot": N_BOOT, "budgets": list(e70.BUDGETS),
                          "controls": list(e70.CONTROLS)},
               "results": {"groups": results, "failures": failures},
               "verdicts": verdicts(results) if results else {}}
    tag = "_smoke" if getattr(args, "smoke_torch", False) else ""
    with open(os.path.join(OUT, f"exp75_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    if rows:
        with open(os.path.join(OUT, f"exp75_groups{tag}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for k, x in summary["verdicts"].items():
        print(f"{k}: {x.get('holds', '')}  {json.dumps({a: b for a, b in x.items() if a != 'holds'})[:220]}", flush=True)
    if failures:
        print(f"failed: {sorted(failures)}", flush=True)
    return summary


# ----------------------------------------------------------------------------- smokes
def smoke(args):
    """Torch-free: the readings, the interleaving, the missed-error shares and the verdicts on planted data."""
    rng = np.random.default_rng(0)
    n, C = 4000, 4
    y = rng.integers(0, C, n)
    p_ref = rng.dirichlet(np.ones(C) * 0.5, size=n)
    dec = p_ref.argmax(1)
    # plant: a view that agrees where the reference is right and disagrees on a confident subset of its errors
    err = dec != y
    p_view = p_ref.copy()
    conf_err = err & (np.sort(p_ref, 1)[:, -1] > 0.7)
    p_view[conf_err] = np.eye(C)[y[conf_err]] * 0.9 + 0.1 / C
    rd = view_readings(p_ref, {"a": p_view, "b": p_ref})
    assert set(rd) == {"dis_a", "kl_a", "dis_b", "kl_b", "view_vote_dis", "view_kl"}
    assert np.allclose(rd["kl_b"], 0) and rd["kl_a"][conf_err].mean() > rd["kl_a"][~err].mean()
    sig, dec2 = e70.readings_from_probs(p_ref)
    assert np.array_equal(dec, dec2)
    sig.update(e70.controls(rng.standard_normal((n, 8)), rng.standard_normal((100, 8)), dec))
    sig.update(rd)
    sig["interleave"] = interleave(sig["margin"], sig["view_kl"], np.random.default_rng(0))
    top = np.argsort(-sig["interleave"])[:200]
    ra, rb = ranks(sig["margin"], np.random.default_rng(0)), ranks(sig["view_kl"], np.random.default_rng(0))
    assert (np.minimum(ra, rb)[top] < 110).all(), "the interleaving's top 200 must come from the two top-100 lists, up to ties"
    ms = missed_shares(sig, err.astype(float), 0.10, np.random.default_rng(1), np.arange(n))
    assert ms["review_k"] == 400 and ms["missed_by_margin"] > 0
    assert ms["share_found"]["view_kl"] > ms["share_found"][ms["best_control_for_misses"]], ms["share_found"]
    assert ms["view_kl_minus_control"]["ci95"][0] > 0, ms["view_kl_minus_control"]
    sc = e70.score_task(sig, err.astype(float), C)
    assert sc["view_kl"]["excess_aurc"] < sc["ctl_class_rarity"]["excess_aurc"]

    def planted(kl_lead, ci_lo, g5, g10, k=5):
        return {f"g{i}": {"lead_over_control": {"view_kl": kl_lead}, "signals": {"margin": {"excess_aurc": 0.1}, "view_kl": {"excess_aurc": 0.09}},
                          "misses": {"view_kl_minus_control": {"point": 0.1, "ci95": [ci_lo, 0.2]}, "share_found": {"view_kl": 0.3, "kl_a": 0.2}, "missed_by_margin": 10},
                          "capture_gain_interleave": {"0.05": g5, "0.1": g10, "0.2": 0.0}} for i in range(k)}
    v = verdicts(planted(0.02, 0.01, 0.03, 0.02))
    assert v["P1"]["holds"] and v["P2"]["holds"] and v["P3"]["holds"]
    v = verdicts(planted(-0.01, -0.02, 0.01, -0.01))
    assert not v["P1"]["holds"] and not v["P2"]["holds"] and not v["P3"]["holds"]
    print("smoke OK: view readings, interleaving, missed-error shares with a bootstrap, scoring, and the verdicts on planted groups")


def smoke_torch(args):
    """A two-view classification group and a two-view segmentation group through the real fitting code, on synthetic data."""
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54
    import exp73_alternatives_suite as e73
    e51.EPOCHS, e54.EPOCHS = 2, 2
    rng = np.random.default_rng(0)
    N, D, C = 400, 16, 3
    ytr = torch.tensor(rng.integers(0, C, N)); yte = torch.tensor(rng.integers(0, C, 150))
    mk = lambda n: torch.tensor(rng.standard_normal((n, D)), dtype=torch.float32)
    p_ref = e73._train_cls_probe(mk(N), ytr, C, 0)(mk(150))
    p_v = {"a": e73._train_cls_probe(mk(N), ytr, C, 0)(mk(150))}
    rd = view_readings(p_ref, p_v)
    assert rd["view_kl"].shape == (150,)
    pp, h = 4, 8
    emb = torch.tensor(rng.standard_normal((6, h, h, D)), dtype=torch.float32)
    lab = torch.tensor(rng.integers(0, C, (6, h * pp, h * pp)), dtype=torch.int64); lab[:, :4, :4] = -1
    probe = e54.train_probe(emb, lab, pp, C, 0.05, 0)
    q = seg_window_probs(probe, emb, lab, pp, C)
    w = e54.window_quantities(probe, emb, lab, pp, C)
    assert np.array_equal(q["ok"], w["ok"].reshape(-1)) and np.array_equal(q["P"].argmax(1), w["dec"].reshape(-1))
    assert np.allclose(np.sort(q["P"], 1)[:, -1] - np.sort(q["P"], 1)[:, -2], w["margin"].reshape(-1), atol=1e-5)
    assert q["tile"].shape == q["ok"].shape
    print("smoke-torch OK: classification and segmentation views fit and read through the real code; window probabilities match exp54")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="run only these groups")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="torch-free checks of readings, shares and verdicts")
    ap.add_argument("--smoke-torch", action="store_true", help="the fitting code on synthetic data")
    args = ap.parse_args()
    if args.smoke:
        smoke(args); return
    if args.smoke_torch:
        smoke_torch(args); return
    cmd_all(args)


if __name__ == "__main__":
    main()
