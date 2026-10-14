#!/usr/bin/env python
"""exp56: does the audit's ranking save labels when fine-tuning? Which pool tiles to label, chosen three ways.

Why. The audit's review set is the least confident fifth of the frozen head's windows and the windows on its prediction
boundary: on Bolivia hand labels those two cues carry 3.6x and 3.5x the error rate of the rest (exp37) and the
boundary-first order captures more errors than confidence at the 5-10% budgets (exp36). That is a statement about WHERE
the frozen model is wrong. Fine-tuning the encoder on labels corrects 66% of those errors on Bolivia and 44% on the
multi-region split (exp52; 55.6% on AWF, exp21), so the next use of the ranking is the labelling decision itself: when
labels are the cost, does labelling the tiles the audit flags teach the model more per label than labelling tiles at
random? If the flagged tiles are the ones whose labels move the model, the same ranking buys label efficiency; if they
only name errors that any label fixes, or errors that no label fixes, it does not. Active learning asks this question
with predictive entropy as its classic acquisition, so entropy is the second comparator.

Design. Pool: the bucket's train split (6,790 tiles), which the frozen head never saw. Frozen scores on the pool: v1 Base
at crop offset 0, exp18's head (the exp18 cache's training features, 600 valid-split tiles), per-window probability and
logit for every pool tile. Three ways to choose the B pool tiles to label, none of which sees a pool label:
  random   a seeded permutation of the pool, its first B tiles, so one seed's budgets nest;
  audit    the share of a tile's windows that are in the least confident fifth of the pool's windows (|logit| at or
           below the pool's 20th percentile) or on a prediction boundary (exp18.boundary > 0), the B highest: the
           repository's review-set logic applied to the labelling decision;
  entropy  the tile's mean binary predictive entropy of the head's window probabilities, the B highest.
Budgets B in (100, 300, 500, 1000), seeds 0, 1, 2 for every strategy (audit and entropy select deterministically; the
fine-tune seed still varies). Every run reloads v1 Base and fine-tunes it on Sentinel-2 only with exp52's recipe
(per-pixel linear head, AdamW at 3e-4, backbone frozen for the first fifth of the epochs then 0.1 x LR, plateau
scheduler on validation mIoU, best checkpoint kept), steps-matched: epochs = max(12, ceil(1000 x 32 / B)), so every run
takes about 1,000 optimizer steps of batch 32 (1,024 at B = 500 and 1,000; the partial last batch makes it 1,070 at
B = 300 and 1,280 at B = 100, with the same 32,000 tile passes). The scheduler and the checkpoint use ONE fixed
validation subset, 200 valid-split tiles (seed 0), for every run, so the selection labels are equal across strategies;
those 200 labels are not counted in the budget, and the per-epoch scheduler takes more decisions at small budgets: both
stated limitations, equal across strategies at a given budget. Grading: Bolivia (441 tiles) and the test split as exp18
sampled it (800 tiles, seed 1), never used for anything else; per-pixel probabilities pooled to the 4-px window grid at
offset 0, majority labels, the interior windows 1..14 as exp47, window accuracy, and the cross-tab against the frozen
head's offset-0 decision on identical windows (share of its errors corrected, share broken, phi). Per strategy and
budget: the selected tiles' frozen-head window error rate against the pool labels (held here, never shown to a strategy)
says how much harder the chosen tiles were; the Jaccard overlap of the audit and entropy sets says whether they are the
same tiles. Aggregates: mean and min over seeds of the test-split and Bolivia window accuracy; labels_to_parity, for
audit and entropy the smallest budget whose seed-mean test-split accuracy reaches random's at B = 1000.

Preregistered, stated before the run:
  P1  at B = 300 the audit selection beats random on test-split window accuracy by >= 0.005 on the seed mean AND every
      seed's audit run beats the same-seed random run.
  P2  labels_to_parity for audit <= 500: the audit reaches random-at-1000 accuracy with at most half the labels.
  Falsification: the audit's ranking says where the frozen model is wrong, not which labels teach the model; then label
  efficiency is not a product of the audit and the review set stays a review set.
Secondary, descriptive: entropy against audit (accuracy, and the Jaccard overlap of the selected sets at each budget);
the frozen error rate of the selected tiles; Bolivia; the share of frozen errors corrected.

Inputs: data/floods/ (train and test downloaded from the bucket when absent), exp/out/exp18_feats.npz (the head's
training features and the frozen decisions on the grading splits; re-encoded when absent). Outputs:
exp/out/exp56_summary.json, exp/out/exp56_label_efficiency.csv. One B200: 36 fine-tunes of about 4 minutes each plus
the per-epoch validation, about 3 hours (--steps lowers the step count). --smoke: CPU, pool = 4 valid-split tiles,
budgets (2, 3), seed 0, one epoch, 4 Bolivia tiles, the exp42 smoke cache, no downloads, _smoke outputs.
"""
import csv
import gc
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp47_served_ranker as e47  # noqa: E402
import exp51_their_probe as e51  # noqa: E402
import exp52_finetune as e52  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.stats import spearman  # noqa: E402

PATCH, CROP, G, DEV = exp18.PATCH, exp18.CROP, exp18.G, exp18.DEV
LR, BATCH, FREEZE_FRAC = e52.LR, e52.BATCH, e52.FREEZE_FRAC
STRATEGIES = ("random", "audit", "entropy")
BUDGETS, SEEDS, STEPS = (100, 300, 500, 1000), (0, 1, 2), 1000
SMOKE_BUDGETS, SMOKE_SEEDS = (2, 3), (0,)
N_VALID, LOW_CONF_Q, MIN_LEAD, P1_BUDGET, P2_MAX_BUDGET = 200, 0.20, 0.005, 300, 500
EMBED_CHUNK = 256


def epochs_for(budget, steps):
    """Steps-matched schedule: about `steps` optimizer steps of BATCH tiles whatever the budget, never fewer than exp52's epochs."""
    return max(e52.EPOCHS, int(math.ceil(steps * BATCH / budget)))


def embed0(model, s2):
    """exp18.embed at offset 0 in chunks, keeping only the pooled tokens: (N, G, G, D) float16."""
    return np.concatenate([exp18.embed(model, s2[i:i + EMBED_CHUNK], 0)[0] for i in range(0, len(s2), EMBED_CHUNK)])


def frozen_scores(feats, head):
    """Per-window probability and logit of the frozen head, (N, G, G) each; the float16 features go to float32 in chunks."""
    p, lg = [], []
    for i in range(0, len(feats), EMBED_CHUNK):
        a, b = exp18.head_prob_logit(np.asarray(feats[i:i + EMBED_CHUNK], dtype=np.float32), *head)
        p.append(a); lg.append(b)
    return np.concatenate(p), np.concatenate(lg)


def audit_suspicion(p, logit, thr):
    """Share of a tile's windows in the least confident fifth of the pool (|logit| <= thr) or on a prediction boundary."""
    flagged = (np.abs(logit) <= thr) | (exp18.boundary(p) > 0)
    return flagged.reshape(len(p), -1).mean(axis=1)


def entropy_score(p):
    """A tile's mean binary predictive entropy over its windows."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return (-(q * np.log(q) + (1 - q) * np.log(1 - q))).reshape(len(p), -1).mean(axis=1)


def select(strategy, budget, seed, n, scores):
    """Indices of the budget pool tiles a strategy labels; random is a seeded permutation (nested budgets), the others the highest scores."""
    if strategy == "random":
        return np.random.default_rng(seed).permutation(n)[:budget]
    return np.argsort(-scores[strategy], kind="stable")[:budget]


def window_grade(p, lab):
    """Per-pixel probabilities (N, 60, 60) -> interior-window error map, valid mask, and the pixel accuracy at offset 0."""
    n = len(p)
    w = p.reshape(n, G, PATCH, G, PATCH).mean(axis=(2, 4))[:, 1:G, 1:G]
    y, ok = exp18.patch_labels(lab[:, :CROP, :CROP])
    y, ok = y[:, 1:G, 1:G], ok[:, 1:G, 1:G]
    err = ((w > 0.5) != (y > 0.5)).astype(np.float64)
    pix_ok = lab[:, :CROP, :CROP] >= 0
    return err, ok, float(((p > 0.5) == (lab[:, :CROP, :CROP] == 1))[pix_ok].mean())


def jaccard(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return float(len(a & b) / max(len(a | b), 1))


def free():
    """After `del` of a model: collect, and give the CUDA cache back."""
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def main():
    args = hb.make_parser(__doc__, lambda ap: ap.add_argument("--steps", type=int, default=STEPS, help="optimizer steps per fine-tune; epochs = max(12, ceil(steps * 32 / B))")).parse_args()
    suffix = "_smoke" if args.smoke else ""
    budgets, seeds = (SMOKE_BUDGETS, SMOKE_SEEDS) if args.smoke else (BUDGETS, SEEDS)
    p1_budget, p2_max = (budgets[1], budgets[-1]) if args.smoke else (P1_BUDGET, P2_MAX_BUDGET)
    primary = "bolivia" if args.smoke else "test"                               # the split labels_to_parity and P1 read
    summary = {"experiment": "exp56 does the audit's ranking save labels when fine-tuning", "smoke": args.smoke, "device": DEV,
               "config": {"strategies": STRATEGIES, "budgets": budgets, "seeds": seeds, "steps": args.steps, "epochs_rule": "max(12, ceil(steps * batch / B)); 1 in smoke",
                          "lr": LR, "batch": BATCH, "freeze_fraction": FREEZE_FRAC, "sensors": ["s2"], "crop": CROP, "validation_tiles": N_VALID, "low_confidence_quantile": LOW_CONF_Q,
                          "primary_split": primary, "p1_budget": p1_budget, "p2_max_budget": p2_max, "min_lead": MIN_LEAD,
                          "prereg": "P1 at B = 300 audit beats random on test-split window accuracy by >= 0.005 on the seed mean and on every seed; "
                                    "P2 labels_to_parity for audit <= 500 (random's seed-mean test-split accuracy at B = 1000 reached with at most half the labels)"},
               "pool": {}, "frozen": {}, "runs": {}, "aggregate": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    n_smoke = args.smoke_tiles if args.smoke else None
    t_start = time.time()

    # ---- tiles: the pool, the fixed validation subset, the grading splits
    pool_s1, pool_s2, pool_lab = e51.load_ours(floods_dir, "valid" if args.smoke else "train", n_smoke, seed=3)
    va_s1, va_s2, va_lab = e51.load_ours(floods_dir, "valid", n_smoke if args.smoke else N_VALID, seed=0)
    bo = e51.load_ours(floods_dir, "bolivia", None)
    if args.smoke:
        bo = tuple(a[:args.smoke_tiles] for a in bo)                             # the first tiles, the ones the exp42 smoke cache holds
    splits = {"bolivia": bo}
    if not args.smoke:
        splits["test"] = e51.load_ours(floods_dir, "test", exp18.N_TEST_TILES, seed=1)
    N = len(pool_s2)
    print(f"pool {N} tiles, validation {len(va_s2)}, grading " + ", ".join(f"{k} {len(v[0])}" for k, v in splits.items()), flush=True)

    # ---- the frozen head, its scores on the pool, its offset-0 decisions on the grading splits (one model load, before any fine-tune mutates it)
    model = hb.load_model().to(DEV)
    cache_path = os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz" if args.smoke else "exp18_feats.npz")
    cache = np.load(cache_path) if os.path.exists(cache_path) else None
    head_s2, head_lab = hb.load_floods_split(os.path.join(floods_dir, "flood_valid_data.pt"), args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES)
    if cache is not None and "tr_base" in cache.files and len(cache["tr_base"]) == len(head_s2):
        trf, head_src = np.asarray(cache["tr_base"], dtype=np.float32), cache_path
    else:                                                                        # never slice a cache to fit: re-encode the head's training tiles
        trf, head_src = np.asarray(embed0(model, head_s2), dtype=np.float32), "re-encoded with exp18.embed"
    head = e47.head_from(trf, head_lab)
    del trf, head_s2
    t0 = time.time()
    p_pool, logit_pool = frozen_scores(embed0(model, pool_s2), head)
    thr = float(np.quantile(np.abs(logit_pool), LOW_CONF_Q))
    scores = {"audit": audit_suspicion(p_pool, logit_pool, thr), "entropy": entropy_score(p_pool)}
    y_pool, ok_pool = exp18.patch_labels(pool_lab[:, :CROP, :CROP])
    err_pool = ((p_pool > 0.5) != (y_pool > 0.5)).astype(np.float64)              # held here for the descriptives; no strategy sees it
    rho = float(spearman(scores["audit"], scores["entropy"])) if N > 2 else float("nan")
    summary["pool"] = {"n_tiles": N, "head_features": head_src, "embed_seconds": time.time() - t0, "confidence_threshold": thr,
                       "share_windows_low_confidence": float((np.abs(logit_pool) <= thr).mean()), "share_windows_boundary": float((exp18.boundary(p_pool) > 0).mean()),
                       "frozen_window_accuracy": float(1 - err_pool[ok_pool].mean()), "frozen_error_rate": float(err_pool[ok_pool].mean()), "spearman_audit_entropy": rho}
    print(f"pool: frozen window accuracy {summary['pool']['frozen_window_accuracy']:.4f}, |logit| 20th percentile {thr:.3f}, "
          f"{summary['pool']['share_windows_boundary']:.3f} of windows on a boundary, audit/entropy tile-score Spearman {rho:.3f}", flush=True)
    frozen = {}
    for name, (s1, s2, lab) in splits.items():
        try:
            key = f"{name}_base0"
            if cache is not None and key in cache.files and len(cache[key]) == len(s2):
                p0, src = exp18.head_prob_logit(np.asarray(cache[key], dtype=np.float32), *head)[0], cache_path
            else:
                p0, src = frozen_scores(embed0(model, s2), head)[0], "re-encoded with exp18.embed"
            y, ok = exp18.patch_labels(lab[:, :CROP, :CROP]); y, ok = y[:, 1:G, 1:G], ok[:, 1:G, 1:G]
            err = ((p0[:, 1:G, 1:G] > 0.5) != (y > 0.5)).astype(np.float64)
            frozen[name] = {"err": err, "ok": ok, "acc": float(1 - err[ok].mean())}
            summary["frozen"][name] = {"window_accuracy": frozen[name]["acc"], "n_windows": int(ok.sum()), "n_errors": int(err[ok].sum()), "source": src}
            print(f"frozen head/{name}: window accuracy {frozen[name]['acc']:.4f} ({src})", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": f"frozen reference {name}", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"frozen reference {name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    del model; free()

    # ---- inputs normalised once at offset 0
    x2_pool, x1_pool = e52.prep(pool_s2, pool_s1, 0); lab_pool = pool_lab[:, :CROP, :CROP]
    va = (*e52.prep(va_s2, va_s1, 0), va_lab[:, :CROP, :CROP])
    grade_in = {name: e52.prep(s2, s1, 0) for name, (s1, s2, lab) in splits.items()}
    del pool_s2, pool_s1, va_s2, va_s1

    # ---- the runs
    for strategy in STRATEGIES:
        for B in budgets:
            for seed in seeds:
                tag = f"{strategy}/B={B}/seed {seed}"
                t0 = time.time()
                try:
                    idx = select(strategy, B, seed, N, scores)
                    sel_err = float(err_pool[idx][ok_pool[idx]].mean()) if ok_pool[idx].any() else float("nan")
                    water = float((lab_pool[idx] == 1).sum() / max((lab_pool[idx] >= 0).sum(), 1))
                    epochs = 1 if args.smoke else epochs_for(B, args.steps)
                    steps = epochs * int(math.ceil(B / BATCH))
                    model = hb.load_model().to(DEV)
                    ft, history, best = e52.finetune(model, ["s2"], (x2_pool[idx], x1_pool[idx], lab_pool[idx]), va, epochs, seed=seed, log=lambda m: None)
                    res = {"strategy": strategy, "budget": B, "seed": seed, "epochs": epochs, "steps": steps, "train_seconds": time.time() - t0, "best_val_miou": best,
                           "best_epoch": int(np.argmax([h["val_miou"] for h in history])), "selected_frozen_error_rate": sel_err, "selected_water_share": water,
                           "selected": [int(i) for i in idx], "history": {k: [round(float(h[k]), 6) for h in history] for k in ("train_loss", "val_miou", "lr")}}
                    row = {"strategy": strategy, "budget": B, "seed": seed, "epochs": epochs, "steps": steps, "train_seconds": res["train_seconds"], "best_val_miou": best,
                           "selected_frozen_error_rate": sel_err, "selected_water_share": water}
                    line = []
                    for name, (s1, s2, lab) in splits.items():
                        p = e52.predict(ft, *grade_in[name])
                        err, ok, pix = window_grade(p, lab)
                        r = {"window_accuracy": float(1 - err[ok].mean()), "pixel_accuracy": pix, "n_windows": int(ok.sum()), "n_errors": int(err[ok].sum())}
                        if name in frozen:
                            r["frozen_window_accuracy"] = frozen[name]["acc"]
                            r["vs_frozen_head"] = e52.crosstab(frozen[name]["err"], err, ok & frozen[name]["ok"])
                        res[name] = r
                        ct = r.get("vs_frozen_head", {})
                        row.update({f"{name}_window_acc": r["window_accuracy"], f"{name}_frozen_window_acc": r.get("frozen_window_accuracy"), f"{name}_pixel_acc": pix,
                                    f"{name}_share_corrected": ct.get("share_corrected"), f"{name}_broken": ct.get("broken"), f"{name}_phi": ct.get("phi")})
                        line.append(f"{name} {r['window_accuracy']:.4f}" + (f" (frozen {r['frozen_window_accuracy']:.4f}; corrects {ct['share_corrected']:.3f}, breaks {ct['broken']}, phi {ct['phi']:.3f})" if ct else ""))
                    summary["runs"].setdefault(strategy, {}).setdefault(str(B), {})[str(seed)] = res
                    rows.append(row)
                    print(f"{tag}: epochs {epochs} ({steps} steps), {res['train_seconds']:.0f}s, best val mIoU {best:.4f} at epoch {res['best_epoch']} | "
                          f"selected frozen error rate {sel_err:.3f} (pool {summary['pool']['frozen_error_rate']:.3f}), water {water:.3f} | " + " | ".join(line), flush=True)
                    del ft, model; free()
                except Exception as ex:  # noqa: BLE001
                    summary["failures"].append({"part": tag, "error": repr(ex), "traceback": traceback.format_exc()})
                    print(f"{tag} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
                    ft = model = None; free()

    # ---- aggregates over seeds, the overlap of the selected sets, labels to parity, the preregistered tests
    agg = summary["aggregate"]
    for strategy in STRATEGIES:
        for B in budgets:
            runs = summary["runs"].get(strategy, {}).get(str(B), {})
            a = {"n_runs": len(runs), "selected_frozen_error_rate_mean": float(np.mean([r["selected_frozen_error_rate"] for r in runs.values()])) if runs else None}
            for name in splits:
                accs = {s: r[name]["window_accuracy"] for s, r in runs.items() if name in r}
                a[name] = {"mean": float(np.mean(list(accs.values()))), "min": float(np.min(list(accs.values()))), "std": float(np.std(list(accs.values()))), "per_seed": accs} if accs else None
                shares = [r[name]["vs_frozen_head"]["share_corrected"] for r in runs.values() if name in r and "vs_frozen_head" in r[name]]
                a[f"{name}_share_corrected_mean"] = float(np.mean(shares)) if shares else None
            if strategy != "audit":
                a["jaccard_with_audit"] = float(np.mean([jaccard(select("audit", B, 0, N, scores), select(strategy, B, s, N, scores)) for s in (seeds if strategy == "random" else seeds[:1])]))
            agg.setdefault(strategy, {})[str(B)] = a
    ref = agg.get("random", {}).get(str(max(budgets)), {}).get(primary)
    ltp = {"split": primary, "reference": {"strategy": "random", "budget": max(budgets), "mean_accuracy": ref["mean"] if ref else None}, "complete": {}}
    for strategy in ("audit", "entropy"):
        ltp[strategy] = None
        means = {B: (agg.get(strategy, {}).get(str(B), {}).get(primary) or {}).get("mean") for B in budgets}
        ltp["complete"][strategy] = bool(ref is not None and all(v is not None for v in means.values()))
        if ref is not None:
            for B in budgets:
                if means[B] is not None and means[B] >= ref["mean"]:
                    ltp[strategy] = B
                    break
    summary["labels_to_parity"] = ltp
    p1, p1_detail = None, {"budget": p1_budget, "split": primary, "min_lead": MIN_LEAD}
    ra, rr = summary["runs"].get("audit", {}).get(str(p1_budget), {}), summary["runs"].get("random", {}).get(str(p1_budget), {})
    if all(str(s) in ra and str(s) in rr and primary in ra[str(s)] and primary in rr[str(s)] for s in seeds):
        per_seed = {s: ra[str(s)][primary]["window_accuracy"] - rr[str(s)][primary]["window_accuracy"] for s in seeds}
        lead = float(np.mean(list(per_seed.values())))
        p1 = bool(lead >= MIN_LEAD and all(v > 0 for v in per_seed.values()))
        p1_detail.update({"seed_mean_lead": lead, "per_seed_lead": per_seed, "every_seed_wins": bool(all(v > 0 for v in per_seed.values()))})
    p2 = None
    if ltp["audit"] is not None:
        p2 = bool(ltp["audit"] <= p2_max)
    elif ltp["complete"]["audit"]:
        p2 = False
    summary["prereg"] = {"P1": p1, "P1_detail": p1_detail, "P2": p2, "P2_detail": {"labels_to_parity_audit": ltp["audit"], "max_budget": p2_max},
                         "supported": bool(p1 and p2) if (p1 is not None and p2 is not None) else None, "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    summary["runtime_s"] = time.time() - t_start

    names = list(splits)
    print("\nstrategy  B      " + "".join(f"{n + ' mean (min)':<26}" for n in names) + "sel. frozen err  corrected  Jaccard vs audit")
    nan = float("nan")
    for strategy in STRATEGIES:
        for B in budgets:
            a = agg[strategy][str(B)]
            cells = "".join((f"{a[n]['mean']:.4f} ({a[n]['min']:.4f})" if a[n] else "n/a").ljust(26) for n in names)
            sel_err, corr, jac = a["selected_frozen_error_rate_mean"], a.get(f"{primary}_share_corrected_mean"), a.get("jaccard_with_audit", 1.0)
            print(f"{strategy:<9} {B:<6} {cells}{nan if sel_err is None else sel_err:<16.3f} {nan if corr is None else corr:<10.3f} {jac:.3f}")
    print(f"labels to parity on {primary} (random at B={max(budgets)}: {ref['mean'] if ref else nan:.4f}): audit {ltp['audit']}, entropy {ltp['entropy']}")
    print(f"P1 (B={p1_budget}, audit - random, {primary}): {p1_detail.get('seed_mean_lead', nan):+.4f} seed mean, per seed {p1_detail.get('per_seed_lead')} -> {p1}; P2 (audit parity <= {p2_max}): {p2}", flush=True)

    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp56_label_efficiency{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp56_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    short = {k: v for k, v in summary["prereg"].items() if not k.endswith("_detail")}
    print(f"wrote exp56_summary{suffix}.json; prereg {short}; failures {summary['n_failures']}; runtime {summary['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
