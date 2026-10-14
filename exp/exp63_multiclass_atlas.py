#!/usr/bin/env python
"""exp63: the difference atlas on multi-class tasks: MADOS (15 classes) and PASTIS (19 classes) from Ai2's embeddings.

Why. Every comparison finding in this repository is a binary-water finding: exp57's atlas (disagreement sits on
boundaries, the sensor and backbone differences are different sets, the label decides which side is right only where
the model changed) and exp60 to exp62 all graded water. exp54 took the ranking half to many classes on Ai2's paper
embeddings (allenai/olmoearth-paper-embeddings) and found the boundary cue weaker there; the comparison half has never
met a task with many classes, and exp16's collapse of the boundary cue at nine classes says its first claim may fail.

Design. exp54's probes (their recipe in fp32, 50 epochs, the task's probe learning rate) per task and encoder on their
train split, decisions on the test split pooled to 4-px windows (class-probability means, argmax, majority label,
validity), rows aligned across encoders by the label tile. OlmoEarth Base is fitted with three probe seeds (draws);
the other encoders with one. Pairs through oe_inferencex.compare on identical windows, groups = tile: each other
encoder against OlmoEarth Base (five per task); Base against its own second draw (the head-draw floor of a
difference rate, which no earlier pair had); on PASTIS the sensor pair, Base on Sentinel-2 against Base on
Sentinel-1 plus Sentinel-2, the same scenes. Cues of the first side: boundary (exp54.boundary_share > 0), low margin
(bottom quintile) and high entropy (top quintile). Graded: which side is right (with the multi-class 'neither'),
the cross-tab, per tile with compare.over_groups. Stability: the pairwise phi of the five encoder disagreement sets;
of Base-vs-encoder across Base's three draws; and the sensor set against each encoder set.

Preregistered (one-sided):
  P1  the disagreement windows are boundary-enriched more than twofold on every pair of both tasks (exp57's P1 at 15
      and 19 classes; the encoder pairs, the draw pair and the sensor pair, 13 tests).
  P2  on PASTIS the sensor difference and the encoder difference (Base against Galileo) are different sets: phi below
      0.5 pooled and below 0.5 on more tiles than not (tiles with at least MIN_TILE_WINDOWS windows in both sets).
  Falsification: P1 fails if any pair's enrichment is at or below 2 (the boundary cue would then not locate
  disagreement at many classes, as exp16 found for errors); P2 fails if the two sets overlap at phi 0.5 or above.
  Stated predictions, not tested: which side is right stays between 40% and 60% on the encoder pairs, as on floods;
  the 'neither' share of the encoder disagreement windows exceeds one fifth; the draw pair differs on fewer windows
  than any encoder pair.

Inputs. $HF_HOME/paper_embeddings (exp54's cache). Run with ~/oe12/.venv. Outputs. exp/out/exp63_summary.json,
exp/out/exp63_multiclass_atlas.csv (one row per pair), exp/out/exp63_masks.npz (decisions, margins, labels and
validity per task, encoder and draw). --smoke: synthetic embeddings on CPU as exp54's, _smoke outputs.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp54_multiclass_embeddings as e54  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import compare_inferences, over_groups, phi, stability  # noqa: E402
from oe_inferencex.explain import top_fraction  # noqa: E402

BASE, DRAWS = e54.BASE, (0, 1, 2)
TASKS = {"mados": e54.TASKS["mados"], "pastis_sentinel2": e54.TASKS["pastis_sentinel2"]}
SENSOR_TASK = "pastis_sentinel1_sentinel2"
MIN_ENRICHMENT, MAX_PHI, MIN_TILE_WINDOWS = 2.0, 0.5, 20
fmt = e57.fmt


def fit_and_decide(task, model, seed, args, cache_dir):
    """exp54's probe and window quantities for one task, encoder and probe seed: decisions, margins, labels, validity, tile keys."""
    if args.smoke:
        rng = np.random.default_rng(abs(hash((task, model))) % 1000)
        C, N, h, D = 5, 6, 8, 16
        emb_tr, emb_te = (torch.tensor(rng.standard_normal((N, h, h, D)), dtype=torch.float32) for _ in range(2))
        base = np.repeat(np.repeat(np.random.default_rng(0).integers(0, C, (N, h, h)), 4, 1), 4, 2)
        lab_tr, lab_te = torch.tensor(base), torch.tensor(base)
        lab_te[:, :3, :] = -1
    else:
        emb_tr, lab_tr = e54.load_theirs(model, task, "train", cache_dir)
        emb_te, lab_te = e54.load_theirs(model, task, "test", cache_dir)
    C = int(max(int(lab_tr.max()), int(lab_te.max())) + 1)
    pp = lab_tr.shape[-1] // emb_tr.shape[1]
    probe = e54.train_probe(emb_tr, lab_tr, pp, C, e54.TASK_LR.get(task, 0.01), seed=seed)
    q = e54.window_quantities(probe, emb_te, lab_te, pp, C)
    return {"dec": q["dec"], "margin": q["margin"], "entropy": q["entropy"], "y": q["y"], "ok": q["ok"], "keys": e54.label_keys(lab_te.numpy()),
            "pixel_accuracy": q["pixel_accuracy"], "pixel_miou": q["pixel_miou"], "C": C}


def align(ref, other):
    """Reorder `other`'s tiles to `ref`'s by the label tile; None when the tiles do not match one to one."""
    pos = {k: i for i, k in enumerate(other["keys"])}
    idx = np.array([pos.get(k, -1) for k in ref["keys"]])
    if (idx < 0).any() or other["dec"].shape != ref["dec"].shape:
        return None
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(idx),) else v) for k, v in other.items()}


def distribution(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()) if v.size else None, "q10": float(np.percentile(v, 10)) if v.size else None,
            "q50": float(np.median(v)) if v.size else None, "q90": float(np.percentile(v, 90)) if v.size else None}


def row_of(task, name, r):
    w, g = r["where"], r["graded"]
    return {"task": task, "pair": name, "n_windows": r["n_windows"], "n_disagree": r["n_disagree"], "rate": r["disagreement_rate"], "tile_median_rate": r["per_group_rate"]["q50"],
            "boundary_enrichment": w["boundary"]["enrichment"], "low_margin_enrichment": w["low_margin"]["enrichment"], "entropy_enrichment": w["high_entropy"]["enrichment"],
            "share_a_right": g["which_side"]["share_a_right"], "share_b_right": g["which_side"]["share_b_right"], "errors_phi": g["crosstab"]["phi"], "accuracy_a": r["accuracy_a"], "accuracy_b": r["accuracy_b"]}


def pair(name, A, B, groups):
    """compare on identical windows with the first side's cues; the multi-class which-side and cross-tab with the label."""
    ok = A["ok"] & B["ok"]
    cues = {"boundary": e54.boundary_share(A["dec"]) > 0, "low_margin": top_fraction(-A["margin"], 0.2, ok), "high_entropy": top_fraction(A["entropy"], 0.2, ok)}
    out = compare_inferences(A["dec"], B["dec"], ok, groups=groups, labels=A["y"], cues=cues)
    r = {k: v for k, v in out.items() if k != "arrays"}
    r["accuracy_a"], r["accuracy_b"] = float((A["dec"] == A["y"])[ok].mean()), float((B["dec"] == A["y"])[ok].mean())
    r["per_group_rate"] = distribution([g["rate"] for g in out["per_group"].values() if g["n"] >= MIN_TILE_WINDOWS])
    w, g = r["where"], r["graded"]
    print(f"{name}: differ {r['n_disagree']} of {r['n_windows']} ({fmt(100 * r['disagreement_rate'], '.2f')}%) | boundary {fmt(w['boundary']['enrichment'], '.1f')}x, low margin {fmt(w['low_margin']['enrichment'], '.1f')}x, "
          f"entropy {fmt(w['high_entropy']['enrichment'], '.1f')}x | a right {fmt(g['which_side']['share_a_right'], '.3f')}, b right {fmt(g['which_side']['share_b_right'], '.3f')}, neither {fmt(1 - (g['which_side']['share_a_right'] or 0) - (g['which_side']['share_b_right'] or 0), '.3f')} | "
          f"errors phi {fmt(g['crosstab']['phi'])} | acc {r['accuracy_a']:.4f} / {r['accuracy_b']:.4f}", flush=True)
    return r, out["arrays"]["disagree"]


def prereg(summary):
    R = summary["results"]
    detail, tested = {}, 0
    for task, T in R.items():
        if not isinstance(T, dict) or "pairs" not in T:
            continue
        for name, r in T["pairs"].items():
            e = r["where"]["boundary"]["enrichment"]
            detail[f"{task}/{name}"] = e
            tested += 1
    p1 = bool(detail) and all(e is not None and np.isfinite(e) and e > MIN_ENRICHMENT for e in detail.values())
    st = R.get("pastis_sentinel2", {}).get("sensor_vs_encoder")
    p2 = bool(st and st["pooled_phi"] is not None and np.isfinite(st["pooled_phi"]) and st["pooled_phi"] < MAX_PHI and st["sign_p"] < 0.05) if st else None
    summary["prereg"] = {"P1": p1 if detail else None, "P1_detail": detail, "P1_n_tested": tested, "P2": p2, "min_enrichment": MIN_ENRICHMENT, "max_phi": MAX_PHI,
                         "complete": bool(detail and p2 is not None and not summary["failures"])}
    summary["n_failures"] = len(summary["failures"])
    print(f"prereg: P1 {summary['prereg']['P1']} ({tested} pairs, min enrichment {fmt(min(detail.values()) if detail else None)}) | P2 {p2} {st and {k: st[k] for k in ('pooled_phi', 'tiles_below', 'tiles_at_or_above', 'sign_p')}} | complete {summary['prereg']['complete']}", flush=True)


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp63 the difference atlas on multi-class tasks (MADOS, PASTIS) from Ai2's embeddings", "smoke": args.smoke, "device": e54.DEV,
               "config": {"tasks": {t: TASKS[t] for t in args.tasks}, "draws": list(DRAWS), "sensor_task": SENSOR_TASK, "min_enrichment": MIN_ENRICHMENT, "max_phi": MAX_PHI, "min_tile_windows": MIN_TILE_WINDOWS,
                          "lr": e54.TASK_LR, "epochs": e54.EPOCHS, "window": e54.WIN,
                          "prereg": "P1 boundary enrichment > 2 on every pair of both tasks; P2 on PASTIS the sensor and the Base-vs-Galileo disagreement sets overlap at phi < 0.5 pooled and on more tiles than not"},
               "results": {}, "failures": []}
    rows, masks = [], {}
    cache_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "paper_embeddings")
    for task in args.tasks:
        T = {"pairs": {}, "encoders": {}, "seconds": {}}
        summary["results"][task] = T
        try:
            models = [BASE, "other_a", "other_b"] if args.smoke else TASKS[task]
            held = {}
            for model in models:
                for seed in (DRAWS if model == BASE else (0,)):
                    t0 = time.time()
                    try:
                        held[(model, seed)] = fit_and_decide(task, model, seed, args, cache_dir)
                        T["seconds"][f"{model}/{seed}"] = time.time() - t0
                        T["encoders"][f"{model}/{seed}"] = {k: held[(model, seed)][k] for k in ("pixel_accuracy", "pixel_miou", "C")}
                        print(f"{task} {model} seed {seed}: pixel acc {held[(model, seed)]['pixel_accuracy']:.4f}, mIoU {held[(model, seed)]['pixel_miou']:.3f} ({time.time() - t0:.0f}s)", flush=True)
                    except Exception as ex:  # noqa: BLE001
                        e57.fail(summary, f"{task} {model} seed {seed}", ex)
            ref = held[(BASE, 0)]
            groups = np.arange(len(ref["dec"]))[:, None, None]
            for key, h in held.items():
                # Base's draws share the reference order; aligning them by label hash would permute tiles with identical labels
                hh = h if key[0] == BASE else (align(ref, h) or h)
                masks[f"{task}/{key[0]}/{key[1]}/dec"] = hh["dec"].astype(np.int16)
                masks[f"{task}/{key[0]}/{key[1]}/margin"] = hh["margin"].astype(np.float32)
            masks[f"{task}/y"], masks[f"{task}/ok"] = ref["y"].astype(np.int16), ref["ok"]
            dis = {}
            for model in models:
                if model == BASE:
                    continue
                other = align(ref, held[(model, 0)])
                if other is None:
                    summary["failures"].append({"part": f"{task} {model}", "error": "tiles do not align with OlmoEarth Base's"})
                    continue
                T["pairs"][f"{BASE} vs {model}"], dis[model] = pair(f"{task}: {BASE} vs {model}", ref, other, groups)
                rows.append(row_of(task, f"{BASE} vs {model}", T["pairs"][f"{BASE} vs {model}"]))
                # the same pair across Base's draws
                drawn = {0: dis[model]}
                for seed in DRAWS[1:]:
                    if (BASE, seed) in held:
                        drawn[seed] = ((held[(BASE, seed)]["dec"] != other["dec"]) & held[(BASE, seed)]["ok"] & other["ok"])
                T["pairs"][f"{BASE} vs {model}"]["stability_across_draws"] = stability(drawn) if len(drawn) > 1 else None
            if (BASE, 1) in held:
                T["pairs"]["draw 0 vs draw 1"], dis["draw"] = pair(f"{task}: {BASE} draw 0 vs draw 1", ref, held[(BASE, 1)], groups)
                rows.append(row_of(task, "draw 0 vs draw 1", T["pairs"]["draw 0 vs draw 1"]))
            enc_sets = {m: d for m, d in dis.items() if m != "draw"}
            T["stability_across_encoders"] = stability(enc_sets) if len(enc_sets) > 1 else None
            if T["stability_across_encoders"]:
                print(f"{task}: pairwise phi of the {len(enc_sets)} encoder disagreement sets, median {fmt(T['stability_across_encoders']['median_pairwise_phi'])}", flush=True)
            if task == "pastis_sentinel2":
                t0 = time.time()
                try:
                    both = fit_and_decide(SENSOR_TASK, BASE, 0, args, cache_dir)
                    T["seconds"][f"{SENSOR_TASK}/{BASE}/0"] = time.time() - t0
                    T["encoders"][f"{SENSOR_TASK}/{BASE}/0"] = {k: both[k] for k in ("pixel_accuracy", "pixel_miou", "C")}
                    other = align(ref, both)
                    if other is None:
                        raise RuntimeError("the S1+S2 tiles do not align with the S2 tiles")
                    masks[f"{SENSOR_TASK}/{BASE}/0/dec"], masks[f"{SENSOR_TASK}/{BASE}/0/margin"] = other["dec"].astype(np.int16), other["margin"].astype(np.float32)
                    T["pairs"]["S2 vs S1+S2"], dis["sensor"] = pair(f"{task}: {BASE} S2 vs S1+S2", ref, other, groups)
                    rows.append(row_of(task, "S2 vs S1+S2", T["pairs"]["S2 vs S1+S2"]))
                    galileo = "galileo_base" if "galileo_base" in dis else next(m for m in enc_sets)
                    per = {}
                    for t in range(len(ref["dec"])):
                        m = (groups[:, 0, 0] == t)[:, None, None] & ref["ok"]
                        if (dis["sensor"] & m).sum() >= MIN_TILE_WINDOWS and (dis[galileo] & m).sum() >= MIN_TILE_WINDOWS:
                            per[t] = phi(dis["sensor"][m], dis[galileo][m])
                    finite = {t: v for t, v in per.items() if np.isfinite(v)}
                    og = over_groups({t: MAX_PHI - v for t, v in finite.items()})
                    T["sensor_vs_encoder"] = {"encoder": galileo, "pooled_phi": phi(dis["sensor"][ref["ok"]], dis[galileo][ref["ok"]]), "n_tiles_defined": len(finite),
                                              "tiles_below": og["w"], "tiles_at_or_above": og["l"], "ties": og["t"], "sign_p": og["sign_p"],
                                              "per_tile_phi": {"median": float(np.median(list(finite.values()))) if finite else None},
                                              "sensor_vs_each_encoder": {m: phi(dis["sensor"][ref["ok"]], d[ref["ok"]]) for m, d in enc_sets.items()}}
                    print(f"{task}: sensor set vs {galileo} set phi {fmt(T['sensor_vs_encoder']['pooled_phi'])} pooled, tiles below 0.5: {og['w']}/{og['l']} (p={og['sign_p']:.2g})", flush=True)
                except Exception as ex:  # noqa: BLE001
                    e57.fail(summary, f"{task} sensor pair", ex)
        except Exception as ex:  # noqa: BLE001
            e57.fail(summary, task, ex)
    np.savez_compressed(os.path.join(hb.OUT, f"exp63_masks{suffix}.npz"), **masks)
    try:
        prereg(summary)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp63_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    if rows:
        with open(os.path.join(hb.OUT, f"exp63_multiclass_atlas{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
