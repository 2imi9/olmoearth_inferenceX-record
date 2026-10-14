"""exp74: the suite under every encoder Ai2 published it for.

exp70 showed the margin beats the best no-model control on all 24 of Ai2's tasks with OlmoEarth Base. exp73
showed it beats the ensemble, nearest-neighbour and Mahalanobis alternatives on the same tasks. Both are about
one encoder, so the sharpest remaining objection is that the finding is a property of OlmoEarth. Ai2 published
the same task embeddings for twenty-six other models, with their own per-model probe settings, so the objection
is answerable at no encoder cost: exp70's protocol, unchanged, on every encoder with enough of the suite.

Which encoders. Every model directory on the Hub that carries at least 20 of exp70's 24 tasks, a rule set before
the listing was read against it; that admits fifteen: three families with all 24 (AnySat, Clay large, Panopticon),
the OlmoEarth size series (tiny, nano, large; base is exp70), and Galileo (three sizes), CROMA (two), TerraMind
(two), Satlas and Copernicus-FM with 20 or 21. DINOv3, Presto, Prithvi and TESSERA carry 3 to 15 tasks and are
excluded by the rule, which is recorded rather than the tasks being cherry-picked among them.

Which probes. Ai2's own per-model settings from eval_settings/: the merged file for the outside families, the
per-size files for OlmoEarth. Where a task names a probe learning rate it is used; where it does not, exp54's
default of 0.1 on the window path and exp51's classification recipe, as exp70 used them. Nothing else changes:
the same graded units, the same no-model controls, the same scoring, the same tie-aware statistics.

Preregistered, one-sided, before the run.
  P1  for every encoder scored, the margin beats the best no-model control on at least 75% of that encoder's
      scored tasks, sign test p < 0.05 (exp70's 18-of-24 bar, scaled to the tasks the encoder has).
  P2  for each of the three outside families that carry all 24 tasks, on at least 22 of 24.
  P3  descriptive: across the OlmoEarth size series the margin's median lead, beside each size's accuracy, so
      whether the signal's advantage grows, shrinks or holds with encoder capacity is on the record.
Falsification. If any encoder falls below 75%, the front page's "on tasks we did not choose" must name the
encoders it holds for; if an outside family fails P2, the result is a property of the OlmoEarth family and
the claim narrows to it. A task an encoder does not carry is recorded as absent, never as a loss or a win.

Each (encoder, task) result is checkpointed the moment it finishes, as exp73 learned to do.
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
from oe_inferencex import stats                   # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PARTS = os.path.join(OUT, "exp74_parts")
MIN_TASKS = 20
ENCODERS = ["anysat", "clay_large", "panopticon",
            "olmoearth_tiny", "olmoearth_nano", "olmoearth_large",
            "galileo_tiny", "galileo_nano", "galileo_base",
            "croma_base", "croma_large", "terramind_base", "terramind_large",
            "satlas_base", "copernicusfm"]
OUTSIDE_FULL = ("anysat", "clay_large", "panopticon")
SIZE_SERIES = ("olmoearth_tiny", "olmoearth_nano", "olmoearth_base", "olmoearth_large")
SETTINGS_MERGED = "eval_settings/max_eval_settings_per_group_merged.enriched.json"
SETTINGS_BY_SIZE = {"olmoearth_tiny": "eval_settings/tiny_settings.enriched.json",
                    "olmoearth_nano": "eval_settings/nano_settings.enriched.json",
                    "olmoearth_large": "eval_settings/large_settings.enriched.json"}


def probe_lrs(enc, cache):
    """Ai2's per-task probe learning rates for one encoder, from their published settings files."""
    from huggingface_hub import hf_hub_download
    if enc in SETTINGS_BY_SIZE:
        path = hf_hub_download(e70.HUB, SETTINGS_BY_SIZE[enc], repo_type="dataset", cache_dir=cache)
        block = next(iter(json.load(open(path)).values()))      # one run group per size file
    else:
        path = hf_hub_download(e70.HUB, SETTINGS_MERGED, repo_type="dataset", cache_dir=cache)
        block = json.load(open(path))[enc]
    return parse_lrs(block)


def parse_lrs(block):
    return {t: (v.get("settings") or {}).get("probe_lr") for t, v in block.items() if isinstance(v, dict)}


def window_embeddings(x, hw, ww, H, W, win):
    """The patch embedding under each window, for encoders whose patch grid is not the window grid.

    exp70 took one patch per window, which holds for OlmoEarth's grid and failed with an index error on AnySat,
    Clay, Panopticon, CROMA, TerraMind and Satlas, whose patches are coarser or finer than 4 px: 54 of the first
    run's 82 absences were this. Window (i, j) covers pixels [i*win, (i+1)*win); the patch containing its first
    pixel is (i*win*h // H, j*win*w // W). When the grid coincides the map is the identity, so exp70's numbers
    are reproduced exactly on OlmoEarth."""
    N, h, w, D = x.shape
    pr = np.minimum(np.arange(hw) * win * h // H, h - 1)
    pc = np.minimum(np.arange(ww) * win * w // W, w - 1)
    return x[:, pr[:, None], pc[None, :], :]                  # (N, hw, ww, D)


def run_segmentation_any_grid(task, cache, seed=0):
    """exp70.run_segmentation with the embedding control read per window rather than per patch."""
    import torch
    import exp54_multiclass_embeddings as e54
    import exp73_alternatives_suite as e73
    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
    pp = int(round(ytr.shape[-1] / xtr.shape[1]))
    lr = e54.TASK_LR.get(task, 0.1)
    probe = e54.train_probe(xtr, ytr, pp, C, lr, seed=seed)
    q_te = e54.window_quantities(probe, xte, yte, pp, C)
    q_tr = e54.window_quantities(probe, xtr, ytr, pp, C)
    ok = q_te["ok"].ravel()
    err = q_te["err"].ravel()[ok]
    dec = q_te["dec"].ravel()[ok]
    sig = {"margin": -q_te["margin"].ravel()[ok].astype(np.float64),
           "entropy": q_te["entropy"].ravel()[ok].astype(np.float64)}
    hw, ww = q_te["hw"]
    H, W = yte.shape[1], yte.shape[2]
    xte32 = xte.to(torch.float32).numpy()
    ete = window_embeddings(xte32, hw, ww, H, W, e54.WIN).reshape(-1, xte.shape[-1])[ok]
    etr = xtr.to(torch.float32).numpy().reshape(-1, xtr.shape[-1])
    sig.update(e73.controls_chunked(ete, etr, dec))            # exp70's controls, accumulated in chunks
    ok_tr = q_tr["ok"]
    return {"family": "segmentation", "n_units": int(ok.sum()), "n_classes": C, "patch_px": pp,
            "patch_grid": [int(xte.shape[1]), int(xte.shape[2])], "window_grid": [int(hw), int(ww)],
            "test_accuracy": float(1.0 - err.mean()),
            "train_accuracy": float(1.0 - q_tr["err"][ok_tr].mean()),
            "pixel_accuracy": q_te["pixel_accuracy"], "pixel_miou": q_te["pixel_miou"],
            "error_rate": float(err.mean()), "signals": e70.score_task(sig, err, C)}


def run_encoder(enc, cache, redo=False):
    """exp70's two runners on one encoder, with that encoder's probe settings, checkpointed per task."""
    import exp54_multiclass_embeddings as e54
    lrs = probe_lrs(enc, cache)
    e70.MODEL = enc
    e54.TASK_LR = {t: float(lr) for t, lr in lrs.items() if lr is not None}
    results, absent = {}, []
    for task in e70.TASKS_CLS + e70.TASKS_SEG:
        part = os.path.join(PARTS, f"{enc}__{task}.json")
        if os.path.exists(part) and not redo:
            with open(part) as fh:
                r = json.load(fh)
            if r.get("absent"):
                absent.append(task)
            else:
                results[task] = r
            continue
        fn = e70.run_classification if task in e70.TASKS_CLS else run_segmentation_any_grid
        t0 = time.time()
        try:
            r = fn(task, cache, seed=0)
        except Exception as exc:                                   # a task the encoder does not carry
            msg = f"{type(exc).__name__}: {str(exc)[:160]}"
            with open(part, "w") as fh:
                json.dump({"absent": True, "reason": msg}, fh)
            absent.append(task)
            print(f"  {enc:<18} {task:44s} absent ({msg[:60]})", flush=True)
            continue
        cname, cval = e70.best_control(r["signals"])
        r["best_control"], r["margin_lead"] = cname, float(cval - r["signals"]["margin"]["excess_aurc"])
        r["source"], r["probe_lr"] = e70.source_of(task), e54.TASK_LR.get(task)
        r["seconds"] = round(time.time() - t0, 1)
        with open(part, "w") as fh:
            json.dump(r, fh, default=float)
        results[task] = r
        print(f"  {enc:<18} {task:44s} {r['family'][:3]} n={r['n_units']:7d} acc {r['test_accuracy']:.3f} "
              f"lead {r['margin_lead']:+.4f} over {cname[4:]} ({r['seconds']:.0f}s)", flush=True)
    return results, absent


def encoder_verdict(results):
    tasks = sorted(results)
    wins = [t for t in tasks if results[t]["margin_lead"] > 0]
    losses = [t for t in tasks if results[t]["margin_lead"] < 0]
    p = float(stats.sign_test(len(wins), len(losses), alternative="greater")) if tasks else 1.0
    by_src = collections.defaultdict(list)
    for t in tasks:
        by_src[results[t]["source"]].append(results[t]["margin_lead"])
    return {"scored": len(tasks), "wins": len(wins), "losses": len(losses), "p": p,
            "share": len(wins) / len(tasks) if tasks else float("nan"),
            "median_lead": float(np.median([results[t]["margin_lead"] for t in tasks])) if tasks else float("nan"),
            "median_accuracy": float(np.median([results[t]["test_accuracy"] for t in tasks])) if tasks else float("nan"),
            "source_wins": sum(1 for v in by_src.values() if float(np.mean(v)) > 0), "distinct_sources": len(by_src),
            "lost_on": sorted(losses)}


def verdicts(per_encoder, base=None):
    """P1 over every encoder, P2 over the three outside families, P3 the size series."""
    v = {"P1": {"holds": bool(per_encoder) and all(x["share"] >= 0.75 and x["p"] < 0.05 for x in per_encoder.values()),
                "per_encoder": {e: {k: x[k] for k in ("scored", "wins", "losses", "p", "share")} for e, x in per_encoder.items()},
                "encoders_below_bar": sorted(e for e, x in per_encoder.items() if not (x["share"] >= 0.75 and x["p"] < 0.05))}}
    outside = {e: per_encoder[e] for e in OUTSIDE_FULL if e in per_encoder}
    v["P2"] = {"holds": bool(outside) and all(x["scored"] == 24 and x["wins"] >= 22 for x in outside.values()),
               "per_family": {e: f"{x['wins']}/{x['scored']}" for e, x in outside.items()}}
    series = {}
    for e in SIZE_SERIES:
        x = per_encoder.get(e) or (base if e == "olmoearth_base" else None)
        if x:
            series[e] = {"median_lead": x["median_lead"], "median_accuracy": x["median_accuracy"], "wins": x["wins"],
                         "scored": x["scored"]}
    v["P3"] = {"note": "descriptive: the margin's median lead and the accuracy across the OlmoEarth sizes", "series": series}
    return v


def cmd_all(args):
    cache = os.environ.get("HF_HOME")
    os.makedirs(PARTS, exist_ok=True)
    per_encoder, all_results, absent = {}, {}, {}
    for enc in (args.only or ENCODERS):
        t0 = time.time()
        try:
            res, abs_ = run_encoder(enc, cache, redo=args.redo)
        except Exception as exc:
            print(f"  {enc:<18} FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            absent[enc] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        all_results[enc], absent[enc] = res, abs_
        per_encoder[enc] = encoder_verdict(res)
        x = per_encoder[enc]
        print(f"== {enc:<18} {x['wins']}/{x['scored']} tasks, p={x['p']:.1e}, median lead {x['median_lead']:+.4f}, "
              f"median acc {x['median_accuracy']:.3f}, absent {len(abs_)} ({(time.time()-t0)/60:.0f} min)", flush=True)
    base = None
    p70 = os.path.join(OUT, "exp70_summary.json")
    if os.path.exists(p70):
        r70 = json.load(open(p70))["results"]["tasks"]
        base = encoder_verdict({t: {"margin_lead": r70[t]["margin_lead"], "source": r70[t]["source"],
                                    "test_accuracy": r70[t]["test_accuracy"]} for t in r70})
    summary = {"experiment": "exp74 the suite under every encoder Ai2 published it for",
               "config": {"encoders": ENCODERS, "rule": f"at least {MIN_TASKS} of exp70's 24 tasks on the Hub",
                          "settings": [SETTINGS_MERGED] + sorted(SETTINGS_BY_SIZE.values()),
                          "controls": list(e70.CONTROLS), "budgets": list(e70.BUDGETS)},
               "results": {"per_encoder": per_encoder, "tasks": all_results, "absent": absent,
                           "olmoearth_base_from_exp70": base},
               "verdicts": verdicts(per_encoder, base) if per_encoder else {}}
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp74_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    rows = [{"encoder": e, **{k: x[k] for k in ("scored", "wins", "losses", "p", "share", "median_lead", "median_accuracy",
                                                   "source_wins", "distinct_sources")}} for e, x in per_encoder.items()]
    if rows:
        with open(os.path.join(OUT, f"exp74_encoders{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for k, x in summary["verdicts"].items():
        print(f"{k}: {x.get('holds', '')} {json.dumps({a: b for a, b in x.items() if a not in ('holds', 'per_encoder', 'note')})[:200]}",
              flush=True)
    return summary


def smoke(args):
    """The settings parser and the verdicts on planted per-encoder results, torch-free."""
    lrs = parse_lrs({"a": {"settings": {"probe_lr": 0.5}}, "b": {"settings": {"probe_lr": None}}, "c": "junk"})
    assert lrs == {"a": 0.5, "b": None}
    rng0 = np.random.default_rng(1)
    x = rng0.standard_normal((2, 16, 16, 3))                     # 16x16 patches of 4 px on a 64 px tile
    same = window_embeddings(x, 16, 16, 64, 64, 4)
    assert np.array_equal(same, x), "on OlmoEarth's grid the map is the identity"
    coarse = rng0.standard_normal((2, 8, 8, 3))                   # 8x8 patches of 8 px on the same tile
    m = window_embeddings(coarse, 16, 16, 64, 64, 4)
    assert m.shape == (2, 16, 16, 3) and np.array_equal(m[:, 0], m[:, 1]) and np.array_equal(m[:, 0, 0], coarse[:, 0, 0])
    fine = rng0.standard_normal((2, 32, 32, 3))                   # 32x32 patches of 2 px
    f = window_embeddings(fine, 16, 16, 64, 64, 4)
    assert f.shape == (2, 16, 16, 3) and np.array_equal(f[:, 1, 1], fine[:, 2, 2])
    rng = np.random.default_rng(0)
    per = {}
    for i, enc in enumerate(ENCODERS):
        n = 24 if enc in OUTSIDE_FULL or enc.startswith("olmoearth") else 21
        leads = rng.uniform(0.01, 0.2, n)
        if enc == "satlas_base":
            leads[:8] = -0.01                                # planted: one encoder below the bar
        res = {f"t{j}": {"margin_lead": float(leads[j]), "source": f"s{j % 14}", "test_accuracy": 0.8} for j in range(n)}
        per[enc] = encoder_verdict(res)
    v = verdicts(per)
    assert v["P1"]["holds"] is False and v["P1"]["encoders_below_bar"] == ["satlas_base"]
    assert v["P2"]["holds"] is True and set(v["P2"]["per_family"]) == set(OUTSIDE_FULL)
    assert set(v["P3"]["series"]) == {"olmoearth_tiny", "olmoearth_nano", "olmoearth_large"}
    per["satlas_base"] = encoder_verdict({f"t{j}": {"margin_lead": 0.05, "source": "s", "test_accuracy": 0.8} for j in range(21)})
    assert verdicts(per)["P1"]["holds"] is True
    print("smoke OK: settings parsed, per-encoder verdicts, P1 fails on a planted encoder and holds once it is fixed, P2 and P3 shaped")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="run only these encoders")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    cmd_all(args)


if __name__ == "__main__":
    main()
