#!/usr/bin/env python
"""exp54: the window protocol on Ai2's published embeddings for the dense multi-class tasks.

Why. Every dense hand-labelled result here is binary flood water on one dataset, so the confidence ranking, the
boundary-first review order and the shared-error finding have never met a task with many classes; exp16 found the
boundary cue collapsing into low margin on nine WorldCover classes against a weak reference, and issue 7 has asked
for a dense few-class expert testbed since. Ai2's paper embeddings (allenai/olmoearth-paper-embeddings) carry the
Table-2 segmentation tasks with labels: MADOS (marine debris, 15 classes), PASTIS (crop types, 19 classes, Sentinel-2
alone and with Sentinel-1), and GEO-Bench's m_cashew_plant and m_sa_crop_type, for two dozen encoders. exp51's port
of their probe recipe reads them. So the protocol can be run on four more tasks, multi-class, several encoders, with
no encoder pass, at the price of one limitation stated up front: the embeddings carry no pixels, so the no-model
control is the embedding-space one (distance to the k-means centroids of the training embeddings, SHRUG-FM's
normalized distance, rejected as a ranker on floods in exp49) and the model-side one (disagreement between encoders),
not a pixel statistic.

Design. Per task and encoder: their probe recipe in fp32 (LinearProbe, classes x pp x pp logits per token, AdamW at
the task's paper-best probe lr from eval_settings/base_settings, warm-up then cosine, cross-entropy ignoring -1,
50 epochs) on their train split, graded on their test split. Pixel mIoU and accuracy as they report; then 4-px
windows: window class probabilities are the mean of the pixel probabilities, the decision their argmax, the label the
majority class of the window's labelled pixels (at least eight), the error set decision != label. Rankers on that
error set: confidence = -(top-1 - top-2 window probability margin); predictive entropy; the boundary-first order
(boundary = share of the eight neighbouring windows with a different decision, then confidence within each band,
assess.boundary_first_score); embedding normalized distance to the nearest of 64 training centroids (the control);
cross-encoder disagreement (share of the other encoders on the task whose window decision differs; rows aligned by
the label tile). Scores: pooled excess AURC, per-tile exact sign tests against confidence, error capture at 5/10/20%
budgets, and for boundary-first the per-tile capture comparison at 5% and 10% (exp36's test). Cross-encoder phi of
the window errors against OlmoEarth Base per task.
Encoders: olmoearth_base, galileo_base, croma_base, terramind_base, clay_large, anysat on MADOS and PASTIS-S2;
olmoearth_base alone on m_cashew_plant (9 GB of training embeddings), m_sa_crop_type (20 GB) and PASTIS-S1+S2.

Preregistered, one-sided, stated before the run:
  P1  on every task, OlmoEarth Base's probe confidence beats the embedding-distance control: pooled lead >= 0.001
      and a per-tile sign test (confidence better) p < 0.05.
Stated prediction, graded descriptively: boundary-first loses to confidence at the 5% budget on the tasks with 15
and 19 classes (MADOS, PASTIS) and is open on 7 and 10 classes (cashew, SA crop type); exp16's collapse says the
former, exp36 says the latter.
Secondary: entropy against confidence; cross-encoder disagreement where two or more encoders ran; the phi table;
PASTIS S2 against S1+S2 for OlmoEarth (the frozen form of the sensor question).

Inputs: the Hub dataset, downloaded into HF_HOME/paper_embeddings. Outputs: exp/out/exp54_summary.json,
exp/out/exp54_multiclass.csv. Run with ~/oe12/.venv. --smoke: synthetic embeddings on CPU exercising every code
path, no downloads, _smoke outputs.
"""
import csv
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp51_their_probe as e51  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, oracle_aurc  # noqa: E402
from oe_inferencex.reliability import centroid_signals, kmeans  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

DEV = e51.DEV
HUB = e51.HUB
WIN = 4
EPOCHS, BATCH = 50, 64
TASK_LR = {"mados": 0.005, "pastis_sentinel2": 0.01, "pastis_sentinel1_sentinel2": 0.01, "m_cashew_plant": 0.1, "m_sa_crop_type": 0.5}   # eval_settings/base_settings, OlmoEarth Base
TASKS = {"mados": ["olmoearth_base", "galileo_base", "croma_base", "terramind_base", "clay_large", "anysat"],
         "pastis_sentinel2": ["olmoearth_base", "galileo_base", "croma_base", "terramind_base", "clay_large", "anysat"],
         "m_cashew_plant": ["olmoearth_base"], "m_sa_crop_type": ["olmoearth_base"], "pastis_sentinel1_sentinel2": ["olmoearth_base"]}
BASE = "olmoearth_base"
CONF, ENT, BFIRST, CTRL_E, DISAGREE = "probe confidence", "predictive entropy", "boundary first, then confidence", "control embedding distance", "cross-encoder disagreement"
BUDGETS = (0.05, 0.10, 0.20)


def load_theirs(model, task, split, cache_dir):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HUB, f"{model}/{task}/{split}.pt", repo_type="dataset", cache_dir=cache_dir)
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["embeddings"], d["labels"].to(torch.int64)


def train_probe(emb, lab, pp, C, lr, seed=0):
    """Their segmentation probe recipe in fp32 for C classes (exp51's, generalised)."""
    torch.manual_seed(seed)
    N, h, w, D = emb.shape
    probe = e51.LinearProbe(in_dim=D, out_dim=C * pp * pp).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    steps = int(np.ceil(N / BATCH))
    g = torch.Generator().manual_seed(seed)
    probe.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(N, generator=g)
        for i in range(steps):
            idx = order[i * BATCH:(i + 1) * BATCH]
            x = emb[idx].to(DEV, dtype=torch.float32); y = lab[idx].to(DEV)
            lg = probe(x)["logits"].reshape(len(idx), h, w, C, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(len(idx), C, h * pp, w * pp)
            if lg.shape[-2:] != y.shape[-2:]:
                lg = torch.nn.functional.interpolate(lg, size=tuple(y.shape[-2:]), mode="bilinear", align_corners=True)
            loss = loss_fn(lg, y)
            loss.backward()
            e51.adjust_learning_rate(optimizer=opt, epoch=epoch + i / steps, total_epochs=EPOCHS, warmup_epochs=int(EPOCHS * 0.1), max_lr=lr, min_lr=1.0e-5)
            opt.step(); opt.zero_grad()
    return probe.eval()


def window_quantities(probe, emb, lab, pp, C):
    """Per tile, 4-px windows: class-probability means, decision, majority label, validity; pixel accuracy and mIoU."""
    N, H, W = lab.shape
    hw, ww = H // WIN, W // WIN
    wp = np.zeros((N, C, hw, ww), dtype=np.float32)
    pix_correct = 0; pix_total = 0
    inter = np.zeros(C); union = np.zeros(C)
    with torch.no_grad():
        for i in range(0, N, 128):
            x = emb[i:i + 128].to(DEV, dtype=torch.float32)
            n, h, w = x.shape[:3]
            lg = probe(x)["logits"].reshape(n, h, w, C, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(n, C, h * pp, w * pp)
            if lg.shape[-2:] != (H, W):
                lg = torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=True)
            p = torch.softmax(lg, dim=1)
            y = lab[i:i + 128].to(DEV)
            pred = p.argmax(1)
            m = y >= 0
            pix_correct += int(((pred == y) & m).sum()); pix_total += int(m.sum())
            for c in range(C):
                inter[c] += int(((pred == c) & (y == c) & m).sum()); union[c] += int((((pred == c) | (y == c)) & m).sum())
            wp[i:i + n] = p[:, :, :hw * WIN, :ww * WIN].reshape(n, C, hw, WIN, ww, WIN).mean(dim=(3, 5)).cpu().numpy()
    valid_c = union > 0
    miou = float(np.mean(inter[valid_c] / union[valid_c])) if valid_c.any() else float("nan")
    # majority label per window
    l = lab[:, :hw * WIN, :ww * WIN].numpy().reshape(N, hw, WIN, ww, WIN).transpose(0, 1, 3, 2, 4).reshape(N, hw, ww, WIN * WIN)
    valid = l >= 0
    n_valid = valid.sum(-1)
    ok = n_valid >= (WIN * WIN) // 2
    counts = np.zeros((N, hw, ww, C), dtype=np.int32)
    for c in range(C):
        counts[..., c] = ((l == c) & valid).sum(-1)
    y_win = counts.argmax(-1)
    dec = wp.argmax(1)
    srt = np.sort(wp, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    ent = -(np.clip(wp, 1e-7, 1) * np.log(np.clip(wp, 1e-7, 1))).sum(1)
    err = ((dec != y_win) & ok).astype(np.float64)
    return {"dec": dec, "y": y_win, "ok": ok, "err": err, "margin": margin, "entropy": ent, "pixel_accuracy": pix_correct / max(pix_total, 1), "pixel_miou": miou, "hw": (hw, ww)}


def boundary_share(dec):
    """Share of the eight neighbours whose decision differs, edge-padded; multi-class form of exp18.boundary."""
    pad = np.pad(dec, ((0, 0), (1, 1), (1, 1)), mode="edge")
    N, h, w = dec.shape
    nb = np.zeros(dec.shape, dtype=np.float64)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[:, 1 + di:1 + di + h, 1 + dj:1 + dj + w] != dec)
    return nb / 8.0


def per_tile_capture(sig, err, ok, budget):
    """Share of a tile's errors inside the top-`budget` fraction of its labelled windows by the ranker (ties broken by order)."""
    out = []
    for t in range(len(err)):
        m = ok[t]
        if m.sum() < 8 or err[t][m].sum() == 0:
            out.append(np.nan); continue
        s, e = np.asarray(sig[t])[m], err[t][m]
        k = max(1, int(round(budget * len(s))))
        top = np.argsort(-s, kind="stable")[:k]
        out.append(float(e[top].sum() / e.sum()))
    return np.array(out)


def label_keys(lab):
    return [hashlib.sha1(np.ascontiguousarray(t.astype(np.int16)).tobytes()).hexdigest() for t in lab]


def score(sig, err, ok, ref, primary):
    r = e51.score(sig, err, ok, ref, primary)
    r.pop("per_tile_eaurc_ref", None); r.pop("tiles", None)
    r["capture_pooled"] = {k: {str(b): float(capture_at_budget_expected(np.asarray(v)[ok], err[ok], budgets=(b,))[b]) for b in BUDGETS} for k, v in sig.items()}
    return r


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), help="tasks to run")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp54 the window protocol on Ai2's multi-class embeddings", "smoke": args.smoke,
               "config": {"tasks": {t: TASKS[t] for t in args.tasks}, "lr": TASK_LR, "epochs": EPOCHS, "window": WIN, "budgets": BUDGETS,
                          "prereg": "P1 on every task OlmoEarth Base's probe confidence beats the embedding-distance control (pooled lead >= 0.001, one-sided per-tile sign test); "
                                    "stated prediction: boundary-first loses to confidence at the 5% budget on 15 and 19 classes, open on 7 and 10"},
               "results": {}, "failures": []}
    rows = []
    cache_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "paper_embeddings")
    for task in args.tasks:
        models = ["olmoearth_base", "other_a", "other_b"] if args.smoke else TASKS[task]
        held = {}                                                            # model -> (dec, err, ok, keys)
        for model in models:
            t0 = time.time()
            try:
                if args.smoke:
                    rng = np.random.default_rng(abs(hash(model)) % 1000)
                    C, N, h, D = 5, 6, 8, 16
                    emb_tr, emb_te = torch.tensor(rng.standard_normal((N, h, h, D)), dtype=torch.float32), torch.tensor(rng.standard_normal((N, h, h, D)), dtype=torch.float32)
                    base = np.repeat(np.repeat(np.random.default_rng(0).integers(0, C, (N, h, h)), 4, 1), 4, 2)   # the same labels for every model, as on the Hub
                    lab_tr, lab_te = torch.tensor(base), torch.tensor(base)
                    lab_te[:, :3, :] = -1
                else:
                    emb_tr, lab_tr = load_theirs(model, task, "train", cache_dir)
                    emb_te, lab_te = load_theirs(model, task, "test", cache_dir)
                C = int(max(int(lab_tr.max()), int(lab_te.max())) + 1)
                pp = lab_tr.shape[-1] // emb_tr.shape[1]
                probe = train_probe(emb_tr, lab_tr, pp, C, TASK_LR.get(task, 0.1)) if not args.smoke else train_probe(emb_tr, lab_tr, pp, C, 0.01)
                q = window_quantities(probe, emb_te, lab_te, pp, C)
                # embedding-distance control on the token grid, pooled to the 4-px window grid
                cent, _, intra = kmeans(emb_tr.reshape(-1, emb_tr.shape[-1]).float().numpy(), k=min(64, len(emb_tr) * emb_tr.shape[1] ** 2 // 2), iters=25, seed=0)
                nd = centroid_signals(emb_te.reshape(-1, emb_te.shape[-1]).float().numpy(), cent, intra)["normalized distance"].reshape(len(emb_te), emb_te.shape[1], emb_te.shape[2])
                hw, ww = q["hw"]
                nd_win = torch.nn.functional.interpolate(torch.tensor(nd)[:, None], size=(hw, ww), mode="bilinear", align_corners=True)[:, 0].numpy()
                bnd = boundary_share(q["dec"])
                sig = {CONF: -q["margin"], ENT: q["entropy"], CTRL_E: nd_win,
                       BFIRST: np.stack([boundary_first_score(-q["margin"][t], bnd[t]) for t in range(len(bnd))])}
                r = score(sig, q["err"], q["ok"], CONF, [CTRL_E])
                r.update({"seconds": time.time() - t0, "classes": C, "train_tiles": int(len(emb_tr)), "test_tiles": int(len(emb_te)), "token_grid": list(emb_tr.shape[1:3]),
                          "pixels_per_token_side": int(pp), "window_grid": [hw, ww], "pixel_accuracy": q["pixel_accuracy"], "pixel_miou": q["pixel_miou"],
                          "boundary_windows_share": float(bnd[q["ok"]].__gt__(0).mean()), "share_of_errors_on_boundary": float((bnd[q["ok"]] > 0)[q["err"][q["ok"]] > 0].mean()) if q["err"][q["ok"]].sum() else None})
                # exp36's test: boundary-first against confidence at fixed budgets, per tile
                r["boundary_first_vs_confidence"] = {}
                for b in (0.05, 0.10):
                    ca, cb = per_tile_capture(sig[BFIRST], q["err"], q["ok"], b), per_tile_capture(sig[CONF], q["err"], q["ok"], b)
                    g = (ca - cb)[np.isfinite(ca) & np.isfinite(cb)]
                    w_, l_, t_ = wins_losses_ties(g)
                    r["boundary_first_vs_confidence"][str(b)] = {"w": w_, "l": l_, "t": t_, "sign_p": sign_test(w_, l_, "greater"), "mean_gain": float(g.mean()) if len(g) else None,
                                                                 "pooled_boundary_first": r["capture_pooled"][BFIRST][str(b)], "pooled_confidence": r["capture_pooled"][CONF][str(b)]}
                held[model] = (q["dec"], q["err"], q["ok"], label_keys(lab_te.numpy()), sig, q)
                summary["results"].setdefault(task, {})[model] = r
                bf = r["boundary_first_vs_confidence"]["0.05"]
                print(f"{task}/{model}: {C} classes, grid {r['token_grid']}, test {r['test_tiles']} tiles | pixel mIoU {r['pixel_miou']:.3f} acc {r['pixel_accuracy']:.4f} | window acc {r['accuracy']:.4f}, "
                      f"{r['n_errors']} errors | E-AURC " + ", ".join(f"{k} {v:.4f}" for k, v in r["pooled_eaurc"].items()) +
                      f" | lead over embedding distance {r['tests'][CTRL_E]['pooled_lead']:+.4f} (p={r['tests'][CTRL_E]['sign_p']:.2g}) | boundary-first at 5%: {bf['pooled_boundary_first']:.3f} vs {bf['pooled_confidence']:.3f}, tiles {bf['w']}/{bf['l']}, p={bf['sign_p']:.2g}", flush=True)
                rows.append({"task": task, "model": model, "classes": C, "test_tiles": r["test_tiles"], "pixel_miou": r["pixel_miou"], "window_acc": r["accuracy"],
                             **{f"eaurc {k}": v for k, v in r["pooled_eaurc"].items()}, "capture5 confidence": r["capture_pooled"][CONF]["0.05"], "capture5 boundary first": r["capture_pooled"][BFIRST]["0.05"],
                             "boundary_first_5pct_w": bf["w"], "boundary_first_5pct_l": bf["l"], "boundary_first_5pct_p": bf["sign_p"]})
                del emb_tr, emb_te
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": f"{task}/{model}", "error": repr(ex), "traceback": traceback.format_exc()})
                print(f"{task}/{model} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
        # cross-encoder disagreement and phi against OlmoEarth Base, rows aligned by the label tile
        if BASE in held and len(held) >= 2:
            try:
                dec0, err0, ok0, k0, sig0, q0 = held[BASE]
                pos0 = {k: i for i, k in enumerate(k0)}
                aligned = {}
                for m, (dec, err, ok, k, _, _) in held.items():
                    idx = np.array([pos0.get(x, -1) for x in k])
                    # The guard must include "idx is a permutation". Two tiles with identical labels hash to one key,
                    # so pos0 keeps only the last of them; where such tiles are not adjacent, np.argsort(idx) permutes
                    # rows even when a model is aligned against itself, and every other condition here still passes.
                    # That is what corrupted this block's phi on PASTIS, reported 2026-09-20; exp63 measures the pair
                    # at 0.50-0.68 where this reported 0.05-0.08, and 0.08 is what phi becomes under scrambling.
                    is_perm = len(np.unique(idx)) == len(idx) == len(k0)
                    if (idx >= 0).all() and is_perm and dec.shape == dec0.shape:
                        aligned[m] = (dec[np.argsort(idx)], err[np.argsort(idx)], ok[np.argsort(idx)])
                    elif (idx >= 0).all() and not is_perm:
                        summary["failures"].append({"part": f"{task}/{m}", "error": "label-tile keys are not unique: "
                                                    f"{len(k0) - len(np.unique(idx))} of {len(k0)} tiles collide, so rows "
                                                    "cannot be aligned by label hash; skipped rather than scrambled"})
                others = [m for m in aligned if m != BASE]
                if others:
                    dis = np.mean([(aligned[m][0] != dec0) for m in others], axis=0)
                    sig = dict(sig0); sig[DISAGREE] = dis
                    r = score(sig, err0, ok0, CONF, [CTRL_E])
                    summary["results"][task][BASE]["with_disagreement"] = {"n_other_encoders": len(others), "pooled_eaurc": r["pooled_eaurc"], "tests": {DISAGREE: r["tests"][DISAGREE]}, "capture_pooled": r["capture_pooled"][DISAGREE]}
                    summary["results"][task]["phi_vs_olmoearth_base"] = {}
                    for m in others:
                        both = aligned[m][2] & ok0
                        summary["results"][task]["phi_vs_olmoearth_base"][m] = {"phi": e51.phi(aligned[m][1][both], err0[both]), "p_base_wrong_given_model_wrong": float(err0[both][aligned[m][1][both] > 0].mean()) if (aligned[m][1][both] > 0).any() else None}
                    print(f"{task}: disagreement over {len(others)} encoders E-AURC {r['pooled_eaurc'][DISAGREE]:.4f} vs confidence {r['pooled_eaurc'][CONF]:.4f} (lead {r['tests'][DISAGREE]['pooled_lead']:+.4f}); phi vs olmoearth_base " +
                          ", ".join(f"{m} {v['phi']:.3f}" for m, v in summary["results"][task]["phi_vs_olmoearth_base"].items()), flush=True)
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": f"{task}/cross-encoder", "error": repr(ex), "traceback": traceback.format_exc()})
        with open(os.path.join(hb.OUT, f"exp54_summary{suffix}.partial.json"), "w") as f:
            json.dump(hb.json_ready(summary), f, indent=1)

    p1 = [summary["results"].get(t, {}).get(BASE, {}).get("tests", {}).get(CTRL_E) for t in args.tasks]
    p1 = [bool(x["pooled_lead"] >= 0.001 and x["sign_p"] < 0.05) if x else None for x in p1]
    summary["prereg"] = {"P1_per_task": dict(zip(args.tasks, p1)), "P1": bool(all(p1)) if all(x is not None for x in p1) else None, "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp54_multiclass{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp54_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp54_summary{suffix}.json; prereg {summary['prereg']}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
