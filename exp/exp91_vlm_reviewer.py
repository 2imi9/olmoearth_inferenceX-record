#!/usr/bin/env python
"""exp91: a vision-language model as the reviewer, on Sen1Floods11 Bolivia (pilot, preregistered 4 October 2026).

Question. The package's labelled routes need a reviewer to say, for each sampled window, what is really there. Can a
general-purpose vision-language model (VLM) supply those labels from the imagery alone, and do the package's
reviewer-error bounds (`estimate --reviewer-false-alarm E0 --reviewer-miss E1`) keep the error-rate interval honest
when E0 and E1 are measured on a small calibration set? The harness follows VISTA (Han et al. 2026, arXiv:2610.02200):
the model sees the imagery directly, every view is indexed by the window it shows, and the model may request further
views (other bands, other extents) of any window before it answers.

Population and map. The 81,984 windows of 4 x 4 pixels of exp37's Bolivia table (exp/out/exp37_patches_bolivia.npz;
Sen1Floods11 Bolivia, a region held out from training, cut into 64 x 64 chips), each with exp18's map: a water head on
frozen OlmoEarth v1-Base tokens. The windows are exp18's 15 x 15 grid on each chip's top-left 60 x 60 pixels, kept
where at least half of a window's pixels are hand-labelled. A window's true class is the majority of its labelled
pixels (LabelHand: water or not), as exp18 defined it; `err` is 1 where the map's class differs. The population
error rate is 7,248 / 81,984 = 8.84%.

Samples (seeds fixed here, drawn by `prepare`).
  S: a simple random sample of 150 windows (oe_inferencex.estimate.sample_for_estimation, design "random", seed 91).
  C: a reviewer-calibration set, disjoint from S: 30 windows drawn at random among the windows with err = 1 and 30
     among those with err = 0 (seed 92). It measures how the VLM errs; it is not a sample of the map.
The 210 windows are shuffled together (seed 93) and renamed V001..V210, so the reviewer cannot tell S from C.

Reviewer. Claude (the session's model, recorded in the summary) as workflow agents, about 21 windows each, blind to
the map's class and to the hand labels. For each window it gets one image: the whole 64 x 64 chip and a 16 x 16
zoom, each in true colour (B04, B03, B02) and in a short-wave-infrared composite (B12, B08, B04; water is dark),
with the window outlined. It may run `python exp/exp91_vlm_reviewer.py view Vnnn --context K --bands rgb|swir|nir`
for more views. It answers, per window, water, land or ? (cannot judge). Its labels are graded against nothing it
can see. A window's `wrong` is 1 where the VLM's class differs from the map's, ? where the VLM answered ?.

Endpoints (`grade`), all on the labels as returned, nothing refitted.
  A. On S: the VLM's class against the hand-label class, on windows it judged; the share answered ?.
  B. On C: E0 = the share of truly correct windows (err = 0) the VLM's labels mark wrong, E1 = the share of truly
     wrong windows (err = 1) they mark right, each over judged windows, with one-sided 95% Clopper-Pearson upper
     bounds U0 and U1.
  C. On S, three 95% intervals for the population error rate, from `estimate_error_rate`: with the hand labels (the
     reference), with the VLM's labels (? counted both ways), and with the VLM's labels widened by
     reviewer_false_alarm = U0 and reviewer_miss = U1. For each: estimate, ends, width, whether it holds 8.84%.

Predictions.
  P1. On S, the VLM's class agrees with the hand-label class on at least 90% of the windows it judged.
  P2. The widened interval (VLM labels, U0, U1) holds the population error rate, 8.84%.
  P3. E1 > E0: the VLM misses the map's errors more often than it raises false alarms, since the map errs where water
      and land meet and the hand labels are hardest to reproduce.

What this cannot show. One draw of S: whether an interval holds the truth is one event, not a coverage rate. C has
30 windows per side, so U0 and U1 are wide. U0 and U1 are themselves estimated: the package's bound assumes the
stated rates hold, so with two one-sided 95% bounds the widened interval's guarantee is about 85% by the union bound,
not 95%. The reviewer sees L1C top-of-atmosphere imagery of one flood event; hand labellers saw the same scenes.
Report-only beyond P1-P3; no claim about other models, regions or classes.

Usage.
    uv run --no-sync python exp/exp91_vlm_reviewer.py prepare          # samples, key (outside the repo), views
    uv run --no-sync python exp/exp91_vlm_reviewer.py view V017 --context 32 --bands swir
    uv run --no-sync python exp/exp91_vlm_reviewer.py grade LABELS.json --key KEY.csv
Torch is not needed: the .pt file is read with the standard library and numpy.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from oe_inferencex import estimate as est  # noqa: E402

DATA = os.path.join(ROOT, "data", "floods", "flood_bolivia_data.pt")
TABLE = os.path.join(ROOT, "exp", "out", "exp37_patches_bolivia.npz")
VIEWS = os.path.join(ROOT, "exp", "out", "exp91_views")
L1C = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
BANDS = {"rgb": ("B04", "B03", "B02"), "swir": ("B12", "B08", "B04"), "nir": ("B08", "B04", "B03")}
N_S, N_C, SEED_S, SEED_C, SEED_ORDER = 150, 30, 91, 92, 93
P = 4                                                   # window side in pixels
_STORAGE = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64, "LongStorage": np.int64,
            "IntStorage": np.int32, "ShortStorage": np.int16, "ByteStorage": np.uint8, "CharStorage": np.int8,
            "BoolStorage": np.bool_}


def load_pt(path):
    """A torch.save file as a dict of numpy arrays, without torch: the zip's pickle with tensors rebuilt from storages."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *_):
                    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                                           strides=tuple(s * storage.itemsize for s in stride)).copy()
                return rebuild
            if module == "torch" and name in _STORAGE:
                return name
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage, key, _, _ = pid
            return np.frombuffer(z.read(f"{root}/data/{key}"), dtype=_STORAGE[storage])

    return Unpickler(z.open(f"{root}/data.pkl")).load()


def population():
    """The exp37 Bolivia windows with the hand-label class and the map's class per window."""
    t = np.load(TABLE)
    d = load_pt(DATA)
    lab = d["labels"][:, 0]
    tile, row, col = t["tile"].astype(int), t["row"].astype(int), t["col"].astype(int)
    err = (t["err"] > 0.5).astype(int)
    hand = np.zeros(len(tile), int)
    for i, (k, r, c) in enumerate(zip(tile, row, col)):
        block = lab[k, r * P:(r + 1) * P, c * P:(c + 1) * P]
        known = block[block >= 0]
        hand[i] = int((known > 0).mean() > 0.5) if known.size else -1
    mapc = np.where(err == 1, 1 - hand, hand)
    return {"tile": tile, "row": row, "col": col, "err": err, "hand": hand, "map": mapc, "conf": t["conf"], "s2": d["s2"]}


def _stretch(img, idx):
    """Three bands of a chip, each stretched between its own 2nd and 98th percentiles (top-of-atmosphere haze would
    otherwise tint a joint stretch blue)."""
    out = []
    for b in idx:
        x = img[L1C.index(b)].astype(np.float32)
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        out.append(np.clip((x - lo) / max(hi - lo, 1.0), 0, 1) ** 0.8)
    return (np.stack(out, -1) * 255).astype(np.uint8)


def render(s2, tile, row, col, context, bands, size):
    """A square view of `context` pixels around window (row, col) of a chip, the window outlined in yellow."""
    img = _stretch(s2[tile], BANDS[bands])
    H = img.shape[0]
    cy, cx = row * P + P / 2, col * P + P / 2
    half = context / 2
    y0 = int(round(min(max(cy - half, 0), H - context))); x0 = int(round(min(max(cx - half, 0), H - context)))
    crop = Image.fromarray(img[y0:y0 + context, x0:x0 + context]).resize((size, size), Image.NEAREST)
    s = size / context
    d = ImageDraw.Draw(crop)
    box = [(col * P - x0) * s, (row * P - y0) * s, ((col + 1) * P - x0) * s - 1, ((row + 1) * P - y0) * s - 1]
    d.rectangle(box, outline=(255, 230, 0), width=max(2, int(s / 4)))
    return crop


def composite(s2, tile, row, col):
    """One image per window: chip and zoom, true colour and short-wave infrared, with captions."""
    W = 300
    out = Image.new("RGB", (2 * W + 30, 2 * W + 70), "white")
    d = ImageDraw.Draw(out)
    panels = [("whole chip, true colour", 64, "rgb"), ("whole chip, SWIR (water dark)", 64, "swir"),
              ("zoom, true colour", 16, "rgb"), ("zoom, SWIR (water dark)", 16, "swir")]
    for k, (cap, ctx, bands) in enumerate(panels):
        x, y = 10 + (k % 2) * (W + 10), 25 + (k // 2) * (W + 25)
        out.paste(render(s2, tile, row, col, ctx, bands, W), (x, y))
        d.text((x, y - 15), cap, fill=(0, 0, 0))
    return out


def prepare(key_path):
    pop = population()
    n = len(pop["err"])
    valid = pop["hand"] >= 0
    s = est.sample_for_estimation(pop["conf"], N_S, design="random", valid=valid, seed=SEED_S)
    in_s = np.zeros(n, bool); in_s[s["indices"]] = True
    rng = np.random.default_rng(SEED_C)
    wrong_pool = np.flatnonzero((pop["err"] == 1) & valid & ~in_s)
    right_pool = np.flatnonzero((pop["err"] == 0) & valid & ~in_s)
    c_idx = np.concatenate([rng.choice(wrong_pool, N_C, replace=False), rng.choice(right_pool, N_C, replace=False)])
    rows = [("S", int(i)) for i in s["indices"]] + [("C", int(i)) for i in c_idx]
    order = np.random.default_rng(SEED_ORDER).permutation(len(rows))
    os.makedirs(VIEWS, exist_ok=True)
    manifest, key = {}, []
    for k, j in enumerate(order, 1):
        part, i = rows[j]
        vid = f"V{k:03d}"
        t, r, c = int(pop["tile"][i]), int(pop["row"][i]), int(pop["col"][i])
        composite(pop["s2"], t, r, c).save(os.path.join(VIEWS, f"{vid}.png"))
        manifest[vid] = {"chip": t, "row": r, "col": c}
        key.append({"id": vid, "set": part, "index": i, "chip": t, "row": r, "col": c, "hand": int(pop["hand"][i]),
                    "map": int(pop["map"][i]), "err": int(pop["err"][i])})
    json.dump(manifest, open(os.path.join(VIEWS, "manifest.json"), "w"), indent=0)
    with open(key_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader(); w.writerows(key)
    json.dump({"sample": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()}},
              open(key_path.replace(".csv", "_sample.json"), "w"))
    print(f"population {n} windows, {int(pop['err'].sum())} wrong ({100 * pop['err'].mean():.2f}%); "
          f"S {len(s['indices'])}, C {len(c_idx)}; views in {VIEWS}; key in {key_path}")


def view(vid, context, bands):
    m = json.load(open(os.path.join(VIEWS, "manifest.json")))[vid]
    s2 = load_pt(DATA)["s2"]
    out = os.path.join(VIEWS, "extra", f"{vid}_{bands}_{context}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(s2, m["chip"], m["row"], m["col"], int(context), bands, 400).save(out)
    print(out)


def cp_upper(k, n, a=0.05):
    """One-sided Clopper-Pearson upper bound for a binomial share, by bisection on the beta tail (no scipy)."""
    if n == 0 or k == n:
        return 1.0
    def tail(p):                                   # P(X <= k | n, p)
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > a else (lo, mid)
    return hi


def grade(labels_path, key_path, out_path):
    lab = {d["id"]: d["label"] for d in json.load(open(labels_path))}
    key = list(csv.DictReader(open(key_path)))
    pop = np.load(TABLE)
    n_pop = len(pop["err"]); true_rate = float((pop["err"] > 0.5).mean())
    cls = {"water": 1, "land": 0}

    def vlm_wrong(k):
        v = lab[k["id"]]
        return None if v == "?" else int(cls[v] != int(k["map"]))

    S = [k for k in key if k["set"] == "S"]; C = [k for k in key if k["set"] == "C"]
    judged_s = [k for k in S if lab[k["id"]] != "?"]
    agree = sum(cls[lab[k["id"]]] == int(k["hand"]) for k in judged_s)
    c0 = [k for k in C if k["err"] == "0" and lab[k["id"]] != "?"]
    c1 = [k for k in C if k["err"] == "1" and lab[k["id"]] != "?"]
    fa = sum(vlm_wrong(k) == 1 for k in c0); miss = sum(vlm_wrong(k) == 0 for k in c1)
    U0, U1 = cp_upper(fa, len(c0)), cp_upper(miss, len(c1))
    sample = json.load(open(key_path.replace(".csv", "_sample.json")))["sample"]
    sample["indices"] = np.asarray(sample["indices"])
    by_index = {int(k["index"]): k for k in S}
    order = [by_index[int(i)] for i in sample["indices"]]
    hand_wrong = np.array([int(k["err"]) for k in order], float)
    v = [vlm_wrong(k) for k in order]
    v_wrong = np.array([0 if x is None else x for x in v], float)
    v_unj = np.array([x is None for x in v])

    def iv(r):
        return {"estimate": r.get("estimate"), "estimate_range": r.get("estimate_range"), "low": r["low"], "high": r["high"],
                "width": r["high"] - r["low"], "holds_truth": bool(r["low"] <= true_rate <= r["high"])}
    ref = iv(est.estimate_error_rate(sample, hand_wrong))
    vlm = iv(est.estimate_error_rate(sample, v_wrong, unjudged=v_unj))
    wid = iv(est.estimate_error_rate(sample, v_wrong, unjudged=v_unj, reviewer_false_alarm=U0, reviewer_miss=U1))
    out = {"experiment": "exp91", "preregistered": True, "population": {"windows": n_pop, "error_rate": true_rate},
           "S": {"n": len(S), "judged": len(judged_s), "unjudged_share": 1 - len(judged_s) / len(S),
                 "class_agreement": agree / max(len(judged_s), 1)},
           "C": {"n0_judged": len(c0), "false_alarms": fa, "E0": fa / max(len(c0), 1), "U0": U0,
                 "n1_judged": len(c1), "misses": miss, "E1": miss / max(len(c1), 1), "U1": U1},
           "intervals": {"hand_labels": ref, "vlm_labels": vlm, "vlm_widened": wid},
           "descriptive_added_after_the_result": {
               "S_map_agreement_on_vlm_judged": sum(int(k["hand"]) == int(k["map"]) for k in judged_s) / max(len(judged_s), 1),
               "S_vlm_agrees_with_map_on_judged": sum(cls[lab[k["id"]]] == int(k["map"]) for k in judged_s) / max(len(judged_s), 1),
               "S_true_errors": sum(k["err"] == "1" for k in S),
               "S_true_errors_judged": sum(k["err"] == "1" and lab[k["id"]] != "?" for k in S),
               "S_true_errors_flagged": sum(k["err"] == "1" and vlm_wrong(k) == 1 for k in S),
               "S_false_flags": sum(k["err"] == "0" and vlm_wrong(k) == 1 for k in S),
               "C_unjudged_share_wrong": sum(k["err"] == "1" and lab[k["id"]] == "?" for k in C) / max(sum(k["err"] == "1" for k in C), 1),
               "C_unjudged_share_correct": sum(k["err"] == "0" and lab[k["id"]] == "?" for k in C) / max(sum(k["err"] == "0" for k in C), 1),
               "S_unjudged_share_wrong": sum(k["err"] == "1" and lab[k["id"]] == "?" for k in S) / max(sum(k["err"] == "1" for k in S), 1),
               "S_unjudged_share_correct": sum(k["err"] == "0" and lab[k["id"]] == "?" for k in S) / max(sum(k["err"] == "0" for k in S), 1),
               "pooled_unjudged_wrong": [sum(k["err"] == "1" and lab[k["id"]] == "?" for k in key), sum(k["err"] == "1" for k in key)],
               "pooled_unjudged_correct": [sum(k["err"] == "0" and lab[k["id"]] == "?" for k in key), sum(k["err"] == "0" for k in key)],
               "S_map_agreement_all": sum(int(k["hand"]) == int(k["map"]) for k in S) / len(S),
               "S_reviewer_right_counting_unjudged_right": sum(lab[k["id"]] == "?" or cls[lab[k["id"]]] == int(k["hand"]) for k in S),
               "S_judged_discordant_reviewer_wrong_map_right": sum(cls[lab[k["id"]]] != int(k["hand"]) and int(k["map"]) == int(k["hand"]) for k in judged_s),
               "S_judged_discordant_reviewer_right_map_wrong": sum(cls[lab[k["id"]]] == int(k["hand"]) and int(k["map"]) != int(k["hand"]) for k in judged_s),
               "S_reviewer_errors_water_where_hand_land": sum(lab[k["id"]] == "water" and k["hand"] == "0" for k in judged_s),
               "S_reviewer_errors": sum(cls[lab[k["id"]]] != int(k["hand"]) for k in judged_s),
               "reviewer": json.load(open(os.path.join(os.path.dirname(labels_path), "exp91_run.json")))},
           "prereg": {"P1": {"holds": agree / max(len(judged_s), 1) >= 0.90},
                      "P2": {"holds": wid["holds_truth"]},
                      "P3": {"holds": miss / max(len(c1), 1) > fa / max(len(c0), 1)}},
           "inputs_sha256": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
                             for p in (labels_path, key_path, os.path.join(os.path.dirname(labels_path), "exp91_run.json"))}}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare"); pr.add_argument("--key", required=True)
    vw = sub.add_parser("view"); vw.add_argument("id"); vw.add_argument("--context", type=int, default=32)
    vw.add_argument("--bands", choices=sorted(BANDS), default="rgb")
    gr = sub.add_parser("grade"); gr.add_argument("labels"); gr.add_argument("--key", required=True)
    gr.add_argument("--out", default=os.path.join(ROOT, "exp", "out", "exp91_summary.json"))
    a = p.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.key)
    elif a.cmd == "view":
        view(a.id, a.context, a.bands)
    else:
        grade(a.labels, a.key, a.out)


if __name__ == "__main__":
    main()
