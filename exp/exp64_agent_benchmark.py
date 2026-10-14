#!/usr/bin/env python
"""exp64: does giving an agent this package as tools make its answers better, or only more fluent?

Why. Every number in this repository answers whether the MEASUREMENT is right: does the margin rank errors, on which
references, by how much. None of them answers whether the TOOL helps. Those are different claims, and only the first is
in the ledger. A person handed this package might review no better than one without it, and an agent given these tools
might answer no more correctly than one given a Python sandbox and the same arrays. This experiment measures the second,
because it can be measured reproducibly; the first needs people.

Preregistered in docs/plan/agent_benchmark.md before any run. That page is the specification and this file implements
it; where the implementation had to choose, the choice is recorded in the summary rather than left in the code.

Design. Forty cards, each a directory the agent can read: `scores.npy` (the model's per-window suspicion and decision),
`meta.json` (task, sensor, dates, the review budget, what a second inference is if there is one), optionally
`second.npy`, and `labels.npy`, which the agent never sees and which grades it. Cards come from testbeds this
repository already holds with expert labels, and nothing in a card is derived from a label except the hidden file.

  Three tasks per card. Name the 5% of windows a reviewer should check, graded by error capture against the labels with
  the package's own capture as the ceiling and a random set at 5% as the floor. Say why each named window is suspect,
  graded on whether the stated cue actually holds for that window. And on cards with a second inference, say how much
  the two differ, where the differences sit, and which side to believe, where the third has one correct answer without
  labels, which is to decline, because exp58 showed the more confident side is right on 51 to 70% of windows and
  confidence does not order the set. Declining scores full marks; picking a side scores the share it gets right, so a
  confident wrong pick is penalised.

  Four arms. A has the package's outputs as tools. B has a sandbox with the same arrays and no package. C is a fixed
  template over the package's outputs with no language model, the groundedness ceiling. D has the card's meta.json and
  no rasters, the floor for task accuracy and the control for prior knowledge of these testbeds.

  Two scores. Task accuracy, per card and pooled. And the claims audit: every number and every window reference in an
  answer is matched against that run's tool outputs, so an unsupported number is counted and a window outside the
  card's grid is a fabrication. That is the contract page turned into a metric.

Preregistered (one-sided, from the plan page, restated here unchanged):
  P1  arm A's claims audit exceeds arm B's by at least 0.2 pooled, and on more cards than not (exact sign test p<0.05).
  P2  arm A's review-set capture exceeds arm B's by at least 0.05 pooled and on more cards than not; arm A reaches at
      least 80% of the package's own capture on the median card.
  P3  on the comparison task arm A declines to pick a side on at least 80% of cards; arm B on fewer than half.
  Falsification. P1 fails if the sandbox agent grounds its numbers as well as the tool agent, in which case the
  contract adds nothing to an agent that can compute. P2 fails if the tool agent does not use the review set it is
  given, or the sandbox agent rediscovers confidence ranking on its own. P3 fails if the tool agent picks sides anyway,
  in which case the contract's "this resolves nothing" is not reaching it, and the safety property claimed for the
  comparison half is not real.

Cards, and why these. Twenty come from GEOID-Flood through exp60 and exp62, which committed per-window decisions,
margins, labels and event ids for 4,489 and 544 chips: those need no encoder and they carry the second inference the
comparison task requires, the other sensor of the same period or the same sensor of the other period. Twenty come from
Sen1Floods11 through exp18's probe on the committed tiles, ten from the Bolivia hand labels and ten from the
multi-region test split, which do need an encoder pass.

Serving. The cluster has a CUDA driver and no toolkit, so vLLM cannot build the JIT attention kernels its Blackwell
path wants and this uses transformers with PyTorch SDPA instead: slower per token and entirely sufficient at this
scale, since the runs are short. The model is named in the summary, never inferred from a default.

Outputs: exp/out/exp64_summary.json, exp/out/exp64_cards.csv, exp/out/exp64_answers.jsonl.
Stages: --stage cards (build them), --stage run (the four arms), --stage grade.
--smoke: synthetic cards, a stub model, no GPU, _smoke outputs.

Model-size ablation, preregistered 2026-09-16 before its run, after the Qwen3.8-27B result was recorded. The 27B
run failed P1 and P2: a numpy sandbox rediscovered the ranking and grounded its numbers nearly as well as the
tool arm, and the reading offered was that the package is a floor whose value over a sandbox shrinks as the model
strengthens. That reading is testable: the same forty cards and five arms with Qwen2.5-7B-Instruct, outputs under
exp/out/exp64_qwen25_7b/, graded by the same code.
  Predicted. At 7B the three preregistered tests hold: P1 the tool arm grounds its claims better than the sandbox
  by at least 0.2 pooled and on more cards than not; P2 it captures at least 0.05 more and reaches 80% of the
  package's capture on the median card; P3 it declines to pick a side on at least 80% of comparison cards and the
  sandbox on fewer than half. The agent arm reproduces the package at 7B as it did at 27B.
  Falsification. If the 7B sandbox also matches the tool arm, the package adds nothing at either size tested and
  the floor reading is wrong. If the tool arm itself fails at 7B (parse failures, fabricated windows), the
  package's value is bound to a model that can follow its contract, which is a limit to state on the front page.
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics, stats  # noqa: E402
from oe_inferencex.signals import boundary_indicator  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
CARDS = "/scratch/qi_zim_neu/olmoearth_inferenceX/exp64_cards"
BUDGET = 0.05
N_PER_SOURCE = 10
SEED = 0


# ----------------------------------------------------------------------------- cards
def _suspicion(margin):
    """Higher = more suspect, the package's own convention."""
    return -np.asarray(margin, dtype=np.float64)


def _card(name, dec, margin, ok, lab, meta, second=None):
    """One card: what the agent may read, and the labels it may not."""
    return {"name": name, "dec": np.asarray(dec), "margin": np.asarray(margin, dtype=np.float32),
            "ok": np.asarray(ok, dtype=bool), "labels": np.asarray(lab), "meta": meta, "second": second}


MIN_ERRORS = 15          # a card with no errors cannot grade a review set; see the note in _qualifying
MAX_ERROR_RATE = 0.30    # above this the card measures a broken model, not whether a review set is well targeted
MIN_VALID = 100


def _qualifying(m, arm, lab, ok):
    """Chips with enough valid windows and enough errors to grade a review set, and their error counts.

    The median GEOID chip has ZERO errors under these heads: 4,482 usable chips and only 777 carry a single one. A card
    with no errors cannot grade a review set at all, and one with two makes capture a coin flip, so cards are drawn only
    from chips with at least MIN_ERRORS. That biases the card set toward the harder chips, which is the right bias for a
    benchmark of finding errors and the wrong one for estimating how common errors are; nothing here estimates the
    latter, and the selection rule and its counts go into the summary so a reader can see the bias rather than infer it.
    Within the qualifying chips one is drawn at RANDOM per event, not the worst, so the set is not the extreme tail."""
    err = ((m[arm].astype(int) != m[lab].astype(int)) & ok).reshape(len(ok), -1).sum(1)
    valid = ok.reshape(len(ok), -1).sum(1)
    rate = err / np.maximum(valid, 1)
    return err, (valid >= MIN_VALID) & (err >= MIN_ERRORS) & (rate <= MAX_ERROR_RATE)


def _one_per_event(qual, ev, rng, exclude):
    """One chip index per event, drawn at random among that event's qualifying chips."""
    by_event = collections.defaultdict(list)
    for i in np.flatnonzero(qual):
        if ev[i] not in exclude:
            by_event[ev[i]].append(i)
    events = sorted(by_event)
    rng.shuffle(events)
    return [(e, int(rng.choice(by_event[e]))) for e in events]


def geoid_cards(rng):
    """Twenty cards from the committed GEOID-Flood artifacts: ten single-inference, ten with a second inference.

    exp60 holds three inferences of identical windows (pre-event optical, pre-event radar, post-event radar) with their
    margins and two labels. The second inference is drawn from the same cell family so the pair is a real comparison
    and not a strawman: the other sensor of the same period, or the same sensor of the other period."""
    m60 = np.load(os.path.join(OUT, "exp60_masks.npz"), allow_pickle=True)
    ok, ev = m60["ok"], m60["event"].astype(str)
    cards, used_events = [], set()

    err_s2, qual_s2 = _qualifying(m60, "A_s2pre", "y_permanent", ok)
    for e, i in _one_per_event(qual_s2, ev, rng, used_events):
        if len(cards) >= N_PER_SOURCE:
            break
        used_events.add(e)
        cards.append(_card(f"geoid_single_{e}", m60["A_s2pre"][i], m60["margin_A_s2pre"][i], ok[i],
                           m60["y_permanent"][i],
                           {"task": "permanent water", "sensor": "Sentinel-2, pre-event composite",
                            "classes": ["not water", "water"], "review_budget": BUDGET,
                            "second_inference": None, "source": "GEOID-Flood via exp60", "event": e,
                            "chip_index": i}))

    # the comparison cards, on events the single cards did not use
    pairs = [("A_s2pre", "A_s1pre", "y_permanent", "the other sensor of the same period"),
             ("A_s1pre", "B_s1post", "y_after", "the same sensor of the other period")]
    for k, (a_, b_, lab, what) in enumerate(pairs):
        err, qual = _qualifying(m60, a_, lab, ok)
        want = N_PER_SOURCE // len(pairs) + (1 if k < N_PER_SOURCE % len(pairs) else 0)
        got = 0
        for e, i in _one_per_event(qual, ev, rng, used_events):
            if got >= want:
                break
            used_events.add(e)
            got += 1
            cards.append(_card(f"geoid_pair_{k}_{e}", m60[a_][i], m60[f"margin_{a_}"][i], ok[i], m60[lab][i],
                               {"task": "water", "sensor": a_, "classes": ["not water", "water"],
                                "review_budget": BUDGET, "second_inference": what,
                                "source": "GEOID-Flood via exp60", "event": e, "chip_index": i},
                               second={"dec": m60[b_][i], "margin": m60[f"margin_{b_}"][i], "name": b_}))
    return cards


def sen1floods_cards(rng):
    """Twenty cards from exp18's cached features: ten Bolivia hand-label tiles, ten multi-region test tiles.

    exp18 committed the encoder pass (exp/out/exp18_feats.npz, 3.9 GB, on the cluster) and its head is a logistic
    regression refit from the cached training features in seconds, so no encoder runs here. The tiles are the ones
    exp18 evaluated, loaded with the same calls and seeds so labels align with the cached features. A binary head's
    top-1 minus top-2 logit is the absolute logit, which is exp18's own confidence reading negated."""
    import torch
    import exp18_sen1floods_expert as e18
    from oe_inferencex.evidence import train_logistic_head

    z = np.load(e18.CACHE)
    _, tr_lab = e18.load_split("valid", e18.N_TRAIN_TILES)
    tr_y, tr_ok = e18.patch_labels(tr_lab[:, :e18.CROP, :e18.CROP])
    D = z["tr_base"].shape[-1]
    sel = tr_ok.flatten()
    torch.manual_seed(0)
    w, b = train_logistic_head(torch.tensor(np.asarray(z["tr_base"], dtype=np.float32)).reshape(-1, D)[sel],
                               tr_y.flatten()[sel])
    cards = []
    for name, (_, lab), src in (
            ("bolivia", e18.load_split("bolivia"), "Sen1Floods11 Bolivia hand labels via exp18"),
            ("test", e18.load_split("test", e18.N_TEST_TILES, seed=1), "Sen1Floods11 multi-region test split via exp18")):
        y, ok = e18.patch_labels(lab[:, :e18.CROP, :e18.CROP])
        p, logit = e18.head_prob_logit(z[f"{name}_base0"], w, b)
        dec = (p > 0.5).astype(np.int64)
        ylab = (y > 0.5).astype(np.int64)
        err, qual = _qualifying({"dec": dec, "y": ylab}, "dec", "y", ok)
        idx = np.flatnonzero(qual)
        rng.shuffle(idx)
        print(f"  sen1floods {name}: {len(lab)} tiles, {int(qual.sum())} qualifying, drawing {N_PER_SOURCE}", flush=True)
        for i in idx[:N_PER_SOURCE]:
            cards.append(_card(f"sen1_{name}_{int(i):04d}", dec[i], np.abs(logit[i]), ok[i], ylab[i],
                               {"task": "water", "sensor": "Sentinel-2, exp18 tiles, OlmoEarth v1 Base features, "
                                                           "shift 0, exp18's logistic head",
                                "classes": ["not water", "water"], "review_budget": BUDGET,
                                "second_inference": None, "source": src, "tile_index": int(i)}))
    return cards


def write_cards(cards, root):
    os.makedirs(root, exist_ok=True)
    rows = []
    for c in cards:
        d = os.path.join(root, c["name"])
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "scores.npy"), np.stack([c["margin"], c["dec"].astype(np.float32)]))
        np.save(os.path.join(d, "valid.npy"), c["ok"])
        np.save(os.path.join(d, "labels.npy"), c["labels"])      # the agent never reads this
        if c["second"] is not None:
            np.save(os.path.join(d, "second.npy"),
                    np.stack([c["second"]["margin"], np.asarray(c["second"]["dec"], dtype=np.float32)]))
        meta = dict(c["meta"])
        meta["grid"] = list(np.asarray(c["dec"]).shape)
        meta["n_valid_windows"] = int(c["ok"].sum())
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        err = (np.asarray(c["dec"]).astype(int) != np.asarray(c["labels"]).astype(int)) & c["ok"]
        n_valid, n_err = int(c["ok"].sum()), int(err.sum())
        # A 5% budget on this grid buys k slots, so no ranker can capture more than k / n_errors. Reporting the raw
        # capture without that ceiling makes an easy card look like a good ranker; both arms face the same ceiling on
        # the same card, but the record should carry it.
        k = max(1, int(round(BUDGET * n_valid)))
        ceiling = min(1.0, k / max(n_err, 1))
        cap = package_capture(c)
        rows.append({"card": c["name"], "source": meta["source"], "grid": "x".join(map(str, meta["grid"])),
                     "n_valid": n_valid, "n_errors": n_err, "review_slots_at_budget": k,
                     "error_rate": float(n_err / max(n_valid, 1)),
                     "has_second": c["second"] is not None,
                     "attainable_capture_ceiling": float(ceiling),
                     "package_capture_at_budget": cap,
                     "package_share_of_ceiling": float(cap / ceiling) if np.isfinite(cap) and ceiling > 0 else float("nan")})
    return rows


def package_capture(c):
    """The ceiling for task 1: what the package's own review set captures on this card."""
    ok = c["ok"].ravel()
    if ok.sum() == 0:
        return float("nan")
    sus = _suspicion(c["margin"]).ravel()[ok]
    err = ((np.asarray(c["dec"]).astype(int) != np.asarray(c["labels"]).astype(int)).ravel()[ok]).astype(float)
    if err.sum() == 0:
        return float("nan")
    return float(metrics.capture_at_budget_expected(sus, err, (BUDGET,))[BUDGET])


def cmd_cards(args):
    rng = np.random.default_rng(SEED)
    cards = []
    if "geoid" in args.sources:
        cards += geoid_cards(rng)
    if "sen1" in args.sources:
        cards += sen1floods_cards(rng)
    root = os.path.join(CARDS, "smoke" if args.smoke else "v1")
    rows = write_cards(cards, root)
    tag = "_smoke" if args.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"exp64_cards{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    gradeable = [r for r in rows if np.isfinite(r["package_capture_at_budget"])]
    print(f"wrote {len(rows)} cards to {root}")
    print(f"  with a second inference : {sum(r['has_second'] for r in rows)}")
    print(f"  gradeable (have errors) : {len(gradeable)}")
    print(f"  median error rate       : {np.median([r['error_rate'] for r in rows]):.4f}")
    print(f"  median review slots     : {int(np.median([r['review_slots_at_budget'] for r in rows]))}")
    print(f"  median package capture  : {np.median([r['package_capture_at_budget'] for r in gradeable]):.4f}"
          f"  (a random {BUDGET:.0%} captures {BUDGET:.2f})")
    print(f"  median share of ceiling : {np.median([r['package_share_of_ceiling'] for r in gradeable]):.4f}"
          f"  (1.0 = the best any ranker could do at this budget)")
    return rows


def smoke(args):
    """Synthetic cards end to end: the selection rule, the ceiling arithmetic, and what a card exposes."""
    import tempfile
    rng = np.random.default_rng(0)
    n, g = 40, 14
    ok = np.ones((n, g, g), bool)
    lab = (rng.random((n, g, g)) < 0.3)
    dec = lab.copy()
    for i in range(n):                                   # plant a controlled number of errors per chip
        flip = rng.choice(g * g, size=i, replace=False)
        dec[i].ravel()[flip] = ~dec[i].ravel()[flip]
    m = {"ok": ok, "A_s2pre": dec, "y_permanent": lab,
         "margin_A_s2pre": rng.random((n, g, g)).astype(np.float32)}
    err, qual = _qualifying(m, "A_s2pre", "y_permanent", ok)
    assert not qual[:MIN_ERRORS].any(), "chips below the error floor must not qualify"
    rate = err / (g * g)
    assert not qual[rate > MAX_ERROR_RATE].any(), "chips above the error ceiling must not qualify"
    assert qual[(err >= MIN_ERRORS) & (rate <= MAX_ERROR_RATE)].all()

    ev = np.array([f"E{i // 3}" for i in range(n)])
    picked = _one_per_event(qual, ev, np.random.default_rng(1), exclude={"E0"})
    assert all(e != "E0" for e, _ in picked), "an excluded event must not be drawn"
    assert len({e for e, _ in picked}) == len(picked), "one chip per event"
    assert all(qual[i] for _, i in picked), "only qualifying chips may be drawn"

    # the ceiling: with k slots and E errors no ranker can capture more than k/E
    c = _card("t", dec[20], m["margin_A_s2pre"][20], ok[20], lab[20], {"source": "smoke", "review_budget": BUDGET})
    root = tempfile.mkdtemp()
    rows = write_cards([c], root)
    r = rows[0]
    assert r["review_slots_at_budget"] == max(1, round(BUDGET * r["n_valid"]))
    assert abs(r["attainable_capture_ceiling"] - min(1.0, r["review_slots_at_budget"] / r["n_errors"])) < 1e-12
    assert r["package_capture_at_budget"] <= r["attainable_capture_ceiling"] + 1e-12, "capture cannot beat the ceiling"
    assert sorted(os.listdir(os.path.join(root, "t"))) == ["labels.npy", "meta.json", "scores.npy", "valid.npy"]
    print("smoke OK: error floor and ceiling, one chip per event, exclusions, capture bounded by the attainable ceiling")
    _smoke_pipeline()


def _smoke_pipeline():
    """run and grade end to end against a STUB model, so the harness is checked without a GPU.

    The stub is written to make arm A ground itself in its tools and arm B state a number no tool produced. That is
    a test of the grading, NOT a preview of the result: the arms behave that way here because this function makes
    them, and what a real served model does is the experiment's open question."""
    import shutil
    import tempfile
    import types

    import exp64_arms as arms

    root = tempfile.mkdtemp()
    cards_root = os.path.join(root, "smoke")
    rng = np.random.default_rng(0)
    g, made = 16, []
    for i in range(6):
        lab = (rng.random((g, g)) < 0.35).astype(int)
        dec = lab.copy()
        ei = rng.choice(g * g, size=30, replace=False)
        dec.ravel()[ei] = 1 - dec.ravel()[ei]
        margin = rng.random((g, g)).astype(np.float32) * 0.5 + 0.5
        margin.ravel()[ei] -= 0.45
        sec = dec.copy()
        fl = rng.choice(g * g, size=20, replace=False)
        sec.ravel()[fl] = 1 - sec.ravel()[fl]
        made.append(_card(f"c{i}", dec, margin, np.ones((g, g), bool), lab,
                          {"source": "smoke", "review_budget": BUDGET, "task": "water", "sensor": "S1"},
                          second={"dec": sec, "margin": margin * 0.9, "name": "s"}))
    write_cards(made, cards_root)

    def stub(endpoint, model, messages, tools=None, temperature=0.0, seed=0, timeout=300):
        txt = json.dumps(messages)
        if tools and '"role": "tool"' not in txt:
            name = tools[0]["function"]["name"]
            call = {"id": "1", "function": {"name": name, "arguments": "{}" if name != "python" else
                    json.dumps({"code": "print(float(margin.mean()))"})}}
            extra = [{"id": "2", "function": {"name": "compare", "arguments": "{}"}}] if name == "assess" else []
            return {"role": "assistant", "content": "", "tool_calls": [call] + extra}
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if tools and tools[0]["function"]["name"] == "assess":
            a, cmp_ = json.loads(tool_msgs[0]["content"]), json.loads(tool_msgs[-1]["content"])
            ans = {"review_windows": a["review_windows"],
                   "explanations": [{"window": w["window"], "cue": "boundary" if w["boundary"] else "low_margin",
                                     "value": w["margin"]} for w in a["window_cues"]],
                   "comparison": {"n_differing": cmp_["n_differing"], "where": "boundaries",
                                  "believe": "decline"}}
            return {"role": "assistant", "content": json.dumps(ans)}
        if tools:
            ans = {"review_windows": [[i // g, i % g] for i in range(12)],
                   "explanations": [{"window": [0, 0], "cue": "low_margin", "value": 0.4242}],
                   "comparison": {"n_differing": 999, "where": "everywhere", "believe": "first"}}
            return {"role": "assistant", "content": json.dumps(ans) + " I measured 0.8137 of windows."}
        ans = {"review_windows": [[0, 0], [1, 1]],
               "explanations": [{"window": [0, 0], "cue": "other", "value": None}],
               "comparison": {"n_differing": 50, "where": "unknown", "believe": "second"}}
        return {"role": "assistant", "content": json.dumps(ans)}

    real_chat, real_cards, real_out = arms.chat, CARDS, OUT
    arms.chat = stub
    globals()["CARDS"], globals()["OUT"] = root, tempfile.mkdtemp()
    try:
        a = types.SimpleNamespace(smoke=True, endpoint="stub", model="stub-model", arms="ABCD",
                                  samples=2, temperature=0.0, seed=0, cards=0, card_prefix="", answers_suffix="",
                                  sources="", out_dir="")
        cmd_run(a)
        # a second answer file, as the agent driver writes one: grade must see the extra arm without being told
        with open(os.path.join(OUT, "exp64_answers_smoke.jsonl")) as fh:
            one = next(json.loads(ln) for ln in fh if '"arm": "A"' in ln)
        one["arm"] = "E"
        with open(os.path.join(OUT, "exp64_answers_smoke_E.jsonl"), "w") as fh:
            fh.write(json.dumps(one, default=float) + "\n")
        s = cmd_grade(a)
        assert "E" in s["claims_audit"] and s["exploratory"]["E"]["note"].startswith("not preregistered")
    finally:
        arms.chat = real_chat
        globals()["CARDS"], globals()["OUT"] = real_cards, real_out
        shutil.rmtree(root, ignore_errors=True)

    assert s["claims_audit"]["C"]["pooled"] == 1.0, "arm C states only what its tools produced"
    assert s["claims_audit"]["A"]["pooled"] > s["claims_audit"]["B"]["pooled"], "the audit must separate them"
    assert abs(s["share_of_package"]["A"]["pooled"] - 1.0) < 1e-9, "an arm copying the review set reproduces it"
    assert s["review_capture"]["D"]["pooled"] < s["review_capture"]["A"]["pooled"], "the no-raster floor is a floor"
    assert s["verdicts"]["P3_declines"]["declined_A"] == 1.0 and s["verdicts"]["P3_declines"]["declined_B"] == 0.0
    assert {k: v for k, v in s["parse_failures"].items() if k in ARMS} == {"A": 0, "B": 0, "C": 0, "D": 0}, \
        s["parse_failures"]
    assert _sign_p(6, 6) < 0.05 and _sign_p(3, 6) > 0.05, "the sign test must be one-sided and exact"
    print("smoke OK: run and grade end to end on a stub model, the audit separates a grounded arm from an "
          "ungrounded one, and the three verdicts compute")


ARMS = ("A", "B", "C", "D")                      # the preregistered arms cmd_run drives


def _out(args):
    """The output directory of this run; a second model writes beside the first, never over it."""
    d = getattr(args, "out_dir", "") or OUT
    os.makedirs(d, exist_ok=True)
    return d
ARMS_ALL = ("A", "B", "C", "D", "E", "E_forced")  # plus the OlmoEarth Agent, run by exp64_arm_e.py


def cmd_run(args):
    """Every arm over every card, `--samples` times, appended to exp64_answers.jsonl as they finish.

    Written as it goes rather than at the end: a thousand short runs against a served model is long enough that a
    preempted job should cost the runs it had not reached, not the ones it had."""
    import exp64_arms as arms
    root = os.path.join(CARDS, "smoke" if args.smoke else "v1")
    dirs = sorted(d for d in (os.path.join(root, x) for x in os.listdir(root)) if os.path.isdir(d))
    if args.card_prefix:
        dirs = [d for d in dirs if os.path.basename(d).startswith(args.card_prefix)]
    if args.cards:
        dirs = dirs[:args.cards]
    tag = "_smoke" if args.smoke else ""
    # A second batch of cards (the Sen1Floods11 half, built after the first run started) writes its own file,
    # which the grade stage already gathers, rather than overwriting the first batch's answers.
    suffix = f"_{args.answers_suffix}" if args.answers_suffix else ""
    path = os.path.join(_out(args), f"exp64_answers{tag}{suffix}.jsonl")
    want = [a for a in ARMS if a in set(args.arms)]
    n = 0
    with open(path, "w") as fh:
        for d in dirs:
            card = arms.load_card(d)
            for arm in want:
                # Arm C has no model and no sampling temperature, so one run of it is all there is.
                for sample in range(1 if arm == "C" else args.samples):
                    if arm == "C":
                        ans, text, tools = arms.arm_c_answer(card)
                        run = {"arm": "C", "answer": ans, "parse_error": None, "text": text,
                               "tool_outputs": tools, "n_tool_calls": len(tools), "n_steps": 0}
                    else:
                        # One arm failing on one card must not end a run of a thousand: the failure is recorded
                        # as that run's result, which is what it is, and the remaining arms still execute.
                        try:
                            run = arms.run_llm_arm(card, arm, args.endpoint, args.model,
                                                   seed=args.seed + sample, temperature=args.temperature)
                        except Exception as exc:  # noqa: BLE001
                            run = {"arm": arm, "answer": None, "text": "",
                                   "parse_error": f"harness error: {type(exc).__name__}: {exc}",
                                   "tool_outputs": {}, "n_tool_calls": 0, "n_steps": 0}
                            print(f"    {card['name']} arm {arm} sample {sample}: "
                                  f"{type(exc).__name__}: {exc}", flush=True)
                    fh.write(json.dumps({"card": card["name"], "sample": sample, **run}, default=float) + "\n")
                    fh.flush()
                    n += 1
            print(f"  {card['name']:<34} {len(want)} arms done", flush=True)
    print(f"wrote {n} runs to {path}")
    return path


def cmd_grade(args):
    """Grade every recorded run and decide the three preregistered predictions."""
    import exp64_arms as arms
    tag = "_smoke" if args.smoke else ""
    root = os.path.join(CARDS, "smoke" if args.smoke else "v1")
    # The four preregistered arms land in exp64_answers.jsonl; other drivers append their own file beside it
    # (exp64_answers_E.jsonl for the OlmoEarth Agent), so one grade pass sees every arm.
    import glob
    out = _out(args)
    files = sorted(set([os.path.join(out, f"exp64_answers{tag}.jsonl")]
                       + glob.glob(os.path.join(out, f"exp64_answers{tag}_*.jsonl"))))
    runs = [json.loads(ln) for f in files if os.path.exists(f) for ln in open(f) if ln.strip()]
    present = [a for a in ARMS_ALL if any(r["arm"] == a for r in runs)]
    cards = {}
    graded = []
    for r in runs:
        if r["card"] not in cards:
            cards[r["card"]] = arms.load_card(os.path.join(root, r["card"]))
        g = arms.grade_run(r, cards[r["card"]])
        g["sample"] = r.get("sample", 0)
        graded.append(g)

    def by_arm(fn):
        """Mean over cards of the per-card mean over samples, so a card counts once however often it was sampled."""
        out = {}
        for arm in present:
            per_card = collections.defaultdict(list)
            for g in graded:
                if g["arm"] != arm:
                    continue
                v = fn(g)
                if v is not None and np.isfinite(v):
                    per_card[g["card"]].append(v)
            means = {c: float(np.mean(v)) for c, v in per_card.items() if v}
            out[arm] = {"pooled": float(np.mean(list(means.values()))) if means else float("nan"),
                        "median": float(np.median(list(means.values()))) if means else float("nan"),
                        "n_cards": len(means), "per_card": means}
        return out

    claims = by_arm(lambda g: g["claims"]["supported_share"])
    capture = by_arm(lambda g: g["review_set"]["capture"] if g["review_set"].get("gradeable") else None)
    share_pkg = by_arm(lambda g: g["review_set"]["share_of_package"] if g["review_set"].get("gradeable") else None)
    cue_acc = by_arm(lambda g: g["explanation"]["accuracy"])
    declined = by_arm(lambda g: float(g["comparison"]["declined"]) if g["comparison"].get("gradeable") else None)
    unanswered = by_arm(lambda g: float(not g["comparison"].get("answered", True))
                        if g["comparison"].get("gradeable") else None)

    def paired(metric, a="A", b="B"):
        """Wins, losses and an exact sign test on the cards both arms were graded on."""
        shared = sorted(set(metric[a]["per_card"]) & set(metric[b]["per_card"]))
        wins = sum(metric[a]["per_card"][c] > metric[b]["per_card"][c] for c in shared)
        losses = sum(metric[a]["per_card"][c] < metric[b]["per_card"][c] for c in shared)
        return {"n_cards": len(shared), "wins": wins, "losses": losses,
                "p_value": _sign_p(wins, wins + losses)}

    cl_p, cap_p = paired(claims), paired(capture)
    p1 = {"holds": bool(claims["A"]["pooled"] - claims["B"]["pooled"] >= 0.2
                        and cl_p["wins"] > cl_p["losses"] and cl_p["p_value"] < 0.05),
          "pooled_A": claims["A"]["pooled"], "pooled_B": claims["B"]["pooled"],
          "difference": claims["A"]["pooled"] - claims["B"]["pooled"], **cl_p}
    p2 = {"holds": bool(capture["A"]["pooled"] - capture["B"]["pooled"] >= 0.05
                        and cap_p["wins"] > cap_p["losses"]
                        and share_pkg["A"]["median"] >= 0.8),
          "pooled_A": capture["A"]["pooled"], "pooled_B": capture["B"]["pooled"],
          "difference": capture["A"]["pooled"] - capture["B"]["pooled"],
          "median_share_of_package_A": share_pkg["A"]["median"], **cap_p}
    p3 = {"holds": bool(declined["A"]["pooled"] >= 0.8 and declined["B"]["pooled"] < 0.5),
          "declined_A": declined["A"]["pooled"], "declined_B": declined["B"]["pooled"],
          "n_cards_A": declined["A"]["n_cards"], "n_cards_B": declined["B"]["n_cards"]}

    parse_fail = {arm: sum(1 for g in graded if g["arm"] == arm and g["parse_error"]) for arm in present}
    fabricated = {arm: int(sum(g["claims"]["n_fabricated_windows"] for g in graded if g["arm"] == arm))
                  for arm in present}
    # The OlmoEarth Agent arms were left open on the plan page, so nothing about them was preregistered:
    # they are reported against arm A descriptively, and the summary says so.
    exploratory = {}
    for x in ("E", "E_forced"):
        if x in present and "A" in present:
            exploratory[x] = {"note": "not preregistered; descriptive only",
                              "claims_vs_A": paired(claims, x, "A"), "capture_vs_A": paired(capture, x, "A"),
                              "pooled_claims": claims[x]["pooled"], "pooled_capture": capture[x]["pooled"],
                              "median_share_of_package": share_pkg[x]["median"],
                              "declined": declined[x]["pooled"], "comparison_unanswered": unanswered[x]["pooled"],
                              "parse_failures": parse_fail[x],
                              "fabricated_windows": fabricated[x]}
    summary = {"experiment": "exp64 agent benchmark", "model": args.model, "endpoint_named": bool(args.endpoint),
               "samples_per_card": args.samples, "temperature": args.temperature, "budget": BUDGET,
               "n_runs": len(graded), "n_cards": len(cards),
               "claims_audit": claims, "review_capture": capture, "share_of_package": share_pkg,
               "cue_accuracy": cue_acc, "declined_share": declined, "comparison_unanswered_share": unanswered,
               "parse_failures": parse_fail, "fabricated_windows": fabricated,
               "answer_files": [os.path.basename(f) for f in files if os.path.exists(f)],
               "verdicts": {"P1_claims_audit": p1, "P2_review_capture": p2, "P3_declines": p3},
               "exploratory": exploratory}
    with open(os.path.join(out, f"exp64_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    print(f"\n{'arm':<9} {'claims':>8} {'capture':>8} {'of pkg':>8} {'cue acc':>8} {'declined':>9} "
          f"{'parse!':>7} {'fabric':>7}")
    for arm in present:
        print(f"{arm:<9} {claims[arm]['pooled']:>8.3f} {capture[arm]['pooled']:>8.3f} "
              f"{share_pkg[arm]['pooled']:>8.3f} {cue_acc[arm]['pooled']:>8.3f} "
              f"{declined[arm]['pooled']:>9.3f} {parse_fail[arm]:>7d} {fabricated[arm]:>7d}")
    print(f"\nP1 claims audit A - B >= 0.2 and A wins more cards : {p1['holds']}  "
          f"({p1['difference']:+.3f}, {p1['wins']}/{p1['losses']}, p={p1['p_value']:.4g})")
    print(f"P2 capture A - B >= 0.05 and >= 0.8 of the package  : {p2['holds']}  "
          f"({p2['difference']:+.3f}, {p2['wins']}/{p2['losses']}, median share {p2['median_share_of_package_A']:.3f})")
    print(f"P3 A declines >= 80%, B on fewer than half          : {p3['holds']}  "
          f"(A {p3['declined_A']:.3f}, B {p3['declined_B']:.3f})")
    return summary


def _sign_p(wins, decisive):
    """One-sided exact sign test; no scipy, so the binomial tail is summed directly."""
    if decisive == 0:
        return 1.0
    total = 0.0
    for k in range(wins, decisive + 1):
        c = 1.0
        for i in range(k):
            c = c * (decisive - i) / (i + 1)
        total += c
    return min(1.0, total * 0.5 ** decisive)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("cards", "run", "grade", "all"), default="cards")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--endpoint", default=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--arms", default="ABCD", help="which arms to run")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cards", type=int, default=0, help="cap the number of cards (0 = all)")
    ap.add_argument("--sources", default="geoid,sen1", help="card sources to build: geoid, sen1")
    ap.add_argument("--card-prefix", default="", help="run only cards whose name starts with this")
    ap.add_argument("--answers-suffix", default="", help="write exp64_answers_<suffix>.jsonl instead")
    ap.add_argument("--out-dir", default="", help="output directory for run and grade (default exp/out)")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    if args.stage in ("cards", "all"):
        cmd_cards(args)
    if args.stage in ("run", "all"):
        cmd_run(args)
    if args.stage in ("grade", "all"):
        cmd_grade(args)


if __name__ == "__main__":
    main()
