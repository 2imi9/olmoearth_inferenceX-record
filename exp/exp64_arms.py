"""exp64's four arms, their tools, and the grading. Imported by exp64_agent_benchmark.py.

Split out of the experiment file so the card builder stays readable. Everything here is numpy and the standard
library except the HTTP call to the served model, which is the only part that needs the cluster.

The answer contract. Grading free prose is not reproducible, so every arm emits ONE json object and the prompt says
so. A model that emits something else is not silently rescued: the parse failure is recorded per card and counted,
because "the agent could not produce a gradeable answer" is a result about the arm and not a bug in the harness.

  {"review_windows": [[r, c], ...],
   "explanations":   [{"window": [r, c], "cue": "<name>", "value": <number or null>}, ...],
   "comparison":     {"n_differing": <int>, "where": "<short phrase>", "believe": "decline"|"first"|"second"}}

The claims audit tolerance. exp72 measured the two implementations of these statistics against each other on real
probe output and found the review set bit-identical but the summary statistics apart by up to 2.7e-05, because the
probe emits float32 and a pure-Python tool works in double. So a number in an answer is matched against the tool
outputs with a tolerance, never by string equality: an exact-match audit would score a correct agent as unsupported
for reporting the same quantity at a different precision.
"""
import json
import math
import os
import re

import numpy as np

from oe_inferencex import metrics

BUDGET = 0.05
CUES = ("boundary", "low_margin", "high_entropy", "isolated", "other")
#: A number in an answer counts as supported if it matches some tool output to within this RELATIVE tolerance,
#: or this ABSOLUTE one for values near zero. See the module docstring: exp72 is why this is not equality.
REL_TOL, ABS_TOL = 1e-3, 1e-4


# --------------------------------------------------------------------------------------------- card access
def load_card(d):
    """Everything a card holds, labels included; the arms are handed only what their arm allows."""
    scores = np.load(os.path.join(d, "scores.npy"))
    card = {"dir": d, "name": os.path.basename(d), "margin": scores[0], "dec": scores[1].astype(int),
            "ok": np.load(os.path.join(d, "valid.npy")).astype(bool),
            "labels": np.load(os.path.join(d, "labels.npy")).astype(int),
            "meta": json.load(open(os.path.join(d, "meta.json")))}
    p2 = os.path.join(d, "second.npy")
    if os.path.exists(p2):
        s2 = np.load(p2)
        card["second"] = {"margin": s2[0], "dec": s2[1].astype(int)}
    else:
        card["second"] = None
    card["errors"] = ((card["dec"] != card["labels"]) & card["ok"])
    return card


def n_slots(card):
    """Review slots the budget buys on this card."""
    return max(1, int(round(BUDGET * int(card["ok"].sum()))))


# --------------------------------------------------------------------------------------------- cues
def boundary_map(dec):
    """Windows whose 8-neighbourhood contains a different decision: the package's boundary cue."""
    d = np.asarray(dec)
    out = np.zeros(d.shape, bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(d, dr, 0), dc, 1)
            inside = np.ones(d.shape, bool)
            if dr == 1:
                inside[0, :] = False
            elif dr == -1:
                inside[-1, :] = False
            if dc == 1:
                inside[:, 0] = False
            elif dc == -1:
                inside[:, -1] = False
            out |= (shifted != d) & inside
    return out


def cue_table(card):
    """Per-window cue values the tools report and the explanation task is graded against."""
    return {"boundary": boundary_map(card["dec"]),
            "low_margin": card["margin"] <= np.quantile(card["margin"][card["ok"]], 0.25),
            "high_entropy": card["margin"] <= np.quantile(card["margin"][card["ok"]], 0.10),
            "isolated": ~boundary_map(card["dec"]) & (card["margin"] <=
                                                      np.quantile(card["margin"][card["ok"]], 0.25))}


# --------------------------------------------------------------------------------------------- the tools (arm A)
def tool_assess(card):
    """What `oe-inferencex assess` reports for this card: the review set at the budget and its cues."""
    ok = card["ok"]
    sus = (-card["margin"]).astype(np.float64)
    flat = np.flatnonzero(ok.ravel())
    order = flat[np.argsort(-sus.ravel()[flat], kind="stable")]
    k = n_slots(card)
    picked = [[int(i // ok.shape[1]), int(i % ok.shape[1])] for i in order[:k]]
    cues = cue_table(card)
    bmap = cues["boundary"]
    return {"review_budget": BUDGET, "n_valid_windows": int(ok.sum()), "review_slots": k,
            "review_windows": picked,
            "window_cues": [{"window": w, "suspicion": round(float(sus[w[0], w[1]]), 6),
                             "margin": round(float(card["margin"][w[0], w[1]]), 6),
                             "boundary": bool(bmap[w[0], w[1]])} for w in picked],
            "n_boundary_in_review_set": int(sum(bool(bmap[r, c]) for r, c in picked)),
            "boundary_share_all": round(float(bmap[ok].mean()), 6),
            "margin_min": round(float(card["margin"][ok].min()), 6),
            "margin_median": round(float(np.median(card["margin"][ok])), 6),
            "grid": list(card["dec"].shape)}


def tool_compare(card):
    """What `oe-inferencex compare` reports: how much two inferences of the same windows differ, and where."""
    if card["second"] is None:
        return {"comparable": False, "reason": "this card has one inference"}
    ok = card["ok"]
    diff = (card["dec"] != card["second"]["dec"]) & ok
    bmap = boundary_map(card["dec"])
    n = int(diff.sum())
    return {"comparable": True, "n_differing": n,
            "share_differing": round(float(n / max(int(ok.sum()), 1)), 6),
            "n_valid_windows": int(ok.sum()),
            "boundary_share_of_differing": round(float(bmap[diff].mean()), 6) if n else 0.0,
            "boundary_share_overall": round(float(bmap[ok].mean()), 6),
            "differing_windows": [[int(r), int(c)] for r, c in np.argwhere(diff)][:200],
            "which_side_is_right": "not resolvable without labels",
            "note": "exp58 measured that the more confident side is right on 51 to 70 percent of differing windows "
                    "and that confidence does not order the set, so this comparison does not identify a winner; "
                    "declining to pick a side is the correct answer here."}


def tool_python(code, card):
    """Arm B's sandbox: numpy and the same arrays, no package. Returns whatever the snippet prints."""
    import contextlib
    import io
    env = {"np": np, "margin": card["margin"].copy(), "decision": card["dec"].copy(),
           "valid": card["ok"].copy(), "budget": BUDGET, "grid": card["dec"].shape}
    if card["second"] is not None:
        env["second_margin"] = card["second"]["margin"].copy()
        env["second_decision"] = card["second"]["dec"].copy()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, env)  # noqa: S102 - the arm IS a sandbox, by design
    except Exception as exc:  # noqa: BLE001 - handed back to the model so it can recover
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "stdout": buf.getvalue()[:4000]}
    return {"ok": True, "stdout": buf.getvalue()[:4000]}


# --------------------------------------------------------------------------------------------- arm C: no model
def arm_c_answer(card):
    """The package's outputs narrated by a fixed template. No language model, so the claims audit is 1 by
    construction: every number it states is copied from a tool output rather than generated."""
    a = tool_assess(card)
    cmp_ = tool_compare(card)
    ans = {"review_windows": a["review_windows"],
           "explanations": [{"window": w["window"],
                             "cue": "boundary" if w["boundary"] else "low_margin",
                             "value": w["margin"]} for w in a["window_cues"]]}
    if cmp_["comparable"]:
        ans["comparison"] = {"n_differing": cmp_["n_differing"],
                             "where": ("mostly on class boundaries"
                                       if cmp_["boundary_share_of_differing"] > cmp_["boundary_share_overall"]
                                       else "spread across the scene"),
                             "believe": "decline"}
    text = ("Review these {n} windows, ordered by the model's own suspicion. {b} of them sit on a class boundary. "
            "The scene's median margin is {m}.").format(n=len(a["review_windows"]),
                                                        b=a["n_boundary_in_review_set"], m=a["margin_median"])
    if cmp_["comparable"]:
        text += (" The two inferences differ on {d} windows; which side is right is not resolvable "
                 "without labels.".format(d=cmp_["n_differing"]))
    return ans, text, {"assess": a, "compare": cmp_}


# --------------------------------------------------------------------------------------------- grading
def grade_review_set(answer, card):
    """Error capture of the agent's named windows, with the package's capture as the ceiling."""
    ok, err = card["ok"], card["errors"]
    total = int(err.sum())
    if total == 0:
        return {"gradeable": False, "reason": "card has no errors"}
    k = n_slots(card)
    picked, off_grid, invalid, dup = [], 0, 0, 0
    seen = set()
    R, C = card["dec"].shape
    for w in (answer.get("review_windows") or [])[:k]:
        try:
            r, c = int(w[0]), int(w[1])
        except (TypeError, ValueError, IndexError):
            off_grid += 1
            continue
        if not (0 <= r < R and 0 <= c < C):
            off_grid += 1
            continue
        if not ok[r, c]:
            invalid += 1
            continue
        if (r, c) in seen:
            dup += 1
            continue
        seen.add((r, c))
        picked.append((r, c))
    caught = int(sum(bool(err[r, c]) for r, c in picked))
    sus = (-card["margin"]).astype(np.float64).ravel()[ok.ravel()]
    e = err.ravel()[ok.ravel()].astype(float)
    pkg = float(metrics.capture_at_budget_expected(sus, e, (BUDGET,))[BUDGET])
    ceiling = min(1.0, k / total)
    return {"gradeable": True, "n_named": len(picked), "n_slots": k, "capture": caught / total,
            "package_capture": pkg, "attainable_ceiling": ceiling,
            "share_of_package": (caught / total) / pkg if pkg > 0 else float("nan"),
            "random_capture": BUDGET, "off_grid": off_grid, "invalid_window": invalid, "duplicate": dup}


def grade_explanation(answer, card):
    """Does the stated cue actually hold for the window it is stated about?"""
    cues = cue_table(card)
    R, C = card["dec"].shape
    n, right, unknown = 0, 0, 0
    for e in (answer.get("explanations") or []):
        w = e.get("window")
        try:
            r, c = int(w[0]), int(w[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if not (0 <= r < R and 0 <= c < C):
            continue
        cue = str(e.get("cue", "")).strip().lower()
        n += 1
        if cue in cues:
            right += bool(cues[cue][r, c])
        elif cue == "other":
            unknown += 1
        else:
            unknown += 1
    return {"n_explained": n, "n_cue_holds": right, "n_uncheckable_cue": unknown,
            "accuracy": right / n if n else float("nan")}


def grade_comparison(answer, card, tol=0.25):
    """Difference count within tolerance, and the side question, where declining is the one correct answer."""
    truth = tool_compare(card)
    if not truth["comparable"]:
        return {"gradeable": False}
    if not isinstance(answer.get("comparison"), dict):
        # The question was asked and not answered: scored as not declining, and marked apart from a pick so
        # the summary can say how often an arm left the question alone rather than chose a side.
        return {"gradeable": True, "answered": False, "true_n_differing": truth["n_differing"],
                "said_n_differing": None, "count_within_tolerance": False, "believe": "",
                "believe_raw": "", "declined": False, "side_score": 0.0}
    c = answer["comparison"]
    said = c.get("n_differing")
    try:
        count_ok = abs(float(said) - truth["n_differing"]) <= tol * max(truth["n_differing"], 1)
    except (TypeError, ValueError):
        count_ok = False
    believe = str(c.get("believe", "")).strip().lower()
    picks = {"first": "first", "a": "first", "second": "second", "b": "second"}
    # The pilot's tool arm answered "not resolvable without labels", the tool's own phrase, and was scored as
    # a wrong pick. A pick must be one of the exact words; anything else that says it will not choose is the
    # correct answer the contract asks for, and the raw string is kept so a reader can check the reading.
    declines = ("decline", "neither", "none", "cannot", "can't", "unresolv", "not resolv", "without label",
                "no way to", "insufficient", "unknown", "not possible")
    if believe in picks:
        believe = picks[believe]
        diff = (card["dec"] != card["second"]["dec"]) & card["ok"]
        if diff.sum() == 0:
            side_score, declined = float("nan"), False
        else:
            first_right = (card["dec"][diff] == card["labels"][diff]).mean()
            side_score = float(first_right if believe == "first" else 1.0 - first_right)
            declined = False
    elif believe and any(w in believe for w in declines):
        side_score, declined = 1.0, True
    else:
        side_score, declined = 0.0, False
    return {"gradeable": True, "answered": True, "true_n_differing": truth["n_differing"],
            "said_n_differing": said, "count_within_tolerance": bool(count_ok), "believe": believe,
            "believe_raw": str(c.get("believe", "")), "declined": declined, "side_score": side_score}



# --------------------------------------------------------------------------------------------- claims audit
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
_WIN = re.compile(r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]")


_NUM_IN_TEXT = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _values_in(obj, out):
    """Every numeric value anywhere in a tool output, so a stated number can be matched against it.

    Numbers inside strings count too. The sandbox arm's only tool output is what it printed, which is text,
    and the first grading of the 27B run scored that arm at zero grounding while 97% of the values it stated
    were in its own printed output: the audit had read JSON fields and skipped strings, which is a bias toward
    the arm whose tools happen to answer in JSON. A number the run's tools produced is evidence whatever the
    container."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)) and math.isfinite(obj):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in _NUM_IN_TEXT.finditer(obj):
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            if math.isfinite(v):
                out.append(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _values_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _values_in(v, out)


def claims_audit(text, tool_outputs, grid):
    """Share of stated numbers that appear in this run's tool outputs, and window references off the grid.

    A number matches if some tool output is within REL_TOL relatively or ABS_TOL absolutely; percentages are also
    tried as their fraction, since an agent may write 12% for 0.12. Small integers are exempt from counting as
    unsupported: they are overwhelmingly list positions and counts of the agent's own sentences rather than claims
    about the card, and penalising them would measure prose style instead of grounding.
    """
    pool = []
    _values_in(tool_outputs, pool)
    pool_arr = np.asarray(sorted(set(pool)), dtype=np.float64) if pool else np.zeros(0)

    def supported(v):
        if pool_arr.size == 0:
            return False
        d = np.abs(pool_arr - v)
        return bool((d <= ABS_TOL).any() or (d <= REL_TOL * np.abs(v)).any())

    stated, unsupported = 0, []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        val = float(raw[:-1]) / 100.0 if raw.endswith("%") else float(raw)
        if raw.endswith("%"):
            cands = (val, val * 100.0)
        else:
            if float(raw).is_integer() and abs(val) <= 8:
                continue                       # see the docstring: list positions, not claims
            cands = (val, val / 100.0)
        stated += 1
        if not any(supported(c) for c in cands):
            unsupported.append(raw)

    R, C = grid
    fabricated = [(int(a), int(b)) for a, b in _WIN.findall(text) if not (0 <= int(a) < R and 0 <= int(b) < C)]
    return {"n_numbers": stated, "n_unsupported": len(unsupported),
            "supported_share": 1.0 - len(unsupported) / stated if stated else float("nan"),
            "unsupported_examples": unsupported[:8],
            "n_fabricated_windows": len(fabricated), "fabricated_examples": fabricated[:8]}


# --------------------------------------------------------------------------------------------- the served model
#: Completion budget per turn. Qwen3-family models reason before answering, and the first 27B run showed a
#: 2,000-token cap spent entirely inside the reasoning on the no-tool arm (120 of 120 empty answers), so the
#: budget matches what the OlmoEarth Agent's own client allows rather than what a 7B needed.
MAX_TOKENS = 8000
MAX_STEPS = 10


def chat(endpoint, model, messages, tools=None, temperature=0.0, seed=0, timeout=600, max_tokens=MAX_TOKENS):
    """One completion from an OpenAI-compatible endpoint. urllib, so the experiment adds no dependency."""
    import urllib.error
    import urllib.request
    body = {"model": model, "messages": messages, "temperature": temperature, "seed": seed,
            "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "EMPTY")})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            choice = json.loads(r.read().decode())["choices"][0]
            msg = choice["message"]
            msg["_finish_reason"] = choice.get("finish_reason")   # kept for the record; never sent back
            return msg
    except urllib.error.HTTPError as exc:
        return {"role": "assistant", "content": f"__HTTP_ERROR__ {exc.code} {exc.read()[:300]!r}"}
    except Exception as exc:  # noqa: BLE001 - a dead endpoint is recorded, not raised, so one card cannot end the run
        return {"role": "assistant", "content": f"__ERROR__ {type(exc).__name__}: {exc}"}


ANSWER_RULE = (
    "Answer with ONE json object and nothing else, in this exact shape:\n"
    '{"review_windows": [[row, col], ...], '
    '"explanations": [{"window": [row, col], "cue": "boundary|low_margin|high_entropy|isolated|other", '
    '"value": number or null}, ...], '
    '"comparison": {"n_differing": int, "where": "short phrase", "believe": "decline|first|second"}}\n'
    'Omit "comparison" when the card has only one inference. Name exactly the number of review windows the '
    "budget allows. Rows and columns are zero-based and must lie inside the grid."
)

TOOLSPEC_A = [
    {"type": "function", "function": {
        "name": "assess",
        "description": "Rank this card's windows by the model's own suspicion and return the review set the budget "
                       "buys, with each window's margin and whether it sits on a class boundary.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "compare",
        "description": "Compare the card's two inferences on identical windows: how many differ, where they sit, "
                       "and what is known about which side to believe.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

TOOLSPEC_B = [
    {"type": "function", "function": {
        "name": "python",
        "description": "Run numpy over THIS CARD'S arrays, which are already loaded as variables: margin, "
                       "decision, valid (2-D arrays of shape grid), budget (float), grid (tuple), and on "
                       "two-inference cards second_margin and second_decision. Do not create or simulate data; "
                       "compute on these variables and print() what you want back. Only printed output returns.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]


def _prompt(card, arm):
    m = card["meta"]
    head = (f"You are auditing a {m.get('task', 'land cover')} prediction map. Grid {list(card['dec'].shape)} "
            f"windows, {int(card['ok'].sum())} of them valid. Sensor: {m.get('sensor', 'unspecified')}. "
            f"The review budget is {BUDGET:.0%}, which buys {n_slots(card)} windows.\n\n"
            "Name the windows a reviewer should check first, say why each is suspect, and")
    head += (" say how the two inferences of this scene differ, where the differences sit, and which side to "
             "believe.\n\n" if card["second"] is not None else " nothing about a second inference.\n\n")
    if arm == "A":
        head += "Use the assess and compare tools to ground every number you state.\n\n"
    elif arm == "B":
        head += "Use the python tool to compute whatever you need from the arrays.\n\n"
    elif arm == "D":
        head += ("You have no rasters for this card, only the description above. Answer as well as you can "
                 "from it.\n\n")
    elif arm in ("E", "E_forced"):
        # The OlmoEarth Agent as shipped: its own registry, its own system prompt. It is told where the scores
        # are and nothing about which of its tools to use; the forced variant pins its review-set skill.
        head += (f"The card's per-class scores are in the JSON file {card.get('scores_json', '<missing>')}: an "
                 "object with 'grid' [rows, cols] and 'scores', one row per window in row-major order, two "
                 "numbers per window (logit-like class scores). ")
        if card["second"] is not None:
            head += (f"The second inference of the same windows is in {card.get('second_scores_json', '<missing>')}, "
                     "same layout. ")
        head += "Use your tools to ground every number you state; do not estimate values you have not computed.\n\n"
    return head + ANSWER_RULE


def _parse_answer(text):
    """The one json object the contract asks for; a parse failure is recorded, never rescued."""
    if not isinstance(text, str):
        return None, "no content"
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None, "no json object in the answer"
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError as exc:
        return None, f"json decode failed: {exc}"
    return (obj, None) if isinstance(obj, dict) else (None, "top-level json is not an object")


def _turn_record(msg):
    return {"finish_reason": msg.get("_finish_reason"), "content_len": len(msg.get("content") or ""),
            "reasoning_len": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
            "n_calls": len(msg.get("tool_calls") or [])}


def run_llm_arm(card, arm, endpoint, model, seed=0, temperature=0.0, max_steps=MAX_STEPS):
    """One agent run: a minimal tool-calling loop, so the benchmark is reproducible without any agent framework."""
    tools = {"A": TOOLSPEC_A, "B": TOOLSPEC_B, "D": None}[arm]
    messages = [{"role": "system", "content": "You are a careful remote-sensing analyst. State only what your "
                                              "tools support."},
                {"role": "user", "content": _prompt(card, arm)}]
    used = {}
    calls_made = []                      # what the model asked for, kept beside what it got back
    transcript = []
    turns = []
    for _ in range(max_steps):
        msg = chat(endpoint, model, messages, tools=tools, temperature=temperature, seed=seed)
        transcript.append(msg)
        turns.append(_turn_record(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name")
            try:
                cargs = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                cargs = {}
            # Qwen double-encodes: `arguments` arrives as a JSON string that itself holds the JSON object, so
            # one decode yields a str. The pilot showed what happens if that str is taken as code: the sandbox
            # evaluates the literal text {"code": "..."} as a dict expression, does nothing, and returns empty
            # output four times in a row. Decode again; only a string that is not JSON is the bare value.
            if isinstance(cargs, str):
                try:
                    again = json.loads(cargs)
                    cargs = again if isinstance(again, dict) else {"code": cargs}
                except json.JSONDecodeError:
                    cargs = {"code": cargs} if name == "python" else {}
            if not isinstance(cargs, dict):
                cargs = {"code": str(cargs)} if name == "python" else {}
            calls_made.append({"name": name, "arguments": {k: (v[:4000] if isinstance(v, str) else v)
                                                           for k, v in cargs.items()}})
            if name == "assess":
                out = tool_assess(card)
            elif name == "compare":
                out = tool_compare(card)
            elif name == "python":
                out = tool_python(cargs.get("code", ""), card)
            else:
                out = {"error": f"unknown tool {name!r}"}
            used.setdefault(name, []).append(out)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "content": json.dumps(out)[:12000]})
    else:
        # Every step was a tool call. One last turn without tools asks for the answer, the same for every arm,
        # so a run that computed and never concluded is scored on what it concludes rather than on silence.
        messages.append({"role": "user", "content": "The tool budget is spent. " + ANSWER_RULE})
        msg = chat(endpoint, model, messages, tools=None, temperature=temperature, seed=seed)
        transcript.append(msg)
        turns.append(_turn_record(msg))
    text = (transcript[-1].get("content") if transcript else "") or ""
    answer, parse_error = _parse_answer(text)
    return {"arm": arm, "answer": answer, "parse_error": parse_error, "text": text,
            "tool_outputs": used, "tool_calls": calls_made, "turns": turns,
            "n_tool_calls": sum(len(v) for v in used.values()), "n_steps": len(transcript)}


def grade_run(run, card):
    """Every score for one run of one card: the three tasks and the claims audit."""
    ans = run.get("answer") or {}
    # Arm A and B are audited against what they actually called; a run that called nothing has an empty pool,
    # so every number it states is unsupported, which is the honest reading of an ungrounded answer.
    audit = claims_audit(run.get("text") or "", run.get("tool_outputs") or {}, card["dec"].shape)
    return {"card": card["name"], "arm": run["arm"], "parse_error": run.get("parse_error"),
            "n_tool_calls": run.get("n_tool_calls", 0),
            "review_set": grade_review_set(ans, card),
            "explanation": grade_explanation(ans, card),
            "comparison": grade_comparison(ans, card),
            "claims": audit}
