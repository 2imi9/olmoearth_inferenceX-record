"""exp119: the measured costs olmoearth-inferencex 1.20.0's CHANGELOG and Usage quote (descriptive release checks,
aicr jobs 1554014 and 1553048). The summary against the raw outputs: the files it read, every ratio by a second
traversal of the JSON, the job's printed text against the JSON, and the times by a second parse of the log.
Standard library only."""
import hashlib
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
RAW = os.path.join(OUT, "exp119")


def _json(*p):
    with open(os.path.join(*p), encoding="utf-8") as f:
        return json.load(f)


S = _json(OUT, "exp119_summary.json")


def test_the_summary_read_the_committed_files():
    for rel, digest in S["sources"].items():
        with open(os.path.join(RAW, rel), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, rel


def _intervals_with_a_box(node):
    """Every object holding an interval, the box and normal theory's interval, wherever it sits in the JSON."""
    if isinstance(node, dict):
        if {"interval", "box", "normal_theory"} <= set(node):
            yield node
        for v in node.values():
            yield from _intervals_with_a_box(v)
    elif isinstance(node, list):
        for v in node:
            yield from _intervals_with_a_box(v)


def test_the_direct_widths_by_a_second_traversal():
    for name, D in S["ratio_direct"].items():
        rows = list(_intervals_with_a_box(_json(RAW, "j3", f"{name}.json")))
        assert len(rows) == D["n_direct"] == 26, name
        width = lambda iv: iv[1] - iv[0]
        for key, other in (("width_over_box", "box"), ("width_over_normal", "normal_theory")):
            r = [width(x["interval"]) / width(x[other]) for x in rows]
            got = {"median": statistics.median(r), "min": min(r), "max": max(r)}
            assert all(abs(got[k] - D[key][k]) < 1e-12 for k in got), (name, key, got)


def test_the_printed_text_matches_the_json():
    """The job's own printed table (one decimal, in percent) against the JSON's direct intervals."""
    G = _json(RAW, "j3", "nlcd2016_8_direct.json")
    text = open(os.path.join(RAW, "j3", "nlcd2016_8_direct.txt"), encoding="utf-8").read()
    checked = 0
    for block in re.split(r"\n(?=\w+ per class:)", text):
        m = re.match(r"(\w+) per class:", block)
        if not m or m.group(1) not in G["per_class"]:
            continue
        for line in block.splitlines()[1:]:
            row = re.match(r"\s+(\d+): (.*)", line)
            if not row:
                continue
            parts = row.group(2).split(" | ")
            for kind, part in zip(("user", "producer"), parts[:2]):
                q = G["per_class"][m.group(1)][row.group(1)][kind]
                if q.get("method") != "betting test inverted over the ratio":
                    continue
                lo, hi = (float(v) for v in re.search(r"\[([\d.]+), ([\d.]+)\]", part).groups())
                # printed to one decimal, possibly rounded outward: within 0.1 point of the JSON's end
                assert abs(lo - 100 * q["interval"][0]) <= 0.1 and abs(hi - 100 * q["interval"][1]) <= 0.1, \
                    (m.group(1), row.group(1), kind)
                checked += 1
    assert checked == 26


def test_the_valid_widths_and_times_by_a_second_parse():
    log = open(os.path.join(RAW, "j2", "oeix-j2-1554014.out"), encoding="utf-8").read().split("\n")
    seconds = {}
    for line in log:
        words = line.split()
        if len(words) > 4 and words[1] == "s" and words[2] == "exit" and "--out" in words:
            seconds[words[words.index("--out") + 1]] = float(words[0])
    for name, V in S["valid"].items():
        E = _json(RAW, "j2", f"{name}_valid.json")
        ratio = (E["valid_interval"]["high"] - E["valid_interval"]["low"]) / (E["high"] - E["low"])
        assert abs(ratio - V["width_ratio_valid_over_design"]) < 1e-12, name
        assert seconds[f"{name}_valid.json"] == V["seconds_whole_map"], name
        assert seconds[f"{name}_valid_pc.json"] == V["seconds_per_class"], name
