"""exp119's summary: the measured costs the package's 1.20.0 CHANGELOG and Usage quote, read from the raw outputs in
exp/out/exp119/ (standard library only). Descriptive release checks, not an experiment with predictions. Run on the
cluster.

    python exp/exp119_summary.py > exp/out/exp119_summary.json

j2 (aicr job 1554014): `estimate --valid` on the demo tile's three samples (the confidence design at 300 and 1,000
labels, the class design at 1,000), each JSON holding the design's own interval (`low`, `high`) beside the valid one
(`valid_interval`); the times are the job log's, one line per command.
j3 (aicr job 1553048, tasks 0 and 2): `grade --ratio direct` on NLCD 2016's eight-class two-map grade (set B, NLCD
against LCMAP) at tolerances 1e-6 and 1e-4; each direct interval has the box and normal theory's interval beside it,
and its time is one line of the timings file."""
import hashlib
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "exp", "out", "exp119")


def path(*p):
    return os.path.join(RAW, *p)


def load(*p):
    with open(path(*p), encoding="utf-8") as f:
        return json.load(f)


def sha256(*p):
    with open(path(*p), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def spread(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


S = {"what": "descriptive release checks of olmoearth-inferencex 1.20.0, not an experiment with predictions",
     "jobs": {"valid": 1554014, "ratio_direct": "1553048 (tasks 0 and 2; GFC2020's tasks were cancelled once their "
                                                "work bounds were known)"},
     "sources": {}}

# j2: estimate --valid, width against the design's own interval, and time per command
times = {}
for line in open(path("j2", "oeix-j2-1554014.out"), encoding="utf-8"):
    m = re.match(r"\s*([\d.]+) s\s+exit (\d+)\s+\S+ estimate (\S+)\.csv(.*?)--out (\S+)", line)
    if m:
        times[m.group(5)] = {"seconds": float(m.group(1)), "exit": int(m.group(2))}
S["sources"]["j2/oeix-j2-1554014.out"] = sha256("j2", "oeix-j2-1554014.out")
valid = {}
for name in ("confidence300", "confidence1000", "class1000"):
    E = load("j2", f"{name}_valid.json")
    S["sources"][f"j2/{name}_valid.json"] = sha256("j2", f"{name}_valid.json")
    S["sources"][f"j2/{name}_valid_pc.json"] = sha256("j2", f"{name}_valid_pc.json")
    V = E["valid_interval"]
    valid[name] = {"design_interval": [E["low"], E["high"]], "design_method": E["method"],
                   "valid_interval": [V["low"], V["high"]], "valid_form": V["form"],
                   "width_ratio_valid_over_design": (V["high"] - V["low"]) / (E["high"] - E["low"]),
                   "seconds_whole_map": times[f"{name}_valid.json"]["seconds"],
                   "seconds_per_class": times[f"{name}_valid_pc.json"]["seconds"]}
S["valid"] = valid
S["valid_seconds"] = {"whole_map_max": max(v["seconds_whole_map"] for v in valid.values()),
                      "per_class_max": max(v["seconds_per_class"] for v in valid.values())}

# j3: grade --ratio direct, widths against the box and normal theory, and time in all
direct = {}
for name, tol in (("nlcd2016_8_direct", 1e-6), ("nlcd2016_8_direct_tol4", 1e-4)):
    G = load("j3", f"{name}.json")
    S["sources"][f"j3/{name}.json"] = sha256("j3", f"{name}.json")
    S["sources"][f"j3/{name}.timings.jsonl"] = sha256("j3", f"{name}.timings.jsonl")
    rows = []
    for m, classes in G["per_class"].items():
        for c, q in classes.items():
            for kind in ("user", "producer"):
                x = q.get(kind) if isinstance(q, dict) else None
                if isinstance(x, dict) and x.get("method") == "betting test inverted over the ratio":
                    w = x["interval"][1] - x["interval"][0]
                    rows.append({"map": m, "class": c, "kind": kind,
                                 "over_box": w / (x["box"][1] - x["box"][0]),
                                 "over_normal": w / (x["normal_theory"][1] - x["normal_theory"][0])})
    secs = [json.loads(line)["seconds"] for line in open(path("j3", f"{name}.timings.jsonl"), encoding="utf-8")
            if line.strip()]
    direct[name] = {"tolerance": tol, "n_direct": len(rows), "n_timed": len(secs), "seconds_in_all": sum(secs),
                    "seconds_largest": max(secs),
                    "width_over_box": spread([r["over_box"] for r in rows]),
                    "width_over_normal": spread([r["over_normal"] for r in rows])}
S["ratio_direct"] = direct

print(json.dumps(S, indent=1, sort_keys=True))
