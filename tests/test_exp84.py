"""exp84's statistics by a second route (docs/plan/where_the_lead_holds.md): the confound correlations and the
group medians of one export recomputed with explicit loops, and Holm's step-down against an independent
implementation on a case with a known answer."""
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))


def _an_export():
    for pattern in ("exp/out/exp79_seeds/*.json", "exp/out/exp79_engine/*.json"):
        files = sorted(glob.glob(os.path.join(ROOT, pattern)))
        if files:
            return json.load(open(files[0]))
    pytest.skip("no exp79 export present")


def _spearman(x, y):
    def ranks(v):
        return [sum(1 for u in v if u < w) + (sum(1 for u in v if u == w) + 1) / 2 for w in v]
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def test_the_confound_and_the_group_medians_recompute_by_explicit_loops():
    import exp84_where_the_lead_holds as e84
    export = _an_export()
    rows = e84.encoder_rows(export)
    for s in (0, len(rows) - 1):
        acc, au, ld, by = [], [], [], {}
        for t, v in export["tasks"].items():
            r = v["seeds"][s]
            sig = r["signals"]
            lead = min(sig["ctl_embedding_distance"]["excess_aurc"], sig["ctl_class_rarity"]["excess_aurc"]) - sig["margin"]["excess_aurc"]
            acc.append(r["test_accuracy"]); au.append(sig["margin"]["auroc"]); ld.append(lead)
            by.setdefault(e84.sensor_group(t), []).append(lead)
        assert abs(rows[s]["rho_accuracy_auroc"] - _spearman(acc, au)) < 1e-9
        assert abs(rows[s]["rho_accuracy_lead"] - _spearman(acc, ld)) < 1e-9
        for g, v in by.items():
            assert abs(rows[s]["groups"]["sensor"][g]["median_lead"] - float(np.median(v))) < 1e-12
        assert rows[s]["losses"] == sum(1 for x in ld if x <= 0)
        assert abs(rows[s]["sensor_range"] - (max(np.median(v) for v in by.values()) - min(np.median(v) for v in by.values()))) < 1e-12


def test_holm_matches_an_independent_step_down_and_a_known_answer():
    import exp84_where_the_lead_holds as e84
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
    h = e84.holm(p)
    # by hand: sorted 0.01, 0.03, 0.04, 0.20 with multipliers 4, 3, 2, 1 -> 0.04, 0.09, 0.09, 0.20 (monotone)
    assert abs(h["a"]["adjusted"] - 0.04) < 1e-12 and abs(h["c"]["adjusted"] - 0.09) < 1e-12
    assert abs(h["b"]["adjusted"] - 0.09) < 1e-12 and abs(h["d"]["adjusted"] - 0.20) < 1e-12
    assert h["a"]["survives"] and not h["c"]["survives"] and not h["d"]["survives"]
    # an independent implementation on random p-values
    rng = np.random.default_rng(0)
    q = {f"t{i}": float(x) for i, x in enumerate(rng.random(12) * 0.1)}
    h = e84.holm(q)
    items = sorted(q.items(), key=lambda kv: kv[1])
    m, run = len(items), 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, v * (m - i)))
        assert abs(h[k]["adjusted"] - run) < 1e-12 and h[k]["survives"] == (run < 0.05)
