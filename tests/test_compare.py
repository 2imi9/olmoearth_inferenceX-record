"""oe_inferencex.compare on synthetic fixtures, the phi overflow regression, and the recorded exp52 / exp55 cross-tabs.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import json
import os

import numpy as np
import pytest
from oe_inferencex.compare import compare_inferences, crosstab, disagreement, phi

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def test_crosstab_reproduces_the_recorded_exp52_cross_tabs():
    """exp52 stored counts, not windows: rebuild each 2x2 table from n_windows, corrected, broken and both."""
    s = json.load(open(_need("exp52_summary.json")))
    for arm, res in s["results"].items():
        for split in ("bolivia", "test"):
            r = res[split]
            ct = r["vs_frozen_head"]
            n = r["n_windows"]
            neither = n - ct["frozen_errors"] - ct["broken"]
            ea = np.repeat([True, True, False, False], [ct["both"], ct["corrected"], ct["broken"], neither])
            eb = np.repeat([True, False, True, False], [ct["both"], ct["corrected"], ct["broken"], neither])
            got = crosstab(ea, eb, np.ones(n, dtype=bool))
            assert got["errors_a"] == ct["frozen_errors"] and (got["corrected"], got["broken"], got["both"]) == (ct["corrected"], ct["broken"], ct["both"])
            assert got["share_corrected"] == pytest.approx(ct["share_corrected"], abs=1e-12)
            assert got["phi"] == pytest.approx(ct["phi"], abs=1e-12), (arm, split)
            assert 1 - ct["frozen_errors"] / n == pytest.approx(r["frozen_window_accuracy"], abs=1e-12)


def test_phi_reproduces_the_recorded_exp55_pooled_cross_tab():
    """exp55's pooled S2-head vs S1-head cross-tab from its window count, accuracies and conditionals."""
    x = json.load(open(_need("exp55_summary.json")))["cross_tab_A_vs_B"]
    n = x["n_windows"]
    ea_n = round(n * (1 - x["acc_A_S2_head"]))
    both = round(ea_n * x["p_b_wrong_given_a_wrong"])
    broken = round((n - ea_n) * x["p_b_wrong_given_a_right"])
    ea = np.repeat([True, True, False, False], [both, ea_n - both, broken, n - ea_n - broken])
    eb = np.repeat([True, False, True, False], [both, ea_n - both, broken, n - ea_n - broken])
    assert phi(ea, eb) == pytest.approx(x["phi"], abs=1e-6)


def _exp57(suffix=""):
    s = json.load(open(_need(f"exp57_summary{suffix}.json")))
    z = np.load(_need(f"exp57_masks{suffix}.npz"))
    return s, {k: z[k] for k in z.files}


def _recompute(a, b, ok, y, keep):
    """exp57's per-pair statistics from the committed decisions alone: the boundary cue of a's own map, tile groups."""
    from oe_inferencex.signals import boundary_indicator
    out = compare_inferences(a, b, ok, groups=np.arange(len(a)), labels=y, cues={"boundary": boundary_indicator(a) > 0})
    return {"n_windows": out["n_windows"], "n_disagree": out["n_disagree"], "disagreement_rate": out["disagreement_rate"],
            "boundary_enrichment": out["where"]["boundary"]["enrichment"],
            "crosstab": {k: out["graded"]["crosstab"][k] for k in ("errors_a", "errors_b", "corrected", "broken", "both", "phi")},
            "which_side": {k: out["graded"]["which_side"][k] for k in ("a_right", "b_right", "neither")}} if keep else out


def _close(got, rec, tol=1e-9):
    """A recorded null is an undefined value (NaN in the module)."""
    return (rec is None and np.isnan(got)) or (rec is not None and got == pytest.approx(rec, abs=tol))


def _assert_pair(rec, got):
    assert (got["n_windows"], got["n_disagree"]) == (rec["n_windows"], rec["n_disagree"])
    assert _close(got["disagreement_rate"], rec["disagreement_rate"], 1e-12)
    assert _close(got["boundary_enrichment"], rec["where"]["boundary"]["enrichment"])
    for k, v in got["crosstab"].items():
        assert _close(v, rec["graded"]["crosstab"][k]), k
    for k, v in got["which_side"].items():
        assert v == rec["graded"]["which_side"][k], k


def check_exp57_atlas_recomputes_from_the_masks(suffix=""):
    """Every Sen1Floods11, encoder and GEOID pair of exp57's summary is the module's output on the committed decisions."""
    s, z = _exp57(suffix)
    R = s["results"]
    pairs = {"offsets": [(f"offset0", f"offset{k}", f"offsets 0 vs {k}") for k in (1, 2, 3)], "backbones": [("v1", "v1_2", None)],
             "sensors": [("s2", "s1", None)], "finetune": [("frozen", "ft", None)]}
    n_checked = 0
    for name in ("bolivia", "test"):
        if f"{name}_ok" not in z:
            continue
        ok, y = z[f"{name}_ok"], z[f"{name}_y"]
        for pair, specs in pairs.items():
            for ka, kb, sub in specs:
                if f"{name}_{ka}" not in z or f"{name}_{kb}" not in z or name not in R.get(pair, {}):
                    continue
                rec = R[pair][name][sub] if sub else R[pair][name]
                _assert_pair(rec, _recompute(z[f"{name}_{ka}"], z[f"{name}_{kb}"], ok, y, True))
                n_checked += 1
    for model, rec in R.get("encoders", {}).items():
        if isinstance(rec, dict) and f"encoders_{model}" in z and f"encoders_ok_{model}" in z:
            _assert_pair(rec, _recompute(z["encoders_olmoearth_base"], z[f"encoders_{model}"], z[f"encoders_ok_{model}"], z["encoders_y"], True))
            n_checked += 1
    if "geoid_ok" in z and R.get("geoid", {}).get("n_windows"):
        g = R["geoid"]
        d = disagreement(z["geoid_s2"], z["geoid_s1"], z["geoid_ok"])
        assert (d["n"], d["n_disagree"]) == (g["n_windows"], g["n_disagree"])
        ct = crosstab(z["geoid_s2"] != z["geoid_y_permanent"], z["geoid_s1"] != z["geoid_y_after"], z["geoid_ok"])
        assert ct["phi"] == pytest.approx(g["graded"]["crosstab_own_labels"]["phi"], abs=1e-9)
        assert set(np.unique(z["geoid_event"]).tolist()) >= set(g["per_event"])
        n_checked += 1
    assert n_checked > 0
    return n_checked


def test_exp57_atlas_recomputes_from_the_masks():
    check_exp57_atlas_recomputes_from_the_masks("")
