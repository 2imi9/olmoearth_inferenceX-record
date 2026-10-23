"""exp113 (docs/plan/planteur_transfer.md): PLANTEUR leave-one-island-out with three embeddings. The run's outputs
against the independent second route, the outputs' privacy, the verdicts recomputed from the per-map numbers, and the
two readings of P3's interval."""
import ast
import hashlib
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
SUMMARY = os.path.join(OUT, "exp113_summary.json")
CROSSCHECK = os.path.join(OUT, "exp113_crosscheck.json")
CROSSCHECK_SCRIPT = os.path.join(ROOT, "exp", "exp113_crosscheck.py")
TERRITORIES = ("Guadeloupe", "Martinique", "Guyane", "Réunion", "Mayotte")
CONFIGS = tuple(f"loio:{t}" for t in TERRITORIES) + ("default",)
# the run's embedding keys and the second route's
EMBS = {"oe13": "v1.3", "tessera2": "tessera_v2", "aef": "alphaearth"}
CONTROLS = {"rarity": "ctl_class_rarity", "distance": "ctl_distance_from_fit_mean", "boundary": "ctl_boundary"}


def _load():
    with open(SUMMARY, encoding="utf-8") as f:
        S = json.load(f)
    with open(CROSSCHECK, encoding="utf-8") as f:
        C = json.load(f)
    return S, C


def test_exp113_recorded_by_a_second_route():
    """The second route (written by an agent that saw neither the run's script nor its outputs, importing neither)
    draws the same fit pixels and reaches the same verdicts. It fits the stored float32 vectors where the run fits
    float64, so a few windows flip: the deterministic numbers agree to that precision, the Monte Carlo ones within
    their errors."""
    src = open(CROSSCHECK_SCRIPT, encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"oe_inferencex", "exp113_planteur_transfer"}, "the second route imports neither"
    S, C = _load()
    assert C["script_sha256"] == hashlib.sha256(src.encode("utf-8")).hexdigest(), "the committed crosscheck ran"
    assert C["script_uncommitted"] is False and C["smoke"] is False and S["smoke"] is False
    assert S["inputs"]["loio_json_sha256"] == C["inputs"]["splits/loio.json"]
    assert S["inputs"]["metadata_sha256"] == C["inputs"]["metadata.geojson"]
    assert S["environment"]["scikit_learn"] == C["environment"]["scikit_learn"] == "1.9.1"

    for cfg in CONFIGS:
        f = S["fit"][cfg]
        assert f["fit_pixels"] == C["configurations"][cfg]["fit_set"]["size"] == 100000
        assert f["selection_sha256"] == C["configurations"][cfg]["fit_set"]["sha256_patch_ids_then_flat_pixel_index_sorted"]
        assert all(f[f"converged_{e}"] for e in EMBS)
        for e, x in EMBS.items():
            m, y = S["maps"][cfg][e], C["populations"][cfg][x]
            a, b = m["arm1"], y["arm1"]
            assert a["n"] == b["N"]
            assert abs(a["errors"] - b["errors"]) <= 0.01 * b["errors"]
            assert abs(a["theta"] - b["theta"]) < 0.002
            assert abs(a["own"]["auroc"] - b["auroc"]) < (0.003 if e == "oe13" else 0.005)
            assert abs(a["own"]["capture_10"] - b["capture_10"]) < 0.01
            assert abs(a["lead"] - b["lead_over_best_control"]) < 0.008
            assert CONTROLS[a["best_control"]] == b["best_control"]
            for k, kk in CONTROLS.items():
                assert abs(a["controls"][k]["auroc"] - b["controls"][kk]["auroc"]) < 0.006
            assert abs(a["confident_error_share"] - b["confident_error_share"]) < 0.008
            assert abs(a["band"]["interior"]["error_rate"] - y["edge_band"]["interior"]["error_rate"]) < 0.001
            assert abs(a["band"]["edge_band"]["error_rate"] - y["edge_band"]["band"]["error_rate"]) < 0.025
            assert abs(m["miou"] - y["miou_pixels_nonvoid"]["miou_8_absent_0"]) < 0.002
            for B in ("100", "400"):
                # exact coverage jumps with K, so a few flipped windows can move it by about a point
                assert abs(m["arm2"][B] - y["arm2"][B]["exact_coverage"]) < 0.015
            for al in ("0.05", "0.1"):
                r, z = m["arm3"]["by_alpha"][al], y["arm3"][al]
                p = z["exact_p_certify_any_guard_aside"]
                assert abs(r["p_certify_any"] - p) <= 4 * math.sqrt(p * (1 - p) / 2000) + 0.005
                v = max(r["violation_share"], z["violation_share"])
                assert abs(r["violation_share"] - z["violation_share"]) <= 4 * math.sqrt(2 * v * (1 - v) / 2000) + 0.001
            if "boot90" in a:
                for i in (0, 1):
                    assert abs(a["boot90"]["auroc"][i] - b["bootstrap"]["auroc"][i]) < 0.005
    for t in TERRITORIES:
        for pair, v in S["which_map"][t].items():
            ea, eb = pair.split("_vs_")
            w = C["arm4"][f"loio:{t}"][f"{EMBS[ea]}~{EMBS[eb]}"]
            assert v["N"] == w["N"] and abs(v["D"] - w["D"]) <= 0.1 * w["D"]
            assert v["better"] == w["better_agreeing"]
            assert abs(v["acc_A"] - w["agreement_A"]) < 0.002 and abs(v["acc_B"] - w["agreement_B"]) < 0.002

    V, W = S["verdicts"], C["verdicts"]
    assert V["P1"]["holds"] is W["P1"]["holds"] is True
    assert V["P2"]["holds"] is W["P2"]["holds"] is False
    assert V["P3"]["holds"] is W["P3"]["holds"] is True
    assert V["P4"]["holds"] is W["P4"]["holds"] is True
    assert V["P5"]["holds"] is W["P5"]["holds"] is True


def test_exp113_outputs_hold_aggregates_only():
    """No patch or window list and no coordinate in either output."""
    S, C = _load()

    def longest(o):
        if isinstance(o, dict):
            return max([longest(v) for v in o.values()] + [0])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0

    def keys(o):
        if isinstance(o, dict):
            return set(o) | set().union(*(keys(v) for v in o.values()))
        if isinstance(o, list):
            return set().union(*(keys(v) for v in o)) if o else set()
        return set()
    assert longest(S) <= 64 and longest(C) <= 64, "aggregates only"
    for k in keys(S) | keys(C):
        assert k.lower() not in {"lon", "lat", "x", "y", "geometry", "coordinates", "bbox", "id_patch"}, k


def test_exp113_verdicts_from_the_per_map_numbers():
    """P1, P2, P4 and P5 recomputed from the run's per-map numbers with the page's bars, without the run's verdict
    code."""
    S, _ = _load()
    v13 = {t: S["maps"][f"loio:{t}"]["oe13"]["arm1"] for t in TERRITORIES}
    assert all(a["errors"] >= 200 for a in v13.values())
    meets = [t for t, a in v13.items()
             if a["own"]["auroc"] >= 0.70 and a["own"]["capture_10"] >= 0.25
             and a["own"]["auroc"] - max(c["auroc"] for c in a["controls"].values()) >= 0.05]
    assert meets == list(TERRITORIES) and S["verdicts"]["P1"]["territories_meeting"] == meets
    for a in v13.values():
        best = max(a["controls"], key=lambda k: a["controls"][k]["auroc"])
        assert best == a["best_control"] and abs(a["own"]["auroc"] - a["controls"][best]["auroc"] - a["lead"]) < 1e-12
    drop = S["maps"]["default"]["oe13"]["arm1"]["own"]["auroc"] - sum(a["own"]["auroc"] for a in v13.values()) / 5
    assert 0.0469 < drop < 0.05 and abs(drop - S["verdicts"]["P2"]["drop"]) < 1e-12
    viol = [S["maps"][f"loio:{t}"][e]["arm3"]["by_alpha"]["0.05"]["violation_share"] for t in TERRITORIES for e in EMBS]
    assert len(viol) == 15 and max(viol) == 0.0505 <= 0.115
    worse = [v["p_names_worse"] for t in TERRITORIES for pr in S["which_map"][t].values() for v in pr["n"].values()]
    assert len(worse) == 30 and max(worse) <= 0.025


def test_exp113_p3_interval_readings():
    """P3 was graded on the hypergeometric interval, which covers at least 95% for every (N, K, B); the package's
    exact_coverage_srs defaults to the finite-population Wilson interval, under which Mayotte's TESSERA v2 map at
    100 windows covers 0.896, as the second route's 0.895 (its K is one window off)."""
    from oe_inferencex import estimate as est
    S, C = _load()
    a = S["maps"]["loio:Mayotte"]["tessera2"]["arm1"]
    wilson = est.exact_coverage_srs(a["n"], a["errors"], 100)
    exact = est.exact_coverage_srs(a["n"], a["errors"], 100, interval=est.hypergeom_interval)
    assert round(wilson, 3) == 0.896 and wilson < 0.95
    assert abs(exact - S["maps"]["loio:Mayotte"]["tessera2"]["arm2"]["100"]) < 1e-8 and exact >= 0.95
    assert abs(wilson - C["verdicts"]["P3"]["reading_wilson_fpc"]["min_exact_coverage"]) < 0.002
    r = S["maps"]["loio:Réunion"]["aef"]["arm1"]
    low = est.exact_coverage_srs(r["n"], r["errors"], 100, interval=est.hypergeom_interval)
    assert abs(low - S["verdicts"]["P3"]["min_exact_coverage"]) < 1e-8 and low >= 0.95
