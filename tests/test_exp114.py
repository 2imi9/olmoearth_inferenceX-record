"""exp114 (docs/plan/prior_shift_pathway.md): prior shift or feature shift on exp113's PLANTEUR maps. The run's outputs
against the independent second route, the outputs' privacy, and the verdicts recomputed from the per-map numbers."""
import ast
import hashlib
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
SCRIPT = os.path.join(ROOT, "exp", "exp114_crosscheck.py")
EMBS = {"oe13": "v1.3", "tessera2": "tessera_v2", "aef": "alphaearth"}


def _load():
    with open(os.path.join(OUT, "exp114_summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    with open(os.path.join(OUT, "exp114_crosscheck.json"), encoding="utf-8") as f:
        C = json.load(f)
    return S, C


def _tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def test_exp114_recorded_by_a_second_route():
    """The second route (written without the run's script or outputs, importing neither it nor the package) refits the
    maps from exp113's saved fit pixels; every number it computed matches the run's, the run's own draws agree with
    the exact rejection probabilities, and the verdicts are the same. (The label-corrected maps have no second route.)"""
    src = open(SCRIPT, encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"oe_inferencex", "exp114_prior_shift", "exp113_planteur_transfer"}
    S, C = _load()
    assert C["script_sha256"] == hashlib.sha256(src.encode("utf-8")).hexdigest(), "the committed crosscheck ran"
    assert C["smoke"] is False and S["smoke"] is False
    assert len(S["reproduction"]) == 18 and all(v["window_class_agreement"] == 1.0 for v in S["reproduction"].values())
    zs = []
    for cfg, maps in S["maps"].items():
        cc = C["maps"][cfg.replace(":", "_")]
        for e, r in maps.items():
            x = cc[EMBS[e]]
            assert r["n_pixels"] == x["target_nonvoid_pixels"]
            assert _tv(r["pi_t"], x["pi_t"]) == 0 and _tv(r["pi_s"], x["pi_s"]) < 1e-12
            u, xu, o, xo = r["uncorrected"], x["uncorrected"], r["oracle"], x["oracle"]
            for a, b in ((u["pixel_accuracy"], xu["pixel_accuracy"]), (u["miou"], xu["miou_8"]),
                         (u["window_error_rate"], xu["window_error_rate"]), (o["pixel_accuracy"], xo["pixel_accuracy"]),
                         (o["window_error_rate"], xo["window_error_rate"]),
                         (r["pathway_gap_uncorrected"], xu["gap_tv_mean_posterior_to_pi_t"]),
                         (r["pathway_gap"], xo["gap_tv_mean_posterior_to_pi_t"]), (r["em"]["tv_to_pi_t"], x["tv"]["em_to_pi_t"]),
                         (r["bbse"]["tv_to_pi_t"], x["tv"]["bbse_to_pi_t"]),
                         (r["tv_predicted_shares"], x["tv"]["predicted_shares_to_pi_t"])):
                assert abs(a - b) < 1e-9
            assert r["em"]["iterations"] == x["em"]["iterations"] and _tv(r["em"]["pi"], x["em"]["pi"]) < 1e-12
            for n in ("100", "400"):
                a, y = r["random"][n], x["random_labels"][n]
                assert abs(a["size_exact"] - y["size_joint_exact"]) < 2e-9
                assert abs(a["size_union_bound"] - y["size_bonferroni_sum_exact"]) < 2e-9
                assert abs(a["reject_em_exact"] - y["em_rejection_exact"]) < 2e-9
                p = y["em_rejection_exact"]
                se = math.sqrt(max(p * (1 - p), 1e-12) / a["draws"])
                zs.append(abs(a["reject_em"] - p) / se if se > 1e-6 else 0.0)
    assert max(zs) < 4 and sum(z > 2 for z in zs) <= 4
    V, W = S["verdicts"], C["verdicts"]
    assert V["P1"]["holds"] is W["P1"]["holds"] is True
    assert V["P2"]["holds"] is W["P2"]["holds"] is False
    assert V["P3"]["holds"] is W["P3"]["holds"] is True
    assert V["P4"]["holds"] is W["P4"]["holds"] is True


def test_exp114_outputs_hold_aggregates_only():
    S, C = _load()

    def longest(o):
        if isinstance(o, dict):
            return max([longest(v) for v in o.values()] + [0])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0
    assert longest(S) <= 64 and longest(C) <= 64


def test_exp114_verdicts_from_the_per_map_numbers():
    """P1 to P4 recomputed from the run's per-map numbers with the page's bars, without the run's verdict code."""
    S, _ = _load()
    m = S["maps"]
    dflt = m["default"]["oe13"]["pathway_gap"]
    meets = [t for t in ("Guyane", "Réunion", "Mayotte")
             if m[f"loio:{t}"]["oe13"]["pathway_gap"] >= 0.05 and m[f"loio:{t}"]["oe13"]["pathway_gap"] >= 3 * dflt]
    assert meets == ["Guyane", "Réunion"]
    held = [(c, e) for c in m if c != "default" for e in m[c]]
    assert len(held) == 15
    assert sum(m[c][e]["em"]["tv_to_pi_t"] < m[c][e]["tv_predicted_shares"] for c, e in held) == 3
    far = [c for c in m if c != "default" and m[c]["oe13"]["em"]["tv_to_pi_t"] >= 0.05]
    assert len(far) == 5 and all(m[c]["oe13"]["random"]["400"]["reject_em"] >= 0.8 for c in far)
    assert max(r["random"][n]["size_exact"] for c in m.values() for r in c.values() for n in ("100", "400")) <= 0.05
