"""exp88's measures and grades before the preregistration is frozen and before any real run
(docs/plan/missing_modality.md). Two synthetic cases go through every measure and every grade: in one the extra
errors under a missing input are confident ones, so P2 must hold; in the other the confidence falls with the
accuracy, so P2 must fail. The grades are also checked on planted numbers at their thresholds, the confident-error
share and the estimation study by a second route, and the guard that refuses a full run while the plan is a draft."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp88_missing_modality as e88   # noqa: E402  (module level is torch-free; torch is imported inside the real modes)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return e88.smoke(out_dir=str(tmp_path_factory.mktemp("exp88")))


def test_p2_holds_when_the_extra_errors_are_confident(smoke):
    g = smoke["cases"]["confident"]["prereg"]
    assert g["P2"]["holds"] is True
    assert g["P2"]["confident_share_optical_missing"] - g["P2"]["confident_share_full"] >= 0.30, \
        "the planted effect is large, so the grade does not rest on the threshold"
    assert g["P3"]["holds"] is False, "confident errors sit where the margin trusts the map, so it cannot rank them"


def test_p2_fails_when_the_confidence_falls_with_the_accuracy(smoke):
    g = smoke["cases"]["tracks"]["prereg"]
    assert g["P2"]["holds"] is False
    assert g["P2"]["rise_points"] < 0.03, "the planted effect is far from the threshold"
    assert "tracks the lost information" in g["P2"]["reading"]
    assert g["P1"]["holds"] is True and g["P3"]["holds"] is True


def test_every_grade_is_decided_in_both_cases_as_designed(smoke):
    for case, want in e88.SMOKE_EXPECTED.items():
        g = smoke["cases"][case]["prereg"]
        assert g["complete"] is True and g["graded_on"] == "synthetic, probe seed 0"
        assert {p: g[p]["holds"] for p in want} == want, case
        fam = smoke["cases"][case]["results"]["synthetic"]["seeds"]["0"]["families"]
        assert set(fam) == set(e88.FAMILIES), "Togo is measured and reported though not graded"
        for f in fam.values():
            assert set(f["conditions"]) == set(e88.CONDITIONS)
            assert "shift_from_full" in f["conditions"]["optical_missing"] and "shift_from_full" not in f["conditions"]["full"]
    mm = smoke["cases"]["tracks"]["results"]["synthetic"]["seeds"]["0"]["mixed_map"]
    assert mm["n_cloudy_tiles"] == mm["n_tiles"] // 2
    assert {"cloudy_errors_in_review_share", "cloudy_share_of_all_errors", "cloudy_share_of_review_errors"} <= set(mm["review_set"])
    assert smoke["planted"]["P1_at_threshold"] is True and smoke["units_npz"]["written"] is True


def test_the_confident_share_by_explicit_loops():
    """Measure 3 recomputed unit by unit: the threshold is the median margin of the units the full input gets right."""
    rng = np.random.default_rng(3)
    n, C, D = 400, 3, 4
    y = rng.integers(0, C, n)
    conds = {}
    for c in e88.CONDITIONS:
        p = rng.dirichlet(np.ones(C) * (0.4 if c == "optical_missing" else 2.0), size=n)
        conds[c] = {"p": p, "y": y, "tile": None, "emb_te": rng.standard_normal((n, D)).astype(np.float32)}
    meas, units = e88.evaluate_family({"conditions": conds, "emb_tr": rng.standard_normal((50, D)).astype(np.float32), "C": C})
    full = conds["full"]["p"]
    correct_margins = []
    for i in range(n):
        s = sorted(full[i])
        if int(np.argmax(full[i])) == y[i]:
            correct_margins.append(s[-1] - s[-2])
    thr = float(np.median(correct_margins))
    assert meas["threshold_confident"] == pytest.approx(thr, abs=1e-12)
    for c in e88.CONDITIONS:
        p = conds[c]["p"]
        n_err = n_conf = 0
        for i in range(n):
            if int(np.argmax(p[i])) != y[i]:
                n_err += 1
                s = sorted(p[i])
                n_conf += (s[-1] - s[-2]) >= thr
        got = meas["conditions"][c]
        assert got["n_errors"] == n_err and got["confident_errors"]["n"] == n_conf
        assert got["confident_errors"]["share"] == pytest.approx(n_conf / n_err, abs=1e-12)
        lead = got["ranking"]["controls_excess_aurc"][got["ranking"]["best_control"]] - got["ranking"]["margin_excess_aurc"]
        assert got["ranking"]["lead_over_best_control"] == pytest.approx(lead, abs=1e-12)


def test_the_mixed_map_and_the_estimation_study():
    rng = np.random.default_rng(5)
    n_tiles, per = 30, 40
    tile = np.repeat(np.arange(n_tiles), per)
    full = {"margin": rng.random(tile.size), "p1": rng.random(tile.size), "err": (rng.random(tile.size) < 0.1).astype(float)}
    cloud = {"margin": rng.random(tile.size), "p1": rng.random(tile.size), "err": (rng.random(tile.size) < 0.4).astype(float)}
    mm = e88.mixed_map(full, cloud, tile, n_tiles)
    assert mm["cloudy_tiles"].size == 15 and np.array_equal(mm["cloudy_tiles"], e88.mixed_map(full, cloud, tile, n_tiles)["cloudy_tiles"])
    for t in range(n_tiles):
        src = cloud if t in set(mm["cloudy_tiles"].tolist()) else full
        assert np.array_equal(mm["err"][tile == t], src["err"][tile == t])
    es = e88.estimation_study(mm["err"], mm["cloudy"], budget=300, draws=2000, seed=0)
    e, cl = mm["err"] > 0.5, mm["cloudy"]
    assert es["truth"] == pytest.approx({"cloudy": e[cl].mean(), "clear": e[~cl].mean(), "whole": e.mean()})
    # a simple random sample's mean is unbiased for the whole map, so read as the cloudy part it is off by the gap
    assert es["pooled"]["bias_for_cloudy_part"] == pytest.approx(e.mean() - e[cl].mean(), abs=0.006)
    for h in ("cloudy", "clear"):
        assert es["stratified"][h]["coverage"] >= 0.93, "the exact interval per part covers at its nominal rate"
        assert es["stratified"][h]["draws_without_a_label"] == 0


def test_planted_grades_at_their_thresholds():
    got = e88.planted_checks()
    assert got["P1_at_threshold"] is True and got["P2_below"] is False and got["P3_tie_is_not_a_win"] is False
    assert got["P4_one_part_under_covers"] is False and got["P1_without_pastis"] is None


def test_the_cli_smoke_runs_in_seconds_and_writes_its_outputs(tmp_path):
    assert e88.main(["--smoke", "--out-dir", str(tmp_path)]) == 0
    out = json.load(open(tmp_path / "exp88_summary_smoke.json"))
    assert {c: out["cases"][c]["prereg"]["P2"]["holds"] for c in out["cases"]} == {"confident": True, "tracks": False}
    assert out["seconds"] < 30 and (tmp_path / "exp88_units_smoke.npz").exists()


def test_the_full_run_is_refused_while_the_plan_is_a_draft(tmp_path, monkeypatch):
    assert e88.prereg_status("**Status: frozen on 3 October 2026, before any run.**") == "frozen"
    assert e88.prereg_status("**Status:** frozen on 3 October 2026.") == "frozen"
    assert e88.prereg_status("**Status: frozen on 3 October 2026; the draft of 28 September is in the history.**") == "frozen"
    assert e88.prereg_status("**Status: draft, not frozen.** Written 28 September 2026.") == "draft"
    assert e88.prereg_status("**Status: not frozen.**") == "draft"
    assert e88.prereg_status("no status line") == "unknown"
    with open(os.path.join(ROOT, "docs", "plan", "missing_modality.md"), encoding="utf-8") as f:
        assert e88.prereg_status(f.read()) in ("draft", "frozen"), "the plan page's status line is readable"
    plan = tmp_path / "missing_modality.md"
    plan.write_text("# exp88\n\n**Status: draft, not frozen.** Written 28 September 2026.\n")
    monkeypatch.setattr(e88, "PLAN", str(plan))
    assert e88.main([]) == 2, "no run on real embeddings before the page is frozen"


def test_the_units_file_drops_replicate_seeds_before_it_drops_the_graded_one(tmp_path):
    a = {"m/f/y": np.zeros(10, np.int16), "m/f/seed0/full/err": np.zeros(10, np.uint8),
         "m/f/seed1/full/err": np.random.default_rng(0).integers(0, 255, 200_000).astype(np.uint8)}
    big = e88.write_units(a, str(tmp_path / "u.npz"), limit=1e9)
    assert big["written"] and not big["replicate_seeds_dropped"] and big["n_arrays"] == 3
    small = e88.write_units(a, str(tmp_path / "u.npz"), limit=50_000)
    assert small["written"] and small["replicate_seeds_dropped"] and small["n_arrays"] == 2
    none = e88.write_units(a, str(tmp_path / "u.npz"), limit=10)
    assert not none["written"] and not (tmp_path / "u.npz").exists()


def test_the_units_file_drops_the_replication_encoders_seeds_first(tmp_path):
    rng = np.random.default_rng(1)
    big = lambda: rng.integers(0, 255, 200_000).astype(np.uint8)            # noqa: E731
    a = {f"{e88.BASE}/f/seed0/full/err": np.zeros(10, np.uint8), f"{e88.BASE}/f/seed1/full/err": big(),
         f"{e88.LARGE}/f/seed0/full/err": np.zeros(10, np.uint8), f"{e88.LARGE}/f/seed1/full/err": big()}
    one = e88.write_units(a, str(tmp_path / "u.npz"), limit=300_000)
    assert one["written"] and one["replicate_seeds_dropped_for"] == "replication encoder" and one["n_arrays"] == 3
    with np.load(tmp_path / "u.npz") as z:
        assert f"{e88.BASE}/f/seed1/full/err" in z.files and f"{e88.LARGE}/f/seed1/full/err" not in z.files
    both = e88.write_units(a, str(tmp_path / "u.npz"), limit=50_000)
    assert both["written"] and both["replicate_seeds_dropped_for"] == "all" and both["n_arrays"] == 2


RUN_UNITS = os.path.join(ROOT, "exp", "out", "exp88_units.npz")
RUN_SUMMARY = os.path.join(ROOT, "exp", "out", "exp88_summary.json")


@pytest.mark.skipif(not os.path.exists(RUN_UNITS), reason="the recorded run's per-unit file is not present")
def test_the_recorded_run_by_a_second_route():
    """The recorded numbers recomputed from the per-unit file with plain numpy, none of exp88's code: the error
    rates and confident shares on every family and seed, the mixed map's truths, the review set's capture, and the
    one-class collapse the record quotes."""
    z = np.load(RUN_UNITS)
    s = json.load(open(RUN_SUMMARY))
    for enc, res in s["results"].items():
        for seed, rs in res["seeds"].items():
            for fam, rf in rs["families"].items():
                base = f"{enc}/{fam}/seed{seed}/"
                if base + "full/err" not in z.files:
                    continue
                full_err = z[base + "full/err"].astype(bool)
                thr = np.median(z[base + "full/margin"][~full_err])
                assert abs(thr - rf["threshold_confident"]) < 1e-6
                for cond, rc in rf["conditions"].items():
                    err = z[base + cond + "/err"].astype(bool)
                    margin = z[base + cond + "/margin"]
                    assert abs(err.mean() - rc["error_rate"]) < 1e-12
                    confident = sum(1 for m, e in zip(margin, err) if e and m >= thr) if len(err) < 10000 \
                        else int(((margin >= thr) & err).sum())
                    assert abs(confident / err.sum() - rc["confident_errors"]["share"]) < 1e-12
    b = "olmoearth_base/pastis/"
    cloudy = np.isin(z[b + "tile"], z[b + "cloudy_tiles"])
    err = np.where(cloudy, z[b + "seed0/optical_missing/err"], z[b + "seed0/full/err"]).astype(bool)
    margin = np.where(cloudy, z[b + "seed0/optical_missing/margin"], z[b + "seed0/full/margin"])
    mm = s["results"]["olmoearth_base"]["seeds"]["0"]["mixed_map"]
    assert abs(err[cloudy].mean() - mm["error_rate"]["cloudy"]) < 1e-12
    assert abs(err[~cloudy].mean() - mm["error_rate"]["clear"]) < 1e-12
    review = np.zeros(len(err), bool)
    review[np.argsort(margin, kind="stable")[:round(0.05 * len(err))]] = True
    assert abs((review & err & cloudy).sum() / (err & cloudy).sum()
               - mm["review_set"]["cloudy_errors_in_review_share"]) < 2e-3, "ties at the cut may split differently"
    assert abs((review & err & ~cloudy).sum() / (err & ~cloudy).sum()
               - mm["review_set"]["clear_errors_in_review_share"]) < 2e-3
    dec = z[b + "seed0/optical_missing/dec"]
    assert round(float((dec == 1).mean()), 3) == 0.826 and round(float((z[b + "y"] == 1).mean()), 3) == 0.207


MATCHED = os.path.join(ROOT, "exp", "out", "exp88_matched_head.json")


def _midrank_auroc(score, positive):
    """Mann-Whitney AUROC of `score` for `positive`, ties at their mid-rank, written without the package."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, bool)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    before = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = (before + (counts + 1) / 2.0)[inv]
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return (ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _tiles_of(export):
    """The tile of every exported window, from the packed validity mask and the (tiles, rows, cols) grid."""
    n, h, w = (int(v) for v in export["grid"])
    ok = np.unpackbits(export["ok_packed"])[:int(export["ok_len"][0])].astype(bool)
    return np.repeat(np.arange(n), h * w)[ok]


@pytest.mark.skipif(not (os.path.exists(RUN_UNITS) and os.path.exists(MATCHED)), reason="the recorded files are not present")
def test_the_matched_head_by_a_second_route():
    """exp88_matched_head.json recomputed with plain numpy from the per-unit files: exp88's probe and the probe
    trained on each input, on the same PASTIS windows, the confident share at exp88's threshold, a mid-rank AUROC,
    and the half-cloudy map with a plain stable-sort review set. Large is read only where its export is present."""
    z = np.load(RUN_UNITS)
    rec = json.load(open(MATCHED))
    sources = {"olmoearth_base": os.path.join(ROOT, "exp", "out", "exp78_units"),
               "olmoearth_large": os.path.join(ROOT, "exp", "out", "exp79_units", "olmoearth_large")}
    checked = []
    for enc, d in sources.items():
        s1_path = os.path.join(d, "pastis_sentinel1.npz")
        if not os.path.exists(s1_path):
            continue
        b = f"{enc}/pastis/"
        y, tile = z[b + "y"].astype(int), z[b + "tile"]
        s1 = np.load(s1_path)
        assert np.array_equal(_tiles_of(s1), tile)
        s1_err = s1["err"].astype(bool)
        assert np.array_equal(s1["dec"].astype(int) == y, ~s1_err), "the S1 export is not in exp88's unit order"
        full_err = z[b + "seed0/full/err"].astype(bool)
        full_m = z[b + "seed0/full/margin"].astype(np.float64)
        thr = np.median(full_m[~full_err])
        R = rec["encoders"][enc]["families"]["pastis"]
        assert abs(thr - R["threshold_confident"]) < 1e-9
        heads = {"s1s2_head_on_s1s2": (full_m, full_err),
                 "s1s2_head_on_s1": (z[b + "seed0/optical_missing/margin"].astype(np.float64),
                                     z[b + "seed0/optical_missing/err"].astype(bool)),
                 "s1_head_on_s1": (s1["margin"].astype(np.float64), s1_err)}
        for name, (m, e) in heads.items():
            r = R["rows"][name]
            assert abs(e.mean() - r["error_rate"]) < 1e-12, (enc, name)
            assert abs(np.count_nonzero(m[e] >= thr) / e.sum() - r["confident_share"]) < 1e-12, (enc, name)
            assert abs(_midrank_auroc(-m, e) - r["margin_auroc"]) < 1e-9, (enc, name)
        cloudy = np.isin(tile, z[b + "cloudy_tiles"])
        for variant, cloud in (("mismatched", "s1s2_head_on_s1"), ("matched", "s1_head_on_s1")):
            m = np.where(cloudy, heads[cloud][0], full_m)
            e = np.where(cloudy, heads[cloud][1], full_err)
            mm = rec["encoders"][enc]["mixed_map"][variant]
            assert abs(e[cloudy].mean() - mm["error_rate"]["cloudy"]) < 1e-12
            assert abs(e[~cloudy].mean() - mm["error_rate"]["clear"]) < 1e-12
            review = np.zeros(e.size, bool)
            review[np.argsort(m, kind="stable")[:round(0.05 * e.size)]] = True
            for part, mask in (("cloudy", cloudy), ("clear", ~cloudy)):
                got = (review & e & mask).sum() / (e & mask).sum()
                assert abs(got - mm["review_set"][f"{part}_errors_in_review_share"]) < 2e-3, (enc, variant, part)
            assert abs((review & e).sum() / e.sum() - mm["review_set"]["all_errors_in_review_share"]) < 2e-3
        checked.append(enc)
        # CropHarvest China 6: every head on its input, by the same plain route (samples, so no tiles)
        c1 = os.path.join(d, "cropharvest_Peoples_Republic_of_China_6_sentinel1.npz")
        c2 = os.path.join(d, "cropharvest_Peoples_Republic_of_China_6.npz")
        b = f"{enc}/china6/"
        y = z[b + "y"].astype(int)
        R = rec["encoders"][enc]["families"]["china6"]
        full_err = z[b + "seed0/full/err"].astype(bool)
        thr = np.median(z[b + "seed0/full/margin"].astype(np.float64)[~full_err])
        assert abs(thr - R["threshold_confident"]) < 1e-9
        rows = {name: (z[b + f"seed0/{key}/margin"].astype(np.float64), z[b + f"seed0/{key}/err"].astype(bool))
                for name, key in (("s1s2_head_on_s1s2", "full"), ("s1s2_head_on_s1", "optical_missing"),
                                  ("s1s2_head_on_s2", "radar_missing"))}
        for name, path in (("s1_head_on_s1", c1), ("s2_head_on_s2", c2)):
            ex = np.load(path)
            err = ex["err"].astype(bool)
            assert np.array_equal(ex["dec"].astype(int) == y, ~err), (enc, name, "not in exp88's sample order")
            rows[name] = (ex["margin"].astype(np.float64), err)
        for name, (m, e) in rows.items():
            r = R["rows"][name]
            assert abs(e.mean() - r["error_rate"]) < 1e-12, (enc, "china6", name)
            assert abs(np.count_nonzero(m[e] >= thr) / e.sum() - r["confident_share"]) < 1e-12, (enc, "china6", name)
            assert abs(_midrank_auroc(-m, e) - r["margin_auroc"]) < 1e-9, (enc, "china6", name)
        # with a head trained on each input, the optical input is the more useful one on China 6
        assert rows["s2_head_on_s2"][1].mean() < rows["s1_head_on_s1"][1].mean()
    assert "olmoearth_base" in checked, "OlmoEarth Base's export is committed and must be checked"
