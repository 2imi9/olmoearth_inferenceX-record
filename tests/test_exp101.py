"""exp101 (docs/plan/two_step.md): filter, then shake. Written before the page was frozen and before any perturbed
reading at exp99's plots: the view layout; the instability and the rule R(q) on hand-made inputs with known answers;
the record's capture convention; q's tie rule; the dihedral round trip and the patch grid mapping onto itself; the
season builder keeping each group's own month index; the view machinery against an independent loop and against a
pixelwise model (every view with the same months must give V0's logits exactly, clamped or not); the paired bootstrap
on constructed cases; the smoke end to end (the file path on exp98's and exp99's smoke inputs, a planted set where
both predictions hold and a null where neither does); the refusal before freezing; and nothing written carries a
position. No network. Needs rasterio, shapely and pyproj (the geo extra); skipped without them. The GPU path's tensor
mover is tested where torch imports."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp101_two_step as e101          # noqa: E402
import exp98_awf_deployment as e98      # noqa: E402
from oe_inferencex import metrics       # noqa: E402


# ----------------------------------------------------------------------------- the views
def test_the_view_layout_is_the_pages():
    assert e101.N_VIEWS == 320 == 8 * 5 * 2 * 4
    assert e101.view_settings(e101.V0) == (0, 0, 0, 0)
    assert e101.ORIENTATIONS[0] == (0, False) and len(set(e101.ORIENTATIONS)) == 8
    assert e101.season_months(0) == tuple(range(12)) and e101.CONTEXTS[0] == 16 and e101.PHASES[0] == 0
    for v in range(e101.N_VIEWS):
        assert e101.view_index(*e101.view_settings(v)) == v
    assert {k: len(v) for k, v in e101.KINDS.items()} == {"orientation": 7, "time": 4, "context": 1, "phase": 3}
    for kind, views in e101.KINDS.items():                  # one kind moves, the other three stay at V0's setting
        pos = {"orientation": 0, "time": 1, "context": 2, "phase": 3}[kind]
        for v in views:
            s = e101.view_settings(v)
            assert s[pos] != 0 and all(s[i] == 0 for i in range(4) if i != pos)


def test_dihedral_round_trip_is_exact_and_the_patch_grid_maps_onto_itself():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 16, 3, 2))
    seen = set()
    for k, f in e101.ORIENTATIONS:
        t = e101.apply_t(a, k, f, axes=(0, 1))
        assert np.array_equal(e101.invert_t(t, k, f, axes=(0, 1)), a)
        seen.add(t.tobytes())
        g = rng.normal(size=(5, 16, 16, 10))                # a batch of output grids, spatial axes (1, 2)
        assert np.array_equal(e101.invert_t(e101.apply_t(g, k, f, axes=(1, 2)), k, f, axes=(1, 2)), g)
    assert len(seen) == 8
    for S in e101.CONTEXTS:
        r, c = np.mgrid[0:S, 0:S]
        pid = (r // 4) * (S // 4) + c // 4
        for k, f in e101.ORIENTATIONS:
            t = e101.apply_t(pid, k, f, axes=(0, 1))
            blocks = t.reshape(S // 4, 4, S // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)
            assert all(len(set(b.tolist())) == 1 for b in blocks)        # every 4 x 4 block is one patch
            assert sorted(blocks[:, 0].tolist()) == list(range((S // 4) ** 2))


def test_the_season_builder_keeps_each_groups_own_month_index():
    series = np.zeros((5, 5, 12, 12))
    series[..., :, :] = np.arange(12)[:, None]              # each group's values carry its month index
    for t in range(1, 5):
        x, months = e101.season_inputs(series, t)
        assert x.shape == (5, 5, 9, 12) and len(months) == 9
        assert set(months) == set(range(12)) - set(e101.QUARTERS[t - 1])
        assert list(months) == sorted(months)
        assert np.array_equal(x[0, 0, :, 0], np.array(months, dtype=float))
        ts = e101.timestamps(months)
        assert ts.shape == (9, 3) and list(ts[:, 1]) == list(months)
        assert set(ts[:, 0]) == {15} and set(ts[:, 2]) == {2023}            # exp21's legacy timestamps
    x, months = e101.season_inputs(series, 0)
    assert x.shape[2] == 12 and months == tuple(range(12))


def test_crop_plan_puts_v0s_unit_at_8_8_and_clamps_at_the_edge():
    plan, box, clamped = e101.crop_plan(1024, 1024, 500, 600)
    assert not clamped and plan[(0, 0)][2:] == (8, 8)
    for ci, S in enumerate(e101.CONTEXTS):
        for p in e101.PHASES:
            r0, c0, rr, cc = plan[(ci, p)]
            assert (rr, cc) == (S // 2 - p, S // 2 - p) and (r0 + rr, c0 + cc) == (500, 600)
    assert box == (500 - 16, 600 - 16, 500 + 19, 600 + 19)                # 35 x 35 holds every crop
    plan, box, clamped = e101.crop_plan(1024, 1024, 3, 1020)
    assert clamped and box[0] == 0 and box[3] == 1024
    for (ci, p), (r0, c0, rr, cc) in plan.items():
        assert (r0 + rr, c0 + cc) == (3, 1020) and 0 <= r0 and c0 + e101.CONTEXTS[ci] <= 1024


class PixelModel:
    """Logits at a pixel from that pixel's own series only: every view with the same months must give V0's logits."""
    def __init__(self):
        self.A = np.random.default_rng(1).normal(size=(12, 10))

    def normalize(self, raw):
        return (np.asarray(raw, np.float64) / 1000.0).astype(np.float32)

    def pixel_logits(self, x, months):
        w = 1.0 + np.asarray(months, np.float64) / 12.0
        return np.einsum("bijtc,bt->bijc", np.asarray(x, np.float64), w) @ self.A


def _units(rng, places, H=64, W=64):
    out = []
    for r, c in places:
        plan, box, clamped = e101.crop_plan(H, W, r, c)
        raw = rng.integers(0, 3000, size=(H, W, 12, 12)).astype(np.float32)
        out.append(e101.make_unit(raw[box[0]:box[2], box[1]:box[3]], plan, box, clamped))
    return out


def test_every_view_with_the_same_months_reads_v0s_pixel_under_a_pixelwise_model():
    rng = np.random.default_rng(2)
    units = _units(rng, [(32, 32), (2, 61), (60, 1), (17, 45)])
    assert [u["clamped"] for u in units] == [False, True, True, False]
    L, stats = e101.run_views(PixelModel(), units, batch_tokens=5000, log=lambda *a: None)
    assert stats["views"] == 4 * 320 and np.isfinite(L).all()
    for t in range(5):
        vs = [v for v in range(e101.N_VIEWS) if e101.view_settings(v)[1] == t]
        ref = L[:, vs[0]]
        for v in vs:
            assert np.allclose(L[:, v], ref, rtol=0, atol=1e-9), (t, e101.view_settings(v))
    cls, tv = e101.view_readings(L)
    sh, _ = e101.instability(cls, tv, e101.KINDS["orientation"] + e101.KINDS["context"] + e101.KINDS["phase"])
    assert np.array_equal(sh, np.zeros(4))


def test_run_views_agrees_with_one_view_at_a_time():
    rng = np.random.default_rng(3)
    units = _units(rng, [(30, 30), (5, 40)])
    model = e101.StandIn()
    L, _ = e101.run_views(model, units, batch_tokens=3000, log=lambda *a: None)
    for i, u in enumerate(units):
        block = model.normalize(u["raw"])
        for v in rng.choice(e101.N_VIEWS, 25, replace=False):
            x, months, (k, f), (rr, cc) = e101.view_input(block, u["plan"], v)
            g = model.pixel_logits(x[None], np.asarray([months]))[0]
            # the stand-in sums in float32, so one view alone and a batch differ in the last bits only
            assert np.allclose(L[i, v], e101.invert_t(g, k, f, axes=(0, 1))[rr, cc], rtol=0, atol=1e-5)


# ----------------------------------------------------------------------------- the instability and the rule
def test_instability_on_hand_made_classes():
    classes = np.array([[3, 3, 3, 3, 3], [3, 1, 3, 2, 3], [0, 1, 2, 1, 1]])
    tv = np.array([[0, .1, .1, .1, .1], [0, .4, .0, .6, .2], [0, .9, .9, .9, .5]])
    share, mean_tv = e101.instability(classes, tv)
    assert np.allclose(share, [0, 0.5, 1.0]) and np.allclose(mean_tv, [0.1, 0.3, 0.8])
    share, mean_tv = e101.instability(classes, tv, [1, 2])
    assert np.allclose(share, [0, 0.5, 1.0]) and np.allclose(mean_tv, [0.1, 0.2, 0.9])


def test_view_readings_class_over_ten_channels_and_tv_to_v0():
    L = np.zeros((1, 3, 10))
    L[0, :, 9] = 5.0                                        # the untrained channel can be the class (rslearn's argmax)
    L[0, 2, 9], L[0, 2, 2] = 0.0, 5.0
    cls, tv = e101.view_readings(L)
    assert cls.tolist() == [[9, 9, 2]]
    P = e101.softmax(L[0])
    assert tv[0, 0] == 0 and tv[0, 1] == 0 and np.isclose(tv[0, 2], 0.5 * np.abs(P[2] - P[0]).sum())


def test_instability_score_takes_the_share_first_and_breaks_ties_by_distance():
    s = e101.instability_score(np.array([0.5, 0.5, 0.2, 0.9, 0.5]), np.array([0.1, 0.3, 0.9, 0.0, 0.1]))
    assert list(np.argsort(-s, kind="stable")) == [3, 1, 0, 4, 2]
    assert s[0] == s[4]                                     # equal on both keys: a tie


def test_set_aside_takes_the_most_confident_share_in_weight_and_whole_tie_groups():
    u = np.arange(10, dtype=float)                          # unit 0 the most confident
    assert np.flatnonzero(e101.set_aside(u, 0.5, np.ones(10))).tolist() == [0, 1, 2, 3, 4]
    u2 = u.copy()
    u2[4] = u2[5] = 4.5                                     # a tie across the cut: left in, both
    assert np.flatnonzero(e101.set_aside(u2, 0.5, np.ones(10))).tolist() == [0, 1, 2, 3]
    w = np.ones(10)
    w[0] = 4.0                                              # 4 of 13 in weight: q = 0.5 sets aside 0, 1 and 2 (6/13)
    assert np.flatnonzero(e101.set_aside(u, 0.5, w)).tolist() == [0, 1, 2]


def test_the_two_step_order_on_a_known_case():
    u = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05])   # unit 0 the least confident
    share = np.array([0.0, 0.5, 0.5, 0.2, 0.9, 0.0, 0.0, 0.0])
    tv = np.array([0.0, 0.1, 0.1, 0.3, 0.9, 0.0, 0.0, 0.0])
    score, aside = e101.two_step_score(u, share, tv, 0.5, np.ones(8))
    assert np.flatnonzero(aside).tolist() == [4, 5, 6, 7]
    # kept: by share (1 and 2 tie on share and distance: the less confident, 1, first), then 3, then 0; then the units
    # set aside, least confident first, the most unstable of them (4) no earlier for it
    assert list(np.argsort(-score, kind="stable")) == [1, 2, 3, 0, 4, 5, 6, 7]


def test_capture_is_the_records_and_review_inclusion_matches_it():
    rng = np.random.default_rng(4)
    for _ in range(20):
        n = 37
        score = rng.integers(0, 8, n).astype(float)        # many ties
        err = (rng.random(n) < 0.3).astype(float)
        w = rng.choice([0.5, 1.0, 2.0], n)
        for b in e101.BUDGETS:
            inc = e101.review_inclusion(score, w, b)
            assert np.isclose((inc * w).sum(), b * w.sum())
            want = metrics.weighted_capture_at_budget(score, err, w, (b,))[b]
            assert np.isclose(e101.capture(score, err, w, (b,))[b], want)
            if err.sum():
                assert np.isclose((inc * w * err).sum() / (w * err).sum(), want)


def _q_case(errors, unstable, n=20):
    u = np.linspace(2.0, 0.1, n)                            # unit 0 the least confident
    err = np.zeros(n)
    err[list(errors)] = 1.0
    share = np.zeros(n)
    share[list(unstable)] = 0.8
    return u, share, np.zeros(n), err


def test_q_takes_the_most_errors_held_and_ties_go_to_the_larger_q():
    # q 0.5 keeps units 0-9 and q 0.6 units 0-7: both review units 5 and 6 (2 errors); q 0.7 keeps 0-5 (1 error)
    q, rows = e101.choose_q(*_q_case([5, 6], [5, 6]))
    assert [round(rows[x]["errors_held"], 9) for x in e101.Q_GRID] == [2.0, 2.0, 1.0, 0.0]
    assert q == 0.6
    q, rows = e101.choose_q(*_q_case([8, 9], [8, 9]))      # only q 0.5 keeps them
    assert rows[0.5]["errors_held"] == 2.0 and rows[0.6]["errors_held"] == 0.0 and q == 0.5
    q, _ = e101.choose_q(*_q_case([], []))                  # no error: every q ties, the largest is taken
    assert q == 0.8


# ----------------------------------------------------------------------------- the bootstrap and the grades
def test_the_stratified_resampler_keeps_each_stratums_size():
    st = np.array(["a"] * 7 + ["b"] * 3)
    draw = e101.stratified_resampler(st, 10)
    rng = np.random.default_rng(5)
    for _ in range(50):
        idx = draw(rng)
        assert (st[idx] == "a").sum() == 7 and (st[idx] == "b").sum() == 3
    assert len(e101.stratified_resampler(None, 4)(rng)) == 4


def test_the_paired_bootstrap_on_constructed_cases():
    n = 60
    u = np.linspace(2.0, 0.1, n)
    strata = np.array(["kenya", "tanzania"] * (n // 2))
    w = np.where(strata == "kenya", 1.0, 2.0)
    err = np.zeros(n)
    err[[0, 3, 8, 15, 20, 22, 25, 40]] = 1.0
    # no instability anywhere: within the kept part R(q) is confidence's order, so the review is confidence's
    zero = np.zeros(n)
    bt = e101.paired_bootstrap(u, zero, zero, err, w, strata, 0.5, n_boot=100, seed=1)
    for b in e101.BUDGETS:
        iv = bt["review_difference"][e101.bkey(b)]
        assert iv["n_resamples"] == 100 and abs(iv["lo90"]) < 1e-12 and abs(iv["hi90"]) < 1e-12
    # instability exactly on the errors of the kept half: R(q) reviews them first, confidence does not
    share = np.where((err > 0) & (np.arange(n) < 30), 1.0, 0.0)
    rec = e101.ranking_record(u, share, share * 0.5, err, w, strata, 0.5, n_boot=200, seed=2)
    assert rec["review"]["0.1"]["difference"] > 0 and rec["bootstrap"]["review_difference"]["0.1"]["lo90"] > 0
    assert rec["not_set_aside"]["auroc_instability"] == pytest.approx(1.0)
    assert rec["bootstrap"]["auroc_difference"]["lo90"] > 0


def _rec(diff, lo, adiff, alo):
    return {"review": {"0.1": {"difference": diff, "two_step": 0.3, "confidence": 0.3 - diff}},
            "not_set_aside": {"difference": adiff, "auroc_instability": 0.7, "auroc_confidence": 0.7 - adiff},
            "bootstrap": {"review_difference": {"0.1": {"lo90": lo, "hi90": 1.0}},
                          "auroc_difference": {"lo90": alo, "hi90": 1.0}}}


def test_grades_at_their_bars_and_floors():
    g = e101.grade(_rec(0.06, 0.01, 0.05, 0.001), 271, 91, 180)
    assert g["P1"]["holds"] and g["P2"]["holds"] and e101.reading(g) == e101.READINGS["both"]
    g = e101.grade(_rec(0.049, 0.01, 0.05, 0.01), 271, 91, 180)        # below the 5-point bar
    assert not g["P1"]["holds"] and g["P2"]["holds"] and e101.reading(g) == e101.READINGS["P2 only"]
    g = e101.grade(_rec(0.10, -0.01, 0.05, -0.01), 271, 91, 180)       # intervals reach 0
    assert not g["P1"]["holds"] and not g["P2"]["holds"] and e101.reading(g) == e101.READINGS["neither"]
    for n, k, c in ((249, 91, 158), (271, 19, 252), (271, 252, 19)):
        g = e101.grade(_rec(0.2, 0.1, 0.2, 0.1), n, k, c)
        assert not g["P1"]["graded"] and not g["P2"]["graded"] and e101.reading(g) == e101.READINGS["not graded"]


def test_recorded_counts_are_exp98s_and_exp99s():
    p98, p99 = (os.path.join(ROOT, "exp", "out", f"exp{n}_summary.json") for n in (98, 99))
    if not (os.path.exists(p98) and os.path.exists(p99)):
        pytest.skip("exp98 or exp99 not recorded")
    g = json.load(open(p98))["graded_validation"]
    assert (g["n"], g["accuracy"]["n_errors"]) == (e101.RECORDED["F"]["n"], e101.RECORDED["F"]["errors"])
    r = json.load(open(p99))["rules"]
    E = e101.RECORDED["E"]
    assert (r["strict"]["n"], r["strict"]["errors"], r["lenient"]["errors"]) == (E["n"], E["strict"], E["lenient"])
    assert r["strict"]["ranking"]["weighted"]["capture"]["0.1"] == E["confidence_capture_0.1_strict"]


# ----------------------------------------------------------------------------- the smoke, end to end
@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("exp101"))


@pytest.fixture(scope="module")
def smoke(out_dir):
    return e101.smoke(out_dir=out_dir, log=lambda *a: None)


def test_the_smoke_runs_the_file_path(smoke, out_dir):
    inv = smoke["inventory"]
    assert inv["F"]["n_units"] == 40 and inv["F"]["read"] == 40 and inv["F"]["in_two_or_more_windows"] > 0
    assert inv["E"]["n_units"] == 22 and inv["E"]["read"] == 21                 # one plot's window lacks a group
    assert smoke["fit"]["q"] in e101.Q_GRID and smoke["fit_record_check"]["q_matches"] is True
    assert smoke["test"]["units"]["n_with_views"] == 21 and smoke["inputs_E"]["only_in_windows_without_12_groups"] == 1
    w = smoke["test"]["weights"]["per_plot"]
    assert set(w) == {"kenya", "tanzania"} and all(v > 0 for v in w.values())
    for p in ("P1", "P2"):
        assert {"rule", "value", "interval"} <= set(smoke["prereg"][p])
    for name in ("exp101_fit_smoke.json", "exp101_summary_smoke.json", "exp101_units_smoke.npz"):
        assert os.path.exists(os.path.join(out_dir, name))
    z = np.load(os.path.join(out_dir, "exp101_units_smoke.npz"))
    assert z["view_class"].shape == (62, 320) and int(z["views_read"].sum()) == 61
    ok = z["views_read"]
    assert np.array_equal(z["v0_class"][ok], z["view_class"][ok, 0])
    assert np.allclose(z["view_tv"][ok, 0], 0.0)
    assert np.allclose(z["instability_share"][ok], (z["view_class"][ok, 1:] != z["view_class"][ok, :1]).mean(1))


def test_the_planted_effect_is_found_and_the_null_is_not(smoke):
    ps = smoke["planted_smoke"]
    assert ps["q"] in e101.Q_GRID
    planted, null = ps["planted"]["prereg"], ps["null"]["prereg"]
    assert planted["P1"]["graded"] and planted["P1"]["holds"] and planted["P2"]["holds"]
    assert null["P1"]["graded"] and not null["P1"]["holds"] and not null["P2"]["holds"]
    assert ps["planted"]["auroc_instability_all_units"] > 0.7 > ps["null"]["auroc_instability_all_units"]


def test_the_report_only_parts_are_there(smoke):
    t = smoke["test"]
    assert set(t["kinds_reported"]) == set(e101.KINDS) and set(t["dose_reported"]) == {str(k) for k in e101.DOSE_K}
    assert t["dose_reported"]["319"]["review_difference_0.1"]["min"] == t["dose_reported"]["319"][
        "review_difference_0.1"]["max"]                                              # all 319 views: one subset
    assert set(t["strict"]["review"]) == {"0.05", "0.1", "0.2"} and "lenient_reported" in t
    assert t["grass_shrub_reported"]["n"] == t["units"]["n_errors"] - t["units"]["lenient_errors"]
    assert set(smoke["fit"]["per_q"]) == {str(q) for q in e101.Q_GRID}


def test_nothing_written_carries_a_position_or_an_identifier(smoke, out_dir, tmp_path):
    import exp99_transfer as e99
    import timesync_awf_crosswalk as tsx
    e98.check_no_coordinates(smoke)
    texts = [open(os.path.join(out_dir, n)).read() for n in ("exp101_summary_smoke.json", "exp101_fit_smoke.json")]
    paths = e99.make_smoke_inputs(str(tmp_path))
    plots = tsx.read_plots(paths["timesync"], 2017)
    coords = {round(v, 6) for v in np.r_[plots["lon"], plots["lat"]]}

    def numbers(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from numbers(v)
        elif isinstance(o, list):
            for v in o:
                yield from numbers(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield float(o)
    for text in texts:
        assert not coords & {round(v, 6) for v in numbers(json.loads(text))}
        assert "kenya:" not in text and "tanzania:" not in text and "_point_" not in text and "windows/" not in text
    z = np.load(os.path.join(out_dir, "exp101_units_smoke.npz"))
    assert not {k.lower() for k in z.files} & (e98.FORBIDDEN_KEYS | {"pid", "plotid", "name", "names", "window"})


def test_the_inventory_from_the_command_line_counts_and_reads_no_outcome(tmp_path, monkeypatch, capsys):
    import functools
    a, _ = e101.make_smoke_inputs(str(tmp_path / "in"))
    # the command line reads exp99's selection with its pins; the smoke's synthetic sample needs exp99's smoke settings
    monkeypatch.setattr(e101, "load_test_units", functools.partial(e101.load_test_units, **e101.SMOKE_KW99))

    def no_outcome(*args, **kw):
        raise AssertionError("the inventory read a class or a probability as an outcome")
    monkeypatch.setattr(e98, "readings", no_outcome)
    out = tmp_path / "out"
    rc = e101.main(["--inventory", "--exp98-scores", *a.exp98_scores, "--labels", a.labels, "--deploy-2023",
                    *a.deploy_2023, "--exp99-scores", *a.exp99_scores, "--timesync", a.timesync, "--request-geometry",
                    a.request_geometry, "--deploy-2017", *a.deploy_2017, "--out-dir", str(out)])
    assert rc == 0
    rec = json.load(open(out / "exp101_inventory.json"))
    assert (rec["F"]["n_units"], rec["F"]["read"], rec["E"]["n_units"], rec["E"]["read"]) == (40, 40, 22, 21)
    assert rec["E"]["selected"] == 24 and rec["F"]["failed_reads"] == {} and rec["E"]["failed_reads"] == {}
    e98.check_no_coordinates(rec)
    assert sorted(os.listdir(out)) == ["exp101_inventory.json"]


def test_the_test_is_refused_while_the_page_is_a_draft(tmp_path, capsys, monkeypatch):
    assert e101.prereg_status("# exp101\n\n**Status: draft, written 10 October 2026.**\n") == "draft"
    assert e101.prereg_status("# exp101\n\n**Status: frozen 11 October 2026.**\n") == "frozen"
    monkeypatch.setattr(e101, "prereg_status", lambda text=None: "draft")
    out = tmp_path / "out"
    rc = e101.main(["--test", "--exp98-scores", str(tmp_path), "--labels", str(tmp_path), "--exp99-scores",
                    str(tmp_path), "--request-geometry", str(tmp_path / "g.geojson"), "--deploy-2017", str(tmp_path),
                    "--boundaries", str(tmp_path / "b.geojson"), "--out-dir", str(out)])
    assert rc == 3 and "refused" in capsys.readouterr().out
    assert not out.exists()


def test_the_stray_tensor_mover():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.idx = torch.tensor([0, 1])                 # a plain attribute: model.to(device) leaves it behind
            self.many = [torch.zeros(1), torch.ones(1)]
    m = M().to("meta")
    assert e101.devices_off(m, "meta") == ["idx"]
    moved = e101.move_stray_tensors(m, "meta")
    assert sorted(moved) == ["idx", "many[0]", "many[1]"]
    assert e101.devices_off(m, "meta") == [] and m.idx.device.type == "meta"


def _fake_windows(monkeypatch, cands):
    wins = [{"dir": d, "crs": "EPSG:32737", "res": (10.0, -10.0), "b": [0, 0, 64, 64], "complete": True}
            for d in ("a", "b")]
    monkeypatch.setattr(e101, "deployed_windows", lambda roots: wins)
    monkeypatch.setattr(e101, "locate_units", lambda positions, windows: cands)
    monkeypatch.setattr(e101, "read_window_block", lambda win, box: np.zeros(
        (box[2] - box[0], box[3] - box[1], e101.N_GROUPS, len(e101.OLMO_BANDS)), np.float32))


def test_a_unit_takes_a_later_window_rather_than_a_clamped_crop(monkeypatch):
    """Choice c: a unit at (2, 2) of one window and at (32, 32) of another reads the second, where no crop is clamped;
    a unit clamped in every complete window keeps the first and is counted."""
    _fake_windows(monkeypatch, [[(0, 2, 2), (1, 32, 32)], [(0, 2, 2), (1, 61, 61)], [(0, 32, 32), (1, 2, 2)],
                                [(0, 30, 30)]])
    S = {"name": "E", "positions": [{}, {}, {}, {}]}
    rec = e101.attach_blocks(S, roots=["root"], log=lambda *a: None)
    assert [u["clamped"] for u in S["units"]] == [False, True, False, False]
    # only the first unit moved to a later window; the third's first window was already unclamped (an inventory on
    # the cluster once counted every unclamped pick as a move, through an identity test on a fresh tuple)
    assert rec["took_a_later_window_to_avoid_clamping"] == 1 and rec["clamped_at_a_window_edge"] == 1


def test_f_without_the_2023_dataset_is_refused_unless_the_fallback_is_asked_for():
    with pytest.raises(ValueError, match="no dataset roots"):
        e101.attach_blocks({"name": "F", "positions": [{}], "points": [{}]}, roots=None, label_fallback=False,
                           log=lambda *a: None)


# ----------------------------------------------------------------------------- the recorded run, by a second route
# The claim ledger's crosschecks for exp101: the review shares, the instability and the set-aside counts recomputed
# from the committed per-unit file with plain numpy (a tie-aware weighted review written here, not the script's).
UNITS = os.path.join(ROOT, "exp", "out", "exp101_units.npz")
SUMMARY = os.path.join(ROOT, "exp", "out", "exp101_summary.json")


def _review(score, w, err, b):
    """Share of the weighted errors in the top b of the weight by score (higher first); a tied group on the cut counts
    in proportion."""
    order = np.argsort(-score, kind="stable")
    s, w2, e = score[order], w[order], err[order]
    budget, used, got, i = b * w.sum(), 0.0, 0.0, 0
    while i < len(s) and used < budget - 1e-12:
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        gw, ge = w2[i:j + 1].sum(), (w2[i:j + 1] * e[i:j + 1]).sum()
        f = min(1.0, (budget - used) / gw)
        got, used, i = got + f * ge, used + f * gw, j + 1
    return got / (w * err).sum()


@pytest.mark.skipif(not os.path.exists(UNITS), reason="exp101 not recorded")
def test_exp101_review_and_instability_by_a_second_route():
    z = np.load(UNITS)
    rec = json.load(open(SUMMARY))
    assert rec["prereg_status"] == "frozen" and rec["dirty"] is False and rec["commit"].startswith("a1b878e")
    E, F = z["set"] == 1, z["set"] == 0
    assert (int(E.sum()), int(z["wrong"][E].sum()), int(F.sum()), int(z["wrong"][F].sum())) == (271, 91, 259, 33)
    share = (z["view_class"][:, 1:] != z["v0_class"][:, None]).mean(1)
    assert np.allclose(share, z["instability_share"])
    w, err = z["weight"][E], z["wrong"][E]
    conf, two = _review(z["suspicion"][E], w, err, 0.1), _review(z["two_step_rank"][E], w, err, 0.1)
    assert round(conf, 4) == 0.1787 and round(two, 4) == 0.1348
    P1 = rec["prereg"]["P1"]
    assert P1["holds"] is False and abs((two - conf) - P1["value"]) < 1e-9
    assert (int(z["set_aside_at_q"][E].sum()), int(err[z["set_aside_at_q"][E]].sum())) == (159, 38)
    assert [round(share[E][err > 0].mean(), 3), round(share[E][err == 0].mean(), 3)] == [0.174, 0.097]
    wf = z["wrong"][F]
    assert [round(share[F][wf > 0].mean(), 3), round(share[F][wf == 0].mean(), 3)] == [0.263, 0.035]
    assert _review(z["two_step_rank"][F], z["weight"][F], wf, 0.1) == pytest.approx(0.512, abs=1e-3)
    assert _review(z["suspicion"][F], z["weight"][F], wf, 0.1) == pytest.approx(0.424, abs=1e-3)
    # the plots kept, the grass/shrub plots (STRICT wrong, LENIENT right) and the AUROCs among the kept plots
    keep, len_ = ~z["set_aside_at_q"][E], z["wrong_lenient"][E]
    sE = share[E]
    assert [round(sE[keep & (err > 0)].mean(), 3), round(sE[keep & (err == 0)].mean(), 3)] == [0.264, 0.224]
    gs, other = (err == 1) & (len_ == 0), (err == 1) & (len_ == 1)
    assert (int(gs.sum()), int(other.sum())) == (54, 37)
    assert [round(sE[gs].mean(), 3), round(sE[other].mean(), 3)] == [0.098, 0.285]
    key = sE * 1e6 + z["instability_mean_tv"][E]           # the share, ties broken by the mean distance to V0

    def wauc(score, e, m):
        pos, neg = m & (e == 1), m & (e == 0)
        num = sum(wi * (w[neg][score[neg] < si].sum() + 0.5 * w[neg][score[neg] == si].sum())
                  for si, wi in zip(score[pos], w[pos]))
        return num / (w[pos].sum() * w[neg].sum())
    assert [round(wauc(key, err, keep), 3), round(wauc(z["suspicion"][E], err, keep), 3)] == [0.541, 0.528]
    assert [round(wauc(key, len_, keep), 3), round(wauc(z["suspicion"][E], len_, keep), 3)] == [0.676, 0.518]
