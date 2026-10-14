"""exp89's grades, guards and per-arm rules before the preregistration is frozen and before any real run
(docs/plan/finetuned_checkpoints.md, amended 1 October 2026). Synthetic units with known answers go through every
measure and every grade: in one case the model's confidence beats an informative control on every prediction, in one
the control wins, in one nothing is informative. Arms A and F are report-only: their cases go through every measure
and must carry no verdict anywhere, in what is written or printed. The grades are also checked on planted numbers at
their thresholds. The guards are checked without a model: no full run before the page is frozen and the gate has
passed (or, for a report-only arm, an alignment check is recorded), and a gate file that holds counts and accuracy
only. Arm M is reported as not run until Ai2's split is pinned. The channel, fill and label rules of each arm, the
readers on rslearn layouts written to disk (arm F's and arm A's included), arm F's tar filter and pin, and the
extraction manifest that catches the scratch purge are checked here too. K3's fit and the replica's forward pass run
only where torch is installed."""
import argparse
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp89_finetuned_checkpoints as e89   # noqa: E402  (module level is torch-free)
from oe_inferencex import metrics, stats    # noqa: E402

FROZEN = "# exp89\n\n**Status: frozen on 2 October 2026, before any run.**\n"
DRAFT = "# exp89\n\n**Status: DRAFT, not frozen.** Written 1 October 2026.\n"
PINNED_MANGROVE = e89.ARMS["mangrove"]["model_sha256"]
PINNED_FLD = e89.ARMS["fld"]["model_sha256"]
MANGROVE_SPLIT_AS_SHIPPED = e89.ARMS["mangrove"]["split_source"]


@pytest.fixture(autouse=True)
def _ledger_beside_the_gate_file(monkeypatch):
    """The gate's ledger defaults to the gate file's directory; a cluster setting in the shell must not leak in."""
    monkeypatch.delenv("E89_GATE_LEDGER", raising=False)


@pytest.fixture(autouse=True)
def _mangrove_split_pinned(monkeypatch):
    """The guard tests below drive arm M's gate and run, which wait for Ai2's split; here a split is taken as pinned.
    test_arm_m_waits_for_ai2s_split checks the shipped value and the not-run path."""
    monkeypatch.setitem(e89.ARMS["mangrove"], "split_source", "a split pinned for the tests")


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return e89.smoke(out_dir=str(tmp_path_factory.mktemp("exp89")))


# ----------------------------------------------------------------------------- S1: every grade as designed
def test_every_case_grades_as_designed(smoke):
    for case, want in e89.SMOKE_EXPECTED.items():
        g = smoke["cases"][case]["prereg"]
        assert {p: g[p]["holds"] for p in want} == want, case
        assert g["complete"] is True, case


def test_each_prediction_both_passes_and_fails_somewhere(smoke):
    for p in ("P1", "P2", "P3", "P4", "P5"):
        seen = {smoke["cases"][c]["prereg"][p]["holds"] for c in e89.SMOKE_EXPECTED}
        assert {True, False} <= seen, p


def test_the_grades_do_not_rest_on_their_thresholds(smoke):
    beats, wins, weak = (smoke["cases"][c]["prereg"] for c in ("beats", "control_wins", "weak"))
    assert beats["P1"]["capture_10"] >= 0.9 and weak["P1"]["capture_10"] <= 0.2
    assert beats["P3"]["capture_difference_10"] >= 0.5 and beats["P3"]["lower_one_sided_95"] > 0.3
    assert wins["P3"]["capture_difference_10"] <= -0.3, "the control wins by far"
    assert weak["P2"]["auroc"] <= 0.60
    assert beats["P4"]["width_ratio_300"] <= 0.70 and weak["P4"]["width_ratio_300"] >= 1.0
    assert min(beats["P4"]["coverage"].values()) >= 0.95, "2,000 draws: the coverage is not a coin flip at 0.93"
    assert beats["P5"]["lead_over_k3a"] >= 0.2 and wins["P5"]["lead_over_k3a"] <= -0.05
    assert weak["P5"]["median_coverage"] == 0.0


@pytest.mark.parametrize("case", e89.SMOKE_REPORT_ONLY)
def test_the_report_only_arms_carry_no_verdict(smoke, case):
    """Arms A and F: every measure is computed and reported with its interval, and nothing says pass, fail or holds,
    though the synthetic numbers would clear some of the page's thresholds."""
    res = smoke["cases"][case]
    assert res["report_only"] is True and "prereg" not in res
    assert e89.verdicts_in(res) == []
    rep = res["reported"]
    assert "report-only" in rep["why"] and "none is graded" in rep["why"]
    assert set(rep["review_order"]["capture"]) == {"0.05", "0.1", "0.2"}
    lead = rep["confidence_minus_best_control"]["0.1"]["bootstrap"]
    assert lead["lo95"] <= lead["hi95"], "the interval is reported"
    assert rep["review_order"]["capture"]["0.1"] >= 0.2, "a case that would clear P1's bar, and is still not graded"
    assert set(rep["certify_best_coverage"]) == {f"{o}/{a:g}" for o in ("confidence", "k3a")
                                                 for a in rep["alphas"]}
    if case == "awf_report":
        assert res["estimate"]["budgets_run"] == [300] and rep["estimate"]["300"]["run"] is True
        assert rep["certify"]["300/0.05"]["violation_rate"] is not None
    else:
        assert res["estimate"]["budgets_run"] == [] and rep["certify"]["run"] is False


def test_the_verdict_finder_finds_verdicts():
    """verdicts_in is what keeps the report-only outputs clean, so it must catch a verdict where one is."""
    assert e89.verdicts_in({"a": {"holds": None}}) == ["/a/holds"]
    assert e89.verdicts_in({"x": [{"note": "the gate: PASS"}]}) == ["/x[0]/note='the gate: PASS'"]
    assert e89.verdicts_in({"alignment": "replica not aligned", "aligned": False, "capture": 0.4}) == []


def test_planted_grades_at_their_thresholds(smoke):
    assert all(v is True for v in smoke["planted"].values()), smoke["planted"]


def test_the_graded_confidence_is_the_package_top1_over_the_trained_channels():
    """The untrained channel counts in the argmax (an error when it wins) and never in the confidence."""
    L = np.array([[5.0, 1.0, 0.5, 0.0],      # Mangrove: channel 0 (untrained) wins
                  [-9.0, 2.0, 1.0, 0.0],
                  [-9.0, 0.0, 0.0, 3.0]])
    rd = e89.readings(L, "mangrove")
    assert rd["pred"].tolist() == [0, 1, 3] and rd["n_pred_untrained"] == 1
    z = L[:, 1:] - L[:, 1:].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert np.allclose(rd["p1"], p.max(1), atol=1e-12)
    srt = np.sort(L[:, 1:], axis=1)
    assert np.allclose(rd["margin"], srt[:, -1] - srt[:, -2])
    assert rd["n_untrained_in_top2"] == 1
    with pytest.raises(ValueError):
        e89.readings(L[:, :3], "mangrove")


@pytest.mark.parametrize("arm,n_out,trained,untrained", [("mangrove", 4, (1, 2, 3), (0,)),
                                                         ("nandi", 11, tuple(range(10)), (10,)),
                                                         ("awf", 10, tuple(range(9)), (9,)),
                                                         ("fld", 10, tuple(range(10)), ())])
def test_each_arms_channels(arm, n_out, trained, untrained):
    A = e89.ARMS[arm]
    assert (A["n_out"], A["trained"], A["untrained"]) == (n_out, trained, untrained)
    assert set(A["trained"]) | set(A["untrained"]) == set(range(n_out))


def test_rslearn_centre_pad_and_the_mangrove_label_rule():
    a = np.arange(16).reshape(4, 4)
    assert e89.pad_center(a, 2).tolist() == [[5, 6], [9, 10]], "a 4x4 window is cropped to its centre 2x2"
    # rslearn: extra = 2 - 3 = -1, before = -1 // 2 = -1, after = 0, so the first row and column go
    b = np.arange(9).reshape(3, 3)
    assert e89.pad_center(b, 2).tolist() == [[4, 5], [7, 8]], "an odd crop drops the first row and column"
    # extra = 1, before = 0, after = 1: the padding goes after
    c = np.ones((1, 1), int)
    assert e89.pad_center(c, 2).tolist() == [[1, 0], [0, 0]], "an odd pad goes after"
    stack = np.arange(4 * 4 * 3).reshape(3, 4, 4)
    assert e89.pad_center(stack, 2).shape == (3, 2, 2)
    assert e89.mangrove_unit_label(np.full((2, 2), 2)) == (2, None)
    assert e89.mangrove_unit_label(np.array([[1, 1], [1, 0]]))[0] is None
    assert e89.mangrove_unit_label(np.array([[1, 1], [1, 3]]))[1] == "label pixels disagree"
    assert e89.mangrove_unit_label(np.full((2, 2), 4))[0] is None


def test_the_single_pixel_rule_and_fill_values():
    lab = np.full((5, 5), 10)
    lab[2, 3] = 7
    assert e89.single_pixel_label(lab, 10, e89.ARMS["nandi"]["trained"]) == ((2, 3, 7), None)
    lab[0, 0] = 1
    assert e89.single_pixel_label(lab, 10, e89.ARMS["nandi"]["trained"])[1] == "2 labelled pixels"
    awf = np.full((5, 5), 9)
    awf[1, 1] = 8
    assert e89.single_pixel_label(awf, 9, e89.ARMS["awf"]["trained"])[0] == (1, 1, 8)
    assert e89.single_pixel_label(np.full((3, 3), 9), 9, e89.ARMS["awf"]["trained"])[1] == "0 labelled pixels"


def test_exp21_crop_rule_puts_the_label_at_row_8_and_clamps():
    assert e89.crop_at(63, 63, 30, 40, 16, 0) == (22, 32, 8, 8)
    assert e89.crop_at(63, 63, 30, 40, 16, 3) == (25, 35, 5, 5)
    assert e89.crop_at(63, 63, 2, 61, 16, 0) == (0, 47, 2, 14)


# ----------------------------------------------------------------------------- controls
def _series(months):
    """(1, 12, 12) raw series from a list of {band: value} per month (missing bands 0)."""
    s = np.zeros((1, 12, 12))
    for t, d in enumerate(months):
        for b, v in d.items():
            s[0, t, e89.OLMO_BANDS.index(b)] = v
    return s


def test_mangrove_index_controls_and_the_cloud_count_by_hand():
    months = []
    for t in range(12):
        if t in (3, 4):
            months.append({})                                            # two empty months
        elif t < 6:
            months.append({"B02": 500, "B03": 900, "B11": 300, "B04": 400, "B08": 2000})     # MNDWI > 0
        else:
            months.append({"B02": 2500 if t == 11 else 500, "B03": 300, "B11": 900, "B04": 400, "B08": 1200})
    k = e89.index_controls("mangrove", _series(months))
    f = 4 / 10                                                          # 4 of the 10 valid months have MNDWI > 0
    assert k["k4_inundation_ambiguity"][0] == pytest.approx(min(f, 1 - f))
    mndwi = [(900 - 300) / 1200] * 4 + [(300 - 900) / 1200] * 6
    assert k["k4_mndwi_near_zero"][0] == pytest.approx(-abs(np.median(mndwi)))
    ndvi = [(2000 - 400) / 2400] * 4 + [(1200 - 400) / 1600] * 6
    assert k["k4_ndvi_month_std"][0] == pytest.approx(np.std(ndvi))
    assert k["k5_cloud_months"][0] == 3, "two empty months and one with B02 above 0.2 reflectance"


def test_nandi_controls_use_exp21s_ndvi_and_the_3x3_block():
    s = _series([{"B04": 400, "B08": 2000 + 100 * t} for t in range(12)])
    nd = (2000 + 100 * np.arange(12) - 400) / (2000 + 100 * np.arange(12) + 400)
    block = np.full((1, 3, 3, 12), np.nan)
    block[0, 1:, 1:] = np.arange(4)[:, None, None].reshape(2, 2, 1) * np.ones(12)       # a block clipped at an edge
    k = e89.index_controls("nandi", s, block)
    assert k["k4_ndvi_temporal_std"][0] == pytest.approx(np.std(nd))
    assert k["k4_ndvi_3x3_std"][0] == pytest.approx(np.std([0, 1, 2, 3])), "read over the pixels it has"


def test_k3_features_fill_empty_months_with_the_units_median():
    s = _series([{} if t == 5 else {"B02": 100 * (t + 1), "B04": 300, "B08": 900} for t in range(12)])
    x = e89.k3_features(s, lambda a: a / 1000.0, "nandi")
    assert x.shape == (1, 12 * 13 + 1)
    per = x[0, :-1].reshape(12, 13)
    b02 = [100 * (t + 1) / 1000 for t in range(12) if t != 5]
    assert per[5, e89.OLMO_BANDS.index("B02")] == pytest.approx(np.median(b02))
    assert per[5, 12] == pytest.approx(0.5), "NDVI of the empty month is the median NDVI, (900 - 300) / 1200"
    assert x[0, -1] == 1, "the count of empty months"
    assert e89.k3_features(s, lambda a: a, "mangrove").shape == (1, 12 * 14 + 1), "Mangrove adds MNDWI"


def test_k2_and_k3_signals():
    k2, freq = e89.class_rarity(np.array([1, 2, 0]), np.array([1, 1, 1, 2]), (1, 2, 3))
    assert freq == {1: 0.75, 2: 0.25, 3: 0.0} and k2.tolist() == [-0.75, -0.25, -0.0]
    prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    sig, cls = e89.k3_signals(prob, (1, 2, 3), np.array([1, 2]))
    assert cls.tolist() == [1, 3]
    assert sig["k3a_no_encoder_uncertainty"] == pytest.approx([0.3, 0.4])
    assert sig["k3b_no_encoder_disagreement"].tolist() == [0.0, 1.0]


def test_k3_fit_is_deterministic_at_seed_89():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((700, 15)), rng.integers(1, 4, 700)
    x[y == 2] += 1.0
    p_a, fit_a = e89.fit_no_encoder_classifier(x, y, (1, 2, 3), x[:60])
    p_b, fit_b = e89.fit_no_encoder_classifier(x, y, (1, 2, 3), x[:60])
    assert np.array_equal(p_a, p_b) and fit_a == fit_b
    assert p_a.shape == (60, 3) and np.allclose(p_a.sum(1), 1)
    assert fit_a["loss_last_epoch"] < fit_a["loss_first_epoch"]
    p_c, _ = e89.fit_no_encoder_classifier(x, y, (1, 2, 3), x[:60], seed=90)
    assert not np.array_equal(p_a, p_c), "the seed is what fixes the fit"


# ----------------------------------------------------------------------------- the statistics are the package's
def test_the_capture_bootstrap_draws_the_aurc_bootstraps_resamples():
    """A manual loop drawing exactly as stats.cluster_bootstrap_difference does reproduces both bootstraps."""
    rng = np.random.default_rng(5)
    n = 400
    err = (rng.random(n) < 0.08).astype(float)
    a = rng.random(n) + 0.8 * err
    b = rng.random(n) + 0.3 * err
    cl = rng.integers(0, 12, n)
    got = e89.capture_bootstrap(a, b, err, cl, 0.10, n_boot=300, seed=89)
    lo, hi, p = stats.cluster_bootstrap_difference(a, b, err, cl, n_boot=300, seed=89)
    r = np.random.default_rng(89)
    ids = np.unique(cl)
    cap, aurc = [], []
    for _ in range(300):
        sel = np.concatenate([np.flatnonzero(cl == c) for c in r.choice(ids, size=len(ids), replace=True)])
        if err[sel].sum() == 0:
            continue
        cap.append(metrics.capture_at_budget_expected(a[sel], err[sel], (0.10,))[0.10]
                   - metrics.capture_at_budget_expected(b[sel], err[sel], (0.10,))[0.10])
        aurc.append(metrics.aurc_expected(a[sel], err[sel]) - metrics.aurc_expected(b[sel], err[sel]))
    assert got["n_resamples"] == len(cap)
    assert got["lower_one_sided_95"] == pytest.approx(np.percentile(cap, 5))
    assert (lo, hi) == pytest.approx((np.percentile(aurc, 2.5), np.percentile(aurc, 97.5)))


def test_the_estimate_draws_are_seeded_per_arm_design_and_budget():
    s = {(a, d, B): tuple(e89.draw_seeds(a, d, B, 3)) for a in e89.ARM_INDEX for d in e89.DESIGN_INDEX
         for B in e89.EST_BUDGETS}
    assert len(set(s.values())) == len(s)
    assert tuple(np.random.SeedSequence([89, 0, 1, 300]).generate_state(3)) == s[("mangrove", "confidence", 300)]


def test_the_certificate_reads_violation_and_coverage_from_the_true_zone():
    rng = np.random.default_rng(1)
    N = 3000
    err = np.zeros(N)
    err[rng.choice(N, 90, replace=False)] = 1                     # 3%: the whole map is above alpha = 2%
    score = rng.random(N) - err                                     # errors least trusted
    draws = {300: np.stack([rng.choice(N, 300, replace=False) for _ in range(40)])}
    out = e89.certify_study(err, {"confidence": score}, draws, (0.02,))
    c = out["confidence/prefix/300/0.02"]
    assert c["draws"] == 40 and 0 <= c["violation_rate"] <= 0.12
    assert c["best_coverage_possible"] == 0.95, "the 2,850 most trusted windows hold no error; all 3,000 hold 3%"
    assert c["min_labels_to_certify"] == 114 and "confidence/bonferroni/300/0.02" in out


def test_the_certificate_counts_violations_from_the_true_zone_rate(monkeypatch):
    """certify_zone is replaced by a script of outcomes, one per draw, so every count is known by hand: a clean zone, a
    zone at exactly alpha, one between alpha and 2 alpha, one far above, a draw that certifies nothing and a draw the
    review-set guard refuses. A violation is a certified zone whose true rate exceeds alpha, and only that."""
    N, alpha = 1000, 0.02
    err = np.zeros(N)
    err[:50] = 1
    z_clean = np.arange(100, 400)                                        # 0 of 300
    z_at = np.r_[np.arange(0, 6), np.arange(100, 394)]                   # 6 of 300 = alpha: not a violation
    z_slight = np.r_[np.arange(0, 9), np.arange(100, 400)]               # 9 of 309 = 0.029: between alpha and 2 alpha
    z_bad = np.r_[np.arange(0, 50), np.arange(100, 150)]                 # 50 of 100
    script = {0: (0.30, z_clean), 1: (0.30, z_at), 2: (0.31, z_slight), 3: (0.10, z_bad), 4: None, 5: "refuse"}

    def fake_certify(score, idx, wrong, a, delta, rule, cut=None):
        assert a == alpha and delta == e89.DELTA
        out = script[int(idx[0])]
        if out == "refuse":
            raise ValueError("the review set is not a random sample of the map: it looks enriched")
        if out is None:
            return {"coverage": None, "zone_indices_in_order": np.array([], int)}
        return {"coverage": out[0], "zone_indices_in_order": out[1]}
    monkeypatch.setattr(e89.est, "certify_zone", fake_certify)
    draws = {300: np.stack([np.r_[r, np.arange(500, 799)] for r in range(6)])}
    res = e89.certify_study(err, {"confidence": np.linspace(1, 0, N)}, draws, (alpha,))
    for rule in ("prefix", "bonferroni"):
        c = res[f"confidence/{rule}/300/0.02"]
        assert c["draws"] == 6
        assert c["violation_rate"] == pytest.approx(2 / 6), "the 0.029 zone and the 0.5 zone, not the one at alpha"
        assert c["median_coverage"] == pytest.approx(np.median([0.30, 0.30, 0.31, 0.10, 0.0, 0.0]))
        assert c["share_certifying_nothing"] == pytest.approx(2 / 6) and c["refused_by_review_set_guard"] == 1
        assert c["median_true_rate_of_certified_zones"] == pytest.approx(np.median([0.0, 0.02, 9 / 309, 0.5]))


def test_the_best_control_is_the_lowest_aurc_where_the_aurc_and_the_auroc_disagree():
    """The page picks the best informative control by the lowest AURC. Here one control has the higher AUROC and the
    higher AURC (two errors at its trusted end), so a pick by AUROC would choose the other control."""
    N, K = 200, 20
    err = np.zeros(N)
    err[:K] = 1
    corr = np.arange(K, N)
    high_auroc = np.zeros(N)
    high_auroc[corr] = np.linspace(0.1, 0.9, N - K)
    high_auroc[:18] = 1.0 + np.arange(18) * 0.01                          # 18 errors at the suspect end
    high_auroc[18:20] = [-1.0, -0.9]                                     # 2 errors at the trusted end
    low_aurc = np.zeros(N)
    low_aurc[corr] = np.linspace(0.0, 1.0, N - K)
    low_aurc[:K] = 0.62 + np.arange(K) * 0.001                           # every error in the middle
    ones = np.ones(N)
    assert metrics.aurc_expected(low_aurc, err) < metrics.aurc_expected(high_auroc, err)
    assert metrics.weighted_auroc(low_aurc, err, ones) < metrics.weighted_auroc(high_auroc, err, ones)
    controls = {"k4_high_auroc": high_auroc, "k5_low_aurc": low_aurc}
    conf = np.where(err > 0, 0.9, np.linspace(0, 0.5, N))
    rank = e89.ranking(err, conf, conf, conf, controls, np.arange(N) % 10, n_boot=20)
    assert rank["best_control"] == "k5_low_aurc"
    p2 = e89.grade_p2(rank, 40, "mangrove")
    assert p2["best_control"] == "k5_low_aurc" and p2["auroc"] == pytest.approx(rank["signals"]["k5_low_aurc"]["auroc"])


def test_the_random_designs_coverage_matches_its_exact_coverage(smoke):
    """The study's coverage is read against the population's theta. For the random design the package computes the
    exact coverage of the hypergeometric interval, so the 2,000 draws must agree with it within Monte Carlo error.
    A coverage read against anything else (the draw's own estimate covers itself) sits 8 or more errors away."""
    n_cells = 0
    for case, res in smoke["cases"].items():
        for key, cell in res["estimate"]["cells"].items():
            if not (key.startswith("random/") and cell["run"]):
                continue
            ex = cell["exact_coverage"]
            se = np.sqrt(ex * (1 - ex) / cell["draws"])
            assert 1 - ex >= 6 * se, (case, key, "the check has teeth only if a coverage of 1 is far from exact")
            assert abs(cell["coverage"] - ex) <= 4 * se, (case, key, cell["coverage"], ex)
            n_cells += 1
    assert n_cells >= 7


# ----------------------------------------------------------------------------- the guards
def _page(tmp_path, monkeypatch, text):
    page = tmp_path / "page.md"
    page.write_text(text)
    monkeypatch.setattr(e89, "PLAN", str(page))


def _args(tmp_path, arm="mangrove", **kw):
    return argparse.Namespace(arm=arm, out_dir=str(tmp_path), data=None, ckpt=None, offline=True, from_units=None,
                              real=False, **kw)


def test_the_status_reader():
    assert e89.prereg_status(FROZEN) == "frozen"
    assert e89.prereg_status(DRAFT) == "draft"
    assert e89.prereg_status("**Status: frozen on 2 October; the DRAFT is in the history.**") == "frozen"
    assert e89.prereg_status("no status line") == "unknown"
    assert e89.prereg_status() in ("draft", "frozen"), "the real page has a status line"


def test_the_run_and_the_gate_are_refused_while_the_page_is_a_draft(tmp_path, monkeypatch, capsys):
    _page(tmp_path, monkeypatch, DRAFT)
    called = []
    monkeypatch.setattr(e89, "run_arm", lambda *a, **k: called.append(a))
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: called.append("build"))
    json.dump({"passed": True, "attempts": [{}]}, open(tmp_path / "exp89_gate_mangrove.json", "w"))
    assert e89.cmd_run(_args(tmp_path)) == 2
    assert e89.cmd_gate(_args(tmp_path)) == 2
    assert not called and capsys.readouterr().out.count("refused") == 2


def test_the_run_is_refused_until_the_gate_passes(tmp_path, monkeypatch, capsys):
    _page(tmp_path, monkeypatch, FROZEN)
    called = []
    monkeypatch.setattr(e89, "run_arm", lambda *a, **k: called.append(a) or {"prereg": {"complete": True}})
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: (None, []))
    assert e89.cmd_run(_args(tmp_path)) == 2, "no gate record"
    gate = tmp_path / "exp89_gate_mangrove.json"
    json.dump({"passed": False, "closed": False, "attempts": [{"pass": False}]}, open(gate, "w"))
    assert e89.cmd_run(_args(tmp_path)) == 2, "a failed gate"
    json.dump({"passed": False, "closed": True, "attempts": [{"pass": False}] * 3}, open(gate, "w"))
    assert e89.cmd_run(_args(tmp_path)) == 2, "a closed gate"
    assert not called
    assert "the gate is closed" in capsys.readouterr().out
    os.remove(gate)
    _pass_gate(tmp_path, monkeypatch)
    assert e89.cmd_run(_args(tmp_path)) == 0 and len(called) == 1


def test_no_validation_window_is_scored_before_freezing(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, DRAFT)
    monkeypatch.setattr(e89, "read_units", lambda *a, **k: pytest.fail("no window may be read"))
    with pytest.raises(RuntimeError, match="not frozen"):
        e89.predict_population("mangrove", None, [{"split": "val"}])


def test_arm_n_is_reported_as_not_run(tmp_path, monkeypatch, capsys):
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "nandi_status", lambda: {"status": "not run", "reason": e89.ARMS["nandi"]["not_run_reason"]})
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: pytest.fail("arm N must not be built"))
    assert e89.cmd_run(_args(tmp_path, arm="nandi")) == 0
    assert e89.cmd_gate(_args(tmp_path, arm="nandi")) == 0
    assert capsys.readouterr().out.count("arm N not run") == 2
    assert e89.ARMS["nandi"]["model_revision"] is None and e89.ARMS["nandi"]["data_revision"] is None


def _fake_population(n, n_err, n_out=4, label=1, n_no_imagery=0):
    """(records, records with imagery, logits) for the gate: n Mangrove windows, the first n_err predicted wrong, one
    window dropped by the label rule, and n_no_imagery windows without imagery."""
    recs = [{"label": label, "drop": None, "pixel_labels": np.full(4, label), "input": 0} for _ in range(n)]
    recs.append({"label": None, "drop": "label pixels disagree", "pixel_labels": np.array([1, 1, 2, 2]), "input": 0})
    lg = np.zeros((n + 1, n_out))
    lg[:, label] = 5.0
    lg[:n_err, label] = -5.0
    lg[:n_err, 3] = 5.0
    blind = [{"label": label, "drop": "no imagery", "no_imagery": True} for _ in range(n_no_imagery)]
    return recs + blind, recs, lg


class _Rep:
    """A replica as the gate sees it: only the scored checkpoint's sha256."""
    def __init__(self, sha=PINNED_MANGROVE):
        self.checkpoint_sha256 = sha


def _pass_gate(out_dir, monkeypatch, rep=None):
    """A real pass: run_gate on a frozen page, 24 of 1000 windows wrong (97.6% pixel accuracy), the pinned checkpoint."""
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 24))
    rec = e89.run_gate("mangrove", rep or _Rep(), [], str(out_dir), log=lambda *a: None)
    assert rec["passed"] is True
    return rec


def test_a_rolled_back_gate_file_cannot_reset_the_attempt_count(tmp_path, monkeypatch):
    """The job resets the checkout to origin/main, which restores a committed gate file over a later attempt. The ledger
    keeps the count: the file is restored from it, and the third failure still closes the gate."""
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 100))
    gate = tmp_path / "exp89_gate_mangrove.json"
    e89.run_gate("mangrove", _Rep(), [], str(tmp_path))
    committed = gate.read_text()                                          # attempt 1, as committed and pushed
    e89.run_gate("mangrove", _Rep(), [], str(tmp_path))                   # attempt 2, never committed
    gate.write_text(committed)                                            # git reset --hard origin/main
    rec = e89.run_gate("mangrove", _Rep(), [], str(tmp_path))
    assert [a["attempt"] for a in rec["attempts"]] == [1, 2, 3] and rec["closed"] is True
    ledger, _ = e89.read_ledger("mangrove", str(tmp_path))
    assert [a["attempt"] for a in ledger] == [1, 2, 3]
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("a closed gate does not run"))
    gate.write_text(committed)
    assert e89.run_gate("mangrove", _Rep(), [], str(tmp_path))["closed"] is True


def test_a_gate_pass_its_ledger_does_not_hold_is_refused(tmp_path, monkeypatch, capsys):
    """A pass written by hand, or carried from elsewhere, is not a pass: neither the gate nor the run accepts it."""
    _page(tmp_path, monkeypatch, FROZEN)
    rec = _pass_gate(tmp_path, monkeypatch)
    os.remove(e89.read_ledger("mangrove", str(tmp_path))[1])
    monkeypatch.setattr(e89, "run_arm", lambda *a, **k: pytest.fail("no run without a verified pass"))
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: pytest.fail("no build without a verified pass"))
    assert e89.cmd_run(_args(tmp_path)) == 2
    assert "ledger" in capsys.readouterr().out
    with pytest.raises(e89.GateRefused, match="ledger"):
        e89.run_gate("mangrove", _Rep(), [], str(tmp_path))
    with pytest.raises(e89.GateRefused, match="ledger"):
        e89.require_gate("mangrove", str(tmp_path))
    assert rec["attempts"][0]["checkpoint_sha256"] == PINNED_MANGROVE


@pytest.mark.parametrize("field,value,why", [("prereg_status", "draft", "frozen"),
                                             ("checkpoint_sha256", "0" * 64, "pinned checkpoint"),
                                             ("checkpoint_sha256", None, "pinned checkpoint")])
def test_the_run_checks_how_the_pass_was_made(tmp_path, monkeypatch, field, value, why):
    """The pass must have been made on a frozen page and on the pinned checkpoint; the file and its ledger are edited
    together here, so only this rule can refuse it."""
    _page(tmp_path, monkeypatch, FROZEN)
    _pass_gate(tmp_path, monkeypatch)
    gate = tmp_path / "exp89_gate_mangrove.json"
    rec = json.load(open(gate))
    rec["attempts"][-1][field] = value
    json.dump(rec, open(gate, "w"))
    with open(e89.read_ledger("mangrove", str(tmp_path))[1], "w") as f:
        f.write("".join(json.dumps(a) + "\n" for a in rec["attempts"]))
    with pytest.raises(e89.GateRefused, match=why):
        e89.require_gate("mangrove", str(tmp_path))
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: pytest.fail("no build without a verified pass"))
    assert e89.cmd_run(_args(tmp_path)) == 2


def test_run_arm_and_compute_units_check_the_gate_themselves(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("no window is scored without a pass"))
    monkeypatch.setattr(e89, "read_units_file", lambda *a, **k: pytest.fail("nothing is graded without a pass"))
    with pytest.raises(e89.GateRefused):
        e89.run_arm("mangrove", str(tmp_path), units_file=str(tmp_path / "units.npz"))
    with pytest.raises(e89.GateRefused):
        e89.run_arm("mangrove", str(tmp_path), rep=_Rep(), windows=[])
    with pytest.raises(e89.GateRefused):
        e89.compute_units("mangrove", _Rep(), [], out_dir=str(tmp_path))


def test_the_gate_refuses_a_population_with_no_imagery_and_counts_windows_without_it(tmp_path, monkeypatch):
    """No kept window with imagery: nothing to gate on, and no attempt is spent. Windows without imagery are counted."""
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: ([{"label": 1, "drop": "no imagery"}], [],
                                                                     np.zeros((0, 4))))
    with pytest.raises(e89.GateRefused, match="imagery"):
        e89.run_gate("mangrove", _Rep(), [], str(tmp_path))
    assert not (tmp_path / "exp89_gate_mangrove.json").exists() and not e89.read_ledger("mangrove", str(tmp_path))[0]
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 24, n_no_imagery=7))
    att = e89.run_gate("mangrove", _Rep(), [], str(tmp_path))["attempts"][0]
    assert att["n_windows_no_imagery"] == 7 and att["n_windows"] == 1000 and att["n_windows_dropped"] == 8


def test_a_local_checkpoint_must_match_the_pin(tmp_path, monkeypatch):
    """--ckpt used to skip the sha256 check in the gate and the run; now a local file is hashed and refused unless it
    is the pinned checkpoint, before anything is loaded."""
    ck = tmp_path / "model.ckpt"
    ck.write_bytes(b"not the pinned checkpoint")
    monkeypatch.setattr(e89, "resolve_data", lambda *a, **k: (str(tmp_path), None))
    monkeypatch.setattr(e89, "Replica", lambda *a, **k: pytest.fail("a checkpoint off the pin is never loaded"))
    args = _args(tmp_path)
    args.ckpt = str(ck)
    with pytest.raises(RuntimeError, match="sha256"):
        e89.build_arm(args)


def test_the_encoder_config_is_read_at_the_pinned_revision(monkeypatch):
    """load_model_from_id reads OlmoEarth v1-Base's config.json at the Hub's current main; the page names 4bd1392a. The
    replica builds the encoder from the config at that revision, and the pin agrees with upstream_revisions.json."""
    import enum
    import inspect
    import types
    calls = {}

    class ModelID(str, enum.Enum):
        OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"
        OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"

        def repo_id(self):
            return f"allenai/{self.value}"

    def fake_download(repo_id, filename, revision=None, **kw):
        calls["download"] = (repo_id, filename, revision)
        return f"/hub/models--allenai--x/snapshots/{revision}/{filename}"

    def fake_from_path(path, load_weights=True):
        calls["from_path"] = (path, load_weights)
        return "encoder"
    ml = types.ModuleType("olmoearth_pretrain.model_loader")
    ml.ModelID, ml.load_model_from_path = ModelID, fake_from_path
    ml.load_model_from_id = lambda *a, **k: pytest.fail("the unpinned loader is not used")
    monkeypatch.setitem(sys.modules, "olmoearth_pretrain", types.ModuleType("olmoearth_pretrain"))
    monkeypatch.setitem(sys.modules, "olmoearth_pretrain.model_loader", ml)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert e89.encoder_skeleton("OLMOEARTH_V1_BASE") == "encoder"
    pin = "4bd1392a4539404d2c74276c39f3cb4cfff466cc"
    assert calls["download"] == ("allenai/OlmoEarth-v1-Base", "config.json", pin)
    assert calls["from_path"] == (f"/hub/models--allenai--x/snapshots/{pin}", False)
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "upstream_revisions.json")))["repos"]
    for name, rev in e89.ENCODER_REVISIONS.items():
        assert rec[ModelID[name].repo_id()]["revision"] == rev, name
    src = inspect.getsource(e89)
    assert "load_model_from_id" not in src, "every encoder skeleton is built at a pinned revision"


def test_the_page_and_the_script_agree_on_what_the_owner_settled():
    """The page is the preregistration: what the script reads by interpretation is written on it before freezing."""
    with open(e89.PLAN, encoding="utf-8") as f:
        page = f.read()
    assert "(threshold: owner to confirm)" not in page, "K5 was confirmed with the other thresholds"
    assert "optional, owner to decide" not in page, "the owner chose to run arm A (status line)"
    assert "89.5%" in page and "AWF (arm A) passes" in page, "arm A's gate is on the page"
    assert e89.MAX_GATE_ATTEMPTS == 3 and "at most three attempts in all" in page
    assert "retried at most three times" not in page
    assert "n_p1_saturated" in page, "the graded top-1 probability can tie at 1.0, and the page says so"
    assert "Readings fixed by the run script" in page


def test_the_page_states_the_amendment_the_script_follows():
    """The amendment of 1 October 2026 is on the page, dated, and the page is now frozen: arms F and A report-only
    and why, their alignment tolerances and the label a check outside them gives, arm M waiting for Ai2's split, arm N
    not run, and arm A's data path."""
    with open(e89.PLAN, encoding="utf-8") as f:
        page = f.read()
    assert e89.prereg_status(page) == "frozen"
    assert "## Amendment of 1 October 2026" in page
    assert "report-only" in page and "no pass or fail verdict is drawn" in page
    assert "109 windows" in page and "344 points" in page and "error floor of 40" in page and "N/5" in page
    assert "within 2.0 points of Ai2's 76.1%" in page and "within 2.0 points of Ai2's 89.5%" in page
    assert '"replica not aligned"' in page and e89.NOT_ALIGNED == "replica not aligned"
    assert "waits for Ai2's validation split" in page and "Slack on 1 October 2026" in page
    assert "Arm N (Nandi) stays not run" in page
    assert "no longer reads exp21's `data/awf`" in page
    assert "olmoearth_evals_split" in page and "It never chooses a window" in page
    assert "—" not in page, "no em dashes"
    for arm in ("awf", "fld"):
        A = e89.ARMS[arm]
        assert f"{A['ai2_accuracy'] * 100:.1f}%" in page and A["gate_tolerance"]["tar"] == 0.02


def test_the_gate_writes_counts_and_accuracy_only_and_closes_after_three_failures(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 100))
    for i in range(3):
        rec = e89.run_gate("mangrove", None, [], str(tmp_path))
        assert rec["passed"] is False and len(rec["attempts"]) == i + 1
    assert rec["closed"] is True
    assert set(rec) == set(e89.GATE_KEYS)
    att = rec["attempts"][0]
    assert set(att) == set(e89.ATTEMPT_KEYS), "no confidence, margin or control in the gate file"
    assert att["n_windows"] == 1000 and att["n_errors"] == 100 and att["n_windows_dropped"] == 1
    assert att["n_pixels"] == 4004 and att["n_pixel_errors"] == 402, "every valid pixel, kept window or not"
    text = (tmp_path / "exp89_gate_mangrove.json").read_text()
    for word in ("p1", "margin", "confidence", "k3", "auroc", "logit"):
        assert word not in text.lower(), word
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("a closed gate does not run"))
    assert e89.run_gate("mangrove", None, [], str(tmp_path))["closed"] is True


def test_the_gate_passes_within_tolerance_two_sided_and_is_not_rerun(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, FROZEN)
    # 97.6% pixel accuracy needs 2.4% pixel errors: 24 of 1000 windows wrong, plus the dropped window's 2 pixels
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 24))
    rec = e89.run_gate("mangrove", None, [], str(tmp_path))
    assert rec["passed"] is True and abs(rec["attempts"][0]["gap_points"]) <= 0.5
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("a passed gate is not rerun"))
    assert e89.run_gate("mangrove", None, [], str(tmp_path))["passed"] is True
    other = tmp_path / "far_above"
    other.mkdir()
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 0))
    assert e89.run_gate("mangrove", None, [], str(other))["passed"] is False, "100% is as suspect as 96%"


# ----------------------------------------------------------------------------- the reader on an rslearn layout
def test_the_reader_maps_band_sets_to_olmoearths_order_and_resamples(tmp_path):
    pytest.importorskip("rasterio")
    root, names = e89.synthetic_dataset(str(tmp_path), "nandi", n_train=3, n_val=2, size=24, dataset_bands="three")
    windows = e89.list_windows(os.path.join(root, "windows"), "spatial_split")
    assert [w["split"] for w in windows] == ["train"] * 3 + ["val"] * 2
    w = windows[0]
    stack, n_groups = e89.read_stack(w["dir"], w["meta"])
    assert stack.shape == (24, 24, 12, 12) and n_groups == 12
    import rasterio
    with rasterio.open(os.path.join(w["dir"], "layers", "sentinel2.4", "B02_B03_B04_B08", "geotiff.tif")) as src:
        b04 = src.read(3)
    assert np.array_equal(stack[:, :, 4, e89.OLMO_BANDS.index("B04")], b04), "B04 lands at OlmoEarth index 2"
    assert stack[:, :, :, e89.OLMO_BANDS.index("B01")].std() > 0, "the 60 m band is warped onto the 10 m grid"
    rec = e89.unit_of("nandi", windows[3]["dir"], windows[3]["meta"])
    assert rec["label"] is not None and len(rec["crops"]) == 4 and rec["crops"][0].shape == (16, 16, 12, 12)
    assert rec["series"].shape == (12, 12) and rec["ndvi3x3"].shape == (3, 3, 12)
    _, _, empty = e89.unit_indices(rec["series"][None])
    assert empty[0, 5], "the month with no mosaic is empty"
    os.remove(os.path.join(w["dir"], "layers", "sentinel2.7", "completed"))
    groups = e89.completed_groups(w["dir"], "sentinel2")
    assert 7 not in groups and len(groups) == 11, "a group without rslearn's marker is not loaded"
    _, n = e89.read_stack(w["dir"], w["meta"])
    assert n == 11


def test_the_mangrove_reader_crops_the_centre_and_counts_drops(tmp_path):
    pytest.importorskip("rasterio")
    root, names = e89.synthetic_dataset(str(tmp_path), "mangrove", n_train=14, n_val=4)
    windows = e89.list_windows(os.path.join(root, "windows"), "sample_100K")
    big = [w for w in windows if w["meta"]["bounds"][2] - w["meta"]["bounds"][0] == 4]
    assert big, "the synthetic set holds a 4x4 window"
    rec = e89.unit_of("mangrove", big[0]["dir"], big[0]["meta"])
    assert rec["input"].shape == (2, 2, 12, 12) and rec["pixel_labels"].size == 4
    stack, _ = e89.read_stack(big[0]["dir"], big[0]["meta"])
    assert np.array_equal(rec["input"], stack[1:3, 1:3]), "rslearn's centre crop of a 4x4 window"
    drops = [e89.unit_of("mangrove", w["dir"], w["meta"], with_stack=False)["drop"] for w in windows]
    assert "a label pixel is 0" in drops and "label pixels disagree" in drops


class _FakeReplica:
    """The replica as compute_units uses it: the arm's channels, a normalisation and logits from the inputs."""
    def __init__(self, arm):
        self.arm, self.A = arm, e89.ARMS[arm]

    def normalize(self, raw):
        return np.asarray(raw, dtype=np.float64) / 10000.0

    def logits(self, stacks, locs=None, timestamps="rslearn"):
        m = np.asarray(stacks, dtype=np.float64).mean(axis=tuple(range(1, np.ndim(stacks))))
        L = np.zeros((len(m), self.A["n_out"]))
        L[:, 1], L[:, 2], L[:, 3] = m / 1000.0, 2.0, 1.5
        return L


def test_k2_and_k3_are_fitted_on_training_windows_only(tmp_path, monkeypatch):
    """compute_units on a synthetic rslearn dataset with class_rarity and the K3 fit spied on: K2's frequencies come
    from the training labels, and K3 is fitted on the training windows' features and labels, never a validation one."""
    pytest.importorskip("rasterio")
    root, _ = e89.synthetic_dataset(str(tmp_path), "mangrove", n_train=14, n_val=8)
    windows = e89.list_windows(os.path.join(root, "windows"), "sample_100K")
    seen = {}
    real_rarity = e89.class_rarity

    def spy_rarity(pred, train_labels, trained):
        seen["k2_labels"] = np.asarray(train_labels).copy()
        return real_rarity(pred, train_labels, trained)

    def spy_fit(x_train, y_train, classes, x_eval, seed=e89.SEED):
        seen["x_train"], seen["y_train"] = np.asarray(x_train).copy(), np.asarray(y_train).copy()
        return np.full((len(x_eval), len(classes)), 1.0 / len(classes)), {"n_train": len(y_train)}
    monkeypatch.setattr(e89, "class_rarity", spy_rarity)
    monkeypatch.setattr(e89, "fit_no_encoder_classifier", spy_fit)
    rep = _FakeReplica("mangrove")
    units, meta = e89.compute_units("mangrove", rep, windows, log=lambda *a: None, synthetic=True)
    train = [w for w in windows if w["split"] == "train"]
    tr = [r for r in e89.read_units("mangrove", train, need="model", log=lambda *a: None)
          if "series" in r and r["label"] is not None]
    assert len(tr) >= 8 and len(units["label"]) >= 4
    assert np.array_equal(seen["k2_labels"], [r["label"] for r in tr]), "K2 reads the training labels"
    assert np.array_equal(seen["y_train"], [r["label"] for r in tr]), "K3's labels are the training labels"
    want = e89.k3_features(np.stack([r["series"] for r in tr]), rep.normalize, "mangrove")
    assert np.array_equal(seen["x_train"], want), "K3's features are the training windows' features"
    assert meta["n_train_for_k3"] == len(tr) and meta["n_train_windows_labelled"] == len(tr)


def test_the_tar_round_trip_and_the_inventory_without_a_model(tmp_path):
    pytest.importorskip("rasterio")
    tar, names = e89.synthetic_tar(str(tmp_path), "mangrove", n_train=8, n_val=4)
    info = e89.extract_tar(tar, str(tmp_path / "data"))
    assert info["n_members"] > 0 and os.path.exists(tmp_path / "data" / ".exp89_extracted")
    s = e89.dataset_summary("mangrove", e89.find_windows_root(str(tmp_path / "data")))
    assert s["windows_per_split"] == {"train": 8, "val": 4}
    assert s["item_groups_per_window"] == {"12": 12} and s["layers"]["sentinel2"]["completed"] == 12 * 12
    assert s["band_sets"] == {"B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12": 12}
    assert sum(s["units_per_split"].values()) + sum(sum(d.values()) for d in s["dropped_per_split"].values()) == 12


def test_an_extraction_is_dated_now_not_with_its_members_times(tmp_path):
    """The scratch purge removes files by age; a member packed in 1970 must not keep that date once extracted."""
    import io
    import tarfile
    import time
    tar = tmp_path / "old.tar"
    with tarfile.open(tar, "w") as tf:
        for name in ("dataset/windows/g/w0/metadata.json", "dataset/config.json"):
            data = b"{}"
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime = len(data), 0
            tf.addfile(ti, io.BytesIO(data))
    t0 = time.time()
    e89.extract_tar(str(tar), str(tmp_path / "data"))
    for name in ("dataset/windows/g/w0/metadata.json", "dataset/config.json"):
        st = os.stat(tmp_path / "data" / name)
        assert st.st_mtime >= t0 - 2 and st.st_atime >= t0 - 2, name


# ----------------------------------------------------------------------------- the pins
def test_every_repo_exp89_names_is_pinned_or_says_why():
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "upstream_revisions.json")))["repos"]
    for arm, A in e89.ARMS.items():
        for key, rev in (("model", A["model_revision"]), ("data", A["data_revision"])):
            if A[key] is None:                          # arm F's dataset is not on the Hub; pinned below instead
                continue
            entry = rec[A[key]]
            assert entry["revision"] == rev, (arm, key)
            if rev is None:
                assert "401" in entry["why"]
    F = e89.ARMS["fld"]
    assert F["data"] is None and F["data_url"].startswith("https://storage.googleapis.com/")
    assert (F["data_generation"], F["data_md5"], F["data_bytes"]) == (
        "1761857427036506", "6abc5b944c64330f0a545d5ed7ddddf4", 42214604800), "the HEAD of 1 October 2026"
    assert (F["model_revision"], F["model_bytes"]) == ("15502f8acb4caed6e3a7d777b0fb569c1f7eb791", 381184175)


# ----------------------------------------------------------------------------- the amendment of 1 October 2026
def test_arm_m_waits_for_ai2s_split(tmp_path, monkeypatch, capsys):
    """Arm M keeps its graded predictions and is reported as not run, by every mode that scores a window, until Ai2's
    validation split is pinned."""
    assert MANGROVE_SPLIT_AS_SHIPPED is None, "no split is pinned yet"
    monkeypatch.setitem(e89.ARMS["mangrove"], "split_source", None)
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: pytest.fail("arm M must not be built"))
    assert e89.cmd_run(_args(tmp_path)) == 0
    assert e89.cmd_gate(_args(tmp_path)) == 0
    args = _args(tmp_path)
    args.real = True
    assert e89.smoke_torch(args) == 0
    out = capsys.readouterr().out
    assert out.count("arm M not run") == 3 and "validation split" in out
    with pytest.raises(RuntimeError, match="not run"):
        e89.predict_population("mangrove", None, [{"split": "val"}])
    assert e89.ARMS["mangrove"]["graded"] == ("P1", "P2", "P3", "P4", "P5") and not e89.is_report_only("mangrove")


def test_arms_a_and_f_are_report_only():
    for arm in ("awf", "fld"):
        A = e89.ARMS[arm]
        assert e89.is_report_only(arm) and A["graded"] == ()
    assert (e89.ARMS["fld"]["ai2_accuracy"], e89.ARMS["fld"]["gate_tolerance"]["tar"]) == (0.761, 0.020)
    assert (e89.ARMS["awf"]["ai2_accuracy"], e89.ARMS["awf"]["gate_tolerance"]["tar"]) == (0.895, 0.020)
    assert not e89.is_report_only("nandi") and e89.ARMS["nandi"]["graded"] == ("P1", "P2", "P3", "P4", "P5")


def _fld_population(n, n_err, flip_err=None):
    """(records, records with imagery, logits) for arm F's gate: n kept windows of class 0 (agriculture), the first
    n_err predicted as burned, one window dropped by the layer rule and one without imagery."""
    recs = [{"label": 0, "drop": None, "input": 0, "n_groups": 8} for _ in range(n)]
    lg = np.zeros((n, 10))
    lg[:, 0] = 5.0
    lg[:n_err, 0], lg[:n_err, 5] = -5.0, 5.0
    blind = [{"label": None, "drop": "8 of the 8 image layers not completed", "no_imagery": True},
             {"label": None, "drop": "1 of the 8 image layers not completed"}]
    return recs + blind, recs, lg


class _RepF:
    checkpoint_sha256 = PINNED_FLD


def test_the_alignment_check_says_aligned_or_replica_not_aligned_never_pass_or_fail(tmp_path, monkeypatch, capsys):
    """Arm F's gate is an alignment check within 2.0 points of Ai2's 76.1%: 80 of 109 (73.4%) is outside it and reads
    "replica not aligned"; the record holds "aligned", never "pass" or "passed", and the log never says PASS or FAIL.
    The flips are reported beside, and only the unflipped accuracy is checked."""
    _page(tmp_path, monkeypatch, FROZEN)
    calls = []

    def fake(arm, rep, windows, log=print, synthetic=False, flip="none"):
        calls.append(flip)
        return _fld_population(109, 29 if flip == "none" else 26)
    monkeypatch.setattr(e89, "predict_population", fake)
    rec = e89.run_gate("fld", _RepF(), [], str(tmp_path))
    assert set(rec) == set(e89.ALIGN_KEYS) and "passed" not in rec
    att = rec["attempts"][0]
    assert set(att) == set(e89.ALIGN_ATTEMPT_KEYS) and "pass" not in att
    assert att["aligned"] is False and rec["alignment"] == "replica not aligned"
    assert att["n_windows"] == 109 and att["n_errors"] == 29 and att["n_windows_no_imagery"] == 1
    assert att["n_windows_dropped"] == 2 and att["accuracy"] == pytest.approx(80 / 109)
    assert att["accuracy_by_flip"]["none"] == pytest.approx(80 / 109)
    assert att["accuracy_by_flip"]["hv"] == pytest.approx(83 / 109), "reported beside, never checked"
    assert calls == ["none", "h", "v", "hv"]
    out = capsys.readouterr().out
    assert "replica not aligned" in out and "PASS" not in out and "FAIL" not in out
    assert e89.verdicts_in(json.load(open(tmp_path / "exp89_gate_fld.json"))) == []
    # a second attempt at Ai2's 83 of 109 reads aligned, and is not rerun
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fld_population(109, 26))
    rec = e89.run_gate("fld", _RepF(), [], str(tmp_path))
    assert rec["aligned"] is True and rec["alignment"] == "aligned" and len(rec["attempts"]) == 2
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("an aligned check is not rerun"))
    assert e89.run_gate("fld", _RepF(), [], str(tmp_path))["aligned"] is True


def test_a_report_only_run_goes_ahead_after_a_check_outside_the_tolerance(tmp_path, monkeypatch, capsys):
    """The run needs a recorded alignment check (frozen page, pinned checkpoint, in the ledger), not an aligned one;
    its numbers then carry "replica not aligned", and neither the summary nor the log holds a verdict."""
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setitem(e89.ARMS["mangrove"], "split_source", None)       # as shipped: arm M waits
    with pytest.raises(e89.GateRefused, match="no alignment check"):
        e89.require_gate("fld", str(tmp_path))
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fld_population(109, 40))
    e89.run_gate("fld", _RepF(), [], str(tmp_path))
    assert e89.require_gate("fld", str(tmp_path))["alignment"] == "replica not aligned"
    units = e89.synthetic_units("fld_report")
    path = tmp_path / "units_fld.npz"
    e89.write_units(units, str(path))
    capsys.readouterr()
    res = e89.run_arm("fld", str(tmp_path), units_file=str(path), draws=40, n_boot=40)
    assert res["alignment"] == "replica not aligned" and res["report_only"] is True
    summary = json.load(open(tmp_path / "exp89_summary.json"))
    assert e89.verdicts_in(summary["arms"]["fld"]) == []
    assert summary["arms"]["mangrove"]["status"] == "not run" and "split" in summary["arms"]["mangrove"]["reason"]
    out = capsys.readouterr().out
    assert "report-only, replica not aligned" in out
    for word in ("PASS", "FAIL", "holds", " True", " False"):
        assert word not in out, word
    # a record made before freezing, or on another checkpoint, is still refused
    gate = tmp_path / "exp89_gate_fld.json"
    rec = json.load(open(gate))
    rec["attempts"][-1]["checkpoint_sha256"] = "0" * 64
    json.dump(rec, open(gate, "w"))
    with open(e89.read_ledger("fld", str(tmp_path))[1], "w") as f:
        f.write("".join(json.dumps(a) + "\n" for a in rec["attempts"]))
    with pytest.raises(e89.GateRefused, match="pinned checkpoint"):
        e89.require_gate("fld", str(tmp_path))


def test_cmd_gate_and_cmd_run_return_zero_on_a_report_only_arm(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fld_population(109, 40))
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: (_RepF(), []))
    assert e89.cmd_gate(_args(tmp_path, arm="fld")) == 0, "a check outside the tolerance is reported, not failed"
    monkeypatch.setattr(e89, "run_arm", lambda *a, **k: {"report_only": True})
    assert e89.cmd_run(_args(tmp_path, arm="fld")) == 0


# ----------------------------------------------------------------------------- arm F's rules
def _ft(label=None, props=True):
    return {"type": "Feature", "properties": ({"new_label": label} if label is not None else {}) if props else None}


def test_arm_f_label_is_classification_tasks_first_valid_new_label():
    assert e89.FLD_CLASSES[0] == "agriculture" and e89.FLD_CLASSES[-1] == "none" and len(e89.FLD_CLASSES) == 10
    assert e89.fld_label_from_features([_ft("burned")])[:2] == (5, None)
    assert e89.fld_label_from_features([_ft(props=False), _ft(), _ft("unknown"), _ft("river")]) == (8, None, "river")
    assert e89.fld_label_from_features([_ft("agriculture-generic"), _ft("coca")]) == (
        None, "new_label outside the ten classes", "agriculture-generic"), "ClassificationTask has no remap"
    assert e89.fld_label_from_features([_ft(props=False)])[1] == "no feature has new_label"
    assert e89.fld_label_from_features([])[0] is None


def test_arm_f_features_by_hand():
    """A 64-px crop with known values: the composites, NDVI and NBR, the centre means, the difference, the empty
    timesteps, K4's drops and K5's cloudy timesteps."""
    S = 64
    crop = np.zeros((S, S, 8, 12))
    b = {n: e89.OLMO_BANDS.index(n) for n in ("B02", "B04", "B08", "B12")}
    for t in range(8):
        if t == 2:
            continue                                                   # an empty pre timestep
        crop[:, :, t, :] = 500.0
        crop[:, :, t, b["B04"]] = 400.0
        crop[:, :, t, b["B08"]] = 3000.0 if t < 4 else 1500.0
        crop[:, :, t, b["B12"]] = 1000.0 if t < 4 else 2000.0
    crop[24:40, 24:40, 6, b["B02"]] = 2500.0                           # a cloudy post timestep at the centre
    x, idx = e89.fld_window_features(crop)
    assert x.shape == (128,) and len(e89.FLD_FEATURE_NAMES) == 128
    f = dict(zip(e89.FLD_FEATURE_NAMES, x))
    ndvi_pre, ndvi_post = (3000 - 400) / 3400, (1500 - 400) / 1900
    nbr_pre, nbr_post = (3000 - 1000) / 4000, (1500 - 2000) / 3500
    assert f["pre_crop_mean_NDVI"] == pytest.approx(ndvi_pre) and f["post_centre_mean_NBR"] == pytest.approx(nbr_post)
    assert f["diff_crop_mean_NDVI"] == pytest.approx(ndvi_post - ndvi_pre)
    assert f["pre_crop_std_B08"] == 0.0 and f["pre_empty_timesteps"] == 1 and f["post_empty_timesteps"] == 0
    assert f["pre_crop_mean_B08"] == 3000.0, "the median over the three non-empty pre timesteps"
    k = e89.fld_index_controls([idx])
    assert k["k4_ndvi_drop_weak"][0] == pytest.approx(-(ndvi_pre - ndvi_post))
    assert k["k4_nbr_drop_weak"][0] == pytest.approx(-(nbr_pre - nbr_post))
    assert k["k5_cloud_timesteps"][0] == 2, "one empty timestep and one cloudy at the centre"
    with pytest.raises(ValueError):
        e89.fld_window_features(np.zeros((64, 64, 12, 12)))


def test_arm_f_reader_on_an_rslearn_layout(tmp_path):
    """The population is options.split "val" with all eight layers and the label completed and a valid new_label;
    olmoearth_evals_split and label.json are never read. The crop is rslearn's centre Pad to 64 (rows and columns 32 to
    95 of a 128-px window), and the stack runs pre_sentinel2, .1 to .3, then post_sentinel2, .1 to .3."""
    pytest.importorskip("rasterio")
    import rasterio
    root, names = e89.synthetic_fld_dataset(str(tmp_path), n_train=1, n_val=5, size=128)
    windows = e89.arm_windows("fld", os.path.join(root, "windows"))
    assert {w["group"] for w in windows} == set(e89.FLD_SYN_GROUPS), "every group is listed"
    val = [w for w in windows if w["split"] == "val"]
    assert len(val) == 5 and all(w["meta"]["options"]["olmoearth_evals_split"] == "train" for w in val)
    recs = e89.read_units("fld", val, need="model", log=lambda *a: None)
    by = {r["name"]: r for r in recs}
    v = sorted(by)
    assert by[v[0]]["drop"] == "1 of the 8 image layers not completed" and "input" not in by[v[0]]
    assert by[v[1]]["drop"] == "new_label outside the ten classes" and by[v[1]]["new_label"] == "unknown"
    assert by[v[2]]["label"] is not None, "a feature without properties is passed over"
    with open(os.path.join(next(w["dir"] for w in val if w["name"] == v[2]), "label.json")) as f:
        assert e89.FLD_CLASSES.index(json.load(f)["new_label"]) != by[v[2]]["label"], "label.json is stale"
    r = by[v[4]]
    assert r["input"].shape == (64, 64, 8, 12) and r["n_groups"] == 8 and r["features"].shape == (128,)
    w = next(w for w in val if w["name"] == v[4])
    with rasterio.open(os.path.join(w["dir"], "layers", "post_sentinel2.1", e89.FLD_BAND_SET, "geotiff.tif")) as src:
        b08 = src.read(e89.FLD_BAND_SET.split("_").index("B08") + 1)
    assert np.array_equal(r["input"][:, :, 5, e89.OLMO_BANDS.index("B08")], b08[32:96, 32:96]), \
        "layer 5 is post_sentinel2.1, B08 sits at OlmoEarth's index 3, and the crop is rows and columns 32 to 95"
    assert by[v[3]]["indices"]["pre"]["n_empty"] == 1, "the all-zero pre_sentinel2.2 is an empty timestep"
    # rslearn's check_window: the population ignores olmoearth_evals_split, whatever it says
    tr = [w for w in windows if w["split"] == "train"]
    assert tr and all(w["meta"]["options"]["olmoearth_evals_split"] == "val" for w in tr)


def _fld_tar(tmp_path, **kw):
    pytest.importorskip("rasterio")
    tar, names = e89.synthetic_tar(str(tmp_path / "src"), "fld", **kw)
    return tar


def _pin_tar(monkeypatch, tar, sha=None):
    import hashlib
    monkeypatch.setitem(e89.ARMS["fld"], "data_bytes", os.path.getsize(tar))
    monkeypatch.setitem(e89.ARMS["fld"], "data_md5", hashlib.md5(open(tar, "rb").read()).hexdigest())
    monkeypatch.setitem(e89.ARMS["fld"], "data_sha256", sha)


def test_arm_f_tar_is_checked_against_its_pin(tmp_path, monkeypatch):
    """Size, then MD5, then (once pinned) SHA-256; the SHA-256 is recorded either way."""
    import hashlib
    tar = _fld_tar(tmp_path, n_train=2, n_val=1)
    with pytest.raises(RuntimeError, match="bytes, pinned 42214604800"):
        e89.verify_tar(tar, "fld", log=lambda *a: None)
    _pin_tar(monkeypatch, tar)
    info = e89.verify_tar(tar, "fld", log=lambda *a: None)
    sha = hashlib.sha256(open(tar, "rb").read()).hexdigest()
    assert info["sha256"] == sha and info["sha256_pinned"] is None and info["generation"] == "1761857427036506"
    monkeypatch.setitem(e89.ARMS["fld"], "data_sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        e89.verify_tar(tar, "fld", log=lambda *a: None)
    monkeypatch.setitem(e89.ARMS["fld"], "data_sha256", sha)
    monkeypatch.setitem(e89.ARMS["fld"], "data_md5", "0" * 32)
    with pytest.raises(RuntimeError, match="md5"):
        e89.verify_tar(tar, "fld", log=lambda *a: None)


def test_arm_f_extraction_keeps_only_what_the_run_reads_and_survives_the_purge(tmp_path, monkeypatch):
    """The tar is streamed once: the eight image layers, the label layer, the window files and the dataset's config
    are written, every other layer is counted and skipped. A file the scratch purge removes later is caught by the
    manifest, and the extraction is made again from the tar."""
    tar = _fld_tar(tmp_path, n_train=2, n_val=2)
    _pin_tar(monkeypatch, tar)
    data = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="--tar"):
        e89.resolve_data("fld", str(data), download=False)
    root, info = e89.resolve_data("fld", str(data), tar=tar, log=lambda *a: None)
    assert set(info["skipped"]) == set(e89.FLD_SYN_EXTRA_LAYERS) and info["md5"] == e89.ARMS["fld"]["data_md5"]
    assert set(info["kept"]) == set(e89.FLD_KEEP_LAYERS) | {"window and dataset files"}
    w = os.path.join(root, e89.FLD_SYN_GROUPS[0], "fld_0002")
    assert os.path.exists(os.path.join(w, "layers", "post_sentinel2.3", e89.FLD_BAND_SET, "geotiff.tif"))
    assert not os.path.exists(os.path.join(w, "layers", "pre_sentinel2.4"))
    assert os.path.exists(os.path.join(w, "metadata.json")) and os.path.exists(data / "dataset" / "config.json")
    assert e89.manifest_missing(str(data)) == []
    victim = os.path.join(w, "layers", "pre_sentinel2.1", e89.FLD_BAND_SET, "geotiff.tif")
    os.remove(victim)                                                 # the 30-day purge
    assert e89.manifest_missing(str(data)) != []
    said = []
    root2, _ = e89.resolve_data("fld", str(data), download=False, log=said.append)
    assert os.path.exists(victim) and "extracted again" in said[0] and root2 == root
    assert e89.member_layer("dataset/windows/g/w/layers/pre_sentinel2.4/B/geotiff.tif") == "pre_sentinel2.4"
    assert e89.member_layer("dataset/windows/g/w/metadata.json") is None
    assert e89.member_layer("dataset/windows/g/layers/metadata.json") is None, "a window named layers is a window"


def test_the_page_says_what_the_inventory_reads_and_what_is_still_proposed():
    """The inventory reads olmoearth_evals_split and label.json, to report them beside split and data.geojson, so the
    page says they never choose or label a window, not that they are never read. The extraction keeps every file
    outside the windows, and the page says so. The plan index does not call arm F's readings or the alignment
    tolerances confirmed."""
    import inspect
    counts = inspect.getsource(e89.fld_window_counts)
    assert "olmoearth_evals_split" in counts and '"label.json"' in counts, "the inventory reads both"
    for fn in (e89.fld_unit, e89.fld_label, e89.fld_label_from_features, e89.arm_windows, e89.list_windows,
               e89.predict_population, e89.compute_units):
        src = inspect.getsource(fn)
        assert "olmoearth_evals_split" not in src and '"label.json"' not in src, fn.__name__
    with open(e89.PLAN, encoding="utf-8") as f:
        page = f.read()
    assert "and is never read" not in page and "`old_label` are never read" not in page
    assert "the inventory only reports it beside `split`" in page
    assert "every file outside the windows" in page
    with open(os.path.join(ROOT, "docs", "plan", "index.md"), encoding="utf-8") as f:
        row = next(line for line in f if "| exp89 |" in line)
    status = row.rstrip().rstrip("|").rsplit("|", 1)[-1]
    assert "Frozen" in status and "proposed" not in status, status   # frozen on 1 October 2026 with every reading confirmed


class _FakeReplicaF:
    """Arm F's replica as compute_units uses it: ten channels, logits from the input's mean."""
    def __init__(self):
        self.arm, self.A = "fld", e89.ARMS["fld"]

    def logits(self, stacks, locs=None, timestamps="rslearn"):
        m = np.asarray(stacks, dtype=np.float64).mean(axis=tuple(range(1, np.ndim(stacks))))
        L = np.zeros((len(m), 10))
        L[:, 0], L[:, 5], L[:, 9] = m / 1000.0, 0.6, 0.4
        return L


def test_arm_f_k2_and_k3_are_fitted_on_training_windows_only(tmp_path, monkeypatch):
    """compute_units on arm F's rslearn layout with class_rarity and the K3 fit spied on: K2's frequencies come from
    the training windows' labels, and K3 is fitted on the training windows' 128 crop statistics, never a validation
    window's."""
    pytest.importorskip("rasterio")
    root, _ = e89.synthetic_fld_dataset(str(tmp_path), n_train=8, n_val=6, size=72)
    windows = e89.arm_windows("fld", os.path.join(root, "windows"))
    seen = {}
    real_rarity = e89.class_rarity

    def spy_rarity(pred, train_labels, trained):
        seen["k2_labels"] = np.asarray(train_labels).copy()
        return real_rarity(pred, train_labels, trained)

    def spy_fit(x_train, y_train, classes, x_eval, seed=e89.SEED):
        seen["x_train"], seen["y_train"], seen["x_eval"] = (np.asarray(x_train).copy(), np.asarray(y_train).copy(),
                                                            np.asarray(x_eval).copy())
        return np.full((len(x_eval), len(classes)), 1.0 / len(classes)), {"n_train": len(y_train)}
    monkeypatch.setattr(e89, "class_rarity", spy_rarity)
    monkeypatch.setattr(e89, "fit_no_encoder_classifier", spy_fit)
    units, meta = e89.compute_units("fld", _FakeReplicaF(), windows, log=lambda *a: None, synthetic=True)
    train = [w for w in windows if w["split"] == "train"]
    tr = [r for r in e89.read_units("fld", train, need="features", log=lambda *a: None) if r["label"] is not None]
    va = [r for r in e89.read_units("fld", [w for w in windows if w["split"] == "val"], need="features",
                                    log=lambda *a: None) if r["label"] is not None]
    assert len(tr) == 8 and len(va) == 4 == len(units["label"])
    assert np.array_equal(seen["k2_labels"], [r["label"] for r in tr]), "K2 reads the training labels"
    assert np.array_equal(seen["y_train"], [r["label"] for r in tr]), "K3's labels are the training labels"
    assert np.array_equal(seen["x_train"], np.stack([r["features"] for r in tr]))
    assert np.array_equal(seen["x_eval"], np.stack([r["features"] for r in va]))
    assert not any((seen["x_train"] == v).all(1).any() for v in seen["x_eval"]), "no validation window in the fit"
    assert set(units["stratum"]) == {e89.FLD_SYN_GROUPS[0]} and units["clusters_coarse"] is None


# ----------------------------------------------------------------------------- arm A's data path
def test_arm_a_never_reads_the_old_data_awf(tmp_path, monkeypatch):
    """The inventory of 1 October failed on data/awf, which the scratch purge had partly removed. Arm A now reads only
    the pinned tar's extraction under E89_DATA: a bare windows directory without the extraction's marker is not used,
    and the job points every arm at E89_DATA/<arm>."""
    monkeypatch.delenv("E89_DATA", raising=False)
    assert e89.default_data_dir("awf") == os.path.join(e89.ROOT, "data", "exp89_awf")
    monkeypatch.setenv("E89_DATA", str(tmp_path / "e89"))
    assert e89.default_data_dir("awf") == str(tmp_path / "e89" / "awf")
    old = tmp_path / "awf" / "dataset" / "windows" / "spatial_split" / "task_0_point_0"
    old.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        e89.resolve_data("awf", str(tmp_path / "awf"), download=False)
    job = open(os.path.join(ROOT, "exp", "jobs", "e89.sh")).read()
    assert "data/awf" not in job.replace("never reads data/awf", "") and "DATA=$E89_DATA/$ARM" in job


def test_arm_a_reads_the_extracted_tar_without_touching_the_awf_module(tmp_path, monkeypatch):
    """A tar off the pin is refused; the pinned one is extracted, and arm_windows gives oe_inferencex.awf.list_windows
    the extraction's group directory. The module's ROOT is not changed, so its other callers keep exp21's default."""
    pytest.importorskip("torch")
    pytest.importorskip("olmoearth_pretrain")
    pytest.importorskip("rasterio")
    from oe_inferencex import awf
    before = awf.ROOT
    tar, _ = e89.synthetic_tar(str(tmp_path / "src"), "awf", n_train=4, n_val=3, size=63, dataset_bands="three")
    with pytest.raises(RuntimeError, match="pinned d0837f14"):
        e89.resolve_data("awf", str(tmp_path / "data"), tar=tar)
    monkeypatch.setitem(e89.ARMS["awf"], "data_sha256", e89.sha256(tar))
    root, info = e89.resolve_data("awf", str(tmp_path / "data"), tar=tar)
    assert info["sha256"] and e89.manifest_missing(str(tmp_path / "data")) == []
    ws = e89.arm_windows("awf", root)
    assert awf.ROOT == before == "data/awf/dataset/windows/spatial_split"
    assert sorted(w["split"] for w in ws) == ["train"] * 3 + ["val"] * 3, "the two-pixel window is not listed"
    assert all(w["dir"].startswith(str(tmp_path / "data")) for w in ws)


def test_the_awf_module_lists_its_default_root_as_before(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("olmoearth_pretrain")
    from oe_inferencex import awf
    monkeypatch.setattr(awf, "ROOT", str(tmp_path / "nothing_here"))
    assert awf.list_windows() == [], "with no argument it reads ROOT at call time"


# ----------------------------------------------------------------------------- arm F's replica against rslearn's path
def test_arm_f_replica_matches_rslearns_simple_time_series_path(tmp_path):
    """The replica's forward pass against a direct transcription of rslearn's: the 96 channels concatenated layer by
    layer, SimpleTimeSeries' reshape into two 48-channel images, the wrapper's rearrange "b (t c) h w -> b h w t c",
    the mean over timesteps and band sets, the concatenation of the two maps, the PoolingDecoder."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("olmoearth_pretrain")
    from einops import rearrange
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    ck, D = e89.synthetic_checkpoint(str(tmp_path / "f.ckpt"), "fld")
    rep = e89.Replica("fld", str(ck), model_id="OLMOEARTH_V1_NANO", device="cpu")
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 4000, (2, 64, 64, 8, 12)).astype(np.float32)
    ours = rep.logits(x)
    # rslearn: CHW with the eight layers' 12 bands one after another, normalised per band and timestep
    chw = np.stack([np.concatenate([np.moveaxis(rep.normalize(x[b, :, :, t, :]), -1, 0) for t in range(8)])
                    for b in range(2)])                                                   # (2, 96, 64, 64)
    images = torch.tensor(chw, dtype=torch.float32).reshape(2 * 2, 48, 64, 64)            # SimpleTimeSeries
    cur = rearrange(images, "b (t c) h w -> b h w t c", t=4)
    ts = torch.zeros((4, 4, 3), dtype=torch.int32)
    ts[:, :, 0], ts[:, :, 1], ts[:, :, 2] = 1, torch.arange(4)[None, :], 2024
    sample = MaskedOlmoEarthSample(sentinel2_l2a=cur, timestamps=ts, sentinel2_l2a_mask=torch.ones(
        cur.shape[:4] + (rep.n_band_sets,), dtype=torch.int32) * MaskValue.ONLINE_ENCODER.value)
    with torch.no_grad():
        tok = rep.model.encoder(sample, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a
        pooled = rearrange(tok.mean(dim=[3, 4]), "b h w c -> b c h w")
        maps = pooled.reshape(2, 2, pooled.shape[1], 16, 16)
        feat = torch.cat([maps[:, 0], maps[:, 1]], dim=1)                                 # groups [[0], [1]]
        theirs = rep.head(feat).double().numpy()
    assert ours.shape == (2, 10)
    assert np.allclose(ours, theirs, atol=1e-4), np.abs(ours - theirs).max()
    swapped = rep.logits(np.concatenate([x[..., 4:, :], x[..., :4, :]], axis=3))
    assert not np.allclose(ours, swapped, atol=1e-4), "pre and post are not interchangeable"


def test_a_synthetic_attempt_in_the_real_ledger_neither_counts_nor_blocks(tmp_path, monkeypatch):
    """Job 1151373 was refused because the synthetic smoke of job 1151248 had written its gate attempt (a synthetic
    checkpoint, a draft page) into the real ledger. Only attempts on the pinned checkpoint count; the synthetic one
    stays in the ledger as history."""
    import json as _json
    ledger_dir = tmp_path / "ledger"
    monkeypatch.setenv(e89.GATE_LEDGER_ENV, str(ledger_dir))
    pinned = e89.ARMS["awf"]["model_sha256"]
    synthetic = {"attempt": 1, "aligned": False, "checkpoint_sha256": "6ce4" + "0" * 60, "prereg_status": "draft",
                 "accuracy": 0.17}
    real = {"attempt": 2, "aligned": True, "checkpoint_sha256": pinned, "prereg_status": "frozen", "accuracy": 0.88}
    for a in (synthetic, real):
        e89.append_ledger("awf", str(tmp_path), a)
    rec = {"arm": "awf", "attempts": [synthetic, real], "aligned": True, "closed": False, "report_only": True}
    (tmp_path / "exp89_gate_awf.json").write_text(_json.dumps(rec))
    assert e89.counted_attempts("awf", [synthetic, real]) == [real]
    assert e89.require_gate("awf", str(tmp_path))["attempts"][-1]["accuracy"] == 0.88
    # the synthetic attempt alone is no alignment check
    (tmp_path / "exp89_gate_awf.json").write_text(_json.dumps({**rec, "attempts": [synthetic]}))
    e89.ledger_path("awf", str(tmp_path))
    with open(e89.ledger_path("awf", str(tmp_path)), "w") as f:
        f.write(_json.dumps(synthetic, sort_keys=True) + "\n")
    with pytest.raises(e89.GateRefused):
        e89.require_gate("awf", str(tmp_path))


def test_the_synthetic_smoke_never_writes_the_real_ledger(tmp_path, monkeypatch):
    real_dir = tmp_path / "real_ledger"
    monkeypatch.setenv(e89.GATE_LEDGER_ENV, str(real_dir))
    seen = {}

    def fake(args):
        seen["ledger"] = os.environ.get(e89.GATE_LEDGER_ENV)
        e89.append_ledger("awf", str(tmp_path), {"attempt": 1, "checkpoint_sha256": "x"})
        return 0

    monkeypatch.setattr(e89, "_smoke_torch", fake)
    args = type("A", (), {"real": False})()
    assert e89.smoke_torch(args) == 0
    assert seen["ledger"] != str(real_dir) and not real_dir.exists()
    assert os.environ.get(e89.GATE_LEDGER_ENV) == str(real_dir)


def _plain_auroc(score, err):
    """The AUROC for errors by counting every (error, correct) pair, a tie counting one half."""
    score, err = np.asarray(score, dtype=np.float64), np.asarray(err, bool)
    d = score[err][:, None] - score[~err][None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

def test_the_report_only_numbers_by_a_second_route():
    """exp89's report-only numbers recomputed with plain numpy from the per-unit files: the top-1 probability over the
    trained channels, the errors, a mid-rank AUROC for confidence and every control, the capture at each review budget
    by a stable sort, the confidence's AURC as the mean running risk, the clusters that hold an error, and c*(alpha)
    for both certification orders from all labels."""
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp89_summary.json")))["arms"]
    for arm, trained in (("awf", 9), ("fld", 10)):
        z = np.load(os.path.join(ROOT, "exp", "out", f"exp89_units_{arm}.npz"))
        R = rec[arm]
        logits, y = z["logits"].astype(np.float64), z["label"].astype(int)
        t = logits[:, :trained]
        p = np.exp(t - t.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        conf = p.max(axis=1)
        err = logits.argmax(axis=1) != y
        assert (err.size, int(err.sum())) == (R["n_units"], R["n_errors"])
        assert len(np.unique(conf)) == conf.size            # no ties, so plain sorts suffice below
        sig = R["ranking"]["signals"]
        assert abs(_plain_auroc(-conf, err) - sig["confidence"]["auroc"]) < 1e-9, arm
        for name in R["ranking"]["candidates"]:
            assert abs(_plain_auroc(z["controls/" + name], err) - sig[name]["auroc"]) < 1e-9, (arm, name)
        order = np.argsort(-conf, kind="stable")             # most confident first
        for b in (0.05, 0.1, 0.2):
            k = max(1, int(round(b * err.size)))
            assert abs(err[order[::-1]][:k].sum() / err.sum() - sig["confidence"]["capture"][str(b)]) < 1e-12, (arm, b)
        running = np.cumsum(err[order]) / np.arange(1, err.size + 1)
        assert abs(running.mean() - sig["confidence"]["aurc"]) < 1e-9, arm
        clusters = z["clusters"]
        assert (len(np.unique(clusters)), len(np.unique(clusters[err]))) == (
            R["accuracy"]["n_clusters"], R["accuracy"]["n_clusters_with_an_error"])
        for oname, score in (("confidence", conf), ("k3a", 1.0 - z["controls/k3a_no_encoder_uncertainty"])):
            zone = np.argsort(-score, kind="stable")         # descending, ties by index
            for alpha in R["reported"]["alphas"]:
                best = 0.0
                for j in range(1, 21):
                    n = max(1, int(round(j / 20 * zone.size)))
                    if err[zone[:n]].mean() <= alpha + 1e-12:
                        best = round(j / 20, 2)
                assert best == R["certify_best_coverage"][f"{oname}/{alpha:g}"], (arm, oname, alpha)
