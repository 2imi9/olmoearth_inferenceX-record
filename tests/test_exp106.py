"""exp106 (docs/plan/second_opinion.md): the scores, the confident rows, the verdicts and the inventory, against
second routes on synthetic rows; the grade is refused while the page is a draft. No test reads a reference label
together with the disagreement or the confidence of the real cells."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp106_second_opinion as e106  # noqa: E402


def test_the_constants_are_the_pages():
    text = open(e106.PLAN, encoding="utf-8").read()
    assert "AUROC(c) − AUROC(u) ≥ 0.02 on at least 2 of the 3 cells" in text
    assert "f_err ≥ 2 f_ok on all" in text and "f_err > 0 there" in text and "at least 20" in text
    assert "2,000 resamples, seed 106" in text
    assert (e106.P1_GAIN, e106.P1_CELLS, e106.P2_RATIO, e106.P2_FLOOR, e106.R_BOOT, e106.SEED) == (0.02, 2, 2.0, 20, 2000, 106)
    for n in ("12,574", "4,807", "4,794"):
        assert n in text
    assert e106.EXPECTED_ROWS == {"east_africa_cgls": 12574, "europe_odse": 4807, "conus_lcmap": 4794}


def _brute_auroc(s, e):
    P, N = s[e], s[~e]
    return float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (P.size * N.size))


def test_the_scores_and_the_auroc_by_a_second_route():
    cell = e106.synthetic_cell(seed=3, n=600)
    err = cell["prod"] != cell["ref"]
    u, d, c = e106.scores(cell)
    assert np.allclose(u, 101 - cell["conf"]) and set(np.unique(d)) <= {0.0, 1.0}
    # every disputed row above every undisputed one, and within a group the least confident first
    assert c[d == 1].min() > c[d == 0].max()
    m = d == 0
    assert np.array_equal(np.argsort(c[m], kind="stable"), np.argsort(u[m], kind="stable"))
    for s in (u, d, c):
        assert e106.auroc(s, err) == pytest.approx(_brute_auroc(s, err), abs=1e-12)


def test_the_confident_rows_and_the_flag_shares():
    conf = np.array([90, 80, 70, 60, 95, 85, 50, 99], float)
    ref = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    prod = np.array([1, 1, 1, 1, 2, 2, 2, 2])          # rows 4-7 are errors
    sec = np.array([1, 2, 1, 1, 1, 1, 2, 1])
    cell = dict(ref=ref, prod=prod, conf=conf, sec=sec, group=np.zeros(8, int), boot="simple")
    err = prod != ref
    _, d, _ = e106.scores(cell)
    fs = e106.flag_shares(cell, err, d)
    assert fs["median_correct_confidence"] == 75.0     # median of 90, 80, 70, 60
    # confident errors: conf >= 75 among errors -> rows 4 (95), 5 (85), 7 (99); flagged where prod != sec: 4, 5, 7 all
    assert fs["confident_errors"] == 3 and fs["f_err"] == 1.0
    # confident correct: rows 0 (90) and 1 (80); flagged: row 1 only
    assert fs["confident_correct"] == 2 and fs["f_ok"] == 0.5 and fs["ratio"] == 2.0


def _rows(gains, ratios, conf_errors=(50, 50, 50)):
    return {c: {"gain": g, "confident": {"f_err": r * 0.1, "f_ok": 0.1, "confident_errors": n}}
            for c, g, r, n in zip(e106.CELLS, gains, ratios, conf_errors)}


def test_the_verdicts_on_constructed_rows():
    v = e106.verdicts(_rows([0.03, 0.02, 0.0], [2.5, 2.0, 2.1]))
    assert v["P1"]["holds"] and len(v["P1"]["cells_meeting"]) == 2 and v["P2"]["holds"]
    v = e106.verdicts(_rows([0.03, 0.019, 0.0], [2.5, 1.9, 2.1]))
    assert not v["P1"]["holds"] and not v["P2"]["holds"] and v["P2"]["determined"]
    v = e106.verdicts({c: {"gain": 0.05, "confident": {"f_err": 0.0, "f_ok": 0.0, "confident_errors": 50}} for c in e106.CELLS})
    assert not v["P2"]["holds"], "a flag that never fires among confident rows does not pass P2"
    v = e106.verdicts(_rows([0.05, 0.05, 0.05], [3, 3, 3], conf_errors=(50, 19, 50)))
    assert v["P1"]["holds"] and v["P2"]["by_cell"]["europe_odse"] is None
    assert not v["P2"]["holds"] and not v["P2"]["determined"], "a cell under the floor is not determined, P2 cannot hold"


def test_the_bootstrap_resamples_within_countries_and_by_tiles():
    rng = np.random.default_rng(0)
    g = np.repeat(np.array(list("abcdefg")), [10, 20, 30, 5, 5, 15, 15])
    i = e106.boot_index(g, rng, "within each country")
    assert all((g[i] == x).sum() == (g == x).sum() for x in "abcdefg"), "within each country"
    tiles = np.repeat(np.arange(30).astype(str), 4)
    j = e106.boot_index(tiles, rng, "by tile")
    assert j.size % 4 == 0 and all((tiles[j] == t).sum() % 4 == 0 for t in np.unique(tiles)), "whole tiles"
    k = e106.boot_index(np.zeros(50, int), rng, "simple")
    assert k.size == 50
    few = np.repeat(np.arange(5).astype(str), 3)
    jj = e106.boot_index(few, rng, "by tile")
    assert all((few[jj] == t).sum() % 3 == 0 for t in np.unique(few)), "five tiles are still resampled whole"
    with pytest.raises(ValueError):
        e106.boot_index(few, rng, "other")


def test_the_smoke_and_the_refusal_while_a_draft(tmp_path, monkeypatch):
    rec = e106.main(["--smoke", "--out-dir", str(tmp_path), "--draws", "20"])
    assert set(rec["cells"]) == set(e106.CELLS)
    assert rec["cells"]["east_africa_cgls"]["gain"] > 0, "an informative second product raises the order"
    monkeypatch.setattr(e106, "prereg_status", lambda path=e106.PLAN: "draft")
    with pytest.raises(SystemExit):
        e106.main([])


def test_the_cells_load_with_the_pages_row_counts():
    """Label-free: the rows each cell keeps (presence of the columns, not their agreement)."""
    cells = e106.load_cells()
    assert {k: v["ref"].size for k, v in cells.items()} == e106.EXPECTED_ROWS
    assert np.unique(cells["east_africa_cgls"]["group"]).size == 7
    assert np.unique(cells["europe_odse"]["group"]).size == 55
    assert [cells[c]["boot"] for c in e106.CELLS] == ["within each country", "by tile", "simple"]
    for v in cells.values():
        assert v["conf"].min() >= 0 and v["conf"].max() <= 100
    assert {k: v["ref"].size for k, v in e106.load_control().items()} == e106.EXPECTED_CONTROL


def test_the_decomposition_identity_and_the_tied_part_by_a_second_route():
    """AUROC(c) - AUROC(u) equals r(1-q)(1-A10) - (1-r)q A01, and the tied part equals the gain counted over the
    error/correct pairs u ties, by brute force; on synthetic rows with heavy ties, as in cell 3."""
    rng = np.random.default_rng(9)
    n = 500
    err = rng.random(n) < 0.2
    conf = np.where(rng.random(n) < 0.5, 99.0, rng.integers(40, 100, n).astype(float))
    d = (rng.random(n) < np.where(err, 0.5, 0.15)).astype(float)
    u, c = 101 - conf, d + (101 - conf) / 102
    dec = e106.decomposition(u, d, err)
    gain = _brute_auroc(c, err) - _brute_auroc(u, err)
    assert dec["identity_gain"] == pytest.approx(gain, abs=1e-12)
    P, N = np.flatnonzero(err), np.flatnonzero(~err)
    tied = sum((np.sign(d[i] - d[N]))[u[i] == u[N]].sum() * 0.5 for i in P) / (P.size * N.size)
    assert dec["gain_from_pairs_u_ties"] == pytest.approx(tied, abs=1e-12)


def test_the_control_crosswalk_on_a_tiny_file(tmp_path, monkeypatch):
    f = tmp_path / "nlcd.csv"
    f.write_text("set,year,ref_nlcd,nlcd,lcmap,lcpconf,weight\n"
                 "A,2001,21,90,1,50,1\nA,2001,90,21,6,99,1\nA,2001,99,21,1,50,1\nA,2001,21,21,1,0,1\nA,2001,21,21,1,150,1\n")
    monkeypatch.setattr(e106, "NLCD", str(f))
    ctl = e106.load_control()
    assert list(ctl) == ["nlcd_A2001"]
    c = ctl["nlcd_A2001"]
    assert c["ref"].tolist() == [1, 6] and c["sec"].tolist() == [6, 1] and c["conf"].tolist() == [50.0, 99.0]


def test_exp106_recorded_by_a_second_route():
    """The recorded cells rebuilt from the point files with code written here (csv, numpy ranks; no import of exp106's
    scoring or the package's metrics): rows, AUROCs of u and c, the gain, and the confident flag shares."""
    import csv
    import json
    S = json.load(open(os.path.join(ROOT, "exp", "out", "exp106_summary.json"), encoding="utf-8"))
    assert S["prereg_status"] == "frozen" and S["dirty"] is False and S["commit"].startswith("5169e46")
    out = os.path.join(ROOT, "exp", "out")

    def num(v, missing):
        v = (v or "").strip()
        return float(v) if v else missing

    def rows(path, keep, cols):
        R = [r for r in csv.DictReader(open(os.path.join(out, path), encoding="utf-8")) if keep(r)]
        return [np.array([num(r[c], m) for r in R]) for c, m in cols]

    ea = rows("exp94_eastafrica_points.csv", lambda r: r["year"] == "2017",
              (("ref", 0), ("cgls", 0), ("cgls_proba", 255), ("esri", 0)))
    eu = rows("exp94_europe_points.csv", lambda r: r["in_subset"] == "1" and r["odse_prob"].strip() != "",
              (("ref", 0), ("odse", 0), ("odse_prob", -1), ("esri", 0)))
    us = rows("exp92_plots.csv", lambda r: r["in_conf_subset"] == "1", (("ref", 0), ("lcmap", 0), ("lcpconf", 0), ("esri", 0)))

    def rank_auroc(score, pos):
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty(score.size)
        sorted_s = score[order]
        i = 0
        while i < score.size:                                  # average ranks over ties
            j = i
            while j + 1 < score.size and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        n1, n0 = pos.sum(), (~pos).sum()
        return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    for name, (ref, prod, conf, sec), lo in (("east_africa_cgls", ea, 0), ("europe_odse", eu, 0), ("conus_lcmap", us, 1)):
        k = (ref > 0) & (prod > 0) & (sec > 0) & (conf >= lo) & (conf <= 100)
        ref, prod, conf, sec = ref[k], prod[k], conf[k], sec[k]
        err = prod != ref
        u = 101 - conf
        d = (prod != sec).astype(float)
        c = d + u / 102
        R = S["cells"][name]
        assert R["n"] == err.size and R["errors"] == int(err.sum())
        assert rank_auroc(u, err) == pytest.approx(R["auroc_own"], abs=1e-12)
        assert rank_auroc(c, err) == pytest.approx(R["auroc_combined"], abs=1e-12)
        m = np.median(conf[~err])
        ce, cok = err & (conf >= m), ~err & (conf >= m)
        assert d[ce].mean() == pytest.approx(R["confident"]["f_err"], abs=1e-12)
        assert d[cok].mean() == pytest.approx(R["confident"]["f_ok"], abs=1e-12)
    V = S["verdicts"]
    gains = {c: S["cells"][c]["gain"] for c in e106.CELLS}
    assert V["P1"]["holds"] == (sum(g >= 0.02 for g in gains.values()) >= 2) is True
    assert V["P2"]["holds"] is False and S["cells"]["east_africa_cgls"]["confident"]["ratio"] < 2
