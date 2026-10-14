"""exp107 (docs/plan/which_map_baselines.md): the sign test's critical values, the exact arms against Monte Carlo with the
package's own sampler and interval, the sign test's size under sampling without replacement, the verdicts, the refusal
while a draft. No test computes an arm on the nine recorded pairs."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp107_which_map_baselines as e107  # noqa: E402
from oe_inferencex import estimate as est, stats  # noqa: E402


def test_the_constants_are_the_pages():
    text = open(e107.PLAN, encoding="utf-8").read()
    assert "at least 7 of 9 pairs at 100 labels" in text and "at most 5% of draws" in text
    assert "at least 5 points more often than the tool's interval on at least 5 of 9 pairs" in text
    assert (e107.P1_PAIRS, e107.P2_PAIRS, e107.P2_GAIN, e107.WRONG_MAX, e107.HEADLINE) == (7, 5, 0.05, 0.05, 100)
    assert "at least as often as Az on at least 7 of the 9" in text and "probability at most 0.05 on every pair" in text
    assert "at least 0.05 more often than B on at least 5 of" in text and "C names the other map with probability at most 0.05" in text
    assert "probability 0.8 and 0.9" in text and "n = 50, 200, 400 and 800" in text
    assert (e107.ALPHA, e107.POWERS, e107.BUDGETS) == (0.05, (0.8, 0.9), (50, 100, 200, 400, 800))
    for name, (N, D, KA, KB) in e107.EXPECTED.items():
        assert f"{N:,} | {D:,} | {KA:,} | {KB:,} |" in text, name
    assert e107.load_pairs() == e107.EXPECTED, "the page's table is the recorded Q1 counts"


def test_the_critical_values_are_the_sign_tests():
    for m in range(0, 151):
        k = e107.critical(m)
        for a in range(m + 1):
            rejects = stats.sign_test(a, m - a) < 0.05
            assert rejects == (min(a, m - a) <= k), (m, a)
    for m in range(151, 1101, 7):                     # the boundary only, past the budgets and past 2**1024
        k = e107.critical(m)
        assert stats.sign_test(k, m - k) < 0.05 if k >= 0 else True
        assert stats.sign_test(k + 1, m - k - 1) >= 0.05


def _maps(N, D, KA, KB, seed=0):
    """Constructed maps over N rows: D differing rows (KA where only A agrees, KB where only B does, the rest neither),
    the other rows agreeing with each other (right on most)."""
    rng = np.random.default_rng(seed)
    ref = rng.integers(0, 3, N)
    a, b = ref.copy(), ref.copy()
    idx = rng.permutation(N)
    dA, dB, dN, same = idx[:KA], idx[KA:KA + KB], idx[KA + KB:D], idx[D:]
    b[dA] = (ref[dA] + 1) % 3
    a[dB] = (ref[dB] + 1) % 3
    a[dN] = (ref[dN] + 1) % 3
    b[dN] = (ref[dN] + 2) % 3
    wrong_same = same[: len(same) // 10]
    a[wrong_same] = b[wrong_same] = (ref[wrong_same] + 1) % 3
    return a, b, ref


@pytest.mark.parametrize("KA,KB", [(300, 200), (200, 300)])
def test_the_exact_arms_against_monte_carlo_with_the_package(KA, KB):
    """Second route, both orientations: B by the package's own sampler and interval on constructed maps; A, Az and C by
    drawing rows and the package's sign test (or the z form); each within 4 Monte Carlo se of the exact probabilities."""
    N, D, n, R = 3000, 600, 100, 6000
    a, b, ref = _maps(N, D, KA, KB)
    ex = e107.arms(N, D, KA, KB, n)
    rng = np.random.default_rng(1)
    named = {k: [0, 0] for k in ("A", "Az", "B", "C")}
    diff = np.flatnonzero(a != b)
    for r in range(R):
        s = est.sample_disagreement(a, b, n, seed=r)
        j = np.asarray(s["indices"], int)
        v = est.compare_from_disagreement(s, a[j], b[j], ref[j], rule="interval")["verdict"]
        if v in ("a", "b"):
            named["B"][0 if v == "a" else 1] += 1
        for arm, pool in (("A", np.arange(N)), ("C", diff)):
            i = rng.choice(pool, n, replace=False)
            ba, bb = int(((a[i] == ref[i]) & (b[i] != ref[i])).sum()), int(((b[i] == ref[i]) & (a[i] != ref[i])).sum())
            if stats.sign_test(ba, bb) < 0.05:
                named[arm][0 if ba > bb else 1] += 1
            if arm == "A" and ba + bb > 0 and (ba - bb) ** 2 > e107.Z2 * (ba + bb):
                named["Az"][0 if ba > bb else 1] += 1
    for arm in ("A", "Az", "B", "C"):
        pa, pb = named[arm][0] / R, named[arm][1] / R
        right, wrong = (pa, pb) if KA > KB else (pb, pa)
        for got, want in ((right, ex[arm]["right"]), (wrong, ex[arm]["wrong"])):
            assert abs(got - want) < 4 * np.sqrt(max(want * (1 - want), 1e-4) / R), (arm, got, want)


def test_the_sign_test_keeps_its_size_without_replacement():
    """Under equal counts (K_A = K_B) drawing from a finite set, the sign test names a map with probability at most
    0.05, on a grid of sizes and budgets (exact)."""
    for T in (10, 41, 100, 401, 2000):
        for n in (5, 10, 30, 60, 100, 400, 800):
            if n > T:
                continue
            for KA in (T // 2, (T + 1) // 2):
                pa, pb = e107.sign_arm(T, T, KA, n)
                # equal counts: each direction at most 0.025; one more in A: naming B is at most 0.025
                assert pb <= 0.025 + 1e-12, (T, n, KA, pb)
                if 2 * KA == T:
                    assert pa <= 0.025 + 1e-12, (T, n, KA, pa)


def _rows(right_a, right_b, right_c, wrong=(0.0, 0.0, 0.0)):
    return {f"p{i}": {"budgets": {"100": {"A": {"right": ra, "wrong": wrong[0]}, "Az": {"right": ra, "wrong": wrong[0]},
                                          "B": {"right": rb, "wrong": wrong[1]}, "C": {"right": rc, "wrong": wrong[2]}}}}
            for i, (ra, rb, rc) in enumerate(zip(right_a, right_b, right_c))}


def test_the_verdicts_on_constructed_rows():
    v = e107.verdicts(_rows([0.1] * 9, [0.5] * 7 + [0.05] * 2, [0.56] * 5 + [0.5] * 2 + [0.05] * 2))
    assert v["P1"]["holds"] and v["P2"]["holds"] and len(v["P2"]["pairs_meeting"]) == 5
    v = e107.verdicts(_rows([0.1] * 9, [0.5] * 6 + [0.05] * 3, [0.54] * 9))
    assert not v["P1"]["holds"] and not v["P2"]["holds"] and v["P2"]["decision"] == "the package is unchanged"
    v = e107.verdicts(_rows([0.1] * 9, [0.5] * 9, [0.6] * 9, wrong=(0.0, 0.0, 0.06)))
    assert v["P1"]["holds"] and not v["P2"]["holds"], "C may not name the wrong map more than 5% of the time"
    for w in ((0.06, 0, 0), (0, 0.06, 0)):
        v = e107.verdicts(_rows([0.1] * 9, [0.5] * 9, [0.6] * 9, wrong=w))
        assert not v["P1"]["holds"] and not v["P1"]["wrong_ok"]
    v = e107.verdicts(_rows([0.5] * 9, [0.5] * 7 + [0.4] * 2, [0.5] * 9))
    assert v["P1"]["holds"], "a tie counts as meeting P1"
    rows = _rows([0.1] * 9, [0.5] * 9, [0.5] * 9)
    for i, p in enumerate(rows):
        rows[p]["budgets"]["100"]["Az"]["right"] = 0.6 if i < 3 else 0.1
    assert not e107.verdicts(rows)["P1"]["holds"], "P1 asks B to meet Foody's z form too"


def test_the_smoke_and_the_refusal_while_a_draft(tmp_path, monkeypatch):
    rec = e107.main(["--smoke", "--out-dir", str(tmp_path)])
    big = rec["pairs"]["constructed_large_gap"]["budgets"]["100"]
    assert big["B"]["right"] > big["A"]["right"] and big["C"]["right"] >= big["B"]["right"]
    flat = rec["pairs"]["constructed_no_gap"]["budgets"]["100"]
    assert all(flat[k]["right"] + flat[k]["wrong"] <= 0.05 + 1e-9 for k in "AC")
    monkeypatch.setattr(e107, "prereg_status", lambda path=e107.PLAN: "draft")
    with pytest.raises(SystemExit):
        e107.main([])
