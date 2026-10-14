"""Cross-unit tests (oe_inferencex.stats): exact values, and the recorded headline tests recomputed from
the committed exp13 table.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import csv
import os
import pytest

from oe_inferencex.stats import clustered_sign_test, sign_test, wins_losses_ties

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")
RIVER = {"barotse": "Zambezi", "delta": "Zambezi", "kazungula": "Zambezi", "vicfalls_up": "Zambezi",
         "zambezi_20": "Zambezi", "zambezi_50": "Zambezi", "zambezi_80": "Zambezi",
         "cuando_20": "Cuando", "cuando_50": "Cuando", "cuando_80": "Cuando",
         "kafue_20": "Kafue", "kafue_50": "Kafue", "kafue_80": "Kafue", "luangwa_conf": "Luangwa",
         "okavango_50": "Okavango", "okavango_80": "Okavango", "okavango_sep": "Okavango",
         "rovuma_20": "Rovuma", "rovuma_50": "Rovuma", "rovuma_80": "Rovuma",
         "save_20": "Save", "save_50": "Save", "save_80": "Save",
         "shire_20": "Shire", "shire_50": "Shire", "shire_80": "Shire", "shire_liwonde": "Shire"}


def _exp13_table():
    path = os.path.join(OUT, "exp13_corrected_stats.csv")
    if not os.path.exists(path):
        pytest.skip("exp13 table not committed")
    per = {}
    for r in csv.DictReader(open(path)):
        per.setdefault(r["scene"], {})[r["signal"]] = float(r["eaurc"])
    return per


def test_exp13_headline_tests_from_the_committed_table():
    per = _exp13_table()
    gains = {s: v["baseline"] - v["tile-phase (aligned)"] for s, v in per.items()}
    w, l, t = wins_losses_ties(list(gains.values()))
    assert (w, l, t) == (26, 1, 0) and sign_test(w, l) == pytest.approx(4.17e-07, rel=0.01)   # protocol: 26/27, p = 4e-07
    river = clustered_sign_test(gains, RIVER, aggregate="majority", alternative="two-sided")
    assert (river["w"], river["l"], river["n_clusters"]) == (8, 0, 8) and river["p"] == pytest.approx(0.0078, abs=1e-4)
    assert clustered_sign_test(gains, RIVER)["p"] == pytest.approx(1 / 256)               # mean aggregate, one-sided
    e_dist = {s: v["baseline"] - v["E_dist"] for s, v in per.items()}
    assert wins_losses_ties(list(e_dist.values()))[:2] == (13, 14)                          # ledger: E_dist 13/27
