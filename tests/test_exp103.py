"""exp103 (docs/plan/v12_regrade.md): exp98's and exp99's frozen rules applied to the record's v1.2 fine-tune. The
claim ledger's crosschecks: the graded statistics recomputed from the committed per-unit files with plain numpy, by a
route that does not call exp98's or exp99's code (a weighted AUROC and a tie-aware weighted review written here)."""
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_DIR = os.path.join(ROOT, "exp", "out", "exp103", "awf2023")
T_DIR = os.path.join(ROOT, "exp", "out", "exp103", "transfer2017")


def _wauc(score, err, w):
    pos, neg = err > 0, err == 0
    num = sum(wi * (w[neg][score[neg] < si].sum() + 0.5 * w[neg][score[neg] == si].sum())
              for si, wi in zip(score[pos], w[pos]))
    return num / (w[pos].sum() * w[neg].sum())


def _review(score, err, w, b):
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


@pytest.mark.skipif(not os.path.exists(os.path.join(A_DIR, "exp98_units.npz")), reason="exp103 not recorded")
def test_exp103_ai2_points_by_a_second_route():
    S = json.load(open(os.path.join(A_DIR, "exp98_summary.json")))
    assert S["prereg_status"] == "frozen" and S["dirty"] is False and S["commit"].startswith("b83a5b9")
    z = np.load(os.path.join(A_DIR, "exp98_units.npz"))
    V = (z["split"] == "val") & z["covered"]
    err = (z["pred_deployed"][V] != z["label"][V]).astype(float)
    u, w = z["suspicion"][V], np.ones(int(V.sum()))
    assert (int(V.sum()), int(err.sum())) == (259, 32)
    assert round(_wauc(u, err, w), 3) == 0.814 == round(S["prereg"]["P3"]["value"], 3)
    k = round(0.1 * len(u))                      # exp98 reviews round(b n) whole points (26 of 259)
    assert round(_review(u, err, w, k / len(u)), 3) == 0.375 == round(S["prereg"]["P4"]["value"], 3)
    assert [S["prereg"][p]["holds"] for p in ("P3", "P4", "P5")] == [True, True, True]


@pytest.mark.skipif(not os.path.exists(os.path.join(T_DIR, "exp99_units.npz")), reason="exp103 not recorded")
def test_exp103_transfer_plots_by_a_second_route():
    S = json.load(open(os.path.join(T_DIR, "exp99_summary.json")))
    assert S["prereg_status"] == "frozen" and S["dirty"] is False and S["commit"].startswith("b83a5b9")
    z = np.load(os.path.join(T_DIR, "exp99_units.npz"))
    K = z["covered"]
    u, w = z["suspicion"][K], z["weight"][K]
    st, le = z["wrong_strict"][K], z["wrong_lenient"][K]
    assert (int(K.sum()), int(st.sum()), int(le.sum())) == (271, 95, 45)
    assert round(_wauc(u, st, w), 3) == 0.642 and round(_wauc(u, le, w), 3) == 0.730
    cap, ceil = _review(u, st, w, 0.1), min(1.0, 0.1 / ((w * st).sum() / w.sum()))
    assert round((cap - 0.1) / (ceil - 0.1), 3) == 0.267
    assert (S["prereg"]["P1"]["holds"], S["prereg"]["P2"]["holds"]) == (False, True)
