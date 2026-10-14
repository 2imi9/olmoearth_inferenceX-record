"""Which of two maps is more accurate, from labels on the windows where they differ (sample --other, estimate).

The interval on the accuracy difference adds each map's exact interval over the differing windows at 97.5%, so it
covers at least 95% by the union bound; ? rows widen each end the way that can only hold more. These tests check the
coverage exactly (multivariate hypergeometric probabilities, no Monte Carlo), with an adversary that marks ? after
seeing each window's truth, and run the route through the command line and the MCP server.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import json
import math

import numpy as np
import pytest

from oe_inferencex import estimate as est


def test_exp90_by_a_second_route():
    """exp90's per-pair truth recomputed from exp79's per-unit files (where present and byte-identical), and the
    coverage of a few pairs computed exactly, by summing the multivariate hypergeometric probabilities of every
    outcome at 50 labels, against the Monte Carlo share the artifact records."""
    import hashlib
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(root, "exp", "out", "exp90_which_map.json")))
    units = os.path.join(root, "exp", "out", "exp79_units")
    cells = [c for c in d["cells"] if c["task"] in ("m_eurosat", "mados") and c["D"]]
    checked = 0
    for c in cells:
        paths = [os.path.join(units, e, c["task"] + ".npz") for e in (c["a"], c["b"])]
        if not all(os.path.exists(p) for p in paths):
            continue
        if any(hashlib.sha256(open(p, "rb").read()).hexdigest() != d["inputs_sha256"][f"{e}/{c['task']}"]
               for p, e in zip(paths, (c["a"], c["b"]))):
            continue
        za, zb = (np.load(p) for p in paths)
        y = za["y"].astype(int)
        da, db = za["dec"].astype(int), zb["dec"].astype(int)
        dis = da != db
        assert (c["N"], c["D"]) == (y.size, int(dis.sum()))
        assert (c["KA"], c["KB"]) == (int((dis & (da == y)).sum()), int((dis & (db == y)).sum()))
        checked += 1
    if checked == 0:
        pytest.skip("exp79_units is not here (it is not committed)")
    # exact coverage at 50 labels for three pairs, by enumeration, beside the recorded Monte Carlo share
    for c in sorted(cells, key=lambda c: c["D"])[:3]:
        N, D, KA, KB, n = c["N"], c["D"], c["KA"], c["KB"], 50
        m = min(n, D)
        lgD = math.lgamma(D + 1) - math.lgamma(m + 1) - math.lgamma(D - m + 1)
        lc = lambda K, k: math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)
        tab = [est.hypergeom_interval(a, m, D, conf=0.975) for a in range(m + 1)]
        cov = 0.0
        for a in range(min(m, KA) + 1):
            for b in range(min(m - a, KB) + 1):
                rest = m - a - b
                if rest > D - KA - KB:
                    continue
                p = math.exp(lc(KA, a) + lc(KB, b) + lc(D - KA - KB, rest) - lgD)
                lo = (tab[a][0] - tab[b][1]) * D / N
                hi = (tab[a][1] - tab[b][0]) * D / N
                cov += p * (lo - 1e-12 <= c["delta"] <= hi + 1e-12)
        assert cov >= 0.95
        se = math.sqrt(cov * (1 - cov) / 400) + 1e-3
        assert abs(cov - c["disagree_50"]["coverage"]) <= 4 * se, (c["task"], c["a"], c["b"], cov)


def test_the_sign_verdict_and_its_probability_are_exp107s_arm_c():
    """Second route: plan's probability under the sign test equals exp107's own implementation (float tails through
    stats.sign_test's critical values, written separately), and estimate's verdict, drawn by the package's sampler on
    constructed maps, matches it within 4 Monte Carlo se."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))
    import exp107_which_map_baselines as e107
    from oe_inferencex import plan
    for D, T, KA, n in ((600, 500, 300, 100), (600, 500, 200, 100), (982, 871, 454, 100), (1467, 1234, 452, 200)):
        assert np.allclose(plan.which_map_probability(5000, D, n, KA, T - KA), e107.sign_arm(D, T, KA, n), atol=1e-12)
    N, D, KA, KB, n, R = 3000, 600, 300, 200, 100, 3000
    rng = np.random.default_rng(7)
    ref = rng.integers(0, 3, N)
    a, b = ref.copy(), ref.copy()
    idx = rng.permutation(N)
    b[idx[:KA]] = (ref[idx[:KA]] + 1) % 3
    a[idx[KA:KA + KB]] = (ref[idx[KA:KA + KB]] + 1) % 3
    a[idx[KA + KB:D]], b[idx[KA + KB:D]] = (ref[idx[KA + KB:D]] + 1) % 3, (ref[idx[KA + KB:D]] + 2) % 3
    named = [0, 0]
    for r in range(R):
        s = est.sample_disagreement(a, b, n, seed=r)
        j = np.asarray(s["indices"], int)
        res = est.compare_from_disagreement(s, a[j], b[j], ref[j])
        assert res["verdict_rule"] == "sign"
        if res["verdict"]:
            named["ab".index(res["verdict"])] += 1
    want = plan.which_map_probability(N, D, n, KA, KB)
    for got, w in zip(named, want):
        assert abs(got / R - w) < 4 * np.sqrt(max(w * (1 - w), 1e-4) / R), (got / R, w)
