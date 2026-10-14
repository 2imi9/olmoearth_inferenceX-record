"""exp108 (report-only): the two conformal arms keep their own guarantees by exact enumeration on small maps, and
certify's arm in the harness equals the package's certify_zone; the smoke runs."""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp108_zone_vs_conformal as e108  # noqa: E402


def _exact(err, n, alpha):
    """Over every labelled subset of size n: the mean share of unlabelled windows wrong and in CRC's zone, and the mean
    share of wrong windows among BH's selected unlabelled windows (0 when none)."""
    N = err.size
    joint, fdp = [], []
    for sub in itertools.combinations(range(N), n):
        pos = np.array(sub)
        w = err[pos]
        e = e108.crc_joint_end(pos, w, n, N, alpha)
        unl = np.setdiff1d(np.arange(e), pos)
        joint.append(err[unl].sum() / (N - n))
        e2, r = e108.bh_select_end(pos, w, n, N, alpha)
        unl2 = np.setdiff1d(np.arange(e2), pos)
        assert unl2.size == r
        fdp.append(err[unl2].mean() if r else 0.0)
    return float(np.mean(joint)), float(np.mean(fdp))


def test_the_conformal_arms_keep_their_guarantees_exactly_on_small_maps():
    rng = np.random.default_rng(3)
    for N, n, alpha in ((10, 4, 0.3), (11, 5, 0.25), (12, 6, 0.2)):
        for _ in range(6):
            err = (rng.random(N) < np.linspace(0.05, 0.7, N)).astype(float)
            joint, fdp = _exact(err, n, alpha)
            assert joint <= alpha + 1e-12, (N, n, alpha, joint)
            assert fdp <= alpha + 1e-12, (N, n, alpha, fdp)


def test_crc_and_bh_ends_on_a_constructed_draw():
    pos = np.array([0, 2, 5, 7, 9])
    w = np.array([0, 1, 0, 1, 0])                # labelled errors at positions 2 and 7
    # alpha (n + 1) - 1 = 0.4 * 6 - 1 = 1.4 -> m = 1: the zone ends before the second labelled error (position 7)
    assert e108.crc_joint_end(pos, w, 5, 12, 0.4) == 7
    assert e108.crc_joint_end(pos, w, 5, 12, 0.1) == 0          # m < 0: empty
    end, r = e108.bh_select_end(pos, w, 5, 12, 0.5)
    # segment 0 ends at 2 with 1 unlabelled ahead: (1)/6 <= 0.5 * 1/7? no; segment 1 ends at 7 with 4 ahead: 2/6 <= 0.5*4/7 = .286? no
    # segment 2 ends at 12 with 7 ahead: 3/6 <= 0.5 * 7/7 = 0.5 -> yes
    assert (end, r) == (12, 7)


def test_certify_arm_equals_the_package():
    rng = np.random.default_rng(5)
    N, B = 2000, 300
    err = (rng.random(N) < np.linspace(0.01, 0.4, N)).astype(float)
    alpha = float(err.mean() / 2)
    margin = np.linspace(1, 0, N)                     # zone order = index order
    from oe_inferencex import estimate as est
    cov, sizes, _ = est.zone_levels(N, B, alpha, e108.DELTA, cut="standard")
    for seed in range(20):
        pos = np.sort(np.random.default_rng(seed).choice(N, B, replace=False))
        b, k = est.zone_counts(pos, err[pos], sizes)
        p = [est.zone_pvalue(kk, bb, nn, alpha) for bb, kk, nn in zip(b, k, sizes)]
        _, best = est.apply_zone_rule(p, b, k, alpha, e108.DELTA, "prefix")
        z = est.certify_zone(margin, pos, err[pos], alpha, e108.DELTA)
        got = z.get("coverage") if isinstance(z, dict) else None
        want = None if best is None else cov[best]
        assert (got is None and want is None) or (got is not None and want is not None and abs(got - want) < 1e-9), (seed, got, want)


def test_the_smoke(tmp_path):
    rec = e108.main(["--smoke", "--out-dir", str(tmp_path), "--draws", "50"])
    c = rec["cells"]["synthetic"]["300"]
    assert set(c) >= {"certify", "crc_joint", "bh_select"}
