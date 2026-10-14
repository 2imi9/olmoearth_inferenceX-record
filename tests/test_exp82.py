"""exp82's cues and bootstrap, checked by a second route (docs/plan/cue_verification.md).

The boundary cue is recomputed with explicit neighbour shifts on the tile grid, no call into the package; the
low-confidence cue's share among errors must equal exp70's recorded capture at 20% (the gate); and the tile-count
bootstrap equals explain.cue_enrichment's window bootstrap on MADOS, where the latter is affordable."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
UNITS = os.path.join(ROOT, "exp", "out", "exp78_units")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(UNITS, "mados.npz")), reason="exp78 export not present")


def _boundary_by_shifts(hard, valid):
    """A window sits on a boundary when any of its 8 in-grid, valid neighbours holds another class."""
    N, H, W = hard.shape
    out = np.zeros(hard.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            for i in range(N):
                for r in range(H):
                    rr = r + dy
                    if not 0 <= rr < H:
                        continue
                    for c in range(W):
                        cc = c + dx
                        if not 0 <= cc < W or not valid[i, r, c] or not valid[i, rr, cc]:
                            continue
                        if hard[i, rr, cc] != hard[i, r, c]:
                            out[i, r, c] = True
    return out[valid]


def test_the_boundary_cue_equals_an_explicit_neighbour_scan_on_mados_tiles():
    import exp82_cue_verification as e82
    u = e82.load_export("mados", UNITS)
    n = 60                                                    # the first sixty tiles: 20 x 20 windows each
    keep = np.arange(u["hard"].shape[0]) < n
    hard, valid = u["hard"][:n], u["valid"][:n]
    pkg = e82.boundary_cue(hard, valid)
    ref = _boundary_by_shifts(hard, valid)
    assert pkg.shape == ref.shape and np.array_equal(pkg, ref)
    assert 0 < pkg.mean() < 1


def test_the_low_confidence_cue_reproduces_exp70s_capture_at_twenty_percent():
    import exp82_cue_verification as e82
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))["results"]["tasks"]
    for task in ("mados", "sen1floods11"):
        u = e82.load_export(task, UNITS)
        low = e82.low_confidence_cue(u["margin"])
        share = float(low[u["err"] > 0.5].mean())
        assert abs(share - rec[task]["signals"]["margin"]["capture"]["0.2"]) <= e82.GATE_TOL


def test_the_tile_count_bootstrap_matches_the_window_bootstrap_on_mados():
    import exp82_cue_verification as e82
    from oe_inferencex.explain import cue_enrichment
    u = e82.load_export("mados", UNITS)
    bnd = e82.boundary_cue(u["hard"], u["valid"])
    a = e82.enrichment_from_counts(e82.per_tile_counts(bnd, u["err"], u["tile"]), n_boot=400, seed=1)
    b = cue_enrichment(bnd, u["err"], clusters=u["tile"], n_boot=400, seed=1)
    for k in ("share_errors", "share_correct", "enrichment", "precision", "n", "n_errors", "n_with_cue"):
        assert abs(a[k] - b[k]) < 1e-12, k
    assert abs(a["boot_lo"] - b["boot_lo"]) < 0.05 * a["enrichment"] and abs(a["boot_hi"] - b["boot_hi"]) < 0.05 * a["enrichment"]


def test_coarsening_is_majority_pooling_by_brute_force():
    """The addendum's coarse grid: each k x k block's decision and reference are the majority over its valid fine
    windows, ties to the smallest class; validity is any valid fine window; the confidence is the mean margin."""
    import exp82_cue_verification as e82
    rng = np.random.default_rng(3)
    N, H, W, C, k = 3, 8, 12, 4, 4
    valid = rng.random((N, H, W)) > 0.3
    hard = rng.integers(0, C, (N, H, W)); hard[~valid] = 0
    y = rng.integers(0, C, int(valid.sum())); margin = rng.random(int(valid.sum()))
    u = {"hard": hard, "valid": valid, "y": y, "margin": margin, "n_classes": C, "family": "t",
         "tile": np.broadcast_to(np.arange(N)[:, None, None], (N, H, W))[valid]}
    c = e82.coarsen_export(u, k)
    full_y = np.zeros((N, H, W), int); full_y[valid] = y
    full_m = np.zeros((N, H, W)); full_m[valid] = margin
    got = iter(zip(c["dec"], c["y"], c["margin"], c["err"], c["tile"]))
    for i in range(N):
        for r in range(H // k):
            for q in range(W // k):
                v = valid[i, r * k:(r + 1) * k, q * k:(q + 1) * k]
                if not v.any():
                    continue
                d, yy, m, e, t = next(got)
                maj = lambda a: int(np.argmax(np.bincount(a[v], minlength=C)))     # argmax: the smallest class on a tie
                assert d == maj(hard[i, r * k:(r + 1) * k, q * k:(q + 1) * k]) and yy == maj(full_y[i, r * k:(r + 1) * k, q * k:(q + 1) * k])
                assert abs(m - full_m[i, r * k:(r + 1) * k, q * k:(q + 1) * k][v].mean()) < 1e-12 and e == float(d != yy) and t == i
    assert next(got, None) is None
    with pytest.raises(ValueError):
        e82.coarsen_export(u, 5)
