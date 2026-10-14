"""Signals and pixel controls (oe_inferencex.signals): properties, and equality with the experiment
modules that produced the recorded results when the encoder stack is installed.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import os
import sys

import numpy as np
import pytest

from oe_inferencex.signals import S2_BANDS, aligned_tile_phase, boundary_indicator, ndwi_gradient

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp")


# ----------------------------------------------------------------------------- equality with the experiment code
@pytest.fixture(scope="module")
def exp_modules():
    pytest.importorskip("olmoearth_pretrain")
    sys.path.insert(0, EXP)
    import exp13_stat_corrections as exp13
    import exp18_sen1floods_expert as exp18
    return exp13, exp18


def test_band_order_matches_the_encoder(exp_modules):
    from olmoearth_pretrain.data.constants import Modality
    assert tuple(Modality.SENTINEL2_L2A.band_order) == S2_BANDS


def test_equals_exp13_and_exp18_implementations(exp_modules):
    exp13, exp18 = exp_modules
    rng = np.random.default_rng(123)
    ps = [rng.random((exp13.GRID, exp13.GRID)) for _ in exp13.SHIFTS]
    assert np.allclose(aligned_tile_phase(ps), exp13.aligned_tile_phase(ps))
    img = rng.integers(0, 6000, (12, exp13.SIZE + exp13.PAD, exp13.SIZE + exp13.PAD)).astype(np.int32)
    assert np.allclose(ndwi_gradient(img, size=exp13.SIZE), exp13.ndwi_gradient(img))
    tiles = rng.integers(0, 6000, (3, 12, 64, 64)).astype(np.float32)
    assert np.allclose(ndwi_gradient(tiles, size=exp18.CROP), exp18.ndwi_gradient(tiles))
    p = rng.random((2, exp18.G, exp18.G))
    assert np.array_equal(boundary_indicator(p, probabilities=True), exp18.boundary(p))
    ps60 = np.stack([rng.random((2, exp18.G, exp18.G)) for _ in exp13.SHIFTS]).astype(np.float32)
    assert np.array_equal(aligned_tile_phase(ps60, dtype=np.float32), exp18.aligned_tile_phase(ps60))   # bit for bit
    assert np.allclose(aligned_tile_phase(ps60), exp18.aligned_tile_phase(ps60), atol=1e-6)               # float64 form
