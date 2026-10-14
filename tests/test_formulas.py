"""The closed forms the record divides by, each checked against a computation that shares no code with it.

Why this file exists. On 2026-09-20 a recorded claim was false and its own check passed, because the check read the
number back from the artifact that produced it. A check that cannot fail for a reason its author did not anticipate
is not a check. These tests are the counterpart for the formulas: every one is compared against a brute-force
version written here, not against the package's own implementation of the same idea.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import numpy as np
import pytest

from oe_inferencex.metrics import oracle_aurc


def _extract(path, name):
    """Lift one pure-numpy function out of an experiment script without importing it.

    The experiment scripts import torch at module level, so importing them would skip this test in every
    environment that lacks the encoder stack, which is where it is most likely to be run. The copied oracle is
    pure numpy, so the function definition is compiled on its own."""
    import ast, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "exp", path), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns[name]
    raise AssertionError(f"{path} no longer defines {name}(); if it was removed, drop this test with a reason")


@pytest.mark.parametrize("path,name", [
    ("exp13_stat_corrections.py", "oracle_aurc"),
    ("exp18_sen1floods_expert.py", "oracle"),
    ("exp19_v1_vs_v12.py", "oracle"),
])
def test_the_experiment_copies_of_the_oracle_still_equal_the_package(path, name):
    """oracle_aurc is copied verbatim into three experiment scripts. Copies drift; this is what notices."""
    fn = _extract(path, name)
    for n, k in ((1024, 97), (225, 3), (50, 50), (50, 0), (7, 4)):
        assert fn(n, k) == pytest.approx(oracle_aurc(n, k), abs=1e-15), f"exp/{path} has drifted from the package"
