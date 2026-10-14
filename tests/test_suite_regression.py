"""The suite regression runner's torch-free path: candidate scoring beside the margin, and the per-model verdict."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_runner_smoke():
    spec = importlib.util.spec_from_file_location("suite_regression", os.path.join(ROOT, "scripts", "suite_regression.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.smoke()
