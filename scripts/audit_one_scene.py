#!/usr/bin/env python
"""Audit one scene end to end.

Fetches a Sentinel-2 window and its ESA WorldCover reference, embeds the window with the frozen OlmoEarth encoder,
trains a small water head on a nearby scene, predicts, and scores the audit signals against the reference with
risk-coverage curves. Writes exp/out/exp02_full_slice.png and exp/out/exp02_full_slice.json.

This is the named entry point for the numbered experiment exp/exp02_full_slice.py, which stays the record of that
experiment; the two run the same code. Run from the repository root, in the full experiment environment:

    uv sync --extra encoder --extra geo
    uv run python scripts/audit_one_scene.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp02_full_slice  # noqa: E402

if __name__ == "__main__":
    exp02_full_slice.main()
