"""Rasters and numbers for the comparison diagram (compare_inferences.tex): the two-period, two-sensor square on one
GEOID-Flood chip where all four cells exist, event EMSR273-1 (Gruemirë on Lake Shkodër, Albania), tile EMSR273-1-0, rows 512-575,
columns 512-575: exp62 chip 7 (its exp60 index is stored in exp/out/exp62_masks.npz). Evidence: the Sentinel-2
composite and the Sentinel-1 pass of the pre-event date, the Sentinel-1 pass after the event (GEOID), and WorldFloods
v2's Sentinel-2 scene of 2018-03-28 (the fourth cell, exp62). Decisions: exp60's three (exp/out/exp60_masks.npz) and
exp62's fourth (exp/out/exp62_masks.npz). The four differences that each isolate one axis, drawn on the scene, and the
label bridge. Imagery: rasters/cmp_square7.npz, cut once on the cluster from the shard tree and the WorldFloods repo
(S1 in dB with exp55's rules, S2 digital numbers). Fixed stretches, surface reflectance 0-1200 DN, top-of-atmosphere 0-1400 DN, VH -32 to -10 dB, so
water is dark in radar (VH) and blue-grey in optics on every panel alike. Numbers from exp/out/exp60_summary.json and
exp/out/exp62_summary.json, written as TeX macros so the figure cannot drift from the ledger.

    uv run python docs/figures/tikz/make_compare_rasters.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_rasters import OUT, classmap, save, tex_list  # noqa: E402

CHIP62, EVENT, TILE = 7, "EMSR273-1", "EMSR273-1-0"
OFFSET = 1                                  # the 14 x 14 W1 windows are windows 1..14 of the 16-cell grid over the 64-px chip
FLOOD = (109, 40, 217)


def cells(mask):
    return [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(mask))]


def optical(dn, idx, offset=0.0, scale=1200.0):
    """Fixed stretch per processing level: surface reflectance 0-1200 DN; top-of-atmosphere L1C with its haze offset removed."""
    return (np.clip((dn[idx] - offset) / scale, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def radar(db):
    """VH backscatter, -32 to -10 dB: open water is dark, land bright, the same stretch on both dates."""
    g = (np.clip((db[1] + 32.0) / 22.0, 0, 1) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    m62 = np.load(os.path.join(ROOT, "exp", "out", "exp62_masks.npz"))
    z60 = np.load(os.path.join(ROOT, "exp", "out", "exp60_masks.npz"))
    assert m62["event"][CHIP62] == EVENT and m62["tile"][CHIP62] == TILE
    k = int(m62["chip_index"][CHIP62])
    ok, ya, yb = m62["ok"][CHIP62], z60["y_permanent"][k], z60["y_after"][k]
    dec = {"A_s2pre": z60["A_s2pre"][k], "A_s1pre": z60["A_s1pre"][k], "B_s1post": z60["B_s1post"][k], "B_s2post": m62["B_s2post"][CHIP62]}
    z = np.load(os.path.join(OUT, "cmp_square7.npz"))
    names = [str(n) for n in z["names"]]
    date = lambda n: re.search(r"_(?:pre|post)_(\d{4})(\d{2})(\d{2})T", n).groups()
    save("cmp_s2_t1.png", optical(z["s2pre"], [2, 1, 0]), 8)                       # encoder order: B02, B03, B04 first
    save("cmp_s2_t2.png", optical(z["wf_S2"].astype(np.float32), [3, 2, 1], scale=1400.0), 8)     # WorldFloods order: B01..B12, B8A ninth; L1C top-of-atmosphere
    save("cmp_s1_t1.png", radar(z["s1pre"]), 8)
    save("cmp_s1_t2.png", radar(z["s1post"]), 8)
    for key, d in dec.items():
        im = classmap(d.astype(float))
        im[~ok] = (200, 200, 200)
        save(f"cmp_{key}.png", im, 32)
    lab = z["label"]
    lab_img = classmap((lab == 1).astype(float))
    lab_img[lab == 2] = FLOOD
    lab_img[lab == 255] = (200, 200, 200)
    save("cmp_label.png", lab_img, 8)
    flooded = yb & ~ya
    pairs = {"SensA": ("A_s2pre", "A_s1pre"), "SensB": ("B_s2post", "B_s1post"), "TimeSS": ("A_s2pre", "B_s2post"), "TimeSO": ("A_s1pre", "B_s1post")}
    macros = {"cmpEvent": EVENT, "cmpDateSTwoPre": "-".join(date(names[0])), "cmpDateSOnePre": "-".join(date(names[1])), "cmpDateSOnePost": "-".join(date(names[2])), "cmpDateSTwoPost": "2018-03-28",
              "cmpWindows": str(int(ok.sum())), "cmpFloodedWindows": str(int((flooded & ok).sum()))}
    for name, (a, b) in pairs.items():
        d = (dec[a] != dec[b]) & ok
        tex_list(f"cmp_{name.lower()}.tex", f"cmpCells{name}", cells(d))
        macros[f"cmpN{name}"] = str(int(d.sum()))
        macros[f"cmpFl{name}"] = f"{100 * flooded[d].mean():.0f}" if d.any() else "0"
    tex_list("cmp_time_flooded.tex", "cmpCellsTimeFlooded", cells((dec["A_s1pre"] != dec["B_s1post"]) & ok & flooded))
    tex_list("cmp_time_error.tex", "cmpCellsTimeError", cells((dec["A_s1pre"] != dec["B_s1post"]) & ok & ~flooded))
    R60 = json.load(open(os.path.join(ROOT, "exp", "out", "exp60_summary.json")))["results"]["pairs"]
    S62 = json.load(open(os.path.join(ROOT, "exp", "out", "exp62_summary.json")))
    R62 = S62["results"]["pairs"]
    macros.update({"cmpPooledSensPreRate": f"{100 * R60['sensor_only']['disagreement_rate']:.1f}", "cmpPooledSensPreFlooded": f"{100 * R60['sensor_only']['flooded_share']:.1f}",
                   "cmpPooledTimeSORate": f"{100 * R60['time_only']['disagreement_rate']:.1f}", "cmpPooledTimeSOFlooded": f"{100 * R60['time_only']['flooded_share']:.0f}",
                   "cmpSquareChips": str(S62["config"]["fourth_cell"]["chips_with_clear_post_optical"]), "cmpSquareEvents": str(len(S62["config"]["fourth_cell"]["events"])),
                   "cmpSquareSensBFlooded": f"{100 * R62['sensor_only_post']['flooded_share']:.0f}", "cmpSquareTimeSOFlooded": f"{100 * R62['time_only_s1']['flooded_share']:.0f}"})
    with open(os.path.join(OUT, "cmp_numbers.tex"), "w") as f:
        for key, v in macros.items():
            f.write(f"\\def\\{key}{{{v}}}\n")
    print(f"chip {CHIP62} ({EVENT}, exp60 index {k}): {int(ok.sum())} valid windows, {int((flooded & ok).sum())} flooded; differences " +
          ", ".join(f"{n} {macros['cmpN' + n]} ({macros['cmpFl' + n]}% flooded)" for n in pairs))


if __name__ == "__main__":
    main()
