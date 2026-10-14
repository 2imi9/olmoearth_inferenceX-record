"""Rasters for the TikZ diagrams, from the repository's own data: one WorldCover rule scene (okavango_80, exp11
scene archive + exp37 per-window table) and one Sen1Floods11 Bolivia tile (tile 218, hand labels + exp37 table).
Writes PNGs and small TeX lists (review-set and error windows) into docs/figures/tikz/rasters/.

    uv run python docs/figures/tikz/make_rasters.py
"""
import os
import sys

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from oe_inferencex.assess import boundary_first_score, review_mask  # noqa: E402
from oe_inferencex.signals import S2_BANDS, ndwi_gradient  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures", "tikz", "rasters")
SCENE, TILE, BUDGET = "okavango_80", 218, 0.05
WATER, LAND = (37, 99, 235), (243, 239, 226)          # prediction / label palette
ERR = (220, 38, 38)


def save(name, arr, scale):
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(os.path.join(OUT, name))


def stretch(band, lo=2, hi=98):
    a, b = np.percentile(band, [lo, hi])
    return np.clip((band - a) / max(b - a, 1e-6), 0, 1)


def rgb(img12, size):
    """True colour from the S2_BANDS order (B04, B03, B02), percentile-stretched, cropped to size."""
    i = [S2_BANDS.index(b) for b in ("B04", "B03", "B02")]
    x = np.stack([stretch(img12[k, :size, :size]) for k in i], -1)
    return (x * 255).astype(np.uint8)


def gray(band, size):
    return (stretch(band[:size, :size]) * 255).astype(np.uint8)


def classmap(binary):
    out = np.zeros(binary.shape + (3,), np.uint8)
    out[binary > 0.5] = WATER
    out[binary <= 0.5] = LAND
    return out


def cmap(arr, name, vmin=None, vmax=None, valid=None):
    a = np.asarray(arr, dtype=np.float64)
    lo = np.nanmin(a[valid]) if valid is not None else np.nanmin(a)
    hi = np.nanmax(a[valid]) if valid is not None else np.nanmax(a)
    if vmin is not None:
        lo = vmin
    if vmax is not None:
        hi = vmax
    x = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = (colormaps[name](x) * 255).astype(np.uint8)
    if valid is not None:
        rgba[~valid] = (255, 255, 255, 255)
    return rgba[..., :3]


def tex_list(name, macro, rows_cols):
    """A TeX file defining \\macro as a r/c list, for \\foreach in the figures (no extra packages needed)."""
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\\def\\" + macro + "{" + ", ".join(f"{r}/{c}" for r, c in rows_cols) + "}\n")


def grid_from_table(z, mask, G, cols):
    """Per-window arrays (G, G) from the exp37 table rows in `mask`; NaN where no row."""
    out = {c: np.full((G, G), np.nan) for c in cols}
    r, c = z["row"][mask].astype(int), z["col"][mask].astype(int)
    for k in cols:
        out[k][r, c] = z[k][mask]
    return out


def scene():
    zs = np.load(os.path.join(ROOT, "exp", "out", "exp11_scenes.npz"), allow_pickle=True)
    zt = np.load(os.path.join(ROOT, "exp", "out", "exp37_patches_scenes.npz"))
    img, lab = zs[f"{SCENE}_img"], zs[f"{SCENE}_lab"]
    G, size = lab.shape[0], lab.shape[0] * 4
    m = zt["scene"] == SCENE
    g = grid_from_table(zt, m, G, ["err", "conf", "tile_phase", "boundary", "ndwi_level", "dihedral"])
    valid = ~np.isnan(g["err"])
    pred = np.where(g["err"] > 0.5, 1 - lab, lab)
    save("scene_rgb.png", rgb(img, size), 4)
    save("scene_b08.png", gray(img[S2_BANDS.index("B08")], size), 4)
    save("scene_b11.png", gray(img[S2_BANDS.index("B11")], size), 4)
    save("scene_label.png", classmap(lab), 16)
    save("scene_pred.png", classmap(pred), 16)
    save("scene_conf.png", cmap(g["conf"], "viridis", valid=valid), 16)          # -|logit|: bright = least confident
    save("scene_boundary.png", cmap(g["boundary"], "Oranges", vmin=0, vmax=1, valid=valid), 16)
    save("scene_tilephase.png", cmap(g["tile_phase"], "magma", valid=valid), 16)
    save("scene_ndwi.png", cmap(g["ndwi_level"], "cividis", valid=valid), 16)   # -|NDWI|: bright = ambiguous
    save("scene_dihedral.png", cmap(g["dihedral"], "magma", valid=valid), 16)
    save("scene_ndwigrad.png", cmap(ndwi_gradient(img, patch=4, size=size), "Greys"), 16)   # the no-model control
    rs = review_mask(boundary_first_score(g["conf"], g["boundary"]), valid, BUDGET)
    tex_list("scene_review.tex", "sceneReview", zip(*np.nonzero(rs)))
    tex_list("scene_errors.tex", "sceneErrors", zip(*np.nonzero(g["err"] > 0.5)))
    print(f"scene {SCENE}: {G}x{G} windows, {int((g['err'] > 0.5).sum())} errors, review set {int(rs.sum())} windows, "
          f"{int((g['err'][rs] > 0.5).sum())} of them errors")


def bolivia():
    d = torch.load(os.path.join(ROOT, "data", "floods", "flood_bolivia_data.pt"), weights_only=True)
    s2 = d["s2"].numpy().astype(np.float32)          # (N, 13, 64, 64) in Sentinel-2 L1C band order
    lab = d["labels"].numpy()[:, 0]                  # -1 no label, 0 land, 1 water
    l1c = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    img = s2[TILE][[l1c.index(b) for b in S2_BANDS]]  # reorder to S2_BANDS
    zt = np.load(os.path.join(ROOT, "exp", "out", "exp37_patches_bolivia.npz"))
    m = zt["tile"] == TILE
    G = 16
    g = grid_from_table(zt, m, G, ["err", "conf", "tile_phase", "boundary", "ndwi_level", "dihedral"])
    valid = ~np.isnan(g["err"])
    lab_w = np.zeros((G, G))
    for r in range(G):
        for c in range(G):
            block = lab[TILE][r * 4:(r + 1) * 4, c * 4:(c + 1) * 4]
            lab_w[r, c] = (block[block >= 0] > 0.5).mean() if (block >= 0).any() else 0
    lab_w = (lab_w > 0.5).astype(float)
    pred = np.where(g["err"] > 0.5, 1 - lab_w, lab_w)
    size = 64
    save("bolivia_rgb.png", rgb(img, size), 8)
    hand = classmap(lab[TILE] > 0.5); hand[lab[TILE] < 0] = (200, 200, 200)
    save("bolivia_label.png", hand, 8)
    predimg = classmap(pred); predimg[~valid] = (200, 200, 200)
    save("bolivia_pred.png", predimg, 32)
    save("bolivia_conf.png", cmap(g["conf"], "viridis", valid=valid), 32)
    save("bolivia_boundary.png", cmap(g["boundary"], "Oranges", vmin=0, vmax=1, valid=valid), 32)
    save("bolivia_tilephase.png", cmap(g["tile_phase"], "magma", valid=valid), 32)
    save("bolivia_ndwi.png", cmap(g["ndwi_level"], "cividis", valid=valid), 32)
    rs = review_mask(boundary_first_score(np.nan_to_num(g["conf"], nan=-np.inf), np.nan_to_num(g["boundary"], nan=0)), valid, BUDGET)
    tex_list("bolivia_review.tex", "boliviaReview", zip(*np.nonzero(rs)))
    tex_list("bolivia_errors.tex", "boliviaErrors", zip(*np.nonzero(g["err"] > 0.5)))
    hits = [(r, c) for r, c in zip(*np.nonzero(rs & (g["err"] > 0.5)))]
    zoom = hits[len(hits) // 2]                                    # one flagged window that is an error, for the zoom
    tex_list("bolivia_zoom.tex", "boliviaZoom", [zoom])
    cues = {"boundary": g["boundary"][zoom] > 0, "low_conf": g["conf"][zoom] >= np.nanquantile(g["conf"][valid], 0.8),
            "unstable": g["tile_phase"][zoom] >= np.nanquantile(g["tile_phase"][valid], 0.8), "ndwi": g["ndwi_level"][zoom] > -0.1}
    print("zoom window", zoom, "cues", cues)
    print(f"bolivia tile {TILE}: {int(valid.sum())} valid windows, {int((g['err'] > 0.5).sum())} errors, review set {int(rs.sum())} windows, "
          f"{int((g['err'][rs] > 0.5).sum())} of them errors; label water share {float((lab[TILE] == 1).mean()):.2f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    scene()
    bolivia()
    print("wrote", sorted(os.listdir(OUT)))
