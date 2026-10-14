"""Summary figure: does an audit signal's advantage survive a change of truth?

For each signal, the share of units where it ranks errors better than the
model's own confidence, under a weak reference (ESA WorldCover, the 27
rule-selected scenes) and under hand labels (Sen1Floods11 Bolivia). Every
signal that beats confidence on the left fails to beat it on the right; that
gap is the repository's open question (docs/results/comparisons.md section 3).

DIAGNOSTIC ONLY - do not publish this chart as evidence. Its two columns do
not share a unit: a WorldCover point is one scene aggregating 3000+ patches
into a single vote, a Sen1Floods11 point is one 60x60 tile crop. Joining them
with a line implies a comparability that does not hold, and the monotone drop
reads as though the reference caused it, which is exactly the causal claim
exp23-exp25 failed to establish. The numbers are correct; the graphic form
overstates them. The clustered sign test printed below is the part worth
citing (docs/method/protocol.md, "Known limits of these tests").

Reads committed artifacts only - no network, no encoder, no torch:
exp13_summary.json, exp14_boundary_ablation.csv, exp17_internal_evidence.csv,
exp18_sen1floods.csv. Writes exp/out/summary_transfer.png.

    uv run --extra geo python exp/summary_transfer.py
"""
import collections
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def head_to_head(filename, signal, baseline):
    """Scenes where `signal` scores below `baseline` (lower E-AURC is better)."""
    wins = losses = 0
    with open(os.path.join(OUT, filename)) as fh:
        for row in csv.DictReader(fh):
            wins += float(row[signal]) < float(row[baseline])
            losses += float(row[signal]) > float(row[baseline])
    return wins, losses


with open(os.path.join(OUT, "exp13_summary.json")) as fh:
    exp13 = json.load(fh)["signals"]
with open(os.path.join(OUT, "exp18_sen1floods.csv")) as fh:
    bolivia = next(iter(csv.DictReader(fh)))

# (label, WorldCover wins/losses, the exp18 column holding "W/L" on hand labels)
SIGNALS = [
    ("E_system tiling instability",
     (exp13["tile-phase (aligned)"]["w"], exp13["tile-phase (aligned)"]["l"]),
     "tile-phase (aligned)"),
    ("band-set disagreement",
     head_to_head("exp17_internal_evidence.csv", "band-set disagreement", "baseline"),
     "band-set disagreement"),
    ("prediction boundary",
     head_to_head("exp14_boundary_ablation.csv", "pred-boundary", "baseline"),
     "boundary indicator"),
    ("pixel control (no model)",
     (exp13["control"]["w"], exp13["control"]["l"]),
     "control NDWI gradient"),
    ("E_case cross-model",
     (exp13["E_case"]["w"], exp13["E_case"]["l"]),
     "E_case |Nano-Base|"),
    ("E_dist embedding distance",
     (exp13["E_dist"]["w"], exp13["E_dist"]["l"]),
     "E_dist knn-to-train"),
]

# Nudges (in points) that keep near-coincident labels legible.
LEFT_DY = {"pixel control (no model)": 8, "E_dist embedding distance": -8}
RIGHT_DY = {"pixel control (no model)": -7, "E_case cross-model": 7}

fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.axhspan(0, 0.5, color="#c62828", alpha=0.05, zorder=0)
ax.axhline(0.5, color="#444", lw=1.2, ls="--", zorder=2)

cmap = plt.get_cmap("tab10")
for i, (name, (wins, losses), key) in enumerate(SIGNALS):
    hand_w, hand_l = (int(v) for v in bolivia[f"{key}|W/L"].split("/"))
    hand_p = float(bolivia[f"{key}|sign_p"])
    weak, hand = wins / (wins + losses), hand_w / (hand_w + hand_l)
    color = cmap(i)
    # A hollow endpoint means the signal is not significantly different from
    # confidence there; only filled endpoints are significantly worse.
    worse = hand_p < 0.05
    ax.plot([1, 2], [weak, hand], "-", color=color, lw=2.2, zorder=3)
    ax.plot([1], [weak], "o", color=color, ms=8, zorder=4)
    ax.plot([2], [hand], "o", color=color, ms=8, zorder=4,
            markerfacecolor=color if worse else "white", markeredgewidth=2)
    ax.annotate(f"{name}   {weak:.0%}", (1, weak), textcoords="offset points",
                xytext=(-12, LEFT_DY.get(name, 0)), ha="right", va="center",
                fontsize=9.5, color=color)
    tag = f"{hand:.0%}" if worse else f"{hand:.0%}  n.s."
    ax.annotate(tag, (2, hand), textcoords="offset points",
                xytext=(10, RIGHT_DY.get(name, 0)), ha="left", va="center",
                fontsize=9.5, color=color)

ax.set_xlim(0.32, 2.34)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([1, 2])
ax.set_xticklabels(["ESA WorldCover\n27 river scenes (exp13, exp14, exp17)",
                    "hand labels\n350 Bolivia tiles (exp18)"], fontsize=10)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("share of units where the signal ranks errors\nbetter than the model's own confidence",
              fontsize=10)
ax.set_title("No audit signal beats the model's own confidence against hand labels\n"
             "(hollow endpoint = not significantly different; all others significantly worse)",
             fontsize=11.5, pad=12)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

# Robustness: the 27 scenes are not 27 independent draws. They sample fixed
# fractions along eight named rivers, so scenes on one river share its
# reference errors, its season and its channel morphology. Re-run the sign
# test with one vote per river (majority of that river's scenes) to see how
# much of the significance survives clustering.
RIVER = {"barotse": "Zambezi", "delta": "Zambezi", "kazungula": "Zambezi",
         "vicfalls_up": "Zambezi", "zambezi_20": "Zambezi", "zambezi_50": "Zambezi",
         "zambezi_80": "Zambezi", "cuando_20": "Cuando", "cuando_50": "Cuando",
         "cuando_80": "Cuando", "kafue_20": "Kafue", "kafue_50": "Kafue",
         "kafue_80": "Kafue", "luangwa_conf": "Luangwa", "okavango_50": "Okavango",
         "okavango_80": "Okavango", "okavango_sep": "Okavango", "rovuma_20": "Rovuma",
         "rovuma_50": "Rovuma", "rovuma_80": "Rovuma", "save_20": "Save",
         "save_50": "Save", "save_80": "Save", "shire_20": "Shire",
         "shire_50": "Shire", "shire_80": "Shire", "shire_liwonde": "Shire"}


def _sign_p(wins, losses):
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def river_clustered_check():
    scenes = collections.defaultdict(dict)
    with open(os.path.join(OUT, "exp13_corrected_stats.csv")) as fh:
        for row in csv.DictReader(fh):
            scenes[row["scene"]][row["signal"]] = float(row["eaurc"])
    tp = "tile-phase (aligned)"
    sw = sum(v[tp] < v["baseline"] for v in scenes.values())
    sl = sum(v[tp] > v["baseline"] for v in scenes.values())
    per_river = collections.defaultdict(lambda: [0, 0])
    for name, v in scenes.items():
        per_river[RIVER[name]][0 if v[tp] < v["baseline"] else 1] += 1
    rw = sum(1 for a, b in per_river.values() if a > b)
    rl = sum(1 for a, b in per_river.values() if a < b)
    print(f"  scene level : {sw}/{sl}  sign p = {_sign_p(sw, sl):.2g}  (n = {sw + sl})")
    print(f"  river level : {rw}/{rl}  sign p = {_sign_p(rw, rl):.2g}  (n = {rw + rl})")


print("tiling instability vs confidence, ESA WorldCover:")
river_clustered_check()

dest = os.path.join(OUT, "summary_transfer.png")
fig.savefig(dest, dpi=170)
print("wrote", dest)
