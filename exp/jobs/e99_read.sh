#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J e99read
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp99 stage 5 (exp/jobs/E99_README.md): read the collected run into the rasters exp99 reads, with
# `oe-inferencex from-olmoearth --conditions --inputs sentinel2_scl`, one window per part: the windows lie apart
# (one per plot), and a part holding two distant windows would be a mostly empty grid of their bounding box, which the
# reader holds in memory whole. Each part goes to its own directory, read as one map by exp99 (as exp98 reads its parts):
#   /home/qi_zim_neu/olmoearth_inferenceX/data/exp99/scores/r<root>_w<k>/   (data/ is gitignored)
#   /home/qi_zim_neu/olmoearth_inferenceX/data/exp99/scores/parts.json      which window each part holds (names only)
# and counts only to exp/out/exp99/e99_read.json. E99_REREAD=1 replaces an earlier read. E99_PILOT=1 reads the pilot
# (data/exp99/pilot/) and then checks what was read, counts only and label-free, as e98_read.sh's pilot does: bands,
# band sums, covered share, condition codes, and `oe-inferencex assess --condition` on each part.
#   ssh aicr "E99_SHA=$SHA sbatch --parsable --dependency=afterok:$COLL_JOB" < exp/jobs/e99_read.sh
set -euo pipefail
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e99_code
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
if [ -n "${E99_SHA:-}" ]; then
  git cat-file -e "$E99_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E99_SHA" \
    || { echo "E99_SHA=$E99_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E99_SHA"
  [ "$(git rev-parse HEAD)" = "$E99_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E99_SHA=$E99_SHA (give the full sha)"; exit 2; }
else
  git checkout -q --force --detach origin/main
fi
test -f oe_inferencex/olmoearth.py || { echo "oe_inferencex/olmoearth.py is not in this checkout"; exit 2; }
export E99_PILOT=${E99_PILOT:-0}
case "$E99_PILOT" in 0|1) ;; *) echo "E99_PILOT must be 0 or 1, got '$E99_PILOT'"; exit 2;; esac
if [ "$E99_PILOT" = 1 ]; then
  export DEST=$REPO/data/exp99/pilot/awf2017_run SCORES=$REPO/data/exp99/pilot/scores OUT=$REPO/exp/out/exp99/pilot
  export ASSESS=$REPO/data/exp99/pilot/assess
else
  export DEST=$REPO/data/exp99/awf2017_run SCORES=$REPO/data/exp99/scores OUT=$REPO/exp/out/exp99
fi
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp99 read, commit $(git rev-parse HEAD), pilot $E99_PILOT =="
test -f "$DEST/records/prepare_inventory.json" || { echo "no collected run at $DEST: run e99_collect.sh first"; exit 1; }
test -f "$OUT/e99_collect.json" || { echo "no $OUT/e99_collect.json: run e99_collect.sh first"; exit 1; }
if [ -e "$SCORES" ]; then
  if [ "${E99_REREAD:-0}" = 1 ]; then rm -rf "$SCORES"; else echo "$SCORES exists; E99_REREAD=1 replaces it"; exit 2; fi
fi
rm -rf "$SCORES.partial"
mkdir -p "$SCORES.partial" "$OUT"
uv run --extra geo oe-inferencex from-olmoearth --help > /dev/null \
  || { echo "oe-inferencex from-olmoearth does not run in this checkout"; exit 1; }

uv run --extra geo python - <<'EOF'
import json, os, re, subprocess, sys
dest, scores, out = os.environ["DEST"], os.environ["SCORES"] + ".partial", os.environ["OUT"]
inv = json.load(open(os.path.join(dest, "records", "prepare_inventory.json")))
coll = json.load(open(os.path.join(out, "e99_collect.json")))
assert coll.get("manifest_verified"), "collect is not verified"
pilot = os.environ["E99_PILOT"] == "1"
assert inv.get("experiment") == coll.get("experiment") == "exp99", "not exp99's collection"
assert inv.get("pilot", False) == pilot and coll.get("pilot", False) == pilot, \
    f"inventory pilot {inv.get('pilot', False)}, collect pilot {coll.get('pilot', False)}, this job {pilot}: no mixing"


def esc(s):                          # a window id as a literal fnmatch pattern
    return "".join(f"[{ch}]" if ch in "[]*?" else ch for ch in s)


plan, counts = [], {"roots": [], "rule": "one window per part"}
for ri, r in enumerate(inv["roots"]):
    root = os.path.join(dest, r["root_id"])
    wins = []
    for group in sorted(os.listdir(os.path.join(root, "windows"))):
        gd = os.path.join(root, "windows", group)
        for name in sorted(os.listdir(gd)):
            if os.path.exists(os.path.join(gd, name, "layers", "output", "completed")):
                wins.append(f"{group}/{name}")
    counts["roots"].append({"root_id": r["root_id"], "windows": len(wins)})
    for k, w in enumerate(wins):
        plan.append({"root": r["root_id"], "dir": f"r{ri:02d}_w{k:04d}", "windows": [w]})
json.dump(plan, open(os.path.join(scores, "parts.json"), "w"), indent=1)
print(json.dumps(counts))
for i, p in enumerate(plan):
    cmd = ["oe-inferencex", "from-olmoearth", os.path.join(dest, p["root"]), "--out", os.path.join(scores, p["dir"]),
           "--conditions", "--inputs", "sentinel2_scl", "--window"] + [esc(w) for w in p["windows"]]
    # a window holds one plot, so its name and extent never reach the log: on failure only the error's last line,
    # with the window, the paths and every number masked as redact() does, under the part's own name
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if rc.returncode != 0:
        last = (rc.stderr.strip().splitlines() or [""])[-1][:300]
        for w in p["windows"]:
            last = last.replace(w, "<window>").replace(w.split("/")[-1], "<window>")
        for path, tag in sorted([(scores, "<scores>"), (dest, "<dest>")], key=lambda x: -len(x[0])):
            last = last.replace(path, tag)       # the longer first, in case one path holds the other
        last = re.sub(r"-?[0-9]{4,}", "<n>", re.sub(r"-?[0-9]+\.[0-9]{2,}", "<n>", last))
        print(f"from-olmoearth failed on part {p['dir']} (exit {rc.returncode}); the last line of its error, "
              f"masked: {last}", flush=True)
        sys.exit(1)
    n = len([f for f in os.listdir(os.path.join(scores, p["dir"])) if f.startswith("scores_")])
    assert n == 1, f"{p['dir']}: {n} scores rasters written for one window"
    if (i + 1) % 25 == 0 or i + 1 == len(plan):
        print(f"{i + 1} of {len(plan)} parts read", flush=True)
counts["parts_read"] = len(plan)
counts["pilot"] = pilot
counts["experiment"] = "exp99"
json.dump(counts, open(os.path.join(out, "e99_read.json"), "w"), indent=1)
EOF
mv "$SCORES.partial" "$SCORES"
if [ "$E99_PILOT" = 1 ]; then
  echo "== $(date -Is) pilot checks on what was read (counts only; no plot is read) =="
  rm -rf "$ASSESS"
  uv run --extra geo python - <<'EOF'
import glob, json, os, re, shlex, subprocess
import numpy as np, rasterio
scores, out, assess = os.environ["SCORES"], os.environ["OUT"], os.environ["ASSESS"]
chk = {"windows_read": 0, "parts": [], "ok": True}
for d in sorted(glob.glob(os.path.join(scores, "r*_w*"))):
    so = json.load(open(os.path.join(d, "olmoearth_output.json")))
    co = json.load(open(os.path.join(d, "olmoearth_conditions.json")))
    cond_of = {g["label"]: g for g in co["grids"]}
    chk["windows_read"] += so["windows_read"]
    for g in so["grids"]:
        spath = os.path.join(d, os.path.basename(g["scores"]))
        cpath = os.path.join(d, os.path.basename(cond_of[g["label"]]["condition"]))
        with rasterio.open(spath) as src:
            a = src.read()
        covered = np.isfinite(a).all(axis=0)
        dev = float(np.abs(a[:, covered].sum(axis=0, dtype=np.float64) - 1).max()) if covered.any() else None
        cg = cond_of[g["label"]]
        with rasterio.open(cpath) as src:
            c = src.read(1)
        vals, ns = np.unique(c, return_counts=True)
        e = {"part": os.path.basename(d), "grid": g["label"], "windows": len(g["windows"]), "bands": int(a.shape[0]),
             "summary_bands": so["bands"], "pixels": int(covered.size), "covered_share": float(covered.mean()),
             "max_abs_sum_minus_1": dev, "condition_codes": {str(int(v)): int(n) for v, n in zip(vals, ns)}}
        aout = os.path.join(assess, f"{os.path.basename(d)}_{g['label']}")
        names = shlex.split(cg["condition_names"])[1:]
        cmd = ["oe-inferencex", "assess", spath, "--out", aout, "--condition", cpath]
        cmd += (["--condition-names", *names] if names else [])
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        e["assess_exit"] = rc.returncode
        if rc.returncode == 0:
            e["assess_windows_ranked"] = json.load(open(os.path.join(aout, "assessment.json")))["n_windows"]
        else:
            last = (rc.stderr.strip().splitlines() or [""])[-1][:300]
            last = re.sub(r"-?[0-9]{4,}", "<n>", re.sub(r"-?[0-9]+\.[0-9]{2,}", "<n>", last))
            print(f"assess failed on {e['part']} {e['grid']}; the last line of its error, masked: {last}")
        e["ok"] = bool(e["bands"] == 10 and e["summary_bands"] == 10 and dev is not None and dev <= 1e-3
                       and e["covered_share"] > 0 and rc.returncode == 0)
        chk["ok"] &= e["ok"]
        chk["parts"].append(e)
        print(json.dumps(e))
chk["ok"] = bool(chk["ok"] and chk["parts"])
json.dump(chk, open(os.path.join(out, "e99_pilot_checks.json"), "w"), indent=1)
print(f"windows read {chk['windows_read']}; ok {chk['ok']}")
assert chk["ok"], "a pilot check failed (above; exp/out/exp99/pilot/e99_pilot_checks.json)"
EOF
fi
echo "== $(date -Is) done: $(ls -d "$SCORES"/r*_w* | wc -l) part directories in $SCORES, $OUT/e99_read.json =="
