#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J e98burn
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp98 burned-area stage (exp/jobs/E98_README.md, "Burned area"), after e98_read.sh: for every part directory that
# e98_read.sh wrote, exp/exp98_burned.py fetches MODIS MCD64A1 v061 (monthly burned area, 500 m) for 2023 from
# Planetary Computer's STAC API (no key; the cpu nodes reach it, as prepare's Sentinel-2 reads do), mosaics the tiles
# over the part, and writes beside its scores, on the same grid by nearest neighbour:
#   data/exp98/scores/<part>/burned_<EPSG>.tif        int32: 1 burned in a month read, 0 not burned, -1 unrecorded
#   data/exp98/scores/<part>/burned_conditions.json   the codes' names and the exact --condition-names string
# and counts only to exp/out/exp98/e98_burned.json (share of covered pixels per code, per month burned, QA checks).
# Then, as a check that the layer plugs in, `oe-inferencex assess --condition burned_<EPSG>.tif --condition-names ...`
# on the smallest grid, to data/exp98/burned_assess/, with its exit status, windows ranked and the windows per
# condition in exp/out/exp98/e98_burned_assess.json. exp98's Part H (exp/jobs/e98.sh) reads the burned rasters.
#
#   ssh aicr "sbatch --parsable --dependency=afterok:$READ_JOB" < exp/jobs/e98_burned.sh
# E98_SHA=<full sha> checks out that commit (detached) instead of origin/main, as env, prepare, predict and collect do.
# E98_PILOT=1 reads the pilot's parts (data/exp98/pilot/scores) and records to exp/out/exp98/pilot/; the read record's
# `pilot` must match. E98_REBURN=1 replaces burned rasters written before. E98_BURN_ALLOW_MISSING (default 2023-09)
# names the months that may be absent from the catalogue: on 2026-10-08 Planetary Computer held no September 2023 item
# of the collection anywhere, while NASA's CMR lists those granules; any other missing month stops the job. Set it
# empty once the month is there; a month allowed missing but present is read.
# No coordinate, tile or item id is printed: the script prints counts only, its full output stays in a log on scratch,
# and what this job echoes from that log on a failure is masked (URLs, tile ids, long numbers).
set -euo pipefail
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
cd "$REPO" || exit 1
git fetch -q origin main
if [ -n "${E98_SHA:-}" ]; then
  # a pinned stage checks out its own commit, whatever main has become since the chain was submitted
  git cat-file -e "$E98_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E98_SHA" \
    || { echo "E98_SHA=$E98_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E98_SHA"
  [ "$(git rev-parse HEAD)" = "$E98_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E98_SHA=$E98_SHA (give the full sha)"; exit 2; }
else
  git reset -q --hard origin/main
fi
test -f exp/exp98_burned.py || { echo "exp/exp98_burned.py is not in this checkout"; exit 2; }
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
export E98_PILOT=${E98_PILOT:-0}
case "$E98_PILOT" in 0|1) ;; *) echo "E98_PILOT must be 0 or 1, got '$E98_PILOT'"; exit 2;; esac
if [ "$E98_PILOT" = 1 ]; then
  export SCORES=$REPO/data/exp98/pilot/scores OUT=$REPO/exp/out/exp98/pilot BASSESS=$REPO/data/exp98/pilot/burned_assess
  LOG=$SCRATCH/deploy/logs/pilot_burned.log PILOT_ARG=(--pilot)
else
  export SCORES=$REPO/data/exp98/scores OUT=$REPO/exp/out/exp98 BASSESS=$REPO/data/exp98/burned_assess
  LOG=$SCRATCH/deploy/logs/burned.log PILOT_ARG=()
fi
ALLOW=${E98_BURN_ALLOW_MISSING-2023-09}
read -r -a ALLOW_LIST <<< "$ALLOW"
ALLOW_ARGS=()
if [ ${#ALLOW_LIST[@]} -gt 0 ]; then ALLOW_ARGS=(--allow-missing "${ALLOW_LIST[@]}"); fi
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"
# masks what is echoed from the log: URLs (signed hrefs name the tile), MODIS tile ids, and the jobs' redact() numbers
redact() { sed -E 's#https?://[^ "]*#<url>#g; s/h[0-9]{2}v[0-9]{2}/<tile>/g; s/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g'; }

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp98 burned area, commit $(git rev-parse HEAD), pilot $E98_PILOT, allowed missing '${ALLOW}' =="
test -f "$OUT/e98_read.json" || { echo "no $OUT/e98_read.json: run e98_read.sh first"; exit 1; }
python3 - "$OUT/e98_read.json" "$E98_PILOT" <<'EOF' || exit 2
import json, sys
r = json.load(open(sys.argv[1]))
if bool(r.get("pilot", False)) != (sys.argv[2] == "1"):
    sys.exit("the read record says pilot %s, this job E98_PILOT=%s: no mixing" % (r.get("pilot", False), sys.argv[2]))
EOF
test -d "$SCORES" || { echo "no $SCORES: run e98_read.sh first"; exit 1; }
compgen -G "$SCORES/r*_p*/scores_*.tif" > /dev/null || { echo "no part directory with scores_<EPSG>.tif in $SCORES"; exit 1; }
REDO=()
if compgen -G "$SCORES/r*_p*/burned_*.tif" > /dev/null; then
  if [ "${E98_REBURN:-0}" = 1 ]; then REDO=(--redo); else echo "burned rasters exist in $SCORES; E98_REBURN=1 replaces them"; exit 2; fi
fi
mkdir -p "$OUT" "$(dirname "$LOG")"
uv run --extra geo python -c 'import rasterio, pystac_client, planetary_computer' \
  || { echo "rasterio, pystac-client or planetary-computer does not import (the geo extra)"; exit 1; }

echo "== $(date -Is) burned rasters (full output in the scratch log; counts below) =="
if ! uv run --extra geo python exp/exp98_burned.py --scores "$SCORES" --out-json "$OUT/e98_burned.json" \
     ${ALLOW_ARGS[@]+"${ALLOW_ARGS[@]}"} ${REDO[@]+"${REDO[@]}"} ${PILOT_ARG[@]+"${PILOT_ARG[@]}"} > "$LOG" 2>&1; then
  echo "exp98_burned.py failed; the last lines of its log, masked:"; tail -n 40 "$LOG" | redact; exit 1
fi
uv run --extra geo python - "$OUT/e98_burned.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"names: {r['condition_names']}")
print(f"months missing: {r['source']['months_missing']}; from local files: {r['source']['months_from_files']}")
for e in r["parts"]:
    s = {k: round(v, 4) if v is not None else None for k, v in e["share_of_covered_by_code"].items()}
    print(f"{e['part']} {e['grid']}: covered {e['pixels_covered']}, share by code {s}, QA agrees {e['qa_check']['qa_agrees']}, "
          f"burn dates outside their month {e['burn_date_outside_its_month_pixels']} px")
print("all parts:", json.dumps(r["totals"]["share_of_covered_by_code"]))
EOF

echo "== $(date -Is) check: assess --condition on the smallest grid (counts only) =="
rm -rf "$BASSESS"
uv run --extra geo python - <<'EOF'
import glob, json, os, re, shlex, subprocess
import rasterio
scores, out, bassess = os.environ["SCORES"], os.environ["OUT"], os.environ["BASSESS"]
grids = []
for b in sorted(glob.glob(os.path.join(scores, "r*_p*", "burned_*.tif"))):
    d, lab = os.path.dirname(b), re.sub(r"^burned_|\.tif$", "", os.path.basename(b))
    with rasterio.open(os.path.join(d, f"scores_{lab}.tif")) as src:
        grids.append((src.height * src.width, d, lab))
assert grids, "no burned_<EPSG>.tif written"
_, d, lab = min(grids)
names = json.load(open(os.path.join(d, "burned_conditions.json")))
args = shlex.split(names["condition_names"])[1:]             # "--condition-names a=b ..." without the flag
aout = os.path.join(bassess, f"{os.path.basename(d)}_{lab}")
cmd = ["oe-inferencex", "assess", os.path.join(d, f"scores_{lab}.tif"), "--out", aout,
       "--condition", os.path.join(d, f"burned_{lab}.tif"), "--condition-names", *args]
rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
rec = {"part": os.path.basename(d), "grid": lab, "pixels": min(grids)[0], "assess_exit": rc.returncode,
       "condition_names": names["condition_names"]}
if rc.returncode == 0:
    a = json.load(open(os.path.join(aout, "assessment.json")))
    rec["assess_windows_ranked"] = a["n_windows"]
    per = (a.get("conditions") or {}).get("per_condition") or {}
    rec["per_condition"] = {k: {"n_windows": v.get("n_windows"), "share_of_map": v.get("share_of_map"),
                                "share_of_review_set": v.get("share_of_review_set")} for k, v in per.items()}
else:
    last = (rc.stderr.strip().splitlines() or [""])[-1][:300]           # masked as redact() does
    last = re.sub(r"https?://\S+", "<url>", last)
    last = re.sub(r"-?[0-9]{4,}", "<n>", re.sub(r"-?[0-9]+\.[0-9]{2,}", "<n>", last))
    print(f"assess failed; the last line of its error, masked: {last}")
json.dump(rec, open(os.path.join(out, "e98_burned_assess.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "condition_names"}))
assert rc.returncode == 0, "assess --condition burned failed (above)"
EOF
echo "== $(date -Is) done: burned rasters beside the scores in $SCORES; $OUT/e98_burned.json, $OUT/e98_burned_assess.json =="
[ "$E98_PILOT" = 1 ] || echo "next: E98_MODE=inv sbatch --dependency=afterok:${SLURM_JOB_ID:-<this job>} exp/jobs/e98.sh (Part H reads the burned rasters in the run)"
