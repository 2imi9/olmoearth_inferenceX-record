#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J e99select
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp99 stage 1 (exp/jobs/E99_README.md): the area rule and the request geometry, before any 2017 map exists.
#   1. fetches the East Africa TimeSync sample (github.com/bullocke/eastafrica at fc2014fc, CC0) and Natural Earth's
#      admin-0 countries (v5.1.2, public domain) into data/breadth/ of the home checkout, each checked against its pin
#      (exp/timesync_awf_crosswalk.py: the sample's sha256, computed from the Mac's copy on 9 October 2026, and both
#      files' git blob ids); nothing is copied from the Mac;
#   2. applies the rule docs/plan/awf_transfer.md fixed (D = 100 km, 309 plots: 219 Kenya, 90 Tanzania; 47 inside
#      Ai2's request geometry) and refuses other counts; computes each country's share of the region (Natural Earth,
#      EPSG:6933);
#   3. writes, on scratch only, olmoearth_run's request geometries: one square of about 11 m around each selected plot,
#      in the structure of Ai2's file (a FeatureCollection of Polygon features) with Ai2's property names for 2017
#      (oe_start_time 2017-01-01, oe_end_time 2017-12-31), and the pilot's (4 of the plots, seed 99):
#        /scratch/qi_zim_neu/olmoearth_inferenceX/deploy/e99_select/prediction_request_geometry{,_pilot}.geojson
#   Counts, areas and sha256s only go to exp/out/exp99/e99_select.json and this log; no coordinate does.
# exp99 stages run from their own clone (/scratch/qi_zim_neu/olmoearth_inferenceX/e99_code), never resetting the home
# checkout that exp98's chain resets (E99_README.md); records and data still land in the home checkout.
#   ssh aicr "E99_SHA=$SHA sbatch --parsable" < exp/jobs/e99_select.sh
# E99_SHA=<full sha> checks out that commit (detached) instead of origin/main. Needs exp98's environment
# (exp/jobs/e98_env.sh: the pinned olmoearth_projects clone holds Ai2's request geometry).
set -euo pipefail
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e99_code
DEPLOY=$SCRATCH/deploy
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
if [ -n "${E99_SHA:-}" ]; then
  # A pinned stage checks out its own commit, whatever main has become since the chain was submitted.
  git cat-file -e "$E99_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E99_SHA" \
    || { echo "E99_SHA=$E99_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E99_SHA"
  [ "$(git rev-parse HEAD)" = "$E99_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E99_SHA=$E99_SHA (give the full sha)"; exit 2; }
else
  git checkout -q --force --detach origin/main
fi
SEL=$DEPLOY/e99_select
OUT=$REPO/exp/out/exp99
DATA=$REPO/data/breadth
AI2=$DEPLOY/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp99 select, commit $(git rev-parse HEAD) =="
[ "$(git -C "$DEPLOY/olmoearth_projects" rev-parse HEAD)" = "f3c9b0c89c7670b3525647dda4c14b76245d682d" ] \
  || { echo "no pinned olmoearth_projects clone in $DEPLOY: run exp/jobs/e98_env.sh first"; exit 1; }
test -f "$AI2" || { echo "Ai2's request geometry is missing from the pinned clone"; exit 1; }
if [ -e "$SEL" ] && [ "${E99_RESELECT:-0}" != 1 ]; then
  echo "$SEL exists; E99_RESELECT=1 replaces it (only before prepare has read it)"; exit 2
fi
mkdir -p "$OUT" "$DATA"

echo "== the sample and the boundaries, fetched from their public sources and checked against their pins =="
uv run --extra geo python exp/timesync_awf_crosswalk.py fetch --dest "$DATA"

echo "== the area rule, the region's shares by country, and the request geometries (scratch only) =="
rm -rf "$SEL.partial"
uv run --extra geo python exp/exp99_transfer.py --select --timesync "$DATA/eastafrica_yearly_point_data.csv" \
  --request-geometry "$AI2" --boundaries "$DATA/ne_10m_admin_0_countries.geojson" --write-request "$SEL.partial" \
  --record "$OUT/e99_select.json"
( cd "$SEL.partial" && sha256sum prediction_request_geometry.geojson prediction_request_geometry_pilot.geojson ) \
  > "$SEL.partial/request.sha256"
rm -rf "$SEL"
mv "$SEL.partial" "$SEL"
echo "== $(date -Is) done: $OUT/e99_select.json; the request geometries' sha256 in $SEL/request.sha256 =="
