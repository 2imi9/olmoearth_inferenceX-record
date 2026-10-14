#!/bin/bash
# exp98 (docs/plan/awf_deployment.md): the record's re-run of Ai2's FT-AWF deployment configuration over the AWF
# project's request geometry (2023, 10 m, probabilities written), graded at Ai2's own AWF labels. CPU only: it reads
# files other jobs wrote and writes exp/out/exp98_*.
#
#   E98_MODE=inv   the inventory: label points inside the deployed area by split, how each was mapped, the label
#                  windows' time ranges and, with E98_DEPLOY, the 10 m inputs and the Sentinel-2 configuration at the
#                  label pixels. No class, probability or condition at a label pixel is read out. Allowed before the
#                  page is frozen.
#   E98_MODE=run   the run; exp/exp98_awf_deployment.py refuses it until the page says frozen.
#
#   E98_SCORES     the from-olmoearth output: a directory holding scores_<EPSG>.tif, or (the default,
#                  data/exp98/scores, which exp/jobs/e98_read.sh writes) a directory of part directories r<root>_p<part>,
#                  all read as one map
#   E98_DEPLOY     optional (part G): the rslearn dataset roots olmoearth_run wrote, space-separated, or "auto" for
#                  every root in prepare's scratch inventory, under scratch deploy/awf_run (purged after 30 days); the
#                  roots' real names are never echoed
#   E98_LABELS     the AWF windows root; default exp89's extraction of the pinned tar (exp/jobs/e89.sh, E89_ARM=awf)
#   Part I (report-only) needs the East Africa TimeSync sample and Natural Earth's countries: the job fetches both
#                  from their public sources into data/breadth/ and checks them against their pins
#                  (exp/timesync_awf_crosswalk.py), and reads Ai2's request geometry from the pinned olmoearth_projects
#                  clone on scratch. If either cannot be had, Part I is skipped with a note; nothing else changes.
#
# Submit from the Mac after commit and push (the job resets the checkout to origin/main), chained on e98_burned.sh
# (Part H's layer, itself chained on e98_read.sh, the job that wrote E98_SCORES; without the layer Part H is skipped
# with a note); the login node runs one sbatch and nothing else:
#   ssh aicr "E98_MODE=inv E98_DEPLOY=auto sbatch --parsable --dependency=afterok:$BURN_JOB" < exp/jobs/e98.sh
#   ssh aicr 'E98_MODE=run E98_DEPLOY=auto sbatch --parsable' < exp/jobs/e98.sh
# Both modes write files in exp/out; never let this job overlap another that writes there (chain with --dependency).
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J e98awf
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
MODE=${E98_MODE:-inv}
E98_SCORES=${E98_SCORES:-/home/qi_zim_neu/olmoearth_inferenceX/data/exp98/scores}
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
LABELS=${E98_LABELS:-$SCRATCH/data/exp89/awf/dataset/windows}
cd /home/qi_zim_neu/olmoearth_inferenceX || { echo "no checkout at /home/qi_zim_neu/olmoearth_inferenceX"; exit 1; }
git fetch -q origin main
git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$SCRATCH/uv-cache
export OMP_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"

test -d "$E98_SCORES" || { echo "E98_SCORES=$E98_SCORES is not a directory"; exit 1; }
SCORE_DIRS=()
if compgen -G "$E98_SCORES/scores_*.tif" > /dev/null; then
  SCORE_DIRS=("$E98_SCORES")
else
  for d in "$E98_SCORES"/*/; do
    compgen -G "$d/scores_*.tif" > /dev/null && SCORE_DIRS+=("${d%/}")
  done
fi
[ ${#SCORE_DIRS[@]} -gt 0 ] || { echo "no scores_<EPSG>.tif in $E98_SCORES or its subdirectories"; exit 1; }
if [ -f "$E98_SCORES/parts.json" ]; then
  NPARTS=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$E98_SCORES/parts.json")
  [ "$NPARTS" = "${#SCORE_DIRS[@]}" ] || { echo "parts.json lists $NPARTS parts, $E98_SCORES holds ${#SCORE_DIRS[@]}"; exit 1; }
fi
test -d "$LABELS" || { echo "no AWF windows at $LABELS: extract them with exp/jobs/e89.sh (E89_MODE=inv E89_ARM=awf)"; exit 1; }
DEPLOY_ARGS=()
if [ "${E98_DEPLOY:-}" = auto ]; then
  INV=$SCRATCH/deploy/prepare_inventory.json      # scratch's copy holds the real root names; the tracked one does not
  test -f "$INV" || { echo "E98_DEPLOY=auto needs $INV (e98_prepare.sh; purged after 30 days)"; exit 1; }
  E98_DEPLOY=$(python3 -c 'import json,sys; print(" ".join(sys.argv[2] + "/" + r["root"] for r in json.load(open(sys.argv[1]))["roots"]))' \
    "$INV" "$SCRATCH/deploy/awf_run")
fi
if [ -n "${E98_DEPLOY:-}" ]; then
  read -r -a DEPLOY_LIST <<< "$E98_DEPLOY"
  for d in "${DEPLOY_LIST[@]}"; do
    test -d "$d/windows" || { echo "deployment root $d holds no windows/ (purged after 30 days?)"; exit 1; }
  done
  DEPLOY_ARGS=(--deployment-dataset "${DEPLOY_LIST[@]}")
fi
echo "${#SCORE_DIRS[@]} from-olmoearth part(s); ${#DEPLOY_ARGS[@]} deployment argument(s)"
PY="uv run --extra encoder --extra geo python exp/exp98_awf_deployment.py"

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): mode $MODE, commit $(git rev-parse --short HEAD) =="
echo "== Part I's inputs: the TimeSync sample and the boundaries, fetched and checked against their pins =="
TS_ARGS=()
AI2_GEOMETRY=$SCRATCH/deploy/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
if uv run --extra encoder --extra geo python exp/timesync_awf_crosswalk.py fetch --dest data/breadth \
   && test -f "$AI2_GEOMETRY"; then
  TS_ARGS=(--timesync data/breadth/eastafrica_yearly_point_data.csv
           --boundaries data/breadth/ne_10m_admin_0_countries.geojson --request-geometry "$AI2_GEOMETRY")
else
  echo "Part I's inputs are not available (above); Part I is skipped with a note, nothing else changes"
fi
echo "== smoke =="
$PY --smoke --out-dir "$SCRATCH/exp98_smoke"
case "$MODE" in
  inv)
    echo "== inventory (counts only) =="
    $PY --inventory --scores "${SCORE_DIRS[@]}" --labels "$LABELS" ${DEPLOY_ARGS[@]+"${DEPLOY_ARGS[@]}"} \
      ${TS_ARGS[@]+"${TS_ARGS[@]}"}
    ls -la exp/out/exp98_inventory.json ;;
  run)
    echo "== the run =="
    $PY --scores "${SCORE_DIRS[@]}" --labels "$LABELS" ${DEPLOY_ARGS[@]+"${DEPLOY_ARGS[@]}"} \
      ${TS_ARGS[@]+"${TS_ARGS[@]}"}
    ls -la exp/out/exp98_summary.json exp/out/exp98_units.npz ;;
  *)
    echo "unknown E98_MODE '$MODE': inv or run"; exit 2 ;;
esac
