#!/bin/bash
# exp99 (docs/plan/awf_transfer.md): the transfer test, graded at the East Africa TimeSync plots within 100 km of Ai2's
# AWF request geometry. CPU only: it reads files other jobs wrote and writes exp/out/exp99_* in the home checkout.
#
#   E99_MODE=inv   the inventory: how many selected plots fall in a window and on a covered pixel, by country. No class,
#                  probability or condition at a plot is read out. Allowed before the page is frozen.
#   E99_MODE=run   the run; exp/exp99_transfer.py refuses it until the page says frozen.
#
#   E99_PILOT=1    the pilot's inventory: E99_SCORES defaults to data/exp99/pilot/scores and the record goes to
#                  exp/out/exp99/pilot/exp99_inventory.json (only E99_MODE=inv)
#   E99_SCORES     the from-olmoearth part directories (default data/exp99/scores, which e99_read.sh writes)
#   E99_EXP98      exp98's part directories for Part F (default data/exp98/scores when it exists; "none" skips it)
#
# Submit from the Mac after commit and push, chained on e99_read.sh; the login node runs one sbatch and nothing else:
#   ssh aicr "E99_MODE=inv E99_SHA=$SHA sbatch --parsable --dependency=afterok:$READ_JOB" < exp/jobs/e99.sh
#   ssh aicr 'E99_MODE=run sbatch --parsable' < exp/jobs/e99.sh          # once the owner has frozen the page
# Like every exp99 stage it runs from exp99's own clone and never resets the home checkout (E99_README.md).
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J e99awf
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
MODE=${E99_MODE:-inv}
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
export E99_PILOT=${E99_PILOT:-0}
case "$E99_PILOT" in
  0) E99_SCORES=${E99_SCORES:-$REPO/data/exp99/scores}; OUTDIR=$REPO/exp/out ;;
  1) E99_SCORES=${E99_SCORES:-$REPO/data/exp99/pilot/scores}; OUTDIR=$REPO/exp/out/exp99/pilot
     [ "$MODE" = inv ] || { echo "the pilot runs the inventory only (E99_MODE=inv)"; exit 2; } ;;
  *) echo "E99_PILOT must be 0 or 1, got '$E99_PILOT'"; exit 2 ;;
esac
DATA=$REPO/data/breadth
AI2=$SCRATCH/deploy/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"

test -d "$E99_SCORES" || { echo "E99_SCORES=$E99_SCORES is not a directory"; exit 1; }
SCORE_DIRS=()
for d in "$E99_SCORES"/*/; do
  compgen -G "$d/scores_*.tif" > /dev/null && SCORE_DIRS+=("${d%/}")
done
[ ${#SCORE_DIRS[@]} -gt 0 ] || { echo "no scores_<EPSG>.tif in the subdirectories of $E99_SCORES"; exit 1; }
if [ -f "$E99_SCORES/parts.json" ]; then
  NPARTS=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$E99_SCORES/parts.json")
  [ "$NPARTS" = "${#SCORE_DIRS[@]}" ] || { echo "parts.json lists $NPARTS parts, $E99_SCORES holds ${#SCORE_DIRS[@]}"; exit 1; }
fi
test -f "$AI2" || { echo "Ai2's request geometry is missing from the pinned clone (exp/jobs/e98_env.sh)"; exit 1; }
EXP98_ARGS=()
E99_EXP98=${E99_EXP98:-$REPO/data/exp98/scores}
if [ "$E99_EXP98" != none ] && [ -d "$E99_EXP98" ]; then
  EXP98_DIRS=()
  for d in "$E99_EXP98"/*/; do
    compgen -G "$d/scores_*.tif" > /dev/null && EXP98_DIRS+=("${d%/}")
  done
  if [ ${#EXP98_DIRS[@]} -gt 0 ]; then EXP98_ARGS=(--exp98-scores "${EXP98_DIRS[@]}"); fi
fi
echo "${#SCORE_DIRS[@]} part(s) of the 2017 run; ${#EXP98_ARGS[@]} exp98 argument(s) for Part F"
PY="uv run --extra geo python exp/exp99_transfer.py"

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp99 mode $MODE, commit $(git rev-parse --short HEAD) =="
echo "== the sample and the boundaries, checked against their pins (fetched if absent) =="
uv run --extra geo python exp/timesync_awf_crosswalk.py fetch --dest "$DATA"
ARGS=(--timesync "$DATA/eastafrica_yearly_point_data.csv" --boundaries "$DATA/ne_10m_admin_0_countries.geojson"
      --request-geometry "$AI2" --out-dir "$OUTDIR")
echo "== smoke =="
$PY --smoke --out-dir "$SCRATCH/exp99_smoke"
case "$MODE" in
  inv)
    echo "== inventory (counts only) =="
    $PY --inventory --scores "${SCORE_DIRS[@]}" "${ARGS[@]}"
    ls -la "$OUTDIR/exp99_inventory.json" ;;
  run)
    echo "== the run =="
    $PY --scores "${SCORE_DIRS[@]}" "${ARGS[@]}" ${EXP98_ARGS[@]+"${EXP98_ARGS[@]}"}
    ls -la "$OUTDIR/exp99_summary.json" "$OUTDIR/exp99_units.npz" ;;
  *)
    echo "unknown E99_MODE '$MODE': inv or run"; exit 2 ;;
esac
