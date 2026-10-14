#!/bin/bash
# exp100 (docs/plan/alignment.md): exp98's and exp99's maps read at their label points moved by whole pixels, to see
# where the map and the labels agree best. CPU only, small: it reads the maps and labels exp98's and exp99's jobs left
# (data/exp98/scores and data/exp99/scores, symlinked to scratch; exp89's AWF label windows and Ai2's request
# geometry on scratch; the TimeSync sample in data/breadth, checked against its pin and never fetched here) and writes
# one summary to scratch, $SCRATCH/e100_out/exp100_summary.json, never into a checkout. It runs from its own clone on
# scratch, pinned to a full commit:
#   ssh aicr "E100_SHA=<full sha> sbatch --parsable" < exp/jobs/e100.sh
# The login node runs that one sbatch and nothing else. Any missing input stops the job at once.
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J e100
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e100_code
OUTDIR=$SCRATCH/e100_out
[ -n "${E100_SHA:-}" ] || { echo "E100_SHA (the full commit to run) is required"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
git cat-file -e "$E100_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E100_SHA" \
  || { echo "E100_SHA=$E100_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E100_SHA"
[ "$(git rev-parse HEAD)" = "$E100_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E100_SHA=$E100_SHA (give the full sha)"; exit 2; }
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=2
export PATH="$HOME/.local/bin:$PATH"

E98_SCORES=${E100_EXP98:-$REPO/data/exp98/scores}
E99_SCORES=${E100_EXP99:-$REPO/data/exp99/scores}
LABELS=$SCRATCH/data/exp89/awf/dataset/windows
TS=$REPO/data/breadth/eastafrica_yearly_point_data.csv
AI2=$SCRATCH/deploy/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
parts() {
  for d in "$1"/*/; do
    if compgen -G "$d/scores_*.tif" > /dev/null; then echo "${d%/}"; fi
  done
}
for s in "$E98_SCORES" "$E99_SCORES"; do
  test -d "$s" || { echo "$s is not a directory"; exit 1; }
done
mapfile -t D98 < <(parts "$E98_SCORES")
mapfile -t D99 < <(parts "$E99_SCORES")
for pair in "$E98_SCORES:${#D98[@]}" "$E99_SCORES:${#D99[@]}"; do
  s=${pair%:*}; n=${pair##*:}
  [ "$n" -gt 0 ] || { echo "no part directory with scores_<EPSG>.tif in $s"; exit 1; }
  if [ -f "$s/parts.json" ]; then
    NP=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$s/parts.json")
    [ "$NP" = "$n" ] || { echo "$s/parts.json lists $NP parts, $s holds $n"; exit 1; }
  fi
done
test -d "$LABELS" || { echo "no AWF label windows at $LABELS (scratch purge?): exp/jobs/e89.sh E89_MODE=inv E89_ARM=awf"; exit 1; }
test -f "$TS" || { echo "no TimeSync sample at $TS; this job does not fetch it"; exit 1; }
test -f "$AI2" || { echo "Ai2's request geometry is missing from the pinned clone on scratch (exp/jobs/e98_env.sh)"; exit 1; }
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp100, commit $(git rev-parse --short HEAD); ${#D98[@]} exp98 parts, ${#D99[@]} exp99 parts =="
uv run --extra geo python - "$TS" <<'EOF'
import sys
sys.path.insert(0, "exp")
import timesync_awf_crosswalk as tsx
ok, got = tsx.verify(sys.argv[1], tsx.TIMESYNC)
print("the TimeSync sample against its pin:", "ok" if ok else got)
sys.exit(0 if ok else 1)
EOF
# E100_MODE=run (the default): the preregistered check. E100_MODE=bytask: the descriptive follow-up written after the
# run (exp/exp100_by_task.py: V's and T's best positions per annotation task, V with each task left out).
case "${E100_MODE:-run}" in
  run)
    echo "== smoke =="
    uv run --extra geo python exp/exp100_alignment.py --smoke --out-dir "$OUTDIR/smoke"
    echo "== the run =="
    uv run --extra geo python exp/exp100_alignment.py --exp98-scores "${D98[@]}" --labels "$LABELS" \
      --exp99-scores "${D99[@]}" --timesync "$TS" --request-geometry "$AI2" --out-dir "$OUTDIR"
    ls -la "$OUTDIR/exp100_summary.json" ;;
  bytask)
    echo "== smoke (by task) =="
    uv run --extra geo python exp/exp100_by_task.py --smoke --out-dir "$OUTDIR/smoke"
    echo "== by task =="
    uv run --extra geo python exp/exp100_by_task.py --exp98-scores "${D98[@]}" --labels "$LABELS" --out-dir "$OUTDIR"
    ls -la "$OUTDIR/exp100_by_task.json" ;;
  *) echo "E100_MODE must be run or bytask"; exit 2 ;;
esac
echo "== $(date -Is) done =="
