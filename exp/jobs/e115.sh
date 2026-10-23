#!/bin/bash
# exp115 (docs/plan/per_class_exact.md; the page is the specification): exact per-class report on 40 fully labelled
# maps. Helper: exp/exp115_per_class_exact.py in the record's uv environment. No download: the PLANTEUR windows are
# exp113's private arrays on scratch; the other populations are committed under exp/out/.
#
#   E115_MODE=run         CPU, 8 CPUs, 16 GB, 2 h. Refused unless the page is frozen at E115_SHA; writes
#                         exp115_summary.json to $OUT (copy to exp/out/ when recording).
#   E115_MODE=crosscheck  CPU, 8 CPUs, 16 GB, 2 h. The independent second route (exp/exp115_crosscheck.py); writes
#                         $OUT/exp115_crosscheck.json.
#
# Submit after commit and push; the login node runs one sbatch per line and nothing else:
#   ssh aicr "E115_MODE=run E115_SHA=$SHA sbatch --parsable -J e115run -p cpu -c 8 --mem=16G -t 02:00:00" < exp/jobs/e115.sh
#SBATCH -A p2026_0089_neu
#SBATCH -J e115
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -eu
umask 077
MODE=${E115_MODE:?E115_MODE is run or crosscheck}
E115_SHA=${E115_SHA:?E115_SHA is the commit to run}
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e115_code
E113PRIV=$SCRATCH/e113_private
OUT=$SCRATCH/e115_out
export UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
test -f "$E113PRIV/exp113_windows_default_oe13.npz" || { echo "exp113's private window arrays are missing"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q https://github.com/2imi9/olmoearth_inferenceX-record.git "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E115_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E115_SHA" \
  || { echo "E115_SHA=$E115_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E115_SHA"
[ "$(git rev-parse HEAD)" = "$E115_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E115_SHA"; exit 2; }
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp115 $MODE at $E115_SHA =="
if [ "$MODE" = run ]; then
  grep -q '^\*\*Status: frozen' docs/plan/per_class_exact.md \
    || { echo "refused: docs/plan/per_class_exact.md is not frozen at E115_SHA"; exit 2; }
  uv run --frozen python exp/exp115_per_class_exact.py run --e113-private "$E113PRIV"
  cp exp/out/exp115_summary.json "$OUT/"
  echo "wrote $OUT/exp115_summary.json"
elif [ "$MODE" = crosscheck ]; then
  test -f exp/exp115_crosscheck.py || { echo "E115_SHA has no exp/exp115_crosscheck.py"; exit 2; }
  uv run --frozen python -I exp/exp115_crosscheck.py --e113-private "$E113PRIV" --repo . --out "$OUT/exp115_crosscheck.json"
  echo "wrote $OUT/exp115_crosscheck.json"
else
  echo "E115_MODE is run or crosscheck, got $MODE"; exit 2
fi
