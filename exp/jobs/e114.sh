#!/bin/bash
# exp114 (docs/plan/prior_shift_pathway.md; the page is the specification): prior shift or feature shift on exp113's
# 18 maps. Helper: exp/exp114_prior_shift.py in the record's uv environment, with scikit-learn added by
# `uv run --with` (the record's lock is not changed). No download: exp113's data and private window arrays on scratch.
#
#   E114_MODE=smoke  CPU, 8 CPUs, 16 GB, 30 min. The same code on a few patches per configuration; nothing graded.
#   E114_MODE=run    CPU, 16 CPUs (as exp113's run: the fit's BLAS reductions depend on the thread count), 32 GB,
#                    12 h (the page's Cost said up to 3 hours; the pre-run review put the worst case, every EM at its
#                    10,000-iteration cap, at 6 to 9 hours). Refused unless the page is frozen at E114_SHA. Refits
#                    exp113's maps, checks all 18 against exp113's stored windows before computing anything else,
#                    then corrects, estimates and grades; writes $OUT/exp114_partial.json after each configuration and
#                    exp114_summary.json to $OUT at the end (copy to exp/out/ when recording).
#
# Submit after commit and push; the login node runs one sbatch per line and nothing else:
#   SHA=$(git rev-parse HEAD)
#   ssh aicr "E114_MODE=smoke E114_SHA=$SHA sbatch --parsable -J e114smoke -p cpu -c 8 --mem=16G -t 00:30:00" < exp/jobs/e114.sh
#   ssh aicr "E114_MODE=run E114_SHA=$SHA sbatch --parsable -J e114run -p cpu -c 16 --mem=32G -t 12:00:00" < exp/jobs/e114.sh
#SBATCH -A p2026_0089_neu
#SBATCH -J e114
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -eu
umask 077
MODE=${E114_MODE:?E114_MODE is smoke or run}
E114_SHA=${E114_SHA:?E114_SHA is the commit to run}
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e114_code
DATA=$SCRATCH/e113_data
E113PRIV=$SCRATCH/e113_private
OUT=$SCRATCH/e114_out
export UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
test -d "$DATA/EMB_OLMOEARTH_V1_3" && test -f "$E113PRIV/exp113_windows_default_oe13.npz" \
  || { echo "exp113's data or private window arrays are missing"; exit 2; }

if [ ! -d "$CODE/.git" ]; then
  git clone -q https://github.com/2imi9/olmoearth_inferenceX-record.git "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E114_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E114_SHA" \
  || { echo "E114_SHA=$E114_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E114_SHA"
[ "$(git rev-parse HEAD)" = "$E114_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E114_SHA"; exit 2; }
test -f exp/exp114_prior_shift.py && test -f docs/plan/prior_shift_pathway.md || { echo "E114_SHA predates exp114"; exit 2; }
PY=(uv run --frozen --with "scikit-learn==1.9.1" python exp/exp114_prior_shift.py)
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp114 $MODE at $E114_SHA =="

if [ "$MODE" = smoke ]; then
  "${PY[@]}" smoke --data "$DATA" --e113-private "$E113PRIV"
elif [ "$MODE" = run ]; then
  grep -q '^\*\*Status: frozen' docs/plan/prior_shift_pathway.md \
    || { echo "refused: docs/plan/prior_shift_pathway.md is not frozen at E114_SHA"; exit 2; }
  "${PY[@]}" run --data "$DATA" --e113-private "$E113PRIV" --partial "$OUT/exp114_partial.json"
  cp exp/out/exp114_summary.json "$OUT/"
  echo "wrote $OUT/exp114_summary.json"
else
  echo "E114_MODE is smoke or run, got $MODE"; exit 2
fi
