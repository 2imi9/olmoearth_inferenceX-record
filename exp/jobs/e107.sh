#!/bin/bash
# exp107 (docs/plan/which_map_baselines.md): which of two maps is more accurate, the tool's design against
# McNemar's test, exact probabilities from the counts exp92 and exp94 recorded (in the clone). Numpy only.
# Refused by the script unless the page is frozen at E107_SHA.
#
#   SHA=$(git rev-parse HEAD)
#   ssh aicr "E107_SHA=$SHA sbatch --parsable -J e107 -p cpu -c 2 --mem=8G -t 01:00:00" < exp/jobs/e107.sh
#
# Writes: the clone $SCRATCH/e107_code (its .venv by uv), $SCRATCH/e107_out (exp107_summary.json, the smoke's), the
# Slurm log. Nothing downloaded beyond the clone and its Python packages. Expected: minutes (nine pairs,
# five budgets, exact sums).
#SBATCH -A p2026_0089_neu
#SBATCH -J e107
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e107_code
OUTDIR=$SCRATCH/e107_out
[[ "${E107_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E107_SHA must be the full 40-character commit (got '${E107_SHA:-}')"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E107_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E107_SHA" \
  || { echo "E107_SHA=$E107_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E107_SHA"
[ "$(git rev-parse HEAD)" = "$E107_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E107_SHA"; exit 2; }
grep -q '^\*\*Status: frozen' docs/plan/which_map_baselines.md || { echo "the page is not frozen at E107_SHA"; exit 2; }
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=2
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUTDIR"
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp107, commit $E107_SHA =="
echo "== smoke =="
uv run python exp/exp107_which_map_baselines.py --smoke --out-dir "$OUTDIR/smoke"
echo "== the run =="
uv run python exp/exp107_which_map_baselines.py --out-dir "$OUTDIR"
echo "== $(date -Is) done: $OUTDIR/exp107_summary.json =="
