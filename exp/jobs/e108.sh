#!/bin/bash
# exp108 (report-only, not preregistered): the certified zone against conformal risk control and conformal
# selection on exp95's 34 committed cells (in the clone). Numpy only.
#
#   SHA=$(git rev-parse HEAD)
#   ssh aicr "E108_SHA=$SHA sbatch --parsable -J e108 -p cpu -c 2 --mem=16G -t 04:00:00" < exp/jobs/e108.sh
#
# Writes: the clone $SCRATCH/e108_code (its .venv by uv), $SCRATCH/e108_out (exp108_summary.json, the smoke's), the
# Slurm log. Nothing downloaded beyond the clone and its Python packages. Expected: under two hours (34 cells,
# two budgets, 2,000 draws, three arms).
#SBATCH -A p2026_0089_neu
#SBATCH -J e108
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e108_code
OUTDIR=$SCRATCH/e108_out
[[ "${E108_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E108_SHA must be the full 40-character commit (got '${E108_SHA:-}')"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E108_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E108_SHA" \
  || { echo "E108_SHA=$E108_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E108_SHA"
[ "$(git rev-parse HEAD)" = "$E108_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E108_SHA"; exit 2; }
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=2
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUTDIR"
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp108, commit $E108_SHA =="
echo "== smoke =="
uv run python exp/exp108_zone_vs_conformal.py --smoke --out-dir "$OUTDIR/smoke"
echo "== the run =="
uv run python exp/exp108_zone_vs_conformal.py --out-dir "$OUTDIR"
echo "== $(date -Is) done: $OUTDIR/exp108_summary.json =="
