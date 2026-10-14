#!/bin/bash
# exp106 (docs/plan/second_opinion.md): does disagreement with an independent product find a product's confident
# errors? On the point files exp92 and exp94 committed (in the clone). Numpy and csv only.
# Refused by the script unless the page is frozen at E106_SHA.
#
#   SHA=$(git rev-parse HEAD)
#   ssh aicr "E106_SHA=$SHA sbatch --parsable -J e106 -p cpu -c 2 --mem=8G -t 01:00:00" < exp/jobs/e106.sh
#
# Writes: the clone $SCRATCH/e106_code (its .venv by uv), $SCRATCH/e106_out (exp106_summary.json, the smoke's), the
# Slurm log. Nothing downloaded beyond the clone and its Python packages. Expected: minutes (three cells and
# a control, 2,000 bootstrap resamples each).
#SBATCH -A p2026_0089_neu
#SBATCH -J e106
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e106_code
OUTDIR=$SCRATCH/e106_out
[[ "${E106_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E106_SHA must be the full 40-character commit (got '${E106_SHA:-}')"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E106_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E106_SHA" \
  || { echo "E106_SHA=$E106_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E106_SHA"
[ "$(git rev-parse HEAD)" = "$E106_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E106_SHA"; exit 2; }
grep -q '^\*\*Status: frozen' docs/plan/second_opinion.md || { echo "the page is not frozen at E106_SHA"; exit 2; }
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=2
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUTDIR"
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp106, commit $E106_SHA =="
echo "== smoke =="
uv run python exp/exp106_second_opinion.py --smoke --out-dir "$OUTDIR/smoke"
echo "== the run =="
uv run python exp/exp106_second_opinion.py --out-dir "$OUTDIR"
echo "== $(date -Is) done: $OUTDIR/exp106_summary.json =="
