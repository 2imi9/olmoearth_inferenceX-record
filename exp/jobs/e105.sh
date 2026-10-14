#!/bin/bash
# exp105 (docs/plan/class_strata_baseline.md): confidence strata against map-class strata for a map's overall error
# rate, on exp78's committed per-unit exports of the embedding suite (exp/out/exp78_units, in the clone). Numpy only.
# Refused by the script unless the page is frozen at E105_SHA.
#
#   SHA=$(git rev-parse HEAD)
#   ssh aicr "E105_SHA=$SHA sbatch --parsable -J e105 -p cpu -c 2 --mem=16G -t 02:00:00" < exp/jobs/e105.sh
#
# Writes: the clone $SCRATCH/e105_code (its .venv by uv), $SCRATCH/e105_out (exp105_summary.json, the smoke's), the
# Slurm log. Nothing downloaded beyond the clone and its Python packages. Expected: a few minutes to an hour (24 tasks,
# 3 budgets, 5 arms, 2,000 draws; the 4,096,000-unit task dominates).
#SBATCH -A p2026_0089_neu
#SBATCH -J e105
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e105_code
OUTDIR=$SCRATCH/e105_out
[[ "${E105_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E105_SHA must be the full 40-character commit (got '${E105_SHA:-}')"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E105_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E105_SHA" \
  || { echo "E105_SHA=$E105_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E105_SHA"
[ "$(git rev-parse HEAD)" = "$E105_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E105_SHA"; exit 2; }
grep -q '^\*\*Status: frozen' docs/plan/class_strata_baseline.md || { echo "the page is not frozen at E105_SHA"; exit 2; }
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=2
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUTDIR"
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp105, commit $E105_SHA =="
echo "== smoke =="
uv run python exp/exp105_class_strata.py --smoke --out-dir "$OUTDIR/smoke"
echo "== the run =="
uv run python exp/exp105_class_strata.py --out-dir "$OUTDIR"
echo "== $(date -Is) done: $OUTDIR/exp105_summary.json =="
