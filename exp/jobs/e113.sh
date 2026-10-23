#!/bin/bash
# exp113 (docs/plan/planteur_transfer.md; the page is the specification): PLANTEUR leave-one-island-out with three 2019
# embeddings. Helper: exp/exp113_planteur_transfer.py in the record's uv environment, with scikit-learn and
# huggingface_hub added by `uv run --with` (the record's lock is not changed).
#
#   E113_MODE=fetch  CPU, 8 CPUs, 16 GB, 3 h. Downloads from Hugging Face (allenai/planteur-benchmark, public, no
#                    token) the labels, the three embeddings (OlmoEarth v1.3, TESSERA v2, AlphaEarth), metadata.geojson
#                    and the split files into $DATA, about 58 GB; no imagery. Resumable (the hub skips finished files);
#                    refuses if any folder holds fewer than 2,750 files.
#   E113_MODE=run    CPU, 32 CPUs, 64 GB, 6 h. Refused unless the page is frozen at E113_SHA. Fits the 18 maps, reads
#                    every test patch once, grades, and writes exp113_summary.json to $OUT (copy to exp/out/ when
#                    recording) and the per-window arrays to $PRIV (no coordinate; kept off the repository).
#
# Submit after commit and push; the login node runs one sbatch per line and nothing else:
#   SHA=$(git rev-parse HEAD)
#   F=$(ssh aicr "E113_MODE=fetch E113_SHA=$SHA sbatch --parsable -J e113fetch -p cpu -c 8 --mem=16G -t 03:00:00" < exp/jobs/e113.sh)
#   ssh aicr "E113_MODE=run E113_SHA=$SHA sbatch --parsable -J e113run -p cpu -c 32 --mem=64G -t 06:00:00 --dependency=afterok:$F" < exp/jobs/e113.sh
#SBATCH -A p2026_0089_neu
#SBATCH -J e113
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -eu
umask 077
MODE=${E113_MODE:?E113_MODE is fetch or run}
E113_SHA=${E113_SHA:?E113_SHA is the commit to run}
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e113_code
DATA=$SCRATCH/e113_data
PRIV=$SCRATCH/e113_private
OUT=$SCRATCH/e113_out
export UV_CACHE_DIR=$SCRATCH/uv-cache HF_HOME=$SCRATCH/hf PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$DATA" "$PRIV" "$OUT"
chmod 700 "$PRIV"

if [ ! -d "$CODE/.git" ]; then
  git clone -q https://github.com/2imi9/olmoearth_inferenceX-record.git "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E113_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E113_SHA" \
  || { echo "E113_SHA=$E113_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E113_SHA"
[ "$(git rev-parse HEAD)" = "$E113_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E113_SHA"; exit 2; }
test -f exp/exp113_planteur_transfer.py && test -f docs/plan/planteur_transfer.md || { echo "E113_SHA predates exp113"; exit 2; }
PY=(uv run --frozen --with "scikit-learn>=1.4" --with "huggingface_hub>=0.24" python exp/exp113_planteur_transfer.py)
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp113 $MODE at $E113_SHA =="

if [ "$MODE" = fetch ]; then
  "${PY[@]}" fetch --data "$DATA"
  echo "data: $(du -sh "$DATA" | cut -f1)"
elif [ "$MODE" = run ]; then
  grep -q '^\*\*Status: frozen' docs/plan/planteur_transfer.md \
    || { echo "refused: docs/plan/planteur_transfer.md is not frozen at E113_SHA"; exit 2; }
  "${PY[@]}" run --data "$DATA" --private-dir "$PRIV"
  cp exp/out/exp113_summary.json "$OUT/"
  echo "wrote $OUT/exp113_summary.json; private: $(du -sh "$PRIV" | cut -f1)"
else
  echo "E113_MODE is fetch or run, got $MODE"; exit 2
fi
