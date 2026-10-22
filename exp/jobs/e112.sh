#!/bin/bash
# exp112 (docs/plan/v13_embeddings.md; the page is the specification): a map made from OlmoEarth v1.3 embeddings at
# LCMAP's reference plots. Helper: exp/exp112_v13_lcmap.py in the record's uv environment, with zarr, obstore, pyproj
# and scikit-learn added by `uv run --with` (the record's lock is not changed).
#
#   E112_MODE=extract  CPU, 16 CPUs, 32 GB, 4 h. Fetches LCMAP Reference v1.2 from USGS ScienceBase (public, 29.9 MB)
#                      unless present, checks its sha256 (exp92's), unzips it under $PRIV; draws S (seed 112); reads the
#                      2018 embedding at each plot of S from Ai2's store on Source Cooperative (anonymous, about 6 MB
#                      moved per plot, about 70 GB in all), 16 reads at a time, into $PRIV/exp112_vectors.jsonl, one
#                      line per plot (resumable: a rerun reads only the plots not yet there). Allowed before freezing:
#                      it reads inputs, grades nothing. Stops after 20 failed reads in a row.
#   E112_MODE=analyze  CPU, 4 CPUs, 16 GB, 1 h. Refused unless the page is frozen at E112_SHA; writes
#                      exp112_summary.json and exp112_plots.csv to $OUT, to be copied to exp/out/ when recording.
#
# Positions and embeddings stay in $PRIV (owner only), never in the repository.
#
# Submit after commit and push; the login node runs one sbatch per line and nothing else:
#   SHA=$(git rev-parse HEAD)
#   ssh aicr "E112_MODE=extract E112_SHA=$SHA sbatch --parsable -J e112ext -p cpu -c 16 --mem=32G -t 04:00:00" < exp/jobs/e112.sh
#   ssh aicr "E112_MODE=analyze E112_SHA=$SHA sbatch --parsable -J e112ana -p cpu -c 4 --mem=16G -t 01:00:00" < exp/jobs/e112.sh
#SBATCH -A p2026_0089_neu
#SBATCH -J e112
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -eu
umask 077
MODE=${E112_MODE:?E112_MODE is extract or analyze}
E112_SHA=${E112_SHA:?E112_SHA is the commit to run}
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e112_code
PRIV=$SCRATCH/e112_private
OUT=$SCRATCH/e112_out
ZIP_URL="https://www.sciencebase.gov/catalog/file/get/5e42e54be4b0edb47be84535?name=LCMAP_CU_20211117_V01_REF.zip"
ZIP_SHA256=eb935d96b51e2a6bb60105fd5bb8bf170f4fbcc87318d8abb5746c32022b5b4c
NCPU=${SLURM_CPUS_PER_TASK:-4}
export UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$PRIV/data/lcmap" "$OUT"
chmod 700 "$PRIV"

if [ ! -d "$CODE/.git" ]; then
  git clone -q https://github.com/2imi9/olmoearth_inferenceX-record.git "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E112_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E112_SHA" \
  || { echo "E112_SHA=$E112_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E112_SHA"
[ "$(git rev-parse HEAD)" = "$E112_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E112_SHA"; exit 2; }
test -f exp/exp112_v13_lcmap.py && test -f docs/plan/v13_embeddings.md || { echo "E112_SHA predates exp112"; exit 2; }
ln -sfn "$PRIV/data" data            # the script reads data/lcmap/...; the link is in the scratch checkout only
PY=(uv run --frozen --with "zarr>=3.1,<4" --with "obstore>=0.6" --with "pyproj>=3.6" --with "scikit-learn>=1.4"
    python exp/exp112_v13_lcmap.py)
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp112 $MODE at $E112_SHA =="

if [ "$MODE" = extract ]; then
  Z=$PRIV/data/lcmap/LCMAP_CU_20211117_V01_REF.zip
  if [ ! -f "$Z" ]; then
    curl -sS --fail --max-time 600 -o "$Z.part" "$ZIP_URL" && mv "$Z.part" "$Z"
  fi
  echo "$ZIP_SHA256  $Z" | sha256sum -c --quiet || { echo "LCMAP zip sha256 differs from exp92's"; exit 1; }
  test -f "$PRIV/data/lcmap/LCMAP_CU_20211117_V01_REF/LCMAP_CU_20211117_V01_REF.csv" \
    || (cd "$PRIV/data/lcmap" && unzip -q -o "$Z")
  "${PY[@]}" draw --private-dir "$PRIV"
  "${PY[@]}" extract --private-dir "$PRIV" --workers "$NCPU"
  echo "plots read: $(wc -l < "$PRIV/exp112_vectors.jsonl"); scratch: $(du -sh "$SCRATCH/e112_private" | cut -f1)"
elif [ "$MODE" = analyze ]; then
  grep -q '^\*\*Status: frozen' docs/plan/v13_embeddings.md \
    || { echo "refused: docs/plan/v13_embeddings.md is not frozen at E112_SHA"; exit 2; }
  "${PY[@]}" analyze --private-dir "$PRIV"
  cp exp/out/exp112_summary.json exp/out/exp112_plots.csv "$OUT/"
  echo "wrote $OUT/exp112_summary.json and exp112_plots.csv"
else
  echo "E112_MODE is extract or analyze, got $MODE"; exit 2
fi
