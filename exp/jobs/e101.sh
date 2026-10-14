#!/bin/bash
# exp101 (docs/plan/two_step.md): filter, then shake. The record's replica of FT-AWF on 320 perturbed views of each unit
# (exp98's 259 graded validation points, F; exp99's 271 plots with 2017 input, E), read from the deployed runs' own
# rslearn datasets on scratch. One script; the mode comes from the environment:
#
#   E101_MODE=gpusmoke  F's first 32 units: V0 on the GPU against V0 on the CPU (logits within 1e-3, the same class),
#                       the script's sample builder against exp21.logits_grid on the CPU, and the throughput of all 320
#                       views (views per second, the fit's and the test's estimated minutes). Under 15 minutes. Allowed
#                       before freezing; run it after inv, which builds the environment on the CPU.
#   E101_MODE=inv       counts only, CPU, no model: the units of F and E whose block and twelve item groups can be read.
#                       Run it first: its uv sync builds the torch environment in the scratch clone, so the GPU smoke's
#                       15 minutes are not spent installing. Set fit's and test's -t from the smoke's estimated minutes
#                       with a margin of about 1.5x.
#   E101_MODE=fit       F only, one GPU: views, instability, the choice of q -> exp101_fit.json, exp101_units_fit.npz.
#                       Allowed before freezing.
#   E101_MODE=test      F again (q, checked against exp101_fit.json) and E, one GPU -> exp101_summary.json,
#                       exp101_units.npz. exp/exp101_two_step.py refuses it until the page says frozen.
#
# Submit from the Mac after commit and push, with the full commit; the login node runs one sbatch and nothing else:
#   ssh aicr "E101_MODE=gpusmoke E101_SHA=<full sha> sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -c 8 --mem=64G -t 00:15:00" < exp/jobs/e101.sh
#   ssh aicr "E101_MODE=inv E101_SHA=<full sha> sbatch --parsable -p cpu -c 4 --mem=16G -t 00:30:00" < exp/jobs/e101.sh
#   ssh aicr "E101_MODE=fit E101_SHA=<full sha> sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -c 16 --mem=128G -t 02:00:00" < exp/jobs/e101.sh
#   ssh aicr "E101_MODE=test E101_SHA=<full sha> sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -c 16 --mem=128G -t 02:00:00" < exp/jobs/e101.sh
# The CPU fallback, documented and not the default (a deviation the page names, used only if the GPU path cannot be made
# to work; about 12 hours for the test):
#   ssh aicr "E101_MODE=test E101_DEVICE=cpu E101_SHA=<full sha> sbatch --parsable -p cpu -c 32 --mem=120G -t 12:00:00" < exp/jobs/e101.sh
#
# Inputs; any missing one stops the job at once, and nothing is downloaded:
#   exp98's and exp99's score parts (data/exp98/scores and data/exp99/scores in the home checkout, or E101_EXP98 and
#   E101_EXP99); exp89's AWF label windows on scratch (data/exp89/awf/dataset/windows); the deployed runs' rslearn roots
#   on scratch, named in deploy/prepare_inventory.json (2023, under deploy/awf_run) and deploy/prepare_inventory_e99.json
#   (2017, under deploy/awf2017_run), both purged after 30 days; the TimeSync sample and Natural Earth's boundaries in
#   data/breadth (checked against their pins, never fetched here); Ai2's request geometry in the pinned clone on
#   scratch; the pinned checkpoint and OlmoEarth v1-Base's config.json in the scratch HF cache (HF_HUB_OFFLINE=1).
#   Without the 2023 roots the job stops, unless E101_F_FALLBACK=1, which lets F read exp89's label windows instead (a
#   deviation the page names; the summary records it).
# Outputs: $SCRATCH/e101_out, never a checkout (a few MB of JSON and npz). Memory: the units' blocks are about 0.4 GB.
# E101_BATCH_TOKENS sets the encoder tokens per forward (default 131072 in the script).
#SBATCH -A p2026_0089_neu
#SBATCH -J e101
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
MODE=${E101_MODE:-}
case "$MODE" in
  gpusmoke|inv|fit|test) ;;
  *) echo "E101_MODE must be gpusmoke, inv, fit or test (got '$MODE')"; exit 2 ;;
esac
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e101_code
OUTDIR=$SCRATCH/e101_out
[ -n "${E101_SHA:-}" ] || { echo "E101_SHA (the full commit to run) is required"; exit 2; }
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
git cat-file -e "$E101_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E101_SHA" \
  || { echo "E101_SHA=$E101_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E101_SHA"
[ "$(git rev-parse HEAD)" = "$E101_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E101_SHA=$E101_SHA (give the full sha)"; exit 2; }
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PATH="$HOME/.local/bin:$PATH"
RUN="uv run --extra encoder --extra geo"
PY="$RUN python exp/exp101_two_step.py"

# the device: a GPU for every model mode unless E101_DEVICE=cpu (the documented fallback); the GPU smoke needs a GPU
DEVICE=${E101_DEVICE:-cuda}
case "$MODE" in
  gpusmoke|fit|test)
    if [ "$DEVICE" = cpu ]; then
      [ "$MODE" != gpusmoke ] || { echo "the GPU smoke needs a GPU; E101_DEVICE=cpu is for fit and test only"; exit 2; }
      echo "E101_DEVICE=cpu: the views run on the CPU, a deviation the page names (record it)"
    elif ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
      echo "mode $MODE runs the model on a GPU: submit with -p b200-devel,rtx-devel -q interactive --gpus=1"
      exit 2
    fi ;;
esac

E98_SCORES=${E101_EXP98:-$REPO/data/exp98/scores}
E99_SCORES=${E101_EXP99:-$REPO/data/exp99/scores}
LABELS=$SCRATCH/data/exp89/awf/dataset/windows
TS=$REPO/data/breadth/eastafrica_yearly_point_data.csv
NE=$REPO/data/breadth/ne_10m_admin_0_countries.geojson
AI2=$SCRATCH/deploy/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
INV23=$SCRATCH/deploy/prepare_inventory.json          # scratch's copies hold the real root names; the tracked ones do not
INV17=$SCRATCH/deploy/prepare_inventory_e99.json
parts() {
  for d in "$1"/*/; do
    if compgen -G "$d/scores_*.tif" > /dev/null; then echo "${d%/}"; fi
  done
}
roots() {   # the dataset roots an inventory names, one per line; the names are never echoed
  python3 -c 'import json,sys; print("\n".join(sys.argv[2] + "/" + r["root"] for r in json.load(open(sys.argv[1]))["roots"]))' "$1" "$2"
}
check_parts() {   # $1 the scores directory; the parts in it must match parts.json when it is there
  local s=$1 n=$2
  [ "$n" -gt 0 ] || { echo "no part directory with scores_<EPSG>.tif in $s"; exit 1; }
  if [ -f "$s/parts.json" ]; then
    NP=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$s/parts.json")
    [ "$NP" = "$n" ] || { echo "$s/parts.json lists $NP parts, $s holds $n"; exit 1; }
  fi
}

# F's inputs (every mode)
test -d "$E98_SCORES" || { echo "$E98_SCORES is not a directory"; exit 1; }
mapfile -t D98 < <(parts "$E98_SCORES")
check_parts "$E98_SCORES" "${#D98[@]}"
test -d "$LABELS" || { echo "no AWF label windows at $LABELS (scratch purge?): exp/jobs/e89.sh E89_MODE=inv E89_ARM=awf"; exit 1; }
DEP23=()
if [ -f "$INV23" ]; then
  mapfile -t DEP23 < <(roots "$INV23" "$SCRATCH/deploy/awf_run")
  for d in "${DEP23[@]}"; do
    if [ ! -d "$d/windows" ]; then DEP23=(); break; fi
  done
fi
DEP23_ARGS=()
if [ ${#DEP23[@]} -gt 0 ]; then
  DEP23_ARGS=(--deploy-2023 "${DEP23[@]}")
elif [ "${E101_F_FALLBACK:-0}" = 1 ]; then
  echo "NOTE: the 2023 rslearn roots are missing (scratch purge?); E101_F_FALLBACK=1: F reads exp89's label windows, a deviation the page names"
  DEP23_ARGS=(--f-fallback)
else
  echo "the 2023 rslearn roots named in $INV23 are missing (scratch purge?); set E101_F_FALLBACK=1 to read exp89's label windows instead (a deviation the page names)"
  exit 1
fi

# E's inputs (inventory and test)
E_ARGS=()
if [ "$MODE" = inv ] || [ "$MODE" = test ]; then
  test -d "$E99_SCORES" || { echo "$E99_SCORES is not a directory"; exit 1; }
  mapfile -t D99 < <(parts "$E99_SCORES")
  check_parts "$E99_SCORES" "${#D99[@]}"
  test -f "$INV17" || { echo "no $INV17 (e99_prepare.sh; purged after 30 days)"; exit 1; }
  mapfile -t DEP17 < <(roots "$INV17" "$SCRATCH/deploy/awf2017_run")
  for d in "${DEP17[@]}"; do
    test -d "$d/windows" || { echo "a 2017 rslearn root named in $INV17 holds no windows/ (scratch purge?)"; exit 1; }
  done
  test -f "$TS" || { echo "no TimeSync sample at $TS; this job does not fetch it"; exit 1; }
  test -f "$AI2" || { echo "Ai2's request geometry is missing from the pinned clone on scratch (exp/jobs/e98_env.sh)"; exit 1; }
  CHECK=("$TS" timesync)
  if [ "$MODE" = test ]; then
    test -f "$NE" || { echo "no Natural Earth boundaries at $NE; this job does not fetch them"; exit 1; }
    CHECK+=("$NE" boundaries)
  fi
  $RUN python - "${CHECK[@]}" <<'EOF'
import sys
sys.path.insert(0, "exp")
import timesync_awf_crosswalk as tsx
args = sys.argv[1:]
for path, which in zip(args[::2], args[1::2]):
    ok, got = tsx.verify(path, tsx.SOURCES[which])
    print(f"the {which} file against its pin:", "ok" if ok else got)
    if not ok:
        sys.exit(1)
EOF
  E_ARGS=(--exp99-scores "${D99[@]}" --timesync "$TS" --request-geometry "$AI2" --deploy-2017 "${DEP17[@]}")
  if [ "$MODE" = test ]; then E_ARGS+=(--boundaries "$NE"); fi
fi

# the pinned checkpoint and the encoder's config, from the scratch HF cache only (every model mode)
if [ "$MODE" != inv ]; then
  $RUN python - <<'EOF'
import os, sys
sys.path.insert(0, "exp")
import exp89_finetuned_checkpoints as e89
from huggingface_hub import hf_hub_download
from olmoearth_pretrain.model_loader import ModelID
A = e89.ARMS["awf"]
try:
    e89.hub_file(A["model"], A["model_file"], A["model_revision"], "model", A["model_sha256"])
    hf_hub_download(ModelID["OLMOEARTH_V1_BASE"].repo_id(), "config.json",
                    revision=e89.ENCODER_REVISIONS["OLMOEARTH_V1_BASE"])
except Exception as ex:
    print(f"the pinned checkpoint or the encoder config is not in the HF cache at {os.environ.get('HF_HOME')} "
          f"(HF_HUB_OFFLINE=1, nothing is downloaded): {type(ex).__name__}: {ex}")
    sys.exit(1)
print("the pinned checkpoint (sha256 checked) and the encoder config are in the scratch HF cache")
EOF
fi

mkdir -p "$OUTDIR"
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp101 mode $MODE, device $DEVICE, commit $(git rev-parse --short HEAD); ${#D98[@]} exp98 parts, ${#DEP23[@]} 2023 root(s) =="
$RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
case "$MODE" in
  gpusmoke)
    $PY --gpu-smoke --exp98-scores "${D98[@]}" --labels "$LABELS" ${DEP23_ARGS[@]+"${DEP23_ARGS[@]}"} \
      --device "$DEVICE" --out-dir "$OUTDIR"
    ls -la "$OUTDIR/exp101_gpusmoke.json" ;;
  inv)
    echo "== smoke =="
    $PY --smoke --out-dir "$OUTDIR/smoke"
    echo "== inventory (counts only) =="
    $PY --inventory --exp98-scores "${D98[@]}" --labels "$LABELS" ${DEP23_ARGS[@]+"${DEP23_ARGS[@]}"} "${E_ARGS[@]}" \
      --out-dir "$OUTDIR"
    ls -la "$OUTDIR/exp101_inventory.json" ;;
  fit)
    echo "== smoke =="
    $PY --smoke --out-dir "$OUTDIR/smoke"
    echo "== the fit (F only) =="
    $PY --fit --exp98-scores "${D98[@]}" --labels "$LABELS" ${DEP23_ARGS[@]+"${DEP23_ARGS[@]}"} --device "$DEVICE" \
      --out-dir "$OUTDIR"
    ls -la "$OUTDIR/exp101_fit.json" "$OUTDIR/exp101_units_fit.npz" ;;
  test)
    echo "== smoke =="
    $PY --smoke --out-dir "$OUTDIR/smoke"
    FIT_ARGS=()
    if [ -f "$OUTDIR/exp101_fit.json" ]; then FIT_ARGS=(--fit-record "$OUTDIR/exp101_fit.json"); fi
    echo "== the test (refused unless the page is frozen) =="
    $PY --test --exp98-scores "${D98[@]}" --labels "$LABELS" ${DEP23_ARGS[@]+"${DEP23_ARGS[@]}"} "${E_ARGS[@]}" \
      ${FIT_ARGS[@]+"${FIT_ARGS[@]}"} --device "$DEVICE" --out-dir "$OUTDIR"
    ls -la "$OUTDIR/exp101_summary.json" "$OUTDIR/exp101_units.npz" ;;
esac
du -sh "$OUTDIR" | sed 's/^/outputs: /'
df -h "$SCRATCH" | tail -1 | sed 's/^/scratch: /'
echo "== $(date -Is) done =="
