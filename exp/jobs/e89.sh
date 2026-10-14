#!/bin/bash
# exp89 (docs/plan/finetuned_checkpoints.md, amended 1 October 2026): Forest Loss Driver (arm F) and FT-AWF (arm A)
# report-only; Mangrove (arm M, graded) waits for Ai2's validation split; Nandi (arm N) is not public. One script; the
# mode and arm come from the environment.
#
#   E89_MODE=inv    the inventory: downloads the pinned checkpoint and dataset to scratch, checks them, extracts the
#                   dataset, lists the checkpoint keys and counts windows and labels per split. No model. Arm F: the
#                   42 GB tar is downloaded first with a resumable curl at its pinned object generation (resubmit to
#                   resume), then checked against its pinned size and MD5 (its SHA-256 is recorded), and only the
#                   layers the run reads are extracted.
#   E89_MODE=smoke  S1 (numpy), the torch smoke on a synthetic tar and checkpoint, then S2: the pinned checkpoint on
#                   512 real TRAINING windows (in-sample, never graded). Allowed before freezing.
#   E89_MODE=gate   G: accuracy only on the validation windows. For arms A and F it is the alignment check, reported
#                   with its tolerance. The script refuses it until the page is frozen.
#   E89_MODE=run    the full run. Refused until the page is frozen and exp89_gate_<arm>.json records a pass (arm M)
#                   or an alignment check (arms A and F; outside the tolerance, the numbers read "replica not aligned").
#   E89_ARM=fld|awf|mangrove|nandi (default fld). mangrove and nandi are reported as not run by every scoring mode.
#
# Submit from the Mac; the login node runs one sbatch and nothing else. The script travels on stdin, and the job resets
# the checkout to origin/main, so commit and push first. The partition is given per mode: the inventory on cpu (arm F's
# download takes hours), every model pass on a GPU partition, as scripts/score_area.sh:
#   ssh aicr 'E89_MODE=inv E89_ARM=fld sbatch --parsable -p cpu -t 12:00:00' < exp/jobs/e89.sh
#   ssh aicr 'E89_MODE=inv E89_ARM=awf sbatch --parsable -p cpu -t 01:00:00' < exp/jobs/e89.sh
#   ssh aicr 'E89_MODE=smoke E89_ARM=fld sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -t 01:00:00' < exp/jobs/e89.sh
#   ssh aicr 'E89_MODE=gate E89_ARM=fld sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -t 01:00:00' < exp/jobs/e89.sh
#   On the CPU instead (used since job 1151050): E89_DEVICE=cpu ... sbatch -p cpu -c 32 --mem=120G
#   ssh aicr 'E89_MODE=run E89_ARM=fld sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -t 03:00:00 --dependency=afterok:GATE_JOB' < exp/jobs/e89.sh
#
# Every mode writes tracked files in exp/out, and the hard reset below restores tracked files: never let two of these
# jobs, or this and another job that writes tracked outputs, overlap (chain with --dependency=afterany).
# The gate's attempts are also appended to a ledger outside the checkout (E89_GATE_LEDGER below). A reset that restores
# an older committed exp89_gate_<arm>.json cannot lower the attempt count: the gate restores the file from the ledger,
# and the run accepts a pass (or an alignment check) only when the ledger holds it. Never delete that directory.
#SBATCH -A p2026_0089_neu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J e89ft
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -u
MODE=${E89_MODE:-inv}
ARM=${E89_ARM:-fld}
cd /home/qi_zim_neu/olmoearth_inferenceX || exit 1
git fetch -q origin main; git reset -q --hard origin/main
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
export PYTHONUNBUFFERED=1
export HF_HOME=$SCRATCH/hf                 # every Hub download (checkpoint, tar, geojson, encoder config) lands on scratch
export UV_CACHE_DIR=$SCRATCH/uv-cache
export E89_DATA=$SCRATCH/data/exp89        # each arm's tar is extracted under $E89_DATA/<arm>; arm A never reads data/awf
export E89_GATE_LEDGER=/home/qi_zim_neu/exp89_gate_ledger   # outside the checkout and off the purged scratch
export PATH="$HOME/.local/bin:$PATH"
DATA=$E89_DATA/$ARM
RUN="uv run --extra encoder --extra geo"
PY="$RUN python exp/exp89_finetuned_checkpoints.py"

# arm F's dataset: on Google Cloud Storage, pinned by object generation, size and MD5 (the script holds the same pins)
FLD_URL=https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/forest_loss_driver/20251029/dataset.tar
FLD_GENERATION=1761857427036506
FLD_BYTES=42214604800
FLD_TAR=$SCRATCH/downloads/fld_dataset_20251029.tar
TAR_ARGS=()
if [ "$ARM" = "fld" ] && [ -f "$FLD_TAR" ]; then TAR_ARGS=(--tar "$FLD_TAR"); fi

fetch_fld() {
  # Resumable: a partial file is continued with -C -, and an interrupted job is resubmitted to go on. A GET at another
  # object generation returns 404, so a replaced object is never appended to the bytes already on disk.
  mkdir -p "$(dirname "$FLD_TAR")"
  for attempt in $(seq 1 30); do
    have=$(stat -c %s "$FLD_TAR" 2>/dev/null || echo 0)
    if [ "$have" = "$FLD_BYTES" ]; then echo "arm F tar: $have bytes, the pinned size"; return 0; fi
    if [ "$have" -gt "$FLD_BYTES" ]; then echo "arm F tar: $have bytes, more than the pinned $FLD_BYTES; remove $FLD_TAR"; return 1; fi
    echo "== $(date -Is) arm F tar: $have of $FLD_BYTES bytes, curl attempt $attempt =="
    curl -fL --connect-timeout 60 --speed-limit 100000 --speed-time 120 -C - -o "$FLD_TAR" \
      "$FLD_URL?generation=$FLD_GENERATION" || sleep 30
  done
  echo "arm F tar: still incomplete after 30 attempts; resubmit the inv job to resume"
  return 1
}

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): mode $MODE, arm $ARM, data $DATA, commit $(git rev-parse --short HEAD) =="
case "$MODE" in
  smoke|gate|run)
    if [ "${E89_DEVICE:-}" = "cpu" ]; then
      echo "E89_DEVICE=cpu: the model runs on the CPU (the GPU path hit a device mismatch in job 1151050)"
    elif ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
      echo "mode $MODE runs the model and needs a GPU: submit with -p b200-devel,rtx-devel -q interactive --gpus=1"
      exit 2
    fi ;;
esac
$RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
mkdir -p "$E89_DATA"

case "$MODE" in
  inv)
    echo "== S1 =="
    $PY --smoke --out-dir "$SCRATCH/exp89_smoke" || exit 1
    if [ "$ARM" = "fld" ] && [ ! -f "$DATA/.exp89_extracted" ]; then
      echo "== arm F: the pinned tar =="
      fetch_fld || exit 1
      TAR_ARGS=(--tar "$FLD_TAR")
      df -h "$SCRATCH" | tail -1
    fi
    echo "== inventory (no model on any window) =="
    $PY --inventory --arm "$ARM" --data "$DATA" ${TAR_ARGS[@]+"${TAR_ARGS[@]}"} || exit 1
    ls -la "exp/out/exp89_inventory_$ARM.json" ;;
  smoke)
    echo "== S1 =="
    $PY --smoke --out-dir "$SCRATCH/exp89_smoke" || exit 1
    echo "== the torch smoke on a synthetic tar and checkpoint =="
    $PY --smoke-torch --out-dir "$SCRATCH/exp89_smoke" || exit 1
    echo "== S2: the pinned checkpoint on real training windows =="
    $PY --smoke-torch --real --arm "$ARM" --data "$DATA" ${TAR_ARGS[@]+"${TAR_ARGS[@]}"}
    rc=$?
    ls -la exp/out/exp89_s2_"$ARM".json 2>/dev/null
    exit $rc ;;
  gate)
    echo "== G: accuracy only (arms A and F: the alignment check) =="
    $PY --gate --arm "$ARM" --data "$DATA" ${TAR_ARGS[@]+"${TAR_ARGS[@]}"}
    rc=$?
    cat exp/out/exp89_gate_"$ARM".json 2>/dev/null
    echo "== the gate's ledger =="
    cat "$E89_GATE_LEDGER/exp89_gate_$ARM.ledger.jsonl" 2>/dev/null
    exit $rc ;;
  run)
    echo "== the full run =="
    $PY --arm "$ARM" --data "$DATA" ${TAR_ARGS[@]+"${TAR_ARGS[@]}"}
    rc=$?
    ls -la exp/out/exp89_summary.json exp/out/exp89_units_"$ARM".npz 2>/dev/null
    exit $rc ;;
  *)
    echo "unknown E89_MODE '$MODE': inv, smoke, gate or run"; exit 2 ;;
esac
