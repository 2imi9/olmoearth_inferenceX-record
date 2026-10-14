#!/bin/bash
# exp102: fine-tune OlmoEarth v1.2 Base on Ai2's AWF labels with Ai2's FT-AWF recipe (exp/jobs/e102_config/model.yaml),
# and retrain v1 Base the same way as the control. rslearn 0.1.17 in its own venv on scratch; the AWF rslearn dataset
# is exp89's extraction of the pinned tar ($SCRATCH/data/exp89/awf/dataset, 1,459 label windows: 1,115 train, 344 val).
#
#   E102_MODE=smoke  both encoders through `rslearn model fit` with fast_dev_run (one batch of each split), then one
#                    full epoch of v1.2 for the time per epoch. Prints the package versions and whether olmoearth's
#                    ModelID knows OLMOEARTH_V1_2_BASE. Run it first.
#   E102_MODE=train  one fine-tune: E102_MODEL (v1_2 or v1) and E102_SEED (an integer), 100 epochs as Ai2's recipe,
#                    checkpoints and a CSV log in $SCRATCH/e102/runs/<model>_s<seed>/.
#
# Submit from the Mac after commit and push, with the full commit; the login node runs one sbatch and nothing else:
#   ssh aicr "E102_MODE=smoke E102_SHA=<sha> sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -c 8 --mem=64G -t 00:45:00" < exp/jobs/e102_train.sh
#   ssh aicr "E102_MODE=train E102_MODEL=v1_2 E102_SEED=0 E102_SHA=<sha> sbatch --parsable -p b200-batch --gpus=1 -c 8 --mem=64G -t <from the smoke>" < exp/jobs/e102_train.sh
# Downloads: the Python packages (PyPI) into the venv, and the two encoders' weights from Hugging Face into the
# scratch HF cache when they are not there (allenai/OlmoEarth-v1-Base, allenai/OlmoEarth-v1_2-Base).
#SBATCH -A p2026_0089_neu
#SBATCH -J e102
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
MODE=${E102_MODE:-}
case "$MODE" in smoke|train) ;; *) echo "E102_MODE must be smoke or train (got '$MODE')"; exit 2 ;; esac
[ -n "${E102_SHA:-}" ] || { echo "E102_SHA (the full commit) is required"; exit 2; }
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e102_code
VENV=$SCRATCH/e102/venv
RUNS=$SCRATCH/e102/runs
export HF_HOME=$SCRATCH/hf UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "no GPU: submit with --gpus=1 on b200-devel, rtx-devel or b200-batch"; exit 2; }

if [ ! -d "$CODE/.git" ]; then git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"; fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E102_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E102_SHA" || { echo "E102_SHA is not on origin"; exit 2; }
git checkout -q --force --detach "$E102_SHA"
[ "$(git rev-parse HEAD)" = "$E102_SHA" ] || { echo "checkout is not E102_SHA (give the full sha)"; exit 2; }
CONFIG=$CODE/exp/jobs/e102_config/model.yaml

export DATASET_PATH=$SCRATCH/data/exp89/awf/dataset
test -f "$DATASET_PATH/config.json" && test -d "$DATASET_PATH/windows/spatial_split" \
  || { echo "no AWF rslearn dataset at $DATASET_PATH (scratch purge?): exp/jobs/e89.sh E89_MODE=inv E89_ARM=awf"; exit 1; }
N_WIN=$(ls "$DATASET_PATH/windows/spatial_split" | wc -l)
[ "$N_WIN" = 1459 ] || { echo "$N_WIN label windows, expected 1,459"; exit 1; }
export NUM_WORKERS=${E102_WORKERS:-8} PREDICTION_OUTPUT_LAYER=output_e102

if [ ! -x "$VENV/bin/rslearn" ]; then
  echo "== $(date -Is) building the venv (rslearn 0.1.17) =="
  uv venv -q --python 3.12 "$VENV"
  uv pip install -q --python "$VENV/bin/python" "rslearn[extra]==0.1.17"
fi
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
for p in ("rslearn", "olmoearth-pretrain-minimal", "olmoearth-pretrain", "torch", "lightning"):
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "not installed")
try:
    from olmoearth_pretrain.model_loader import ModelID
except ImportError:
    from olmoearth_pretrain_minimal.model_loader import ModelID   # noqa
print("ModelID:", [m.name for m in ModelID])
print("knows OLMOEARTH_V1_2_BASE:", hasattr(ModelID, "OLMOEARTH_V1_2_BASE"))
EOF

id_of() { case "$1" in v1_2) echo OLMOEARTH_V1_2_BASE ;; v1) echo OLMOEARTH_V1_BASE ;; *) echo ""; esac; }
echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp102 $MODE, commit $(git rev-parse --short HEAD) =="
case "$MODE" in
  smoke)
    for m in v1_2 v1; do
      export MODEL_ID=$(id_of $m) TRAINER_DATA_PATH=$SCRATCH/e102/smoke/$m
      rm -rf "$TRAINER_DATA_PATH"; mkdir -p "$TRAINER_DATA_PATH"
      echo "== fast_dev_run: $m ($MODEL_ID) =="
      "$VENV/bin/rslearn" model fit --config "$CONFIG" --seed_everything 0 --trainer.fast_dev_run true
    done
    export MODEL_ID=OLMOEARTH_V1_2_BASE TRAINER_DATA_PATH=$SCRATCH/e102/smoke/v1_2_epoch
    rm -rf "$TRAINER_DATA_PATH"; mkdir -p "$TRAINER_DATA_PATH"
    echo "== one epoch of v1.2, timed =="
    t0=$(date +%s)
    "$VENV/bin/rslearn" model fit --config "$CONFIG" --seed_everything 0 --trainer.max_epochs 1
    echo "one epoch (train + val): $(( $(date +%s) - t0 )) s; 100 epochs about $(( ($(date +%s) - t0) * 100 / 60 )) min"
    find "$TRAINER_DATA_PATH" -name "metrics.csv" -exec tail -3 {} \; ;;
  train)
    MID=$(id_of "${E102_MODEL:-}"); [ -n "$MID" ] || { echo "E102_MODEL must be v1_2 or v1"; exit 2; }
    [[ "${E102_SEED:-}" =~ ^[0-9]+$ ]] || { echo "E102_SEED must be an integer"; exit 2; }
    export MODEL_ID=$MID TRAINER_DATA_PATH=$RUNS/${E102_MODEL}_s${E102_SEED}
    [ ! -e "$TRAINER_DATA_PATH/last.ckpt" ] || { echo "$TRAINER_DATA_PATH already holds a run; remove it to retrain"; exit 2; }
    mkdir -p "$TRAINER_DATA_PATH"
    git rev-parse HEAD > "$TRAINER_DATA_PATH/commit.txt"
    "$VENV/bin/rslearn" model fit --config "$CONFIG" --seed_everything "$E102_SEED"
    ls -la "$TRAINER_DATA_PATH"
    find "$TRAINER_DATA_PATH" -name "metrics.csv" -exec tail -2 {} \; ;;
esac
du -sh "$SCRATCH/e102" | sed 's/^/e102 on scratch: /'
echo "== $(date -Is) done =="
