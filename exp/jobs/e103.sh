#!/bin/bash
# exp103 (docs/plan/v12_regrade.md, frozen; the page is the specification): exp98's and exp99's frozen rules applied to
# the record's v1.2 fine-tune (exp102: epoch 60, sha256 5ce88827...). `rslearn model predict` (rslearn 0.1.17, the venv
# exp102 trained in, $SCRATCH/e102/venv) writes a new output layer output_v12 (the ten-channel softmax, float32, as
# exp/jobs/e102_config/model.yaml's output_probs and writer layer_config give it) into exp98's 2023 deployment dataset
# and exp99's 2017 one on scratch; nothing else in the datasets changes. `oe-inferencex from-olmoearth --layer
# output_v12` reads them in parts as e98_read.sh and e99_read.sh did, and exp98's and exp99's own scripts grade them.
# Helpers: exp/exp103_jobs.py (counts, the smoke's two windows, the output check, the read view and plan, masking).
#
#   E103_MODE=smoke      GPU, at most 30 min. For each year: rslearn 0.1.17 parses the dataset (config.json written by
#                        rslearn 0.0.27 through olmoearth_run) and its own check_window counts the windows it will
#                        predict (precheck); `model predict` with the v1.2 checkpoint on two windows of one group,
#                        chosen by predict_config.groups + .names in a copy of model.yaml that differs in nothing else,
#                        into layer output_v12_smoke; the check (exactly those two written, 10 float32 bands, finite,
#                        in [0, 1], summing to 1 within 1e-3 at covered pixels); from-olmoearth with the condition
#                        layer on each of the two; the seconds per two windows. The smoke layer is removed on exit
#                        (E103_KEEP_SMOKE=1 keeps it).
#   E103_MODE=predict23  GPU. `model predict` over every window of the 2023 dataset with input (270), layer output_v12,
#   E103_MODE=predict17  or the 2017 one (313 windows; rslearn skips the 42 without Sentinel-2 item groups itself:
#                        check_window, rslearn/train/dataset.py:905-943; 271 with all twelve are predicted). Then
#                        e98_predict.sh's check: windows with input, with output, missing output, output without input,
#                        band set and header of every output, eight sampled windows in full. Refuses if any window
#                        already holds layers/output_v12/ (a config with output_layer_name_skip_inference_if_exists
#                        would write nothing over it), unless E103_REDO=1, which removes that layer first.
#   E103_MODE=read23     CPU. from-olmoearth --layer output_v12 --conditions --inputs sentinel2_scl on a view of the
#   E103_MODE=read17     predicted windows (symlinks to each window's metadata.json and output_v12, and to the SCL layers
#                        of exp98's or exp99's collected copy, whose metadata.json must be identical), in parts: 2023
#                        with e98_read.sh's plan (at most 3 GiB of grid per part, r<root>_p<part>; exp98's MODIS burned
#                        layer is copied beside each part only where exp98 read the same windows onto the identical
#                        grid, for Part H), 2017 one window per part as e99_read.sh (r<root>_w<k>). To
#                        $SCRATCH/e103_out/scores23/ and scores17/, with parts.json; counts to e103_out/records/.
#   E103_MODE=grade23    CPU. exp/exp98_awf_deployment.py --scores <the 2023 parts> --labels <exp89's AWF windows>
#                        (its default --replica; Part I with the TimeSync sample, Natural Earth and Ai2's geometry,
#                        checked against their pins, never fetched) --out-dir e103_out/awf2023. exp98's P1 and P2 are
#                        computed by the script against the v1 replica and are not applicable (the page).
#   E103_MODE=grade17    CPU. exp/exp99_transfer.py --scores <the 2017 parts> --timesync --boundaries
#                        --request-geometry (required, checked) --out-dir e103_out/transfer2017, Part F on the v1.2
#                        2023 parts (--exp98-scores e103_out/scores23; E103_PARTF=none skips it).
#   Both grading scripts refuse unless their own pages are frozen; each runs its --smoke first.
#
# Every mode needs E103_SHA (the full commit, pushed). It checks that commit out, detached, in its own clone on scratch,
# never the home checkout: $SCRATCH/e103_code/smoke, .../y2023 (predict23, read23, grade23) and .../y2017 (predict17,
# read17, grade17), so the two years' chains may run side by side; two jobs of one year must not overlap (chain them).
# Outputs go to $SCRATCH/e103_out only (records/, logs/, smoke/, scores23/, scores17/, awf2023/, transfer2017/, views,
# work and trainer dirs); the only other writes are the layers output_v12 (and the smoke's output_v12_smoke) in the two
# deployment datasets, the clones' .venv (uv, from $SCRATCH/uv-cache) and the Slurm log. Nothing is downloaded on
# purpose: the encoder's base weights come from $SCRATCH/hf with HF_HUB_OFFLINE=1 (E103_HF_OFFLINE=0 allows the Hub,
# as exp102's training did). No window name, group name, real root name, bound or coordinate is echoed: rslearn's own
# log stays in e103_out/logs/ and what is shown of it is masked (the 2017 windows each hold a TimeSync plot).
#
# Submit from the Mac after commit and push; the login node runs one sbatch per line and nothing else:
#   SHA=$(git rev-parse HEAD)
#   SM=$(ssh aicr "E103_MODE=smoke E103_SHA=$SHA sbatch --parsable -J e103smoke -p b200-devel,rtx-devel -q interactive --gpus=1 -c 8 --mem=64G -t 00:30:00" < exp/jobs/e103.sh)
#   P23=$(ssh aicr "E103_MODE=predict23 E103_SHA=$SHA sbatch --parsable -J e103p23 -p b200-batch --gpus=1 -c 16 --mem=128G -t 08:00:00 --dependency=afterok:$SM" < exp/jobs/e103.sh)
#   P17=$(ssh aicr "E103_MODE=predict17 E103_SHA=$SHA sbatch --parsable -J e103p17 -p b200-batch --gpus=1 -c 16 --mem=128G -t 08:00:00 --dependency=afterok:$SM" < exp/jobs/e103.sh)
#   R23=$(ssh aicr "E103_MODE=read23 E103_SHA=$SHA sbatch --parsable -J e103r23 -p cpu -c 8 --mem=64G -t 04:00:00 --dependency=afterok:$P23" < exp/jobs/e103.sh)
#   R17=$(ssh aicr "E103_MODE=read17 E103_SHA=$SHA sbatch --parsable -J e103r17 -p cpu -c 8 --mem=32G -t 03:00:00 --dependency=afterok:$P17" < exp/jobs/e103.sh)
#   G23=$(ssh aicr "E103_MODE=grade23 E103_SHA=$SHA sbatch --parsable -J e103g23 -p cpu -c 8 --mem=32G -t 02:00:00 --dependency=afterok:$R23" < exp/jobs/e103.sh)
#   G17=$(ssh aicr "E103_MODE=grade17 E103_SHA=$SHA sbatch --parsable -J e103g17 -p cpu -c 8 --mem=32G -t 03:00:00 --dependency=afterok:$R17:$R23" < exp/jobs/e103.sh)
# Order: smoke; then predict23 and predict17 (side by side, two GPUs); read23 after predict23, read17 after predict17;
# grade23 after read23; grade17 after read17 and read23 (Part F reads the 2023 parts). At most two jobs run at once.
# Limits: b200-devel and rtx-devel (QOS interactive) allow 4 h and 4 interactive jobs per user; b200-batch allows 24 h
# (`-p b200-batch` without `-q interactive`). Read the smoke's log before the chain goes on if it is not chained: its
# seconds per two windows give each predict's time; if 270 windows would pass about 3.5 h, submit predict on
# b200-batch with -t from that estimate. The two predicts may land on different GPU models (B200, RTX PRO 6000); each
# record names its GPU.
#
# Expected (estimates, not measured): smoke 10 to 20 min (venv and model load, two windows per year, two uv syncs);
# predict 0.5 to 2 h per year (exp98's olmoearth_run predicted the same 270 windows with a v1 Base encoder in under an
# hour; 7,225 crops of 16 px per 1024 px window at batch 4); read23 20 to 60 min, read17 15 to 40 min; grades 10 to 40
# min each. Volume on scratch: output_v12 about 10 to 12 GB in each deployment dataset (10 float32 bands of 1024 x 1024
# per window, LZW; exp99's collected output and SCL were 14.3 GB for 271 windows); scores23 about 10 to 20 GB (11 parts,
# about 5.2e8 grid pixels, deflate), scores17 about 8 to 12 GB (271 windows); the views are symlinks; records and grade
# outputs under 100 MB. Purge: the deployment datasets were built on 8 and 9 October 2026 and the collected copies
# (SCL) moved to scratch on 10 October; scratch is purged after 30 days, so the chain must run before about 7 November.
#
# Reruns: E103_REDO=1 (predict: remove output_v12 and predict again; a read after it needs E103_REREAD=1),
# E103_REREAD=1 (read: replace scores23/ or scores17/), E103_REGRADE=1 (grade: replace awf2023/ or transfer2017/).
# Options: E103_CONDITIONS=0 reads without the condition layer; E103_BURNED=0 skips exp98's burned layer;
# E103_PART_GIB (default 3); E103_PARTF=none or a directory of 2023 parts.
#SBATCH -A p2026_0089_neu
#SBATCH -J e103
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
# The soft limit on open files is 1,024 on the cpu nodes (hard 131,072); data-loader workers exhaust it (job 1243542).
ulimit -n 65536
MODE=${E103_MODE:-}
case "$MODE" in
  smoke) STREAM=smoke ;;
  predict23|read23|grade23) STREAM=y2023 ;;
  predict17|read17|grade17) STREAM=y2017 ;;
  *) echo "E103_MODE must be smoke, predict23, predict17, read23, read17, grade23 or grade17 (got '$MODE')"; exit 2 ;;
esac
[[ "${E103_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E103_SHA must be the full 40-character commit (got '${E103_SHA:-}')"; exit 2; }
for v in E103_REDO E103_REREAD E103_REGRADE E103_KEEP_SMOKE E103_CONDITIONS E103_BURNED E103_HF_OFFLINE; do
  case "${!v:-}" in ""|0|1) ;; *) echo "$v must be 0 or 1 (got '${!v}')"; exit 2 ;; esac
done
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
DEPLOY=$SCRATCH/deploy
OUT=$SCRATCH/e103_out
REC=$OUT/records
CODE=$SCRATCH/e103_code/$STREAM
VENV=$SCRATCH/e102/venv
CKPT=$SCRATCH/e102/runs/v1_2_s0/epoch=60-step=17019.ckpt
CKPT_SHA256=5ce8882718f8b3c5f772fd310fb49affe0aadffaf88cec6474af1442b7324816   # exp/out/exp102/checkpoint.json
LABELS=$SCRATCH/data/exp89/awf/dataset/windows
BREADTH=$REPO/data/breadth
AI2_GEOMETRY=$DEPLOY/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
EXP98_SCORES=$REPO/data/exp98/scores
JOB=${SLURM_JOB_ID:-none}
NCPU=${SLURM_CPUS_PER_TASK:-8}
case "$MODE" in smoke|predict*) THREADS=4 ;; *) THREADS=$NCPU ;; esac   # GPU modes: data-loader workers use the CPUs
export HF_HOME=$SCRATCH/hf UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS
export HF_HUB_OFFLINE=${E103_HF_OFFLINE:-1} WANDB_MODE=disabled
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$REC" "$OUT/logs"

if [ ! -d "$CODE/.git" ]; then
  mkdir -p "$(dirname "$CODE")"
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E103_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E103_SHA" \
  || { echo "E103_SHA=$E103_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E103_SHA"
[ "$(git rev-parse HEAD)" = "$E103_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E103_SHA"; exit 2; }
test -f exp/exp103_jobs.py || { echo "E103_SHA predates exp/exp103_jobs.py"; exit 2; }
grep -q '^\*\*Status: frozen' docs/plan/v12_regrade.md || { echo "docs/plan/v12_regrade.md is not frozen at E103_SHA"; exit 2; }
CONFIG=$CODE/exp/jobs/e102_config/model.yaml
HELP=$CODE/exp/exp103_jobs.py
echo "== $(date -Is) job $JOB on $(hostname): exp103 $MODE, commit $E103_SHA =="

year_paths() {   # year_paths 23|17: INV (the scratch inventory, real root names), RUN (the roots' parent), COLL
  case "$1" in
    23) INV=$DEPLOY/prepare_inventory.json; RUN=$DEPLOY/awf_run; COLL=$REPO/data/exp98/awf_run ;;
    17) INV=$DEPLOY/prepare_inventory_e99.json; RUN=$DEPLOY/awf2017_run; COLL=$REPO/data/exp99/awf2017_run ;;
  esac
  test -s "$INV" || { echo "no 20$1 inventory in $DEPLOY (the 30-day scratch purge?)"; exit 1; }
  test -d "$RUN" || { echo "no 20$1 deployment run in $DEPLOY (the 30-day scratch purge?)"; exit 1; }
  DS=(--inv "$INV" --run "$RUN" --year "$1")
}

# ----------------------------------------------------------------------------- GPU modes
gpu_prep() {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || { echo "no GPU: submit with --gpus=1 on b200-devel, rtx-devel or b200-batch"; exit 2; }
  test -x "$VENV/bin/rslearn" || { echo "no rslearn venv at $VENV: exp103 predicts in exp102's (exp/jobs/e102_train.sh builds it)"; exit 1; }
  test -s "$CKPT" || { echo "no v1.2 checkpoint at $CKPT (purged? the same file is model.ckpt of 2imi9/OlmoEarth-v1_2-FT-AWF-Base)"; exit 1; }
  echo "$CKPT_SHA256  $CKPT" | sha256sum -c --quiet - || { echo "the checkpoint's sha256 is not exp102's"; exit 1; }
  E103_COMMIT=$E103_SHA "$VENV/bin/python" - "$REC/e103_${MODE}_env.json" <<'EOF'
import importlib.metadata as md, json, os, sys
import torch
def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
v = {p: ver(p) for p in ("rslearn", "olmoearth-pretrain-minimal", "torch", "lightning", "jsonargparse")}
assert v["rslearn"] == "0.1.17", f"exp102's venv holds rslearn {v['rslearn']}, not 0.1.17"
assert torch.cuda.is_available(), "torch sees no GPU"
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
assert arch in torch.cuda.get_arch_list(), f"this torch build has no kernels for {arch}"
rec = {"job": os.environ.get("SLURM_JOB_ID"), "commit": os.environ["E103_COMMIT"], "versions": v,
       "gpu": torch.cuda.get_device_name(0), "arch": arch, "checkpoint_sha256": "5ce8882718f8b3c5f772fd310fb49affe0aadffaf88cec6474af1442b7324816",
       "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")}
json.dump(rec, open(sys.argv[1], "w"), indent=1)
print(json.dumps(rec))
EOF
}

masked() {   # masked <year>: stdin to stdout with the dataset's names and long numbers masked
  "$VENV/bin/python" "$HELP" mask --inv "$INV" --run "$RUN" --year "$1"
}

run_predict() {   # run_predict <year> <layer> <config> <root index> <tag>; sets PRED_SECONDS
  local y=$1 layer=$2 cfg=$3 i=$4 tag=$5
  export MODEL_ID=OLMOEARTH_V1_2_BASE PREDICTION_OUTPUT_LAYER=$layer NUM_WORKERS=$NCPU
  export TRAINER_DATA_PATH=$OUT/trainer_data/$tag-$JOB    # model.yaml's CSVLogger and ModelCheckpoint; predict writes none
  # rslearn's ModelDataset writes every window's group, name, bounds and projection to tempfile.gettempdir()
  # (rslearn/train/dataset.py:1007-1019 at v0.1.17) and never deletes them: keep them on scratch, not the node's /tmp
  export TMPDIR=$OUT/tmp/$tag-$JOB
  mkdir -p "$TMPDIR"
  DATASET_PATH=$("$VENV/bin/python" "$HELP" root-path "${DS[@]}" --index "$i")
  export DATASET_PATH
  local work=$OUT/work/$tag-$JOB log=$OUT/logs/$tag-$JOB.log t0
  mkdir -p "$TRAINER_DATA_PATH" "$work"      # rslearn saves the resolved config.yaml in the working directory
  echo "== $(date -Is) rslearn model predict: 20$y, root $i, layer $layer (log e103_out/logs/$tag-$JOB.log) =="
  t0=$(date +%s)
  if ! (cd "$work" && "$VENV/bin/rslearn" model predict --config "$cfg" --ckpt_path "$CKPT") > "$log" 2>&1; then
    echo "rslearn model predict failed; the last lines of its log, masked:"
    tail -n 60 "$log" | masked "$y"
    exit 1
  fi
  PRED_SECONDS=$(( $(date +%s) - t0 ))
  echo "$(date -Is) predicted in $PRED_SECONDS s; rslearn's own counts, masked:"
  { grep -E "examples in split predict|Skipped [0-9]+ windows|Loaded [0-9]+ windows|Writing [0-9]+ dataset examples" "$log" \
      || echo "(no count line found)"; } | masked "$y" | sed -E 's/^.*(logger=)/\1/; s/^/  /'
}

smoke_cleanup() {   # the EXIT trap of the smoke: removes the smoke layer and keeps the job's exit status
  local rc=$?
  if [ "${E103_KEEP_SMOKE:-0}" = 1 ]; then
    echo "E103_KEEP_SMOKE=1: layers/output_v12_smoke/ left in the smoke's windows"
  else
    for y in 23 17; do
      year_paths "$y"
      "$VENV/bin/python" "$HELP" remove-layer "${DS[@]}" --layer output_v12_smoke || true
    done
  fi
  exit "$rc"
}

case "$MODE" in
  smoke)
    gpu_prep
    SM=$OUT/smoke
    rm -rf "$SM"
    mkdir -p "$SM"
    trap smoke_cleanup EXIT
    SECS=()
    for y in 23 17; do
      year_paths "$y"
      echo "== 20$y: rslearn 0.1.17 on the dataset (config.json, windows, check_window) =="
      "$VENV/bin/python" "$HELP" precheck "${DS[@]}" --config "$CONFIG" --workers "$NCPU" --out "$REC/e103_precheck$y.json"
      "$VENV/bin/python" "$HELP" remove-layer "${DS[@]}" --layer output_v12_smoke
      "$VENV/bin/python" "$HELP" smoke-config "${DS[@]}" --config "$CONFIG" --out "$SM/model_smoke$y.yaml" --ids-out "$SM/ids$y.json"
      run_predict "$y" output_v12_smoke "$SM/model_smoke$y.yaml" 0 "smoke$y"
      SECS+=("$PRED_SECONDS")
      echo "== 20$y: the two windows' output, checked =="
      "$VENV/bin/python" "$HELP" check "${DS[@]}" --layer output_v12_smoke --ids "$SM/ids$y.json" --out "$REC/e103_smoke_check$y.json"
      echo "== 20$y: from-olmoearth on the two windows, with the condition layer =="
      uv run --extra geo python "$HELP" read "${DS[@]}" --layer output_v12_smoke --collected "$COLL" --conditions \
        --view "$SM/view$y" --scores "$SM/scores$y" --plan window --ids "$SM/ids$y.json" --record "$REC/e103_smoke_read$y.json"
      rm -rf "$SM/view$y"               # symlinks into the layer the cleanup removes
    done
    E103_COMMIT=$E103_SHA "$VENV/bin/python" - "$REC" "${SECS[@]}" <<'EOF'
import json, os, sys
rec_dir, s23, s17 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = {"job": os.environ.get("SLURM_JOB_ID"), "commit": os.environ["E103_COMMIT"], "ok": True,
       "seconds_two_windows": {"2023": s23, "2017": s17},
       "predict_hours_upper_estimate": {"2023": round(s23 / 2 * 270 / 3600, 2), "2017": round(s17 / 2 * 271 / 3600, 2)},
       "note": "two windows' wall time includes loading the model and the dataset, so the estimate is an upper bound"}
for y in ("23", "17"):
    for k in ("precheck", "smoke_check", "smoke_read"):
        r = json.load(open(os.path.join(rec_dir, f"e103_{k}{y}.json")))
        out["ok"] &= bool(r.get("ok", True))
json.dump(out, open(os.path.join(rec_dir, "e103_smoke.json"), "w"), indent=1)
print(json.dumps(out))
assert out["ok"], "a smoke record is not ok"
EOF
    echo "== $(date -Is) smoke done; records in e103_out/records/ (e103_smoke*.json, e103_precheck*.json) ==" ;;

  predict23|predict17)
    y=${MODE#predict}
    gpu_prep
    year_paths "$y"
    test -s "$REC/e103_smoke.json" || { echo "no smoke record: run E103_MODE=smoke first"; exit 1; }
    "$VENV/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["ok"], "the smoke is not ok"
if r["commit"] != sys.argv[2]: print("note: the smoke ran at another commit, " + r["commit"])' "$REC/e103_smoke.json" "$E103_SHA"
    rc=0
    "$VENV/bin/python" "$HELP" count "${DS[@]}" --layer output_v12 --refuse-layer || rc=$?
    case "$rc" in
      0) ;;
      3) [ "${E103_REDO:-0}" = 1 ] || exit 2
         "$VENV/bin/python" "$HELP" remove-layer "${DS[@]}" --layer output_v12 ;;
      *) exit 1 ;;
    esac
    "$VENV/bin/python" "$HELP" precheck "${DS[@]}" --config "$CONFIG" --workers "$NCPU" --out "$REC/e103_precheck$y.json"
    NROOTS=$("$VENV/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["roots"]))' "$INV")
    for ((i = 0; i < NROOTS; i++)); do
      run_predict "$y" output_v12 "$CONFIG" "$i" "predict$y"
    done
    echo "== the output layers, checked =="
    "$VENV/bin/python" "$HELP" check "${DS[@]}" --layer output_v12 --out "$REC/e103_predict${y}_check.json"
    echo "== $(date -Is) done: e103_out/records/e103_predict${y}_check.json ==" ;;

  # --------------------------------------------------------------------------- CPU modes
  read23|read17)
    y=${MODE#read}
    year_paths "$y"
    test -s "$REC/e103_predict${y}_check.json" || { echo "no predict check: run E103_MODE=predict$y first"; exit 1; }
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["ok"], "the predict check is not ok"' \
      "$REC/e103_predict${y}_check.json"
    SC=$OUT/scores$y
    if [ -e "$SC" ]; then
      if [ "${E103_REREAD:-0}" = 1 ]; then rm -rf "$SC"; else echo "e103_out/scores$y exists; E103_REREAD=1 replaces it"; exit 2; fi
    fi
    ARGS=("${DS[@]}" --layer output_v12 --view "$OUT/view$y" --scores "$SC" --record "$REC/e103_read$y.json")
    if [ "${E103_CONDITIONS:-1}" = 1 ]; then
      test -d "$COLL" || { echo "no collected copy for 20$y (the SCL layers) at $COLL; E103_CONDITIONS=0 reads without them"; exit 1; }
      ARGS+=(--collected "$COLL" --conditions)
    fi
    if [ "$y" = 23 ]; then
      ARGS+=(--plan grid --part-gib "${E103_PART_GIB:-3}")
      if [ "${E103_BURNED:-1}" = 1 ] && [ -f "$EXP98_SCORES/parts.json" ]; then ARGS+=(--burned-from "$EXP98_SCORES"); fi
    else
      ARGS+=(--plan window)
    fi
    uv run --extra geo oe-inferencex from-olmoearth --help > /dev/null \
      || { echo "oe-inferencex from-olmoearth does not run in this checkout"; exit 1; }
    uv run --extra geo python "$HELP" read "${ARGS[@]}"
    echo "== $(date -Is) done: $(find "$SC" -mindepth 1 -maxdepth 1 -type d -name 'r*_*' | wc -l) part directories in e103_out/scores$y, records/e103_read$y.json ==" ;;

  grade23|grade17)
    y=${MODE#grade}
    SC=$OUT/scores$y
    test -s "$REC/e103_read$y.json" && test -s "$SC/parts.json" || { echo "no read of 20$y: run E103_MODE=read$y first"; exit 1; }
    SCORE_DIRS=()
    for d in "$SC"/r*_*/; do
      compgen -G "$d/scores_*.tif" > /dev/null && SCORE_DIRS+=("${d%/}")
    done
    NPARTS=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$SC/parts.json")
    [ "$NPARTS" = "${#SCORE_DIRS[@]}" ] || { echo "parts.json lists $NPARTS parts, scores$y holds ${#SCORE_DIRS[@]}"; exit 1; }
    if [ "$y" = 23 ]; then GO=$OUT/awf2023; else GO=$OUT/transfer2017; fi
    if [ -e "$GO" ]; then
      if [ "${E103_REGRADE:-0}" = 1 ]; then rm -rf "$GO"; else echo "$(basename "$GO") exists; E103_REGRADE=1 replaces it"; exit 2; fi
    fi
    rm -rf "$GO.partial"
    echo "== the TimeSync sample, Natural Earth's countries and Ai2's geometry, checked against their pins (no fetch) =="
    BREADTH_OK=1
    uv run --extra geo python "$HELP" verify-breadth --dir "$BREADTH" --request-geometry "$AI2_GEOMETRY" || BREADTH_OK=0
    TS_ARGS=(--timesync "$BREADTH/eastafrica_yearly_point_data.csv" --boundaries "$BREADTH/ne_10m_admin_0_countries.geojson"
             --request-geometry "$AI2_GEOMETRY")
    if [ "$y" = 23 ]; then
      test -d "$LABELS/spatial_split" || { echo "no AWF label windows at $LABELS (purged? exp/jobs/e89.sh E89_ARM=awf)"; exit 1; }
      N_LAB=$(find "$LABELS/spatial_split" -mindepth 1 -maxdepth 1 -type d | wc -l)
      [ "$N_LAB" -eq 1459 ] || { echo "$N_LAB AWF label windows at $LABELS, expected 1,459 (purged? exp/jobs/e89.sh E89_ARM=awf)"; exit 1; }
      if [ "$BREADTH_OK" = 0 ]; then
        echo "Part I's inputs are not available or not as pinned (above); Part I is skipped with a note, nothing else changes"
        TS_ARGS=()
      fi
      PY=(uv run --extra encoder --extra geo python exp/exp98_awf_deployment.py)
      echo "== smoke =="
      "${PY[@]}" --smoke --out-dir "$OUT/grade_smoke/awf2023"
      echo "== exp98's grading of the v1.2 2023 map =="
      # Part G (report-only, model-independent) reads the 2023 deployment dataset's inputs, as exp98 did
      year_paths 23
      DEP_ARGS=(--deployment-dataset "$("$VENV/bin/python" "$HELP" root-path "${DS[@]}")")
      "${PY[@]}" --scores "${SCORE_DIRS[@]}" --labels "$LABELS" "${DEP_ARGS[@]}" ${TS_ARGS[@]+"${TS_ARGS[@]}"} \
        --out-dir "$GO.partial"
    else
      [ "$BREADTH_OK" = 1 ] || { echo "exp99's grading needs the pinned TimeSync sample, boundaries and geometry (above)"; exit 1; }
      PARTF=${E103_PARTF:-$OUT/scores23}
      F_ARGS=()
      if [ "$PARTF" != none ]; then
        for d in "$PARTF"/r*_*/; do
          compgen -G "$d/scores_*.tif" > /dev/null && F_ARGS+=("${d%/}")
        done
        if [ ${#F_ARGS[@]} -gt 0 ]; then F_ARGS=(--exp98-scores "${F_ARGS[@]}"); else echo "no 2023 parts in $PARTF: Part F reads the 2017 map only"; fi
      fi
      PY=(uv run --extra geo python exp/exp99_transfer.py)
      echo "== smoke =="
      "${PY[@]}" --smoke --out-dir "$OUT/grade_smoke/transfer2017"
      echo "== exp99's grading of the v1.2 2017 map (Part F on $([ ${#F_ARGS[@]} -gt 0 ] && echo "the v1.2 2023 map" || echo "the 2017 map only")) =="
      "${PY[@]}" --scores "${SCORE_DIRS[@]}" "${TS_ARGS[@]}" ${F_ARGS[@]+"${F_ARGS[@]}"} --out-dir "$GO.partial"
    fi
    mv "$GO.partial" "$GO"
    ls -la "$GO"
    echo "== $(date -Is) done: e103_out/$(basename "$GO")/ (copy to exp/out/exp103/ when recording) ==" ;;
esac
du -sh "$OUT" 2>/dev/null | sed 's/^/e103_out on scratch: /' || true
