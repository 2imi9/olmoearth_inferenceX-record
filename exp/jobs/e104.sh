#!/bin/bash
# exp104 (docs/plan/forest_loss_2026.md; the page is the specification): Ai2's Forest Loss Driver model
# (allenai/OlmoEarth-v1-FT-ForestLossDriver-Base @ 15502f8a, sha256 682c20b9...) graded at a stratified sample of 2026
# MapBiomas Alerta alerts in the Amazon. exp/exp104_forest_loss.py does the frame, the windows, the counts and the grade;
# rslearn 0.1.17 (exp102's venv, $SCRATCH/e102/venv) prepares, materializes and predicts; `oe-inferencex
# from-olmoearth` reads the per-window probabilities. Configs: exp/jobs/e104_config/ (Ai2's FLD dataset.json with the
# Planetary Computer source, Ai2's own c698da3c version byte for byte; Ai2's model.yaml with a CSVLogger, nothing else).
#
#   E104_MODE=frame    CPU, minutes. The synthetic smoke, then the frame and the sample from the attribute snapshot at
#                      $PRIV/alerts2026_attrs.json (E104_FETCH=1 fetches the attributes from MapBiomas's WFS instead;
#                      refused unless the frame hashes to the page's 12bf6fd243b27a9a). Codes go to $PRIV only.
#   E104_MODE=windows  CPU, minutes; downloads the 696 sampled alerts' polygons (14 WFS requests, a few MB) into $PRIV.
#                      Builds $PRIV/dataset (one 128-px EPSG:3857 window per alert) and the GPU smoke's two synthetic
#                      windows ($OUT/smoke_ds); checks the configs against Ai2's pins and lets rslearn 0.1.17 parse both
#                      datasets (projection, size, time range, each layer's request range, the centres).
#   E104_MODE=recheck  CPU, minutes. The windows step's checks again on the dataset it built, nothing rebuilt or
#                      fetched (after a change to the checks; prepare gates on the dataset check's record).
#   E104_MODE=prepare  CPU, hours; downloads Sentinel-2 L2A from Microsoft Planetary Computer (no key), about 2 GB.
#                      rslearn dataset prepare, ingest (a no-op: ingest false) and materialize on both datasets, each
#                      run twice (the second pass retries windows whose requests failed); then the windows with all
#                      eight item groups, by stratum (counts).
#   E104_MODE=smoke    GPU, at most 45 min; allowed before freezing (synthetic windows only). The FLD checkpoint from
#                      Hugging Face into the scratch HF cache (about 0.4 GB; sha256 checked), `rslearn model predict`
#                      on the two synthetic windows, from-olmoearth's read, the check (ten classes in Ai2's order).
#   E104_MODE=predict  GPU (b200-batch). Refused unless the page is frozen at E104_SHA and the smoke passed.
#                      `rslearn model predict` over $PRIV/dataset into layer "output" (Ai2's vector layer, new in this
#                      dataset; refused if any window holds it, unless E104_REDO=1, which removes it first), then the
#                      counts: windows with output against windows with all eight item groups.
#   E104_MODE=read     CPU, minutes. Refused unless frozen. from-olmoearth --prob-property probs --class-property
#                      new_label on $PRIV/dataset into $PRIV/probs (window_probs.npy and .csv; the csv holds window
#                      centres, so it stays private).
#   E104_MODE=grade    CPU, minutes. Refused unless frozen. The smoke, then exp104 --grade into $OUT/grade/
#                      (exp104_summary.json, exp104_units.npz: no code, no position; copy to exp/out/ when recording).
#
# Every mode needs E104_SHA (the full commit, pushed): its own clone on scratch, $SCRATCH/e104_code, checked out detached
# at that commit; never the home checkout. Writes: $PRIV=$SCRATCH/e104_private (mode 700: the snapshot, the sample with
# alert codes, the polygons, the rslearn dataset with its imagery and predictions, rslearn's logs and temp files, the
# probabilities), $OUT=$SCRATCH/e104_out (records/ of counts, logs/ of this job's own steps, smoke_ds/, smoke/, grade/),
# the clone's .venv (uv), $SCRATCH/hf (the checkpoint) and the Slurm log. Nothing echoed holds an alert code, a window
# bound, a coordinate or a Sentinel-2 tile or item name: rslearn's logs stay in $PRIV/logs and what is shown is masked.
#
# Before the first job (the owner, from the Mac; the snapshot is not in git, it holds bounding boxes):
#   chmod 600 <scratchpad>/mapbiomas/alerts2026_attrs.json
#   scp -p <scratchpad>/mapbiomas/alerts2026_attrs.json aicr:/scratch/qi_zim_neu/olmoearth_inferenceX/e104_alerts2026_attrs.json
# (a file transfer only: the login node runs no command; the frame job moves it into $PRIV, mode 700)
# Submit after commit and push; the login node runs one sbatch per line and nothing else:
#   SHA=$(git rev-parse HEAD)
#   F=$(ssh aicr "E104_MODE=frame E104_SHA=$SHA sbatch --parsable -J e104frame -p cpu -c 2 --mem=8G -t 00:20:00" < exp/jobs/e104.sh)
#   W=$(ssh aicr "E104_MODE=windows E104_SHA=$SHA sbatch --parsable -J e104win -p cpu -c 4 --mem=16G -t 01:00:00 --dependency=afterok:$F" < exp/jobs/e104.sh)
#   P=$(ssh aicr "E104_MODE=prepare E104_SHA=$SHA sbatch --parsable -J e104prep -p cpu -c 32 --mem=64G -t 10:00:00 --dependency=afterok:$W" < exp/jobs/e104.sh)
#   S=$(ssh aicr "E104_MODE=smoke E104_SHA=$SHA sbatch --parsable -J e104smoke -p b200-devel,rtx-devel -q interactive --gpus=1 -c 8 --mem=64G -t 00:45:00 --dependency=afterok:$P" < exp/jobs/e104.sh)
# After the page is frozen (a new commit; SHA is that commit; the frame, windows and prepare above need not rerun):
#   PR=$(ssh aicr "E104_MODE=predict E104_SHA=$SHA sbatch --parsable -J e104pred -p b200-batch --gpus=1 -c 8 --mem=64G -t 02:00:00" < exp/jobs/e104.sh)
#   R=$(ssh aicr "E104_MODE=read E104_SHA=$SHA sbatch --parsable -J e104read -p cpu -c 2 --mem=8G -t 00:30:00 --dependency=afterok:$PR" < exp/jobs/e104.sh)
#   G=$(ssh aicr "E104_MODE=grade E104_SHA=$SHA sbatch --parsable -J e104grade -p cpu -c 4 --mem=16G -t 01:00:00 --dependency=afterok:$R" < exp/jobs/e104.sh)
# Limits: the cpu partition caps each user's CPUs (QOSMaxCpuPerUserLimit; exp99's prepare used 48 and blocked our other
# cpu jobs), so prepare asks 32; b200-devel and rtx-devel (QOS interactive) allow 4 h; b200-batch 24 h. At most one
# chain of these runs at a time; no step polls anything.
#
# Expected (estimates, not measured): frame under 1 min; windows 2 to 10 min (14 WFS requests); prepare 1 to 4 h (1,392
# STAC searches, then 8 scenes x 12 bands of 128 x 128 px and 8 product XMLs per window; Planetary Computer throttles);
# smoke 10 to 20 min; predict 10 to 40 min (about 690 windows, 8 timesteps of 64 x 64 px, batch 8); read and grade a
# few minutes each. Volume: $PRIV/dataset about 2 to 3 GB (696 windows of 8 x 12 uint16 bands of 128 x 128 px), the
# checkpoint 0.38 GB, everything else under 100 MB. Network: MapBiomas's GeoServer (attributes and the sampled
# polygons), Planetary Computer's STAC and blob storage, the Hugging Face Hub (one file); no account and no key. Purge:
# scratch is purged after 30 days, so the chain must run within 30 days of the prepare.
#
# Reruns: E104_REDO=1 (windows: refetch the polygons and rebuild the dataset; predict: remove layer output first),
# E104_REREAD=1 (read: replace $PRIV/probs), E104_REGRADE=1 (grade: replace $OUT/grade). Options: E104_FETCH=1 (frame),
# E104_WORKERS (the data loader's workers at predict, default 8; recorded, since Ai2's random Flip at predict draws per
# worker), E104_HF_OFFLINE=0 (predict and smoke may reach the Hub for the encoder's base weights if not cached).
#SBATCH -A p2026_0089_neu
#SBATCH -J e104
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
umask 077
# The soft limit on open files is 1,024 on the cpu nodes (hard 131,072); worker pools exhaust it (job 1243542).
ulimit -n 65536
MODE=${E104_MODE:-}
case "$MODE" in frame|windows|recheck|prepare|smoke|predict|read|grade) ;;
  *) echo "E104_MODE must be frame, windows, recheck, prepare, smoke, predict, read or grade (got '$MODE')"; exit 2 ;; esac
[[ "${E104_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E104_SHA must be the full 40-character commit (got '${E104_SHA:-}')"; exit 2; }
for v in E104_REDO E104_REREAD E104_REGRADE E104_FETCH E104_HF_OFFLINE; do
  case "${!v:-}" in ""|0|1) ;; *) echo "$v must be 0 or 1 (got '${!v}')"; exit 2 ;; esac
done
WORKERS=${E104_WORKERS:-8}
[[ "$WORKERS" =~ ^[0-9]+$ ]] || { echo "E104_WORKERS must be an integer"; exit 2; }
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e104_code
PRIV=$SCRATCH/e104_private
OUT=$SCRATCH/e104_out
REC=$OUT/records
VENV=$SCRATCH/e102/venv
DS=$PRIV/dataset
SMOKE_DS=$OUT/smoke_ds
JOB=${SLURM_JOB_ID:-none}
NCPU=${SLURM_CPUS_PER_TASK:-4}
CKPT_REPO=allenai/OlmoEarth-v1-FT-ForestLossDriver-Base
CKPT_REV=15502f8acb4caed6e3a7d777b0fb569c1f7eb791
CKPT_SHA256=682c20b99f44a81769ad7d3ce1b64d933fae3810bcf5978f9999cfebf1c11855
export HF_HOME=$SCRATCH/hf UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$PRIV/logs" "$PRIV/tmp"     # umask 077: owner only
chmod 700 "$PRIV"
mkdir -p "$REC" "$OUT/logs"

if [ ! -d "$CODE/.git" ]; then
  mkdir -p "$(dirname "$CODE")"
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E104_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E104_SHA" \
  || { echo "E104_SHA=$E104_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E104_SHA"
[ "$(git rev-parse HEAD)" = "$E104_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E104_SHA"; exit 2; }
test -f exp/exp104_forest_loss.py && test -f docs/plan/forest_loss_2026.md || { echo "E104_SHA predates exp104"; exit 2; }
FROZEN=0
grep -q '^\*\*Status: frozen' docs/plan/forest_loss_2026.md && FROZEN=1
PY=(uv run python exp/exp104_forest_loss.py)
VPY=("$VENV/bin/python" exp/exp104_forest_loss.py)
CONFIG=$CODE/exp/jobs/e104_config/model.yaml
echo "== $(date -Is) job $JOB on $(hostname): exp104 $MODE, commit $E104_SHA, page $([ $FROZEN = 1 ] && echo frozen || echo draft) =="

# rslearn's output, masked before anything of it is shown: dates and numbers of 4+ digits and decimals, Sentinel-2 item
# and tile names, the private directory
masked() { sed -E "s#$PRIV#<private>#g; s/S2[ABC]_MSIL2A_[A-Za-z0-9_.]+/<item>/g; s/\bT[0-9]{2}[A-Z]{3}\b/<tile>/g; \
  s/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g"; }

need_venv() {
  test -x "$VENV/bin/rslearn" || { echo "no rslearn venv at $VENV (exp/jobs/e102_train.sh builds it; purged?)"; exit 1; }
  "$VENV/bin/python" -c 'import importlib.metadata as m, sys; v = m.version("rslearn"); print("rslearn", v); sys.exit(v != "0.1.17")' \
    || { echo "exp102's venv does not hold rslearn 0.1.17"; exit 1; }
}

need_frozen() {
  [ "$FROZEN" = 1 ] || { echo "refused: docs/plan/forest_loss_2026.md is not frozen at E104_SHA; no prediction at a sampled alert before it is"; exit 2; }
}

rec_ok() {   # rec_ok <record>: the record exists and says "ok": true (a record is written even when its check fails)
  test -s "$1" && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") is True else 1)' "$1"
}

run_logged() {   # run_logged <tag> <command...>: the command's output to $PRIV/logs/<tag>-$JOB.log, masked tail on failure
  local tag=$1; shift
  local log=$PRIV/logs/$tag-$JOB.log t0
  t0=$(date +%s)
  if ! "$@" > "$log" 2>&1; then
    echo "$tag failed after $(( $(date +%s) - t0 )) s; the last lines of its log, masked:"
    tail -n 40 "$log" | masked
    exit 1
  fi
  echo "$tag: $(( $(date +%s) - t0 )) s"
}

prepare_ds() {   # prepare_ds <dataset> <tag>
  local ds=$1 tag=$2 pass
  export TMPDIR=$PRIV/tmp/$tag-$JOB
  mkdir -p "$TMPDIR"
  for pass in 1 2; do
    # --ignore-errors: a window whose Planetary Computer request still fails after the retries is left for the second
    # pass (and, failing again, counted by --imagery against the floor) instead of ending the job
    run_logged "$tag-prepare$pass" "$VENV/bin/rslearn" dataset prepare --root "$ds" --workers "$NCPU" \
      --retry-max-attempts 5 --retry-backoff-seconds 10 --ignore-errors
    run_logged "$tag-ingest$pass" "$VENV/bin/rslearn" dataset ingest --root "$ds" --workers "$NCPU"
    run_logged "$tag-materialize$pass" "$VENV/bin/rslearn" dataset materialize --root "$ds" --workers "$NCPU" \
      --retry-max-attempts 5 --retry-backoff-seconds 10 --ignore-errors
    { grep -E "Layer (pre|post)_sentinel2: prepared=|Total windows requested" "$PRIV/logs/$tag-prepare$pass-$JOB.log" \
        || echo "(no prepare summary line found)"; } | sed -E 's/^.*(Layer|Total)/\1/' | masked | sed 's/^/  /'
  done
}

checkpoint() {   # sets CKPT: the FLD checkpoint in the scratch HF cache, at the pinned revision, sha256 checked
  CKPT=$(HF_HUB_OFFLINE=0 "$VENV/bin/python" -c "from huggingface_hub import hf_hub_download as d; print(d('$CKPT_REPO', 'model.ckpt', revision='$CKPT_REV'))")
  echo "$CKPT_SHA256  $CKPT" | sha256sum -c --quiet - || { echo "the FLD checkpoint's sha256 is not the pinned one"; exit 1; }
  echo "checkpoint: $CKPT_REPO @ ${CKPT_REV:0:8}, sha256 ${CKPT_SHA256:0:8}... verified"
}

gpu_prep() {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || { echo "no GPU: submit with --gpus=1"; exit 2; }
  need_venv
  "$VENV/bin/python" -c 'import torch, sys; assert torch.cuda.is_available(), "torch sees no GPU"; c = torch.cuda.get_device_capability(0); a = f"sm_{c[0]}{c[1]}"; assert a in torch.cuda.get_arch_list(), f"no kernels for {a}"; print(torch.cuda.get_device_name(0), a)'
  "${PY[@]}" --check-configs --record "$REC/e104_configs.json"
  checkpoint
  export HF_HUB_OFFLINE=${E104_HF_OFFLINE:-1}
}

run_predict() {   # run_predict <dataset> <tag>: rslearn model predict, layer output, Ai2's config with the CSV logger
  local ds=$1 tag=$2 log work t0
  export DATASET_PATH=$ds PREDICTION_OUTPUT_LAYER=output NUM_WORKERS=$WORKERS EXTRA_FILES_PATH=""
  export TRAINER_DATA_PATH=$OUT/trainer_data/$tag-$JOB
  # rslearn's ModelDataset writes every window's name, bounds and projection to tempfile.gettempdir()
  # (rslearn/train/dataset.py:1007-1019 at v0.1.17): keep them in the private directory
  export TMPDIR=$PRIV/tmp/$tag-$JOB
  work=$PRIV/tmp/work-$tag-$JOB
  mkdir -p "$TRAINER_DATA_PATH" "$TMPDIR" "$work"
  log=$PRIV/logs/$tag-$JOB.log
  echo "== $(date -Is) rslearn model predict ($tag): layer output, --seed_everything 104, $WORKERS workers =="
  t0=$(date +%s)
  if ! (cd "$work" && "$VENV/bin/rslearn" model predict --config "$CONFIG" --ckpt_path "$CKPT" --seed_everything 104) \
      > "$log" 2>&1; then
    echo "rslearn model predict failed; the last lines of its log, masked:"
    tail -n 60 "$log" | masked
    exit 1
  fi
  echo "predicted in $(( $(date +%s) - t0 )) s; rslearn's own counts, masked:"
  { grep -E "examples in split predict|Skipped [0-9]+ windows|Loaded [0-9]+ windows|missing_data_input" "$log" \
      || echo "(no count line found)"; } | masked | sed 's/^/  /'
}

check_datasets() {   # the dataset checks of the windows step; exits 1 unless the sampled alerts' dataset passes
  echo "== rslearn 0.1.17 reads the dataset (config, windows, request ranges, centres: a second route, rslearn, the repaired polygons) =="
  "${VPY[@]}" --check-dataset "$DS" --private-dir "$PRIV" --workers "$NCPU" --record "$REC/e104_dataset_check.json" \
    | python3 -c 'import json, sys; r = json.loads(sys.stdin.read().strip().splitlines()[-1]); print(json.dumps({k: r.get(k) for k in ("rslearn", "windows", "bad", "invalid_polygons", "repaired_centre_off_window_centre", "ai2_code_letter_off_window_centre", "centre_misses", "ok")})); sys.exit(0 if r["ok"] else 1)'
  if [ ! -d "$SMOKE_DS/windows" ]; then "${PY[@]}" --smoke-dataset "$SMOKE_DS"; fi
  "${VPY[@]}" --check-dataset "$SMOKE_DS" --record "$REC/e104_smoke_dataset_check.json" > /dev/null
}

case "$MODE" in
  frame)
    "${PY[@]}" --smoke --out-dir "$OUT/smoke"
    ARGS=(--frame --private-dir "$PRIV" --record "$REC/e104_frame.json")
    if [ "${E104_FETCH:-0}" = 1 ]; then ARGS+=(--fetch); else
      if [ ! -s "$PRIV/alerts2026_attrs.json" ] && [ -s "$SCRATCH/e104_alerts2026_attrs.json" ]; then
        mv "$SCRATCH/e104_alerts2026_attrs.json" "$PRIV/alerts2026_attrs.json" && chmod 600 "$PRIV/alerts2026_attrs.json"
      fi
      test -s "$PRIV/alerts2026_attrs.json" || { echo "no snapshot at <private>/alerts2026_attrs.json (copy it, header) or E104_FETCH=1"; exit 1; }
      ARGS+=(--snapshot "$PRIV/alerts2026_attrs.json"); fi
    "${PY[@]}" "${ARGS[@]}"
    echo "== $(date -Is) done: records/e104_frame.json (counts); the sample is in the private directory ==" ;;

  windows)
    test -s "$PRIV/e104_sample.json" || { echo "no sample: run E104_MODE=frame first"; exit 1; }
    need_venv
    "${PY[@]}" --check-configs --record "$REC/e104_configs.json"
    W_ARGS=(--windows --private-dir "$PRIV" --record "$REC/e104_windows.json")
    [ "${E104_REDO:-0}" = 1 ] && W_ARGS+=(--redo)
    "${PY[@]}" "${W_ARGS[@]}" 2>&1 | masked
    check_datasets
    echo "== $(date -Is) done: records/e104_windows.json, e104_dataset_check.json ==" ;;

  recheck)
    test -d "$DS/windows" && test -s "$PRIV/e104_polygons.json" || { echo "no dataset or polygons: run E104_MODE=windows first"; exit 1; }
    need_venv
    "${PY[@]}" --check-configs --record "$REC/e104_configs.json"
    check_datasets
    echo "== $(date -Is) done: records/e104_dataset_check.json ==" ;;

  prepare)
    rec_ok "$REC/e104_dataset_check.json" || { echo "no passing dataset check: run E104_MODE=windows first"; exit 1; }
    need_venv
    echo "== the GPU smoke's two synthetic windows =="
    prepare_ds "$SMOKE_DS" smoke_ds
    echo "== the sampled alerts' windows =="
    prepare_ds "$DS" dataset
    "${PY[@]}" --imagery --private-dir "$PRIV" --record "$REC/e104_imagery.json"
    du -sh "$DS" | sed "s#$PRIV#<private>#" ;;

  smoke)
    gpu_prep
    SM=$OUT/smoke_ds
    [ -d "$SM/windows" ] || { echo "no smoke dataset: run E104_MODE=windows and prepare first"; exit 1; }
    find "$SM/windows" -mindepth 4 -maxdepth 4 -type d -name output -path '*/layers/*' -exec rm -rf {} +
    run_predict "$SM" smoke
    rm -rf "$OUT/smoke_probs"
    uv run oe-inferencex from-olmoearth "$SM" --out "$OUT/smoke_probs" --layer output --prob-property probs \
      --class-property new_label > "$OUT/logs/smoke_read-$JOB.log" 2>&1 || { tail -n 20 "$OUT/logs/smoke_read-$JOB.log"; exit 1; }
    "${PY[@]}" --check-probs "$OUT/smoke_probs" --record "$REC/e104_smoke.json"
    echo "== $(date -Is) smoke done: records/e104_smoke.json ==" ;;

  predict)
    need_frozen
    rec_ok "$REC/e104_smoke.json" || { echo "no passing smoke record: run E104_MODE=smoke first"; exit 1; }
    test -s "$REC/e104_imagery.json" || { echo "no imagery counts: run E104_MODE=prepare first"; exit 1; }
    gpu_prep
    N_OUT=$(find "$DS/windows" -mindepth 4 -maxdepth 4 -type d -name output -path '*/layers/*' | wc -l)
    if [ "$N_OUT" -gt 0 ]; then
      [ "${E104_REDO:-0}" = 1 ] || { echo "$N_OUT window(s) already hold layers/output/; E104_REDO=1 removes them first"; exit 2; }
      find "$DS/windows" -mindepth 4 -maxdepth 4 -type d -name output -path '*/layers/*' -exec rm -rf {} +
    fi
    run_predict "$DS" predict
    "${PY[@]}" --imagery --private-dir "$PRIV" --record "$REC/e104_predict_counts.json"
    python3 - "$REC/e104_predict_counts.json" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["total"]
print(json.dumps(t))
ok = t["with_output"] == t["all_eight"] and t["output_without_all_eight"] == 0
print("every window with all eight item groups has an output, and no other" if ok else "OUTPUT COUNTS DIFFER")
sys.exit(0 if ok else 1)
EOF
    echo "== $(date -Is) done: records/e104_predict_counts.json ==" ;;

  read)
    need_frozen
    test -s "$REC/e104_predict_counts.json" || { echo "no predict counts: run E104_MODE=predict first"; exit 1; }
    if [ -e "$PRIV/probs" ]; then
      if [ "${E104_REREAD:-0}" = 1 ]; then rm -rf "$PRIV/probs"; else echo "<private>/probs exists; E104_REREAD=1 replaces it"; exit 2; fi
    fi
    rm -rf "$PRIV/probs.partial"
    uv run oe-inferencex from-olmoearth "$DS" --out "$PRIV/probs.partial" --layer output --prob-property probs \
      --class-property new_label > "$PRIV/logs/read-$JOB.log" 2>&1 \
      || { echo "from-olmoearth failed; its last lines, masked:"; tail -n 20 "$PRIV/logs/read-$JOB.log" | masked; exit 1; }
    python3 - "$PRIV/probs.partial/olmoearth_output.json" "$REC/e104_read.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
r = {k: s[k] for k in ("kind", "class_property", "bands", "windows_read", "features", "largest_sum_deviation")}
r["windows_skipped"] = len(s["windows_skipped"])
json.dump(r, open(sys.argv[2], "w"), indent=1)
print(json.dumps(r))
sys.exit(0 if r["bands"] == 10 and r["class_property"] == "new_label" else 1)
EOF
    mv "$PRIV/probs.partial" "$PRIV/probs"
    echo "== $(date -Is) done: records/e104_read.json ==" ;;

  grade)
    need_frozen
    test -s "$PRIV/probs/window_probs.npy" || { echo "no probabilities: run E104_MODE=read first"; exit 1; }
    GO=$OUT/grade
    if [ -e "$GO" ]; then
      if [ "${E104_REGRADE:-0}" = 1 ]; then rm -rf "$GO"; else echo "e104_out/grade exists; E104_REGRADE=1 replaces it"; exit 2; fi
    fi
    rm -rf "$GO.partial" "$OUT/grade_smoke"
    "${PY[@]}" --smoke --out-dir "$OUT/grade_smoke" > /dev/null
    "${PY[@]}" --grade --private-dir "$PRIV" --probs "$PRIV/probs" --out-dir "$GO.partial"
    mv "$GO.partial" "$GO"
    ls -la "$GO"
    echo "== $(date -Is) done: e104_out/grade/ (copy to exp/out/ when recording) ==" ;;
esac
du -sh "$OUT" "$PRIV" 2>/dev/null | sed "s#$PRIV#<private>#; s/^/on scratch: /" || true
