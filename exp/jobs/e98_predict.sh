#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -J e98pred
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp98 stage 3 of 4 (exp/jobs/E98_README.md): olmoearth_run's run_inference on the dataset e98_prepare.sh built, with
# the FT-AWF checkpoint and the e98 model.yaml (output_probs, p0..p9 float32), so every window's output layer holds
# the 10-channel softmax. The partition, GPUs and time are given at submission, per mode, as e89.sh does:
#   E98_STAGE=infer (default), one GPU on rtx-batch (RTX PRO 6000, 96 GB):
#     ssh aicr "sbatch --parsable -p rtx-batch --gpus=1 -t 24:00:00 --dependency=afterok:$PREP_JOB" < exp/jobs/e98_predict.sh
#   the alternative, one B200 (180 GB): the same with -p b200-batch. Both partitions allow 24 h.
#   E98_STAGE=post, olmoearth_run's postprocess and combine (CPU; optional, exp98 does not read their output), chained on
#   collect, never beside it: both would reset the same checkout, and postprocess may touch the windows collect copies:
#     ssh aicr "E98_STAGE=post sbatch --parsable -p cpu -t 04:00:00 --dependency=afterok:$COLL_JOB" < exp/jobs/e98_predict.sh
# E98_SHA=<full sha> checks out that commit instead of origin/main. E98_PILOT=1 reads and writes the pilot paths that
# e98_prepare.sh's E98_PILOT=1 made (awf_pilot_*, *_pilot.json, exp/out/exp98/pilot/) and refuses a full run's inventory;
# without it, it refuses a pilot's.
#
# Expected duration of infer, estimated: predict_config tiles each 1024 px window into 16 px crops with 4 px overlap
# (Ai2's model.yaml:274-283; overlap_ratio 0.25 -> overlap_pixels round(16 * 0.25) = 4, rslearn/train/dataset.py:530-541
# @ v0.0.27), 85 x 85 = 7,225 crops per window, about 1.6 M crops at batch size 4 (Ai2's model.yaml:228) for ~224
# windows. Each crop reads 12 item groups from the GeoTIFFs; the model is OlmoEarth-v1-Base at patch size 4 on 4 x 4
# tokens per timestep. Data loading dominates: 1 to 6 hours. The output is 10 float32 bands x 1024 x 1024 per window,
# 40 MiB raw (LZW, rslearn/utils/raster_format.py:505), about 9 GB for 224 windows.
set -euo pipefail
# The soft limit on open files is 1,024 on the cpu nodes (hard 131,072); olmoearth_run's worker pools exhaust it
# (job 1243542 failed in 26 s on "Too many open files" while starting its dataset-build pool).
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
cd "$REPO" || exit 1
git fetch -q origin main
if [ -n "${E98_SHA:-}" ]; then
  # A pinned stage checks out its own commit, whatever main has become since the chain was submitted (the chain spans
  # many hours and main moves). The checkout is left detached; a later `git reset --hard origin/main` still works on it.
  git cat-file -e "$E98_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E98_SHA" \
    || { echo "E98_SHA=$E98_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E98_SHA"
  [ "$(git rev-parse HEAD)" = "$E98_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E98_SHA=$E98_SHA (give the full sha)"; exit 2; }
else
  git reset -q --hard origin/main
fi
STAGE=${E98_STAGE:-infer}
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
DEPLOY=$SCRATCH/deploy
export E98_PILOT=${E98_PILOT:-0}
case "$E98_PILOT" in 0|1) ;; *) echo "E98_PILOT must be 0 or 1, got '$E98_PILOT'"; exit 2;; esac
if [ "$E98_PILOT" = 1 ]; then TAG=awf_pilot SFX=_pilot LP=pilot_ OUT=$REPO/exp/out/exp98/pilot
else TAG=awf SFX= LP= OUT=$REPO/exp/out/exp98; fi
RUN=$DEPLOY/${TAG}_run
CFG=$DEPLOY/${TAG}_config
INV=$DEPLOY/prepare_inventory$SFX.json
CHECK=$DEPLOY/predict_check$SFX.json
PY=$DEPLOY/venv/bin/python
CLONE=$DEPLOY/olmoearth_projects
export HF_HOME=$SCRATCH/hf
export PYTHONUNBUFFERED=1
NCPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=4
export NUM_WORKERS=$NCPU                       # model.yaml's data.init_args.num_workers
export PREDICTION_OUTPUT_LAYER=output
export TRAINER_DATA_PATH=$DEPLOY/${LP}trainer_data
export EXTRA_FILES_PATH=$DEPLOY/${LP}extra_files
export WANDB_MODE=disabled WANDB_PROJECT=e98 WANDB_NAME=e98_awf WANDB_ENTITY=e98-local
export E98_RUN=$RUN E98_DEPLOY=$DEPLOY E98_INV=$INV E98_CHECK=$CHECK
redact() { sed -E 's/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g'; }

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp98 predict, stage $STAGE, commit $(git rev-parse HEAD), pilot $E98_PILOT =="
[ -s "$INV" ] || { echo "no prepare inventory at $INV: e98_prepare.sh (with the same E98_PILOT) has not finished"; exit 1; }
"$PY" - "$INV" <<'EOF'
import json, os, sys
inv = json.load(open(sys.argv[1]))
assert inv.get("pilot", False) == (os.environ["E98_PILOT"] == "1"), \
    f"the inventory says pilot {inv.get('pilot', False)}, this job E98_PILOT={os.environ['E98_PILOT']}: a run must not mix"
assert "complete" in inv, "the inventory predates the completeness check: rerun e98_prepare.sh"
assert inv["complete"] or inv.get("allow_incomplete"), "prepare found the area incomplete (inventory complete: false)"
assert inv.get("scl_check"), "prepare did not reach its SCL check"
if not inv["complete"]:
    print("WARNING: a partial area (E98_ALLOW_INCOMPLETE=1 at prepare); predict_check.json records area_complete false")
EOF
( cd "$CFG" && sha256sum -c --quiet "$DEPLOY/${TAG}_config.sha256" ) || { echo "$CFG changed since prepare"; exit 1; }
CKPT=$(cat "$DEPLOY/checkpoint_path.txt")
[ -s "$CKPT" ] || { echo "checkpoint missing; rerun e98_env.sh"; exit 1; }
read -r _ ST_INFER ST_POST ST_COMBINE < "$DEPLOY/stage_spelling.txt"
d=$CLONE/olmoearth_projects
while :; do
  if [ -e "$d/.env" ]; then echo "a .env exists in $d, which main.py's load_dotenv would read; move it away"; exit 2; fi
  [ "$d" = / ] && break
  d=$(dirname "$d")
done
mkdir -p "$DEPLOY/logs" "$OUT" "$TRAINER_DATA_PATH" "$EXTRA_FILES_PATH"

run_stage() {   # run_stage <stage value> <log name>
  local log=$DEPLOY/logs/$LP$2-${SLURM_JOB_ID:-none}.log
  echo "== $(date -Is) olmoearth_run one_stage --stage $1 (log $log) =="
  cd "$DEPLOY" || exit 1
  if ! "$PY" -m olmoearth_projects.main olmoearth_run one_stage --config_path "$CFG" --scratch_path "$RUN" \
       --checkpoint_path "$CKPT" --stage "$1" > "$log" 2>&1; then
    echo "$1 failed; the last lines of $log, masked:"; tail -n 60 "$log" | redact; exit 1
  fi
  cd "$REPO" || exit 1
  echo "$(date -Is) $1 done"
}

case "$STAGE" in
  infer)
    if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
      echo "stage infer needs a GPU: submit with -p rtx-batch --gpus=1 (or -p b200-batch --gpus=1)"; exit 2
    fi
    "$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no GPU"
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print("torch", torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0), arch,
      "archs", torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), f"this torch build has no kernels for {arch}"
x = torch.randn(512, 512, device="cuda")
assert torch.isfinite(x @ x).all()
EOF
    run_stage "$ST_INFER" run_inference
    echo "== the output layers, checked =="
    "$PY" - <<'EOF'
import glob, json, os, re
import numpy as np, rasterio
run, deploy = os.environ["E98_RUN"], os.environ["E98_DEPLOY"]
inv = json.load(open(os.environ["E98_INV"]))
P = "_".join(f"p{i}" for i in range(10))
res = {"job": os.environ.get("SLURM_JOB_ID"), "pilot": bool(inv.get("pilot")), "area_complete": inv["complete"],
       "allow_incomplete": bool(inv.get("allow_incomplete")), "windows": 0, "windows_without_input": 0,
       "windows_with_input": 0, "windows_with_output": 0, "missing_output": 0, "output_without_input": 0,
       "bad_bandset": 0, "sampled": []}
outs = []
for r in inv["roots"]:
    for w in sorted(glob.glob(os.path.join(run, r["root"], "windows", "*", "*"))):
        ldir = os.path.join(w, "layers")
        names = os.listdir(ldir) if os.path.isdir(ldir) else []
        has_in = any(re.fullmatch(r"sentinel2(\.\d+)?", n) and os.path.exists(os.path.join(ldir, n, "completed"))
                     for n in names)
        has_out = os.path.exists(os.path.join(ldir, "output", "completed"))
        res["windows"] += 1
        res["windows_without_input"] += not has_in
        res["windows_with_input"] += has_in
        res["windows_with_output"] += has_out
        res["missing_output"] += has_in and not has_out
        res["output_without_input"] += has_out and not has_in
        if has_out:
            sets = [d for d in os.listdir(os.path.join(ldir, "output"))
                    if os.path.isfile(os.path.join(ldir, "output", d, "geotiff.tif"))]
            if sets != [P]:
                res["bad_bandset"] += 1
            else:
                outs.append(os.path.join(ldir, "output", P, "geotiff.tif"))
picks = sorted(set(np.linspace(0, len(outs) - 1, min(8, len(outs))).round().astype(int).tolist())) if outs else []
ok = True
for i in picks:
    with rasterio.open(outs[i]) as src:
        a = src.read()
    s = a.sum(axis=0, dtype=np.float64)
    covered = s > 0
    e = {"window_index": int(i), "bands": int(a.shape[0]), "dtype": str(a.dtype), "finite": bool(np.isfinite(a).all()),
         "min": float(a.min()), "max": float(a.max()), "uncovered_share": float(1 - covered.mean()),
         "max_abs_sum_minus_1": float(np.abs(s[covered] - 1).max()) if covered.any() else None,
         "argmax_is_9_share": float((a.argmax(axis=0)[covered] == 9).mean()) if covered.any() else None,
         "max_p9": float(a[9].max()), "mean_top1": float(a.max(axis=0)[covered].mean()) if covered.any() else None}
    e["ok"] = (e["bands"] == 10 and e["dtype"] == "float32" and e["finite"] and e["min"] >= -1e-6 and e["max"] <= 1 + 1e-6
               and e["max_abs_sum_minus_1"] is not None and e["max_abs_sum_minus_1"] <= 1e-3)
    ok &= e["ok"]
    res["sampled"].append(e)
# a window without input has no output either, so it would not show in missing_output; it counts unless the partial
# area was allowed at prepare
res["ok"] = bool(ok and picks and res["missing_output"] == 0 and res["bad_bandset"] == 0
                 and (res["windows_without_input"] == 0 or res["allow_incomplete"]))
json.dump(res, open(os.environ["E98_CHECK"], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sampled"}))
for e in res["sampled"]:
    print(e)
assert res["ok"], "the output check failed; see predict_check.json"
EOF
    cp "$CHECK" "$OUT/e98_predict_check.json"
    echo "== $(date -Is) done: $CHECK, $OUT/e98_predict_check.json =="
    ;;
  post)
    [ -s "$CHECK" ] || { echo "no predict check: stage infer has not finished"; exit 1; }
    # post runs after collect has copied and verified the windows (README: chain it on collect), never beside it
    "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("manifest_verified"), "collect has not verified its copy"' \
      "$OUT/e98_collect.json" || { echo "run e98_collect.sh first and chain post on it (--dependency=afterok:<collect job>)"; exit 1; }
    run_stage "$ST_POST" postprocess
    run_stage "$ST_COMBINE" combine
    echo "== GeoTIFFs olmoearth_run wrote outside the windows (counts and sizes only) =="
    find "$RUN" -name '*.tif' -not -path '*/windows/*' -printf '%s\n' | awk '{n++; b+=$1} END {printf "%d files, %.2f GiB\n", n, b/2^30}'
    ;;
  *)
    echo "unknown E98_STAGE '$STAGE': infer or post"; exit 2 ;;
esac
