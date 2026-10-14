#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -J e99pred
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp99 stage 3 (exp/jobs/E99_README.md): olmoearth_run's run_inference on the dataset e99_prepare.sh built, with the
# FT-AWF checkpoint and exp98's model.yaml (output_probs, p0..p9 float32), so every window's output layer holds the
# 10-channel softmax. The partition, GPU and time are given at submission:
#   ssh aicr "E99_SHA=$SHA sbatch --parsable -p rtx-batch --gpus=1 -t 24:00:00 --dependency=afterok:$PREP_JOB" < exp/jobs/e99_predict.sh
#   or one B200: -p b200-batch (24 h), or for the pilot -p b200-devel -t 02:00:00 (4 h at most).
# E99_SHA=<full sha> checks out that commit in exp99's clone. E99_PILOT=1 reads and writes the pilot paths and refuses
# a full run's inventory; without it, it refuses a pilot's. olmoearth_run's postprocess and combine are not run: exp99
# reads the windows' own output layers.
#
# Expected duration, from exp98's estimate: 85 x 85 crops of 16 px per window at batch size 4, about 265 to 300
# windows, 1 to 7 hours; the pilot minutes.
set -euo pipefail
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e99_code
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
if [ -n "${E99_SHA:-}" ]; then
  git cat-file -e "$E99_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E99_SHA" \
    || { echo "E99_SHA=$E99_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E99_SHA"
  [ "$(git rev-parse HEAD)" = "$E99_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E99_SHA=$E99_SHA (give the full sha)"; exit 2; }
else
  git checkout -q --force --detach origin/main
fi
DEPLOY=$SCRATCH/deploy
export E99_PILOT=${E99_PILOT:-0}
case "$E99_PILOT" in 0|1) ;; *) echo "E99_PILOT must be 0 or 1, got '$E99_PILOT'"; exit 2;; esac
if [ "$E99_PILOT" = 1 ]; then TAG=awf2017_pilot SFX=_e99_pilot LP=e99_pilot_ OUT=$REPO/exp/out/exp99/pilot
else TAG=awf2017 SFX=_e99 LP=e99_ OUT=$REPO/exp/out/exp99; fi
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
export NUM_WORKERS=$NCPU
export PREDICTION_OUTPUT_LAYER=output
export TRAINER_DATA_PATH=$DEPLOY/${LP}trainer_data
export EXTRA_FILES_PATH=$DEPLOY/${LP}extra_files
export WANDB_MODE=disabled WANDB_PROJECT=e99 WANDB_NAME=e99_awf2017 WANDB_ENTITY=e99-local
export E99_RUN=$RUN E99_DEPLOY=$DEPLOY E99_INV=$INV E99_CHECK=$CHECK
redact() { sed -E 's/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g'; }

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp99 predict, commit $(git rev-parse HEAD), pilot $E99_PILOT =="
[ -s "$INV" ] || { echo "no prepare inventory at $INV: e99_prepare.sh (with the same E99_PILOT) has not finished"; exit 1; }
"$PY" - "$INV" <<'EOF'
import json, os, sys
inv = json.load(open(sys.argv[1]))
assert inv.get("experiment") == "exp99", "not exp99's inventory"
assert inv.get("pilot", False) == (os.environ["E99_PILOT"] == "1"), \
    f"the inventory says pilot {inv.get('pilot', False)}, this job E99_PILOT={os.environ['E99_PILOT']}: a run must not mix"
assert inv["complete"] or inv.get("allow_incomplete"), "prepare found the run incomplete (inventory complete: false)"
assert inv.get("scl_check"), "prepare did not reach its SCL check"
if not inv["complete"]:
    print("WARNING: an incomplete run (E99_ALLOW_INCOMPLETE=1 at prepare); predict_check.json records it")
EOF
( cd "$CFG" && sha256sum -c --quiet "$DEPLOY/${TAG}_config.sha256" ) || { echo "$CFG changed since prepare"; exit 1; }
CKPT=$(cat "$DEPLOY/checkpoint_path.txt")
[ -s "$CKPT" ] || { echo "checkpoint missing; rerun e98_env.sh"; exit 1; }
read -r _ ST_INFER _ _ < "$DEPLOY/stage_spelling.txt"
d=$CLONE/olmoearth_projects
while :; do
  if [ -e "$d/.env" ]; then echo "a .env exists in $d, which main.py's load_dotenv would read; move it away"; exit 2; fi
  [ "$d" = / ] && break
  d=$(dirname "$d")
done
mkdir -p "$DEPLOY/logs" "$OUT" "$TRAINER_DATA_PATH" "$EXTRA_FILES_PATH"

if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  echo "this stage needs a GPU: submit with -p rtx-batch --gpus=1 (or -p b200-batch / b200-devel --gpus=1)"; exit 2
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
LOG=$DEPLOY/logs/${LP}run_inference-${SLURM_JOB_ID:-none}.log
echo "== $(date -Is) olmoearth_run one_stage --stage $ST_INFER (log $LOG) =="
cd "$DEPLOY" || exit 1
if ! "$PY" -m olmoearth_projects.main olmoearth_run one_stage --config_path "$CFG" --scratch_path "$RUN" \
     --checkpoint_path "$CKPT" --stage "$ST_INFER" > "$LOG" 2>&1; then
  echo "$ST_INFER failed; the last lines of $LOG, masked:"; tail -n 60 "$LOG" | redact; exit 1
fi
cd "$CODE" || exit 1
echo "$(date -Is) $ST_INFER done"

echo "== the output layers, checked =="
"$PY" - <<'EOF'
import glob, json, os, re
import numpy as np, rasterio
run = os.environ["E99_RUN"]
inv = json.load(open(os.environ["E99_INV"]))
P = "_".join(f"p{i}" for i in range(10))
res = {"job": os.environ.get("SLURM_JOB_ID"), "experiment": "exp99", "pilot": bool(inv.get("pilot")),
       "area_complete": inv["complete"], "allow_incomplete": bool(inv.get("allow_incomplete")), "windows": 0,
       "windows_without_input": 0, "windows_with_input": 0, "windows_with_output": 0, "missing_output": 0,
       "output_without_input": 0, "bad_bandset": 0, "sampled": []}
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
res["ok"] = bool(ok and picks and res["missing_output"] == 0 and res["bad_bandset"] == 0
                 and (res["windows_without_input"] == 0 or res["allow_incomplete"]))
json.dump(res, open(os.environ["E99_CHECK"], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sampled"}))
for e in res["sampled"]:
    print(e)
assert res["ok"], "the output check failed; see predict_check.json"
EOF
cp "$CHECK" "$OUT/e99_predict_check.json"
echo "== $(date -Is) done: $CHECK, $OUT/e99_predict_check.json =="
