#!/bin/bash
# Scores provider on aicr: Ai2's fine-tuned AWF model over the default area (Namanga, 512 x 512 px at 10 m, 2023),
# written as a (10, H, W) float32 logit raster plus manifest.json, then read back by `oe-inferencex assess` and
# `sample` as the round-trip check. See scripts/score_area.py.
#
# Submit from the Mac; the login node runs this one sbatch and nothing else. The script travels on stdin, so the
# cluster checkout need not have it yet; the job itself resets the checkout to origin/main before running, so
# commit and push scripts/score_area.py first:
#     ssh aicr 'sbatch --parsable' < scripts/score_area.sh
#
# Outputs are untracked and outside the repository, but the hard reset below restores tracked files: do not let this
# job overlap another job that writes tracked outputs in ~/olmoearth_inferenceX (chain with --dependency=afterany).
#SBATCH -A p2026_0089_neu
#SBATCH -p b200-devel,rtx-devel
#SBATCH -q interactive
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J oeix-scores
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX || exit 1
git fetch -q origin main; git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PATH="$HOME/.local/bin:$PATH"
OUT=/scratch/qi_zim_neu/olmoearth_inferenceX/scores/awf_namanga_2023_${SLURM_JOB_ID}
RUN="uv run --extra encoder --extra geo"

echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname), commit $(git rev-parse --short HEAD) =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

echo "== scores =="
$RUN python scripts/score_area.py --out "$OUT" || exit 1

echo "== round trip through the package =="
$RUN oe-inferencex assess "$OUT/scores.tif" --logits --out "$OUT/assess" || exit 1
$RUN oe-inferencex sample "$OUT/scores.tif" --logits --budget 300 --out "$OUT/sample/sample.csv" || exit 1

echo "== summary =="
$RUN python - "$OUT/manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print("model", m["model"]["repo"], "@", m["model"]["revision"], "| matches the record:", m["model"]["revision_matches_record"])
print("device", m["model"]["device"], "| tf32", m["model"]["tf32"], "| crops", m["inference"]["n_crops"], "in", m["inference"]["seconds"], "s")
print("scores", m["scores"]["shape"], m["scores"]["values"], "| no-data pixels", m["scores"]["n_nodata_pixels"], "| sha256", m["scores"]["sha256"])
print("argmax pixels", m["argmax_pixel_counts"])
EOF
ls -la "$OUT" "$OUT/assess" "$OUT/sample"
echo "== done $(date -Is): $OUT =="
