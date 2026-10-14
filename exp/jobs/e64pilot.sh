#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p rtx-batch
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J e64pilot
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e64pilot-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf
export HF_HUB_OFFLINE=1
VENV=/scratch/qi_zim_neu/olmoearth_inferenceX/agentvenv
MODEL=Qwen/Qwen2.5-7B-Instruct
LOG=/home/qi_zim_neu/olmoearth_inferenceX/slurm/serve-$SLURM_JOB_ID.log

echo "=== starting the transformers shim (vLLM needs a toolkit this cluster lacks) ==="
$VENV/bin/python exp/exp64_serve.py --model $MODEL --port 8077 > $LOG 2>&1 &
VPID=$!
for i in $(seq 1 90); do
  curl -sf http://localhost:8077/v1/models >/dev/null 2>&1 && { echo "endpoint up after ${i}0s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "=== SERVER DIED ==="; tail -30 $LOG; exit 1; }
  sleep 10
done
curl -sf http://localhost:8077/v1/models >/dev/null 2>&1 || { echo "=== TIMED OUT ==="; tail -30 $LOG; kill $VPID; exit 1; }

echo "=== pilot: 4 cards, all four arms, 1 sample ==="
export LLM_ENDPOINT=http://localhost:8077/v1
.venv/bin/python exp/exp64_agent_benchmark.py --stage run  --model $MODEL --endpoint $LLM_ENDPOINT --arms ABCD --samples 1 --cards 4
.venv/bin/python exp/exp64_agent_benchmark.py --stage grade --model $MODEL --endpoint $LLM_ENDPOINT --arms ABCD --samples 1 --cards 4
kill $VPID 2>/dev/null
