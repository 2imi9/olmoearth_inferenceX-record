#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p rtx-batch
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -J e64armE
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e64armE-%j.out
set -u
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf
export HF_HUB_OFFLINE=1
UV=$HOME/.local/bin/uv
AG=/scratch/qi_zim_neu/OlmoEarth-Agent
BR=2imi9/feature-inferencex-review-set
MODEL=Qwen/Qwen2.5-7B-Instruct
IX=$HOME/olmoearth_inferenceX
SHIMVENV=/scratch/qi_zim_neu/olmoearth_inferenceX/agentvenv

echo "=== 1. the agent, in its own environment on scratch ==="
if [ ! -d $AG/.git ]; then git clone -q -b $BR https://github.com/2imi9/OlmoEarth-Agent.git $AG || exit 1; fi
cd $AG && git fetch -q origin && git checkout -q $BR && git reset -q --hard origin/$BR && git submodule update --init -q
echo "agent at $(git log --oneline -1)"
$UV sync --no-dev -q || { echo "uv sync failed"; exit 1; }
$UV pip install -q numpy || exit 1
$AG/.venv/bin/python -c "import olmoearth_agent, numpy; print('agent venv ok, python', __import__('sys').version.split()[0])"

echo "=== 2. the served model ==="
cd $IX
LOG=$IX/slurm/serve-$SLURM_JOB_ID.log
$SHIMVENV/bin/python exp/exp64_serve.py --model $MODEL --port 8077 > $LOG 2>&1 &
VPID=$!
for i in $(seq 1 90); do
  curl -sf http://localhost:8077/v1/models >/dev/null 2>&1 && { echo "endpoint up after ${i}0s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "=== SERVER DIED ==="; tail -30 $LOG; exit 1; }
  sleep 10
done
curl -sf http://localhost:8077/v1/models >/dev/null 2>&1 || { echo "=== TIMED OUT ==="; tail -30 $LOG; kill $VPID; exit 1; }

echo "=== 3. arm E: the OlmoEarth Agent on ${CARDS_N:-4} cards ==="
export LLM_ENDPOINT=http://localhost:8077/v1
export LLM_MODEL=$MODEL
export OLMOEARTH_API_KEY=exp64-no-studio-access
export OLMOEARTH_SCORES_ROOT=/scratch/qi_zim_neu/olmoearth_inferenceX/exp64_cards
PYTHONPATH=$IX:$IX/exp $AG/.venv/bin/python exp/exp64_arm_e.py --cards ${CARDS_N:-4} --arms E,E_forced

echo "=== 4. grade every arm present ==="
$IX/.venv/bin/python exp/exp64_agent_benchmark.py --stage grade --model $MODEL --endpoint $LLM_ENDPOINT --cards ${CARDS_N:-4}
kill $VPID 2>/dev/null
