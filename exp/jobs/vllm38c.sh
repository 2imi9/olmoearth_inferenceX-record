#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p b200-batch
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 01:00:00
#SBATCH -J vllm38c
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/vllm38c-%j.out
set -u
B=/scratch/qi_zim_neu/olmoearth_inferenceX
SIF=$B/vllm-openai-nightly.sif
MODEL=nvidia/Qwen3.8-27B-NVFP4
PORT=8077
mkdir -p $B/vllm_cache $B/inductor_cache $B/triton_cache
ENV="--env HF_HOME=$B/hf,HF_HUB_OFFLINE=0,VLLM_CACHE_ROOT=$B/vllm_cache,TORCHINDUCTOR_CACHE_DIR=$B/inductor_cache,TRITON_CACHE_DIR=$B/triton_cache"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

serve() {  # $1 label, rest = extra flags
  local label=$1; shift
  LOG=/home/qi_zim_neu/olmoearth_inferenceX/slurm/vllm38c-$SLURM_JOB_ID-$label.log
  echo "=== attempt $label ==="
  apptainer exec --nv --bind /scratch:/scratch $ENV $SIF \
    vllm serve $MODEL --port $PORT --tensor-parallel-size 1 --max-model-len 65536 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --seed 0 \
      --gpu-memory-utilization 0.85 --max-num-seqs 8 --max-num-batched-tokens 32768 --enable-chunked-prefill \
      "$@" > $LOG 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do
    curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 && { echo "attempt $label: endpoint up after ${i}0s"; return 0; }
    kill -0 $VPID 2>/dev/null || break
    sleep 10
  done
  kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
  echo "attempt $label failed; root cause:"; grep -E "Error|error:|Exception" $LOG | grep -vE "\]\s+(File|return|super|engine_core|\^)" | tail -4 | cut -c1-300
  return 1
}
probe() {
  echo "=== tool-call round trip ==="
  curl -s http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
   "model": "'"$MODEL"'", "temperature": 0, "max_tokens": 400,
   "messages": [{"role":"user","content":"Rank this card. Call the assess tool with budget 0.05, then answer in one json object."}],
   "tools": [{"type":"function","function":{"name":"assess","description":"Rank windows by suspicion.",
      "parameters":{"type":"object","properties":{"budget":{"type":"number"}},"required":["budget"]}}}],
   "tool_choice": "auto"}' | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; print(json.dumps({k:m.get(k) for k in ('content','tool_calls','reasoning_content','reasoning')}, indent=1)[:900]); print('usage:', r.get('usage'))"
  echo "=== throughput: 200 tokens ==="
  T0=$(date +%s.%N); curl -s http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"'"$MODEL"'","temperature":0,"max_tokens":200,"messages":[{"role":"user","content":"Count from 1 to 150, comma separated."}]}' | python3 -c "import sys,json; print('completion tokens:', json.load(sys.stdin)['usage']['completion_tokens'])"; T1=$(date +%s.%N); echo "wall: $(echo "$T1 - $T0" | bc) s"
}
if serve card --kv-cache-dtype fp8_e4m3 --reasoning-parser qwen3; then probe; kill $VPID; exit 0; fi
if serve plain; then probe; kill $VPID; exit 0; fi
echo "=== container attempts failed ==="; exit 1
