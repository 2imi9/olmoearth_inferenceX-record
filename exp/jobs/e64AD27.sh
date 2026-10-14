#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p b200-batch,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -J e64AD27
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e64AD27-%j.out
set -u
B=/scratch/qi_zim_neu/olmoearth_inferenceX
IX=$HOME/olmoearth_inferenceX
AG=$B/OlmoEarth-Agent
BR=2imi9/feature-inferencex-review-set
UV=$HOME/.local/bin/uv
SIF=$B/vllm-openai-nightly.sif
MODEL=nvidia/Qwen3.8-27B-NVFP4
PORT=8077
SERVE_FLAGS="${SERVE_FLAGS:---kv-cache-dtype fp8_e4m3 --reasoning-parser qwen3}"
CARDS_N=${CARDS_N:-0}
SAMPLES=${SAMPLES:-3}
mkdir -p $B/vllm_cache $B/inductor_cache $B/triton_cache
ENV="--env HF_HOME=$B/hf,HF_HUB_OFFLINE=0,VLLM_CACHE_ROOT=$B/vllm_cache,TORCHINDUCTOR_CACHE_DIR=$B/inductor_cache,TRITON_CACHE_DIR=$B/triton_cache"

echo "=== 2. vLLM in the container: $MODEL ==="
LOG=$IX/slurm/serve-$SLURM_JOB_ID.log
apptainer exec --nv --bind /scratch:/scratch $ENV $SIF \
  vllm serve $MODEL --port $PORT --tensor-parallel-size 1 --max-model-len 65536 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --seed 0 \
    --gpu-memory-utilization 0.85 --max-num-seqs 8 --max-num-batched-tokens 32768 --enable-chunked-prefill \
    $SERVE_FLAGS > $LOG 2>&1 &
VPID=$!
for i in $(seq 1 120); do
  curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 && { echo "endpoint up after ${i}0s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "=== SERVER DIED ==="; tail -20 $LOG; exit 1; }
  sleep 10
done
curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 || { echo "=== TIMED OUT ==="; tail -20 $LOG; kill $VPID; exit 1; }
export LLM_ENDPOINT=http://localhost:$PORT/v1 LLM_MODEL=$MODEL

echo "=== preflight: a two-turn tool loop through exp64_arms.chat, exactly as the arms run it ==="
cd $IX
PYTHONPATH=$IX:$IX/exp .venv/bin/python - <<'PYPRE' || { echo "=== PREFLIGHT FAILED ==="; kill $VPID; exit 1; }
import json, os, sys
sys.path.insert(0, "exp"); import exp64_arms as arms
ep, model = os.environ["LLM_ENDPOINT"], os.environ["LLM_MODEL"]
tools = [{"type": "function", "function": {"name": "assess", "description": "Rank windows by suspicion and return the review set.",
          "parameters": {"type": "object", "properties": {}, "required": []}}}]
msgs = [{"role": "system", "content": "You are a careful analyst."},
        {"role": "user", "content": 'Call assess, then answer with ONE json object {"review_windows": [[r, c], ...]} listing the windows it returns.'}]
m1 = arms.chat(ep, model, msgs, tools=tools, temperature=0.0, seed=0)
assert m1.get("tool_calls"), f"turn 1 made no tool call: {json.dumps(m1)[:400]}"
call = m1["tool_calls"][0]; args = json.loads(call["function"]["arguments"]); assert isinstance(args, dict), args
msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": m1["tool_calls"]})
msgs.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"review_windows": [[1, 2], [3, 4]], "n_review": 2})})
m2 = arms.chat(ep, model, msgs, tools=tools, temperature=0.0, seed=0)
text = m2.get("content") or ""; ans, err = arms._parse_answer(text)
assert err is None and ans.get("review_windows"), f"turn 2 is not a json answer: err={err} text={text[:300]!r} raw={json.dumps(m2)[:300]}"
print("preflight OK: tool call ->", args, "| answer ->", ans)
PYPRE

echo "=== 3. arms A-D: cards=${CARDS_N:-all} samples=$SAMPLES ==="
cd $IX
.venv/bin/python exp/exp64_agent_benchmark.py --stage run --model $MODEL --endpoint $LLM_ENDPOINT --arms ABCD --samples $SAMPLES --cards $CARDS_N

echo "=== 5. grade ==="
.venv/bin/python exp/exp64_agent_benchmark.py --stage grade --model $MODEL --endpoint $LLM_ENDPOINT --samples $SAMPLES --cards $CARDS_N
kill $VPID 2>/dev/null
