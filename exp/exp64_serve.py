"""A minimal OpenAI-compatible chat endpoint over transformers, for exp64's arms.

Why this exists. exp64's plan page says the arms talk to "an open instruct model served on one cluster GPU behind
an OpenAI-compatible endpoint", so the agent loop is ordinary HTTP and the benchmark reproduces without any agent
framework. vLLM is the usual way to get that endpoint and it cannot run here: these nodes carry a CUDA driver and
no toolkit, so vLLM's Blackwell path fails first on `ninja` and then on `nvcc` while trying to JIT its attention
kernels, with VLLM_ATTENTION_BACKEND=TORCH_SDPA set. The plan page already called this and named the fallback,
transformers with PyTorch SDPA. This is that fallback with the thin HTTP layer the loop expects.

Scope, deliberately small. One model, greedy or low-temperature sampling, `/v1/models` and `/v1/chat/completions`,
tool calling through the model's own chat template. No batching, no streaming, no KV reuse across requests: exp64's
runs are short and few, and a slow honest endpoint beats a fast one that needs a toolkit the cluster lacks.

Tool calling. Qwen-family templates render tool schemas into the prompt and emit `<tool_call>{json}</tool_call>`.
That is parsed back into OpenAI's `tool_calls` shape so the loop in exp64_arms.py does not care which server it is
talking to. A model that emits a malformed block gets it passed through as ordinary content, where exp64's answer
parser records it as a parse failure rather than the harness inventing a call the model did not make.

Run: python exp/exp64_serve.py --model Qwen/Qwen2.5-7B-Instruct --port 8077
"""
import argparse
import json
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {}
_LOCK = threading.Lock()
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def load(model_id, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map="cuda", attn_implementation="sdpa")
    model.eval()
    STATE.update(tok=tok, model=model, model_id=model_id, torch=torch)
    print(f"loaded {model_id} on {next(model.parameters()).device} "
          f"({next(model.parameters()).dtype}, sdpa)", flush=True)


def generate(messages, tools, temperature, max_tokens, seed):
    tok, model, torch = STATE["tok"], STATE["model"], STATE["torch"]
    # A tool result must reach the template as the role it expects; the loop already emits role="tool".
    # Qwen3-family templates default to a thinking phase; the arms want the answer, and a <think> block full of
    # braces would defeat the one-json-object contract, so it is switched off where the template supports it.
    extra = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    text = tok.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=True,
                                   **extra)
    enc = tok([text], return_tensors="pt").to(model.device)
    if seed is not None:
        torch.manual_seed(int(seed))
    do_sample = bool(temperature and temperature > 0)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=int(max_tokens), do_sample=do_sample,
                             temperature=float(temperature) if do_sample else None,
                             top_p=0.95 if do_sample else None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def to_message(raw):
    """The model's text as an OpenAI assistant message, with any tool-call blocks lifted out."""
    raw = _THINK.sub("", raw)      # a residual thinking block is not part of the answer
    calls = []
    for m in _TOOL_CALL.finditer(raw):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue            # malformed: leave it in the content, do not invent a call
        args = obj.get("arguments", {})
        if isinstance(args, str):
            # Qwen sometimes writes the arguments object as a JSON string. An OpenAI-format server hands the
            # client ONE encoding of an object; passing the string through would hand it two, which the
            # exp64 agent pilot showed crashes a client that expects a dict. Undo the model's extra layer.
            try:
                inner = json.loads(args)
                args = inner if isinstance(inner, dict) else {}
            except json.JSONDecodeError:
                args = {}
        calls.append({"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                      "function": {"name": obj.get("name", ""), "arguments": json.dumps(args)}})
    content = _TOOL_CALL.sub("", raw).strip() if calls else raw.strip()
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return msg


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # noqa: A003 - the slurm log should carry results, not one line per token
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                     # noqa: N802 - BaseHTTPRequestHandler's contract
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": STATE.get("model_id", ""), "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):                    # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return
        try:
            with _LOCK:                   # one GPU, one generation at a time
                raw = generate(req.get("messages", []), req.get("tools"),
                               req.get("temperature", 0.0), req.get("max_tokens", 1024),
                               req.get("seed"))
        except Exception as exc:          # noqa: BLE001 - returned to the caller, which records it
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        # The openai SDK (which the OlmoEarth Agent uses) expects the full envelope, not only choices.
        self._send(200, {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
                         "created": int(time.time()), "model": STATE.get("model_id", ""),
                         "choices": [{"index": 0, "message": to_message(raw), "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    load(args.model, args.dtype)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving {args.model} on port {args.port}", flush=True)
    sys.stdout.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
