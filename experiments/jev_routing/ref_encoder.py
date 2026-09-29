#!/usr/bin/env python3
"""A reference encoder for CLM: Qwen3-8B through Hugging Face transformers.

It serves the same /v1/embeddings API as `vllm serve Qwen/Qwen3-8B --runner pooling`
(CLM's serve_qwen3_8b.sh): the final, normed hidden state of the last token,
L2-normalized, one text per forward pass. That is what CLM's head was trained on,
computed without vLLM, so it tells a wrong encoder from wrong expectations when
clm_smoke.py reports MISMATCH:

  1. stop `vllm serve` (this needs the same ~16 GB) and start this on its port
  2. restart clm-serve, which keeps every embedding it has seen in memory
  3. run clm_smoke.py again
     - the same numbers as before: the vllm serve encoder is fine. (On a Mac Studio
       vllm-metal matched this to cosine 0.9998 or better; CLM's README example
       doesn't reproduce for anyone, see github.com/Contrastive-LM/CLM issue 15.)
     - different numbers: the vllm serve encoder is wrong. This one is slower but
       right, so the benchmark can run on it.

Run (in a venv with torch and transformers):
     python ref_encoder.py      # port 8090; CUDA, else Apple MPS, else CPU
First start loads the weights from the Hugging Face cache, or downloads them. The load
report lists lm_head.weight as UNEXPECTED: an encoder has no use for the LM head.
"""
import argparse, base64, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

DTYPE_ARG = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"  # renamed in 4.56


class Encoder:
    def __init__(self, model_id, device, dtype):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, **{DTYPE_ARG: dtype}).to(device).eval()
        if next(self.model.parameters()).dtype != dtype:
            raise SystemExit(f"loaded {next(self.model.parameters()).dtype}, not {dtype}; upgrade transformers")
        self.device, self.lock = device, threading.Lock()

    def embed(self, text, max_tokens=None):
        """-> (L2-normalized float32 vector, tokens used)."""
        ids = self.tok(text)["input_ids"]
        if max_tokens:
            ids = ids[-max_tokens:]  # as vLLM's truncate_prompt_tokens: keep the last k
        with self.lock, torch.inference_mode():
            h = self.model(input_ids=torch.tensor([ids], device=self.device)).last_hidden_state[0, -1]
        v = h.float().cpu().numpy()
        return v / max(float(np.linalg.norm(v)), 1e-12), len(ids)


def handler(enc, served_name):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":  # clm-serve's health check
                return self.send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            self.send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/embeddings":
                return self.send(404, {"error": f"no route {self.path}"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
                if not all(isinstance(t, str) for t in inputs):
                    return self.send(400, {"error": "input must be a string or a list of strings"})
                b64 = req.get("encoding_format") == "base64"
                data, tokens = [], 0
                for i, text in enumerate(inputs):
                    v, n = enc.embed(text, req.get("truncate_prompt_tokens"))
                    tokens += n
                    e = base64.b64encode(v.astype("<f4").tobytes()).decode() if b64 else v.tolist()
                    data.append({"object": "embedding", "index": i, "embedding": e})
            except Exception as e:  # noqa: BLE001  (reported to clm-serve, which shows it as an encoder error)
                return self.send(500, {"error": f"{type(e).__name__}: {e}"})
            self.send(200, {"object": "list", "model": served_name, "data": data,
                            "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

        def log_message(self, *args):
            pass

    return Handler


def main():
    default = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--served-model-name", default="qwen3-8b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--device", default=default)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="bfloat16 is what vllm serve uses for Qwen3-8B")
    args = ap.parse_args()
    t0 = time.perf_counter()
    enc = Encoder(args.model, args.device, getattr(torch, args.dtype))
    print(f"[ref] {args.model} on {args.device} ({args.dtype}), loaded in {time.perf_counter() - t0:.0f} s; "
          f"POST http://{args.host}:{args.port}/v1/embeddings as {args.served_model_name}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler(enc, args.served_model_name)).serve_forever()


if __name__ == "__main__":
    main()
