#!/usr/bin/env python3
"""Is a local CLM stack wired the way everyone else's is?

Sends CLM's README quickstart request (one support ticket, a Noul, a Choice and a
Score) and compares the answers with what independent stacks get for it with the
released head: CUDA with vLLM 0.30 on a DGX Spark, an RTX 4090 (CLM's own
assets/playground.png), MLX, PyTorch MPS and Hugging Face transformers all land within
the tolerances below (urgency 0.836-0.848, billing 0.987-0.989, frustration 1.9999-2.0,
98 encoder tokens on a cold cache; github.com/Contrastive-LM/CLM issues #3 and #15).
A different encoder, pooling or head moves these a lot, so run it before jev_eval.py.

The README itself says 0.41022 / 0.93878 / 1.98386 / 106 tokens. Nobody reproduces
that, including with the code at the commit the README shipped in (issue #15), so it is
not what this checks. It also does not check that CLM is any good at anything: these
values are the same near-constant answers the head gives for most states.

Run: python3 clm_smoke.py     (stdlib only; http://127.0.0.1:8700/v1/systemone by default)
     JEV_API=http://studio.local:8700/v1/systemone python3 clm_smoke.py
Key: JEV_API_KEY, only if the server was started with CLM_API_KEY. Never printed.
"""
import json, os, sys, time, urllib.error, urllib.request

API = os.environ.get("JEV_API", "http://127.0.0.1:8700/v1/systemone")
MODEL = os.environ.get("JEV_MODEL", "clm-latest")
BODY = {
    "state": "Customer: my invoice was charged twice and nobody answers the phone!",
    "model": MODEL,
    "questions": {
        "urgency": {"type": "noul", "instructions": "Is this urgent?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}
# (name, getter, value independent stacks get, tolerance); Metal, CUDA and CPU round differently, by about 0.005
EXPECTED = [("urgency Noul", lambda a: a["urgency"]["noul"], 0.842, 0.02),
            ("department P(billing)", lambda a: a["department"]["probabilities"]["billing"], 0.988, 0.01),
            ("frustration Score", lambda a: a["frustration"]["score"], 1.9999, 0.01)]

def health(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        sys.exit(f"no CLM server at {base} ({e}); start clm-serve first, see the README")

def main():
    h = health(API.rsplit("/v1/", 1)[0])
    if h.get("mock"):
        print("WARNING: this is CLM's playground mock (fake encoder), so expect a mismatch")
    if not h.get("embedder"):
        print("WARNING: clm-serve cannot reach its encoder; is `vllm serve` up on the --emb-url port?")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("JEV_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JEV_API_KEY']}"
    req = urllib.request.Request(API, data=json.dumps(BODY).encode(), method="POST", headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:  # the first request also warms the encoder up
            resp, server_ms = json.load(r), r.headers.get("X-CLM-Latency-Ms")
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {API}: {e.read()[:500]!r}")
    wall_ms = 1000 * (time.perf_counter() - t0)
    a = resp["answers"]
    print(f"model {resp.get('model')}, encoder tokens {(resp.get('usage') or {}).get('input_tokens')}, "
          f"server {server_ms} ms, wall {wall_ms:.0f} ms (0 tokens once clm-serve has seen these texts)")
    bad = []
    for name, get, want, tol in EXPECTED:
        got = get(a)
        print(f"  {name:22s} {got:.5f}   other stacks {want:.4f} +/- {tol}   diff {got - want:+.5f}")
        if abs(got - want) > tol:
            bad.append(name)
    if bad or a["department"]["choice"] != "billing":
        sys.exit(f"MISMATCH in {bad or ['department choice']}. Check that the encoder is Qwen/Qwen3-8B served with "
                 "--runner pooling --max-model-len 2048, and that clm-serve was restarted after any encoder change. "
                 "ref_encoder.py checks an encoder against an independent implementation (see the README).")
    print("OK: matches what independent stacks get. The wiring is right; this says nothing about accuracy.")

if __name__ == "__main__":
    main()
