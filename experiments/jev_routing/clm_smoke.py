#!/usr/bin/env python3
"""Does a local CLM server give the answers CLM's own README says it should?

Sends the README quickstart request (one support ticket, a Noul, a Choice and a
Score) and compares the answers with the published values. Run it before pointing
jev_eval.py at the server: if these are far off, the encoder is not producing the
embeddings the projection head was trained on (wrong model, pooling or length limit),
and every benchmark number would be noise. The published values are for the
reference head, clm-latest.

Run: python3 clm_smoke.py     (stdlib only; http://127.0.0.1:8700/v1/systemone by default)
     JEV_API=http://studio.local:8700/v1/systemone python3 clm_smoke.py
Key: JEV_API_KEY, only if the server was started with CLM_API_KEY. Never printed.
"""
import json, os, sys, time, urllib.error, urllib.request

API = os.environ.get("JEV_API", "http://127.0.0.1:8700/v1/systemone")
MODEL = os.environ.get("JEV_MODEL", "clm-latest")
TOL = 0.05  # Metal and CUDA round differently, which moves these a little; a wrong encoder moves them far more
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
EXPECTED = [("urgency Noul", lambda a: a["urgency"]["noul"], 0.41022),       # Contrastive-LM/CLM README
            ("department P(billing)", lambda a: a["department"]["probabilities"]["billing"], 0.93878),
            ("frustration Score", lambda a: a["frustration"]["score"], 1.98386)]

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
          f"server {server_ms} ms, wall {wall_ms:.0f} ms (a cold cache embeds the option texts too)")
    worst = 0.0
    for name, get, want in EXPECTED:
        got = get(a)
        worst = max(worst, abs(got - want))
        print(f"  {name:22s} {got:.5f}   README {want:.5f}   diff {got - want:+.5f}")
    if a["department"]["choice"] != "billing" or worst > TOL:
        sys.exit(f"MISMATCH: choice {a['department']['choice']!r}, off by up to {worst:.3f} (tolerance {TOL}). "
                 "Check that the encoder is Qwen/Qwen3-8B served with --runner pooling and --max-model-len 2048.")
    print(f"OK: every value within {TOL} of the README; the encoder and head look right")

if __name__ == "__main__":
    main()
