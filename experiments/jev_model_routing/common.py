"""Shared definitions for the Jev model-tier routing experiment.

Tiers, the model configs they map to, list prices, JSONL helpers and client
construction. Stdlib only; `anthropic` is imported lazily by `anthropic_client()`.
"""
import json, os, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL_EXPERIMENT = HERE.parent / "jev_routing"
sys.path.insert(0, str(TOOL_EXPERIMENT))  # jev_eval.py: GOLD, extract_actions, Jev client

# Cheapest first. Jev and the other routers answer with one of these.
TIERS = ["fast", "capable", "reasoning"]

# What each tier is for, in terms of the work rather than the model, so a model
# upgrade is a MAPPINGS change and the router prompts stay the same. The `capable`
# line comes from the pilot: Haiku's misses were targets it had to infer.
TIER_CRITERIA = {
    "fast": "One obvious step with everything stated: answer directly, run one clearly named action, "
            "classify, reformat, or give a short reply",
    "capable": "A few dependent steps, or a target or parameter that must be inferred from context rather "
               "than read from the request",
    "reasoning": "Planning, debugging, math or code that must be exactly right, search over many possibilities, "
                 "synthesis across many sources, or a costly judgment call",
}

# Every config the labeling run measures. effort=None leaves the API default:
# no thinking on Haiku 4.5, adaptive thinking on Sonnet 5 and Opus 5 (effort high).
CONFIGS = {
    "haiku": {"model": "claude-haiku-4-5", "effort": None},
    "sonnet": {"model": "claude-sonnet-5", "effort": None},
    "opus": {"model": "claude-opus-5", "effort": None},
    "opus-medium": {"model": "claude-opus-5", "effort": "medium"},
    "opus-low": {"model": "claude-opus-5", "effort": "low"},
}

# How a routed tier becomes a request: pick the model, or keep Opus 5 and pick the effort.
MAPPINGS = {
    "model": {"fast": "haiku", "capable": "sonnet", "reasoning": "opus"},
    "effort": {"fast": "opus-low", "capable": "opus-medium", "reasoning": "opus"},
}

# $/MTok (input, output), first-party list prices; same table as ../jev_routing/llm_baseline.py.
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0)}


def cost_usd(model, usage):
    pin, pout = PRICE[model]
    fresh = usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout
    cached = (usage.get("cache_creation_input_tokens") or 0) * pin * 1.25 + (usage.get("cache_read_input_tokens") or 0) * pin * 0.1
    return (fresh + cached) / 1e6


def usage_dict(usage):
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {k: getattr(usage, k, None) or 0 for k in keys}


_write_lock = threading.Lock()


def append_jsonl(path, row):
    line = json.dumps(row, sort_keys=True)
    with _write_lock, open(path, "a") as f:
        f.write(line + "\n")


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn last line from an interrupted run; the rerun redoes that call
    return rows


def anthropic_client(max_retries=6):
    """Same key sources as llm_baseline.py; never prints the key.

    With no ANTHROPIC_API_KEY and no ~/.anthropic_key, the SDK falls back to
    ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
    """
    import anthropic

    p = Path.home() / ".anthropic_key"
    k = os.environ.get("ANTHROPIC_API_KEY") or (p.read_text().strip() if p.exists() else None)
    ws = Path.home() / ".anthropic_workspace"
    kwargs = {"max_retries": max_retries}
    if ws.exists():
        kwargs["default_headers"] = {"anthropic-workspace-id": ws.read_text().strip()}
    if k:
        kwargs["api_key"] = k
    return anthropic.Anthropic(**kwargs)


def api_errors():
    """The exceptions a call may raise after the SDK's own retries: recorded, not labeled."""
    import anthropic

    return (anthropic.APIStatusError, anthropic.APIConnectionError)
