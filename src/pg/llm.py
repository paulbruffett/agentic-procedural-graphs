from __future__ import annotations

import os

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_llm(model: str, temperature: float | None = None) -> ChatOpenAI:
    """Chat model routed through OpenRouter's OpenAI-compatible endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
    kwargs: dict = dict(
        model=model,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=3,
        # Explicit per-phase timeouts: a silently dropped connection must fail fast and retry
        # instead of blocking a rollout forever.
        timeout=httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=15.0),
        # Ask OpenRouter to report cost alongside token usage.
        extra_body={"usage": {"include": True}},
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def usage_of(msg: AIMessage) -> dict:
    """Input/output tokens and (if OpenRouter reported it) dollar cost for one response."""
    um = msg.usage_metadata or {}
    # Some providers return "token_usage": null, so guard both levels.
    token_usage = (msg.response_metadata or {}).get("token_usage") or {}
    cost = token_usage.get("cost") or 0.0
    return {
        "input_tokens": int(um.get("input_tokens", 0)),
        "output_tokens": int(um.get("output_tokens", 0)),
        "cost": float(cost),
    }


def is_fatal(e: Exception) -> bool:
    """Bad credentials (401) or out of credits (402): no retry or later episode can succeed, so the run should stop."""
    return getattr(e, "status_code", None) in (401, 402)


def add_usage(total: dict, part: dict, prefix: str) -> dict:
    out = dict(total)
    for k, v in part.items():
        key = f"{prefix}_{k}"
        out[key] = out.get(key, 0) + v
    return out
