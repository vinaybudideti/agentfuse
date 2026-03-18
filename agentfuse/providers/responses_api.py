"""
OpenAI Responses API adapter — handles the new API format for GPT-5.3+.

GPT-5.3 Codex and GPT-5.4 use the Responses API instead of Chat Completions.
Key differences:
- `text.format` instead of `response_format`
- `response_id` for conversation continuity
- CFG (Context-Free Grammar) support for structured tool calls
- Verbosity control (low/medium/high)

This adapter normalizes Responses API responses to match the Chat Completions
format that AgentFuse's gateway expects.

Usage:
    from agentfuse.providers.responses_api import normalize_response

    # Raw Responses API result
    raw = client.responses.create(model="gpt-5.4", ...)

    # Normalize to Chat Completions format
    normalized = normalize_response(raw)
    # Now works with AgentFuse's _record_cost, _validate_and_cache, etc.
"""

import logging
from typing import Any, Optional
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def is_responses_api_model(model: str) -> bool:
    """Check if a model uses the Responses API instead of Chat Completions.

    GPT-5.3-Codex is Responses API ONLY (no Chat Completions).
    GPT-5.4 supports both but Responses API is recommended.
    """
    responses_only = {"gpt-5.3-codex"}
    responses_preferred = {"gpt-5.4", "gpt-5.3"}

    return model in responses_only or model in responses_preferred


def normalize_response(response: Any) -> SimpleNamespace:
    """
    Normalize a Responses API response to Chat Completions format.

    Responses API returns:
        {output: [{type: "message", content: [{type: "text", text: "..."}]}],
         usage: {input_tokens, output_tokens},
         response_id: "..."}

    We convert to:
        {choices: [{message: {content: "..."}, finish_reason: "stop"}],
         usage: {prompt_tokens, completion_tokens}}
    """
    if response is None:
        return response

    # If it already looks like Chat Completions format, pass through
    if hasattr(response, "choices"):
        return response

    # Extract text from Responses API output
    text = ""
    output = getattr(response, "output", None)
    if output and isinstance(output, list):
        for item in output:
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if item_type == "message":
                content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                if content and isinstance(content, list):
                    for block in content:
                        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                        if block_type == "text":
                            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
                            break

    # Extract usage
    usage = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    if usage:
        prompt_tokens = getattr(usage, "input_tokens", 0)
        completion_tokens = getattr(usage, "output_tokens", 0)

    # Build normalized response
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, role="assistant"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        _agentfuse_normalized=True,
        _response_id=getattr(response, "response_id", None),
    )


def build_responses_params(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    **kwargs,
) -> dict:
    """
    Convert Chat Completions parameters to Responses API format.

    Key mappings:
    - messages → input (text or structured)
    - response_format → text.format
    - max_tokens → max_output_tokens
    """
    params = {
        "model": model,
        "input": messages,  # Responses API accepts Chat format too
    }

    if max_tokens:
        params["max_output_tokens"] = max_tokens

    if temperature > 0:
        params["temperature"] = temperature

    if tools:
        params["tools"] = tools

    # Verbosity control (GPT-5.4 feature)
    verbosity = kwargs.pop("verbosity", None)
    if verbosity:
        params["verbosity"] = verbosity  # "low", "medium", "high"

    params.update(kwargs)
    return params
