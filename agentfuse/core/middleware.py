"""
Middleware Pipeline — composable request/response processing chain.

Inspired by Portkey's gateway architecture and Express.js middleware pattern.
Each middleware is a separate concern (cache, budget, retry, logging) that
processes the request before the LLM call and the response after.

This solves the problem of mixed concerns in wrap_openai/wrap_anthropic
where cache, budget, cost, and validation logic were all in one function.

Usage:
    pipeline = MiddlewarePipeline()
    pipeline.add(CacheMiddlewareStage(cache))
    pipeline.add(BudgetMiddlewareStage(engine))
    pipeline.add(CostTrackingStage(tracker))
    pipeline.add(ValidationStage())

    response = pipeline.execute(request, llm_call_fn)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Normalized request flowing through the middleware pipeline."""
    model: str
    messages: list[dict]
    temperature: float = 0.0
    tools: Optional[list] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    provider: str = "unknown"
    budget_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalized response flowing through the middleware pipeline."""
    raw: Any = None  # original provider response
    text: str = ""
    model: str = ""
    provider: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: Optional[str] = None
    cached: bool = False
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class MiddlewareStage:
    """Base class for middleware stages."""

    def before_call(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Process request before LLM call.
        Return LLMResponse to short-circuit (e.g., cache hit).
        Return None to continue pipeline."""
        return None

    def after_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        """Process response after LLM call. Must return response."""
        return response

    def on_error(self, request: LLMRequest, error: Exception) -> Optional[LLMResponse]:
        """Handle errors. Return LLMResponse to recover, None to re-raise."""
        return None


class MiddlewarePipeline:
    """
    Composable middleware pipeline for LLM request processing.

    Stages execute in order:
    1. before_call (each stage, in order) — can short-circuit
    2. LLM call (actual API call)
    3. after_call (each stage, in REVERSE order)
    4. on_error (each stage, if call failed)
    """

    def __init__(self):
        self._stages: list[MiddlewareStage] = []

    def add(self, stage: MiddlewareStage) -> "MiddlewarePipeline":
        """Add a middleware stage. Returns self for chaining."""
        self._stages.append(stage)
        return self

    def execute(self, request: LLMRequest,
                call_fn: Callable[[LLMRequest], Any]) -> LLMResponse:
        """Execute the pipeline."""
        start = time.monotonic()

        # Before-call phase (in order)
        for stage in self._stages:
            try:
                short_circuit = stage.before_call(request)
                if short_circuit is not None:
                    short_circuit.latency_ms = (time.monotonic() - start) * 1000
                    return short_circuit
            except Exception as e:
                logger.warning("Middleware before_call failed (%s): %s",
                               type(stage).__name__, e)
                raise

        # LLM call
        try:
            raw_result = call_fn(request)
        except Exception as e:
            # Error phase
            for stage in reversed(self._stages):
                try:
                    recovery = stage.on_error(request, e)
                    if recovery is not None:
                        recovery.latency_ms = (time.monotonic() - start) * 1000
                        return recovery
                except Exception:
                    pass
            raise

        # Wrap raw result
        response = LLMResponse(
            raw=raw_result,
            model=request.model,
            provider=request.provider,
            latency_ms=(time.monotonic() - start) * 1000,
        )

        # After-call phase (in REVERSE order — like finally blocks)
        for stage in reversed(self._stages):
            try:
                response = stage.after_call(request, response)
            except Exception as e:
                logger.warning("Middleware after_call failed (%s): %s",
                               type(stage).__name__, e)

        return response


# --- Built-in middleware stages ---

class LoggingStage(MiddlewareStage):
    """Logs every request and response."""

    def before_call(self, request: LLMRequest) -> None:
        logger.info("LLM call: model=%s, messages=%d, provider=%s",
                     request.model, len(request.messages), request.provider)
        return None

    def after_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        logger.info("LLM response: model=%s, cost=$%.6f, latency=%.0fms, cached=%s",
                     response.model, response.cost_usd, response.latency_ms, response.cached)
        return response


class TimingStage(MiddlewareStage):
    """Adds precise timing to responses."""

    def before_call(self, request: LLMRequest) -> None:
        request.metadata["_timing_start"] = time.monotonic()
        return None

    def after_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        start = request.metadata.get("_timing_start", time.monotonic())
        response.latency_ms = (time.monotonic() - start) * 1000
        return response
