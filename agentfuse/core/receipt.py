"""
CostReceiptEmitter — full JSON schema, per-step logging, emit on run completion.

Produces a structured cost receipt for every agent run with all
fields from Section 3.9 of the master plan.
"""

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReceiptStep:
    step: int
    step_type: str              # "tool_call" | "reasoning" | "llm_call"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_tier: Optional[int]   # None, 1, or 2
    latency_ms: int
    tool_name: Optional[str] = None


class CostReceiptEmitter:

    RECEIPT_VERSION = "1.0"

    def __init__(self, run_id: str, agent_type: str = "unknown",
                 user_id: str = "unknown", budget_usd: float = 0.0):
        self.run_id = run_id
        self.agent_type = agent_type
        self.user_id = user_id
        self.budget_usd = budget_usd
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: List[ReceiptStep] = []
        self.total_cost_usd = 0.0
        self.cache_savings_usd = 0.0
        self.retry_cost_usd = 0.0
        self.model_downgrades = 0
        self.context_compressions = 0
        self.cache_hits = 0
        self.total_calls = 0

    def add_step(self, step_type: str, model: str, input_tokens: int,
                 output_tokens: int, cost_usd: float, latency_ms: int,
                 cache_tier=None, tool_name=None):
        step_num = len(self.steps) + 1
        self.steps.append(ReceiptStep(
            step=step_num, step_type=step_type, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, cache_tier=cache_tier,
            latency_ms=latency_ms, tool_name=tool_name,
        ))
        self.total_cost_usd += cost_usd
        self.total_calls += 1
        if cache_tier is not None:
            self.cache_hits += 1

    def record_cache_saving(self, saved_usd: float):
        self.cache_savings_usd += saved_usd

    def record_retry_cost(self, cost_usd: float):
        self.retry_cost_usd += cost_usd

    def record_model_downgrade(self):
        self.model_downgrades += 1

    def record_context_compression(self):
        self.context_compressions += 1

    def emit(self, status: str = "completed") -> dict:
        """Returns the full receipt as a dict matching Section 3.9 schema."""
        completed_at = datetime.now(timezone.utc).isoformat()
        cache_hit_rate = (self.cache_hits / self.total_calls
                          if self.total_calls > 0 else 0.0)
        budget_util = (self.total_cost_usd / self.budget_usd * 100
                       if self.budget_usd > 0 else 0.0)

        return {
            "receipt_version": self.RECEIPT_VERSION,
            "run_id": self.run_id,
            "agent_type": self.agent_type,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "status": status,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": self.budget_usd,
            "budget_utilization_pct": round(budget_util, 2),
            "cache_savings_usd": round(self.cache_savings_usd, 6),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "retry_cost_usd": round(self.retry_cost_usd, 6),
            "model_downgrades": self.model_downgrades,
            "context_compressions": self.context_compressions,
            "steps": [
                {
                    "step": s.step,
                    "step_type": s.step_type,
                    "tool_name": s.tool_name,
                    "model": s.model,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                    "cache_tier": s.cache_tier,
                    "latency_ms": s.latency_ms,
                }
                for s in self.steps
            ],
        }

    def emit_json(self, status: str = "completed") -> str:
        return json.dumps(self.emit(status), indent=2)
