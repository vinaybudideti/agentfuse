# BudgetEngine: per-run state, graduated policies (alert, downgrade, compress, terminate)

import threading
from enum import Enum


class BudgetState(Enum):
    NORMAL = "normal"
    DOWNGRADED = "downgraded"
    COMPRESSED = "compressed"
    EXHAUSTED = "exhausted"


class BudgetExhaustedGracefully(Exception):
    def __init__(self, partial_results, receipt):
        self.partial_results = partial_results
        self.receipt = receipt
        super().__init__("Budget exhausted gracefully")


class BudgetEngine:
    DOWNGRADE_MAP = {
        "gpt-4o": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4o-mini",
        "claude-sonnet-4-6": "claude-haiku-4-5-20251001",
        "claude-opus-4-6": "claude-sonnet-4-6",
        "gemini-1.5-pro": "gemini-1.5-flash",
    }

    def __init__(self, run_id, budget_usd, model, alert_cb=None):
        if budget_usd <= 0:
            raise ValueError(f"Budget must be > 0, got {budget_usd}")
        self._lock = threading.Lock()
        self.run_id = run_id
        self.budget = budget_usd
        self.spent = 0.0
        self.model = model
        self.original_model = model
        self.alert_cb = alert_cb
        self.state = BudgetState.NORMAL
        self.compression_applied = False
        self.partial_results = []

    def check_and_act(self, estimated_cost, messages):
        with self._lock:
            return self._check_and_act_unlocked(estimated_cost, messages)

    def _check_and_act_unlocked(self, estimated_cost, messages):
        pct = (self.spent + estimated_cost) / self.budget

        if pct >= 1.0:
            return self._terminate(messages)
        if pct >= 0.90:
            compressed = self._compress(messages)
            return compressed, self.model
        if pct >= 0.80:
            self._downgrade()
            return messages, self.model
        if pct >= 0.60:
            self._alert(pct)

        return messages, self.model

    def _downgrade(self):
        if self.model in self.DOWNGRADE_MAP and self.state != BudgetState.DOWNGRADED:
            self.model = self.DOWNGRADE_MAP[self.model]
            self.state = BudgetState.DOWNGRADED
            self._alert(self.spent / self.budget, event="model_downgrade")

    def _compress(self, messages):
        if self.compression_applied:
            return messages
        system = [m for m in messages if m["role"] == "system"]
        recent = [m for m in messages if m["role"] != "system"][-6:]
        self.compression_applied = True
        self._alert(self.spent / self.budget, event="context_compression")
        return system + recent

    def _terminate(self, messages):
        self.state = BudgetState.EXHAUSTED
        raise BudgetExhaustedGracefully(
            partial_results=self.partial_results,
            receipt=None,
        )

    def _alert(self, pct, event="budget_alert"):
        if self.alert_cb is not None:
            self.alert_cb(pct, event)

    def record_cost(self, cost_usd):
        with self._lock:
            self.spent += cost_usd

    def add_partial_result(self, result):
        with self._lock:
            self.partial_results.append(result)
