"""
BudgetEngine: per-run state, graduated policies (alert, downgrade, compress, terminate).

FIXED in Phase 0:
- Added ContextVar for per-run isolation in async contexts
- Added asyncio.Lock for async codepaths (threading.Lock causes deadlocks in async)
- Separate sync (check_and_act) and async (check_and_act_async) entry points
- Core logic in _check_and_act_inner, called under appropriate lock
- Instance-level locks (not class-level) — each run gets its own lock
"""

import asyncio
import threading
from contextvars import ContextVar
from enum import Enum


# Module-level ContextVar: each asyncio Task / thread gets its own copy
_current_run_id: ContextVar[str | None] = ContextVar('_current_run_id', default=None)


class BudgetState(Enum):
    NORMAL = "normal"
    DOWNGRADED = "downgraded"
    COMPRESSED = "compressed"
    EXHAUSTED = "exhausted"


class BudgetExhaustedGracefully(Exception):
    def __init__(self, partial_results, receipt, run_id=None, spent=None, budget=None):
        self.partial_results = partial_results
        self.receipt = receipt
        self.run_id = run_id
        self.spent = spent
        self.budget = budget
        msg = "Budget exhausted gracefully"
        if run_id and budget:
            msg = f"Budget exhausted for run '{run_id}': ${spent:.4f} of ${budget:.2f} spent"
        super().__init__(msg)


class BudgetEngine:
    DOWNGRADE_MAP = {
        "gpt-4o": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4o-mini",
        "gpt-4.1": "o4-mini",
        "o1": "o3",
        "o3": "o4-mini",
        "claude-sonnet-4-6": "claude-haiku-4-5-20251001",
        "claude-opus-4-6": "claude-sonnet-4-6",
        "gemini-1.5-pro": "gemini-1.5-flash",
        "gemini-2.5-pro": "gemini-2.0-flash",
    }

    def __init__(self, run_id, budget_usd, model, alert_cb=None):
        if budget_usd <= 0:
            raise ValueError(f"Budget must be > 0, got {budget_usd}")
        self.run_id = run_id
        self.budget = budget_usd
        self.spent = 0.0
        self.model = model
        self.original_model = model
        self.alert_cb = alert_cb
        self.state = BudgetState.NORMAL
        self.compression_applied = False
        self.partial_results = []

        # Instance-level locks — each BudgetEngine gets its own lock
        # so different runs don't serialize through a shared lock
        self._sync_lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None  # created lazily

        # Set the ContextVar so async tasks know which run they belong to
        _current_run_id.set(run_id)

    def _get_async_lock(self) -> asyncio.Lock:
        """Create asyncio.Lock lazily — can't create at __init__ time outside event loop."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def check_and_act(self, estimated_cost, messages):
        """Sync version — uses threading.Lock."""
        with self._sync_lock:
            return self._check_and_act_inner(estimated_cost, messages)

    async def check_and_act_async(self, estimated_cost, messages):
        """Async version — uses asyncio.Lock (NEVER use threading.Lock in async)."""
        async with self._get_async_lock():
            return self._check_and_act_inner(estimated_cost, messages)

    def _check_and_act_inner(self, estimated_cost, messages):
        """Core logic — called under lock from both sync and async paths."""
        pct = (self.spent + estimated_cost) / self.budget

        if pct >= 1.0:
            return self._terminate(messages)
        if pct >= 0.90:
            self._downgrade()  # Downgrade if not already
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
            run_id=self.run_id,
            spent=self.spent,
            budget=self.budget,
        )

    def _alert(self, pct, event="budget_alert"):
        if self.alert_cb is not None:
            self.alert_cb(pct, event)

    def record_cost(self, cost_usd):
        with self._sync_lock:
            self.spent += cost_usd

    async def record_cost_async(self, cost_usd):
        """Async version of record_cost."""
        async with self._get_async_lock():
            self.spent += cost_usd

    def add_partial_result(self, result):
        with self._sync_lock:
            self.partial_results.append(result)

    @staticmethod
    def get_current_run_id() -> str | None:
        """Get the current run_id from ContextVar (async-safe)."""
        return _current_run_id.get()

    def __repr__(self) -> str:
        return (f"BudgetEngine(run_id={self.run_id!r}, budget=${self.budget:.2f}, "
                f"spent=${self.spent:.4f}, model={self.model!r}, state={self.state.value})")
