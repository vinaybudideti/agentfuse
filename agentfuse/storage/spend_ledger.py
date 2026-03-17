"""
SpendLedger — persistent cost tracking that survives process restarts.

Production systems like LiteLLM use PostgreSQL, Helicone uses ClickHouse.
This provides a file-based ledger that can be upgraded to a database later.

The ledger records every cost event as an append-only log. This is the
pattern used by financial systems — never update, only append. This
guarantees auditability and crash recovery.

Usage:
    ledger = SpendLedger("~/.agentfuse/spend.jsonl")
    ledger.record(run_id="run_1", model="gpt-4o", cost_usd=0.05,
                  input_tokens=100, output_tokens=50)

    # Survives restart:
    total = ledger.get_total_spend()
    by_model = ledger.get_spend_by_model()
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional


class SpendLedger:
    """
    Append-only spend ledger with file persistence.

    Each entry is a JSON line in a JSONL file. This pattern:
    - Survives process crashes (append-only, no corruption)
    - Is auditable (every event recorded)
    - Can be replayed to reconstruct state
    - Can be migrated to PostgreSQL/ClickHouse later
    """

    def __init__(self, path: str = "~/.agentfuse/spend.jsonl"):
        self._path = Path(os.path.expanduser(path))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # In-memory aggregates (rebuilt from ledger on init)
        self._total_usd = 0.0
        self._total_calls = 0
        self._by_model: dict[str, float] = {}
        self._by_run: dict[str, float] = {}
        self._by_provider: dict[str, float] = {}

        # Rebuild state from existing ledger
        self._rebuild_from_file()

    def record(
        self,
        run_id: str,
        model: str,
        cost_usd: float,
        provider: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached: bool = False,
    ):
        """Record a cost event. Appends to file and updates in-memory state."""
        entry = {
            "ts": time.time(),
            "run_id": run_id,
            "model": model,
            "provider": provider,
            "cost_usd": round(cost_usd, 8),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached": cached,
        }

        with self._lock:
            # Append to file
            try:
                with open(self._path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as e:
                import logging
                logging.getLogger(__name__).warning("Ledger write failed: %s", e)

            # Update in-memory aggregates
            self._total_usd += cost_usd
            self._total_calls += 1
            self._by_model[model] = self._by_model.get(model, 0.0) + cost_usd
            self._by_run[run_id] = self._by_run.get(run_id, 0.0) + cost_usd
            self._by_provider[provider] = self._by_provider.get(provider, 0.0) + cost_usd

    def get_total_spend(self) -> float:
        """Total spend across all runs."""
        with self._lock:
            return self._total_usd

    def get_spend_by_model(self) -> dict[str, float]:
        """Spend breakdown by model."""
        with self._lock:
            return dict(self._by_model)

    def get_spend_by_run(self) -> dict[str, float]:
        """Spend breakdown by run_id."""
        with self._lock:
            return dict(self._by_run)

    def get_spend_by_provider(self) -> dict[str, float]:
        """Spend breakdown by provider."""
        with self._lock:
            return dict(self._by_provider)

    def get_run_spend(self, run_id: str) -> float:
        """Get total spend for a specific run."""
        with self._lock:
            return self._by_run.get(run_id, 0.0)

    def get_entries(self, run_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Read raw ledger entries. Optionally filter by run_id."""
        entries = []
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if run_id is None or entry.get("run_id") == run_id:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return entries[-limit:]

    def _rebuild_from_file(self):
        """Rebuild in-memory aggregates from existing ledger file."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        cost = entry.get("cost_usd", 0.0)
                        model = entry.get("model", "unknown")
                        run_id = entry.get("run_id", "unknown")
                        provider = entry.get("provider", "unknown")

                        self._total_usd += cost
                        self._total_calls += 1
                        self._by_model[model] = self._by_model.get(model, 0.0) + cost
                        self._by_run[run_id] = self._by_run.get(run_id, 0.0) + cost
                        self._by_provider[provider] = self._by_provider.get(provider, 0.0) + cost
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass
