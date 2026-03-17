"""
CostAlertManager — configurable cost alerts with webhook delivery.

Monitors budget usage across runs and fires alerts at configurable
thresholds. Supports webhook URLs, callback functions, and structured
alert payloads.

This is what production systems need for cost governance — not just
per-run exceptions, but proactive alerts before budget is exhausted.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostAlert:
    """Structured cost alert payload."""
    alert_type: str           # "threshold", "spike", "anomaly"
    tenant_id: str
    run_id: str
    threshold_pct: float      # e.g. 0.50 for 50%
    current_pct: float        # actual percentage used
    spent_usd: float
    budget_usd: float
    model: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "tenant_id": self.tenant_id,
            "run_id": self.run_id,
            "threshold_pct": self.threshold_pct,
            "current_pct": round(self.current_pct, 4),
            "spent_usd": round(self.spent_usd, 6),
            "budget_usd": self.budget_usd,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class CostAlertManager:
    """
    Monitors budget usage and fires alerts at configurable thresholds.

    Usage:
        manager = CostAlertManager(
            thresholds=[0.50, 0.75, 0.90],
            webhook_url="https://hooks.slack.com/...",
        )
        manager.check(engine)  # call after each API response
    """

    def __init__(
        self,
        thresholds: list[float] = None,
        callback: Optional[Callable[[CostAlert], None]] = None,
        webhook_url: Optional[str] = None,
        tenant_id: str = "default",
    ):
        self._thresholds = sorted(thresholds or [0.50, 0.75, 0.90])
        self._callback = callback
        self._webhook_url = webhook_url
        self._tenant_id = tenant_id
        self._fired: set[tuple[str, float]] = set()  # (run_id, threshold) already fired
        self._lock = threading.Lock()

    def check(self, engine) -> Optional[CostAlert]:
        """Check if any threshold has been crossed and fire alert if so.

        Call this after every API response cost is recorded.
        Returns the alert if one was fired, None otherwise.
        """
        if not engine or engine.budget <= 0:
            return None

        pct = engine.spent / engine.budget if engine.budget > 0 else 0.0

        with self._lock:
            for threshold in self._thresholds:
                key = (engine.run_id, threshold)
                if pct >= threshold and key not in self._fired:
                    self._fired.add(key)
                    alert = CostAlert(
                        alert_type="threshold",
                        tenant_id=self._tenant_id,
                        run_id=engine.run_id,
                        threshold_pct=threshold,
                        current_pct=pct,
                        spent_usd=engine.spent,
                        budget_usd=engine.budget,
                        model=engine.model,
                    )
                    self._fire(alert)
                    return alert
        return None

    def _fire(self, alert: CostAlert):
        """Fire the alert via callback and/or webhook."""
        # Callback (sync, in-process)
        if self._callback:
            try:
                self._callback(alert)
            except Exception as e:
                logger.warning("Cost alert callback failed: %s", e)

        # Webhook (async, non-blocking)
        if self._webhook_url:
            thread = threading.Thread(
                target=self._send_webhook,
                args=(alert,),
                daemon=True,
            )
            thread.start()

    def _send_webhook(self, alert: CostAlert):
        """Send alert to webhook URL. Best-effort, non-blocking."""
        try:
            import httpx
            httpx.post(
                self._webhook_url,
                json=alert.to_dict(),
                timeout=5.0,
            )
        except Exception as e:
            logger.warning("Cost alert webhook failed: %s", e)

    def reset(self, run_id: Optional[str] = None):
        """Reset fired alerts for a run or all runs."""
        with self._lock:
            if run_id:
                self._fired = {k for k in self._fired if k[0] != run_id}
            else:
                self._fired.clear()
