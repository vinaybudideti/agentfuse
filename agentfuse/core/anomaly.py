"""
CostAnomalyDetector — statistical anomaly detection for LLM costs.

Uses exponential moving average (EMA) + z-score to detect when a
single API call costs significantly more than the historical baseline.

This catches:
1. Accidental large-context calls (200K+ tokens)
2. Model configuration errors (using opus when haiku was intended)
3. Infinite loops generating massive output
4. Prompt injection causing excessive tool calls

Production systems need this because a single misconfigured call
can cost $50+ and go unnoticed until the monthly bill arrives.
"""

import math
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Report generated when a cost anomaly is detected."""
    run_id: str
    model: str
    cost_usd: float
    mean_cost: float
    std_dev: float
    z_score: float
    severity: str  # "warning", "critical"
    message: str


class CostAnomalyDetector:
    """
    Detects cost anomalies using exponential moving average and z-score.

    Maintains a running baseline of per-call costs per model. When a new
    cost is significantly above the baseline (z-score > threshold), an
    anomaly is reported.

    Uses EMA (exponential moving average) to adapt to gradual cost changes
    while still detecting sudden spikes.

    Usage:
        detector = CostAnomalyDetector(z_threshold=3.0)
        anomaly = detector.record("gpt-4o", 0.05, run_id="run_123")
        if anomaly:
            print(f"ANOMALY: {anomaly.message}")
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        critical_threshold: float = 5.0,
        ema_alpha: float = 0.1,
        min_samples: int = 5,
        callback: Optional[Callable[[AnomalyReport], None]] = None,
    ):
        """
        Args:
            z_threshold: Z-score above which a cost is flagged as warning
            critical_threshold: Z-score above which a cost is critical
            ema_alpha: EMA smoothing factor (0.1 = slow adapt, 0.5 = fast)
            min_samples: Minimum samples before anomaly detection activates
            callback: Optional callback for anomaly reports
        """
        self._z_threshold = z_threshold
        self._critical_threshold = critical_threshold
        self._alpha = ema_alpha
        self._min_samples = min_samples
        self._callback = callback
        self._lock = threading.Lock()

        # Per-model statistics: {model: {mean, var, count}}
        self._stats: dict[str, dict] = {}

    def record(self, model: str, cost_usd: float, run_id: str = "") -> Optional[AnomalyReport]:
        """
        Record a cost observation and check for anomalies.

        Returns AnomalyReport if anomaly detected, None otherwise.
        """
        if cost_usd <= 0:
            return None

        with self._lock:
            stats = self._get_or_create_stats(model)
            stats["count"] += 1

            if stats["count"] <= self._min_samples:
                # Still warming up — just accumulate
                self._update_ema(stats, cost_usd)
                return None

            # Check for anomaly
            z_score = self._z_score(stats, cost_usd)
            anomaly = None

            if z_score >= self._critical_threshold:
                anomaly = AnomalyReport(
                    run_id=run_id, model=model, cost_usd=cost_usd,
                    mean_cost=stats["mean"], std_dev=math.sqrt(max(stats["var"], 1e-12)),
                    z_score=z_score, severity="critical",
                    message=f"CRITICAL cost anomaly: ${cost_usd:.4f} is {z_score:.1f}x std dev above mean ${stats['mean']:.4f} for {model}",
                )
            elif z_score >= self._z_threshold:
                anomaly = AnomalyReport(
                    run_id=run_id, model=model, cost_usd=cost_usd,
                    mean_cost=stats["mean"], std_dev=math.sqrt(max(stats["var"], 1e-12)),
                    z_score=z_score, severity="warning",
                    message=f"Cost anomaly: ${cost_usd:.4f} is {z_score:.1f}x std dev above mean ${stats['mean']:.4f} for {model}",
                )

            # Always update EMA (even on anomalies)
            self._update_ema(stats, cost_usd)

            if anomaly and self._callback:
                try:
                    self._callback(anomaly)
                except Exception:
                    pass

            return anomaly

    def _get_or_create_stats(self, model: str) -> dict:
        if model not in self._stats:
            self._stats[model] = {"mean": 0.0, "var": 0.0, "count": 0}
        return self._stats[model]

    def _update_ema(self, stats: dict, cost: float):
        """Update exponential moving average of mean and variance."""
        if stats["count"] <= 1:
            stats["mean"] = cost
            stats["var"] = 0.0
            return

        alpha = self._alpha
        old_mean = stats["mean"]
        stats["mean"] = alpha * cost + (1 - alpha) * old_mean
        # Welford's online variance with EMA
        diff = cost - old_mean
        stats["var"] = (1 - alpha) * (stats["var"] + alpha * diff * diff)

    def _z_score(self, stats: dict, cost: float) -> float:
        """Calculate z-score of cost relative to EMA baseline."""
        std = math.sqrt(max(stats["var"], 1e-12))
        if std < 1e-9:
            return 0.0
        return (cost - stats["mean"]) / std

    def get_baseline(self, model: str) -> Optional[dict]:
        """Get current baseline statistics for a model."""
        with self._lock:
            stats = self._stats.get(model)
            if stats is None:
                return None
            return {
                "model": model,
                "mean_cost": stats["mean"],
                "std_dev": math.sqrt(max(stats["var"], 1e-12)),
                "samples": stats["count"],
            }

    def reset(self, model: Optional[str] = None):
        """Reset anomaly detection state."""
        with self._lock:
            if model:
                self._stats.pop(model, None)
            else:
                self._stats.clear()
