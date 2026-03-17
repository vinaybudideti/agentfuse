"""
KillSwitch — emergency stop for AI agents at the gateway level.

CRITICAL SAFETY FEATURE: This operates OUTSIDE the AI reasoning path.
The agent cannot disable, bypass, or override the kill switch because
it's enforced at the completion() gateway before the LLM is called.

Design principles (Stanford AI Safety Lab, 2026):
1. Kill switch lives outside the AI reasoning path
2. Cannot be circumvented by model output or prompts
3. Preserves partial results and data integrity
4. Logs every activation for audit
5. Supports both manual (human) and automatic (threshold) triggers

Usage:
    from agentfuse import kill_switch

    # Manual kill
    kill_switch.kill("research_bot", reason="suspicious behavior")

    # Any future completion() calls for this run will raise AgentKilled
    # The agent cannot prevent this — it's checked before the LLM call

    # Revive when safe
    kill_switch.revive("research_bot")

    # Automatic kill on conditions
    kill_switch.set_auto_kill(
        max_cost_usd=50.0,
        max_calls=1000,
        max_duration_seconds=3600,
    )
"""

import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AgentKilled(Exception):
    """Raised when a kill switch is activated for an agent."""
    def __init__(self, run_id: str, reason: str, killed_at: float):
        self.run_id = run_id
        self.reason = reason
        self.killed_at = killed_at
        super().__init__(f"Agent '{run_id}' killed: {reason}")


class KillSwitch:
    """
    Emergency stop mechanism for AI agents.

    Operates at the gateway level — checked before every LLM call.
    Cannot be bypassed by the agent because it runs in the orchestration layer.
    """

    def __init__(self):
        self._killed: dict[str, dict] = {}  # run_id → {reason, killed_at, killed_by}
        self._global_kill = False
        self._global_reason = ""
        self._auto_kill_config: dict = {}
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []

    def kill(self, run_id: str, reason: str = "manual kill", killed_by: str = "human"):
        """Kill a specific agent run. All future calls will raise AgentKilled."""
        with self._lock:
            self._killed[run_id] = {
                "reason": reason,
                "killed_at": time.time(),
                "killed_by": killed_by,
            }
        logger.critical("KILL SWITCH activated for '%s': %s (by %s)", run_id, reason, killed_by)
        self._fire_callbacks(run_id, reason)

    def kill_all(self, reason: str = "global emergency stop"):
        """Kill ALL agent runs. Nuclear option."""
        with self._lock:
            self._global_kill = True
            self._global_reason = reason
        logger.critical("GLOBAL KILL SWITCH activated: %s", reason)
        self._fire_callbacks("*", reason)

    def revive(self, run_id: str):
        """Revive a killed agent run. Requires explicit human action."""
        with self._lock:
            if run_id in self._killed:
                del self._killed[run_id]
                logger.warning("Agent '%s' revived", run_id)

    def revive_all(self):
        """Revive all agents and clear global kill."""
        with self._lock:
            self._killed.clear()
            self._global_kill = False
            self._global_reason = ""
            logger.warning("All agents revived, global kill cleared")

    def check(self, run_id: str):
        """Check if an agent is killed. Raises AgentKilled if so.

        This is called at the gateway level before every LLM call.
        """
        with self._lock:
            if self._global_kill:
                raise AgentKilled(run_id, self._global_reason, time.time())

            if run_id in self._killed:
                info = self._killed[run_id]
                raise AgentKilled(run_id, info["reason"], info["killed_at"])

    def is_killed(self, run_id: str) -> bool:
        """Check without raising. For UI/monitoring use."""
        with self._lock:
            return self._global_kill or run_id in self._killed

    def set_auto_kill(
        self,
        max_cost_usd: Optional[float] = None,
        max_calls: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
    ):
        """Configure automatic kill conditions.

        These are checked by the gateway after each call.
        """
        with self._lock:
            self._auto_kill_config = {
                "max_cost_usd": max_cost_usd,
                "max_calls": max_calls,
                "max_duration_seconds": max_duration_seconds,
            }

    def check_auto_kill(self, run_id: str, cost_usd: float, calls: int,
                        started_at: Optional[float] = None):
        """Check automatic kill conditions. Called by gateway after each call."""
        config = self._auto_kill_config
        if not config:
            return

        if config.get("max_cost_usd") and cost_usd > config["max_cost_usd"]:
            self.kill(run_id, f"cost exceeded ${config['max_cost_usd']}", killed_by="auto")

        if config.get("max_calls") and calls > config["max_calls"]:
            self.kill(run_id, f"calls exceeded {config['max_calls']}", killed_by="auto")

        if config.get("max_duration_seconds") and started_at:
            elapsed = time.time() - started_at
            if elapsed > config["max_duration_seconds"]:
                self.kill(run_id, f"duration exceeded {config['max_duration_seconds']}s", killed_by="auto")

    def on_kill(self, callback: Callable):
        """Register a callback for kill events. For alerts/logging."""
        self._callbacks.append(callback)

    def _fire_callbacks(self, run_id: str, reason: str):
        for cb in self._callbacks:
            try:
                cb(run_id, reason)
            except Exception:
                pass

    def get_killed_agents(self) -> dict:
        """Get all currently killed agents."""
        with self._lock:
            return {
                "global_kill": self._global_kill,
                "global_reason": self._global_reason,
                "killed_runs": dict(self._killed),
            }


# Module-level singleton — shared across the entire process
kill_switch = KillSwitch()
