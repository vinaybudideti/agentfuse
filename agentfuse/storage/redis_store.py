"""
RedisBudgetStore — atomic budget enforcement using Lua scripts.

Budgets stored as integer microdollars (× 1,000,000) for precision.
Key pattern: {agentfuse:run:{run_id}}:budget (hash tag for Redis Cluster).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lua script: atomic check-and-deduct
# KEYS[1] = budget key, ARGV[1] = cost in microdollars, ARGV[2] = optional TTL
# Returns: 1=success, 0=insufficient, -1=key missing, -2=invalid
CHECK_AND_DEDUCT_SCRIPT = """
local cost = tonumber(ARGV[1])
if not cost or cost < 0 then return -2 end

local remaining = tonumber(redis.call('GET', KEYS[1]))
if not remaining then return -1 end

if remaining < cost then return 0 end

redis.call('SET', KEYS[1], tostring(remaining - cost))
local ttl = tonumber(ARGV[2])
if ttl and ttl > 0 then redis.call('EXPIRE', KEYS[1], ttl) end
return 1
"""

# Lua script: reconcile estimated vs actual cost
# KEYS[1] = budget key, ARGV[1] = estimated microdollars, ARGV[2] = actual microdollars
RECONCILE_SCRIPT = """
local estimated = tonumber(ARGV[1])
local actual = tonumber(ARGV[2])
if not estimated or not actual then return -2 end

local remaining = tonumber(redis.call('GET', KEYS[1]))
if not remaining then return -1 end

local new_remaining = remaining + (estimated - actual)
redis.call('SET', KEYS[1], tostring(new_remaining))
return new_remaining
"""


def _to_microdollars(usd: float) -> int:
    """Convert USD to integer microdollars for precision."""
    return int(round(usd * 1_000_000))


def _from_microdollars(micro: int) -> float:
    """Convert microdollars back to USD."""
    return micro / 1_000_000


class RedisBudgetStore:
    """
    Atomic budget enforcement via Redis Lua scripts.
    Uses register_script for EVALSHA with automatic EVAL fallback.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        import redis
        self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self._check_script = self._redis.register_script(CHECK_AND_DEDUCT_SCRIPT)
        self._reconcile_script = self._redis.register_script(RECONCILE_SCRIPT)

    def _key(self, run_id: str) -> str:
        return f"{{agentfuse:run:{run_id}}}:budget"

    def create_run(self, run_id: str, budget_usd: float, ttl: int = 3600) -> bool:
        """Create a new budget run. Returns False if run already exists."""
        key = self._key(run_id)
        micro = _to_microdollars(budget_usd)
        result = self._redis.set(key, str(micro), nx=True, ex=ttl)
        return result is not None and result is not False

    def check_and_deduct(self, run_id: str, estimated_usd: float) -> int:
        """
        Atomically check and deduct budget.
        Returns: 1=success, 0=insufficient, -1=run not found, -2=invalid
        """
        key = self._key(run_id)
        micro = _to_microdollars(estimated_usd)
        result = self._check_script(keys=[key], args=[str(micro), "0"])
        return int(result)

    def reconcile(self, run_id: str, estimated_usd: float, actual_usd: float) -> float:
        """
        Adjust budget after actual cost is known.
        Returns new remaining balance in USD.
        """
        key = self._key(run_id)
        est_micro = _to_microdollars(estimated_usd)
        act_micro = _to_microdollars(actual_usd)
        result = self._reconcile_script(keys=[key], args=[str(est_micro), str(act_micro)])
        return _from_microdollars(int(result))

    def get_remaining(self, run_id: str) -> Optional[float]:
        """Get remaining budget in USD, or None if run not found/expired."""
        key = self._key(run_id)
        val = self._redis.get(key)
        if val is None:
            return None
        return _from_microdollars(int(val))
