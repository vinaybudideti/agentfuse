"""
Loop 27 — Redis store key format tests (no Redis needed).
"""

from agentfuse.storage.redis_store import _to_microdollars, _from_microdollars


def test_microdollar_conversion_roundtrip():
    """USD → microdollars → USD must round-trip correctly."""
    assert _to_microdollars(1.0) == 1_000_000
    assert _to_microdollars(0.001) == 1_000
    assert _to_microdollars(0.0) == 0
    assert _from_microdollars(1_000_000) == 1.0
    assert _from_microdollars(1_000) == 0.001


def test_microdollar_precision():
    """Microdollar precision must handle typical API costs."""
    # $0.0025 (GPT-4o input cost for 1K tokens)
    micro = _to_microdollars(0.0025)
    assert micro == 2_500
    assert _from_microdollars(micro) == 0.0025


def test_key_format_has_hash_tag():
    """Redis key must include hash tag {run_id} for cluster compatibility."""
    from agentfuse.storage.redis_store import RedisBudgetStore
    # We can't create a full store without Redis, but we can test the key method
    # by accessing the class method signature
    # The key format should be: {agentfuse:run:{run_id}}:budget
    # Let's verify the format string
    import inspect
    source = inspect.getsource(RedisBudgetStore._key)
    assert "{agentfuse:run:" in source
    assert "}:budget" in source
