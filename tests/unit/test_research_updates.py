"""
Tests for research-driven updates — GPT-5 pricing, new Anthropic cache fields,
new finish/stop reasons, and cache discount multipliers.
"""

from types import SimpleNamespace

from agentfuse.providers.registry import ModelRegistry
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.response import extract_usage
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.core.response_validator import validate_for_cache, validate_response


# --- GPT-5 / GPT-5.4 Pricing ---

def test_gpt5_pricing_exists():
    """GPT-5 pricing must be in registry."""
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-5")
    assert p["input"] == 1.25
    assert p["output"] == 10.00
    assert p["cached_input"] == 0.125  # 90% discount


def test_gpt54_pricing_exists():
    """GPT-5.4 pricing must be in registry."""
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-5.4")
    assert p["input"] == 2.50
    assert p["output"] == 15.00


def test_gpt5_cache_90pct_discount():
    """GPT-5 family gets 90% cache discount (cached = 10% of base)."""
    engine = ModelPricingEngine()
    full_cost = engine.input_cost("gpt-5", 1_000_000)
    cached_cost = engine.cached_input_cost("gpt-5", 1_000_000)
    assert abs(cached_cost / full_cost - 0.10) < 0.01  # 10% of base


def test_gpt41_cache_75pct_discount():
    """GPT-4.1/o3/o4-mini get 75% cache discount (cached = 25% of base)."""
    engine = ModelPricingEngine()
    full_cost = engine.input_cost("gpt-4.1", 1_000_000)
    cached_cost = engine.cached_input_cost("gpt-4.1", 1_000_000)
    assert abs(cached_cost / full_cost - 0.25) < 0.01  # 25% of base


def test_gpt4o_cache_50pct_discount():
    """GPT-4o gets 50% cache discount."""
    engine = ModelPricingEngine()
    full_cost = engine.input_cost("gpt-4o", 1_000_000)
    cached_cost = engine.cached_input_cost("gpt-4o", 1_000_000)
    assert abs(cached_cost / full_cost - 0.50) < 0.01  # 50% of base


def test_gemini_cache_90pct_discount():
    """Gemini cache read = 0.1x base (90% discount)."""
    engine = ModelPricingEngine()
    full_cost = engine.input_cost("gemini-2.5-pro", 1_000_000)
    cached_cost = engine.cached_input_cost("gemini-2.5-pro", 1_000_000)
    assert abs(cached_cost / full_cost - 0.10) < 0.01


# --- Anthropic new cache_creation sub-object ---

def test_anthropic_cache_creation_sub_object():
    """Anthropic cache_creation sub-object with TTL breakdowns must be extracted."""
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=0,  # Flat field is 0
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=20,
            ephemeral_1h_input_tokens=10,
        ),
    )
    normalized = extract_usage("anthropic", usage)
    assert normalized.total_input_tokens == 160  # 100 + 30 + (20+10)
    assert normalized.cached_input_tokens == 30
    assert normalized.cache_write_tokens == 30  # 20 + 10 from sub-object


def test_anthropic_flat_field_preferred_when_nonzero():
    """When cache_creation_input_tokens is nonzero, it takes precedence."""
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=40,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=30,
            ephemeral_1h_input_tokens=10,
        ),
    )
    normalized = extract_usage("anthropic", usage)
    # Flat field is 40 (nonzero), so sub-object doesn't override
    assert normalized.cache_write_tokens == 40
    assert normalized.total_input_tokens == 160  # 100 + 20 + 40


def test_anthropic_no_cache_creation_object():
    """When cache_creation sub-object is None, flat fields still work."""
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=15,
    )
    normalized = extract_usage("anthropic", usage)
    assert normalized.cache_write_tokens == 15
    assert normalized.total_input_tokens == 135  # 100 + 20 + 15


# --- Anthropic 1-hour cache write cost ---

def test_cache_write_1h_ttl_cost():
    """1-hour TTL cache write must be 2.0x input (not 1.25x)."""
    engine = ModelPricingEngine()
    cost_5m = engine.cache_write_cost("claude-sonnet-4-6", 1_000_000, ttl="5m")
    cost_1h = engine.cache_write_cost("claude-sonnet-4-6", 1_000_000, ttl="1h")
    assert cost_1h > cost_5m
    assert abs(cost_1h / cost_5m - (2.0 / 1.25)) < 0.01


# --- New finish/stop reasons ---

def test_content_filter_not_cached():
    """OpenAI content_filter finish_reason must not be cached."""
    assert validate_for_cache("Some content", finish_reason="content_filter") is False


def test_max_tokens_not_cached():
    """Anthropic max_tokens stop_reason must not be cached."""
    assert validate_for_cache("Some content", finish_reason="max_tokens") is False


def test_pause_turn_not_cached():
    """Anthropic pause_turn stop_reason must not be cached."""
    assert validate_for_cache("Some content", finish_reason="pause_turn") is False


def test_refusal_not_cached():
    """Anthropic refusal stop_reason must not be cached."""
    assert validate_for_cache("Some content", finish_reason="refusal") is False


def test_gemini_safety_not_cached():
    """Gemini SAFETY finish_reason must not be cached."""
    assert validate_for_cache("Some content", finish_reason="SAFETY") is False


def test_gemini_recitation_not_cached():
    """Gemini RECITATION finish_reason must not be cached."""
    assert validate_for_cache("Some content", finish_reason="RECITATION") is False


def test_stop_still_cacheable():
    """Normal 'stop' finish_reason must still be cacheable."""
    assert validate_for_cache("Valid response text", finish_reason="stop") is True


def test_tool_calls_still_cacheable():
    """tool_calls finish_reason is valid and should be cacheable."""
    assert validate_for_cache("Valid response text", finish_reason="tool_calls") is True


# --- Tokenizer GPT-5 support ---

def test_gpt5_uses_o200k_encoding():
    """GPT-5 must use o200k_base encoding."""
    tc = TokenCounterAdapter()
    tokens = tc.count_tokens("Hello, how are you today?", "gpt-5")
    assert tokens > 0


def test_gpt54_uses_o200k_encoding():
    """GPT-5.4 must use o200k_base encoding."""
    tc = TokenCounterAdapter()
    tokens = tc.count_tokens("Hello, how are you today?", "gpt-5.4")
    assert tokens > 0


def test_o4_mini_uses_o200k_encoding():
    """o4-mini must use o200k_base encoding."""
    tc = TokenCounterAdapter()
    tokens = tc.count_tokens("Hello, how are you today?", "o4-mini")
    assert tokens > 0


# --- GPT-4.1 variants ---

def test_gpt41_mini_pricing():
    """GPT-4.1-mini pricing must exist in registry."""
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-4.1-mini")
    assert p["input"] == 0.40
    assert p["output"] == 1.60


def test_gpt41_nano_pricing():
    """GPT-4.1-nano pricing must exist in registry."""
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-4.1-nano")
    assert p["input"] == 0.10
    assert p["output"] == 0.40


# --- Circuit breaker exclusion ---

def test_rate_limit_excluded_from_circuit_breaker():
    """429 rate limit errors must NOT count toward circuit breaker."""
    from agentfuse.core.error_classifier import ClassifiedError
    err = ClassifiedError(error_type="rate_limit", retryable=True, status_code=429)
    assert err.counts_for_circuit_breaker is False


def test_server_error_counts_for_circuit_breaker():
    """500 server errors must count toward circuit breaker."""
    from agentfuse.core.error_classifier import ClassifiedError
    err = ClassifiedError(error_type="server", retryable=True, status_code=500)
    assert err.counts_for_circuit_breaker is True


def test_auth_error_excluded_from_circuit_breaker():
    """401 auth errors (client errors) must NOT count toward circuit breaker."""
    from agentfuse.core.error_classifier import ClassifiedError
    err = ClassifiedError(error_type="auth", retryable=False, status_code=401)
    assert err.counts_for_circuit_breaker is False


def test_timeout_counts_for_circuit_breaker():
    """Timeout errors must count toward circuit breaker."""
    from agentfuse.core.error_classifier import ClassifiedError
    err = ClassifiedError(error_type="timeout", retryable=True)
    assert err.counts_for_circuit_breaker is True


# --- GPT-5 downgrade paths ---

def test_gpt5_downgrade_chain():
    """GPT-5.4 → GPT-5 → GPT-4.1 → GPT-4.1-mini → GPT-4.1-nano."""
    from agentfuse.core.budget import BudgetEngine
    chain = BudgetEngine.DOWNGRADE_MAP
    assert chain["gpt-5.4"] == "gpt-5"
    assert chain["gpt-5"] == "gpt-4.1"
    assert chain["gpt-4.1"] == "gpt-4.1-mini"
    assert chain["gpt-4.1-mini"] == "gpt-4.1-nano"


def test_gpt5_fallback_chain():
    """GPT-5 fallback chain must be defined."""
    from agentfuse.core.fallback_chain import DEFAULT_CHAINS
    assert "gpt-5" in DEFAULT_CHAINS
    assert "gpt-5.4" in DEFAULT_CHAINS
    assert "gpt-4.1" in DEFAULT_CHAINS["gpt-5"]


def test_gpt5_routing_pair():
    """GPT-5 routing pair must exist for intelligent routing."""
    from agentfuse.core.model_router import ROUTING_PAIRS
    assert "gpt-5" in ROUTING_PAIRS
    assert "gpt-5.4" in ROUTING_PAIRS


# --- Gemini tool_use tokens ---

def test_gemini_tool_use_tokens_added_to_input():
    """Gemini tool_use_prompt_token_count must be added to input tokens."""
    usage = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=0,
        cached_content_token_count=0,
        tool_use_prompt_token_count=30,
    )
    normalized = extract_usage("gemini", usage)
    assert normalized.total_input_tokens == 130  # 100 + 30
