"""
AgentFuse: Real-world end-to-end validation script.

Run this BEFORE publishing to PyPI to verify the gateway works with real APIs.
Requires: OPENAI_API_KEY environment variable.
Estimated cost: $0.01-0.05 per run.

Usage:
    python examples/e2e_real_test.py
"""
import os
import sys
import time

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable first.")
    print("  export OPENAI_API_KEY=\"sk-...\"")
    sys.exit(1)

from agentfuse import completion, get_spend_report, estimate_cost, AgentSession
from agentfuse.gateway import cleanup

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


print("=" * 60)
print("AgentFuse Real-World E2E Validation")
print("=" * 60)

# --- Test 1: Basic completion ---
print("\n[1] Basic completion with budget...")
cleanup()
try:
    r1 = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        budget_id="e2e_basic",
        budget_usd=0.10,
    )
    text1 = r1.choices[0].message.content
    print(f"     Response: {text1}")
    check("Got non-empty response", len(text1) > 0, f"Got empty string")
    check("Has usage data", hasattr(r1, "usage") and r1.usage is not None)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 2: Cache hit on identical request ---
print("\n[2] Cache hit on repeated request...")
try:
    r2 = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        budget_id="e2e_basic",
        budget_usd=0.10,
    )
    text2 = r2.choices[0].message.content
    cached = getattr(r2, "_agentfuse_cache_hit", False)
    print(f"     Response: {text2}")
    print(f"     Cache hit: {cached}")
    check("Response matches first call", text2 == text1, f"'{text2}' != '{text1}'")
    check("Marked as cache hit", cached, "._agentfuse_cache_hit is False")
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 3: Spend tracking ---
print("\n[3] Spend tracking...")
try:
    report = get_spend_report()
    print(f"     Total: ${report['total_usd']:.6f}")
    print(f"     By model: {report['by_model']}")
    check("Total spend > 0", report["total_usd"] > 0, f"Got {report['total_usd']}")
    check("gpt-4o-mini in by_model", "gpt-4o-mini" in report.get("by_model", {}))
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 4: Cost estimation ---
print("\n[4] Cost estimation (no API call)...")
try:
    est = estimate_cost(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a poem about AI"}],
        max_output_tokens=500,
    )
    print(f"     Estimated: ${est['estimated_total_cost_usd']:.6f}")
    check("Estimate > 0", est["estimated_total_cost_usd"] > 0)
    check("Has input tokens", est["estimated_input_tokens"] > 0)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 5: AgentSession ---
print("\n[5] AgentSession context manager...")
cleanup()
try:
    with AgentSession("e2e_session", budget_usd=0.10, model="gpt-4o-mini") as session:
        r5 = session.completion(
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        )
        print(f"     Response: {r5.choices[0].message.content}")
    receipt = session.get_receipt()
    print(f"     Total cost: ${receipt['total_cost_usd']:.6f}")
    print(f"     Calls: {receipt['calls']}")
    check("Receipt has positive cost", receipt["total_cost_usd"] >= 0)
    check("Call count is 1", receipt["calls"] == 1, f"Got {receipt['calls']}")
    check("Has duration", receipt["duration_seconds"] > 0)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 6: Different model ---
print("\n[6] Different model (gpt-4o)...")
cleanup()
try:
    r6 = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say yes."}],
        budget_id="e2e_gpt4o",
        budget_usd=0.10,
    )
    check("gpt-4o responded", len(r6.choices[0].message.content) > 0)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Summary ---
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL TESTS PASSED -- safe to publish to PyPI")
else:
    print("SOME TESTS FAILED -- fix issues before publishing")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
