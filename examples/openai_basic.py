"""AgentFuse: Basic OpenAI integration with budget enforcement and caching."""
import os
from agentfuse import completion, get_spend_report

# Requires: export OPENAI_API_KEY="sk-..."

# One function replaces openai.chat.completions.create()
# Automatically: caches, enforces budget, tracks cost, validates responses
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python? Answer in one sentence."}],
    budget_id="demo_run",
    budget_usd=1.00,
)

print("Response:", response.choices[0].message.content)
print(f"Spend so far: ${get_spend_report()['total_usd']:.6f}")

# Second identical call hits cache (free, instant)
response2 = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python? Answer in one sentence."}],
    budget_id="demo_run",
    budget_usd=1.00,
)

cached = getattr(response2, "_agentfuse_cache_hit", False)
print(f"Second call cached: {cached}")
print(f"Total spend: ${get_spend_report()['total_usd']:.6f}")
