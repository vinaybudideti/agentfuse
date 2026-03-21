"""AgentFuse: Session-based budget tracking (works with CrewAI or any framework)."""
from agentfuse import AgentSession

# Requires: export OPENAI_API_KEY="sk-..."

# AgentSession wraps budget + cost tracking + tool tracking in one context manager
with AgentSession("research_agent", budget_usd=3.00, model="gpt-4o-mini") as session:
    response = session.completion(
        messages=[{"role": "user", "content": "What are the latest AI trends?"}],
    )
    print("Response:", response.choices[0].message.content)

    # Track non-LLM tool costs too
    session.record_tool_call("web_search", cost=0.01)

receipt = session.get_receipt()
print(f"Total cost: ${receipt['total_cost_usd']:.6f}")
print(f"LLM cost: ${receipt['llm_cost_usd']:.6f}")
print(f"Tool cost: ${receipt['tool_cost_usd']:.6f}")
print(f"Cache hit rate: {receipt['cache_hit_rate']:.1%}")
