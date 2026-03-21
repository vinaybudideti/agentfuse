"""AgentFuse: LangChain integration with budget-enforced chat model."""
from agentfuse.integrations.langchain import AgentFuseChatModel

# Requires: pip install langchain-core
# Requires: export OPENAI_API_KEY="sk-..."

# Wrap any LangChain-compatible model with budget enforcement + caching
model = AgentFuseChatModel(
    inner_model_name="gpt-4o-mini",
    budget_usd=5.00,
    run_id="langchain_demo",
)

# Use in any LangChain chain, agent, or direct invocation
from langchain_core.messages import HumanMessage
response = model.invoke([HumanMessage(content="Explain quantum computing briefly")])
print("Response:", response.content)
