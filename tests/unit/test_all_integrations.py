"""
Comprehensive test that ALL 6 framework integrations are importable and functional.
"""


def test_langchain_importable():
    from agentfuse.integrations.langchain import AgentFuseChatModel
    model = AgentFuseChatModel(budget=5.0)
    assert model._llm_type == "agentfuse"


def test_crewai_importable():
    from agentfuse.integrations.crewai import create_agentfuse_hooks
    before, after = create_agentfuse_hooks(budget=5.0)
    assert callable(before)
    assert callable(after)


def test_openai_agents_importable():
    from agentfuse.integrations.openai_agents import AgentFuseModelProvider
    provider = AgentFuseModelProvider(budget=5.0)
    assert provider is not None


def test_mcp_importable():
    from agentfuse.integrations.mcp import get_mcp_tool_definitions, handle_mcp_tool_call
    tools = get_mcp_tool_definitions()
    assert len(tools) >= 5


def test_langgraph_importable():
    from agentfuse.integrations.langgraph import AgentFuseNode, budget_guard
    node = AgentFuseNode(lambda s: s)
    assert node is not None


def test_pydantic_ai_importable():
    from agentfuse.integrations.pydantic_ai import wrap_pydantic_agent, get_agent_receipt
    assert callable(wrap_pydantic_agent)
    assert callable(get_agent_receipt)


def test_all_six_frameworks():
    """All 6 framework integrations must be importable."""
    import agentfuse.integrations.langchain
    import agentfuse.integrations.crewai
    import agentfuse.integrations.openai_agents
    import agentfuse.integrations.mcp
    import agentfuse.integrations.langgraph
    import agentfuse.integrations.pydantic_ai
