import pytest
import asyncio
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver

from agents.agent.base.agent_base import AgentBase, AgentState
from agents.agent.registry.agent_registry import AgentRegistry
from graph.factory import mk_graph
from agents.config.base_config import BaseAgentConfig

# Define a dummy agent for testing purposes
@AgentRegistry.register("TestUserRegistrationAgent")
class TestUserRegistrationAgent(AgentBase):
    def __init__(self, config: BaseAgentConfig):
        # Call super with a default name if not provided
        if not hasattr(config, 'name') or not config.name:
            config.name = "TestUserRegistrationAgent"
        super().__init__(config)
        self.allowed_tools = 'ALL'

    def get_agent_role_prompt(self) -> str:
        return "You are a test agent."

    async def run(self, state: AgentState) -> AgentState:
        # Simple run implementation for testing
        state["last_result"] = "Test agent executed"
        return state

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_mk_graph_from_yaml():
    """
    Tests if a graph can be successfully created from a YAML file.
    """
    # Ensure the dummy agent is registered before running the test
    assert "TestUserRegistrationAgent" in AgentRegistry.list_agents()

    # Path to the test YAML file
    yaml_path = Path(__file__).parent / "test_graph.yaml"
    assert yaml_path.exists()

    # Create a checkpointer
    checkpointer = MemorySaver()

    # Create the graph
    graph = mk_graph(str(yaml_path), checkpointer=checkpointer)

    # Assertions
    assert graph is not None, "Graph creation should not fail"

    # Check the structure of the compiled graph
    graph_dict = graph.get_graph().to_json()
    
    # In LangGraph, nodes are dictionaries, so we check for the node name
    assert "user_reg_node" in graph_dict["nodes"], "Node 'user_reg_node' should be in the graph"
    
    # Check entry point
    assert graph_dict["entry_point"] == "user_reg_node", "Entry point should be 'user_reg_node'"
