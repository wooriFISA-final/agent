
import yaml
from agents.registry.agent_registry import AgentRegistry
from graph.builder.graph_builder import GraphBuilder
from core.logging.logger import setup_logger
from graph.schemas.state import LLMStateSchema

def mk_graph(yaml_path: str):
    """
    Dynamically builds a langgraph graph from a YAML configuration file.
    """
    logger = setup_logger()
    logger.info(f"🚀 Initializing LLM Graph from '{yaml_path}'...")

    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"❌ Graph configuration file not found at '{yaml_path}'")
        return None
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing YAML file '{yaml_path}': {e}")
        return None

    AgentRegistry.auto_discover("agents.implementations")
    registered_agents = AgentRegistry.list_agents()
    logger.info(f"✅ Registered agents: {registered_agents}")

    builder = GraphBuilder(LLMStateSchema)

    # Add nodes
    for node in config.get('nodes', []):
        if node['agent'] in registered_agents:
            builder.add_agent_node(node['name'], node['agent'], config=node.get('config', {}))
        else:
            logger.error(f"❌ Agent '{node['agent']}' for node '{node['name']}' is not registered.")
            return None

    # Add edges
    for edge in config.get('edges', []):
        builder.add_edge(edge['source'], edge['target'])

    # Set entry and finish points
    entry_point = config.get('entry_point')
    if entry_point:
        builder.set_entry_point(entry_point)
    else:
        logger.error("❌ 'entry_point' not defined in graph configuration.")
        return None

    finish_point = config.get('finish_point')
    if finish_point:
        builder.set_finish_point(finish_point)
    else:
        logger.error("❌ 'finish_point' not defined in graph configuration.")
        return None

    graph = builder.build()
    logger.info("✅ Graph successfully built and ready to use.")
    
    return graph