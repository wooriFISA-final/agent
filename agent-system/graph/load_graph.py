
from agents.registry.agent_registry import AgentRegistry
from graph.builder.graph_builder import GraphBuilder

def from_yaml(cls, yaml_path: str):
    """YAML에서 그래프 로드"""

    AgentRegistry.auto_discover("agents.implementations")
    registered_agents = AgentRegistry.list_agents()
    logger.info(f"✅ Registered agents: {registered_agents}")

    builder = GraphBuilder(LLMStateSchema)
    llm_agent_config = {"timeout": 120, "max_retries": 2}

    if "intent_classifier" in registered_agents:
        builder.add_agent_node("intent", "intent_classifier", config=llm_agent_config)
        builder.set_entry_point("intent")
        builder.set_finish_point("intent")
        graph = builder.build()
        logger.info("✅ Graph successfully built and ready to use.")
    else:
        logger.error("❌ Required agent 'intent_classifier' not found. Please add it.")
        graph = None

    builder = cls(config['state_schema'])
    
    # 노드 추가
    for node in config['nodes']:
        builder.add_agent_node(
            node['name'],
            node['agent'],
            node.get('config')
        )
    
    # 엣지 추가
    for edge in config.get('edges', []):
        builder.add_edge(edge['from'], edge['to'])
    
    # 조건부 엣지 추가
    for cedge in config.get('conditional_edges', []):
        router_class = globals()[cedge['router']]
        router = router_class(cedge.get('router_config', {}))
        builder.add_conditional_edge(
            cedge['from'],
            router,
            cedge['paths']
        )
    
    builder.set_entry_point(config['entry_point'])
    builder.set_finish_point(config['finish_point'])
    
    return builder