"""
Graph Factory Module
This module reads a graph structure from a YAML file and uses a GraphBuilder
to create a compiled LangGraph instance.
"""
from typing import Optional, Any, Dict, List, Union
import yaml
from pathlib import Path
from langgraph.checkpoint.base import BaseCheckpointSaver

from graph.builder.graph_builder import GraphBuilder
from graph.routing.router_registry import RouterRegistry
from core.logging.logger import setup_logger

logger = setup_logger()


def mk_graph(yaml_path: str, checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    Creates an agent graph from a YAML configuration file.

    Args:
        yaml_path: The path to the YAML configuration file.
        checkpointer: An optional checkpointer instance for persisting graph state.
                      If None, a new MemorySaver will be used (for testing).

    Returns:
        A compiled LangGraph object, or None if creation fails.
    """
    try:
        config = _load_yaml_config(yaml_path)
        if not config:
            return None

        builder = GraphBuilder()

        # 1. Build Nodes
        _build_nodes(builder, config.get("nodes", []))

        # 2. Parse Edges Configuration
        # YAML 구조가 리스트(구버전)인지 딕셔너리(신버전: direct/conditional)인지 확인
        edges_config = config.get("edges", {})
        
        direct_edges = []
        conditional_edges = []

        if isinstance(edges_config, list):
            # 기존 방식: edges가 리스트인 경우 모두 direct로 가정
            direct_edges = edges_config
        elif isinstance(edges_config, dict):
            # 신규 방식: direct와 conditional 분리
            direct_edges = edges_config.get("direct", [])
            conditional_edges = edges_config.get("conditional", [])
        else:
            logger.warning(f"Unknown 'edges' format in YAML: {type(edges_config)}")

        # 3. Build Direct Edges
        # None이 들어올 경우를 대비해 빈 리스트 처리
        if direct_edges:
            _build_edges(builder, direct_edges)

        # 4. Build Conditional Edges
        # 사용자가 YAML에서 리스트(-)를 빼먹었을 경우 단일 딕셔너리로 들어올 수 있음 -> 리스트로 변환
        if isinstance(conditional_edges, dict):
            conditional_edges = [conditional_edges]
        
        if conditional_edges:
            _build_conditional_edges(builder, conditional_edges)

        # 5. Set Entry and Finish Points
        _set_entry_and_finish_points(builder, config)

        logger.info("Building graph...")
        if not checkpointer:
            logger.warning("No checkpointer provided. Using MemorySaver (not for production).")
        
        graph = builder.build(checkpointer=checkpointer)
        
        logger.info("Graph built successfully. Visualizing structure:")
        try:
            logger.info("\n" + builder.visualize_structure())
        except Exception:
            logger.info("(Visualization skipped)")
        
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create graph from YAML '{yaml_path}': {e}", exc_info=True)
        return None


def _build_nodes(builder: GraphBuilder, nodes: List[Dict[str, Any]]):
    """Adds nodes to the graph builder from the configuration."""
    if not nodes:
        raise ValueError("No nodes defined in YAML configuration.")
    
    for node_config in nodes:
        node_name = node_config.get("name")
        agent_name = node_config.get("agent")
        
        if not node_name or not agent_name:
            logger.warning(f"Skipping invalid node definition: {node_config}")
            continue
        
        builder.add_agent_node(
            node_name=node_name,
            agent_name=agent_name,
            config=node_config.get("config", {})
        )
        logger.info(f"Added node: {node_name} (agent: {agent_name})")


def _build_edges(builder: GraphBuilder, edges: List[Dict[str, str]]):
    """Adds standard edges to the graph builder."""
    for edge_config in edges:
        from_node = edge_config.get("from")
        to_node = edge_config.get("to")
        
        if not from_node or not to_node:
            logger.warning(f"Skipping invalid edge definition: {edge_config}")
            continue
            
        builder.add_edge(from_node, to_node)
        logger.info(f"Added edge: {from_node} -> {to_node}")


def _build_conditional_edges(builder: GraphBuilder, conditional_edges: List[Dict[str, Any]]):
    """Adds conditional edges to the graph builder."""
    for edge_config in conditional_edges:
        from_node = edge_config.get("from")
        router_class_name = edge_config.get("router")
        path_map = edge_config.get("paths", {})
        
        if not from_node or not router_class_name or not path_map:
            logger.warning(f"Skipping invalid conditional edge: {edge_config}")
            continue
        
        try:
            # Use the registry to get the router class
            router_class = RouterRegistry.get(router_class_name)
            router_instance = router_class() # Instantiate the router
            
            builder.add_conditional_edge(
                from_node=from_node,
                router=router_instance,
                path_map=path_map
            )
            logger.info(f"Added conditional edge from {from_node} using {router_class_name}")
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to create or add conditional edge for router '{router_class_name}': {e}")
            # Continue building the rest of the graph
            continue


def _set_entry_and_finish_points(builder: GraphBuilder, config: Dict[str, Any]):
    """Sets the entry and finish points for the graph."""
    entry_point = config.get("entry_point")
    if entry_point:
        builder.set_entry_point(entry_point)
        logger.info(f"Set entry point: {entry_point}")
    else:
        logger.warning("No explicit entry_point defined in YAML. LangGraph will use the first node added.")

    finish_points = config.get("finish_points", [])
    for finish_point in finish_points:
        builder.set_finish_point(finish_point)
        logger.info(f"Set finish point: {finish_point}")


def _load_yaml_config(yaml_path: str) -> Optional[Dict[str, Any]]:
    """Loads and parses the YAML configuration file."""
    path = Path(yaml_path)
    if not path.exists():
        logger.error(f"YAML file not found: {yaml_path}")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded YAML config from: {yaml_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {yaml_path}: {e}")
        return None