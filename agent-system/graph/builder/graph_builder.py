from typing import Any, Dict, List, Callable, Optional, Type, Union, TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from agents.registry.agent_registry import AgentRegistry
from graph.routing.router_base import RouterBase
from agents.base.state import AgentState
from agents.config.base_config import BaseAgentConfig
from mystatechema import MyStateSchema

from langgraph.checkpoint.memory import MemorySaver 
class GraphBuilder:
    """유연한 그래프 빌더"""
    
    def __init__(self, state_schema: Union[Type[BaseModel], Type[Dict]]):
        self.graph = StateGraph(state_schema)
        self.nodes: Dict[str, Any] = {}
        self.edges: List[tuple] = []
        self.conditional_edges: List[dict] = []
        
    def add_agent_node(
        self, 
        node_name: str, 
        agent_name: str,
        config: Optional[Dict] = None
    ):
        """Agent를 노드로 추가"""
        agent_class = AgentRegistry.get(agent_name)
        agent_config = BaseAgentConfig(name=node_name, **(config or {}))
        agent_instance = agent_class(agent_config)
        
        self.graph.add_node(node_name, agent_instance.run)
        self.nodes[node_name] = agent_instance
        return self
    
    def add_edge(self, from_node: str, to_node: str):
        """단순 엣지 추가"""
        self.graph.add_edge(from_node, to_node)
        self.edges.append((from_node, to_node))
        return self
    
    def add_conditional_edge(
        self,
        from_node: str,
        router: RouterBase,
        path_map: Dict[str, str]
    ):
        """조건부 엣지 추가"""
        self.graph.add_conditional_edges(
            from_node,
            router.route,
            path_map
        )
        self.conditional_edges.append({
            "from": from_node,
            "router": router,
            "paths": path_map
        })
        return self
    
    def set_entry_point(self, node_name: str):
        """시작 노드 설정"""
        self.graph.set_entry_point(node_name)
        return self
    
    def set_finish_point(self, node_name: str):
        """종료 노드 설정"""
        self.graph.add_edge(node_name, END)
        return self
    
    def build(self):
        """그래프 컴파일"""
        # 메모리 관리 필요
        return self.graph.compile(checkpointer=MemorySaver())
    
    def visualize(self, output_path: str = "graph.png"):
        """그래프 시각화"""
        compiled = self.build()
        # Mermaid 또는 graphviz로 시각화
        pass
    
    # @classmethod
    # def from_yaml(cls, yaml_path: str):
    #     """YAML에서 그래프 로드"""
    #     import yaml
    #     with open(yaml_path, 'r') as f:
    #         config = yaml.safe_load(f)
        
    #     builder = cls(config['state_schema'])
        
    #     # 노드 추가
    #     for node in config['nodes']:
    #         builder.add_agent_node(
    #             node['name'],
    #             node['agent'],
    #             node.get('config')
    #         )
        
    #     # 엣지 추가
    #     for edge in config.get('edges', []):
    #         builder.add_edge(edge['from'], edge['to'])
        
    #     # 조건부 엣지 추가
    #     for cedge in config.get('conditional_edges', []):
    #         router_class = globals()[cedge['router']]
    #         router = router_class(cedge.get('router_config', {}))
    #         builder.add_conditional_edge(
    #             cedge['from'],
    #             router,
    #             cedge['paths']
    #         )
        
    #     builder.set_entry_point(config['entry_point'])
    #     builder.set_finish_point(config['finish_point'])
        
    #     return builder