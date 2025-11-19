# linear_flow.py
from typing import List
from graph.builder.graph_builder import GraphBuilder


class LinearFlow:
    """선형 워크플로우 빌더"""
    
    @staticmethod
    def create(agents: List[str], state_schema):
        builder = GraphBuilder(state_schema)
        
        # 순차적으로 연결
        for i, agent in enumerate(agents):
            builder.add_agent_node(f"node_{i}", agent)
            if i > 0:
                builder.add_edge(f"node_{i-1}", f"node_{i}")
        
        builder.set_entry_point("node_0")
        builder.set_finish_point(f"node_{len(agents)-1}")
        
        return builder.build()