# branching_flow.py
from typing import Dict, List
from graph.builder.graph_builder import GraphBuilder
from routing.router_base import RouterBase

class BranchingFlow:
    """분기 워크플로우 빌더"""
    
    @staticmethod
    def create(
        entry_agent: str,
        branches: Dict[str, List[str]],
        router: RouterBase,
        state_schema
    ):
        builder = GraphBuilder(state_schema)
        
        # 진입점
        builder.add_agent_node("entry", entry_agent)
        builder.set_entry_point("entry")
        
        # 분기별 체인 생성
        path_map = {}
        for branch_name, agents in branches.items():
            for i, agent in enumerate(agents):
                node_name = f"{branch_name}_{i}"
                builder.add_agent_node(node_name, agent)
                if i > 0:
                    builder.add_edge(f"{branch_name}_{i-1}", node_name)
            
            path_map[branch_name] = f"{branch_name}_0"
        
        # 조건부 엣지
        builder.add_conditional_edge("entry", router, path_map)
        
        return builder.build()