# cyclic_flow.py
from typing import Callable, List
from graph.builder.graph_builder import GraphBuilder
from routing.router_base import RouterBase

class CyclicFlow:
    """순환 워크플로우 (재시도 로직)"""
    
    @staticmethod
    def create(
        agents: List[str],
        retry_condition: Callable,
        max_iterations: int,
        state_schema
    ):
        builder = GraphBuilder(state_schema)
        
        # 노드 추가
        for i, agent in enumerate(agents):
            builder.add_agent_node(f"node_{i}", agent)
        
        # 순환 로직 추가
        class RetryRouter(RouterBase):
            def __init__(self):
                super().__init__()
                self.iteration = 0
            
            def route(self, state):
                self.iteration += 1
                if self.iteration >= max_iterations:
                    return "end"
                elif retry_condition(state):
                    return "retry"
                else:
                    return "next"
        
        router = RetryRouter()
        builder.add_conditional_edge(
            "node_0",
            router,
            {"retry": "node_0", "next": "node_1", "end": END}
        )
        
        return builder.build()