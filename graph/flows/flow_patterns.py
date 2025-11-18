from typing import Callable, List, Dict, Any, Type, Optional
from graph.builder.graph_builder import GraphBuilder
from graph.routing.router_base import RouterBase
from langgraph.graph import END  # ✅ import 추가
import logging

logger = logging.getLogger(__name__)


class LinearFlow:
    """선형 워크플로우 빌더"""
    
    @staticmethod
    def create(
        agents: List[str], 
        state_schema: Type,
        agent_configs: Optional[Dict[str, Dict]] = None
    ):
        builder = GraphBuilder(state_schema)
        agent_configs = agent_configs or {}
        
        # 순차적으로 연결
        for i, agent in enumerate(agents):
            node_name = f"node_{i}"
            config = agent_configs.get(agent, {})
            builder.add_agent_node(node_name, agent, config)
            
            if i > 0:
                builder.add_edge(f"node_{i-1}", node_name)
        
        builder.set_entry_point("node_0")
        builder.set_finish_point(f"node_{len(agents)-1}")
        
        logger.info(f"✅ LinearFlow created with {len(agents)} agents")
        return builder.build()


class BranchingFlow:
    """분기 워크플로우 빌더"""
    
    @staticmethod
    def create(
        entry_agent: str,
        branches: Dict[str, List[str]],
        router: RouterBase,
        state_schema: Type,
        agent_configs: Optional[Dict[str, Dict]] = None
    ):
        builder = GraphBuilder(state_schema)
        agent_configs = agent_configs or {}
        
        # 진입점
        builder.add_agent_node("entry", entry_agent, agent_configs.get(entry_agent, {}))
        builder.set_entry_point("entry")
        
        # 분기별 체인 생성
        path_map = {}
        for branch_name, agents in branches.items():
            for i, agent in enumerate(agents):
                node_name = f"{branch_name}_{i}"
                config = agent_configs.get(agent, {})
                builder.add_agent_node(node_name, agent, config)
                
                if i > 0:
                    builder.add_edge(f"{branch_name}_{i-1}", node_name)
            
            path_map[branch_name] = f"{branch_name}_0"
        
        # 조건부 엣지
        builder.add_conditional_edge("entry", router, path_map)
        
        # 모든 분기의 끝을 종료점으로
        for branch_name, agents in branches.items():
            last_node = f"{branch_name}_{len(agents)-1}"
            builder.set_finish_point(last_node)
        
        logger.info(f"✅ BranchingFlow created with {len(branches)} branches")
        return builder.build()


class CyclicFlow:
    """순환 워크플로우 (재시도 로직)"""
    
    @staticmethod
    def create(
        agents: List[str],
        max_iterations: int,
        retry_condition_key: str,
        state_schema: Type,
        agent_configs: Optional[Dict[str, Dict]] = None
    ):
        builder = GraphBuilder(state_schema)
        agent_configs = agent_configs or {}
        
        # 노드 추가
        for i, agent in enumerate(agents):
            node_name = f"node_{i}"
            config = agent_configs.get(agent, {})
            builder.add_agent_node(node_name, agent, config)
            
            if i > 0:
                builder.add_edge(f"node_{i-1}", node_name)
        
        # 진입점
        builder.set_entry_point("node_0")
        
        # 재시도 라우터
        class IterationRouter(RouterBase):
            def __init__(self):
                super().__init__()
                self.iteration = 0
            
            def route(self, state: Dict[str, Any]) -> str:
                self.iteration += 1
                
                # 최대 반복 초과
                if self.iteration >= max_iterations:
                    logger.warning(f"Max iterations ({max_iterations}) reached")
                    return "end"
                
                # 성공 조건 확인
                if state.get(retry_condition_key, False):
                    return "success"
                
                # 재시도
                return "retry"
        
        router = IterationRouter()
        last_node = f"node_{len(agents)-1}"
        
        builder.add_conditional_edge(
            last_node,
            router,
            {
                "retry": "node_0",  # 처음으로
                "success": END,
                "end": END
            }
        )
        
        logger.info(f"✅ CyclicFlow created: {len(agents)} agents, max {max_iterations} iterations")
        return builder.build()