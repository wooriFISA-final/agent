from typing import Any, Dict, List, Optional, Type
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.registry.agent_registry import AgentRegistry
from agents.config.base_config import BaseAgentConfig
from agents.config.base_config import AgentState, StateBuilder, ExecutionStatus
from graph.routing.router_base import RouterBase
from core.logging.logger import setup_logger

logger = setup_logger()


class GraphBuilder:
    """
    LangGraph 기반 Agent 그래프 빌더
    
    핵심 개선사항:
    1. AgentState를 기본 상태 스키마로 사용
    2. StateBuilder를 통한 상태 초기화 및 관리
    3. Agent 실행 전후 상태 추적 및 로깅
    """
    
    def __init__(self, state_schema: Optional[Type] = None):
        """
        GraphBuilder 초기화
        
        Args:
            state_schema: 상태 스키마 (기본값: AgentState)
                         - None 또는 AgentState 사용 권장
                         - 커스텀 스키마는 AgentState를 상속해야 함
        """
        # 기본값으로 AgentState 사용
        if state_schema is None:
            state_schema = AgentState
        
        self.state_schema = state_schema
        self.graph = StateGraph(state_schema)
        self.nodes: Dict[str, Any] = {}
        self.edges: List[tuple] = []
        self.conditional_edges: List[dict] = []
        
        logger.info(f"GraphBuilder initialized with schema: {state_schema.__name__}")
    
    def add_agent_node(
        self, 
        node_name: str, 
        agent_name: str,
        config: Optional[Dict] = None
    ) -> 'GraphBuilder':
        """
        Agent를 노드로 추가
        
        Args:
            node_name: 그래프 내 노드 이름
            agent_name: AgentRegistry에 등록된 Agent 이름
            config: Agent 설정 (BaseAgentConfig 필드)
            
        Returns:
            self (메서드 체이닝)
        """
        try:
            # Agent 클래스 조회
            agent_class = AgentRegistry.get(agent_name)
            
            # Agent Config 생성
            agent_config = BaseAgentConfig(
                name=node_name,
                **(config or {})
            )
            
            # Agent 인스턴스 생성
            agent_instance = agent_class(agent_config)
            
            # 래퍼 함수로 상태 추적 추가
            async def agent_wrapper(state: AgentState) -> AgentState:
                """
                Agent 실행 전후 상태 관리 래퍼
                
                - 실행 전: Agent 컨텍스트 업데이트
                - 실행 후: 실행 경로 기록, 에러 처리
                """
                logger.info(f"[Graph] Executing node: {node_name}")
                
                # 실행 전: Agent 컨텍스트 업데이트
                state = StateBuilder.update_agent_context(
                    state, 
                    node_name,
                    ExecutionStatus.RUNNING
                )
                
                try:
                    # Agent 실행
                    result_state = await agent_instance.run(state)
                    
                    logger.info(
                        f"[Graph] Node {node_name} completed with status: "
                        f"{result_state.get('status', 'unknown')}"
                    )
                    
                    return result_state
                    
                except Exception as e:
                    logger.error(f"[Graph] Node {node_name} failed: {e}")
                    
                    # 에러 기록
                    state = StateBuilder.add_error(state, e, node_name)
                    state = StateBuilder.finalize_state(
                        state, 
                        ExecutionStatus.FAILED
                    )
                    
                    return state
            
            # 노드 추가
            self.graph.add_node(node_name, agent_wrapper)
            self.nodes[node_name] = agent_instance
            
            logger.info(
                f"[Graph] Added agent node: {node_name} "
                f"(agent: {agent_name})"
            )
            
        except Exception as e:
            logger.error(f"[Graph] Failed to add agent node {node_name}: {e}")
            raise
        
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> 'GraphBuilder':
        """
        단순 엣지 추가 (무조건 from_node → to_node)
        
        Args:
            from_node: 시작 노드
            to_node: 도착 노드
            
        Returns:
            self (메서드 체이닝)
        """
        self.graph.add_edge(from_node, to_node)
        self.edges.append((from_node, to_node))
        
        logger.info(f"[Graph] Added edge: {from_node} → {to_node}")
        return self
    
    def add_conditional_edge(
        self,
        from_node: str,
        router: RouterBase,
        path_map: Dict[str, str]
    ) -> 'GraphBuilder':
        """
        조건부 엣지 추가 (Router 기반 분기)
        
        Args:
            from_node: 시작 노드
            router: 라우팅 로직을 담은 RouterBase 인스턴스
            path_map: {라우팅 결과: 다음 노드} 매핑
            
        Returns:
            self (메서드 체이닝)
        """
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
        
        logger.info(
            f"[Graph] Added conditional edge from {from_node} "
            f"with paths: {list(path_map.keys())}"
        )
        return self
    
    def set_entry_point(self, node_name: str) -> 'GraphBuilder':
        """
        그래프 시작 노드 설정
        
        Args:
            node_name: 시작 노드 이름
            
        Returns:
            self (메서드 체이닝)
        """
        self.graph.set_entry_point(node_name)
        logger.info(f"[Graph] Set entry point: {node_name}")
        return self
    
    def set_finish_point(self, node_name: str) -> 'GraphBuilder':
        """
        그래프 종료 노드 설정 (해당 노드 → END)
        
        Args:
            node_name: 종료 노드 이름
            
        Returns:
            self (메서드 체이닝)
        """
        self.graph.add_edge(node_name, END)
        logger.info(f"[Graph] Set finish point: {node_name} → END")
        return self
    
    def build(self, checkpointer: Optional[Any] = None):
        """
        그래프 컴파일
        
        Args:
            checkpointer: 체크포인터
                        - None이면 새 MemorySaver 생성 (테스트용, 권장하지 않음)
                        - 프로덕션에서는 전역 Checkpointer 전달 필수
                
        Returns:
            컴파일된 LangGraph 객체
        """
        # checkpointer가 None이면 경고와 함께 새로 생성
        if checkpointer is None:
            logger.warning(
                "[Graph] ⚠️  No checkpointer provided. Creating new MemorySaver. "
                "For persistent memory across requests, pass a global checkpointer instance."
            )
            checkpointer = MemorySaver()
        else:
            logger.info(f"[Graph] ✅ Using provided checkpointer: {type(checkpointer).__name__}")
        
        compiled_graph = self.graph.compile(checkpointer=checkpointer)
        
        logger.info(
            f"[Graph] Graph compiled successfully with "
            f"{len(self.nodes)} nodes, "
            f"{len(self.edges)} edges, "
            f"{len(self.conditional_edges)} conditional edges"
        )
        
        return compiled_graph
    
    def get_summary(self) -> Dict[str, Any]:
        """
        그래프 구조 요약 정보
        
        Returns:
            그래프 구조 정보 딕셔너리
        """
        return {
            "state_schema": self.state_schema.__name__,
            "nodes": list(self.nodes.keys()),
            "edges": self.edges,
            "conditional_edges": [
                {
                    "from": ce["from"],
                    "router": ce["router"].__class__.__name__,
                    "paths": list(ce["paths"].keys())
                }
                for ce in self.conditional_edges
            ],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "conditional_edge_count": len(self.conditional_edges)
        }
    
    def visualize_structure(self) -> str:
        """
        그래프 구조를 텍스트로 시각화
        
        Returns:
            그래프 구조 텍스트
        """
        lines = ["=" * 60]
        lines.append("GRAPH STRUCTURE")
        lines.append("=" * 60)
        
        # 노드
        lines.append("\n[Nodes]")
        for node_name, agent in self.nodes.items():
            lines.append(f"  • {node_name} ({agent.__class__.__name__})")
        
        # 단순 엣지
        if self.edges:
            lines.append("\n[Edges]")
            for from_node, to_node in self.edges:
                lines.append(f"  {from_node} → {to_node}")
        
        # 조건부 엣지
        if self.conditional_edges:
            lines.append("\n[Conditional Edges]")
            for ce in self.conditional_edges:
                lines.append(f"  {ce['from']} → (Router: {ce['router'].__class__.__name__})")
                for condition, target in ce['paths'].items():
                    lines.append(f"    - {condition} → {target}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# 사용 예시
# ============================================================================

async def example_usage():
    """GraphBuilder 사용 예시"""
    from langchain_core.messages import HumanMessage
    
    # 1. GraphBuilder 생성 (AgentState 자동 사용)
    builder = GraphBuilder()
    
    # 2. Agent 노드 추가
    builder.add_agent_node(
        node_name="user_reg",
        agent_name="user_registration",
        config={
            "model_name": "qwen3:8b",
            "max_iterations": 8,
            "timeout": 300
        }
    )
    
    # 3. 엣지 설정
    builder.set_entry_point("user_reg")
    builder.set_finish_point("user_reg")
    
    # 4. 그래프 빌드
    graph = builder.build()
    
    # 5. 그래프 구조 확인
    print(builder.visualize_structure())
    
    # 6. 초기 상태 생성
    initial_messages = [
        HumanMessage(content="김철수(25세) 등록하고 조회해줘")
    ]
    initial_state = StateBuilder.create_initial_state(
        messages=initial_messages,
        max_iterations=10
    )
    
    # 7. 그래프 실행
    config = {"configurable": {"thread_id": "session-123"}}
    result = await graph.ainvoke(initial_state, config=config)
    
    # 8. 결과 확인
    print(f"\n결과 상태: {result.get('status')}")
    print(f"실행 경로: {result.get('execution_path')}")
    print(f"반복 횟수: {result.get('iteration')}")
    print(f"Tool 호출: {len(result.get('tool_calls', []))}")
    
    return result


# ============================================================================
# 복잡한 그래프 예시 (멀티 Agent)
# ============================================================================

# def create_multi_agent_graph() -> GraphBuilder:
#     """
#     복잡한 멀티 Agent 그래프 생성 예시
    
#     흐름:
#     Entry → Classifier → (UserAgent | DataAgent | ReportAgent) → END
#     """
#     from graph.routing.intent_router import IntentRouter
    
#     builder = GraphBuilder()
    
#     # Agent 노드들 추가
#     builder.add_agent_node("classifier", "intent_classifier")
#     builder.add_agent_node("user_agent", "user_registration")
#     builder.add_agent_node("data_agent", "data_analysis")
#     builder.add_agent_node("report_agent", "report_generator")
    
#     # 조건부 라우팅
#     router = IntentRouter()
#     builder.add_conditional_edge(
#         from_node="classifier",
#         router=router,
#         path_map={
#             "user_management": "user_agent",
#             "data_analysis": "data_agent",
#             "report_generation": "report_agent"
#         }
#     )
    
#     # 각 Agent 실행 후 종료
#     builder.set_entry_point("classifier")
#     builder.set_finish_point("user_agent")
#     builder.set_finish_point("data_agent")
#     builder.set_finish_point("report_agent")
    
#     return builder


# if __name__ == "__main__":
#     import asyncio
    
#     # 단순 그래프 테스트
#     asyncio.run(example_usage())