from typing import Any, Dict, List, Optional, Type
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.registry.agent_registry import AgentRegistry
from agents.config.base_config import BaseAgentConfig
from agents.config.base_config import AgentState, StateBuilder, ExecutionStatus
from graph.routing.router_base import RouterBase
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    BaseMessage
)


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
        """
        if state_schema is None:
            state_schema = AgentState
        
        self.state_schema = state_schema
        self.graph = StateGraph(state_schema)
        self.nodes: Dict[str, Any] = {}
        self.edges: List[tuple] = []
        self.conditional_edges: List[dict] = []
        
        logger.info(f"GraphBuilder initialized with schema: {state_schema.__name__}")
        
    @staticmethod
    def _convert_previous_system_to_human(messages: List[BaseMessage], previous_agent: str) -> List[BaseMessage]:
        """
        이전 에이전트의 SystemMessage를 HumanMessage로 변환
        
        Args:
            messages: 메시지 리스트
            previous_agent: 이전 에이전트 이름
            
        Returns:
            변환된 메시지 리스트
        """
        if not messages:
            return messages
        
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # SystemMessage → HumanMessage로 변환
                converted.append(HumanMessage(
                    content=f"[이전 에이전트 역할 - {previous_agent}]\n{msg.content}"
                ))
            else:
                converted.append(msg)
        
        return converted
    
    def add_agent_node(
        self, 
        node_name: str, 
        agent_name: str,
        config: Optional[Dict] = None
    ) -> 'GraphBuilder':
        """Agent를 노드로 추가"""
        try:
            from agents.config.agent_config_loader import AgentConfigLoader
            
            yaml_config = AgentConfigLoader.get_agent_config_from_current(agent_name)
            
            if yaml_config and not yaml_config.enabled:
                logger.warning(
                    f"⚠️  Skipping disabled agent: {agent_name} "
                    f"(enabled: false in agents.yaml)"
                )
                return self
            
            agent_class = AgentRegistry.get(agent_name)
            
            agent_config = BaseAgentConfig(
                name=node_name,
                **(config or {})
            )
            
            agent_instance = agent_class(agent_config)
            
            # ✅ 수정된 agent_wrapper
            async def agent_wrapper(state: AgentState) -> AgentState:
                """Agent 실행 전후 상태 관리 래퍼"""
                logger.info(f"[Graph] Executing node: {node_name}")
                
                # ========================================
                # 실행 전: 메시지 전처리
                # ========================================
                previous_agent = state.get("previous_agent", "")
                global_messages = state.get("global_messages", [])
                
                # 이전 에이전트가 있으면 SystemMessage를 HumanMessage로 변환
                if previous_agent and global_messages:
                    logger.info(f"[Graph] Converting SystemMessage from previous agent: {previous_agent}")
                    global_messages = GraphBuilder._convert_previous_system_to_human(
                        global_messages, 
                        previous_agent
                    )
                    state["global_messages"] = global_messages
                
                # Agent 컨텍스트 업데이트
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
                    state = StateBuilder.add_error(state, e, node_name)
                    state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
                    return state
            
            self.graph.add_node(node_name, agent_wrapper)
            self.nodes[node_name] = agent_instance
            
            logger.info(f"[Graph] Added agent node: {node_name} (agent: {agent_name})")
            
        except Exception as e:
            logger.error(f"[Graph] Failed to add agent node {node_name}: {e}")
            raise
        
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> 'GraphBuilder':
        """단순 엣지 추가"""
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
        
        logger.info(
            f"[Graph] Added conditional edge from {from_node} "
            f"with paths: {list(path_map.keys())}"
        )
        return self
    
    def set_entry_point(self, node_name: str) -> 'GraphBuilder':
        """그래프 시작 노드 설정"""
        self.graph.set_entry_point(node_name)
        logger.info(f"[Graph] Set entry point: {node_name}")
        return self
    
    def set_finish_point(self, node_name: str) -> 'GraphBuilder':
        """그래프 종료 노드 설정"""
        self.graph.add_edge(node_name, END)
        logger.info(f"[Graph] Set finish point: {node_name} → END")
        return self
    
    def build(self, checkpointer: Optional[Any] = None):
        """
        그래프 컴파일
        
        Args:
            checkpointer: 체크포인터 (None이면 새로 생성)
                
        Returns:
            컴파일된 LangGraph 객체
        """
        if checkpointer is None:
            logger.warning(
                "[Graph] ⚠️  No checkpointer provided. Creating new MemorySaver."
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
        """그래프 구조 요약 정보"""
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
        """그래프 구조를 텍스트로 시각화"""
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