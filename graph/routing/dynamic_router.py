"""
Dynamic Router Module

Agent의 delegation 결정을 반영하여 동적으로 다음 노드를 결정하는 Router
"""

from typing import Literal
from agents.config.base_config import AgentState, ExecutionStatus
from graph.routing.router_base import RouterBase
from core.logging.logger import setup_logger

logger = setup_logger()


class DynamicRouter(RouterBase):
    """
    Agent의 delegation 결정을 반영하는 동적 Router
    
    사용 예:
        builder.add_conditional_edge(
            "agent_a",
            DynamicRouter(),
            {
                "agent_b": "agent_b",
                "agent_c": "agent_c",
                "END": END
            }
        )
    
    동작 방식:
        1. Agent가 DELEGATE 액션으로 next_agent를 지정하면 해당 Agent로 이동
        2. Agent가 SUCCESS/FAILED로 종료하면 END로 이동
        3. 그 외의 경우 기본값(END)으로 이동
    """
    
    def __init__(self, default_route: str = "END"):
        """
        Args:
            default_route: 기본 라우팅 경로 (기본값: "END")
        """
        self.default_route = default_route
        logger.info(f"[DynamicRouter] Initialized with default_route: {default_route}")
    
    def route(self, state: AgentState) -> str:
        """
        Agent의 실행 결과를 보고 다음 노드 결정
        
        우선순위:
        1. Agent가 명시적으로 지정한 next_agent (DELEGATE)
        2. 실행 상태 확인 (SUCCESS/FAILED/TIMEOUT → END, RESPONDING → 재진입)
        3. 기본값 (END)
        
        Args:
            state: 현재 Agent 실행 상태
            
        Returns:
            다음 노드 이름 ("agent_name" 또는 "END")
        """
        
        # 2. 실행 상태 확인
        status = state.get("status", ExecutionStatus.PENDING)
        
        if status == ExecutionStatus.RESPONDING:
            # 응답 완료 + 후처리 필요 → 같은 Agent 재진입
            current_agent = state.get("current_agent")
            logger.info(f"⚙️ [DynamicRouter] Status: RESPONDING → Re-entering {current_agent} for post-processing")
            return current_agent
        
        elif status == ExecutionStatus.SUCCESS:
            logger.info(f"✅ [DynamicRouter] Status: SUCCESS → END")
            return "END"
        
        elif status == ExecutionStatus.FAILED:
            logger.warning(f"❌ [DynamicRouter] Status: FAILED → END")
            return "END"
        
        elif status == ExecutionStatus.TIMEOUT:
            logger.warning(f"⏱️  [DynamicRouter] Status: TIMEOUT → END")
            return "END"
        
        elif status == ExecutionStatus.MAX_ITERATIONS:
            logger.warning(f"🔄 [DynamicRouter] Status: MAX_ITERATIONS → END")
            return "END"
        
        elif status == ExecutionStatus.RUNNING:
            # Agent가 작업 중간에 반환했지만 next_agent를 지정하지 않은 경우
            # 1. Agent의 delegation 확인
            next_agent = state.get("next_agent")
            if next_agent:
                logger.info(f"🔀 [DynamicRouter] Delegation detected → {next_agent}")
                delegation_reason = state.get("delegation_reason", "No reason provided")
                logger.debug(f"   Reason: {delegation_reason}")
                
                return next_agent
            
            logger.warning(f"⚠️  [DynamicRouter] Status: RUNNING but no next_agent → {self.default_route}")
            return self.default_route
        
        # 3. 기본값
        logger.info(f"➡️  [DynamicRouter] Default route → {self.default_route}")
        return self.default_route


class IntentBasedRouter(RouterBase):
    """
    사용자 의도 기반 Router (고급)
    
    Agent의 delegation + 메시지 분석을 결합하여 라우팅
    
    사용 예:
        builder.add_conditional_edge(
            "entry",
            IntentBasedRouter(),
            {
                "research": "research_agent",
                "user_mgmt": "user_management_agent",
                "data_analysis": "data_analysis_agent",
                "END": END
            }
        )
    """
    
    def route(self, state: AgentState) -> Literal["research", "user_mgmt", "data_analysis", "END"]:
        """
        다음 노드 결정 (의도 분석 포함)
        
        우선순위:
        1. Agent delegation (가장 우선)
        2. 실행 상태
        3. 메시지 기반 의도 분석 (폴백)
        
        Args:
            state: 현재 상태
            
        Returns:
            다음 노드 이름
        """
        # 1. Agent delegation 우선
        next_agent = state.get("next_agent")
        if next_agent:
            logger.info(f"🔀 [IntentRouter] Agent delegation → {next_agent}")
            state.pop("next_agent", None)
            state.pop("delegation_reason", None)
            
            # next_agent를 표준 형식으로 변환
            if "research" in next_agent.lower():
                return "research"
            elif "user" in next_agent.lower() or "management" in next_agent.lower():
                return "user_mgmt"
            elif "data" in next_agent.lower() or "analysis" in next_agent.lower():
                return "data_analysis"
            else:
                logger.warning(f"⚠️  Unknown agent: {next_agent}, routing to END")
                return "END"
        
        # 2. 실행 상태 확인
        status = state.get("status", ExecutionStatus.PENDING)
        if status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, 
                      ExecutionStatus.TIMEOUT, ExecutionStatus.MAX_ITERATIONS]:
            logger.info(f"[IntentRouter] Status {status} → END")
            return "END"
        
        # 3. 메시지 기반 의도 분석 (폴백)
        messages = state.get("messages", [])
        if not messages:
            logger.info("[IntentRouter] No messages → END")
            return "END"
        
        last_message = str(messages[-1].content).lower()
        
        # 키워드 기반 의도 분석
        if any(kw in last_message for kw in ["조사", "찾아", "검색", "알아봐"]):
            logger.info(f"🔍 [IntentRouter] Intent: research")
            return "research"
        
        elif any(kw in last_message for kw in ["사용자", "계정", "회원", "등록"]):
            logger.info(f"👤 [IntentRouter] Intent: user_mgmt")
            return "user_mgmt"
        
        elif any(kw in last_message for kw in ["분석", "데이터", "통계", "차트"]):
            logger.info(f"📊 [IntentRouter] Intent: data_analysis")
            return "data_analysis"
        
        logger.info("[IntentRouter] No intent matched → END")
        return "END"