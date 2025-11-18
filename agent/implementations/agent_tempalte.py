import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from agent.base.agent_base import AgentBase, BaseAgentConfig,AgentState
from agent.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager

#log 설정
logger = logging.getLogger("agent_system")

@AgentRegistry.register("template_agent")
class TemplateAgent(AgentBase):
    """
    사용자 등록 자동화 Agent
    
    역할:
    - 사용자 등록/조회/수정/삭제 작업 처리
    - MCP 도구를 사용한 데이터베이스 작업
    
    필요한 구현:
    - get_system_prompt(): 도구 선택 프롬프트
    - validate_input(): 입력 검증
    """
    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        #사용 가능한 Tool 목록
        self.allowed_tools = 'ALL'

    # 전처리: 입력 데이터 
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """state에 messages가 있고, HumanMessage가 포함되어 있는지 확인"""
        messages = state.get("messages")
        
        if not messages or not isinstance(messages, list):
            logger.error(f"[{self.name}] 'messages' must be a non-empty list")
            return False
        
        if not any(isinstance(m, HumanMessage) for m in messages):
            logger.error(f"[{self.name}] No HumanMessage in messages")
            return False
        
        return True
    def pre_execute(self, state: AgentState) -> AgentState:
        """
            실행 전 전처리
    
            Override 가능: 구체적인 Agent에서 추가 전처리 구현
        """
        # 기본: 아무것도 하지 않음
        return state
    
        # =============================
    # 구체적인 Agent의 역할 정의
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        Agent의 역할 정의
        
        이 Prompt 하나로 Agent의 모든 행동 원칙이 결정됨
        """
        return """당신은 챗 Agent입니다. 현재까지의 작업을 요약하여 사용자에게 응답하세요."""