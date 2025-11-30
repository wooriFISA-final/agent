import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

logger = logging.getLogger("agent_system")


@AgentRegistry.register("template_agent")
class TemplateAgent(AgentBase):
    """
    Template Agent
    
    역할:
    - 새로운 Agent 구현을 위한 템플릿
    
    필요한 구현:
    - get_agent_role_prompt(): Agent 역할 정의
    - validate_input(): 입력 검증 (선택)
    - pre_execute(): 전처리 (선택)
    """
    
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        # 사용 가능한 Tool 목록 (Tool 이름 리스트 또는 'ALL')
        self.allowed_tools = []
        
        # 위임 가능한 Agent 목록
        self.allowed_agents = []

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
        """실행 전 전처리"""
        return state
    
    def get_agent_role_prompt(self) -> str:
        """
        Agent의 역할 정의
        
        이 Prompt 하나로 Agent의 모든 행동 원칙이 결정됨
        """
        return """당신은 챗 Agent입니다. 현재까지의 작업을 요약하여 사용자에게 응답하세요."""