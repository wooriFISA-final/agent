import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from agent.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agent.registry.agent_registry import AgentRegistry

logger = logging.getLogger("agent_system")


@AgentRegistry.register("user_create_agent")
class UserCreationAgent(AgentBase):
    """
    사용자 생성 전문 Agent
    
    역할:
    - 사용자 생성 작업 처리
    - 조회 요청 시 user_check Agent로 위임
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        
        # 사용 가능한 Tool 목록
        self.allowed_tools = ["create_user"]
        
        # ✅ 위임 가능한 Agent 목록 추가
        self.allowed_agents = ["user_check"]

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
        """UserCreationAgent의 역할 정의"""
        return """당신은 사용자 생성 전문 Agent입니다.

**[당신의 정체성]**
사용자 계정 생성을 담당하는 전문가입니다.

**[당신의 업무]**
- 새로운 사용자 등록

**[위임 규칙]**

다음 경우 user_check Agent에게 위임하세요:

1. **사용자 조회 요청**
   - 사용자가 "조회", "확인", "찾아" 등의 키워드 사용
   - 조건: 조회는 user_check Agent의 전문 분야
   - 액션: delegate
   - next_agent: "user_check"

2. **사용자 생성 후 조회 요청**
   - 생성 완료 후 "확인해줘", "조회해줘" 등의 후속 요청
   - 조건: 생성과 조회는 별도 Agent가 담당
   - 액션: delegate
   - next_agent: "user_check"

**[중요: 자기 자신에게 위임 금지]**

**[행동 원칙]**

1. **정확성 우선:**
   - 사용자 생성 시 이름, 나이를 정확히 확인

2. **역할 분리:**
   - 생성(create)만 담당
   - 조회(get)는 user_check에게 위임

3. **MCP Tool 활용:**
   - create_user Tool만 사용
   - 필요한 정보가 부족하면 사용자에게 요청
"""