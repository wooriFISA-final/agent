import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

logger = logging.getLogger("agent_system")


@AgentRegistry.register("user_check_agent")
class UserCheckAgent(AgentBase):
    """
    사용자 조회 전문 Agent
    
    역할:
    - 사용자 조회 작업만 수행
    - 위임받으면 바로 조회하고 응답
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        self.allowed_tools = ["get_user"]
        
        # 다른 Agent로만 위임이 가능합니다. (자기 자신은 제외됩니다.)
        self.allowed_agents = ["user_creation"]

    def validate_input(self, state: Dict[str, Any]) -> bool:
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
        """UserCheckAgent의 역할 정의"""
        return """당신은 사용자 조회 전문 Agent입니다.

**[당신의 정체성]**
사용자 계정 조회를 담당하는 전문가입니다. 당신은 사용자의 요구사항중 계정 조회가 있을 경우 반드시 이 역할을 수행해야 합니다.

**[당신의 업무]**
1. 사용자 정보 조회 (get_user Tool 사용)
2. 조회 결과가 나오면 바로 사용자에게 응답

**[행동 원칙]**

1. **바로 작업 수행:**
   - 위임받았으면 바로 get_user Tool을 사용해서 조회
   - 추가 분석이나 위임 필요 없음
   - 조회 → 응답으로 끝!

2. **작업 흐름:**
   ```
   1단계: 조회 대상 파악 (이름, 나이)
   2단계: get_user Tool 실행
   3단계: 결과를 사용자에게 응답 (respond)
   ```
```
"""