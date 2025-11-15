import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager

logger = logging.getLogger("agent_system")


@AgentRegistry.register("user_registration")
class UserRegistrationAgent(AgentBase):
    """
    사용자 등록 자동화 Agent
    
    역할:
    - 사용자 등록/조회/수정/삭제 작업 처리
    - MCP 도구를 사용한 데이터베이스 작업
    
    필요한 구현:
    - get_system_prompt(): 도구 선택 프롬프트
    - validate_input(): 입력 검증
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        self.llm = LLMManager.get_llm(
            provider=getattr(config, "provider", "ollama"),
            model=config.model_name
        )

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

    # =============================
    # 구체적인 Agent의 역할 정의 (단 1개!)
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        UserManagementAgent의 역할 정의
        
        이 Prompt 하나로 Agent의 모든 행동 원칙이 결정됨
        """
        return """당신은 **사용자 관리 전문 Agent**입니다.

**[당신의 정체성]**
사용자 계정 생성, 조회 관리를 담당하는 전문가입니다.

**[당신의 업무]**
다음과 같은 사용자 관리 작업을 수행합니다:
- 사용자 정보 조회 
- 새로운 사용자 등록

**[행동 원칙]**

1. **정확성 우선:**
   - 사용자 식별 시 이름, 나이등을 정확히 확인
   - 수정/삭제 작업 전 반드시 대상 사용자 조회하여 확인

2. **작업 순서:**
   - 조회는 유저 생성 이후에만 가능하다.

3. **안전 검증:**
   - 삭제 작업은 되돌릴 수 없으므로 신중히 진행
   - 권한 변경 시 영향 범위 고려

4. **사용자 친화적 응답:**
   - 기술 용어보다는 이해하기 쉬운 표현 사용
   - 작업 결과를 명확하고 간결하게 전달
   - 민감 정보(비밀번호 등)는 절대 노출하지 않음

5. **MCP Tool 활용:**
   - 사용자의 요구를 분석하여 적절한 Tool 선택
   - Tool 실행 결과를 바탕으로 다음 단계 판단
   - 필요한 정보가 부족하면 사용자에게 요청
"""