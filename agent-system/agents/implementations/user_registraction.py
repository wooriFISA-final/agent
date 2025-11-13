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

    def get_system_prompt(
        self,
        tool_names: List[str],
        messages: List
    ) -> SystemMessage:
        """
        사용자 등록 Agent의 시스템 프롬프트
        
        - MCP 도구 목록 제공
        - JSON 출력 형식 지정
        - 도구 선택 가이드라인
        """
        # 이전 도구 실행 결과가 있는지 확인
        has_previous_result = any("Tool Result:" in str(m.content) for m in messages)
        previous_result_note = """
⚠️ 이전 단계에서 MCP tool 실행 결과가 존재합니다.
해당 결과를 참고하여 다음 작업을 결정하거나 최종 응답을 생성하세요.
""" if has_previous_result else ""

        prompt_content = f"""당신은 사용자 등록 및 관리를 담당하는 MCP 에이전트입니다.

**역할:**
- 사용자 메시지를 분석하여 적합한 MCP 도구를 선택
- 도구 실행에 필요한 인자를 JSON 형식으로 생성

**사용 가능한 MCP 도구:**
{', '.join(tool_names)}

**중요한 규칙:**
1. 유저 조회는 사용자의 **이름(name)**으로 수행합니다
2. 응답은 반드시 아래 JSON 형식으로만 작성하세요
3. Markdown 백틱(```)은 사용하지 마세요
4. <think> 태그 안에 추론 과정을 작성할 수 있습니다

**출력 형식:**
{{
    "tool": "<사용할 MCP 도구 이름>",
    "arguments": {{
        "arg1": "값1",
        "arg2": "값2"
    }}
}}

{previous_result_note}

**예시:**
사용자: "홍길동을 조회해줘"
→ {{"tool": "get_user", "arguments": {{"name": "홍길동"}}}}

사용자: "이름은 김철수, 나이는 30, 이메일은 kim@example.com으로 등록해줘"
→ {{"tool": "create_user", "arguments": {{"name": "김철수", "age": 30, "email": "kim@example.com"}}}}
"""
        
        return SystemMessage(content=prompt_content)

    # 기본 process_tool_result()를 그대로 사용
    # 커스터마이징이 필요하면 오버라이드 가능