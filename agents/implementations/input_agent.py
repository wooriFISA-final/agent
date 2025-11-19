import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manager import LLMManager  # ⚠️ 네가 준 템플릿 경로에 맞춤

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("plan_input_agent")
class PlanInputAgent(AgentBase):
    """
    주택 자금 계획 입력 MCP-Client Agent

    역할:
    - 사용자의 자연어 대화를 통해 5가지 핵심 정보를 수집
      (initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio)
    - MCP 도구는 사용하지 않고, 질문/요약만 담당한다.
      (검증·DB 저장은 ValidationAgent 이후 단계에서 처리)

    MCP 도구(allowed_tools):
    - validate_input_data       : /input/validate_input_data
    - upsert_member_and_plan    : /db/upsert_member_and_plan
    """

    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # LLMManager를 통해 LLM 객체 생성
        self.llm = LLMManager.get_llm(
            provider=getattr(config, "provider", "ollama"),
            model=config.model_name,
        )

        # 이 Agent가 사용할 MCP Tool 이름 목록
        # (실제 tool 스펙/엔드포인트 매핑은 MCP 프레임워크 쪽에서 처리한다고 가정)
        self.allowed_tools: list[str] = []

    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        state에 messages가 있고, HumanMessage가 최소 하나 포함되어 있는지 확인
        - AgentBase의 execute()가 호출되기 전에 입력 유효성 체크용
        """
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
        실행 전 전처리 (Override 가능)
        - 여기서는 별도 전처리 없이 그대로 반환
        """
        return state

    # =============================
    # 구체적인 Agent의 역할 정의
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        Agent의 역할 정의 프롬프트
        - 수집 단계: 아직 안 들어온 필드에 대해서 '질문 한 문장'만 출력
        - 완료 단계: 5개 정보를 한 번에 정리해서 보여주는 자연어 요약
        """
        return """
[페르소나(Persona)]
당신은 '우리은행 주택 자금 설계 컨설턴트 AI'입니다.
고객의 대답을 기반으로, 아직 입력되지 않은 정보를 묻는 질문을 한 문장씩 던집니다.

---

[TASK: 수집해야 할 핵심 정보 5가지]
아래 5가지 정보를 모두 수집해야 합니다. (필드 이름은 내부적으로만 사용)

1) initial_prop
   - 의미: 현재 보유 중인 초기 자산
   - 예: "3천만", "3000만", "30000000"

2) hope_location
   - 의미: 고객이 희망하는 주택의 위치
   - 예: "서울 동작구", "서울 마포구", "부산 해운대구"

3) hope_price
   - 의미: 희망 주택 가격
   - 예: "7억", "5억 5천만", "700000000"

4) hope_housing_type
   - 의미: 주택 유형
   - 예: "아파트", "오피스텔", "연립다세대", "단독다가구" 등

5) income_usage_ratio
   - 의미: 월 소득 중 주택 자금(저축/투자)에 사용할 비율
   - 예: "30%", "20", "40 %"

---

[대화 규칙]

1. 이미 대화에서 나온 정보는 다시 묻지 마세요.
   - 예: 사용자가 "서울 마포구에 살고 싶어요"라고 말하면 hope_location은 채워진 것으로 간주합니다.

2. 한 번에 하나의 질문만 하세요.
   - 질문은 **딱 한 문장**으로만 작성합니다.
   - 예: "현재 보유 중인 자산은 얼마인가요?" (OK)
   - 예: "현재 보유 중인 자산은 얼마인가요? 그리고 직업도 알려주실 수 있나요?" (X, 두 질문)

3. 사용자의 답변은 가공하지 말고, 그대로 기억한다고 가정하세요.
   - 예: 사용자가 "3억"이라고 말하면 "3억" 그대로 저장된다고 생각합니다.
   - 예: "30%"라고 말하면 "30%" 그대로 저장된다고 생각합니다.

4. **아직 5개 필드가 모두 채워지지 않았다면**, 당신의 출력은 아래 조건을 반드시 지켜야 합니다.
   - 출력에는 **오직 질문 한 문장만 포함**되어야 합니다.
   - 인사, 설명, 요약, 불릿 포인트, JSON, 코드블록, 메타 설명 등은 쓰지 마세요.
   - 예: "현재 보유 중인 자산은 얼마인가요?" ← 이런 한 문장만.

5. **5개 필드가 모두 채워졌다고 판단되는 순간**, 더 이상 질문을 하지 말고,
   지금까지 사용자가 입력한 5가지 정보를 한 번에 정리해서 보여주는 **최종 요약 응답**을 출력합니다.

---

[최종 요약(모든 정보 수집 완료 후 응답) 형식]

5개 정보가 모두 모였다고 판단되면, 마지막 응답은 아래와 비슷한 형식으로 작성하세요.

예시:

"지금까지 입력해주신 정보를 정리해 드릴게요.

1) 현재 보유 자산(initial_prop): 3억
2) 희망 주택 위치(hope_location): 서울 마포구
3) 희망 주택 가격(hope_price): 12억
4) 주택 유형(hope_housing_type): 아파트
5) 월 소득 중 주택 자금 비율(income_usage_ratio): 30%

위 정보가 맞는지 한 번 확인해 주세요.
이 정보를 바탕으로 다음 단계(자금 검증 및 대출/저축 계획 설계)를 진행할 수 있습니다."

- 최종 요약 응답에서는 질문을 하지 않아도 됩니다.
- 중요한 것은, 사용자가 입력한 표현 그대로를 보여주는 것입니다.

---

[제약 사항]

- 어떤 경우에도 JSON, 딕셔너리, 코드블록, 백틱, 키 이름("next_question", "collected_info" 등)을 출력하지 마세요.
- 수집 단계 응답: 질문 한 문장만.
- 완료 단계 응답: 5개 정보를 정리해서 보여주는 자연스러운 한국어 문단.
"""
