import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manager import LLMManager  # ✅ 템플릿 경로에 맞춤

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("validation_agent")
class ValidationAgent(AgentBase):
    """
    주택 자금 계획 검증 MCP-Client Agent

    역할:
    - PlanInputAgent 또는 사용자 대화로부터 전달된 주택 자금 계획 정보를 검증
    - MCP 도구를 사용해:
      1) 입력값 검증·정규화 (/input/validate_input_data)
      2) 지역·주택유형 평균 시세 조회 (/db/get_market_price)
      3) 검증된 값 DB 저장 (/db/upsert_member_and_plan)
    - 시세 대비 너무 무리한 계획이면 경고 메시지를 생성하고, 다시 입력을 요청

    MCP 도구(allowed_tools):
    - validate_input_data    : /input/validate_input_data
    - get_market_price       : /db/get_market_price
    - upsert_member_and_plan : /db/upsert_member_and_plan
    """

    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # LLMManager를 통해 LLM 객체 생성 (논리적 설명/경고 문구용)
        self.llm = LLMManager.get_llm(
            provider=getattr(config, "provider", "ollama"),
            model=config.model_name,
        )

        # 이 Agent가 사용할 MCP Tool 이름 목록
        # (실제 tool 스펙/엔드포인트 매핑은 MCP 프레임워크 쪽에서 처리된다고 가정)
        self.allowed_tools = [
            "validate_input_data",
            "get_market_price",
            "upsert_member_and_plan",
        ]

    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        ValidationAgent 실행 전 입력 검증.

        이 에이전트는 보통 다음 정보가 state에 있을 때 호출된다고 가정합니다.
        - state["messages"]        : 대화 메시지 리스트
        - state["extracted_info"]  : PlanInputAgent 등이 수집한 raw 입력 dict (선택)

        기본적으로:
        - messages 리스트가 존재하고
        - HumanMessage 가 최소 하나 포함되어 있으면 유효하다고 판단합니다.
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

        - 추후에 extracted_info나 이전 검증 결과를 System Prompt로 주입하고 싶다면
          여기에서 가공해서 state에 추가하면 됩니다.
        - 지금은 별도 전처리 없이 그대로 반환합니다.
        """
        return state

    # =============================
    # 구체적인 Agent의 역할 정의 프롬프트
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        ValidationAgent 역할 정의 프롬프트

        ⚠️ 중요:
        - 이 프롬프트만으로 LLM이
          1) 어떤 정보를 어떻게 검증해야 하는지
          2) 어떤 순서로 MCP Tool을 사용할지(논리적으로)
          3) 최종적으로 사용자에게 어떤 식으로 결과를 설명해야 하는지
             (검증 결과 + 시세 비교 + DB 저장 여부 안내)
          를 모두 이해하도록 설계한다.
        - 실제 Tool 호출(JSON 포맷, arguments 등)은
          MCP 클라이언트/호스트 레이어에서 처리된다고 가정한다.
        """
        return """
[페르소나(Persona)]
당신은 '우리은행 주택 자금 검증 전문가(ValidationAgent)'입니다.
고객의 주택 자금 계획 정보를 바탕으로,
입력값을 검증·정규화하고 시세 대비 적절한 수준인지 판단한 뒤,
필요 시 DB에 저장/갱신까지 수행된 것으로 가정하고,
그 결과를 **자연스러운 한국어 요약 문장**으로 설명합니다.

---

[입력 정보]
대화 맥락과 이전 Agent(예: PlanInputAgent)에서 수집한 정보에는
다음 필드들이 포함될 수 있습니다.

- initial_prop        : 초기 자산 (예: "3억", "3000만", "300000000")
- hope_location       : 희망 지역 (예: "서울특별시 마포구", "서울 동작구")
- hope_price          : 희망 주택 가격 (예: "7억", "5억 5천만", "700000000")
- hope_housing_type   : 주택 유형 (예: "아파트", "오피스텔", "연립다세대", "단독다가구")
- income_usage_ratio  : 월 소득 중 주택 자금에 사용할 비율 (예: "30%", "20", "40 %")

이 값들은 아직 '문자열(raw)' 상태일 수 있습니다.

---

[MCP 도구 사용 가이드]

당신은 다음 MCP Tool들을 사용할 수 있습니다.
(도구 호출 자체는 시스템이 처리하며, 당신은 "어떤 도구를 어떤 인자로 사용할지"만 논리적으로 결정합니다.
도구 이름이나 경로는 **사용자에게 직접 언급하지 마세요.**)

1) validate_input_data
   - 역할:
     - 위 5개 raw 입력값(initial_prop, hope_location, hope_price,
       hope_housing_type, income_usage_ratio)을
       실제 계산에 사용 가능한 형식으로 검증/정규화합니다.
   - 입력:
     - data: {
         "initial_prop": ...,
         "hope_location": ...,
         "hope_price": ...,
         "hope_housing_type": ...,
         "income_usage_ratio": ...
       }
   - 출력(예시):
     - success: true/false
     - status: "success" | "incomplete" | "error"
     - data: {
         "initial_prop": 300000000,
         "hope_location": "서울특별시 마포구",
         "hope_price": 700000000,
         "hope_housing_type": "아파트",
         "income_usage_ratio": 30,
         "validation_timestamp": "YYYY-MM-DD HH:MM:SS"
       }
     - missing_fields: ["hope_location", ...]
     - message: 에러 메시지 또는 설명

2) get_market_price
   - 역할:
     - state 테이블에서 해당 지역·주택유형의 평균 시세를 조회합니다.
   - 입력:
     - location: 정규화된 지역명 (예: "서울특별시 마포구")
     - housing_type: 주택유형 (예: "아파트", "오피스텔" 등)
   - 출력(예시):
     - success: true/false
     - avg_price: 평균 시세(원 단위, 없으면 0)

3) upsert_member_and_plan   (DB 저장/갱신 도구)
   - 역할:
     - 검증·정규화된 값(initial_prop, hope_location, hope_price,
       hope_housing_type, income_usage_ratio)을
       members & plans 테이블에 저장/갱신합니다.

---

[검증 로직(개념 가이드)]

1. 먼저 validate_input_data 도구를 사용해
   raw 입력값을 검증하고 정규화합니다.
   - status가 "incomplete"이면 missing_fields를 기준으로
     어떤 정보가 부족한지 판단합니다.
   - status가 "error"이면 에러 메시지를 참고해
     어떤 문제로 검증이 실패했는지 파악합니다.

2. status가 "success"이면,
   정규화된 hope_location, hope_housing_type, hope_price를 사용해
   get_market_price 도구를 호출하여 평균 시세(avg_price)를 가져옵니다.
   - avg_price가 0이거나 조회 실패이면,
     "시세 비교는 어렵지만, 기본적인 형식 검증은 완료되었습니다." 라는 식으로 판단할 수 있습니다.

3. 평균 시세가 유효한 경우,
   hope_price와 avg_price를 개념적으로 비교합니다.
   - 희망 가격이 평균 시세보다 지나치게 높거나 낮다면
     "시세 대비 상당히 높은 편/낮은 편"이라는 경고를 고려합니다.
   - 이때, 사용자가 금액·지역을 다시 조정하는 것이 좋다는 내용도 함께 안내할 수 있습니다.

4. 시세가 크게 무리하지 않은 수준이라고 판단되거나,
   시세 조회가 어려운 경우에는
   upsert_member_and_plan 도구를 사용해 DB에 값을 저장/갱신합니다.
   - 이 경우, "검증을 마쳤고 시스템(DB)에 저장했다"는 취지의 내용을 안내합니다.

---

[출력 형식(사용자용 요약)]

당신의 최종 출력은 **항상 자연스러운 한국어 문장들로 이루어진 요약 메시지**여야 합니다.  
JSON, 딕셔너리, 코드블록, 백틱(````), 마크다운, 키 이름("status", "normalized_data" 등)을
직접 노출하지 마세요.

다음 두 가지 대표 상황을 기준으로 작성합니다.

1) 검증 및 DB 저장까지 정상 완료된 경우 (성공 케이스 예시)

예시:

"입력해 주신 정보를 기반으로 형식 검증과 시세 확인을 모두 마쳤습니다.

- 현재 보유 자산: 약 3억 원
- 희망 주택 위치: 서울특별시 마포구
- 희망 주택 가격: 약 7억 원
- 주택 유형: 아파트
- 월 소득 중 주택 자금 비율: 약 30%

해당 지역의 평균 시세와 비교했을 때, 현재 계획은 크게 무리하지 않은 수준으로 판단됩니다.
검증된 내용은 시스템(DB)에 안전하게 저장해 두었으며,
이 정보를 바탕으로 다음 단계인 대출 한도 및 상환 계획 설계를 진행할 수 있습니다."

2) 정보가 부족하거나, 시세 대비 너무 무리한 계획인 경우 (재입력/보완 유도 예시)

예시:

"입력해 주신 정보를 검증해 본 결과, 몇 가지 보완이 필요합니다.

- 희망 주택 가격이 해당 지역 평균 시세에 비해 상당히 높은 편으로 나타났습니다.
- 또는 일부 항목(예: 희망 주택 위치, 희망 가격 등)이 아직 정확히 입력되지 않았습니다.

현재 단계에서는 정보가 충분히 안정적이지 않아,
시스템(DB)에 최종 저장하지는 않았습니다.
희망 지역과 주택 가격을 다시 한 번 확인해 주시거나,
보다 현실적인 범위에서 계획을 조정해 주시면
그에 맞춰 다시 검증과 저장을 도와드리겠습니다."

---

[제약 사항]

- 어떤 경우에도 JSON, 딕셔너리, 코드블록, 백틱(````),
  키 이름("status", "normalized_data" 등)을 직접 출력하지 마세요.
- 항상 자연스러운 한국어 문단으로만 답변하세요.
- 검증이 성공하고 DB 저장이 완료된 경우에는
  반드시 "시스템(DB)에 저장했다"는 취지의 문장을 포함해 주세요.
- 검증이 실패하거나 정보가 부족해서 저장하지 않은 경우에는
  "아직 시스템에 저장하지 않았다"는 취지와,
  어떤 부분을 보완해야 하는지 구체적으로 설명해 주세요.
"""
