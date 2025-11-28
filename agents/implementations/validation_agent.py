import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase
from agents.config.base_config import BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("validation_agent")
class ValidationAgent(AgentBase):
    """
    주택 자금 계획 검증 MCP-Client Agent (Template 스타일)

    역할:
    - PlanInputAgent에서 수집한 주택 자금 계획 정보를 검증·정규화
    - 시세와 비교하여 계획이 무리한지 여부를 판단
    - 검증 결과를 바탕으로 members & plans 및
      (필요 시) 예금/적금/펀드 배분 금액을 members 테이블에 저장
    - 최종 결과를 자연스러운 한국어로 요약하여 안내
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # 이 Agent가 사용할 MCP Tool 이름 목록
        self.allowed_tools = [
            "check_plan_completion",    # PlanInput 단계 입력 완료 여부 판단
            "validate_input_data",      # 6개 필드 검증·정규화
            "get_market_price",         # 지역/주택유형 평균 시세 조회
            "upsert_member_and_plan",   # members & plans 기본 계획 저장/갱신
            "save_user_portfolio",      # ✅ 예/적/펀드 배분 금액을 members에 저장
        ]

        # ValidationAgent는 다른 Agent로 delegation 하지 않음
        self.allowed_agents: list[str] = []

    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        messages 또는 global_messages 중 하나에 HumanMessage 가 최소 1개 이상 있는지만 확인
        """
        messages = state.get("messages")
        if messages is None:
            messages = state.get("global_messages")

        if not messages or not isinstance(messages, list):
            logger.error(f"[{self.name}] 'messages' 또는 'global_messages'는 비어 있지 않은 리스트여야 합니다.")
            return False

        if not any(isinstance(m, HumanMessage) for m in messages):
            logger.error(f"[{self.name}] HumanMessage 타입의 메시지가 없습니다.")
            return False

        return True

    def pre_execute(self, state: AgentState) -> AgentState:
        """
        실행 전 전처리 (지금은 그대로 반환)
        """
        return state

    # =============================
    # 역할 정의 프롬프트
    # =============================
    def get_agent_role_prompt(self) -> str:
        return """
[페르소나]
당신은 plan_input_agent에서 입력한 정보를 검증하는 에이전트입니다. 아래 작성된 [사용 Tool과 TASK]와 [delegate 규칙]에 따라 행동하십시오.

---

[PlanInputAgent와의 연계]

이전 단계(PlanInputAgent)에서 보통 다음 6개 값이 대화 맥락과 Tool 결과에 포함되어 있습니다.

1) initial_prop        : 초기 자산 (예: "3천만", 30000000)
2) hope_location       : 희망 지역 (예: "서울 마포구")
3) hope_price          : 희망 주택 가격
4) hope_housing_type   : 주택 유형 (예: "아파트")
5) income_usage_ratio  : 월 소득 중 주택 자금에 쓸 비율(%)
6) investment_ratio    : 예금:적금:펀드 자산 배분 비율 (예: "30:40:30")
   - MCP Tool 입력에서는 주로 **ratio_str** 라는 필드 이름으로 전달됩니다.

---

[사용 Tool과 TASK]

당신은 아래 다섯 가지 도구를 사용할 수 있습니다. 해당 도구들을 사용해서 실행하세요.

1) check_plan_completion
   - 역할: 대화 히스토리와 PlanInputAgent 결과를 보고 6개 입력(initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio, investment_ratio/ratio_str)이 충분히 수집되었는지 판단합니다.

2) validate_input_data
   - 역할: 6개 필드를 파싱·검증·정규화합니다.
   
3) get_market_price
   - 역할: 정규화된 hope_location, hope_housing_type 기준 평균 시세를 조회합니다.

4) upsert_member_and_plan
   - 역할: 검증·정규화된 기본 계획 값 (initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio)을 members테이블과 plans 테이블에 저장/갱신합니다.

5) save_user_portfolio
   - 역할: 사용자가 최종적으로 확정한 예금/적금/펀드 배분 금액을 members 테이블의 deposit_amount, savings_amount, fund_amount 컬럼에 저장합니다.

---

[delegate 규칙]

1. 검증시 입력한 정보가 부족하거나 이상하다면 다시 질문하여 입력받을 수 있게 supervisor_agent로 delegate한 후, plan_input_agent로 delegate하세요.

2. 검증이 완료됐다면 JSON이 아닌 한국어 설명 텍스트만 최종 응답으로 반환하고 종료하세요.
"""
