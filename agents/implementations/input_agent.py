import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase
from agents.config.base_config import BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

logger = logging.getLogger("agent_system")


@AgentRegistry.register("plan_input_agent")
class PlanInputAgent(AgentBase):
    """
    주택 자금 계획 입력 Agent (Template 스타일)

    역할:
    - 사용자와의 자연어 대화를 통해 6가지 핵심 정보를 수집
      (initial_prop, hope_location, hope_price, hope_housing_type,
       income_usage_ratio, investment_ratio)
    - MCP 도구(get_user_profile_for_fund, get_investment_ratio, calculate_portfolio_amounts)를 활용해
      투자 성향과 추천 비율을 참고한 Smart Nudging 수행
    - 필요 시 calculate_portfolio_amounts로 “예금/적금/펀드 예상 금액”을 미리 계산해 보여주되,
      실제 DB 저장은 Validation/다음 단계 Agent에서 수행하도록 넘긴다.
    - 6개 정보의 수집/정제 상태에 따라 supervisor_agent, validation_agent로
      적절히 delegate 한다.
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # ✅ 이 Agent가 사용할 MCP Tool 목록
        #    (MCP 서버 OpenAPI operation_id와 반드시 같아야 함)
        self.allowed_tools: list[str] = [
            "get_user_profile_for_fund",    # DB: members에서 투자 성향 조회
            "get_investment_ratio",         # DB: 성향별 예금/적금/펀드 추천 비율 조회
            "calculate_portfolio_amounts",  # INPUT: 총액 + 비율 → 예금/적금/펀드 금액 계산
        ]

        # ✅ 이 Agent가 위임(delegate)할 수 있는 다른 Agent 목록
        self.allowed_agents: list[str] = [
            "supervisor_agent",
            "validation_agent",
        ]

    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        state에 messages 또는 global_messages가 있고,
        그 안에 HumanMessage가 포함되어 있는지 확인
        """
        messages = state.get("messages")
        if messages is None:
            messages = state.get("global_messages")

        if not messages or not isinstance(messages, list):
            logger.error(
                f"[{self.name}] 'messages' 또는 'global_messages'는 "
                f"비어 있지 않은 리스트여야 합니다."
            )
            return False

        if not any(isinstance(m, HumanMessage) for m in messages):
            logger.error(f"[{self.name}] HumanMessage 타입의 메시지가 없습니다.")
            return False

        return True

    def pre_execute(self, state: AgentState) -> AgentState:
        """
        실행 전 전처리 훅(Hook)

        - 현재는 별도 전처리 없이 state 그대로 반환
        - 필요한 경우 나중에 로깅/초기화 등을 추가할 수 있음
        """
        return state

    # =============================
    # 역할 정의 프롬프트 (Template 스타일)
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        PlanInputAgent의 Persona/목표/도구 사용 규칙/대화 규칙/에이전트 간 delegate 규칙을 정의하는 프롬프트.
        실제 LLM 호출 및 Tool/Agent 사용 로직은 AgentBase + DynamicRouter 쪽에서 처리된다.
        """
        return """[페르소나(Persona)]
당신은 고객이 제공하는 정보를 처리하는 에이전트입니다. 아래 작성된 [사용 Tool과 TASK]와 [delegate 규칙]에 따라 행동하십시오.
---

[사용 Tool과 Task]
1. Task
  1) initial_prop (초기 자산)
    - 의미: 현재 보유 중인 목돈 (예금, 주식 등 포함)
    - 예: "3천만 원", "1억 5천"

  2) hope_location (희망 지역)
    - 의미: 주택을 구매하거나 전세로 살고 싶은 지역
    - 예: "서울 마포구", "경기도 분당"

  3) hope_price (희망 주택 가격)
    - 의미: 목표로 하는 주택의 매매/전세 가격
    - 예: "8억 원 정도", "5억 미만"

  4) hope_housing_type (주택 유형)
    - 의미: 아파트, 오피스텔, 단독주택 등 선호 유형
    - 예: "아파트", "빌라"

  5) income_usage_ratio (저축 가능 비율)
    - 의미: 월 소득 중에서 주택 자금 마련을 위해 저축/투자에 쓸 수 있는 비율(%)
    - 예: "월급의 40%", "30프로 정도"

  6) investment_ratio (자산 배분 비율)
    - 의미: 주택 자금 마련을 위한 저축/투자 금액을 **예금 : 적금 : 펀드** 로 어떻게 나눌지 비율
    - 예: "30:40:30", "50:30:20"
    - 형태는 "예금:적금:펀드" 순서의 비율 문자열로 정리합니다.
    - 중요: 성향별 권장 비율을 참고해 고객에게 제안하고, 고객이 스스로 최종 비율을 선택·확정하도록 도와주세요.
    
2. 사용 가능 Tool
  1) get_user_profile_for_fund
    - 목적: user_id를 기반으로 고객 프로필과 투자 성향(invest_tendency)을 조회합니다.

  2) get_investment_ratio
    - 목적: 투자 성향(invest_tendency)에 따른 권장 자산 배분 비율을 조회합니다.

  3) calculate_portfolio_amounts
    - 목적: total_amount(예: initial_prop)와 비율 문자열(예: "30:40:30")을 이용해
      예금/적금/펀드 각각에 얼마씩 배분되는지 예상 금액을 계산합니다.
  

---

[delegate 규칙]

1. 6개 정보(initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio, investment_ratio) 중 정보가 비어있다면 해당 정보를 추가로 질문·수집할 수 있도록 supervisor_agent 로 delegate 하십시오.

2. 6개 정보가 모두 채워졌다고 판단되면,  validation_agent 로 delegate 하여 검증하세요.

3. validation_agent 에 의해 사용자 정보 검증까지 완료되었다면, 한국어 설명 텍스트만 최종 응답으로 반환하고 종료하세요.
"""
