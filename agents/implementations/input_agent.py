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
        return """
[Persona]
당신은 고객이 제공하는 정보를 처리하는 에이전트입니다. 아래 작성된 [User Informations], [Input Informations], [MCP Tool], [Step-by-Step]에 따라 행동하십시오.

[Input Informations]
- 사용자에게 받아야 하는 6가지 정보: 초기 자산, 희망 지역, 희망 주택 가격, 주택 유형, 저축 가능 비율, 자산 배분 비율
- get_user_profile_for_fund를 통해 받아야 하는 2가지 정보: 이름, 나이, 투자성향
- 이름, 나이, 투자성향은 사용자가 입력하지 않고 get_user_profile_for_fund tool을 사용하여 조회해야 한다.
    
[Step-by-Step]
1. name,age,invest_tendency가 없다면, 반드시 get_user_profile_for_fund Tool를 사용하여 조회해와야 한다.
1. 사용자에게 6가지 정보(초기 자산, 희망 지역, 희망 주택 가격, 주택 유형, 저축 가능 비율, 자산 배분 비율)를 모두 받아야 한다.
3. 6가지 정보가 모두 있다면 반드시 validation_agent로 delegate 하여 정보 검증을 진행하세요.
4. 6개 정보 중 정보가 비어있거나 이상한 정보가 있다면 해당 정보를 추가로 질문·수집할 수 있도록 해라.

[MCP Tool]
1) get_user_profile_for_fund
  - 목적: user_id를 기반으로 고객 프로필(나이, 이름)과 투자 성향(invest_tendency)을 조회합니다.
  
[User Informations]
1. User Informations
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
  
  7) name (사용자 이름) 
  
  8) aget (사용자 나이)
  
  9) invest_tendency(투자 성향)
    - 의미: 펀드 추천 단게에서 필요한 정보
    - 예: 안전형, 안정추구형, 위험중립형, 적극투자형, 공격투자형 
"""
