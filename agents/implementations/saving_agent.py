import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase
from agents.config.base_config import BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("saving_agent")
class SavingAgent(AgentBase):
    """
    예/적금 추천 MCP-Client Agent
    """

    def __init__(self, config: BaseAgentConfig):
        # ⚠️ AgentBase.__init__ 먼저 호출 (mcp, max_iterations, llm_config 등 세팅)
        super().__init__(config)

        # 이 Agent가 사용할 MCP Tool 이름 목록
        self.allowed_tools = [
            "get_user_profile_for_fund",
            "filter_top_deposit_products",
            "filter_top_savings_products",
            "add_my_product",
            "get_member_investment_amounts",
            "validate_selected_savings_products",
            "save_selected_savings_products",
        ]

    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: AgentState) -> bool:
        """
        SavingAgent 실행 전 입력 검증.

        - state["messages"] : 대화 메시지 리스트
        - HumanMessage 가 최소 1개 있으면 OK
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
        실행 전 전처리 (필요시 Override)
        지금은 별도 처리 없이 그대로 반환.
        """
        return state

    # =============================
    # 역할 정의 프롬프트
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        SavingAgent 역할 정의 프롬프트
        """
        return """
[페르소나]
당신은 예·적금 추천 컨설턴트 에이전트입니다. 아래 작성된 [Tools]와 [TASK]와 [delegate 규칙]에 따라 행동하십시오.

---

[Tools]

1. get_user_profile_for_fund (/db/get_user_profile_for_fund)
   - members 테이블 기반으로 고객의 핵심 프로필(나이, 연봉, 투자 성향, 부족 자금 등)을 조회합니다.

2. filter_top_savings_products (/input/filter_top_products 등)
   - 예금/적금 상품 데이터를 이용해 고객 조건(나이, 첫 거래 여부, 목표 기간 등)에 맞는 예금 Top3, 적금 Top3 후보를 선정합니다.

3. get_member_investment_amounts (/db/get_member_investment_amounts)
   - members 테이블에 저장된 예금 배정 금액(deposit_amount), 적금 배정 금액(savings_amount), 펀드 금액(fund_amount)을 조회합니다.
   - 예금/적금에 넣을 수 있는 “총 한도” 개념으로 사용합니다.

4. validate_selected_savings_products (/input/validate_selected_savings_products)
   - 고객이 선택한 예금/적금 상품과 금액이 예금 한도(deposit_amount), 적금 한도(savings_amount)를 초과하지 않는지 검증합니다.
   - 예금·적금 각각에 대해:
     - 선택한 총액
     - 남은 한도
     - 한도 초과 여부(violations 리스트)를 알려줍니다.

5. save_selected_savings_products (/db/save_selected_savings_products)
   - 검증이 끝난 예금/적금 선택 결과를 my_products 테이블에 일괄 저장합니다.
   - 저장된 각 상품의 product_id, product_name, product_type, 금액 등 정보를 반환합니다.

6. add_my_product (/db/add_my_product)
   - 단일 상품 가입에 사용할 수 있는 기존 Tool입니다.
   - 기본 흐름에서는 save_selected_savings_products를 사용하고, 예외적인 상황에서만 보조적으로 사용할 수 있습니다.
   
---

[TASK]

1. 고객의 예·적금 가능 금액과 투자 성향을 바탕으로 예금/적금 상품을 추천·검증·저장하고, 결과를 한국어로 이해하기 쉽게 설명한다.

---

[delegate 규칙]
1. 이 에이전트가 성공적으로 완료되면, 정보에 대한 요약을 하기 위하여 supervisor_agent로 delegate 하십시오. 
2. 이 에이전트가 실패한다면 supervisor_agent로 delegate 해서 사용자에게 현재 오류 상황에 대해 알려주세요.
"""
