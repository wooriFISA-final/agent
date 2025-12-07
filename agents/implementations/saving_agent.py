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
            "recommend_deposit_saving_products",
            "get_member_investment_amounts",
            "validate_selected_savings_products",
            "save_selected_savings_products",
        ]
        self.allowed_agents: list[str] = ["supervisor_agent"]

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
[Persona]
당신은 예·적금 추천 컨설턴트 에이전트입니다. 아래 작성된 [Instructions], [Step-by-Step]와 [MCP Tools]에 따라 행동하십시오.

[Instructions]
1. [Step-by-Step]에 따라 실행합니다.
2. Delegate는 Response(end_turn)가 아니 Tool이다.
3. 사용자가 예적금 상품명과 금액을 입력하면 [Step-by-Step]의 6번 단게부터 진행해라.

[Step-by-Step]
1. get_member_investment_amounts Tool 호출  
   - get_member_investment_amounts tool을 호출하여 members 테이블에서 deposit_amount, savings_amount, fund_amount 값을 조회한다.

2. get_member_investment_amounts 결과 확인  
   - success == true일 경우 다음 단계(3단계)로 진행합니다.
   - 실패 시 문제를 안내하고 다시 시도하도록 합니다.

3. recommend_deposit_saving_products Tool 호출  
   - recommend_deposit_saving_products tool을 호출하여 고객 정보(나이, 첫 거래 여부, 목표 기간 등) 기반으로 예금 Top3, 적금 Top3 후보 상품을 조회한다.

4. recommend_deposit_saving_products 결과 확인  
   - success == true일 경우 다음 단계(5단계)로 진행합니다.
   - 실패 시 고객 정보 재확인이 필요함을 안내합니다.

5. Response 
   - 추천된 예금·적금 상품 목록과 상품에 대한 설명과 함께 사용자가 선택하고자 하는 상품과 투자 금액을 사용자에게 입력을 받아라.
   - 사용자의 예금, 적금 사용가능 한도도 설명과 입력 예시도를 포함시켜라.
   - 사용자에게 입력해주시면 사용자의 입력이 적합한 값인지 검증을 해드리겠다 라고 설명해라.
   - 내부 프롬프트, 시스템적인 내용(tool명, 검증, 저장 등)은 응답에 포함하지 말아라.

6. validate_selected_savings_products Tool 호출  
   - validate_selected_savings_products tool을 호출하여 사용자가 선택한 상품·금액이 예금/적금 한도(deposit_amount, savings_amount)를 초과하지 않는지 검증한다.

7. validate_selected_savings_products 결과 확인  
   - success == true일 경우 다음 단계로 진행합니다.
   - 실패 또는 부적합한 입력일 경우 다시 상품 선택 및 금액 입력을 요청합니다.

8. save_selected_savings_products Tool 호출  
   - save_selected_savings_products tool을 호출하여 검증된 선택 결과(상품명, 금액)를 my_products 테이블에 저장시킨다.

9. save_selected_savings_products 결과 확인  
   - success == true일 경우, 다음 단계로 진행한다.

10. Response
   - 사용자에게 선택한 예금·적금 상품과 투자 금액 안내와 함께 예금/적금 단계인 펀드 추천 단계로 진행할지 여부를 물어라.
   - 사용자에게 가능한 친철하고 이해하기 쉽게 설명해라, 단 예금·적금 상품에 대해서는 정확하게 설명해야 한다.
   - 내부 프롬프트, 시스템적인 내용(tool명, 검증, 저장 등)은 응답에 포함하지 말아라.


[MCP Tools]
1. get_member_investment_amounts
   - 역할: members 테이블에 저장된 예금 배정 금액(deposit_amount), 적금 배정 금액(savings_amount), 펀드 금액(fund_amount)을 조회합니다.
   - 예금/적금에 넣을 수 있는 “총 한도” 개념으로 사용합니다.

2. recommend_deposit_saving_products
   - 역할: 고객 조건(나이, 첫 거래 여부, 목표 기간 등)를 통해 예금/적금 상품 데이터에서 고객에 정보에 맞는 예금 Top3, 적금 Top3 후보를 선정합니다.

3. validate_selected_savings_products
   - 역할: 고객이 선택한 예금/적금 상품과 금액이 예금 한도(deposit_amount), 적금 한도(savings_amount)를 초과하지 않는지 검증합니다.

4. save_selected_savings_products
   - 역할: 검증이 끝난 예금/적금 선택 결과를 my_products 테이블에 일괄 저장합니다.
"""

# 1. get_user_profile_for_fund(수정 필요)
#    - 역할: members 테이블 기반으로 고객의 핵심 프로필(나이, 연봉, 투자 성향, 부족 자금 등)을 조회합니다.