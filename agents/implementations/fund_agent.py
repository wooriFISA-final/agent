import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("fund_agent")
class FundAgent(AgentBase):
    """
    펀드 추천 + 선택 + 검증 + 저장까지 담당하는 MCP-Client Agent

    역할:
    - 사용자 투자 성향과 펀드 한도(fund_amount)를 조회
    - 투자 성향에 맞는 펀드 후보를 추천
    - 사용자가 펀드 상품을 선택하고, 각 상품별 투자 금액을 입력하도록 대화
    - 전체 투자 금액이 fund_amount를 초과하는지 검증
    - 사용자가 선택 완료를 말하면 my_products에 저장
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # 이 Agent가 사용할 MCP Tool 이름 목록
        # (실제 HTTP 경로/스펙 매핑은 MCP-Server에서 처리)
        self.allowed_tools = [
            # 1) 사용자 정보 / 투자 여력 조회
            "get_user_profile_for_fund",        # /db/get_user_profile_for_fund
            "get_member_investment_amounts",    # /db/get_member_investment_amounts

            # 2) 추천 / 비율 정보
            "get_ml_ranked_funds",              # /db/get_ml_ranked_funds
            "get_investment_ratio",             # /db/get_investment_ratio (선택)

            # 3) 선택한 펀드 검증 + 저장
            "validate_selected_funds_products", # /input/validate_selected_funds_products
            "save_selected_funds_products",     # /db/save_selected_funds_products

            # 4) 단일 펀드 바로 가입(예외적 케이스)
            "add_my_product",                   # /db/add_my_product
        ]

    # =============================
    # 1. 입력 검증
    # =============================
    def validate_input(self, state: AgentState) -> bool:
        """
        FundAgent 실행 전 입력 검증.

        기대 state:
        - state["messages"]        : 대화 메시지 리스트
        - state["user_id"]         : (선택) 사용자 ID
        - state["user_data"]       : (선택) 이전 노드에서 전달된 사용자 프로필
        - state["selected_funds"]  : (선택) 사용자가 실제 가입 선택한 펀드 리스트

        규칙:
        - messages 리스트가 존재하고
        - HumanMessage 가 최소 하나 포함되어 있으면 유효
        """
        messages = state.get("messages")

        if not messages or not isinstance(messages, list):
            logger.error(f"[{self.name}] 'messages' must be a non-empty list")
            return False

        if not any(isinstance(m, HumanMessage) for m in messages):
            logger.error(f"[{self.name}] No HumanMessage in messages")
            return False

        return True

    # =============================
    # 2. 실행 전 전처리
    # =============================
    def pre_execute(self, state: AgentState) -> AgentState:
        """
        실행 전 전처리

        - user_id가 없으면 기본값 1로 설정
        """
        if "user_id" not in state or state.get("user_id") is None:
            state["user_id"] = 1
        return state

    # =============================
    # 3. 시스템 프롬프트(역할 정의)
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        FundAgent의 역할 정의 프롬프트.
        - 길이를 줄이고, 각 Tool의 역할과 사용 순서만 명확히 설명
        """
        return """
      [페르소나(Persona)]
      당신은 펀드 상품 분석가입니다. 고객의 실제 투자 성향과 DB에 저장된 펀드 한도를 기반으로, 무리하지 않는 범위에서 펀드 포트폴리오를 추천·검증·저장해야 합니다. 아래 작성된 [Tools]와 [TASK]와 [delegate 규칙]에 따라 행동하십시오.

[사용 Tool 설명]

1) get_user_profile_for_fund (/db/get_user_profile_for_fund)
    - 실제 투자 성향(invest_tendency) 조회. 이후 모든 로직에서 이 값을 사용.

2) get_member_investment_amounts (/db/get_member_investment_amounts)
    - fund_amount(펀드 투자 가능 최대 금액) 확인. 이후 선택·검증 단계에서 이 한도를 절대 넘기지 않아야 한다.

3) get_ml_ranked_funds (/db/get_ml_ranked_funds)
   - 추천 후보를 얻기 위한 도구.

4) get_investment_ratio (/db/get_investment_ratio)
   - "왜 이 정도 펀드 비중이 적절한지" 설명할 때 참고용 설명 자료.

5) validate_selected_funds_products (/input/validate_selected_funds_products)
   - 사용자가 선택한 전체 펀드 금액이 한도 내인지 검증. remaining_fund_amount < 0 또는 violations 존재 시, 초과/문제 상황이므로 사용자에게 상세 설명 후 금액 조정 요청.

6) save_selected_funds_products (/db/save_selected_funds_products)
   - 검증된 최종 선택 펀드를 my_products에 실제 저장.

7) add_my_product (/db/add_my_product)
   - 예외적으로 "특정 펀드 1개만 지금 바로 가입"하는 원샷 케이스에 사용. product_name에는 항상 추천 리스트에 나온 "정확한 펀드 풀네임"을 넣어야 한다.
   
---

[TASK]

1. 고객의 투자 성향과 펀드 한도를 바탕으로 펀드 포트폴리오를 추천·검증·저장하고, 결과와 투자 이유를 한국어로 이해하기 쉽게 설명한다.

---

[delegate 규칙]
1. 이 에이전트가 성공적으로 완료되면, 정보에 대한 요약을 하기 위하여 supervisor_agent로 delegate 하십시오. 
2. 이 에이전트가 실패한다면 supervisor_agent로 delegate 해서 사용자에게 현재 오류 상황에 대해 알려주세요.
"""