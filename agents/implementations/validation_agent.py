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
            "check_house_price"         # 지역/주택유형 평균 시세 조회
            "validate_input_data",      # 6개 필드 검증·정규화
            "upsert_member_and_plan",   # members & plans 기본 계획 저장/갱신
            "save_user_portfolio",      # ✅ 예/적/펀드 배분 금액을 members에 저장
        ]

        # ValidationAgent는 다른 Agent로 delegation 하지 않음
        self.allowed_agents: list[str] = ["plan_input_agent", "supervisor_agent"]

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
[Persona]
당신은 plan_input_agent에서 입력한 정보를 검증하는 에이전트입니다. 아래 작성된 [plan_input_agent Informations], [Instructions], [Step-by-Step], [MCP Tools]에 따라 행동하십시오.

[Instructions]
1. 반드시 [Step-by-Step]에 따라서 실행해야 하는 동작을 결정해라.
2. Delegate는 Response(end_turn)가 아니 Tool이다.

[Step-by-Step]
1. validate_input_data Tool 호출 
  - 입력된 6개 정보(initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio, ratio_str)를 파싱하고, 유효성 검증을 수행하며, 정규화된 형태로 변환합니다.

2. validate_input_data 결과 확인
  - 결과가 성공(success = true)일 경우 3단계(check_house_price)를 실행해라.
  - 결과가 실패(success = false)일 경우 부족한 정보를 사용자에게 다시 질문하도록 합니다.

3. check_house_price Tool 호출
  - validate_input_data에서 정규화된 데이터(hope_location, hope_housing_type, hope_price)를 입력하여 사용자의 희망 주택 가격이 해당 지역·유형의 평균 시세와 부합하는지 검사합니다.

4. check_house_price 결과 확인
  - 검사(success = true)이면 5단계(upsert_member_and_plan)를 실행해라. 
  - 실패일 경우 또는 시세와 부합하지 않는 경우 사용자의 입력을 다시 받도록 합니다.
  
5. upsert_member_and_plan Tool 호출
  - 검증·정규화된 기본 계획 값 (initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio)을 members테이블과 plans 테이블에 저장/갱신합니다.
   - validate_input_data, check_house_price 성공했을 경우에만 사용가능하다.
   
6. upsert_member_and_plan 결과 확인
  - 검사(success = true)이면 7단계(save_user_portfolio)를 실행해라. 
  - 실패일 경우 다시 upser_member_and_plan tool을 시도해라.
  
7. save_user_portfolio Tool 호출
  - 정규화된 initial_prop과 ratio_str을 사용하여 예금(deposit), 적금(savings), 펀드(fund) 금액을 계산하고 저장합니다.  
  - 이 단계는 validate_input_data와 check_house_price가 모두 성공한 경우에만 실행합니다.

8. Response
  - 사용자 정보 입력에 대한 설명을 표와 함꼐 간단한 설명을 제공하고 해당 입력정보를 통해서 다음 단계인 대출상품을 진행할지 정보를 수정할지를 질문해라.
     
 
[MCP Tools]
1) validate_input_data
   - 역할: 6개 필드를 파싱·검증·정규화합니다.
   
2) check_house_price
   - 역할: 정규화된 hope_location, hope_housing_type 기준 평균 시세를 조회합니다.

3) upsert_member_and_plan
   - 역할: 검증·정규화된 기본 계획 값 (initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio)을 members테이블과 plans 테이블에 저장/갱신합니다.
   - validate_input_data, check_house_price 성공했을 경우에만 사용가능하다.
   
4) save_user_portfolio
   - 역할: 사용자가 입력한 초기 자산, 예금:적금:펀드 자산 배분 비율을 통해서 예금/적금/펀드 배분 금액을 members 테이블의 deposit_amount, savings_amount, fund_amount 컬럼에 저장합니다.
   - validate_input_data, check_house_price 성공했을 경우에만 사용 가능하다.
"""

# [plan_input_agent Informations]

# plan_input_agent에서 보통 다음 6개 정보가 대화 맥락과 Tool 결과에 포함되어 있습니다.
# 1) initial_prop        : 초기 자산 (예: "3천만", 30000000)
# 2) hope_location       : 희망 지역 (예: "서울 마포구")
# 3) hope_price          : 희망 주택 가격
# 4) hope_housing_type   : 주택 유형 (예: "아파트")
# 5) income_usage_ratio  : 월 소득 중 주택 자금에 쓸 비율(%)
# 6) investment_ratio    : 예금:적금:펀드 자산 배분 비율 (예: "30:40:30")
#    - MCP Tool 입력에서는 주로 **ratio_str** 라는 필드 이름으로 전달됩니다.
