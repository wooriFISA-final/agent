import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase
from agents.config.base_config import BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("loan_agent")
class LoanAgent(AgentBase):
    """
    주택담보대출 MCP-Client Agent

    역할:
    - MCP Tool을 활용해 고객의 대출 가능 금액과 부족 자금을 계산
    - LTV, DSR, DTI, 신용점수, 지역 규제를 고려한 '현실적인' 대출 한도 산출
    - 결과를 내부적으로는 구조화된 정보(loan_amount, shortage_amount 등)로 정리하고,
      최종적으로는 사용자에게 한국어로 자연스럽게 설명하는 역할을 담당

    MCP 도구(allowed_tools):
    - get_user_loan_overview : /db/get_user_loan_overview
    - calc_shortage_amount   : /input/calc_shortage_amount (또는 /input/calc_shortage 래핑 Tool 이름)
    - update_loan_result     : /db/update_loan_result
    """

    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        # AgentBase가 LLM 설정/llm_config를 이미 처리
        super().__init__(config)

        # 이 Agent가 사용할 MCP Tool 이름 목록
        # (실제 HTTP 경로 매핑은 MCP 프레임워크/호스트 레이어에서 처리한다고 가정)
        self.allowed_tools = [
            "get_user_loan_overview",
            "calc_shortage_amount",  # plan_agent_tools.py의 CalcShortageAmountRequest.tool_name 기준
            "update_loan_result",
        ]
        self.allowed_agents: list[str] = ["supervisor_agent"]
    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        LoanAgent 실행 전 입력 검증.

        기대 state:
        - state["messages"] : 대화 메시지 리스트
        - state["user_id"]  : (선택) 현재 로그인 사용자 ID

        기본적으로:
        - messages 리스트가 존재하고
        - HumanMessage 가 최소 하나 포함되어 있으면 유효하다고 판단.
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

        - 여기서는 별도 전처리 없이 그대로 반환.
        - 필요하다면 나중에:
          - user_id 기본값 주입
          - 이전 노드(ValidationAgent 등)의 결과를 system context로 추가
          등의 작업을 수행 가능.
        """
        return state

    # =============================
    # 구체적인 Agent의 역할 정의 프롬프트
    # =============================
    def get_agent_role_prompt(self) -> str:
        return """
[Persona]
당신은 대출 컨설턴트 에이전트입니다. 희망 주택가격, 예상 대출금, 보유 자산을 기반으로 부족 자금을 계산하고, 부족 자금, 대출 상품 정보를 사용자에게 설명해야 한다.
아래 작성된 [Instructions], [Step-by-Step], [MCP Tools]와 [LTV·DSR·DTI Definition]에 따라 행동하십시오.

[Instructions]
1. [Step-by-Step]에 따라 실행합니다.
2. Delegate는 Response(end_turn)가 아니 Tool이다.


[Step-by-Step]
1. get_user_loan_overview Tool 호출
  - 대출 상품 관련 정보를 가져와야 한다.
  
2. get_user_loan_overview 결과 확인
  - 결과가 성공(success == true)일 경우 다음 단계로 진행해라.  

3. calc_shortage_amount를 Tool 호출
  - 사용자 희망한 주택을 구매하기 위해 calc_shortage_amount를 사용하여 희망 주택가격, 예상 대출금, 보유 자산을 기반으로 부족 자금을 계산해야 한다.
  
4. calc_shortage_amount 결과 확인
  - 결과가 성공(success == true)이고 부족 자금이 상적으로 계산된 경우 다음 단계로 진행해라.
  
5. update_loan_result Tool 호출
  - 대출 가능 금액과 부족 자금을 저장시켜야 한다.

6. update_loan_result 결과 확인
  - 결과가 성공(success == true)일 경우 다음 단계로 진행해라.

7. Response
  - 위 과정이 정상적으로 완료된 경우, 대출상품 설명, 대출 가능 금액, 부족 자금, 그리고 사용자의 LTV·DSR·DTI 정보를 표와 함께 간단히 설명을 해라.
  - 서비스의 다음 단계인 예금/적금을 진행할지의 여부를 확인하는 내용도 추가해라.

[MCP Tools]
1) get_user_loan_overview
   - 역할: DB에서 members, plans, loan_product를 조인하여 한 번에 대출 관련 핵심 정보를 가져옵니다.

2) calc_shortage_amount
   - 역할: 희망 주택가격, 예상 대출금, 보유 자산을 기반으로 부족 자금을 계산합니다.

3) update_loan_result
   - 역할: 대출 가능 금액과 부족 자금을 DB(plans + members)에 반영합니다.


[LTV·DSR·DTI Definition]
1. LTV (Loan To Value): 주택 가격 대비 대출금 비율
2. DSR (Debt Service Ratio): 연소득 대비 모든 금융 부채의 원리금 상환액 비율.
3. DTI (Debt To Income): 연소득 대비 주택담보대출의 원리금 상환액이 차지하는 비율.
"""
