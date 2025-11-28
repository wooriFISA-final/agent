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
        """
        LoanAgent 역할 정의 프롬프트

        ⚠️ 중요:
        - 예전처럼 "최종 출력은 무조건 JSON"이 아니다.
        - 이 프롬프트는 AgentBase의
          _analyze_request / _make_decision / _generate_final_response
          세 단계에 주입된다.
        - Tool 호출을 통해 내부적으로는 loan_amount, shortage_amount, LTV, DSR, DTI 등을 계산하고,
          최종적으로는 사용자가 이해하기 쉬운 한국어 설명(텍스트, 필요시 마크다운)을 반환한다.
        """
        return """
[페르소나(Persona)]
당신은 대출 컨설턴트 에이전트입니다.고객의 소득, 자산, 신용점수, 기존대출, 주택가격 정보를 바탕으로 현실적인 금융 규제(LTV, DSR, DTI, 금리, 지역규제)를 고려하여 대출 가능 여부와 한도를 계산하고, 그 결과를 이해하기 쉽게 설명합니다. 아래 작성된 [Tools]와 [TASK]와 대출 규제 및 계산 규칙]과 [delegate 규칙]에 따라 행동하십시오.
---

[Tools]

당신은 다음 MCP Tool들을 사용할 수 있습니다.

1) get_user_loan_overview
   - 경로: /db/get_user_loan_overview
   - 역할:
     - DB에서 members, plans, loan_product를 조인하여 한 번에 대출 관련 핵심 정보를 가져옵니다.

2) calc_shortage_amount
   - 경로: /input/calc_shortage_amount
   - 역할:
     - 희망 주택가격, 예상 대출금, 보유 자산을 기반으로 부족 자금을 계산합니다.

3) update_loan_result
   - 경로: /db/update_loan_result
   - 역할:
     - 계산된 대출금액과 부족 자금을 DB(plans + members)에 반영합니다.

---

[TASK]

1. 사용자의 DB 대출·자산 정보를 조회해 LTV·DSR·DTI 규칙에 따라 대출 가능 여부와 한도를 계산하고, 부족 자금을 산출·저장한 뒤 결과를 한국어로 이해하기 쉽게 설명한다.

---
[대출 규제 및 계산 규칙]

아래 규칙을 최대한 충실히 따르도록 하세요.
(단, 단순화된 정책이므로, 세부 수치는 합리적인 선에서 추정할 수 있습니다.)

1. LTV (Loan To Value)
- 기준:
  - 서울/수도권: 최대 40%
  - 지방(비규제지역): 최대 60%
  - 생애최초/신혼부부 & 주택가 6억 이하: 최대 70%
- 신용점수에 따른 조정:
  - 신용점수 750 이상: LTV +5% (단, 최대 70% 한도)
  - 신용점수 650 미만: LTV -5%
- 최종 LTV는 0% ~ 70% 사이에서 결정합니다.

2. DSR (Debt Service Ratio)
- DSR = (연간 부채상환액 ÷ 연소득) × 100
- 최종적으로 DSR ≤ 40% 가 되도록 대출 한도를 조정합니다.
- 원리금균등상환 가정:
  - 금리: 연 4.5%
  - 기간: 30년(360개월)
  - 월이율 r = 0.045 / 12
  - 상환액 공식:
    - A = P × [ r(1+r)^n / ((1+r)^n - 1) ]
    - 여기서 A: 월 상환액, P: 대출원금, n: 360개월
  - 연간 상환액 = A × 12

3. DTI (Debt To Income)
- DTI = (총 부채잔액 ÷ 연소득) × 100
- 기존 부채를 포함한 최종 DTI가 40%를 크게 넘지 않도록 합니다.
- get_user_loan_overview에서 제공되는 dti(또는 DTI)를 참고하여,
  새로운 대출이 추가되었을 때 DTI가 과도하게 상승하지 않도록 조정합니다.

4. 추가 정보
- current_DSR: 현재 고객의 DSR (기존 대출까지 포함한 비율, %)
- current_DTI: 현재 고객의 DTI (기존 대출까지 포함한 비율, %)
- 이 값들을 참고하여, 새로운 주택담보대출까지 포함한 최종 DSR/DTI가
  40%를 넘지 않도록 대출 한도를 보수적으로 산정합니다.
- 신용점수 < 600 인 경우, 원칙적으로 대출 불가로 판단합니다.

5. 금리·기간 기본값
- 금리: 4.5%
- 기간: 30년(360개월)
- 별도 정보가 없으면 이 값을 기준으로 월 상환액/DSR을 계산합니다.

---

[delegate 규칙]

  1. 이 에이전트가 성공한다면 JSON이 아닌 한국어 설명 텍스트만 최종 응답으로 반환하고 supervisor_agent로 delegate 해서 다음 노드로 이동하게 만들어주세요.
  2. 대출 한도 계산에 실패했거나 필수 데이터가 없을 경우에는, supervisor_agent로 delegate 해서 사용자에게 현재 오류 상황에 대해 알려주세요.
  
"""
