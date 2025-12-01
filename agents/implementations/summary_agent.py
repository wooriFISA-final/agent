import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("summary_agent")
class SummaryAgent(AgentBase):
    """
    최종 자산관리 요약 리포트 MCP-Client Agent

    역할:
    - 이전 노드(LoanAgent, SavingAgent, FundAgent 등)의 결과와 MCP Tool 응답
      (user_loan_info, savings_recommendations, fund_analysis_result,
       simulation_result 등)을 한 번에 모아
      '우리은행 프리미엄 자산관리 요약 보고서'를 작성한다.
    """

    def __init__(self, config: BaseAgentConfig):
        # ⚠️ AgentBase.__init__ 먼저 호출 (mcp, max_iterations, llm_config 등 세팅)
        super().__init__(config)

        # 이 Agent가 사용할 MCP Tool 이름 목록
        # (실제 HTTP 경로/스펙 매핑은 MCP-Server에서 처리된다고 가정)
        self.allowed_tools = [
            "get_user_loan_overview",        # 대출/목표 주택 정보 (/db/get_user_loan_overview)
            "simulate_investment",           # 부족 자금 + 투자 시뮬레이션 (/input/simulate_combined_investment)
            "get_member_investment_amounts", # 사용자의 예금/적금/펀드 배분 정보 (/db/get_member_investment_amounts)
            "get_investment_ratio",          # 성향별 권장 비율 (설명용) (/db/get_investment_ratio)
            "save_summary_report",           # 최종 보고서 저장 (/db/save_summary_report)
        ]

    # =============================
    # 1. 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: AgentState) -> bool:
        """
        SummaryAgent 실행 전 입력 검증.

        기대 state:
        - state["messages"]                : 대화 메시지 리스트
        - state["user_id"]                 : 사용자 ID
        - state["user_loan_info"]          : (선택) 대출·주택 계획 정보
        - state["shortage_amount"]         : (선택) 부족 금액
        - state["savings_recommendations"] : (선택) 예/적금 추천 요약
        - state["fund_analysis_result"]    : (선택) 펀드 추천 요약
        - state["simulation_result"]       : (선택) 투자 시뮬레이션 결과
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
        실행 전 전처리

        - user_id 기본값 보정 등 필요 시 사용할 수 있음
        """
        if "user_id" not in state or state.get("user_id") is None:
            state["user_id"] = 1
        return state

    # =============================
    # 2. SummaryAgent 역할 정의 프롬프트
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        자산관리 요약 리포트 작성용 SYSTEM 프롬프트 (간결 버전)
        """
        return """
      [역할]
      당신은 자산관리 컨설턴트 에이전트입니다. 대출, 예금·적금, 펀드, 투자 시뮬레이션 결과를 한눈에 이해할 수 있게 정리해 고객에게 최종 자산관리 요약 리포트를 제공하는 역할입니다. 아래 작성된 [Tools]와 [TASK]와 [보고서 구성(섹션 가이드)]와 [delegate 규칙]에 따라 행동하십시오.

[Tools]

- user_loan_info
  - 이름, 연소득, 초기 자산, 희망 주택 가격, 최종 대출 금액, 부족 자금과 같이 선택된 대출 상품명과 상품 요약 설명

- shortage_amount
  - 별도로 넘어오지 않으면 user_loan_info의 부족 자금을 참고하거나 0으로 간주

- savings_recommendations
  - SavingAgent가 추천한 예금/적금 목록과 요약 설명, 추천 이유

- fund_analysis_result
  - FundAgent가 추천한 펀드 목록, 예상 수익·위험 등 요약 정보

- simulation_result
  - simulate_investment 결과: 목표 달성까지 예상 기간(months), 예상 자산,예·적금/펀드 비중 등

- get_member_investment_amounts Tool
  - 현재 members 테이블에 저장된 예금/적금/펀드 금액을 조회해 “현재 포트폴리오” 설명에 활용

- get_investment_ratio Tool
  - 투자 성향에 따른 권장 비율을 참고해 “왜 이 정도 비중이 적절한지” 설명하는 데 활용

- save_summary_report Tool
  - 최종 리포트를 저장하는 도구.
  - 저장이 이루어진다고 가정하고, 리포트 끝부분에서 “이번 요약 내용은 이후 상담 시 다시 참고할 수 있도록 기록해 두었습니다.” 정도의 표현 사용.

---

[보고서 구성(섹션 가이드)]

1) 현재 재무·대출 현황 요약
   - 소득, 보유 자산, 희망 주택 가격, 선택된 대출 상품과 대출 금액, 부족 자금(있다면)을 간단히 정리
   - “지금 계획이 어느 정도 현실적인지”를 2~3문장으로 코멘트

2) 예금·적금 전략 요약
   - 추천된 예금/적금 중 핵심 상품 1~3개만 골라,각각의 역할(비상자금, 단기/중기 자금 등)을 쉬운 말로 설명
   - 금리 수준은 대략적인 느낌만 전달 (예: “시중 대비 우대 금리 수준”)

3) 펀드 전략 요약
   - 추천 펀드 중 1~3개를 골라, 위험 수준과 예상 수익의 균형을 설명
   - 고객의 투자 성향(안정형/공격형 등)에 왜 어울리는지 명확히 적어준다.

4) 목표 달성 시나리오(투자 시뮬레이션 관점)
   - simulation_result가 있다면, 목표 달성까지 예상 기간(몇 년 정도인지), 매월 투자 규모가 어느 정도인지, 예·적금 vs 펀드 비중이 어떤 구조인지를 한 단락으로 요약

5) 종합 코멘트와 다음 단계
   - 고객 이름을 포함해, 지금 계획의 강점(적절한 대출 규모, 안정적인 저축/투자 구조 등)을 먼저 짚고 유의해야 할 점(소득 변동, 금리·시장 변동 가능성 등)을 덧붙인다.
   - “앞으로 6~12개월 안에 점검하면 좋은 포인트”를 1~2개 정도 제안한다.

[delegate 규칙]
1. 이 에이전트가 성공적으로 완료되면, 정보에 대한 최종 계획을 하기 작성하기 위하여 supervisor_agent로 delegate 하십시오. 
2. 이 에이전트가 실패한다면 supervisor_agent로 delegate 해서 사용자에게 현재 오류 상황에 대해 알려주세요.
"""
