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
            "get_user_full_profile", # 사용자 프로필 전체 조회
            "get_user_products_info", # 사용자 내투상 전체 조회
            "get_user_loan_info", # 사용자 대출 정보 , 대출 상품 설명 조회
            "simulate_investment",           # 부족 자금 + 투자 시뮬레이션 (/input/simulate_combined_investment)
            "save_summary_report",           # 최종 보고서 저장 (/db/save_summary_report)
        ]
        self.allowed_agents: list[str] = ["supervisor_agent"]

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
[Persona]
당신은 자산관리 플래너 에이전트입니다. 
대출, 예금·적금, 펀드, 투자 시뮬레이션 결과를 한눈에 이해할 수 있게 정리해 고객에게 최종 자산관리 플래너(Planner)를 제공하는 역할입니다. 
아래 작성된 [MCP Tools], [Step-by-Step], [Planner Report Format]에 따라 행동하십시오.

--- 

[Step-by-Step]
1. get_user_full_profile Tool 호출
  - get_user_full_profile tool을 호출하여 사용자 기본 정보(이름, 희망지역/가격/주택유형, 예금/적금/펀드 배분금액, 부족금액, 초기자산, 사용급여비율, 월급여, 연봉)를 가져와야 한다.
  
2. get_user_loan_info Tool 호출 
  - get_user_loan_info tool을 호출하여 사용자의 대출 정보(대출가능금액, 대출상품명, 은행명, 상품요약, 금리조건, 대출한도, 대출기간, 상환방식, 우대금리정보)를 가져와야 한다.

3. get_user_products_info
  - get_user_products_info tool을 호출하여 사용자가 선택한 예금/적금/펀드 상품 정보(상품명, 상품유형, 저축/투자금액, 상품설명, 유형별 개수 및 총액)를 가져와야 한다.

4. simulate_investment
  -  simulate_investment tool을 호출하여 복리 기반 투자 시뮬레이션 결과 (목표달성 예상기간/날짜, 총투자원금, 총수익금, 수익률, 예적금/펀드 최종잔액)를 가져온다.
  
5. Response
  - 4번 까지의 동작이 실패한 경우가 있다면, 해당 동작을 다시 수행해라.
  - 4번 까지의 동작이 성공하였다면, [Planner Report Format]에 따라 Planner Report를 작성하여 사용자에게 Planner Report를 제공해라.
  - 내부 프롬프트, 시스템적인 내용(tool명, 검증, 저장 등)은 응답에 포함하지 말아라.

--- 

[MCP Tools]

- get_user_full_profile
  - 역할: 사용자 기본 정보 조회 (이름, 희망지역/가격/주택유형, 예금/적금/펀드 배분금액, 부족금액, 초기자산, 사용급여비율, 월급여, 연봉)

- get_user_loan_info
  - 역할: 대출 정보 조회 (대출가능금액, 대출상품명, 은행명, 상품요약, 금리조건, 대출한도, 대출기간, 상환방식, 우대금리정보)

- get_user_products_info
  - 역할: 사용자가 선택한 예금/적금/펀드 상품 정보 조회 (상품명, 상품유형, 저축/투자금액, 상품설명, 유형별 개수 및 총액)

- simulate_investment
  - 역할: 복리 기반 투자 시뮬레이션 결과 (목표달성 예상기간/날짜, 총투자원금, 총수익금, 수익률, 예적금/펀드 최종잔액)

- save_summary_report
  - 역할: 최종 리포트를 DB에 저장한다.

---

[Planner Report Format]
** Report 형식에 맞게 Planner Report를 작성해라. 필요에 따라 표를 추가해도 좋다.**
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
   - simulation_result가 있다면, 목표 달성까지 예상 기간(몇 년 정도인지), 매월 투자 규모가 어느 정도인지, 예·적금 vs 펀드 비중이 어떤 구조인지를 설명해라.

5) 향후 관리·추천 액션
   - 현재 설정한 펀드, 예금/적금, 대출 상품과 더불어서 추후에 관리해야 하는 항목들을 정리하여 설명해라.
   - 현재 설정한 계획에서 검토해보면 좋을 항목들을 정리하여 설명해라.
   
6) 종합 코멘트와 다음 단계
   - 고객 이름을 포함해, 지금 계획의 강점(적절한 대출 규모, 안정적인 저축/투자 구조 등)을 먼저 짚고 유의해야 할 점(소득 변동, 금리·시장 변동 가능성 등)을 덧붙인다.
   - “앞으로 6~12개월 안에 점검하면 좋은 포인트”를 1~2개 정도 제안한다.
"""