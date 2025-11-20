import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manager import LLMManager  # ⚠️ 프로젝트 구조에 맞춘 import
try:
    from mcp.server.api.resources.db_tools import (
        api_get_user_profile_for_fund,
        api_get_ml_ranked_funds,
        api_add_my_product
    )
except ImportError:
    # 경로가 다를 경우를 대비해 예외 처리 (로그만 남김)
    logging.warning("db_tools import failed. Ensure the path is correct.")

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("fund_agent")
class FundAgent(AgentBase):
    """
    펀드 추천 MCP-Client Agent

    역할:
    - MCP Tool들을 활용해 펀드 후보 목록과 사용자 투자 성향을 받아와
      추천 가능한 상품들을 투자 성향에 맞게 상품 위험등급 별로 2개씩 추천
    - 각 펀드에 대해 초보자용 요약과 추천 이유를 JSON으로 생성
    - 사용자가 실제 가입을 선택한 펀드는 my_products 테이블에 저장할 수 있도록
      add_my_product Tool과 매핑하기 쉬운 구조로 응답

    MCP 도구(allowed_tools):
    - get_user_profile_for_fund   : 
    - get_ml_ranked_funds         : -> mcp\server\api\resources\db_tools.py
    - add_my_product              : 
    """

    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # LLMManager를 통해 LLM 객체 생성
        self.llm = LLMManager.get_llm(
            provider=getattr(config, "provider", "ollama"),
            model=config.model_name,
        )

        # 이 Agent가 사용할 MCP Tool 이름 목록
        # (실제 HTTP 경로/스펙 매핑은 MCP 프레임워크에서 처리한다고 가정)
        self.allowed_tools = [
            "get_user_profile_for_fund", # 사용자 성향 조회
            "get_ml_ranked_funds",       # ML 랭킹 기반 추천 조회
            "add_my_product",            # 가입 처리
        ]

    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        FundAgent 실행 전 입력 검증.

        기대 state:
        - state["messages"]        : 대화 메시지 리스트
        - state["user_id"]         : (선택) 사용자 ID
        - state["user_data"]       : (선택) 이전 노드에서 전달된 사용자 프로필
        - state["selected_funds"]  : (선택) 사용자가 실제 가입 선택한 펀드 리스트

        기본 규칙:
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

    def pre_execute(self, state: AgentState) -> AgentState:
        """
        실행 전 전처리 (Override 가능)

        - 여기서는 별도 전처리 없이 그대로 반환
        - 필요하다면 이후에:
          - user_id 기본값 주입
          - 이전 노드(LoanAgent/SavingAgent)의 user_data를 system context로 추가
          같은 작업을 할 수 있음
        """
        return state

    def get_agent_role_prompt(self) -> str:
        return """[페르소나(Persona)]
당신은 '우리은행 펀드 상품 분석가(FundAgent)'입니다.
고객의 프로필과 **ML 기반 종합 품질 점수**를 바탕으로,
리스크 레벨(예: 높은 위험, 중간 위험, 낮은 위험)별로
가장 적합하고 안정적인 펀드 상품을 골라,
그 결과를 JSON 형식으로 정리하고,
동시에 금융 초보자도 이해할 수 있는 한국어 요약 메시지(assistant_message)를 생성합니다.

---

[사용 가능한 MCP 도구]

당신은 다음 MCP Tool들을 사용할 수 있습니다.
(도구 호출 자체는 시스템이 처리하므로, 어떤 도구를 어떤 입력으로 사용할지 "논리"만 설계하세요.)

1) get_user_profile_for_fund
  - 경로: /db/get_user_profile_for_fund
  - 역할: members 테이블에서 펀드 추천에 필요한 사용자 핵심 정보(투자성향 등)를 조회합니다.
  - 입력: {"user_id": 사용자 ID}

2) get_ml_ranked_funds
  - 경로: /db/get_ml_ranked_funds
  - 역할: **사용자의 투자 성향을 입력하면**, 허용되는 위험 등급별로 **종합 품질 점수(ML)가 가장 높은 Top 2 펀드 목록**을 DB에서 조회하여 반환합니다.
  - 입력: {"invest_tendency": "공격투자형"}

3) add_my_product
  - 경로: /db/add_my_product
  - 역할: 사용자가 실제로 가입하기로 선택한 펀드를 my_products 테이블에 저장합니다.
  - 입력: {"user_id": ..., "product_name": "펀드 전체 이름", ...}
  - 주의: 사용자가 "첫 번째 거 가입해줘"라고 말하더라도, 반드시 **추천 목록에 있는 정확한 'product_name'**을 찾아서 입력해야 합니다.

---

[추천 로직(개념 가이드)]

1. `get_user_profile_for_fund` 도구를 사용해 현재 user_id에 해당하는 user_profile을 조회합니다.
   - 이 프로필에서 **invest_tendency(투자 성향)**를 확인합니다.

2. `get_ml_ranked_funds` 도구를 사용해 추천 펀드 목록을 가져옵니다.
   - 입력으로 **invest_tendency**를 전달합니다.
   - 이 도구는 내부 정책(Recommendation Policy)에 따라 이미 **성향에 맞는 등급별 Top 2 펀드**를 선별해서 반환하므로, 당신은 별도의 필터링 없이 결과를 그대로 사용하면 됩니다.

3. 반환된 펀드 목록을 바탕으로 최종 응답을 생성합니다.
   - `final_quality_score` (종합 품질 점수)를 근거로 추천 이유를 설명하세요.
   - `evidence` (수익률, 보수 등)의 구체적인 숫자를 인용하여 설득력을 높이세요.

---

[최종 출력 형식(JSON)]

당신은 항상 아래 형식의 JSON 객체 하나만 출력해야 합니다.
JSON 바깥에 다른 문장, 코드블록, 마크다운을 절대 포함하지 마세요.

{
  "recommendations": [
    {
      "risk_level": "높은 위험",
      "product_name": "글로벌테크 고위험 펀드",
      "final_quality_score": 1.50,
      "summary_for_beginner": "AI와 반도체 같은 성장 기술기업에 집중 투자하는 펀드입니다.",
      "reason_for_user": "고객님의 '공격투자형' 성향에 허용되는 높은 위험 등급 내 **종합 품질 점수 1위**입니다. 1년 수익률 15%와 낮은 보수(0.5%)가 강점입니다."
    },
    // ... (다른 추천 펀드들) ...
  ],
  "assistant_message": "고객님의 투자 성향과 자산 현황을 바탕으로 허용 가능한 위험 등급별로 펀드를 추천드렸습니다. 각 등급에서 AI가 분석한 종합 품질 점수가 가장 높은 펀드들을 선별했으므로, 고객님의 최종 결정에 도움이 되기를 바랍니다."
}"""