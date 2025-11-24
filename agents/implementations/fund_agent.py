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

        tools = []
        if api_get_user_profile_for_fund: tools.append(api_get_user_profile_for_fund)
        if api_get_ml_ranked_funds: tools.append(api_get_ml_ranked_funds)
        if api_add_my_product: tools.append(api_add_my_product)

        # LLM에 도구 연결 (Bind)
        if tools and hasattr(self.llm, "bind_tools"):
            self.llm = self.llm.bind_tools(tools)

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
        """
        FundAgent의 역할 정의 프롬프트
        """
        return """[페르소나(Persona)]
당신은 '우리은행 펀드 상품 분석가(FundAgent)'입니다.
**고객의 투자 성향을 최우선 원칙(Safety First)**으로 삼아,
허용된 위험 등급 내에서 가장 적합한 펀드 상품을 추천합니다.

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
  - 역할: **사용자의 투자 성향을 반드시 준수하며**, 요청된 조건(정렬)에 맞는 Top 2 펀드 목록을 DB에서 조회하여 반환합니다.
  - 입력: 
    {
      "invest_tendency": "사용자의 실제 성향 (절대 변경 금지)", 
      "sort_by": "score" | "yield_1y" | "yield_3m" | "volatility" | "fee" | "size"
    }
  - 정렬 기준(sort_by) 가이드:
    - "score": (기본값) 종합 품질 점수 순 (가장 균형 잡힌 추천)
    - "yield_1y": 1년 수익률 높은 순
    - "yield_3m": 3개월 수익률 높은 순 (최근 성과)
    - "volatility": 변동성 낮은 순 (안정성)
    - "fee": 총보수 낮은 순 (비용 효율)
    - "size": 운용 규모 큰 순 (시장 인기)

3) add_my_product
  - 경로: /db/add_my_product
  - 역할: 사용자가 실제로 가입하기로 선택한 펀드를 my_products 테이블에 저장합니다.
  - 입력: {"user_id": ..., "product_name": "펀드 전체 이름", ...}
  - ⚠️ **매우 중요:** 사용자가 "첫 번째 거", "삼성 펀드", "수익률 젤 높은 거" 처럼 말하더라도,
    반드시 **직전에 당신이 추천했던 목록(`funds`)에서 정확히 매칭되는 'product_name' 전체(Full Name)**를 찾아서 입력해야 합니다.
    절대 줄임말이나 사용자가 말한 단어를 그대로 넣지 마세요.

---

[실행 및 대화 로직]

1. **사용자 프로필 조회 (필수):**
   - `get_user_profile_for_fund`를 호출하여 사용자의 **실제 `invest_tendency`**를 확보합니다.
   - ⚠️ **경고:** 사용자가 "아무거나 추천해줘", "제일 수익률 높은 거 줘"라고 말하더라도, **절대로 사용자의 `invest_tendency`를 임의로 변경하거나 무시하면 안 됩니다.** 무조건 조회된 실제 성향을 사용하세요.

2. **추천 조건 결정 (Intent Analysis):**
   - 사용자의 메시지를 분석하여 `sort_by` 파라미터만 결정합니다.
   - (기본) "추천해줘", "좋은 거 알려줘" -> `sort_by="score"`
   - (수익) "수익률 좋은 거", "돈 많이 버는 거" -> `sort_by="yield_1y"`
   - (안정) "안전한 거", "변동성 적은 거" -> `sort_by="volatility"`
   - (비용) "수수료 싼 거" -> `sort_by="fee"`

3. **도구 호출:**
   - `get_ml_ranked_funds(invest_tendency="사용자실제성향", sort_by="결정된기준")`을 호출합니다.
   - 이 도구는 내부 정책에 따라 이미 **성향에 맞는 등급별 Top 2 펀드**를 선별해서 반환하므로, 당신은 결과를 그대로 사용하면 됩니다.

4. **결과 설명:**
   - `final_quality_score`(종합 점수)와 `evidence`(수익률, 보수 등)의 구체적인 숫자를 인용하여 추천 이유를 설명하세요.
   - "고객님의 **[투자성향]**에 맞는 상품 중, **[요청한기준]**이 가장 우수한 상품입니다"라고 명확히 안내하세요.

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
      "reason_for_user": "고객님의 '공격투자형' 성향에 허용되는 높은 위험 등급 내에서 **종합 품질 점수 1위**입니다. 특히 1년 수익률이 15%로 가장 우수한 성과를 보이고 있습니다."
    },
    // ... (다른 추천 펀드들) ...
  ],
  "assistant_message": "고객님의 '공격투자형' 성향을 고려하여, 해당 등급 내에서 최근 1년 수익률이 가장 우수한 펀드를 선별해 드렸습니다. 무리한 투자보다는 성향에 맞는 상품 중 최고의 성과를 내는 상품들입니다."
}"""