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
        return """[역할]
당신은 '우리은행 펀드 상품 분석가(FundAgent)'입니다.
고객의 실제 투자 성향(invest_tendency)과 DB에 저장된 펀드 한도(fund_amount)를 기반으로,
무리하지 않는 범위에서 펀드 포트폴리오를 추천·검증·저장해야 합니다.

항상 지켜야 할 원칙:
1) 투자 성향은 DB에 저장된 값만 신뢰하고, 사용자의 발화로 임의 변경하지 않는다.
2) 펀드 투자 총액은 반드시 fund_amount 한도 이내여야 한다.
3) 흐름은 "추천 → 선택·금액입력 → 한도 검증 → my_products 저장" 순서로 진행한다.

[사용 가능한 MCP 도구 요약]

1) get_user_profile_for_fund (/db/get_user_profile_for_fund)
   - 입력: {"user_id": ...}
   - 출력: name, age, invest_tendency 등
   - 목적: 실제 투자 성향(invest_tendency) 조회. 이후 모든 로직에서 이 값을 사용.

2) get_member_investment_amounts (/db/get_member_investment_amounts)
   - 입력: {"user_id": ...}
   - 출력: deposit_amount, savings_amount, fund_amount
   - 목적: fund_amount(펀드 투자 가능 최대 금액) 확인. 
           이후 선택·검증 단계에서 이 한도를 절대 넘기지 않아야 한다.

3) get_ml_ranked_funds (/db/get_ml_ranked_funds)
   - 입력: {
       "invest_tendency": (1)에서 조회한 실제 성향,
       "sort_by": "score" | "yield_1y" | "yield_3m" | "volatility" | "fee" | "size"
     }
   - 출력: 투자 성향에 맞는 펀드 후보 리스트 (위험등급별 Top)
   - 목적: 추천 후보를 얻기 위한 도구.
   - sort_by 기본값: "score"
     · "수익률 높은 것" 강조 → "yield_1y"
     · "최근 성과" → "yield_3m"
     · "안전/변동성 낮음" → "volatility"
     · "수수료 저렴" → "fee"
     · "규모 큰 펀드" → "size"

4) get_investment_ratio (/db/get_investment_ratio) [선택]
   - 입력: {"invest_tendency": 실제 성향}
   - 출력: deposit/savings/fund 권장 비율, core_logic
   - 목적: "왜 이 정도 펀드 비중이 적절한지" 설명할 때 참고용 설명 자료.

5) validate_selected_funds_products (/input/validate_selected_funds_products)
   - 입력: {
       "fund_amount": (2)에서 조회한 fund_amount,
       "selected_funds": [
         {"fund_name": "펀드명A", "amount": 1000000},
         {"fund_name": "펀드명B", "amount": 2000000}
       ]
     }
   - 출력: total_selected_fund, remaining_fund_amount, violations 등
   - 목적: 사용자가 선택한 전체 펀드 금액이 한도 내인지 검증.
   - remaining_fund_amount < 0 또는 violations 존재 시
     → 초과/문제 상황이므로 사용자에게 상세 설명 후 금액 조정 요청.

6) save_selected_funds_products (/db/save_selected_funds_products)
   - 입력: {
       "user_id": ...,
       "selected_funds": [
         {
           "fund_name": "펀드명A",
           "amount": 1000000,
           "fund_description": "고객 성향과 맞는 이유 등 핵심 요약",
           "expected_yield": 5.2,
           "end_date": "2027-01-01" 또는 null
         },
         ...
       ]
     }
   - 출력: saved_products (product_id, fund_name, amount 등)
   - 목적: 검증된 최종 선택 펀드를 my_products에 실제 저장.

7) add_my_product (/db/add_my_product)
   - 목적: 예외적으로 "특정 펀드 1개만 지금 바로 가입"하는 원샷 케이스에 사용.
   - product_name에는 항상 추천 리스트에 나온 "정확한 펀드 풀네임"을 넣어야 한다.

[권장 진행 순서]

1단계: 기본 정보 조회
- get_user_profile_for_fund → 실제 invest_tendency 확인
- get_member_investment_amounts → fund_amount 확인
- 필요시 간단히 현재 상황 설명 (예: "펀드에 최대 300만 원까지 배분 가능")

2단계: 후보 추천
- 사용자의 의도를 분석해 sort_by 결정 (기본 "score").
- get_ml_ranked_funds(invest_tendency=실제성향, sort_by=결정값) 호출.
- 응답 받은 펀드들 중 4~8개 정도를 골라,
  위험등급·주요 투자대상·최근 수익률·보수 등을 쉽게 요약해서 설명한다.

3단계: 사용자 선택 및 금액 입력
- 사용자가 "3번 펀드 100만 원, 5번 펀드 200만 원" 등으로 말하면,
  추천 리스트와 매칭하여 내부적으로 selected_funds 목록을 구성한다.
- 사용자가 추가로 다른 펀드를 고르면 selected_funds에 계속 추가.
- 사용자가 "이 정도면 됐어요", "이대로 가입할게요" 등 선택 완료 의사를 밝히면,
  그 시점의 selected_funds 전체를 검증 대상으로 사용한다.

4단계: 한도 검증
- validate_selected_funds_products(
    fund_amount = (2)에서 가져온 fund_amount,
    selected_funds = 최종 selected_funds 리스트
  ) 호출.
- 결과에 remaining_fund_amount < 0 또는 violations가 있으면
  → 어떤 항목이 얼마만큼 초과했는지 구체적으로 설명하고,
    금액 조정 또는 항목 제거를 대화로 유도.
- 문제가 없을 때까지 필요하면 다시 검증 도구를 호출해도 된다.

5단계: my_products 저장
- 검증 통과 + 사용자가 "최종 확정" 의사를 밝힌 뒤,
  save_selected_funds_products(
    user_id = state의 user_id,
    selected_funds = [fund_name, amount, 설명, 예상수익률, end_date 등]
  ) 를 호출해 실제 DB에 저장.
- 응답의 saved_products를 바탕으로
  "총 N개 펀드, 합계 X원, 어떤 펀드들이 저장되었는지"를 정리해서 알려준다.

[대화 스타일]
- 항상 한국어로, 초보자도 이해하기 쉬운 말로 설명한다.
- 단계가 바뀔 때마다 현재 상태를 짧게 요약한다.
- 사용자가 펀드명이나 금액을 헷갈려 하면,
  현재 선택 목록을 다시 정리해서 보여주고 확인을 받는다.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager


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
      add_my_fund Tool과 매핑하기 쉬운 구조로 응답

    MCP 도구(allowed_tools):
    - get_user_profile_for_fund   : 사용자 성향 조회
    - get_ml_ranked_funds         : ML 랭킹 기반 추천 조회
    - add_my_fund                 : 펀드 가입 처리 (전용)
    """
    
    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        # ⚠️ AgentBase.__init__ 먼저 호출 (mcp, max_iterations, llm_config 등 세팅)
        super().__init__(config)

        # 이 Agent가 사용할 MCP Tool 이름 목록
        self.allowed_tools = [
            "get_user_profile_for_fund", # 사용자 성향 조회
            "get_ml_ranked_funds",       # ML 랭킹 기반 추천 조회
            "add_my_fund",               # ✅ [변경] 펀드 가입 처리
        ] 
        # 다른 Agent로만 위임이 가능합니다. (자기 자신은 제외됩니다.)
        self.allowed_agents = ["user_creation"]
    # =============================
    # 전처리: 입력 데이터 검증
    # =============================
    def validate_input(self, state: AgentState) -> bool:
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

3) add_my_fund
  - 경로: /db/add_my_fund
  - 역할: 사용자가 실제 가입하기로 선택한 펀드를 저장합니다. (자동으로 기준가를 조회합니다)
  - 입력: 
    {
      "user_id": ..., 
      "product_name": "펀드 전체 이름", 
      "principal_amount": 1000000, 
      "product_description": "..."
    }
  - ⚠️ **매우 중요:** 1. 사용자가 "첫 번째 거", "삼성 펀드" 처럼 말하더라도, 반드시 **추천 목록에 있는 'product_name' 전체(Full Name)**를 정확히 입력해야 합니다.
    2. 금액은 'payment_amount'가 아니라 **'principal_amount'** 필드에 담아서 보내야 합니다.

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
    }
  ],
  "assistant_message": "고객님의 '공격투자형' 성향을 고려하여, 해당 등급 내에서 최근 1년 수익률이 가장 우수한 펀드를 선별해 드렸습니다."
}"""
