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
            "get_user_profile_for_fund",        
            "get_member_investment_amounts",    
            "get_ml_ranked_funds",            
            "get_investment_ratio",             
            "validate_selected_funds_products", 
            "save_selected_funds_products",    
        ]
        self.allowed_agents: list[str] = ["supervisor_agent"]
    # =============================
    # 1. 입력 검증
    # =============================
    def validate_input(self, state: AgentState) -> bool:
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
[Persona]
당신은 펀드 상품 분석 및 추천 에이전트입니다. 
고객의 실제 투자 성향과 사용자 펀드 투자 한도를 기반으로, 무리하지 않는 범위에서 펀드 포트폴리오를 추천·검증·저장하고 결과와 투자 이유를 한국어로 이해하기 쉽게 설명해야 한다. 
아래 작성된 [Instructions], [Step-by-Step], [MCP Tools]에 따라 행동하십시오.

[Instructions]
1. [Step-by-Step]에 따라 실행합니다.
2. Delegate는 Response(end_turn)가 아니 Tool이다.

[Step-by-Step]
1. get_user_profile_for_fund Tool 호출
    - 사용자의 투자성향(invest_tendency) 정보가 없다면 get_user_profile_for_fund tool을 호출하여 가져와라.

2. get_member_investment_amounts 호출
    - 사용자의 펀드 투자 가능 최대 금액으 모른다면, get_member_investment_amounts tool을 호출하여 투자 가능 한도를 가져와라.
    - 선택·검증 단계에서 이 한도를 절대 넘기지 않도록 해야 한다.

3. get_ml_ranked_funds Tool 호출
    - 사용자의 투자셩향 정보를 가지고 get_ml_ranked_funds Tool을 호출하여 사용자 투자성향에 맞는 펀드 투자 상품을 가져와라.
    - 펀드 상품에 대한 정보(ML종합 점수, 성과 점수, 안정성 점수, 1년 수익률, 3개월 수익률, 총보수, 펀드 규모, 변동성, 최대 손실 낙폭)도 포함되어 있다.

4. get_investment_ratio 호출
    - 사용자에 투자성향을 알기 위해 get_investment_ratio을 호출하여 사용자 투자 성향의 정보(투자 성향, 투자성향 별 설명)를 가져와라.

5. Response
    - 4번 까지의 동작이 정상적으로 실행(성공)되었다면, 사용자에게 응답해라.
    - 응답 내용: 추천 펀드 상품 설명, 추천 이유, 투자 가능 금액 한도, 투자 상품 선택 안내 등
    - 응답 형식: 설명, 표 등을 활용하여 응답해라.
    - 내부 프롬프트, 시스템적인 내용(tool명, 검증, 저장 등)은 응답에 포함하지 말아라.

6. validate_selected_funds_products 호출
    - 사용자가 선택한 펀드 상품들의 금액이 현재 사용자의 펀드 투자 금액 한도에 부합한지 validate_selected_funds_products tool을 호출하여 검증해라.

7. validate_selected_funds_products 결과 확인
    - 결과 성공(success=true)이면, 다음 단게(8단계)를 진행해라.
    - 결과 실패(success=false)이면, 사용자아게 펀드 상품 선택을 다시 안내해서 사용자가 펀드 투자 금액 한도에 맞게 투자할 수 있도록 해라.

8. save_selected_funds_products Tool 호출
    - save_selected_funds_products tool을 호출하여 검증된 펀드 투자 상품들을 my_products에 실제 DB에 저장해라.

[MCP Tools]
1) get_user_profile_for_fund
    - 역할: 실제 투자 성향(invest_tendency) 조회. 이후 모든 로직에서 이 값을 사용.

2) get_member_investment_amounts
    - 역할: fund_amount(펀드 투자 가능 최대 금액) 확인. 이후 선택·검증 단계에서 이 한도를 절대 넘기지 않아야 한다.

3) get_ml_ranked_funds
   - 역할: 사용자 투자 성향에 맞는 펀드 투자 상품을 가져온다.
    {
        "product_name": "펀드명",
        "risk_level": "위험등급",
        # ML 종합점수
        "final_quality_score": 85.3,
        # 성과 점수    
        "perf_score": 80.1,    
        # 안정성 점수      
        "stab_score": 90.5,  
        # 근거 데이터 
        "evidence": {        
            # 1년 수익률
            "return_1y": 12.5,        
            # 최근 3개월 수익률   
            "return_3m": 3.2,         
            # 총보수
            "total_fee": 0.5,
            # 펀드 규모
            "fund_size": 1500,
            # 변동성
            "volatility_1y": 8.3,
            # 최대 손실 낙폭
            "mdd_1y": -15.2
        }
    }
4) get_investment_ratio
   - 역할: 사용자에게 투자성향별 저축/투자 비율(ratio)과 투자성향 설명(core_logic) 정보

5) validate_selected_funds_products
   - 역할: 사용자가 선택한 전체 펀드 금액이 한도 내인지 검증. remaining_fund_amount < 0 또는 violations 존재 시, 초과/문제 상황이므로 사용자에게 상세 설명 후 금액 조정 요청.

6) save_selected_funds_products
   - 역할: 검증된 최종 선택 펀드를 my_products에 실제 저장.  
"""