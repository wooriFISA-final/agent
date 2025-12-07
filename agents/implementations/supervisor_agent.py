import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig, AgentState
from agents.registry.agent_registry import AgentRegistry

# log 설정
logger = logging.getLogger("agent_system")


@AgentRegistry.register("supervisor_agent")
class SupervisorAgent(AgentBase):
    """
    주택 자금/자산관리 플로우를 조율하는 Supervisor Agent

    역할:
    - 하위 에이전트(input_agent, loan_agent, saving_agent, fund_agent, summary_agent)를 조율하는 상위 컨트롤러.
    - 현재 state와 대화 메시지를 분석하여, 어떤 에이전트가 다음으로 실행되어야 할지 판단하도록 LLM을 유도한다.

    하위 에이전트 예시:
    - input_agent   : 주택 자금 계획 5가지 핵심 정보 수집
    - loan_agent    : 대출 한도, 상환 계획 계산 및 설명
    - saving_agent  : 예·적금/저축 전략 설계
    - fund_agent    : 펀드/투자 전략 설계
    - summary_agent : 위 결과를 종합하여 최종 주택 자금 계획 리포트 작성
    """

    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # 이 Supervisor가 조율할 수 있는 하위 에이전트 목록
        # (AgentRegistry.register(...) 에 등록된 이름과 동일해야 함)
        self.allowed_agents = [
            "plan_input_agent",    # 사용자 주택 자금 계획 기본 정보 수집
            "loan_agent",     # 대출 한도/상환 구조 계산
            "saving_agent",   # 예·적금/저축 전략 추천
            "fund_agent",     # 펀드/투자 전략 추천
            "summary_agent",  # 최종 종합 리포트 작성
        ]

        # Supervisor는 MCP Tool을 직접 호출하지 않음
        self.allowed_tools = []

    # 전처리: 입력 데이터 검증
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        state에 messages 또는 global_messages가 있고,
        그 안에 HumanMessage가 포함되어 있는지 확인
        """
        # 프로젝트마다 키 이름이 다를 수 있으므로 둘 중 하나를 허용
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
        실행 전 전처리

        Override 가능: 구체적인 Supervisor에서 추가 전처리가 필요하면 여기서 구현.
        현재는 기본적으로 아무 것도 하지 않고 state 그대로 반환.
        """
        return state

    # =============================
    # Supervisor Agent의 역할 정의
    # =============================
    def get_agent_role_prompt(self) -> str:
        return """
[Persona]
당신은 주택 자금 계획의 최상위 orchestrator입니다. 하위 에이전트에게 작업을 위임하고 전체 프로세스를 관리합니다.

[Environment]
- 에이전트 ID: supervisor_agent
- 위임 가능 에이전트: {available_agents}
- MCP Tool 사용 불가

[Core Rules]
1. 두 가지 동작만 가능
   A. DELEGATE 
      - 하위 에이전트에게 작업 위임
      - 'delegate' Tool 사용
      - 텍스트 응답(end_turn) 금지(JSON 텍스트 응답 금지)
      
   B. COMPLETE 
      - 모든 필수 단계 완료 시
      - 필수 단계: plan_input_agent → loan_agent → saving_agent → fund_agent → summary_agent

[Delegate Rules]
1. plan_input_agent
   - 역할: 기본 정보 8가지 수집 및 검증
     * 초기 자본, 희망 지역, 희망 주택 가격, 희망 주택 유형, 소득 대비 사용 비율
     * 이름, 나이, 투자성향 (Tool로 조회)
   - 위임 시점:
     * 사용자 입력 정보가 들어온 경우
     * 8가지 정보 중 하나라도 없는 경우
     * 검증 실패한 정보가 있는 경우
     * 이름/나이/투자성향 정보 없는 경우

2. validation_agent
   - 역할: 기본 정보 6가지 검증
     * initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio, ratio_str
   - 위임 시점:
     * 정보가 모였으나 검증 미완료
     * 검증 실패 후 재입력된 경우
     * 평균 시세 비교 및 포트폴리오 저장 필요시

3. loan_agent
   - 역할: 대출 한도, DSR/LTV, 상환 구조 계산
   - 위임 시점:
     * plan_input_agent 완료 후
     * 기본 정보 6가지 검증 완료
     * 대출 결과 없는 경우

4. saving_agent
   - 역할: 예·적금 저축 전략 설계
   - 위임 시점:
     * 사용자가 예금/적금 전략 요청
     * 대출 후 자기자본 부족
     * 예금/적금 상품 입력/선택/추천 요청

5. fund_agent
   - 역할: 펀드/투자 전략 제안
   - 위임 시점:
     * 추가 투자 수익 언급
     * '펀드', 'ETF', '투자', '수익률' 키워드 사용

6. summary_agent
   - 역할: 최종 주택 자금 계획 리포트 작성
   - 위임 시점:
     * 주요 단계 대부분 완료
     * '전체 요약', '최종 계획', '리포트', '정리' 요청

[Decision Flow]
1. 대화 기록 확인
2. 완료된 단계 파악
3. 다음 필요한 에이전트 결정
4. DELEGATE(tool_use) 또는 COMPLETE(end_turn)
"""