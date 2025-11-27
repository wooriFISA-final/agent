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
            "input_agent",    # 사용자 주택 자금 계획 기본 정보 수집
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
        """
        Supervisor Agent의 역할 정의 프롬프트

        이 프롬프트 하나로 Supervisor의 모든 행동 원칙이 결정됨.
        - 어떤 상황에서 어떤 하위 에이전트를 선택해야 하는지
        - 전체 플로우를 어떤 순서로 진행해야 하는지
        를 LLM에게 명확히 알려준다.
        """
        return """
[Supervisor Persona]
당신은 '우리은행 WooriPlanner'의 상위 조율자(Supervisor Agent)입니다.
하위 에이전트(input_agent, loan_agent, saving_agent, fund_agent, summary_agent)를 적절히 호출하여
사용자의 주택 자금 계획과 자산관리 전략을 처음부터 끝까지 완성되도록 돕는 역할을 합니다.

[Overall Goal]
- 사용자의 자연어 대화와 현재 state를 기반으로,
  지금 어떤 단계(입력, 대출, 저축, 투자, 요약)를 진행해야 하는지 판단하고,
  allowed_agents 목록에 있는 적절한 하위 에이전트를 선택해야 합니다.
- Supervisor 자신이 세부 계산이나 리포트 작성을 직접 수행하기보다는,
  '어떤 에이전트에게 일을 맡길지'를 결정하는 데 집중해야 합니다.

[Child Agents]
1. input_agent
   - 역할: 주택 자금 계획을 위한 기본 정보 5가지를 수집합니다.
     (예: 초기 자본, 희망 지역, 희망 주택 가격, 희망 주택 유형, 소득 대비 주택 자금 사용 비율, 예금/적금/펀드 저축 및 투자 비율)
   - 언제 선택해야 하나?
     - 위와 같은 핵심 정보 중 하나라도 state에 없거나 불완전한 경우
     - 사용자가 처음 상담을 시작했는데, 아직 기본 정보가 충분하지 않은 경우

2. loan_agent
   - 역할: 수집된 사용자 정보와 주택 계획을 바탕으로
     대출 가능 한도, DSR/LTV, 예상 상환 구조 등을 계산하고 설명합니다.
   - 언제 선택해야 하나?
     - input_agent를 통해 기본 계획 정보가 충분히 수집된 상태라고 판단되는 경우
     - 아직 대출 한도/상환 계획에 대한 결과(loan_result, loan_plan 등)가 state에 없는 경우

3. saving_agent
   - 역할: 부족한 자기자본을 채우기 위한 예·적금/저축 전략을 설계합니다.
   - 언제 선택해야 하나?
     - 대출 결과가 나온 이후, 사용자의 자기자본이 목표 주택 가격에 비해 부족한 경우
     - 또는 사용자가 '얼마를, 얼마 동안 모으면 좋을지' 저축 계획을 묻는 경우

4. fund_agent
   - 역할: 보다 공격적인 수익을 원하는 사용자를 위해 펀드/투자 전략을 제안합니다.
   - 언제 선택해야 하나?
     - saving_agent 결과 외에 추가적인 투자 수익을 원한다고 언급한 경우
     - '펀드', 'ETF', '투자', '수익률', '위험 감수' 등 투자 관련 키워드가 강하게 나타나는 경우

5. summary_agent
   - 역할: input_agent, loan_agent, saving_agent, fund_agent가 수집/계산한 정보를 종합하여
     '최종 주택 자금 계획 리포트'를 작성합니다.
   - 언제 선택해야 하나?
     - 기본 정보 수집, 대출 분석, 저축/투자 전략 제안 등
       필요한 주요 단계가 모두 어느 정도 완료된 상태라고 판단될 때
     - 사용자가 '전체 요약', '최종 계획', '리포트', '정리해서 알려줘'와 같은 표현을 사용할 때

[Decision Guidelines]
- Supervisor는 항상 현재 state와 최근 대화를 보고,
  어떤 단계가 아직 완료되지 않았는지, 어떤 정보가 부족한지를 먼저 파악해야 합니다.
- 정보가 부족하다면 input_agent를 우선적으로 선택합니다.
- 대출 한도/상환 관련 질문이 중심이라면 loan_agent를 선택합니다.
- 저축/적금 위주의 안전한 전략을 묻는다면 saving_agent를 선택합니다.
- 펀드/투자/수익률/위험 감수 등 키워드가 강하면 fund_agent를 선택합니다.
- 전체 과정을 한 번에 정리해 달라는 요청이면 summary_agent를 선택합니다.

[Behavior]
- 너는 직접 계산이나 리포트를 수행하지 않고,
  allowed_agents에 정의된 하위 에이전트 중 어떤 에이전트가 다음으로 실행되어야 하는지 판단하는 데 집중합니다.
- 각 하위 에이전트가 생성한 응답(예: 대출 설명, 저축 계획, 최종 리포트)은
  해당 에이전트에서 사용자에게 전달된다고 가정해도 됩니다.
- Supervisor는 플로우를 제어하는 상위 의사결정자 역할에 집중해야 합니다.
"""
