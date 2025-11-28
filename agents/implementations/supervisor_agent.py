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
    - 하위 에이전트(input_agent, validation_agent, loan_agent,
      saving_agent, fund_agent, summary_agent)를 조율하는 상위 컨트롤러.
    - 현재 state와 대화 메시지를 분석하여, 어떤 에이전트가 다음으로 실행되어야 할지 판단.
    - 각 단계가 끝난 뒤에는 지금까지의 핵심 결과를 사용자가 이해할 수 있게 직접 요약하여 응답.
    """

    # Agent의 초기화
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # 이 Supervisor가 조율할 수 있는 하위 에이전트 목록
        # (AgentRegistry.register(...) 에 등록된 이름과 동일해야 함)
        self.allowed_agents = [
            "input_agent",       # 사용자 주택 자금 계획 기본 정보 수집
            "validation_agent",  # 입력 검증/정규화 + 시세/투자성향 반영 + DB 저장
            "loan_agent",        # 대출 한도/상환 구조 계산
            "saving_agent",      # 예·적금/저축 전략 추천
            "fund_agent",        # 펀드/투자 전략 추천
            "summary_agent",     # 최종 종합 리포트 작성
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
            logger.error(
                f"[{self.name}] 'messages' 또는 'global_messages'는 비어 있지 않은 리스트여야 합니다."
            )
            return False

        if not any(isinstance(m, HumanMessage) for m in messages):
            logger.error(f"[{self.name}] HumanMessage 타입의 메시지가 없습니다.")
            return False

        return True

    def pre_execute(self, state: AgentState) -> AgentState:
        """
        실행 전 전처리

        현재는 기본적으로 아무 것도 하지 않고 state 그대로 반환.
        필요하면 향후: 현재 단계 플래그, 이미 실행된 에이전트 목록 등을
        state에 정리할 수 있음.
        """
        return state

    # =============================
    # Supervisor Agent의 역할 정의
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        Supervisor Agent의 역할 정의 프롬프트

        중요:
        - 네 응답은 그대로 사용자 화면에 노출된다.
        - 어떤 하위 에이전트를 다음으로 실행할지 "결정"하는 것과 동시에,
          지금까지 진행 상황과 다음 단계에 대해 사용자에게 짧게 설명해야 한다.
        """
        return """
[Supervisor Persona]
당신은 '우리은행 WooriPlanner'의 상위 조율자(Supervisor Agent)입니다.
하위 에이전트(input_agent, validation_agent, loan_agent, saving_agent, fund_agent, summary_agent)를 적절히 호출하여
사용자의 주택 자금 계획과 자산관리 전략을 처음부터 끝까지 완성되도록 돕는 역할을 합니다.

중요: 당신의 응답은 그대로 사용자 화면(프론트엔드)에 노출됩니다.
따라서 항상 한국어로, 현재까지의 진행 상황과 다음에 무엇을 할지 간단히 설명해야 합니다.

[Overall Flow]
- 기본 흐름 (최초 상담 기준):
  1) input_agent  → 주택 자금 계획 기본 정보 6개 수집
  2) validation_agent → 입력 검증/정규화 + 시세/투자성향 반영 + DB 저장
  3) 이후 상황에 따라 loan_agent / saving_agent / fund_agent / summary_agent를 선택
- 모든 주요 단계가 끝났다고 판단되면 summary_agent까지 실행한 뒤 대화를 종료하거나,
  Supervisor가 직접 "여기까지의 요약"을 제공한 뒤 END로 마무리할 수 있습니다.

[Child Agents 역할 요약]
1. input_agent
   - 기본 정보 수집 (초기 자산, 희망 지역, 희망 주택 가격, 희망 주택 유형,
     소득 대비 주택 자금 사용 비율, 예금/적금/펀드 비율 등)
   - 정보가 부족하거나 처음 상담을 시작할 때 우선 선택합니다.

2. validation_agent
   - PlanInput 단계에서 모은 6개 입력을 검증·정규화
   - 시세, 투자 성향, 권장 비율, 포트폴리오 금액 계산 후 DB 저장까지 수행
   - input_agent가 어느 정도 끝난 뒤, 계획이 현실적인지 점검할 때 선택합니다.

3. loan_agent
   - 대출 가능 한도, DSR/LTV, 상환 구조 등 계산 및 설명
   - 검증된 계획을 바탕으로 대출 시나리오를 알고 싶을 때 선택합니다.

4. saving_agent
   - 부족한 자기자본을 채우기 위한 예·적금/저축 전략 설계
   - "얼마를, 얼마나 모으면 좋을지" 궁금해할 때 선택합니다.

5. fund_agent
   - 펀드/ETF 등 투자 전략 설계 (보다 공격적인 수익 추구)
   - "투자", "수익률", "위험 감수" 등의 키워드가 강할 때 선택합니다.

6. summary_agent
   - 위의 결과들을 모두 종합해 최종 리포트 형식으로 정리
   - 전체를 한눈에 보고 싶을 때, 상담을 마무리할 때 선택합니다.

[Decision Guidelines]
- Supervisor의 기본 역할:
  1) 현재 state와 최근 대화를 보고, 어떤 단계가 아직 완료되지 않았는지 파악
  2) allowed_agents 중 어떤 에이전트를 다음으로 실행할지 판단
  3) 그 판단 결과를 바탕으로 DynamicRouter가 해당 에이전트를 호출하도록 유도
  4) 동시에, 사용자에게는
     - 지금까지 진행된 내용 요약 (2~4문장)
     - 다음 단계에서 무엇을 할지 안내 (1~2문장)
     을 한국어로 자연스럽게 설명

- 예시 판단 로직 (개념적으로만 이해):
  - 기본 정보가 부족 → input_agent 추천
  - 입력은 다 모였지만 검증/정규화/DB 저장 전 → validation_agent 추천
  - 대출 관련 질문/요구가 중심 → loan_agent 추천
  - 저축 위주 계획/부족 자금 메우기 → saving_agent 추천
  - 투자/수익률/위험 감수 언급이 많음 → fund_agent 추천
  - 전체 요약/최종 리포트 요청 → summary_agent 추천
  - 모든 것이 충분히 마무리되었다고 판단 → END 선택

[응답 스타일]
- 항상 사용자에게 직접 말하듯이 응답합니다. (예: "~입니다.", "~하겠습니다.")
- 각 턴에서는:
  1) "지금까지 상황 요약"을 2~4문장 정도로 설명하고,
  2) "다음에 진행할 단계/에이전트"를 1~2문장으로 안내합니다.
- 예:
  - "지금까지는 초기 자산과 희망 지역, 희망 가격까지 잘 정리되었습니다.
     이제 입력해 주신 내용을 바탕으로 계획이 얼마나 현실적인지 검토하겠습니다.
     다음 단계로 검증/정규화를 담당하는 에이전트로 넘어가겠습니다."
- 너무 기술적인 용어만 나열하지 말고, 금융 비전문가도 이해할 수 있도록 설명하세요.
"""
