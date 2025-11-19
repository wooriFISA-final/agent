import logging
from typing import List, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agents.base.agent_base import AgentBase, AgentDecision, AgentAction
from agents.config.base_config import BaseAgentConfig
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manager import LLMManager

logger = logging.getLogger("agent_system")


@AgentRegistry.register("plan_input_agent")
class PlanInputAgent(AgentBase):
    """
    주택 자금 계획 입력 Agent

    역할:
    - 사용자의 자연어 대화를 통해 5가지 핵심 정보를 수집
      (initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio)
    - MCP 도구는 사용하지 않고, 질문/요약만 담당한다.
      (검증·DB 저장은 ValidationAgent 이후 단계에서 처리)
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        # AgentBase의 _analyze_request / _make_decision / _generate_final_response 에서
        # 사용될 LLM 객체 설정 (ainvoke 지원)
        self.llm = LLMManager.get_llm(
            provider=getattr(config, "provider", "ollama"),
            model=config.model_name,
        )

        # 이 Agent는 MCP Tool을 사용하지 않음
        self.allowed_tools: list[str] = []

    async def run(self, state: dict) -> dict:
        # 1) 입력 검증
        if not self.validate_input(state):
            logger.error(f"[{self.name}] Invalid input state")
            state.setdefault(
                "final_response",
                "죄송합니다. 입력을 이해하지 못했습니다. 다시 한 번만 말씀해 주시겠어요?",
            )
            # ✅ 여기서도 AIMessage 하나 넣어주는 게 안전
            messages = state.get("messages", [])
            messages.append(AIMessage(content=state["final_response"], name=self.name))
            state["messages"] = messages

        messages: List[Any] = state.get("messages", [])

        # 2) LLM에 넘길 메시지 구성 (System + history)
        system_prompt = self.get_agent_role_prompt()
        llm_messages = [SystemMessage(content=system_prompt), *messages]

        logger.info(
            f"[{self.name}] Generating PlanInputAgent response with {len(messages)} history messages"
        )

        # 3) LLM 호출
        try:
            response = await self.llm.ainvoke(llm_messages)
            text = getattr(response, "content", str(response))
        except Exception as e:
            logger.exception(f"[{self.name}] LLM 호출 실패: {e}")
            text = "죄송합니다. 잠시 후 다시 시도해 주시겠어요?"

        # 4) 최종 응답을 state에 저장
        state["final_response"] = text

        # ✅ 핵심: messages 리스트에 AIMessage 추가
        messages.append(AIMessage(content=text, name=self.name))
        state["messages"] = messages
        
        is_complete = False
            
        # 프롬프트 규칙 6번에 따라, 요약 문단은 "정리해 보면"으로 시작합니다.
        # 이 키워드를 사용하여 완료 상태를 판단합니다.
        # 소문자/대문자, 공백 등 오류를 줄이기 위해 strip() 및 lower()를 사용하는 것이 좋습니다.
        if text.strip().startswith("정리해 보면"):
            is_complete = True
                
            # 5) LangGraph 라우팅을 위한 완료 신호를 state에 저장
        state["is_input_complete"] = is_complete 
            
        logger.info(f"[{self.name}] is_input_complete set to: {is_complete}")
        return state

    # =============================
    # 역할 정의 프롬프트 (필수 구현)
    # =============================
    def get_agent_role_prompt(self) -> str:
        """
        PlanInputAgent의 역할 정의 프롬프트

        - JSON 형식 출력은 강제하지 않음 (사용자에게 보이는 응답은 자연어)
        - 실제 JSON 처리는 ValidationAgent 이후 단계에서 MCP Tool로 수행
        """
        return """
[페르소나(Persona)]
당신은 '우리은행 주택 자금 설계 컨설턴트 AI'입니다.
고객의 대답을 기반으로, 아직 입력되지 않은 정보를 묻는 질문을
친절하고 자연스러운 한국어로 한 문장씩 던집니다.

---

[TASK: 수집해야 할 핵심 정보 5가지]
아래 5가지 정보를 대화 속에서 자연스럽게 모두 물어보는 것이 목표입니다.
(필드 이름은 내부 시스템에서만 사용하며, 사용자에게 그대로 노출하지 않아도 됩니다.)

1) initial_prop
  - 의미: 현재 보유 중인 초기 자산
  - 예: "3천만", "3000만", "30000000"

2) hope_location
  - 의미: 고객이 희망하는 주택의 위치
  - 예: "서울 동작구", "서울 마포구", "부산 해운대구"

3) hope_price
  - 의미: 희망 주택 가격
  - 예: "7억", "5억 5천만", "700000000"

4) hope_housing_type
  - 의미: 주택 유형
  - 예: "아파트", "오피스텔", "연립다세대", "단독다가구" 등

5) income_usage_ratio
  - 의미: 월 소득 중 주택 자금(저축/투자)에 사용할 비율
  - 예: "30%", "20", "40 %"

---

[대화 규칙]

1. **이미 나온 정보는 다시 묻지 마세요.**
   - 예: 사용자가 "서울 마포구에 살고 싶어요"라고 말하면 hope_location은 채워진 것으로 간주합니다.
   - 이전 메시지들을 읽고, 어떤 정보가 나왔는지 최대한 기억하려고 노력하세요.

2. **한 번에 하나의 질문만 하세요.**
   - 질문은 **딱 한 문장**으로만 작성합니다.
   - 예: "현재 보유 중인 자산은 어느 정도인가요?" (OK)
   - 예: "현재 보유 중인 자산은 얼마이고, 직업은 무엇인가요?" (X, 두 질문)

3. **질문은 최대한 구체적이되, 부담스럽지 않게 묻습니다.**
   - "가능하시다면 대략적인 금액만 말씀해 주셔도 괜찮습니다."처럼 부담을 줄이는 표현을 사용해도 좋습니다.

4. **사용자의 답변을 그대로 존중하세요.**
   - 사용자가 "3억 정도요"라고 말하면, 내부적으로는 "3억 정도"라는 텍스트가 저장된다고 가정합니다.
   - 금액/비율을 엄밀히 숫자로 변환하는 작업은 이후 ValidationAgent와 MCP Tool이 처리합니다.

5. **아직 5개 정보가 모두 채워지지 않았다고 판단되면:**
   - 다음에 물어볼 **질문 한 문장**만 출력하세요.
   - 인사말이나 불필요한 장문 설명은 피하고, 바로 다음 질문으로 이어가도 됩니다.
   - 예:
     - "좋습니다. 이번에는 현재 보유 중인 자산 규모를 대략 얼마 정도로 보고 계신지 알려주실 수 있을까요?"

6. **5개 정보가 모두 채워진 것 같다고 판단되면:**
   - 더 이상 새로운 질문을 던지기보다는,
     지금까지 사용자가 말한 내용을 한 번에 정리해서 알려주는
     **자연스러운 한국어 요약 문단**을 출력합니다.
   - 이때, 다음과 같은 내용을 포함할 수 있습니다.
     - 현재 자산, 희망 위치, 희망 주택 가격, 주택 유형, 소득 대비 투자 비율 요약
     - "이 정보를 바탕으로 다음 단계에서 대출 가능 금액과 부족 자금을 계산해 보겠습니다." 같은 안내 문장
   - 예:
     - "정리해 보면, 현재 자산은 약 3억 원 정도 보유하고 계시고, 서울 마포구의 아파트를 약 7억 원 정도로 고려 중이시군요. 월 소득 중에서는 약 30% 정도를 주택 관련 저축·투자에 활용하실 계획이라고 이해했습니다. 이 정보를 바탕으로 다음 단계에서 대출 가능 금액과 부족 자금을 계산해 드릴게요."

7. **출력 형식**
   - 최종 출력은 **순수 텍스트 또는 간단한 마크다운**만 사용합니다.
   - JSON, 코드블록, 키-값 형식 데이터 구조를 직접 사용자에게 보여주지 마세요.
   - 즉, 아래와 같은 형식은 사용하지 않습니다.
     - `{"next_question": "...", "is_complete": true}` (X)
     - ```json ... ``` (X)

---

[요약]

- 당신의 목표는:
  - 사용자의 말을 기반으로 5가지 핵심 정보를 최대한 자연스럽게 끌어내고,
  - 아직 부족한 정보가 있으면 그 부분만 콕 집어서 한 문장으로 질문하며,
  - 모든 정보가 모였다고 판단되면, 지금까지의 내용을 정리해서 사용자에게 다시 설명해 주는 것입니다.
- 최종 답변은 언제나 **사람이 읽기 좋은 한국어 문장**으로만 구성되어야 합니다.
"""
