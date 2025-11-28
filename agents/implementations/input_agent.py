import logging
import asyncio
from typing import List, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agents.base.agent_base import AgentBase
from agents.config.base_config import BaseAgentConfig
from agents.registry.agent_registry import AgentRegistry

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
        # AgentBase가 LLM 설정/llm_config까지 이미 처리해줌
        super().__init__(config)

        # 이 Agent는 MCP Tool을 사용하지 않음
        self.allowed_tools = []
        self.allowed_agents = ["validation_agent"]
        
    # async def run(self, state: dict) -> dict:
    #     """
    #     PlanInputAgent는 ReAct 루프를 쓰지 않고,
    #     단일 턴으로 "질문 1개 or 최종 요약"만 생성하는 간단한 로직을 사용한다.
    #     """

    #     # 1) 입력 검증
    #     if not self.validate_input(state):
    #         logger.error(f"[{self.name}] Invalid input state")
    #         state.setdefault(
    #             "final_response",
    #             "죄송합니다. 입력을 이해하지 못했습니다. 다시 한 번만 말씀해 주시겠어요?",
    #         )
    #         messages = state.get("messages", [])
    #         messages.append(AIMessage(content=state["final_response"], name=self.name))
    #         state["messages"] = messages
    #         return state

    #     messages: List[Any] = state.get("messages", [])

    #     # 2) LLM에 보낼 메시지 구성 (System + history)
    #     system_prompt = self.get_agent_role_prompt()
    #     llm_messages: List[Any] = [SystemMessage(content=system_prompt), *messages]

    #     logger.info(
    #         f"[{self.name}] Generating PlanInputAgent response with {len(messages)} history messages"
    #     )
    #     logger.debug(f"[{self.name}] LLM input messages: {llm_messages}")

    #     # 3) LLM 호출 (AgentBase._call_llm 사용 → 내부에서 LLMHelper.invoke_with_history 호출)
    #     try:
    #         # _call_llm 은 sync 함수라, async 컨텍스트에서 to_thread로 돌려줌
    #         text = await asyncio.to_thread(
    #             self._call_llm,
    #             llm_messages,  # messages
    #             False,         # stream=False
    #             ""             # format="" (자유 텍스트)
    #         )
    #     except Exception as e:
    #         logger.exception(f"[{self.name}] LLM 호출 실패: {e}")
    #         text = "죄송합니다. 잠시 후 다시 시도해 주시겠어요?"

    #     # 4) 최종 응답을 state에 저장
    #     state["final_response"] = text

    #     # messages 리스트에 AIMessage 추가
    #     messages.append(AIMessage(content=text, name=self.name))
    #     state["messages"] = messages

    #     # (선택) global_messages에도 반영해두면 나중에 다른 Agent가 이어받기 편함
    #     global_messages = state.get("global_messages", [])
    #     if not global_messages:
    #         global_messages = []
    #     global_messages.extend([AIMessage(content=text, name=self.name)])
    #     state["global_messages"] = global_messages

    #     # 5) 입력 완료 여부 판단
    #     is_complete = False

    #     # 프롬프트 규칙 6번:
    #     # 요약 문단은 "정리해 보면"으로 시작 → 입력이 다 모였다고 판단
    #     if text.strip().startswith("정리해 보면"):
    #         is_complete = True

    #     state["is_input_complete"] = is_complete
    #     logger.info(f"[{self.name}] is_input_complete set to: {is_complete}")

    #     return state

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
당신은 고객이 제공하는 정보를 처리하는 에이전트입니다.
아래 작성된 TASK와 행동 규칙에 따라 행동하십시오.
---

[TASK: 수집해야 할 핵심 정보 5가지]
아래 5가지 정보를 모두 수집하는 것이 목표입니다.
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

[행동 규칙]
1. 5개 정보가 모두 채워지지 않았다면, 사용자에게 추가 요구사항을 요청하기 위하여 supervisor_agent로 delegate 하십시오. 
2. 5개 정보가 모두 채워졌다고 판단되면, 사용자의 제공한 정보가 맞는지 검증하기 위하여 validation_agent로 delegate 하십시오.
3. 사용자 정보 검증까지 완료하였다면, 현재 사용자가 입력한 정보에 대한 요약을 하기 위하여 supervisor_agent로 delegate 하십시오. 
"""
