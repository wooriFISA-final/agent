"""
Agent Template for New Implementations
-------------------------------------
이 템플릿은 새로운 Agent를 추가할 때 표준 구조를 제공합니다.

✅ 포함 내용
- Agent 등록 및 기본 실행 구조
- 입력 검증(validate_input)
- 로깅 일관성
- 예외 처리 및 상태 업데이트
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agents.base.agent_base import AgentBase
from agents.config.config_loader import BaseAgentConfig  # 공통 설정 클래스
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager


# 공용 로거
logger = logging.getLogger("agent_system")


@AgentRegistry.register("template_agent")
class TemplateAgent(AgentBase):
    """
    🧠 TemplateAgent
    ----------------
    새로운 에이전트를 추가할 때 이 클래스를 복사하여 사용하세요.

    입력:
        - query (str): 사용자 입력 또는 요청 내용

    출력:
        - result (str): Agent의 처리 결과
        - messages (str): 로그 또는 요약 정보
    """

    def __init__(self, config: BaseAgentConfig):
        """
        초기화 메서드
        """
        super().__init__(config)
        self.llm = LLMManager.get_llm(
            provider=getattr(config, "provider", "ollama"),
            model=config.model_name
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        메인 실행 함수
        ----------------
        Args:
            state (dict): 그래프 상에서 전달되는 상태 값
        
        Returns:
            dict: 처리 후 업데이트된 상태
        """
        try:
            # 문자열이 들어온 경우 보정
            if isinstance(state, str):
                state = {"query": state}

            query = state.get("query", "")
            logger.info(f"🚀 [{self.name}] Executing agent with query: {query}")

            # 시스템 프롬프트 구성
            system_prompt = """당신은 특정 작업을 수행하는 AI 에이전트입니다.
입력된 정보를 분석하고 적절한 결과를 반환하세요."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"사용자 입력: {query}")
            ]

            # LLM 호출
            response = await self.llm.ainvoke(messages)
            output = response.content.strip()

            # 상태 업데이트
            state["result"] = output
            state["messages"] = f"[{self.name}] 작업 완료"

            logger.info(f"✅ [{self.name}] Execution complete.")
            return state

        except Exception as e:
            logger.error(f"❌ [{self.name}] Execution failed: {e}")
            state["result"] = f"오류 발생: {str(e)}"
            state["messages"] = f"[{self.name}] 실패: {str(e)}"
            return state

    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        입력 검증 함수
        ----------------
        Args:
            state (dict): 그래프 노드로 전달되는 상태
        
        Returns:
            bool: 유효하면 True
        """
        if not isinstance(state, dict):
            logger.error(f"[{self.name}] ❌ Invalid state type: {type(state)}")
            return False

        if "query" not in state:
            logger.error(f"[{self.name}] ❌ Missing key 'query' in state")
            return False

        if not isinstance(state["query"], str) or not state["query"].strip():
            logger.error(f"[{self.name}] ❌ 'query' must be a non-empty string")
            return False

        logger.debug(f"[{self.name}] ✅ Input validation passed")
        return True
