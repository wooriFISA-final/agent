from abc import ABC, abstractmethod
import asyncio
import json
import re
from typing import Any, Dict, Optional, List
from agents.config.base_config import BaseAgentConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from core.mcp.mcp_manager import MCPManager
import logging

logger = logging.getLogger(__name__)


class AgentBase(ABC):
    """
    모든 Agent가 상속받는 공통 베이스 클래스
    
    공통 기능:
    - MCP 도구 목록 조회
    - LLM 호출 및 JSON 파싱
    - MCP 도구 실행
    - 재시도 및 타임아웃 처리
    
    Agent별 구현 필요:
    - get_system_prompt(): 시스템 프롬프트 정의
    - process_tool_result(): 도구 실행 결과 후처리 (선택)
    """

    def __init__(self, config: BaseAgentConfig):
        self.name = config.name
        self.config = config
        self.mcp = MCPManager()
        self._validate_config()

    # =============================
    # 공통 실행 파이프라인
    # =============================
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 실행 메인 플로우"""
        self._log_start(state)

        if not self.validate_input(state):
            raise ValueError(f"Invalid input for {self.name}")

        state = self.pre_execute(state)

        # 재시도 로직
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with asyncio.timeout(self.config.timeout):
                    result = await self.execute(state)
                break
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout} seconds"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                if attempt == self.config.max_retries:
                    raise TimeoutError(f"{self.name} execution timed out")
                await asyncio.sleep(1.5 * attempt)
            except Exception as e:
                logger.warning(f"[{self.name}] attempt {attempt} failed: {e}")
                if attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(1.5 * attempt)

        self._log_end(result)
        return result

    # =============================
    # 공통 실행 로직 (템플릿 메서드)
    # =============================
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        공통 실행 플로우:
        1. 사용자 메시지 추출
        2. MCP 도구 목록 조회
        3. LLM 호출 (도구 선택 + 인자 생성)
        4. MCP 도구 실행
        5. 결과 후처리
        """
        messages = state.get("messages", [])
        
        # 1️⃣ 사용자 메시지 추출
        user_message = self._extract_user_message(messages)
        if not user_message:
            return {"messages": messages, "error": "No HumanMessage found"}

        # 2️⃣ MCP 도구 목록 조회
        tools = await self._list_mcp_tools()
        if not tools:
            return {"messages": messages, "error": "No MCP tools available"}
        
        logger.info(f"User message: {user_message}")
        
        # 3️⃣ LLM 호출하여 도구 선택
        tool_selection = await self._select_tool_with_llm(user_message, tools, messages)
        if not tool_selection:
            return {"messages": messages, "error": "Failed to select tool"}

        # 4️⃣ MCP 도구 실행
        tool_result = await self._execute_mcp_tool(
            tool_selection["tool"],
            tool_selection["arguments"]
        )

        # 5️⃣ 결과 후처리 (Agent별 커스터마이징 가능)
        final_state = await self.process_tool_result(
            state, tool_result, user_message
        )

        return final_state

    # =============================
    # 공통 헬퍼 메서드
    # =============================
    def _extract_user_message(self, messages: List) -> Optional[HumanMessage]:
        """메시지 리스트에서 가장 최근 HumanMessage 추출"""
        return next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None
        )

    async def _list_mcp_tools(self) -> List:
        """MCP 서버의 사용 가능한 도구 목록 조회"""
        try:
            tools = await self.mcp.list_tools()
            tool_names = [t.name for t in tools]
            logger.info(f"[{self.name}] Available MCP tools: {tool_names}")
            return tools
        except Exception as e:
            logger.error(f"[{self.name}] Failed to list MCP tools: {e}")
            return []

    async def _select_tool_with_llm(
        self,
        user_message: HumanMessage,
        tools: List,
        messages: List
    ) -> Optional[Dict[str, Any]]:
        """
        LLM을 사용하여 적합한 MCP 도구와 인자 선택
        
        Returns:
            {"tool": "tool_name", "arguments": {...}}
        """
        tool_names = [t.name for t in tools]
        
        # Agent별 시스템 프롬프트 가져오기
        system_prompt = self.get_system_prompt(tool_names, messages)
        
        llm_messages = [system_prompt, user_message]
        
        try:
            response = await self.llm.ainvoke(llm_messages)
            logger.info(f"[{self.name}] LLM raw response: {response.content}")
            
            # JSON 파싱
            cleaned_content = self._remove_think_tags(response.content)
            llm_output = json.loads(cleaned_content)
            
            tool_name = llm_output.get("tool")
            tool_args = llm_output.get("arguments", {})
            
            logger.info(f"[{self.name}] Selected tool: {tool_name}, args: {tool_args}")
            return {"tool": tool_name, "arguments": tool_args}
            
        except Exception as e:
            logger.error(f"[{self.name}] LLM tool selection failed: {e}")
            return None

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Any:
        """MCP 도구 실행"""
        try:
            result = await self.mcp.call_tool(tool_name, tool_args)
            logger.info(f"[{self.name}] Tool '{tool_name}' result: {result}")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Tool execution failed: {e}")
            raise

    def _remove_think_tags(self, text: str) -> str:
        """<think> 태그 제거"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # =============================
    # Agent별 구현 필요 (추상 메서드)
    # =============================
    @abstractmethod
    def get_system_prompt(
        self,
        tool_names: List[str],
        messages: List
    ) -> SystemMessage:
        """
        Agent별 시스템 프롬프트 정의
        
        Args:
            tool_names: 사용 가능한 MCP 도구 이름 리스트
            messages: 현재까지의 대화 메시지
            
        Returns:
            SystemMessage 객체
        """
        pass

    async def process_tool_result(
        self,
        state: Dict[str, Any],
        tool_result: Any,
        user_message: HumanMessage
    ) -> Dict[str, Any]:
        """
        도구 실행 결과 후처리 (선택적 오버라이드)
        
        기본 동작: 결과를 메시지에 추가하고 LLM으로 최종 응답 생성
        """
        messages = state.get("messages", [])
        
        # 도구 결과를 대화에 추가
        messages.append(ToolMessage(content=f"Tool Result: {tool_result}"))
        
        # LLM으로 최종 사용자 응답 생성
        system_prompt = self.get_system_prompt([], messages)
        response = await self.llm.ainvoke([system_prompt, *messages])
        messages.append(AIMessage(content=response.content))
        
        return {
            "messages": messages,
            "tool_result": tool_result,
            "last_result": response.content
        }

    # =============================
    # 기타 공통 메서드
    # =============================
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """입력 상태 검증 (Agent별 구현)"""
        pass

    def pre_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """실행 전 전처리 (선택적 오버라이드)"""
        return state

    def _validate_config(self):
        """설정 검증"""
        if not self.config.name:
            raise ValueError("Agent name is required")

    def _log_start(self, state):
        logger.info(f"[{self.name}] Starting with keys: {list(state.keys())}")

    def _log_end(self, result):
        logger.info(f"[{self.name}] Finished. Output keys: {list(result.keys())}")