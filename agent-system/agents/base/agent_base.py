from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel
from agents.config.base_config import BaseAgentConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agents.base.messages import ThinkMessage, ResultMessage
import logging
import re
# -------------------------------
# 로그 설정
# -------------------------------
logger = logging.getLogger(__name__)

# -------------------------------
# Agent 추상 베이스 클래스
# -------------------------------
class AgentBase(ABC):
    """
    모든 Agent가 상속받는 공통 베이스 클래스

    특징:
    - 입력 검증(validate_input)
    - pre/post 처리(pre_execute/post_execute)
    - 핵심 실행(execute) 재시도 및 타임아웃 처리
    - 상태(state) dict 기반 관리
    - 로깅 자동화
    """

    def __init__(self, config: BaseAgentConfig):
        self.config = config
        self.name = config.name
        self._validate_config()  # 생성 시 기본 설정 검증

    # -------------------------------
    # 전체 실행 파이프라인
    # -------------------------------
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 실행 흐름:
        1) 입력 검증(validate_input)
        2) 실행 전 전처리(pre_execute)
        3) 핵심 실행(execute) - 재시도 & 타임아웃 포함
        4) 실행 후 후처리(post_execute)
        5) 시작/종료 로깅
        """
        self._log_start(state)

        # 입력 데이터 검증
        if not self.validate_input(state):
            raise ValueError(f"Invalid input for {self.name}")

        # 필요 시 실행 전 상태 전처리
        state = self.pre_execute(state)

        
        ## ** 제거 가능성 있는 코드** ## 
        # 핵심 실행 로직 실행 (재시도 + 타임아웃)
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with asyncio.timeout(self.config.timeout):
                    result = await self.execute(state)
                break  # 성공 시 재시도 루프 탈출
            except asyncio.TimeoutError:
                # 타임아웃 발생 시 로깅 후 재시도
                error_msg = f"Timeout after {self.config.timeout} seconds"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                if attempt == self.config.max_retries:
                    raise TimeoutError(f"{self.name} execution timed out after {self.config.timeout} seconds")
                await asyncio.sleep(1.5 * attempt)  # 지수적 백오프
            except Exception as e:
                # 일반 예외 처리 및 재시도
                error_msg = str(e) if str(e) else f"{type(e).__name__}"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                if attempt == self.config.max_retries:
                    raise e
                await asyncio.sleep(1.5 * attempt)
        
        # 필요 시 실행 후 상태 후처리
        # result = self.post_execute(result)
        self._log_end(result)
        return result

    # -------------------------------
    # 핵심 실행 로직 (Agent별로 구현)
    # -------------------------------
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        기본 실행 구조 (사용자 메시지 기반)
        - HumanMessage 읽기
        - LLM 호출
        - post_execute에서 Think/Result 분류
        """
        messages = state.get("messages", [])
        user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not user_message:
            raise ValueError("No HumanMessage found in state")

        system_prompt = SystemMessage(content="당신은 에이전트입니다.")
        llm_messages = [system_prompt, user_message]

        # LLM 호출
        response = await self.llm.ainvoke(llm_messages)

        # LLM raw content만 상태에 저장, 메시지 분류는 post_execute에서 처리
        state["last_llm_response"] = response.content
        return {"messages": messages, 
                "raw_response": response.content}


    # -------------------------------
    # 입력 검증 (Agent별로 구현)
    # -------------------------------
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """state dict가 유효한지 검사"""
        pass

    # -------------------------------
    # 선택적 전처리
    # -------------------------------
    def pre_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """실행 전 상태 전처리 (필요 시 오버라이드 가능)"""
        return state

    # -------------------------------
    # 선택적 후처리
    # -------------------------------
    def post_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 응답 후처리
        - <think> 태그 분리
        - ThinkMessage / ResultMessage 생성
        """
        raw_content = state.get("last_llm_response", "")
        think_messages = [ThinkMessage(content=m) for m in re.findall(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)]
        cleaned_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        result_message = ResultMessage(content=cleaned_content)

        state["messages"].extend(think_messages + [result_message])
        state["last_result"] = cleaned_content
        return state

    # -------------------------------
    # 설정 검증
    # -------------------------------
    def _validate_config(self):
        """AgentConfig 기본 필수 값 검증"""
        if not self.config.name:
            raise ValueError("Agent name is required")

    # -------------------------------
    # 실행 로깅
    # -------------------------------
    def _log_start(self, state):
        logger.info(f"[{self.name}] Starting with input keys: {list(state.keys())}")

    def _log_end(self, result):
        logger.info(f"[{self.name}] Finished. Output keys: {list(result.keys())}")
