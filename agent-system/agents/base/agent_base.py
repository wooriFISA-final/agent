from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import logging

class AgentConfig(BaseModel):
    """Agent 설정"""
    name: str
    description: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    enabled: bool = True
    dependencies: list[str] = []

logger = logging.getLogger(__name__)

class AgentBase(ABC):
    """모든 Agent의 베이스 클래스"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self._validate_config()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """전체 실행 파이프라인"""
        self._log_start(state)

        if not self.validate_input(state):
            raise ValueError(f"Invalid input for {self.name}")

        state = self.pre_execute(state)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with asyncio.timeout(self.config.timeout):
                    result = await self.execute(state)
                break
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout} seconds"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                if attempt == self.config.max_retries:
                    raise TimeoutError(f"{self.name} execution timed out after {self.config.timeout} seconds")
                await asyncio.sleep(1.5 * attempt)
            except Exception as e:
                error_msg = str(e) if str(e) else f"{type(e).__name__}"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                if attempt == self.config.max_retries:
                    raise e
                await asyncio.sleep(1.5 * attempt)

        result = self.post_execute(result)
        self._log_end(result)
        return result
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """실행 로직 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """입력 검증"""
        pass
    
    def pre_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """실행 전 전처리 (선택적 오버라이드)"""
        return state
    
    def post_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """실행 후 후처리 (선택적 오버라이드)"""
        return state
    
    def _validate_config(self):
        """설정 검증"""
        if not self.config.name:
            raise ValueError("Agent name is required")

    def _log_start(self, state):
        logger.info(f"[{self.name}] Starting with input keys: {list(state.keys())}")

    def _log_end(self, result):
        logger.info(f"[{self.name}] Finished. Output keys: {list(result.keys())}")
