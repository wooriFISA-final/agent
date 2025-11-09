"""
Research Agent - 수정 버전
문제 해결: validate_input에서 dict 타입도 처리
"""
from agents.base.agent_base import AgentBase, AgentConfig
from agents.registry.agent_registry import AgentRegistry
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@AgentRegistry.register("research")
class ResearchAgent(AgentBase):
    """
    리서치 Agent
    
    입력:
        - query: str (필수)
    
    출력:
        - research_result: str
    """
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """리서치 실행"""
        query = state.get("query", "")
        
        logger.info(f"🔍 [{self.name}] Researching: '{query}'")
        
        # 실제 리서치 로직 (여기서는 시뮬레이션)
        result = f"'{query}'에 대한 리서치 완료: AI agents는 자율적으로 작업을 수행하는 시스템입니다."
        
        state["research_result"] = result
        
        logger.info(f"✅ [{self.name}] Research completed")
        return state
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        입력 검증
        
        수정사항: 
        - state가 dict인지 확인
        - "query" 키 존재 확인
        - query가 빈 문자열이 아닌지 확인
        """
        # 디버깅 로그
        logger.debug(f"[{self.name}] Validating input...")
        logger.debug(f"[{self.name}] State type: {type(state)}")
        logger.debug(f"[{self.name}] State keys: {list(state.keys()) if hasattr(state, 'keys') else 'N/A'}")
        
        # dict 타입 확인
        if not isinstance(state, dict):
            logger.error(f"[{self.name}] ❌ State is not a dict: {type(state)}")
            return False
        
        # "query" 키 존재 확인
        if "query" not in state:
            logger.error(f"[{self.name}] ❌ 'query' key not found in state")
            logger.error(f"[{self.name}] Available keys: {list(state.keys())}")
            return False
        
        # query 값 확인
        query = state.get("query")
        if not query or not isinstance(query, str):
            logger.error(f"[{self.name}] ❌ 'query' must be a non-empty string")
            return False
        
        logger.debug(f"[{self.name}] ✅ Input validation passed")
        return True