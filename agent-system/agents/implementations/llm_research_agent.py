"""
LLM Research Agent
Ollama를 사용한 리서치 Agent
"""
from agents.base.agent_base import AgentBase, AgentConfig
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager, LLMHelper
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@AgentRegistry.register("llm_research")
class LLMResearchAgent(AgentBase):
    """
    LLM 기반 리서치 Agent
    
    입력:
        - query: str (필수)
    
    출력:
        - research_result: str
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.llm = None
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """리서치 실행"""
        query = state.get("query", "")
        
        logger.info(f"🔍 [{self.name}] Researching with LLM: '{query}'")
        
        # LLM 인스턴스 가져오기
        if self.llm is None:
            self.llm = LLMManager.get_llm(provider="ollama", model="qwen3:8b")
        
        # 리서치 프롬프트 생성
        system_prompt = """당신은 전문 리서치 어시스턴트입니다. 
사용자의 질문에 대해 상세하고 정확한 정보를 제공해야 합니다.
답변은 구조화되고 명확해야 하며, 가능한 한 구체적인 정보를 포함해야 합니다."""
        
        research_prompt = f"""다음 주제에 대해 상세한 리서치를 수행해주세요:

주제: {query}

다음 항목들을 포함하여 답변해주세요:
1. 주제에 대한 개요
2. 주요 내용 및 특징
3. 관련 정보 및 세부사항
4. 결론 및 요약

답변은 한국어로 작성해주세요."""
        
        try:
            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=research_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            research_result = response.content
            p
            logger.info(f"✅ [{self.name}] Research completed. Result length: {len(research_result)}")
            
            # 상태 업데이트
            state["research_result"] = research_result
            
            return state
            
        except Exception as e:
            logger.error(f"❌ [{self.name}] Research failed: {e}")
            state["research_result"] = f"리서치 중 오류가 발생했습니다: {str(e)}"
            return state
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        입력 검증
        
        - state가 dict인지 확인
        - "query" 키 존재 확인
        - query가 빈 문자열이 아닌지 확인
        """
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
