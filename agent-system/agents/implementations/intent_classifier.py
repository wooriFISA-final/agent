import logging
import json
import re
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agents.base.agent_base import AgentBase, AgentConfig
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager

# 루트 로거를 사용하거나 agent_system 로거를 가져오기
logger = logging.getLogger("agent_system")  # 또는 logging.getLogger()로 루트 로거 사용


def remove_think_tags(text: str) -> str:
    """think 태그 제거"""
    # 여러 종류의 think 태그 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


@AgentRegistry.register("intent_classifier")
class IntentClassifierAgent(AgentBase):
    """
    LLM 기반 의도 분류 Agent
    
    입력:
        - query: str (필수)
    
    출력:
        - intent_result: str
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.llm = None
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """의도 분류 실행"""
        query = state.get("query", "")
        
        logger.info(f"🔍 [{self.name}] Classifying intent: '{query}'")
        
        # LLM 인스턴스 가져오기
        if self.llm is None:
            self.llm = LLMManager.get_llm(provider="ollama", model="qwen3:8b")
        
        # 의도 분류 프롬프트 생성
        system_prompt = """당신은 사용자의 입력을 분석하여 의도를 분류하는 AI입니다.
다음 중 하나로 분류하세요:
- create_plan : 초기 계획을 수립하기
- update_plan : 기존 계획을 수정하기
- investment_advice : 투자 조언 제공
- other : 기타

답변은 JSON 형식으로 반환하세요:
{
  "intent": "create_plan",
  "confidence": 0.9,
  "reason": "사용자가 새로운 계획을 만들고 싶어합니다"
}"""
        
        try:
            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"사용자 입력: {query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            intent_result = remove_think_tags(response.content)
            
            # 로그 출력
            logger.info(f"✅ [{self.name}] Intent classification completed.")
            logger.info(f"   Full result:\n{intent_result}")
            
            # 상태 업데이트
            state["intent_result"] = intent_result
            
            return state
            
        except Exception as e:
            logger.error(f"❌ [{self.name}] Intent classification failed: {e}")
            state["intent_result"] = f"의도 분류 중 오류가 발생했습니다: {str(e)}"
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

