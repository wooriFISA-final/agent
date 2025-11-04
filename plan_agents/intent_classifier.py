import logging, json, re
from typing import Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class IntentResult(BaseModel):
    """사용자의 의도를 분류한 결과"""
    intent: str = Field(description="create_plan, update_plan, investment_advice, other 중 하나")
    confidence: float = Field(description="0~1 사이의 신뢰도")
    reason: str = Field(description="이 의도를 선택한 이유")

def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

class IntentClassifierAgent:
    """사용자 입력으로부터 의도를 분류하는 Agent"""

    def __init__(self):
        self.llm = ChatOllama(model="qwen3:8b", temperature=0.1, top_p=0.1)

    def create_intent_node(self):
        async def intent_node(state: MessagesState):
            logger.info("🔍 IntentClassifier: 입력 분석 중...")
            try:
                messages = state.get("messages", [])
                user_input = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
                if not user_input:
                    raise ValueError("No user input found")

                system_prompt = SystemMessage(content="""
당신은 사용자의 입력을 분석하여 의도를 분류하는 AI입니다.
다음 중 하나로 분류하세요:
- create_plan : 초기 계획을 수립하기
- update_plan : 기존 계획을 수정하기
- investment_advice. : 투자 조언 제공
- other : 기타

JSON 형식으로만 답변하세요:
{
  "intent": "...",
  "confidence": 0.0~1.0,
  "reason": "..."
}
""")

                query = HumanMessage(content=user_input)
                response = await self.llm.ainvoke([system_prompt, query])
                cleaned = remove_think_tags(response.content)

                logger.info(f"✅ Intent 분류 완료: {cleaned[:80]}")
                parsed = json.loads(cleaned)
                intent_result = IntentResult(**parsed)

                return {
                    "messages": [AIMessage(content=f"[Intent: {intent_result.intent}] {intent_result.reason}")],
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "reason": intent_result.reason
                }

            except Exception as e:
                logger.error(f"❌ IntentClassifier 오류: {e}", exc_info=True)
                return {
                    "messages": [AIMessage(content=f"의도 분석 실패: {e}")],
                    "intent": "error",
                    "confidence": 0.0,
                    "reason": str(e)
                }
        return intent_node
