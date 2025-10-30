import logging
import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- 챗봇 상태 정의 ---
class ChatbotState(MessagesState):
    """챗봇의 대화 상태를 관리"""
    conversation_count: int = 0

# --- LLM 설정 ---
def create_llm(temperature: float = 0.7) -> ChatOllama:
    """LLM 인스턴스 생성"""
    return ChatOllama(
        model="qwen3:8b",
        temperature=temperature,
        top_p=0.9
    )

# --- 챗봇 시스템 프롬프트 ---
CHATBOT_PROMPT = """
당신은 친절하고 도움이 되는 AI 어시스턴트입니다.

사용자의 질문에 명확하고 정확하게 답변하세요.
- 친근한 톤으로 대화하세요
- 모르는 것은 솔직히 모른다고 말하세요
- 필요하면 추가 질문을 하세요
"""

def remove_think_tags(text: str) -> str:
    """
    응답에서 <think> 태그와 그 내용을 제거합니다.
    
    Args:
        text: 원본 텍스트
    
    Returns:
        <think> 태그가 제거된 텍스트
    """
    # <think>...</think> 패턴을 찾아서 제거
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 앞뒤 공백 제거
    cleaned_text = cleaned_text.strip()
    return cleaned_text

class SimpleChatbot:
    
    def __init__(self):
        self.llm = create_llm()
    
    def create_chatbot_node(self):
        """챗봇 노드 생성"""
        
        async def chatbot_node(state: ChatbotState):
            logger.info("🤖 챗봇이 응답을 생성하고 있습니다...")
            
            try:
                # 시스템 프롬프트와 대화 기록 결합
                messages = [SystemMessage(content=CHATBOT_PROMPT)] + state.get("messages", [])
                
                # LLM 호출
                response = await self.llm.ainvoke(messages)
                
                # <think> 태그 제거
                cleaned_content = remove_think_tags(response.content)
                
                logger.info(f"✅ 챗봇 응답 완료: {cleaned_content[:50]}...")
                
                # 정제된 내용으로 새 AIMessage 생성
                cleaned_response = AIMessage(
                    content=cleaned_content,
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    id=response.id
                )
                
                # 상태 업데이트
                return {
                    "messages": [cleaned_response],
                    "conversation_count": state.get("conversation_count", 0) + 1
                }
                
            except Exception as e:
                logger.error(f"❌ 챗봇 오류: {e}")
                error_msg = AIMessage(content=f"죄송합니다. 오류가 발생했습니다: {str(e)}")
                return {
                    "messages": [error_msg],
                    "conversation_count": state.get("conversation_count", 0)
                }
        
        return chatbot_node
    
    def build_graph(self):
        """간단한 챗봇 그래프 구성"""
        workflow = StateGraph(ChatbotState)
        
        # 챗봇 노드 추가
        chatbot_node = self.create_chatbot_node()
        workflow.add_node("chatbot", chatbot_node)
        
        # 단순한 흐름: START -> chatbot -> END
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)
        
        return workflow.compile(checkpointer=MemorySaver())

def create_chatbot() -> StateGraph:
    """챗봇 그래프 생성"""
    builder = SimpleChatbot()
    return builder.build_graph()

async def chat(graph, user_input: str, config: Dict[str, Any] = None):
    """챗봇과 대화하기"""
    
    if config is None:
        config = {"configurable": {"thread_id": "simple-chat-1"}}
    
    logger.info(f"💬 사용자: {user_input}")
    
    try:
        # 그래프 실행
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "conversation_count": 0
            },
            config=config
        )
        
        # 응답 추출
        messages = result.get('messages', [])
        if messages:
            ai_response = messages[-1]
            response_text = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
            
            print(f"\n🤖 챗봇: {response_text}\n")
            
            return {
                "status": "success",
                "response": response_text,
                "conversation_count": result.get("conversation_count", 0)
            }
        else:
            return {"status": "error", "message": "응답을 생성할 수 없습니다."}
            
    except Exception as e:
        logger.error(f"❌ 대화 중 오류 발생: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"오류가 발생했습니다: {str(e)}"
        }

async def interactive_chat():
    """대화형 챗봇 실행"""
    print("=" * 60)
    print("🤖 간단한 챗봇입니다. '종료' 또는 'quit'를 입력하면 종료됩니다.")
    print("=" * 60)
    
    graph = create_chatbot()
    config = {"configurable": {"thread_id": "interactive-chat"}}
    
    while True:
        try:
            user_input = input("\n💬 당신: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['종료', 'quit', 'exit', '나가기']:
                print("\n👋 챗봇을 종료합니다. 좋은 하루 되세요!")
                break
            
            await chat(graph, user_input, config)
            
        except KeyboardInterrupt:
            print("\n\n👋 챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류: {e}")

# --- 사용 예시 ---
if __name__ == "__main__":
    import asyncio
    
    # 대화형 모드 실행
    asyncio.run(interactive_chat())
    
    # 또는 단일 질문 모드
    # async def single_question():
    #     graph = create_chatbot()
    #     await chat(graph, "안녕하세요! 오늘 날씨가 어때요?")
    # 
    # asyncio.run(single_question())