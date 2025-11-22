"""
채팅 엔드포인트

사용자와 AI 간의 대화를 처리하는 엔드포인트를 정의합니다.
"""
from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage, AIMessage
import asyncio

from core.logging.logger import setup_logger
from core.config.setting import settings
from agents.config.base_config import StateBuilder
from api.models import ChatRequest, ChatResponse

logger = setup_logger()

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """채팅 엔드포인트
    
    사용자 메시지를 받아 AI 응답을 생성합니다.
    대화 히스토리를 관리하고 Agent 그래프를 실행합니다.
    
    Args:
        request: FastAPI Request 객체
        chat_request: 채팅 요청 데이터
        
    Returns:
        ChatResponse: AI 응답 데이터
    """
    graph = request.app.state.graph
    if not graph:
        logger.error("❌ Agent graph not initialized")
        return ChatResponse(
            response="System is not initialized.",
            status="error",
            metadata={"error": "graph_not_initialized"}
        )

    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"📩 NEW REQUEST | Session: {chat_request.session_id}")
        logger.info(f"   Message: {chat_request.message}")
        logger.info(f"{'='*80}")

        graph_config = {"configurable": {"thread_id": chat_request.session_id}}

        # Check for existing conversation state
        try:
            existing_state = await graph.aget_state(graph_config)
            has_history = existing_state and existing_state.values.get('global_messages')
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing state for session '{chat_request.session_id}': {e}")
            has_history = False

        if has_history:
            logger.info(f"📚 Continuing conversation for session '{chat_request.session_id}'")
            input_state = {"global_messages": [HumanMessage(content=chat_request.message)]}
        else:
            logger.info(f"🆕 Starting new conversation for session '{chat_request.session_id}'")
            input_state = StateBuilder.create_initial_state(
                messages=[HumanMessage(content=chat_request.message)],
                session_id=chat_request.session_id,
                max_iterations=settings.MAX_GRAPH_ITERATIONS
            )

        # Execute the agent graph
        logger.info("🚀 Executing agent graph...")
        result_state = await graph.ainvoke(input_state, config=graph_config)
        logger.info("✅ Graph execution completed.")

        # Extract the final response from global_messages
        all_messages = result_state.get("global_messages", [])
        ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]

        if not ai_messages:
            logger.warning("⚠️ No AI messages found in the final state.")
            # 폴백: last_result 확인
            last_result = result_state.get("last_result")
            if last_result:
                logger.info("📌 Using last_result as fallback response")
                return ChatResponse(
                    response=last_result,
                    status="success",
                    metadata={"session_id": chat_request.session_id, "source": "last_result"}
                )
            return ChatResponse(response="AI did not generate a response.", status="warning")

        final_response = ai_messages[-1].content
        logger.info(f"💬 Returning response for session '{chat_request.session_id}'.")

        return ChatResponse(
            response=final_response,
            status="success",
            metadata={"session_id": chat_request.session_id}
        )

    except asyncio.TimeoutError:
        logger.error(f"❌ Request timeout for session '{chat_request.session_id}'")
        return ChatResponse(
            response="Request timed out.",
            status="error",
            metadata={"error": "timeout", "session_id": chat_request.session_id}
        )
    
    except Exception as e:
        logger.error(f"❌ Chat processing failed for session '{chat_request.session_id}': {e}", exc_info=True)
        return ChatResponse(
            response=f"An internal error occurred: {str(e)}",
            status="error",
            metadata={"error": "processing_error", "detail": str(e), "session_id": chat_request.session_id}
        )
