from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from typing import Optional

from agent.registry.agent_registry import AgentRegistry
from agent.config.base_config import AgentState, StateBuilder, ExecutionStatus
from core.logging.logger import setup_logger
from graph.factory import mk_graph
from core.mcp.mcp_manager import MCPManager
from utils.session_manager import SessionManager

logger = setup_logger()

# =============================
# 전역 변수
# =============================
graph = None
checkpointer = None
session_manager: Optional[SessionManager] = None


# =============================
# Lifespan 이벤트 핸들러
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱 생명주기 관리"""
    global graph, checkpointer, session_manager

    logger.info("🚀 Starting Multi-Agent System...")

    checkpointer = MemorySaver()
    logger.info("✅ Global MemorySaver initialized")

    session_manager = SessionManager(checkpointer)
    logger.info("✅ SessionManager initialized")

    mcp = MCPManager()
    mcp.initialize("http://localhost:8888/mcp/")

    for attempt in range(1, 6):
        try:
            await mcp.connect()
            logger.info("✅ MCP connected successfully!")
            break
        except Exception as e:
            logger.warning(f"⚠️  MCP connection attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                await asyncio.sleep(2)
            else:
                logger.error("❌ Failed to connect to MCP after 5 attempts")
                raise

    logger.info("📦 Discovering agents...")
    AgentRegistry.auto_discover()
    
    logger.info("🔧 Building agent graph from YAML...")
    graph = mk_graph("graph/schemas/graph.yaml", checkpointer=checkpointer)
    if not graph:
        logger.error("❌ Agent graph could not be built. Shutting down.")
        return

    logger.info("✅ Agent graph built successfully!")

    yield

    logger.info("🧹 Shutting down Multi-Agent System...")
    await mcp.close()
    logger.info("✅ MCP connection closed.")


app = FastAPI(
    title="Multi-Agent Planner",
    version="2.1.0",
    description="Multi-Agent system with conversation history",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    mcp_connected: bool
    available_tools: int
    registered_agents: list
    error: str = None


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AI Agent API is running 🚀",
        "version": "2.1.0",
        "agents": AgentRegistry.list_agents(),
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        mcp = MCPManager()
        await mcp.ensure_connected()
        tools = await mcp.list_tools()
        
        return HealthResponse(
            status="healthy",
            mcp_connected=True,
            available_tools=len(tools),
            registered_agents=AgentRegistry.list_agents()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            mcp_connected=False,
            available_tools=0,
            registered_agents=AgentRegistry.list_agents(),
            error=str(e)
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    채팅 엔드포인트 (멀티턴 대화 지원)
    
    대화 기록 관리:
    1. 같은 session_id면 이전 대화 자동 로드
    2. LangGraph Checkpointer가 메시지 히스토리 관리
    3. Agent는 전체 대화 컨텍스트를 받아서 처리
    """
    global graph

    if not graph:
        logger.error("❌ Agent graph not initialized")
        return ChatResponse(
            response="시스템이 초기화되지 않았습니다.",
            status="error",
            metadata={"error": "graph_not_initialized"}
        )

    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"📩 NEW REQUEST")
        logger.info(f"   Message: {request.message}")
        logger.info(f"   Session ID: {request.session_id}")
        logger.info(f"{'='*80}")

        graph_config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }

        # ============================================
        # 🔍 중요: 기존 대화 기록 확인
        # ============================================
        try:
            existing_state = await graph.aget_state(graph_config)
            
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get('messages', [])
                
                if existing_messages:
                    logger.info(f"📚 CONTINUING CONVERSATION")
                    logger.info(f"   Previous messages: {len(existing_messages)}")
                    
                    # 🔍 디버깅: 이전 메시지 내용 출력
                    logger.info(f"   Previous conversation:")
                    for i, msg in enumerate(existing_messages[-5:], 1):  # 마지막 5개만
                        msg_type = type(msg).__name__
                        content_preview = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
                        logger.info(f"      [{i}] {msg_type}: {content_preview}...")
                    
                    # ✅ 핵심: 새 메시지만 추가 (LangGraph가 자동으로 병합)
                    input_state = {
                        "messages": [HumanMessage(content=request.message)]
                    }
                    
                    logger.info(f"   ✅ New message will be appended to existing {len(existing_messages)} messages")
                else:
                    logger.info(f"🆕 STARTING NEW CONVERSATION (empty history)")
                    input_state = StateBuilder.create_initial_state(
                        messages=[HumanMessage(content=request.message)],
                        session_id=request.session_id,
                        max_iterations=10
                    )
            else:
                logger.info(f"🆕 STARTING NEW CONVERSATION (no state)")
                input_state = StateBuilder.create_initial_state(
                    messages=[HumanMessage(content=request.message)],
                    session_id=request.session_id,
                    max_iterations=10
                )
                
        except Exception as e:
            logger.warning(f"⚠️  Could not load existing state: {e}")
            logger.info(f"🆕 STARTING NEW CONVERSATION (error fallback)")
            
            input_state = StateBuilder.create_initial_state(
                messages=[HumanMessage(content=request.message)],
                session_id=request.session_id,
                max_iterations=10
            )

        # ============================================
        # 🚀 Agent 실행
        # ============================================
        logger.info("🚀 Executing agent graph...")
        result_state = await graph.ainvoke(input_state, config=graph_config)

        # ============================================
        # 📊 실행 결과 분석
        # ============================================
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ EXECUTION COMPLETED")
        logger.info(f"   Status: {result_state.get('status')}")
        logger.info(f"   Iterations: {result_state.get('iteration', 0)}")
        logger.info(f"   Tool calls: {len(result_state.get('tool_calls', []))}")

        # 전체 메시지 수 확인
        all_messages = result_state.get("messages", [])
        logger.info(f"   Total messages in state: {len(all_messages)}")
        
        # 🔍 디버깅: 전체 대화 기록 출력
        if all_messages:
            logger.info(f"\n   Full conversation history:")
            for i, msg in enumerate(all_messages, 1):
                msg_type = type(msg).__name__
                content_preview = msg.content[:80] if hasattr(msg, 'content') else str(msg)[:80]
                logger.info(f"      [{i}] {msg_type}: {content_preview}...")
        
        logger.info(f"{'='*80}\n")

        # ============================================
        # 💬 응답 추출
        # ============================================
        if not all_messages:
            logger.warning("⚠️  No messages in result state")
            return ChatResponse(
                response="응답을 생성할 수 없습니다.",
                status="warning",
                metadata={
                    "execution_status": str(result_state.get('status')),
                    "iterations": result_state.get('iteration', 0),
                    "session_id": request.session_id
                }
            )

        # AI 메시지만 필터링
        ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
        
        if not ai_messages:
            logger.warning("⚠️  No AI messages found")
            return ChatResponse(
                response="AI 응답이 생성되지 않았습니다.",
                status="warning",
                metadata={
                    "total_messages": len(all_messages),
                    "execution_status": str(result_state.get('status')),
                    "session_id": request.session_id
                }
            )

        # 마지막 AI 메시지가 최종 응답
        final_response = ai_messages[-1].content
        
        logger.info(f"💬 Returning response: {len(final_response)} chars")
        logger.info(f"   (AI message {len(ai_messages)} of {len(all_messages)} total)")

        return ChatResponse(
            response=final_response,
            status="success",
            metadata={
                "session_id": request.session_id,
                "execution_status": str(result_state.get('status')),
                "iterations": result_state.get('iteration', 0),
                "tool_calls": len(result_state.get('tool_calls', [])),
                "conversation_length": len(all_messages),
                "ai_messages_count": len(ai_messages),
                "execution_path": result_state.get('execution_path', []),
                "has_conversation_history": len(all_messages) > 2  # User + AI = 2, 더 많으면 기록 있음
            }
        )

    except asyncio.TimeoutError:
        logger.error("❌ Request timeout")
        return ChatResponse(
            response="요청 처리 시간이 초과되었습니다.",
            status="error",
            metadata={"error": "timeout", "session_id": request.session_id}
        )
    
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}", exc_info=True)
        
        return ChatResponse(
            response=f"처리 중 오류가 발생했습니다: {str(e)}",
            status="error",
            metadata={
                "error": "processing_error",
                "detail": str(e),
                "session_id": request.session_id
            }
        )


# =============================
# 세션 관리 API
# =============================

@app.get("/chat/sessions")
async def list_sessions():
    """활성 세션 목록 조회"""
    global session_manager
    
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    try:
        sessions = session_manager.list_all_sessions()
        return {"status": "success", "sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/chat/sessions/detailed")
async def list_sessions_detailed():
    """활성 세션 상세 정보 조회"""
    global session_manager
    
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    try:
        sessions = session_manager.list_sessions_with_details()
        return {"status": "success", "sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Failed to list detailed sessions: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/chat/session/{session_id}")
async def get_session_info(session_id: str):
    """특정 세션 정보 조회"""
    global session_manager
    
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    try:
        info = session_manager.get_session_details(session_id)
        
        if not info:
            return {"status": "not_found", "message": f"Session {session_id} not found"}
        
        return {"status": "success", "session": info}
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/chat/session/{session_id}/history")
async def get_conversation_history(session_id: str):
    """
    특정 세션의 대화 기록 조회
    
    Returns:
        대화 메시지 리스트
    """
    global graph
    
    if not graph:
        return {"status": "error", "message": "Graph not initialized"}
    
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)
        
        if not state or not state.values:
            return {
                "status": "not_found",
                "message": f"Session {session_id} not found",
                "messages": []
            }
        
        messages = state.values.get('messages', [])
        
        # 메시지를 JSON 직렬화 가능한 형태로 변환
        message_list = []
        for msg in messages:
            message_list.append({
                "type": type(msg).__name__,
                "role": getattr(msg, 'type', 'unknown'),
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            })
        
        return {
            "status": "success",
            "session_id": session_id,
            "message_count": len(messages),
            "messages": message_list
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        return {"status": "error", "message": str(e)}


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    global session_manager
    
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    try:
        result = session_manager.delete_session(session_id)
        
        if result["deleted"]:
            logger.info(f"🗑️  Session {session_id} deleted")
            return {
                "status": "success",
                "message": f"Session {session_id} deleted",
                "checkpoints_deleted": result["checkpoints_deleted"]
            }
        else:
            return {"status": "not_found", "message": f"Session {session_id} not found"}
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/chat/statistics")
async def get_statistics():
    """전체 세션 통계"""
    global session_manager
    
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    try:
        stats = session_manager.get_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/agents")
async def list_agents():
    """등록된 Agent 목록"""
    agents = AgentRegistry.list_agents()
    return {"agents": agents, "count": len(agents)}


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting API Server on http://localhost:8080")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )