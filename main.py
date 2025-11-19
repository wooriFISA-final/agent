import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from typing import Optional

from agents.registry.agent_registry import AgentRegistry
from agents.config.base_config import AgentState, StateBuilder, ExecutionStatus
from core.logging.logger import setup_logger
from graph.factory import mk_graph
from core.mcp.mcp_manager import MCPManager
from utils.session_manager import SessionManager  # SessionManager 임포트

logger = setup_logger()

# =============================
# 전역 변수
# =============================
graph = None
checkpointer = None  # 전역 Checkpointer
session_manager: Optional[SessionManager] = None  # SessionManager 추가


# =============================
# Lifespan 이벤트 핸들러
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 생명주기 관리
    
    Startup:
    1. 전역 Checkpointer 초기화
    2. SessionManager 초기화
    3. MCP 연결
    4. Agent 자동 등록
    5. LangGraph 빌드
    
    Shutdown:
    1. MCP 연결 종료
    """
    global graph, checkpointer, session_manager

    logger.info("🚀 Starting Multi-Agent System...")

    # 0) 전역 Checkpointer 초기화
    checkpointer = MemorySaver()
    logger.info("✅ Global MemorySaver initialized")

    # 0-1) SessionManager 초기화
    session_manager = SessionManager(checkpointer)
    logger.info("✅ SessionManager initialized")

    # 1) MCP 단일 세션 초기화
    mcp = MCPManager()
    mcp.initialize("http://localhost:8888/mcp/")

    # MCP 연결 재시도
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

    # 2) Agent 자동 등록
    logger.info("📦 Discovering agents...")
    AgentRegistry.auto_discover("agents.implementations")
    
    # 3) 그래프 생성 (YAML 기반) - 전역 Checkpointer 전달
    logger.info("🔧 Building agent graph from YAML...")
    graph = mk_graph("graph/schemas/graph.yaml", checkpointer=checkpointer)
    if not graph:
        logger.error("❌ Agent graph could not be built. Shutting down.")
        return

    logger.info("✅ Agent graph built successfully!")

    yield

    # 종료 시 정리
    logger.info("🧹 Shutting down Multi-Agent System...")
    await mcp.close()
    logger.info("✅ MCP connection closed.")


# =============================
# FastAPI 앱 생성
# =============================
app = FastAPI(
    title="Multi-Agent Planner",
    version="2.1.0",
    description="Multi-Agent system with unified state management and advanced session management",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================
# API 모델
# =============================
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str
    status: str = "success"
    metadata: dict = {}


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    mcp_connected: bool
    available_tools: int
    registered_agents: list
    error: str = None


# =============================
# API 엔드포인트
# =============================
@app.get("/")
async def root():
    """
    루트 엔드포인트 (헬스체크)
    
    Returns:
        시스템 상태 정보
    """
    return {
        "status": "ok",
        "message": "AI Agent API is running 🚀",
        "version": "2.1.0",
        "agents": AgentRegistry.list_agents(),
        "features": [
            "Unified AgentState management",
            "Multi-turn agent execution",
            "MCP tool integration",
            "LangGraph workflow",
            "Advanced session management with SessionManager"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    상세 헬스체크
    
    - MCP 연결 상태
    - 사용 가능한 Tool 개수
    - 등록된 Agent 목록
    """
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
    채팅 엔드포인트 (대화 히스토리 유지)
    
    Flow:
    1. 사용자 메시지 수신
    2. 기존 상태 로드 OR 새 상태 생성 (Checkpointer가 자동 처리)
    3. 새 메시지만 추가
    4. LangGraph 실행 (이전 대화 이어짐)
    5. 최종 응답 추출
    
    ⚠️ 중요: 
    - 전역 Checkpointer가 모든 세션의 상태를 관리
    - 같은 session_id면 이전 대화가 유지됨
    - 매번 초기화하지 않고 새 메시지만 추가
    
    Args:
        request: ChatRequest (message, session_id)
        
    Returns:
        ChatResponse (response, status, metadata)
    """
    global graph

    # 그래프 초기화 확인
    if not graph:
        logger.error("❌ Agent graph not initialized")
        return ChatResponse(
            response="시스템이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
            status="error",
            metadata={"error": "graph_not_initialized"}
        )

    try:
        logger.info(f"📩 Received message: {request.message}")
        logger.info(f"🔑 Session ID: {request.session_id}")

        # 1. LangGraph 설정 (thread_id로 세션 식별)
        graph_config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }

        # 2. 기존 상태 확인 및 새 메시지 추가
        try:
            # 기존 상태가 있는지 확인
            existing_state = await graph.aget_state(graph_config)
            
            if existing_state and existing_state.values and existing_state.values.get('messages'):
                # 기존 대화 이어가기
                existing_messages = existing_state.values.get('messages', [])
                logger.info(f"📚 Continuing existing conversation")
                logger.info(f"   Previous messages: {len(existing_messages)}")
                
                # ⚠️ 핵심: 새 메시지만 추가 (LangGraph가 자동으로 병합)
                input_state = {
                    "messages": [HumanMessage(content=request.message)]
                }
            else:
                # 새 대화 시작
                logger.info(f"🆕 Starting new conversation")
                
                # 초기 상태 생성 (모든 필드 포함)
                input_state = StateBuilder.create_initial_state(
                    messages=[HumanMessage(content=request.message)],
                    session_id=request.session_id,
                    max_iterations=10000000
                )
        except Exception as e:
            # 상태 조회 실패 시 새 대화로 시작
            logger.warning(f"⚠️  Could not load existing state: {e}")
            logger.info(f"🆕 Starting new conversation")
            
            input_state = StateBuilder.create_initial_state(
                messages=[HumanMessage(content=request.message)],
                session_id=request.session_id,
                max_iterations=100000000
            )

        # 3. 그래프 실행 (Checkpointer가 자동으로 상태 병합)
        logger.info("🚀 Executing agent graph...")
        result_state = await graph.ainvoke(input_state, config=graph_config)

        # 4. 실행 결과 로깅
        logger.info(f"✅ Graph execution completed")
        logger.info(f"   Status: {result_state.get('status')}")
        logger.info(f"   Iterations: {result_state.get('iteration')}")
        logger.info(f"   Execution path: {result_state.get('execution_path')}")
        logger.info(f"   Tool calls: {len(result_state.get('tool_calls', []))}")

        # 5. 응답 메시지 추출
        messages = result_state.get("messages", [])
        
        if not messages:
            logger.warning("⚠️  No messages in result state")
            return ChatResponse(
                response="응답을 생성할 수 없습니다.",
                status="warning",
                metadata={
                    "execution_status": str(result_state.get('status')),
                    "iterations": result_state.get('iteration', 0)
                }
            )

        # 6. AI 메시지만 필터링
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        
        if not ai_messages:
            logger.warning("⚠️  No AI messages found")
            return ChatResponse(
                response="AI 응답이 생성되지 않았습니다.",
                status="warning",
                metadata={
                    "total_messages": len(messages),
                    "execution_status": str(result_state.get('status'))
                }
            )

        # 7. 최종 응답 (마지막 AI 메시지)
        final_response = ai_messages[-1].content
        
        logger.info(f"💬 Final response length: {len(final_response)} chars")

        # 8. 메타데이터 포함하여 반환
        return ChatResponse(
            response=final_response,
            status="success",
            metadata={
                "session_id": request.session_id,
                "execution_status": str(result_state.get('status')),
                "iterations": result_state.get('iteration', 0),
                "tool_calls": len(result_state.get('tool_calls', [])),
                "execution_path": result_state.get('execution_path', []),
                "warnings": result_state.get('warnings', []),
                "conversation_length": len(messages)
            }
        )

    except asyncio.TimeoutError:
        logger.error("❌ Request timeout")
        return ChatResponse(
            response="요청 처리 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            status="error",
            metadata={"error": "timeout"}
        )
    
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}", exc_info=True)
        
        # MCP 연결 오류 감지
        if "mcp" in str(e).lower() or "connection" in str(e).lower():
            return ChatResponse(
                response="MCP 서버와의 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.",
                status="error",
                metadata={"error": "mcp_connection_error", "detail": str(e)}
            )
        
        return ChatResponse(
            response=f"처리 중 오류가 발생했습니다: {str(e)}",
            status="error",
            metadata={"error": "processing_error", "detail": str(e)}
        )


# =============================
# 세션 관리 API (SessionManager 사용)
# =============================

@app.get("/chat/sessions")
async def list_sessions():
    """
    활성 세션 목록 조회 (간단 버전)
    
    Returns:
        세션 ID 목록
    """
    global session_manager
    
    if not session_manager:
        return {
            "status": "error",
            "message": "SessionManager not initialized"
        }
    
    try:
        sessions = session_manager.list_all_sessions()
        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/chat/sessions/detailed")
async def list_sessions_detailed():
    """
    활성 세션 목록 조회 (상세 정보 포함)
    
    Returns:
        각 세션의 체크포인트 수, 메시지 수, 타임스탬프 등
    """
    global session_manager
    
    if not session_manager:
        return {
            "status": "error",
            "message": "SessionManager not initialized"
        }
    
    try:
        sessions = session_manager.list_sessions_with_details()
        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Failed to list detailed sessions: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/chat/session/{session_id}")
async def get_session_info(session_id: str):
    """
    특정 세션의 상세 정보 조회
    
    Args:
        session_id: 조회할 세션 ID
        
    Returns:
        세션 상세 정보
    """
    global session_manager
    
    if not session_manager:
        return {
            "status": "error",
            "message": "SessionManager not initialized"
        }
    
    try:
        info = session_manager.get_session_details(session_id)
        
        if not info:
            return {
                "status": "not_found",
                "message": f"Session {session_id} not found"
            }
        
        return {
            "status": "success",
            "session": info
        }
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """
    특정 세션 삭제
    
    Args:
        session_id: 삭제할 세션 ID
        
    Returns:
        삭제 결과
    """
    global session_manager
    
    if not session_manager:
        return {
            "status": "error",
            "message": "SessionManager not initialized"
        }
    
    try:
        result = session_manager.delete_session(session_id)
        
        if result["deleted"]:
            logger.info(f"🗑️  Session {session_id} deleted ({result['checkpoints_deleted']} checkpoints)")
            return {
                "status": "success",
                "message": f"Session {session_id} deleted",
                "checkpoints_deleted": result["checkpoints_deleted"]
            }
        else:
            return {
                "status": "not_found",
                "message": f"Session {session_id} not found"
            }
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/chat/statistics")
async def get_statistics():
    """
    전체 세션 통계 조회
    
    Returns:
        세션 수, 체크포인트 수, 메시지 수 등 통계
    """
    global session_manager
    
    if not session_manager:
        return {
            "status": "error",
            "message": "SessionManager not initialized"
        }
    
    try:
        stats = session_manager.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/chat/sessions/cleanup")
async def cleanup_empty_sessions():
    """
    빈 세션 정리 (체크포인트가 없는 세션)
    
    Returns:
        정리된 세션 목록
    """
    global session_manager
    
    if not session_manager:
        return {
            "status": "error",
            "message": "SessionManager not initialized"
        }
    
    try:
        result = session_manager.cleanup_empty_sessions()
        logger.info(f"🧹 Cleaned up {result['count']} empty sessions")
        return {
            "status": "success",
            "message": f"Cleaned up {result['count']} empty sessions",
            "deleted_sessions": result["deleted_sessions"]
        }
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# =============================
# 기타 API
# =============================

@app.get("/agents")
async def list_agents():
    """
    등록된 모든 Agent 목록 조회
    
    Returns:
        Agent 이름 리스트
    """
    agents = AgentRegistry.list_agents()
    return {
        "agents": agents,
        "count": len(agents)
    }


@app.get("/graph/structure")
async def get_graph_structure():
    """
    현재 그래프 구조 정보 조회
    
    Returns:
        그래프 노드, 엣지 정보
    """
    global graph
    
    if not graph:
        return {"error": "Graph not initialized"}
    
    return {
        "status": "initialized",
        "message": "Graph structure available via /health endpoint"
    }


# =============================
# 개발 서버 실행
# =============================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting API Server on http://localhost:8080")
    logger.info("📚 API Documentation: http://localhost:8080/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )