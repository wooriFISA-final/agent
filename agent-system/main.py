from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager
import asyncio

from agents.registry.agent_registry import AgentRegistry
from core.logging.logger import setup_logger
from graph.builder.graph_builder import GraphBuilder
from graph.schemas.state import LLMStateSchema
from core.mcp.mcp_manager import MCPManager

logger = setup_logger()

# =============================
# Lifespan 이벤트 핸들러
# =============================
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    logger.info("🚀 Starting Multi-Agent System...")

    # 1) MCP 단일 세션 초기화
    mcp = MCPManager()
    mcp.initialize("http://localhost:8888/mcp/")

    for attempt in range(5):
        try:
            await mcp.connect()
            logger.info("🔗 MCP connected!")
            break
        except Exception:
            await asyncio.sleep(2)

    # 2) Agent 자동 등록
    AgentRegistry.auto_discover("agents.implementations")

    # 3) 그래프 생성
    builder = GraphBuilder(LLMStateSchema)
    builder.add_agent_node("user_reg", "user_registration")\
        .set_entry_point("user_reg")\
        .set_finish_point("user_reg")
    graph = builder.build()

    yield

    # 종료
    await mcp.close()
    logger.info("🧹 MCP closed.")


app = FastAPI(title="Multi-Agent Planner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================
# FastAPI 앱 생성
# =============================
app = FastAPI(title="Multi-Agent Planner", lifespan=lifespan)

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
    message: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    response: str


# =============================
# API 엔드포인트
# =============================
@app.get("/")
async def root():
    """헬스체크"""
    return {
        "status": "ok",
        "message": "AI Agent API is running 🚀",
        "agents": AgentRegistry.list_agents()
    }


@app.get("/health")
async def health_check():
    """MCP 연결 상태 확인"""
    try:
        mcp = MCPManager()
        await mcp.ensure_connected()
        tools = await mcp.list_tools()
        return {
            "status": "healthy",
            "mcp_connected": True,
            "available_tools": len(tools)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "mcp_connected": False,
            "error": str(e)
        }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    채팅 엔드포인트
    
    Front → Agent Graph → Response
    """
    global graph

    if not graph:
        logger.error("❌ Agent graph not initialized")
        return ChatResponse(response="❌ Agent graph is not initialized properly.")

    try:
        logger.info(f"📩 User message: {request.message}")
        logger.info(f"🔑 Session ID: {request.session_id}")

        # 그래프 설정
        config = {"configurable": {"thread_id": request.session_id}}

        # 메시지 생성 및 그래프 실행
        messages = [HumanMessage(content=request.message)]
        result = await graph.ainvoke({"messages": messages}, config=config)

        # 응답 추출
        final_response = result.get("messages")

        logger.info(f"최종 응답 결과 포맷: {final_response}")
        if not final_response:
            logger.warning("⚠️ No response generated")
            return ChatResponse(response="응답을 생성할 수 없습니다.")

        logger.info(f"✅ Response generated: {final_response[:100] if isinstance(final_response, str) else 'List'}...")
        
        # AI 메시지 추출
        ai_messages = [m for m in final_response if isinstance(m, AIMessage)]
        if not ai_messages:
            return ChatResponse(response="AI 응답이 없습니다.")
        
        return ChatResponse(response=ai_messages[-1].content)

    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}", exc_info=True)
        
        # MCP 연결 오류인 경우 명확한 메시지 반환
        if "mcp" in str(e).lower() or "connection" in str(e).lower():
            return ChatResponse(response="MCP 서버와의 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.")
        
        return ChatResponse(response=f"오류가 발생했습니다: {str(e)}")


# =============================
# 개발 서버 실행
# =============================
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