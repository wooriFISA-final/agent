from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager

from agents.registry.agent_registry import AgentRegistry
from core.logging.logger import setup_logger
from graph.builder.graph_builder import GraphBuilder
from graph.schemas.state import LLMStateSchema
from core.mcp.mcp_manager import MCPManager

from langchain_core.messages import AIMessage
logger = setup_logger()

# =============================
# Lifespan 이벤트 핸들러
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 및 종료 시 초기화 / 정리 작업"""
    global graph

    logger.info("🚀 Starting Multi-Agent System...")

    # 1️⃣ MCP 클라이언트 초기화
    mcp_manager = MCPManager()
    mcp_manager.initialize(url="http://localhost:8888/mcp/")

    # ✅ MCP 서버 연결
    await mcp_manager.connect()
    logger.info("✅ MCP Manager initialized and connected")

    # 2️⃣ Agent 자동 검색 및 등록
    AgentRegistry.auto_discover("agents.implementations")
    logger.info(f"✅ Registered agents: {AgentRegistry.list_agents()}")

    # 3️⃣ 그래프 빌드
    builder = GraphBuilder(LLMStateSchema)
    builder.add_agent_node("user_regri", "user_registration") \
        .set_entry_point("user_regri") \
        .set_finish_point("user_regri")

    graph = builder.build()
    logger.info("✅ Agent graph built successfully")

    # startup 완료 후 제어권 반환
    yield

    # shutdown 시 처리 (예: MCP 연결 종료)
    await mcp_manager.close()
    logger.info("🧹 MCP connection closed. Application shutdown complete.")


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

        logger.info(f"최종 응답 결과 포멧 : {final_response}")
        if not final_response:
            logger.warning("⚠️ No response generated")
            return ChatResponse(response="응답을 생성할 수 없습니다.")

        # # 메시지 리스트를 문자열로 변환
        # if isinstance(final_response, list):
        #     final_response = " ".join(map(str, final_response))

        logger.info(f"✅ Response generated: {final_response[:100]}...")
        return ChatResponse(response=final_response[AIMessage])

    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}", exc_info=True)
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
