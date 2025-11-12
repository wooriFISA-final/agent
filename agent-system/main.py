import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph.schemas.state import LLMStateSchema
from agents.registry.agent_registry import AgentRegistry
from graph.builder.graph_builder import GraphBuilder
from core.llm.llm_manger import LLMManager
from core.logging.logger import setup_logger
from graph.factory import mk_graph

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

logger = setup_logger()
app = FastAPI(title="Multi-Agent Planner")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# === MCP 클라이언트 객체 정의 추후에 다른 코드 파일에 옮길 예정===
transport = StreamableHttpTransport(
    url="http://localhost:8888/mcp/"
    #headers={"X-Account-Password": "1234"}
)
mcp_client = Client(transport)


# 그래프 초기화
# graph = create_graph()
# graph = mk_graph("graph.yaml")  # UserRegistrationAgent 포함되어 있어야 함

AgentRegistry.auto_discover("agents.implementations")
logger.info(AgentRegistry.list_agents())
# 그래프 빌드
builder = GraphBuilder(LLMStateSchema)
builder.add_agent_node("user_regri", "user_registration") \
    .set_entry_point("user_regri") \
    .set_finish_point("user_regri")

graph = builder.build()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"

class ChatResponse(BaseModel):
    response: str

#예도 변경해야 할듯
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Front → LLM Graph → Response"""
    if not graph:
        return ChatResponse(response="❌ Agent graph is not initialized properly.")

    config = {"configurable": {"thread_id": request.session_id}}
    # # 초기 상태 설정
    # initial_state = {
    #     "query": "계획을 수정하고 싶어"
    # }
    try:
        logger.info(f"유저 메시지 request : {request.message}")
        messages = [HumanMessage(content=request.message)]
        result = await graph.ainvoke(
            {"messages": messages},
            config=config
        )
        # result = await graph.ainvoke({"messages": [HumanMessage(content=request.message)]}, config=config)

        final_response = result.get("messages")

        if not final_response:
            return ChatResponse(response="응답을 생성할 수 없습니다.")
        
        if isinstance(final_response, list):
            # Simple join for now.
            final_response = " ".join(map(str, final_response))

        return ChatResponse(response=final_response)

    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        return ChatResponse(response=f"오류가 발생했습니다: {e}")
@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Agent API is running 🚀"}

# ----------------------------
# 서버 직접 실행용 (선택)
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting API Server on http://localhost:8080")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)