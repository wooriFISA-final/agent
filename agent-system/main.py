"""
api_server.py
Front ↔ Agent System FastAPI API Server
- main.py는 개발용, api_server는 프론트 통신용
"""

import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from agents.registry.agent_registry import AgentRegistry
from graph.builder.graph_builder import GraphBuilder
from core.llm.llm_manger import LLMManager
from core.logging.logger import setup_logger
from graph.factory import mk_graph
# ----------------------------
# 기본 설정
# ----------------------------
app = FastAPI(title="Agent System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스에서는 특정 도메인으로 제한하는 게 좋아
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = setup_logger()

# ----------------------------
# 그래프 초기화
# ----------------------------
graph = mk_graph("graph.yaml")


# ----------------------------
# Request/Response 모델
# ----------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    response: str


# ----------------------------
# Chat API Endpoint
# ----------------------------
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
        result = await graph.ainvoke({"query": request.message}, config=config)
        print(result)
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


# ----------------------------
# 헬스체크 엔드포인트
# ----------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Agent API is running 🚀"}


# ----------------------------
# 서버 직접 실행용 (선택)
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
