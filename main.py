from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from agent.plan_graph import create_graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="Multi-Agent Planner")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 그래프 초기화
graph = create_graph()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Front → AI Graph → Output"""
    config = {"configurable": {"thread_id": request.session_id}}
    result = await graph.ainvoke({"messages": [HumanMessage(content=request.message)]}, config=config)

    messages = result.get("messages", [])
    print(f"messages : {messages}")
    if not messages:
        return ChatResponse(response="응답을 생성할 수 없습니다.")
    return ChatResponse(response=messages[-1].content)

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Agent API is running 🚀"}
