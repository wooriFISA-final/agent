import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# --- 1. (중요) LangGraph 워크플로우 임포트 ---c
# (main.py는 agent/ 폴더에 있으므로,
#  형제 폴더인 'plan_agents'의 모듈을 바로 import 합니다)
try:
    # (plan_graph.py에서 create_workflow, AgentGraphState를 import)
    from plan_graph import create_workflow, AgentGraphState
except ImportError as e:
    print(f"Import Error: {e}")
    print("경로 문제 발생 시, VS Code를 재시작하거나 Python 경로를 확인하세요.")
    # (VS Code가 'agent' 폴더를 루트로 인식하지 못할 경우를 대비한 예외 처리)
    # sys.path.append(str(Path(__file__).resolve().parent.parent)) 
    # from agent.plan_graph import create_workflow, AgentGraphState
    exit()

print("--- LangGraph 'app' 로드 중... ---")
# 1-1. 그래프 워크플로우 컴파일
langgraph_app = create_workflow()
print("--- LangGraph 'app' 로드 완료. FastAPI 서버 설정 ---")

# --- 2. FastAPI 앱 생성 ---
app_fastapi = FastAPI(
    title="Final Project Agent API",
    description="LangGraph 에이전트를 FastAPI로 노출하는 API 서버"
)

# --- 3. (필수!) CORS (Cross-Origin Resource Sharing) 설정 ---
# (보안상의 이유로, 브라우저는 8000번 포트에서 5173번 포트로의 요청을 차단합니다)
# (이 코드는 5173 포트의 요청을 '허용'합니다)
app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # ⬅️ 님의 React 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"], # (GET, POST 등 모든 메소드 허용)
    allow_headers=["*"], # (모든 헤더 허용)
)

# --- 4. 프론트엔드와 통신할 입/출력 모델 정의 ---
# (LangChain의 BaseMessage는 복잡하므로, 간단한 dict로 받습니다)
class AgentInvokeRequest(BaseModel):
    user_id: int = 1 # (임시)
    messages: List[Dict[str, str]] # (예: [{"role": "user", "content": "안녕하세요"}])

# (AgentGraphState의 키들을 포함하는 응답 모델)
class AgentInvokeResponse(BaseModel):
    messages: List[Dict[str, Any]]
    input_completed: bool
    validation_passed: bool
    final_plan: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# --- 5. (핵심) API 엔드포인트 생성 ---
# (프론트엔드는 "http://localhost:8000/invoke_agent"로 POST 요청을 보냄)
@app_fastapi.post("/invoke_agent", response_model=AgentInvokeResponse)
async def invoke_agent(request: AgentInvokeRequest):
    """
    프론트엔드로부터 'messages' 배열을 받아 LangGraph를 실행하고,
    'input_completed' 상태와 'messages' (AI의 새 답변 포함)를 반환합니다.
    """
    print(f"--- API 요청 수신 (User ID: {request.user_id}) ---")
    
    # 1. (초기 상태 구성)
    # (파일 경로는 서버가 알고 있는 절대 경로로 설정)
    project_root = Path(__file__).resolve().parent.parent # (FINAL_PROJECT 폴더)
    
    # (LangChain BaseMessage 객체로 변환)
    from langchain_core.messages import AIMessage, HumanMessage
    messages_for_graph = []
    for msg in request.messages:
        if msg["role"] == "user":
            messages_for_graph.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_for_graph.append(AIMessage(content=msg["content"]))

    initial_state = {
        "user_id": request.user_id,
        "messages": messages_for_graph,
        "fund_data_path": str(project_root / "fund_data.json"),
        "savings_data_path": str(project_root / "saving_data.csv")
        # (다른 키들은 노드가 채울 것이므로 비워둠)
    }
    
    # 2. (LangGraph 실행)
    # .invoke()는 동기식(느림), .ainvoke()는 비동기식(빠름)
    final_state = await langgraph_app.ainvoke(initial_state)
    
    # 3. (결과 반환)
    # (BaseMessage 객체를 다시 프론트가 쓰기 쉬운 dict로 변환)
    final_messages_dict = [
        {"role": msg.type, "content": msg.content} for msg in final_state["messages"]
    ]
    
    return {
        "messages": final_messages_dict, # ⬅️ AI의 새 답변이 포함된 전체 대화
        "final_plan": final_state.get("final_plan"), # ⬅️ 계획이 완성되면 여기 담김
        "input_completed": final_state.get("input_completed", False),
        "validation_passed": final_state.get("validation_passed", False),
        "error_message": final_state.get("error_message")
    }

# --- 6. (VS Code) 이 파일을 직접 실행할 때 서버 켜기 ---
if __name__ == "__main__":
    print("--- FastAPI 서버를 시작하려면 터미널에서 다음을 실행하세요 ---")
    print("--- (위치: C:\\final_project\\agent) ---")
    print("--- uvicorn main:app_fastapi --reload --port 8000 ---")
    
    # (개발 편의를 위해 여기서 바로 실행)
    # uvicorn.run(app_fastapi, host="127.0.0.1", port=8000)