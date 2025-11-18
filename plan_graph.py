import os
import json
import logging
from typing import Dict, Any, Optional, Annotated

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import MessagesState

# LangChain core
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# ----------------------------------
# 1️⃣ 에이전트 임포트
# ----------------------------------
from plan_agents.input_agent import PlanInputAgent
from plan_agents.validation_agent import ValidationAgent
from plan_agents.loan_agent_node import LoanAgent
from plan_agents.saving_agent import SavingAgentNode
from plan_agents.fund_agent import FundAgentNode
from plan_agents.summary_agent import SummaryAgent

# ----------------------------------
# 2️⃣ 로깅 & LangSmith 설정 (env 기반)
# ----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
print("--- ✅ 에이전트 모듈 로드 완료 ---")

load_dotenv()  # .env 로드

# ✨ LangSmith V2 트레이싱: env 만으로 활성화 (별도 Client 불필요)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")  # 트레이싱 on
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "WooriPlanner"))
# API Key는 .env에 LANGCHAIN_API_KEY=... 로 넣어두세요.

if not os.getenv("LANGCHAIN_API_KEY"):
    logger.warning("⚠️ LANGCHAIN_API_KEY가 설정되지 않았습니다. LangSmith 트레이싱이 비활성화될 수 있습니다.")

# ----------------------------------
# 3️⃣ 병합 함수
# ----------------------------------
def update_extracted_info(original: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if original is None:
        original = {}
    if new is None:
        return original
    combined = original.copy()
    combined.update(new)
    return combined

# ----------------------------------
# 4️⃣ GraphState 정의
# ----------------------------------
class GraphState(MessagesState):
    user_id: int
    plan_id: Optional[int] = None
    extracted_info: Annotated[Optional[Dict[str, Any]], update_extracted_info] = None
    input_completed: bool = False
    validated_plan_input: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
    loan_result: Optional[Dict[str, Any]] = None
    savings_recommendations: Optional[Dict[str, Any]] = None
    fund_analysis_result: Optional[Dict[str, Any]] = None
    summary_result: Optional[Dict[str, Any]] = None

# ----------------------------------
# 5️⃣ 에이전트 인스턴스 생성
# ----------------------------------
plan_input_agent = PlanInputAgent()
validator_agent = ValidationAgent()
loan_agent = LoanAgent()
saving_agent = SavingAgentNode()
fund_agent = FundAgentNode()
summary_agent = SummaryAgent()

# ----------------------------------
# 6️⃣ 보조 노드 정의
# ----------------------------------
async def handle_error_node(state: GraphState):
    msg = state.get("final_response", {}).get("message", "⚠️ 알 수 없는 오류가 발생했습니다.")
    return {"messages": [AIMessage(content=msg)]}

async def update_state_after_validation(state: GraphState):
    status = state.get("final_response", {}).get("status", "error")
    if status == "success":
        validated_data = state.get("final_response", {}).get("data", {})
        validated_data["user_id"] = state.get("user_id")
        return {
            "validated_plan_input": validated_data,
            "messages": [AIMessage(content="✅ 입력 검증 완료 — DB 저장 완료")],
        }
    else:
        return {"messages": [AIMessage(content="❌ 유효하지 않은 입력 데이터입니다.")]}

# ----------------------------------
# 7️⃣ 조건부 라우터
# ----------------------------------
def route_after_input(state: GraphState):
    return "validate_input" if state.get("input_completed", False) else END

def route_after_validation(state: GraphState):
    status = state.get("final_response", {}).get("status", "error")
    return "update_state_after_validation" if status == "success" else "handle_error"

# ----------------------------------
# 8️⃣ 그래프 생성 함수
# ----------------------------------
def create_graph():
    workflow = StateGraph(GraphState)

    # 노드 등록
    workflow.add_node("extract_info", plan_input_agent.run)
    workflow.add_node("validate_input", validator_agent.run)
    workflow.add_node("update_state_after_validation", update_state_after_validation)
    workflow.add_node("loan_recommend", loan_agent.run)
    workflow.add_node("saving_recommend", saving_agent.run)
    workflow.add_node("fund_recommend", fund_agent.run)
    workflow.add_node("summary_node", summary_agent.run)
    workflow.add_node("handle_error", handle_error_node)

    # 진입점
    workflow.set_entry_point("extract_info")

    # 연결
    workflow.add_conditional_edges(
        "extract_info",
        route_after_input,
        {"validate_input": "validate_input", END: END}
    )

    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"update_state_after_validation": "update_state_after_validation", "handle_error": "handle_error"}
    )

    workflow.add_edge("update_state_after_validation", "loan_recommend")
    workflow.add_edge("loan_recommend", "saving_recommend")
    workflow.add_edge("saving_recommend", "fund_recommend")
    workflow.add_edge("fund_recommend", "summary_node")
    workflow.add_edge("summary_node", END)
    workflow.add_edge("handle_error", END)

    print("--- ✅ LangGraph Workflow 컴파일 완료 ---")
    return workflow.compile(checkpointer=MemorySaver())
