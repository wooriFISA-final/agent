import json
import re
import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import MessagesState

# -----------------------------------------------------------------
# 1️⃣ 프로젝트 구조 맞춤 임포트
# -----------------------------------------------------------------
from plan_agents.input_agent import PlanInputAgent
from plan_agents.validation_agent import ValidationAgent
from plan_agents.loan_agent_node import LoanAgent
from plan_agents.saving_agent import SavingAgentNode
from plan_agents.fund_agent import FundAgentNode
from plan_agents.summary_agent import SummaryAgent

# -----------------------------------------------------------------
# 2️⃣ 로깅 설정
# -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
print("--- ✅ 에이전트 모듈 로드 완료 ---")

# -----------------------------------------------------------------
# 3️⃣ 병합 함수
# -----------------------------------------------------------------
def update_extracted_info(original: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if original is None:
        original = {}
    if new is None:
        return original
    combined = original.copy()
    combined.update(new)
    return combined

# -----------------------------------------------------------------
# 4️⃣ GraphState 정의
# -----------------------------------------------------------------
class GraphState(MessagesState):
    user_id: int
    plan_id: Optional[int] = None
    extracted_info: Annotated[Optional[Dict[str, Any]], update_extracted_info] = None
    summary_for_validation: Optional[Dict[str, Any]] = None 
    input_completed: bool = False
    validated_plan_input: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
    loan_result: Optional[Dict[str, Any]] = None
    savings_recommendations: Optional[Dict[str, Any]] = None
    fund_analysis_result: Optional[Dict[str, Any]] = None
    summary_result: Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------
# 5️⃣ 파일 경로 설정
# -----------------------------------------------------------------
try:
    CURRENT_SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_SCRIPT_PATH.parents[0]
    SAVING_CSV_PATH = "/Users/yoodongseok/Desktop/WooriAgent/agent/saving_data.csv"
    FUND_JSON_PATH = "/Users/yoodongseok/Desktop/WooriAgent/agent/fund_data.json"
except Exception:
    SAVING_CSV_PATH = "/Users/yoodongseok/Desktop/WooriAgent/agent/saving_data.csv"
    FUND_JSON_PATH = "/Users/yoodongseok/Desktop/WooriAgent/agent/fund_data.json"

# -----------------------------------------------------------------
# 6️⃣ 에이전트 인스턴스 생성
# -----------------------------------------------------------------
plan_input_agent = PlanInputAgent()
validator_agent = ValidationAgent()
loan_agent = LoanAgent()
saving_agent = SavingAgentNode()
fund_agent = FundAgentNode()
summary_agent = SummaryAgent()

# -----------------------------------------------------------------
# 7️⃣ 노드 정의
# -----------------------------------------------------------------
def savings_recommend_node(state: GraphState) -> GraphState:
    try:
        user_id = state.get("user_id")
        user_plan_data = state.get("validated_plan_input", {})
        if not user_plan_data:
            raise ValueError("validated_plan_input 데이터가 없습니다.")

        temp_state = {
            "user_data": {
                "user_id": user_id,
                "age": user_plan_data.get("age", 30),
                "period_goal_months": 36,
                "is_first_customer": user_plan_data.get("is_first_customer", False),
            },
            "csv_file_path": SAVING_CSV_PATH,
        }

        result_dict = saving_agent.run(temp_state)
        rec = result_dict.get("savings_recommendations", {})
        return {
            "savings_recommendations": rec,
            "messages": [AIMessage(content=f"💰 예/적금 추천 완료 ({len(rec.get('top_3_savings', []))}개 상품)")],
        }
    except Exception as e:
        logger.error(f"예/적금 추천 실패: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"예/적금 추천 실패: {e}")]}


def fund_recommend_node(state: GraphState) -> GraphState:
    try:
        result_dict = fund_agent.run({"fund_data_path": FUND_JSON_PATH})
        analysis = result_dict.get("fund_analysis_result", {})
        return {"fund_analysis_result": analysis, "messages": [AIMessage(content="📈 펀드 분석 완료")]}
    except Exception as e:
        logger.error(f"펀드 분석 실패: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"펀드 분석 실패: {e}")]}


async def summary_node(state: GraphState):
    logger.info("🧩 SummaryAgent 실행 중...")
    user_id = state.get("user_id")
    plan_data = state.get("validated_plan_input", {})
    loan_data = state.get("loan_result", {})
    saving_data = state.get("savings_recommendations", {})
    fund_data = state.get("fund_analysis_result", {})

    summary_output = summary_agent.run(
        user_id=user_id,
        plan_data=plan_data,
        loan_data=loan_data,
        saving_results=saving_data,
        fund_results=fund_data
    )

    return {"summary_result": summary_output, "messages": [AIMessage(content=summary_output["summary_text"])]}


async def handle_error_node(state: GraphState):
    msg = state.get("final_response", {}).get("message", "알 수 없는 오류")
    return {"messages": [AIMessage(content=f"⚠️ 입력 검증 실패: {msg}")]}


async def update_state_after_validation(state: GraphState):
    status = state.get("final_response", {}).get("status", "error")
    if status == "success":
        validated_data = state.get("final_response", {}).get("data", {})
        validated_data["user_id"] = state.get("user_id")
        return {
            "validated_plan_input": validated_data,
            "messages": [AIMessage(content="✅ 입력 검증 완료 — DB 저장 준비 완료")],
        }
    else:
        return {"messages": [AIMessage(content="❌ 유효하지 않은 입력 데이터")]}

# -----------------------------------------------------------------
# 8️⃣ 라우터 및 그래프 연결
# -----------------------------------------------------------------
def route_after_input_check(state: GraphState):
    if state.get("input_completed", False):
        return "validate_input"
    return END


def route_after_validation(state: GraphState):
    status = state.get("final_response", {}).get("status", "error")
    return "update_state_after_validation" if status == "success" else "handle_error"

# -----------------------------------------------------------------
# 9️⃣ 그래프 생성 함수
# -----------------------------------------------------------------
def create_graph():
    workflow = StateGraph(GraphState)

    async def completeness_wrapper(state: GraphState):
        node_function = plan_input_agent.create_check_completeness_node()
        result = await node_function(state)
        return result

    # ------------------- 노드 등록 -------------------
    workflow.add_node("extract_info", plan_input_agent.create_extraction_node())
    workflow.add_node("check_completeness", completeness_wrapper)
    workflow.add_node("validate_input", validator_agent.create_validation_node())
    workflow.add_node("update_state_after_validation", update_state_after_validation)
    workflow.add_node("loan_recommend", loan_agent.create_recommendation_node())
    workflow.add_node("saving_recommend", savings_recommend_node)
    workflow.add_node("fund_recommend", fund_recommend_node)
    workflow.add_node("summary_node", summary_node)
    workflow.add_node("handle_error", handle_error_node)

    workflow.set_entry_point("extract_info")

    # ------------------- 노드 연결 -------------------
    workflow.add_edge("extract_info", "check_completeness")

    workflow.add_conditional_edges(
        "check_completeness",
        route_after_input_check,
        {"validate_input": "validate_input", END: END}
    )

    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"update_state_after_validation": "update_state_after_validation", "handle_error": "handle_error"}
    )

    # ✅ save_to_db 제거 후, 바로 loan_recommend로 연결
    workflow.add_edge("update_state_after_validation", "loan_recommend")

    workflow.add_edge("loan_recommend", "saving_recommend")
    workflow.add_edge("saving_recommend", "fund_recommend")
    workflow.add_edge("fund_recommend", "summary_node")
    workflow.add_edge("summary_node", END)
    workflow.add_edge("handle_error", END)

    print("--- ✅ LangGraph Workflow 컴파일 완료 ---")
    return workflow.compile(checkpointer=MemorySaver())