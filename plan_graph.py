import json
import re
import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import MessagesState

# -----------------------------------------------------------------
# 1. 🚀 '하이브리드' 방식 임포트
# -----------------------------------------------------------------
from agent.plan_agents.input_agent import PlanInputAgent
from agent.plan_agents.validation_agent import ValidationAgent
from agent.plan_agents.loan_agent_node import LoanAgent
from agent.plan_agents.saving_agent import SavingAgentNode
from agent.plan_agents.fund_agent import FundAgentNode

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

print("--- (하이브리드 방식) 에이전트 모듈 로드 완료 ---")

# -----------------------------------------------------------------
# 2. 안전한 병합 함수
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
# 3. LangGraph 상태 정의
# -----------------------------------------------------------------
class GraphState(MessagesState):
    user_id: int
    plan_id: Optional[int] = None
    extracted_info: Annotated[Optional[Dict[str, Any]], update_extracted_info] = None
    input_completed: bool = False
    validated_plan_input: Optional[Dict[str, Any]] = None
    original_input: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
    loan_result: Optional[Dict[str, Any]] = None
    savings_recommendations: Optional[Dict[str, Any]] = None
    fund_analysis_result: Optional[Dict[str, Any]] = None
    final_summary: Optional[str] = None

# -----------------------------------------------------------------
# 4. 파일 경로 설정
# -----------------------------------------------------------------
try:
    CURRENT_SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_SCRIPT_PATH.parents[2]
    SAVING_CSV_PATH = "/Users/yoodongseok/Desktop/WooriAgent/saving_data.csv"
    FUND_JSON_PATH = "/Users/yoodongseok/Desktop/WooriAgent/fund_data.json"
except Exception:
    SAVING_CSV_PATH = "/Users/yoodongseok/Desktop/WooriAgent/saving_data.csv"
    FUND_JSON_PATH = "/Users/yoodongseok/Desktop/WooriAgent/fund_data.json"

print(f"--- (plan_graph) 파일 경로 로드 완료 ---")

# -----------------------------------------------------------------
# 5. 에이전트 인스턴스 생성
# -----------------------------------------------------------------
plan_input_agent = PlanInputAgent()
validator_agent = ValidationAgent()
loan_agent = LoanAgent()
saving_agent = SavingAgentNode()
fund_agent = FundAgentNode()
final_summarizer_llm = ChatOllama(model="qwen3:8b", temperature=0.1)

print("--- (하이브리드 방식) 에이전트 인스턴스 생성 완료 ---")

# -----------------------------------------------------------------
# 6. 어댑터 노드 정의 (Saving, Fund)
# -----------------------------------------------------------------
def savings_recommend_node(state: GraphState) -> GraphState:
    print("\n--- [E. 예/적금 추천 노드 시작] ---")
    try:
        user_plan_data = state.get("validated_plan_input", {})
        user_id = state.get("user_id")
        if not user_plan_data:
            raise ValueError("validated_plan_input 데이터가 없습니다.")
        target_years = int(user_plan_data.get("target_period_years", 1))
        period_months = target_years * 12
        temp_state = {
            "user_data": {
                "user_id": user_id,
                "age": user_plan_data.get("age", 30),
                "is_first_customer": user_plan_data.get("is_first_customer", False),
                "period_goal_months": period_months,
            },
            "csv_file_path": SAVING_CSV_PATH,
            "savings_recommendations": {},
        }
        result_dict = saving_agent.run(temp_state)
        rec = result_dict.get("savings_recommendations", {})
        return {
            "savings_recommendations": rec,
            "messages": [AIMessage(content=f"[예/적금 추천 완료] {len(rec.get('top_3_savings', []))}개 상품 추천")],
        }
    except Exception as e:
        logger.error(f"예/적금 추천 실패: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"예/적금 추천 실패: {e}")]}

def fund_recommend_node(state: GraphState) -> GraphState:
    print("\n--- [F. 펀드 추천 노드 시작] ---")
    try:
        temp_state = {"fund_data_path": FUND_JSON_PATH, "fund_analysis_result": {}}
        result_dict = fund_agent.run(temp_state)
        analysis = result_dict.get("fund_analysis_result", {})
        return {
            "fund_analysis_result": analysis,
            "messages": [AIMessage(content="[펀드 분석 완료]")],
        }
    except Exception as e:
        logger.error(f"펀드 분석 실패: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"펀드 분석 실패: {e}")]}

# -----------------------------------------------------------------
# 7. 공통 유틸 노드
# -----------------------------------------------------------------
async def handle_error_node(state: GraphState):
    error_msg = state.get("final_response", {}).get("message", "알 수 없는 오류")
    content = f"입력 검증에 실패했습니다: {error_msg}\n문제가 되는 부분을 수정해 다시 시도해주세요."
    return {"messages": [AIMessage(content=content)]}

# ✅ 수정된 부분 시작
async def create_final_summary_node(state: GraphState):
    logger.info("📋 최종 요약 생성 중...")

    loan_result = state.get("loan_result", {})
    savings_result = state.get("savings_recommendations", {})
    fund_result = state.get("fund_analysis_result", {})

    loan_text = loan_result.get("llm_explanation", "대출 정보 없음")
    saving_text = savings_result.get("llm_output", savings_result if savings_result else "예적금 정보 없음")
    fund_text = fund_result.get("llm_output", fund_result if fund_result else "펀드 정보 없음")

    prompt = f"""
    [페르소나]
    당신은 신뢰감 있고 따뜻한 어조로 고객의 재무 목표를 함께 설계하는
    우리은행의 전문 재무설계사(Financial Planner)입니다.
    고객의 자산 상황, 대출 조건, 투자 성향을 고려해 통합적인 재무 플랜을 제시해야 합니다.

    [TASK]
    1. 아래 [대출 결과], [예/적금 결과], [펀드 결과] 내용을 분석하여
       고객에게 제공할 **최종 재무 계획 요약 보고서**를 작성하세요.
    2. 각 항목별 주요 포인트를 요약하고,
       전체적인 재무 방향(예: 안정형, 성장형, 균형형)을 제안하세요.
    3. 문체는 고객 중심적이고 긍정적이며 신뢰감 있는 상담 어조로 작성하세요.
    4. 반드시 ‘대출 실행 후 남은 금액’을 중심으로 고객이 예/적금 및 펀드 운용을
       어떻게 병행할 수 있을지 제안하세요.

    [출력 형식]
    ① 대출 상품명과 대출 요약  
    ② 예/적금 제안 요약  
    ③ 펀드 제안 요약  
    ④ 종합 재무 계획 제안 (3~4문장)

    [대출 결과]
    {json.dumps(loan_text, ensure_ascii=False, indent=2)}

    [예/적금 결과]
    {json.dumps(saving_text, ensure_ascii=False, indent=2)}

    [펀드 결과]
    {json.dumps(fund_text, ensure_ascii=False, indent=2)}

    [최종 작성 지침]
    - 문체는 “~하실 수 있습니다.” / “~하는 것이 좋습니다.” 형태로 공손하게.
    - 금액 표시는 쉼표(,)를 포함해 정확히.
    - 고객에게 긍정적 인상과 신뢰감을 주는 어조 유지.
    """

    response = await final_summarizer_llm.ainvoke(prompt)
    summary_text = response.content.strip()

    logger.info("✅ 최종 요약 생성 완료")
    return {"final_summary": summary_text, "messages": [AIMessage(content=summary_text)]}
# ✅ 수정된 부분 끝

async def update_state_after_validation(state: GraphState):
    logger.info("✅ [Shim Node] 검증 성공 → DB 저장용 데이터 준비 중")
    status = state.get("final_response", {}).get("status", "error")
    if status == "success":
        validated_data = state.get("final_response", {}).get("data", {})
        validated_data["user_id"] = state.get("user_id")
        return {
            "validated_plan_input": validated_data,
            "messages": [AIMessage(content="[입력 검증 완료] 데이터 저장 준비 완료")],
        }
    else:
        return {"messages": [AIMessage(content="[검증 실패] 유효하지 않은 입력 데이터")]}

# -----------------------------------------------------------------
# 8. 라우터 정의
# -----------------------------------------------------------------
def route_after_input_check(state: GraphState):
    if state.get("input_completed", False):
        return "validate_input"
    return END

def route_after_validation(state: GraphState):
    status = state.get("final_response", {}).get("status", "error")
    if status == "success":
        return "update_state_after_validation"
    else:
        return "handle_error"

# -----------------------------------------------------------------
# 9. 그래프 생성
# -----------------------------------------------------------------
def create_graph():
    print("--- (하이브리드 방식) 그래프 조립 시작 ---")
    workflow = StateGraph(GraphState)

    def ensure_user_id(state: GraphState):
        if not state.get("user_id"):
            state["user_id"] = 1
            print("⚙️ user_id가 없어 기본값 1로 설정했습니다.")
        return {}

    workflow.add_node("ensure_user_id", ensure_user_id)
    workflow.add_node("extract_info", plan_input_agent.create_extraction_node())
    workflow.add_node("check_completeness", plan_input_agent.create_check_completeness_node())
    workflow.add_node("save_to_db", plan_input_agent.create_save_to_db_node())
    workflow.add_node("validate_input", validator_agent.create_validation_node())
    workflow.add_node("loan_recommend", loan_agent.create_recommendation_node())
    workflow.add_node("saving_recommend", savings_recommend_node)
    workflow.add_node("fund_recommend", fund_recommend_node)
    workflow.add_node("update_state_after_validation", update_state_after_validation)
    workflow.add_node("create_final_summary", create_final_summary_node)
    workflow.add_node("handle_error", handle_error_node)

    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "check_completeness")
    workflow.add_conditional_edges(
        "check_completeness",
        route_after_input_check,
        {"validate_input": "validate_input", END: END},
    )
    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "update_state_after_validation": "update_state_after_validation",
            "handle_error": "handle_error",
        },
    )
    workflow.add_edge("update_state_after_validation", "save_to_db")
    workflow.add_edge("save_to_db", "loan_recommend")
    workflow.add_edge("loan_recommend", "saving_recommend")
    workflow.add_edge("saving_recommend", "fund_recommend")
    workflow.add_edge("fund_recommend", "create_final_summary")
    workflow.add_edge("create_final_summary", END)
    workflow.add_edge("handle_error", END)

    print("--- ✅ 그래프 컴파일 완료 ---")
    return workflow.compile(checkpointer=MemorySaver())
