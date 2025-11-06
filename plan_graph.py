# plan_graph.py (최종)

import json
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver 
from pathlib import Path 

# -----------------------------------------------------------------
# 1. 🚀 우리 '노드' 파일들 임포트
# -----------------------------------------------------------------
# plan_graph.py
from agent.plan_agents.input_agent import PlanAgentNode
import agent.plan_agents.validation_agent as validator_agent
from agent.plan_agents.loan_agent_node import LoanAgentNode
from agent.plan_agents.saving_agent import SavingAgentNode
from agent.plan_agents.fund_agent import FundAgentNode
  # 👈 [추가] 1. 펀드 에이전트 클래스 임포트

print("--- 모든 에이전트 모듈 로드 완료 ---")

# -----------------------------------------------------------------
# 2. 📊 LangGraph State 정의
# -----------------------------------------------------------------
class GraphState(TypedDict):
    messages: List[Dict[str, Any]]
    responses: Dict[str, Any] # 👈 모든 결과가 여기에 누적됩니다.
    
    user_id: int  
    plan_id: Optional[int] 
    
    input_completed: bool
    validation_passed: bool
    error_message: str
    
# -----------------------------------------------------------------
# 2-1. 📂 전역 경로 설정 (CSV 및 JSON)
# -----------------------------------------------------------------
try:
    CURRENT_SCRIPT_PATH = Path(__file__).resolve()
    # (주의!) saving_data.csv의 위치에 따라 .parents[N] 숫자를 조정하세요.
    PROJECT_ROOT = CURRENT_SCRIPT_PATH.parents[2] 
    SAVING_CSV_PATH = str(PROJECT_ROOT / 'saving_data.csv')
    FUND_JSON_PATH = str(PROJECT_ROOT / 'fund_data.json') # 👈 [추가] 2. 펀드 JSON 경로
    
    print(f"--- (plan_graph) 예/적금 CSV 파일 경로 로드: {SAVING_CSV_PATH} ---")
    print(f"--- (plan_graph) 펀드 JSON 파일 경로 로드: {FUND_JSON_PATH} ---")
except Exception as e:
    SAVING_CSV_PATH = 'saving_data.csv' # (fallback)
    FUND_JSON_PATH = 'fund_data.json'   # (fallback)
    print(f"--- (plan_graph) 파일 경로 (fallback): {SAVING_CSV_PATH}, {FUND_JSON_PATH} ---")

    
# -----------------------------------------------------------------
# 3. 🤖 에이전트 노드 인스턴스 생성
# -----------------------------------------------------------------
plan_agent = PlanAgentNode()
loan_agent = LoanAgentNode() 
saving_agent = SavingAgentNode()
fund_agent = FundAgentNode() # 👈 [추가] 3. 펀드 에이전트 인스턴스 생성

# -----------------------------------------------------------------
# 4. 🤖 노드(Node) 래퍼(Wrapper) 함수 정의
# -----------------------------------------------------------------

# [노드 1: 상담원]
def input_node(state: GraphState) -> GraphState:
    print("\n--- [A. 입력 노드 시작] ---")
    return plan_agent.run_as_node(state)

# [노드 2: 심사관]
def validation_node(state: GraphState) -> GraphState:
    print("\n--- [B. 검증 노드 시작] ---")
    # (기존 코드와 동일)
    responses_to_validate = state.get("responses", {})
    if not responses_to_validate:
        state["validation_passed"] = False
        state["error_message"] = "수집된 데이터가 없습니다."
        return state
    final_result_dict = validator_agent.run_agent_executor(responses_to_validate)
    if final_result_dict.get("status") == "success":
        state["validation_passed"] = True
        state["responses"] = final_result_dict.get("data", {})
    else:
        state["validation_passed"] = False
        state["error_message"] = final_result_dict.get("message", "알 수 없는 검증 오류")
    return state

# [노드 3: 에러 핸들러]
def handle_error_node(state: GraphState) -> GraphState:
    print("\n--- [C-1. 에러 처리 노드 시작] ---")
    # (기존 코드와 동일)
    error_msg = state.get("error_message", "알 수 없는 오류로 재시작합니다.")
    print(f"오류 발생: {error_msg}")
    state["messages"].append({"role": "assistant", "content": f"오류가 발생했습니다: {error_msg}\n정보를 다시 입력해주세요."})
    return state

# [노드 4: 저장 및 요약]
def save_and_summarize_node(state: GraphState) -> GraphState:
    print("\n--- [C-2. 저장/요약 노드 시작] ---")
    # (기존 코드와 동일)
    final_data = state.get("responses", {})
    user_id_to_save = state.get("user_id")
    if not final_data or not user_id_to_save:
        return state
    try:
        new_plan_id = plan_agent.save_to_db(final_data, user_id_to_save)
        if not new_plan_id or not isinstance(new_plan_id, int):
             raise Exception("'save_to_db'에서 유효한 'plan_id'를 반환하지 않았습니다.")
        state["plan_id"] = new_plan_id
        plan_agent.summarize(final_data)
    except Exception as e:
        state["error_message"] = str(e)
    return state

# [노드 5: 대출 추천]
def loan_recommend_node(state: GraphState) -> GraphState:
    print("\n--- [D. 대출 추천 노드 시작] ---")
    # (기존 코드와 동일)
    user_id = state.get("user_id")
    if not user_id:
        state["error_message"] = "사용자 ID를 찾을 수 없습니다."
        return state
    try:
        loan_result = loan_agent.run(user_id=user_id, plan_id=0) 
        if "message" in loan_result:
            state["error_message"] = loan_result['message']
        else:
            state["responses"].update(loan_result) # 결과를 'responses'에 누적
    except Exception as e:
        state["error_message"] = f"대출 추천 중 오류: {str(e)}"
    return state
    
# [노드 6: 예/적금 추천 (어댑터)]
def savings_recommend_node(state: GraphState) -> GraphState:
    print("\n--- [E. 예/적금 추천 노드 (어댑터) 시작] ---")
    # (기존 코드와 동일)
    try:
        user_plan_data = state.get("responses", {})
        user_id = state.get("user_id")

        target_years = int(user_plan_data.get("target_period_years", 1))
        period_months = target_years * 12

        user_data_for_savings = {
            "user_id": user_id,
            "age": user_plan_data.get("age", 30), 
            "is_first_customer": user_plan_data.get("is_first_customer", False),
            "period_goal_months": period_months
        }
        
        temp_savings_state = {
            "user_data": user_data_for_savings,
            "csv_file_path": SAVING_CSV_PATH,
            "savings_recommendations": {}
        }
        result_dict = saving_agent.run(temp_savings_state)
        
        state["responses"]["savings_recommendations"] = result_dict.get("savings_recommendations", {})
        print("--- [E. 예/적금 추천 노드 (어댑터) 완료] ---")
    except Exception as e:
        print(f"❌ 예/적금 추천 중 심각한 오류 발생: {e}")
        state["error_message"] = f"예/적금 추천 중 오류: {str(e)}"
    return state
    
# -----------------------------------------------------------------
# 4-1. 🤖 [어댑터 노드] 펀드 추천 (새로 추가)
# -----------------------------------------------------------------
def fund_recommend_node(state: GraphState) -> GraphState:
    """
    이것이 펀드 분석을 위한 '어댑터' 함수입니다.
    'saving_agent'와 달리 'user_data'가 필요 없고, 오직 '파일 경로'만 필요합니다.
    """
    print("\n--- [F. 펀드 추천 노드 (어댑터) 시작] ---")
    
    try:
        # 1. 'fund_agent'가 요구하는 '임시 상태' 객체(딕셔너리) 생성
        temp_fund_state = {
            "fund_data_path": FUND_JSON_PATH, # 위에서 정의한 전역 경로 사용
            "fund_analysis_result": {}
        }

        # 2. 'fund_agent'의 'run' 메서드 직접 호출 (import한 클래스 사용)
        result_dict = fund_agent.run(temp_fund_state)
        
        # 3. 결과를 메인 'GraphState'의 'responses'에 병합
        state["responses"]["fund_analysis_result"] = result_dict.get("fund_analysis_result", {})
        print("--- [F. 펀드 추천 노드 (어댑터) 완료] ---")

    except Exception as e:
        print(f"❌ 펀드 추천 중 심각한 오류 발생: {e}")
        state["error_message"] = f"펀드 추천 중 오류: {str(e)}"
    
    return state
    
# -----------------------------------------------------------------
# 5. 🔗 그래프 엣지(Edge) 조립 및 컴파일
# -----------------------------------------------------------------

def create_graph():
    print("--- 그래프 조립 시작 ---")
    workflow = StateGraph(GraphState)

    # 노드 등록
    workflow.add_node("input_node", input_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("handle_error_node", handle_error_node)
    workflow.add_node("save_and_summarize_node", save_and_summarize_node)
    workflow.add_node("loan_recommend_node", loan_recommend_node) 
    workflow.add_node("savings_recommend_node", savings_recommend_node)
    workflow.add_node("fund_recommend_node", fund_recommend_node) # 👈 [추가] 4. 펀드 노드 등록

    # 엣지 연결
    workflow.set_entry_point("input_node") 
    # workflow.add_edge("input_node", "validation_node")
    workflow.add_edge("input_node", END)
    
    # workflow.add_edge("handle_error_node", "input_node") # 재시도

    # # 엣지 수정: ... -> savings -> funds -> END
    # workflow.add_edge("save_and_summarize_node", "loan_recommend_node") 
    # workflow.add_edge("loan_recommend_node", "savings_recommend_node") 
    # workflow.add_edge("savings_recommend_node", "fund_recommend_node") # 👈 [수정] 예/적금이 끝나면 펀드로
    # workflow.add_edge("fund_recommend_node", END)                       # 👈 [추가] 펀드가 끝나면 전체 종료

    # # 조건부 엣지
    # def decide_after_validation(state: GraphState):
    #     if state.get("validation_passed", False):
    #         return "save"
    #     else:
    #         return "retry"

    # workflow.add_conditional_edges(
    #     "validation_node",
    #     decide_after_validation,
    #     {
    #         "save": "save_and_summarize_node",
    #         "retry": "handle_error_node"
    #     }
    # )

    # MemorySaver를 포함하여 그래프 컴파일
    print("--- 그래프 컴파일 (MemorySaver 포함) ---")
    return workflow.compile(checkpointer=MemorySaver())