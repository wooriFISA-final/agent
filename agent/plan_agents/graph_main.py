import json
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# -----------------------------------------------------------------
# 1. 🚀 우리 '노드' 파일들 임포트
# -----------------------------------------------------------------
from input_agent import PlanAgentNode 
import validation_agent as validator_agent 
from loan_agent_node import LoanAgentNode # ✅ [추가] 1. 대출 에이전트 임포트

print("--- 모든 에이전트 모듈 로드 완료 ---")

# -----------------------------------------------------------------
# 2. 📊 LangGraph State 정의 (그래프의 '기억')
# -----------------------------------------------------------------
class GraphState(TypedDict):
    """
    그래프 전체를 흐르는 상태 객체
    """
    messages: List[Dict[str, Any]]
    responses: Dict[str, Any]
    
    user_id: int  
    plan_id: Optional[int] # ✅ [추가] 2. DB 저장 후 'plan_id'를 담을 필드
    
    # --- 플래그 (Flags) ---
    input_completed: bool
    validation_passed: bool
    error_message: str
    
# -----------------------------------------------------------------
# 3. 🤖 노드(Node) 래퍼(Wrapper) 함수 정의
# -----------------------------------------------------------------

plan_agent = PlanAgentNode()
loan_agent = LoanAgentNode() # ✅ [추가] 3. 대출 에이전트 인스턴스 생성

# [노드 1: 상담원]
def input_node(state: GraphState) -> GraphState:
    print("\n--- [A. 입력 노드 시작] ---")
    return plan_agent.run_as_node(state)

# [노드 2: 심사관]
def validation_node(state: GraphState) -> GraphState:
    print("\n--- [B. 검증 노드 시작] ---")
    responses_to_validate = state.get("responses", {})

    if not responses_to_validate:
        print("⚠️ ValidationNode: 검증할 데이터가 없습니다.")
        state["validation_passed"] = False
        state["error_message"] = "수집된 데이터가 없습니다."
        return state

    final_result_dict = validator_agent.run_agent_executor(responses_to_validate)
    
    if final_result_dict.get("status") == "success":
        print("[Node: ValidationNode] ✅ 검증 통과.")
        state["validation_passed"] = True
        corrected_data = final_result_dict.get("data", {})
        state["responses"] = corrected_data
        print(f"보정/검증된 데이터: {corrected_data}")
            
    else:
        print(f"[Node: ValidationNode] ❌ 검증 실패.")
        state["validation_passed"] = False
        error_msg = final_result_dict.get("message", "알 수 없는 검증 오류가 발생했습니다.")
        state["error_message"] = error_msg 

    return state

# [노드 3: 에러 핸들러]
def handle_error_node(state: GraphState) -> GraphState:
    print("\n--- [C-1. 에러 처리 노드 시작] ---")
    error_msg = state.get("error_message", "알 수 없는 오류로 재시작합니다.")
    print(f"오류 발생: {error_msg}")
    
    state["messages"].append({"role": "assistant", "content": f"오류가 발생했습니다: {error_msg}\n정보를 다시 입력해주세요."})
    return state

# [노드 4: 저장 및 요약]
def save_and_summarize_node(state: GraphState) -> GraphState:
    """
    '검증 통과' 시 실행되는 노드.
    최종 데이터를 DB에 저장하고, 다음 노드를 위해 'plan_id'를 state에 저장합니다.
    """
    print("\n--- [C-2. 저장/요약 노드 시작] ---")
    final_data = state.get("responses", {})
    user_id_to_save = state.get("user_id")

    if not final_data:
        print("⚠️ 저장할 데이터가 없습니다.")
        return state
        
    if not user_id_to_save:
        print("⚠️ 'user_id'가 state에 없습니다. DB 저장을 스킵합니다.")
        return state

    try:
        # ✅ [수정] 4. save_to_db가 'plan_id'를 반환한다고 가정하고 값을 캡처
        # (중요: input_agent.py의 save_to_db가 'plan_id'를 return해야 합니다!)
        new_plan_id = plan_agent.save_to_db(final_data, user_id_to_save)
        
        if not new_plan_id or not isinstance(new_plan_id, int):
             raise Exception("'save_to_db'에서 유효한 'plan_id'를 반환하지 않았습니다.")

        print(f"✅ DB 저장 완료 (plan_id: {new_plan_id})")
        state["plan_id"] = new_plan_id # 👈 [수정] 5. state에 plan_id 저장
        
        plan_agent.summarize(final_data) # 요약 함수
        print("--- 저장/요약 작업 완료 ---")
        
    except Exception as e:
        print(f"❌ DB 저장 또는 요약 중 오류 발생: {e}")
        state["error_message"] = str(e)
        # 저장은 성공했으나 요약에 실패해도 대출 추천은 진행하도록 함
        # 만약 저장 자체를 실패하면 state["plan_id"]가 None이 되어 다음 노드에서 처리됨

    return state

# ✅ [추가] 6. 대출 추천 노드
def loan_recommend_node(state: GraphState) -> GraphState:
    """
    '저장/요약' 노드 이후 실행.
    저장된 'plan_id'를 이용해 'loan_agent'를 실행하고 결과를 state에 병합합니다.
    """
    print("\n--- [D. 대출 추천 노드 시작] ---")
    user_id = state.get("user_id")
    plan_id = state.get("plan_id") # 방금 저장된 plan_id

    if not user_id or not plan_id:
        print(f"⚠️ LoanNode: user_id({user_id}) 또는 plan_id({plan_id})가 없습니다. 스킵.")
        state["error_message"] = "플랜 ID가 없어 대출 추천을 스킵합니다."
        # 이 단계에서 오류가 나도 재시도(input)로 돌아갈 필요는 없으므로 END로 진행
        return state

    try:
        print(f"LoanAgent.run(user_id={user_id}, plan_id={plan_id}) 실행...")
        loan_result = loan_agent.run(user_id=user_id, plan_id=plan_id)
        
        if loan_result.get("message"): # loan_agent 내부에서 오류가 발생한 경우
             print(f"⚠️ 대출 추천 실패: {loan_result.get('message')}")
             state["error_message"] = loan_result.get('message')
        else:
            print(f"✅ 대출 추천 완료: {loan_result.get('loan_name')}")
            # 최종 결과를 'responses'에 병합하여 사용자에게 보여줄 수 있도록 함
            state["responses"].update(loan_result)
        
    except Exception as e:
        print(f"❌ 대출 추천 중 심각한 오류 발생: {e}")
        state["error_message"] = f"대출 추천 중 오류: {str(e)}"
    
    return state
    
# -----------------------------------------------------------------
# 4. 🔗 그래프 엣지(Edge) 조립
# -----------------------------------------------------------------
print("--- 그래프 조립 시작 ---")
workflow = StateGraph(GraphState)

workflow.add_node("input_node", input_node)
workflow.add_node("validation_node", validation_node)
workflow.add_node("handle_error_node", handle_error_node)
workflow.add_node("save_and_summarize_node", save_and_summarize_node)
workflow.add_node("loan_recommend_node", loan_recommend_node) # ✅ [추가] 7. 새 노드 등록

workflow.set_entry_point("input_node")

workflow.add_edge("input_node", "validation_node")
# workflow.add_edge("save_and_summarize_node", END) # ❌ [삭제]
workflow.add_edge("save_and_summarize_node", "loan_recommend_node") # ✅ [수정] 8. 엣지 변경
workflow.add_edge("loan_recommend_node", END) # ✅ [추가] 9. 엣지 추가 (대출 추천 후 종료)
workflow.add_edge("handle_error_node", "input_node")

def decide_after_validation(state: GraphState):
    if state.get("validation_passed", False):
        return "save"
    else:
        return "retry"

workflow.add_conditional_edges(
    "validation_node",
    decide_after_validation,
    {
        "save": "save_and_summarize_node",
        "retry": "handle_error_node"
    }
)

app = workflow.compile()
print("--- 그래프 컴파일 완료. 테스트 실행 ---")





# -----------------------------------------------------------------
# 5. ⚡️ 테스트 실행
# -----------------------------------------------------------------

# ✅ [수정] 10. 초기 state에 'user_id'와 'plan_id' (None) 추가
initial_state = {
    "messages": [], 
    "responses": {}, 
    "input_completed": False, 
    "validation_passed": False, 
    "error_message": "",
    "user_id": 1,  # 👈 (예: user_id 1번 사용자로 테스트)
    "plan_id": None # 👈 초기값은 None
}

# (참고: DB의 user_info 테이블에 user_id=1인 사용자가 미리 존재해야 합니다!)

print(f"[테스트 안내] user_id={initial_state['user_id']}로 테스트를 시작합니다.")
print("DB의 'user_info' 테이블에 해당 ID의 사용자가 있는지 확인하세요.")
print("또한 input_agent.py의 save_to_db가 'plan_id'를 반환하는지 확인하세요.")

for event in app.stream(initial_state, {"recursion_limit": 10}):
    if "__end__" in event:
        print("\n--- 그래프 실행 종료 ---")
        print("\n[최종 상태 (State)]")
        final_responses = event['__end__'].get('responses', {})
        print(json.dumps(final_responses, indent=2, ensure_ascii=False))