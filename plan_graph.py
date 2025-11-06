import operator
from typing import TypedDict, Annotated, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from pathlib import Path
import json

# --- 1. (중요) 표준화된 모든 '노드 클래스' 임포트 ---
# (님의 디렉토리 구조 'agent/plan_agents/' 기준)
from plan_agents.input_agent import InputAgentNode
from plan_agents.validation_agent import ValidationAgentNode # (수정되었다고 가정)
from plan_agents.loan_agent_node import LoanAgentNode       # (수정되었다고 가정)
from plan_agents.saving_agent_class_node import SavingAgentNode
from plan_agents.fund_agent_class_node import FundAgentNode
from plan_agents.plan_agent import PlanAgentNode            # (신규 생성 필요)

print("--- 모든 에이전트 노드 클래스 임포트 완료 ---")

# --- 2. '공용 메모리'가 될 통합 GraphState 정의 ---
class AgentGraphState(TypedDict):
    # (Input)
    user_id: int
    messages: Annotated[List[BaseMessage], operator.add] 
    
    # (파일 경로)
    fund_data_path: str
    savings_data_path: str
    
    # (Flags)
    input_completed: bool
    validation_passed: bool
    
    # (Data)
    plan_input_data: Dict[str, Any]
    user_mydata: Dict[str, Any]
    loan_recommendations: Dict[str, Any]
    savings_recommendations: Dict[str, Any]
    fund_analysis_result: Dict[str, Any]
    final_plan: Dict[str, Any]
    error_message: str

# --- 3. 그래프 생성 함수 (FastAPI가 호출할 함수) ---
def create_workflow():
    
    # 3-1. 모든 노드 클래스 인스턴스화
    input_node = InputAgentNode()
    validation_node = ValidationAgentNode()
    loan_node = LoanAgentNode()
    saving_node = SavingAgentNode()
    fund_node = FundAgentNode()
    plan_node = PlanAgentNode()

    # 3-2. 그래프 정의
    workflow = StateGraph(AgentGraphState)

    # 3-3. 노드 등록
    workflow.add_node("input", input_node.run)
    workflow.add_node("validate", validation_node.run)
    workflow.add_node("loan", loan_node.run)
    workflow.add_node("saving", saving_node.run)
    workflow.add_node("fund", fund_node.run)
    workflow.add_node("plan", plan_node.run)
    
    # (필요시 'UserData 수집' 노드 추가)
    # workflow.add_node("get_mydata", get_mydata_node) 

    # 3-4. 엣지(Edge) 연결
    
    # 1. 시작점
    workflow.set_entry_point("input")

    # 2. Input 노드 이후 분기
    def decide_after_input(state: AgentGraphState):
        if state.get("input_completed", False):
            return "go_to_validation" # 정보 수집 완료 -> 검증
        else:
            return "end_turn" # 정보 부족 -> 턴 종료 (사용자 응답 대기)

    workflow.add_conditional_edges(
        "input",
        decide_after_input,
        {
            "go_to_validation": "validate",
            "end_turn": END # ⬅️ 그래프 종료 (API가 AI의 추가 질문을 반환)
        }
    )
    
    # 3. Validation 노드 이후 분기
    def decide_after_validation(state: AgentGraphState):
        if state.get("validation_passed", False):
            return "go_to_loan" # 검증 통과 -> 대출
        else:
            return "end_turn_error" # 검증 실패 -> 턴 종료

    workflow.add_conditional_edges(
        "validate",
        decide_after_validation,
        {
            "go_to_loan": "loan",
            "end_turn_error": END # ⬅️ 그래프 종료 (API가 검증 실패 메시지 반환)
        }
    )

    # 4. (병렬) Loan 노드 이후 Fork
    workflow.add_edge("loan", "saving")
    workflow.add_edge("loan", "fund")

    # 5. (결합) Saving, Fund 노드 이후 Join
    workflow.add_edge(["saving", "fund"], "plan")

    # 6. Plan 노드 이후 종료
    workflow.add_edge("plan", END)

    # 3-5. 그래프 컴파일
    print("--- 🏁 LangGraph 워크플로우 컴파일 완료 🏁 ---")
    return workflow.compile()

# --- 4. (테스트) 이 파일을 VS Code에서 직접 실행할 때 ---
if __name__ == "__main__":
    
    app = create_workflow()

    # (경로 설정)
    # 이 파일은 agent/plan_graph.py에 있음
    current_script_path = Path(__file__).resolve()
    # agent -> FINAL_PROJECT
    project_root = current_script_path.parent.parent
    
    fund_path = str(project_root / "fund_data.json")
    saving_path = str(project_root / "saving_data.csv")

    # (테스트 1: 정보 부족)
    print("\n--- 🏁 테스트 1: 정보 부족 🏁 ---")
    initial_state_1 = {
        "user_id": 1,
        "messages": [HumanMessage(content="10억짜리 아파트 사고 싶어요")],
        "fund_data_path": fund_path,
        "savings_data_path": saving_path
    }
    final_state_1 = app.invoke(initial_state_1)
    print(f"AI 응답: {final_state_1['messages'][-1].content}")

    # (테스트 2: 정보 완료)
    print("\n--- 🏁 테스트 2: 정보 수집 완료 🏁 ---")
    messages_2 = [
        HumanMessage(content="10억짜리 아파트 사고 싶어요"),
        AIMessage(content="... 추가 질문 ..."),
        HumanMessage(content="자산은 2억이고 월급의 50%를 쓸게요. 위치는 서울 송파구, 아파트입니다.")
    ]
    initial_state_2 = {
        "user_id": 1,
        "messages": messages_2,
        "fund_data_path": fund_path,
        "savings_data_path": saving_path
    }
    final_state_2 = app.invoke(initial_state_2)
    print("\n[최종 플랜]")
    print(json.dumps(final_state_2.get('final_plan'), indent=2, ensure_ascii=False))