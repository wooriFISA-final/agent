<<<<<<< HEAD
import json
import logging
from typing import Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage

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
# 2️⃣ 로깅 설정
# ----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
print("--- ✅ 에이전트 모듈 로드 완료 ---")

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
    """검증 결과를 그래프 상태에 반영"""
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
# 7️⃣ 조건부 라우터 정의
# ----------------------------------
def route_after_input(state: GraphState):
    """입력 완료 여부 확인"""
    return "validate_input" if state.get("input_completed", False) else END

def route_after_validation(state: GraphState):
    """검증 결과에 따른 분기"""
    status = state.get("final_response", {}).get("status", "error")
    return "update_state_after_validation" if status == "success" else "handle_error"

# ----------------------------------
# 8️⃣ 그래프 생성 함수
# ----------------------------------
def create_graph():
    workflow = StateGraph(GraphState)

    # ---------------- 노드 등록 ----------------
    workflow.add_node("extract_info", plan_input_agent.run)
    workflow.add_node("validate_input", validator_agent.run)
    workflow.add_node("update_state_after_validation", update_state_after_validation)
    workflow.add_node("loan_recommend", loan_agent.run)
    workflow.add_node("saving_recommend", saving_agent.run)
    workflow.add_node("fund_recommend", fund_agent.run)
    workflow.add_node("summary_node", summary_agent.run)
    workflow.add_node("handle_error", handle_error_node)

    # ---------------- 진입점 설정 ----------------
    workflow.set_entry_point("extract_info")

    # ---------------- 연결 설정 ----------------
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
=======
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
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
