# plan_input_node.py
from input_loan_agent.input_agent import PlanAgentNode

agent = PlanAgentNode()

def plan_input_node(state: dict):
    """
    LangGraph 내에서 실행 가능한 Node 함수
    state: LangGraph의 상태 딕셔너리
    """
    print("\n[PlanInputNode 시작]")
    agent.run()  # PlanAgentNode의 run() 실행 (입력 → 검증 → DB 저장 포함)
    state["plan_completed"] = True
    return state