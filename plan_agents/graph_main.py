from langgraph.graph import StateGraph, END
from input_loan_agent.input_agent_node import plan_input_node
from input_loan_agent.validation_agent_node import validation_node
from typing import TypedDict

# 1️⃣ 상태 스키마 정의
class PlanState(TypedDict):
    plan_completed: bool
    validation_passed: bool

# 2️⃣ 그래프 초기화
graph = StateGraph(PlanState)

# 3️⃣ 노드 추가
graph.add_node("plan_input", plan_input_node)
graph.add_node("validation", validation_node)

# 4️⃣ 노드 연결
graph.set_entry_point("plan_input")
graph.add_edge("plan_input", "validation")
graph.add_edge("validation", END)

# 5️⃣ 그래프 실행
if __name__ == "__main__":
    print("LangGraph 기반 Plan-Validation 플로우 시작\n")
    app = graph.compile()
    final_state = app.invoke({})
    print("\n그래프 실행 완료 — 최종 상태:")
    print(final_state)
