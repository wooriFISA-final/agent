# validation_node.py
from input_loan_agent.validation_agent import ValidationAgent

validator = ValidationAgent()

def validation_node(state: dict):
    """
    LangGraph 내에서 실행 가능한 Validation Node
    state: LangGraph의 상태 딕셔너리
    """
    print("\n[ValidationNode 시작]")
    responses = state.get("responses", {})

    if not responses:
        print("⚠️ 검증할 데이터가 없습니다.")
        state["validation_passed"] = False
        return state

    result = validator.run(responses)
    state["validation_passed"] = result
    return state
