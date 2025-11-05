# report_project/compare/builder.py

from langgraph.graph import StateGraph, END
# 최상위 state.py에서 AgentState를 import 합니다.
from state import AgentState 
# 공통 nodes 디렉토리에서 노드 함수들을 import 합니다.
from nodes.tool_nodes import (
    load_prev_month_report, load_house_info, load_policy_info, load_credit_info
)
# compare_changes 함수가 llm_nodes.py에 정의되어 있다고 가정하고 import 합니다.
from nodes.llm_nodes import compare_changes_node as compare_changes 
# ⚠️ llm_nodes.py에서 함수 이름을 compare_changes_node로 통합했으므로 이름을 맞춰줍니다.


def build_compare_graph():
    """
    Compare 에이전트의 워크플로우 그래프를 LangGraph로 빌드합니다.
    """
    print("🛠️ Compare 에이전트 그래프 빌드 중...")
    
    # LangGraph의 StateGraph에 AgentState를 사용합니다.
    workflow = StateGraph(AgentState)

    # 1. 노드 추가 (기존 코드와 동일)
    workflow.add_node("load_prev_month_report", load_prev_month_report)
    workflow.add_node("load_house_info", load_house_info)
    workflow.add_node("load_policy_info", load_policy_info)
    workflow.add_node("load_credit_info", load_credit_info)
    workflow.add_node("compare_changes", compare_changes)

    # 2. 실행 순서 정의 (기존 코드와 동일)
    workflow.set_entry_point("load_prev_month_report")
    workflow.add_edge("load_prev_month_report", "load_house_info")
    workflow.add_edge("load_house_info", "load_policy_info")
    workflow.add_edge("load_policy_info", "load_credit_info")
    workflow.add_edge("load_credit_info", "compare_changes")
    workflow.add_edge("compare_changes", END)

    app = workflow.compile()
    print("✅ Compare 에이전트 그래프 빌드 완료.")
    return app