from langgraph.graph import StateGraph, END
from report.compare_agent.state import AgentState
from report.compare_agent.nodes.tool_nodes import (
    load_prev_month_report, load_house_info, load_policy_info, load_credit_info)
from report.compare_agent.nodes.llm_nodes import compare_changes


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("load_prev_month_report", load_prev_month_report)
    workflow.add_node("load_house_info", load_house_info)
    workflow.add_node("load_policy_info", load_policy_info)
    workflow.add_node("load_credit_info", load_credit_info)
    workflow.add_node("compare_changes", compare_changes)

    # 실행 순서 정의
    workflow.set_entry_point("load_prev_month_report")
    workflow.add_edge("load_prev_month_report", "load_house_info")
    workflow.add_edge("load_house_info", "load_policy_info")
    workflow.add_edge("load_policy_info", "load_credit_info")
    workflow.add_edge("load_credit_info", "compare_changes")
    workflow.add_edge("compare_changes", END)

    return workflow.compile()
