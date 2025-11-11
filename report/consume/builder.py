# report_project/consume/builder.py

from langgraph.graph import StateGraph, END
# 최상위 state.py에서 AgentState를 import 합니다.
from state import AgentState 
from nodes.tool_nodes import (
    get_user_cluster_node, generate_cluster_nickname_node, analyze_user_spending_node
)
from nodes.llm_nodes import generate_final_report_node


# ⚠️ 시각화 함수는 그래프 로직 외부(execute.py 또는 main_orchestrator.py)에서 호출하는 것이 권장되므로, 
# builder.py에서는 제외합니다.

def build_consume_graph():
    """
    Consume 에이전트의 워크플로우 그래프를 LangGraph로 빌드합니다.
    """
    print("🛠️ Consume 에이전트 그래프 빌드 중...")

    # 1. 랭그래프 정의
    # 통합된 AgentState를 사용합니다.
    graph_builder = StateGraph(AgentState)

    # 2. 노드 추가 (Nodes)
    graph_builder.add_node("predict_cluster", get_user_cluster_node)
    graph_builder.add_node("generate_nickname", generate_cluster_nickname_node)
    graph_builder.add_node("analyze_spending", analyze_user_spending_node)
    graph_builder.add_node("generate_report", generate_final_report_node)

    # 3. 노드 연결 (Edges) - 순차적 워크플로우
    graph_builder.set_entry_point("predict_cluster")
    
    graph_builder.add_edge("predict_cluster", "generate_nickname")
    graph_builder.add_edge("generate_nickname", "analyze_spending")
    graph_builder.add_edge("analyze_spending", "generate_report")
    
    graph_builder.add_edge("generate_report", END)

    # 4. 그래프 컴파일
    app = graph_builder.compile()

    print("✅ Consume 에이전트 그래프 빌드 완료.")
    return app