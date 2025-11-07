# report_project/compare/builder.py (LangGraph 정의 수정)

from langgraph.graph import StateGraph, END
from typing import Dict, Any, Literal
from state import AgentState 
from nodes.tool_nodes import (
    load_prev_month_report, load_house_info, load_policy_info, load_credit_info
)
from nodes.llm_nodes import compare_changes_node # compare_changes 노드

# 🚨 [신규 추가] 정책 로드 성공 여부를 판단하는 라우팅 함수
def check_policy_load_status(state: Dict[str, Any]) -> Literal["compare_changes", "fail_and_end"]:
    """정책 데이터가 비어 있는지 확인하여 LLM 분석 단계를 결정합니다."""
    
    policy_data = state.get("policy_info", {})
    old_chapters = policy_data.get("old_policy", [])
    new_chapters = policy_data.get("new_policy", [])
    
    # 두 리스트 중 하나라도 내용이 있다면 (정책 로드 성공) 비교를 진행합니다.
    if old_chapters and new_chapters:
        print("🧭 [Router] 정책 파일 로드 성공. LLM 비교 분석으로 이동합니다.")
        return "compare_changes"
    else:
        print("🧭 [Router] 정책 파일 로드 실패. 비교 분석을 건너뛰고 종료합니다.")
        # 실패 메시지를 State에 미리 저장하여 최종 아웃풋에 반영되도록 합니다.
        state["comparison_result"] = "정책 파일 로드 실패로 인해 정책 비교 분석을 수행할 수 없습니다."
        return "fail_and_end"


def build_compare_graph():
    """Compare 에이전트의 워크플로우 그래프를 LangGraph로 빌드합니다."""
    print("🛠️ Compare 에이전트 그래프 빌드 중...")
    
    workflow = StateGraph(AgentState)

    # 1. 노드 추가
    workflow.add_node("load_prev_month_report", load_prev_month_report)
    workflow.add_node("load_house_info", load_house_info)
    workflow.add_node("load_policy_info", load_policy_info)
    workflow.add_node("load_credit_info", load_credit_info)
    workflow.add_node("compare_changes", compare_changes_node) # LLM 노드

    # 2. 실행 순서 정의 (데이터 로드 병렬 및 순차)
    workflow.set_entry_point("load_prev_month_report")
    
    # 순차적 데이터 로드
    workflow.add_edge("load_prev_month_report", "load_house_info")
    workflow.add_edge("load_house_info", "load_policy_info")
    workflow.add_edge("load_policy_info", "load_credit_info")
    
    # 🚨 [수정] 정책 로드 후 성공 여부에 따라 분기 처리
    workflow.add_conditional_edges(
        "load_credit_info", # 모든 데이터 로드가 완료된 후 정책 상태 확인
        check_policy_load_status,
        {
            "compare_changes": "compare_changes",
            "fail_and_end": END 
        }
    )
    
    # LLM 분석 후 종료
    workflow.add_edge("compare_changes", END)

    app = workflow.compile()
    print("✅ Compare 에이전트 그래프 빌드 완료.")
    return app