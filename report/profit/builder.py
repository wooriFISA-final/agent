# report_project/profit/builder.py

from langgraph.graph import StateGraph, END
from typing import Dict, Any, Literal
import pandas as pd

from state import AgentState 
from nodes.tool_nodes import (
    aggregate_financial_data_node as aggregate_data_processor, 
    # load_data는 위에서 이미 임포트했으므로 제거
)
from nodes.llm_nodes import (
    generate_visualization_data as vis_data_generator, 
    analyze_investment_results_node as analysis_report_generator
) 

# --- 1. LangGraph Node 함수 정의 (Wrapper Nodes) ---

def load_data_node(state: AgentState) -> dict:
    """Node: 데이터 로드 (is_test 지원)"""
    print("🚀 1. Node: 투자 상품 데이터 로드...")
    
    if state.get("is_test"):
        print("🧪 [TEST MODE] Profit: 더미 금융 데이터를 불러옵니다.")
        raw_data = {
            "report_date": "2025-11-01",
            "deposits": [{"principal": 5000000, "interest_rate": 0.03, "tax_rate": 0.154, "total_period_months": 12, "product_id": "D001"}],
            "savings": [],
            "funds": [],
        }
        return {"raw_data": raw_data}

    try:
        raw_data = load_data() # tool_nodes.load_data 호출 (파일 로드 로직)
        return {"raw_data": raw_data}
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return {"raw_data": {}}


def calculate_data_node(state: AgentState) -> dict:
    """Node: 금융 계산 및 집계 (tool_nodes.aggregate_financial_data_node 호출)"""
    print("🛠️ 2. Node: 수익/손실 계산 및 금융 데이터 집계...")
    
    # 🚨 [수정] aggregate_data_processor는 state를 받아 state를 반환합니다.
    # 따라서, 튜플 언패킹 대신 상태를 직접 업데이트하고 반환 키를 설정합니다.
    
    # temp_state를 만들어 tool_nodes의 노드를 호출하여 상태를 업데이트합니다.
    temp_state = state.copy()
    updated_state = aggregate_data_processor(temp_state) 

    # 필요한 필드를 반환 딕셔너리에 담습니다.
    return {
        "analysis_df": updated_state.get('analysis_df', pd.DataFrame()), 
        "total_principal": updated_state.get('total_principal', 0.0),
        "total_net_profit_loss": updated_state.get('total_net_profit_loss', 0.0),
        # raw_data도 다음 노드에서 필요할 수 있으므로 반환합니다.
        "raw_data": updated_state.get('raw_data', {})
    }


def generate_vis_node(state: AgentState) -> dict:
    """Node: 시각화 데이터 생성"""
    print("📊 3. Node: 시각화 데이터 생성...")
    
    if state['analysis_df'].empty:
         chart_data = {}
         image_tag = "No data to visualize."
    else:
        chart_data, image_tag = vis_data_generator(state['analysis_df']) 
    print(image_tag)
    return {"chart_data": chart_data}


def analyze_llm_node(state: AgentState) -> dict:
    """Node: LLM 분석 보고서 작성"""
    print("🧠 4. Node: LLM 기반 투자 결과 분석 보고서 작성...")
    
    # LLM 노드를 호출하고 결과를 받습니다. (state를 업데이트하는 방식)
    result_state = analysis_report_generator(state.copy())
    
    report = result_state.get('investment_analysis_result', "분석 실패")
    
    print("\n✅ 5. 최종 보고서 출력\n")
    print(report)
    
    return {"investment_analysis_result": report}


# --- 2. LangGraph 워크플로우 빌드 함수 (오류 해결 핵심) ---

# 🚨 [수정] 오류 해결 핵심 함수: 라우터는 오직 하나의 문자열만 반환해야 합니다.
def route_to_analysis(state: Dict[str, Any]) -> Literal["analyze_llm", "stop"]:
    """
    총 원금이 0 초과인지 확인하여 다음 단계를 결정합니다. 
    (오직 하나의 문자열만 반환해야 합니다!)
    """
    total_principal = state.get("total_principal", 0.0)
    
    if total_principal > 0:
        print("🧭 [Router] 총 원금 확인. 투자 분석 LLM으로 이동합니다.")
        return "analyze_llm"
    else:
        # 이 경우 LLM 분석 없이 종료합니다.
        print("🧭 [Router] 총 원금 0 또는 데이터 오류. LLM 분석을 건너뛰고 종료합니다.")
        return "stop"


def build_profit_graph():
    """LangGraph 워크플로우를 정의하고 컴파일합니다."""
    print("🛠️ Profit 에이전트 그래프 빌드 중...")
    
    workflow = StateGraph(AgentState)

    # 1. 노드 추가 
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("calculate_data", calculate_data_node)
    workflow.add_node("generate_vis", generate_vis_node)
    workflow.add_node("analyze_llm", analyze_llm_node)

    # 2. 시작점 및 엣지 설정
    workflow.set_entry_point("load_data")

    # 데이터 로드 후 계산 노드로 이동
    workflow.add_edge("load_data", "calculate_data")
    
    # 계산 후 시각화 데이터 생성 노드로 이동
    workflow.add_edge("calculate_data", "generate_vis")

    # 🚨 [오류 해결] 시각화 데이터 생성 후 LLM 분석 진행 여부를 결정하는 조건부 엣지 설정
    workflow.add_conditional_edges(
        "generate_vis", 
        route_to_analysis,
        {
            # route_to_analysis가 반환하는 문자열과 일치해야 합니다.
            "analyze_llm": "analyze_llm", 
            "stop": END
        }
    )
    
    # LLM 분석 후 종료
    workflow.add_edge("analyze_llm", END)

    app = workflow.compile()
    print("✅ Profit 에이전트 그래프 빌드 완료.")
    return app