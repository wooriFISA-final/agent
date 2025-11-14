# report_project/report/profit/builder.py

from langgraph.graph import StateGraph, END
from typing import Dict, Any, Literal
import pandas as pd

from report.state import AgentState 
from report.nodes.tool_nodes import ( # 🚨 [수정]
    aggregate_financial_data_node as aggregate_data_processor, 
    load_data # load_data 함수를 직접 임포트
)
from report.nodes.llm_nodes import ( # 🚨 [수정]
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
            "savings": [{"monthly_payment": 1000000, "interest_rate": 0.05, "tax_rate": 0.154, "total_period_months": 12, "product_id": "S001"}],
            "funds": [{"purchase_nav": 1000, "current_nav": 1100, "total_shares": 10000, "fee_rate": 0.01, "product_id": "F001", "report_date": "2025-11-01"}],
        }
        return {"raw_data": raw_data}

    try:
        raw_data = load_data() # tool_nodes.py에 정의된 load_data 함수 호출
        return {"raw_data": raw_data}
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return {"raw_data": {}}


def calculate_data_node(state: AgentState) -> dict:
    """Node: 금융 계산 및 집계 (tool_nodes.aggregate_financial_data_node 호출)"""
    print("🛠️ 2. Node: 수익/손실 계산 및 금융 데이터 집계...")
    
    temp_state = state.copy()
    # tool_nodes.py의 aggregate_financial_data_node 함수 호출
    updated_state = aggregate_data_processor(temp_state) 

    principal = updated_state.get('total_principal', 0.0)
    net_pl = updated_state.get('total_net_profit_loss', 0.0)
    
    if principal > 0:
        print(f"--- [계산 결과] 총 수익률: {net_pl / principal * 100:.2f}% ---")

    # LangGraph가 합칠 수 있도록 모든 필드를 딕셔너리로 반환
    return {
        "analysis_df": updated_state.get('analysis_df', pd.DataFrame()), 
        "total_principal": principal,
        "total_net_profit_loss": net_pl,
        "raw_data": updated_state.get('raw_data', {})
    }


def generate_vis_node(state: AgentState) -> dict:
    """Node: 시각화 데이터 생성"""
    print("📊 3. Node: 시각화 데이터 생성...")
    
    if state.get('analysis_df') is None or state['analysis_df'].empty:
         chart_data = {}
         image_tag = "No data to visualize."
    else:
        chart_data, image_tag = vis_data_generator(state['analysis_df']) 
    print(image_tag)
    return {"chart_data": chart_data}


def analyze_llm_node(state: AgentState) -> dict:
    """Node: LLM 분석 보고서 작성"""
    print("🧠 4. Node: LLM 기반 투자 결과 분석 보고서 작성...")
    
    # LLM 노드를 호출하고 결과를 받습니다.
    result_state = analysis_report_generator(state.copy())
    
    report = result_state.get('investment_analysis_result', "분석 실패")
    
    print("\n✅ 5. 최종 보고서 출력\n")
    print(report)
    
    return {"investment_analysis_result": report}


# --- 2. LangGraph 워크플로우 빌드 함수 ---

def route_to_analysis(state: Dict[str, Any]) -> Literal["analyze_llm", "stop"]:
    """
    총 원금이 0 초과인지 확인하여 다음 단계를 결정하는 라우터
    """
    total_principal = state.get("total_principal", 0.0)
    
    if total_principal > 0:
        print("🧭 [Router] 총 원금 확인. 투자 분석 LLM으로 이동합니다.")
        return "analyze_llm"
    else:
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
    workflow.add_edge("load_data", "calculate_data")
    workflow.add_edge("calculate_data", "generate_vis")

    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        "generate_vis", 
        route_to_analysis,
        {
            "analyze_llm": "analyze_llm", 
            "stop": END
        }
    )
    
    # LLM 분석 후 종료
    workflow.add_edge("analyze_llm", END)

    app = workflow.compile()
    print("✅ Profit 에이전트 그래프 빌드 완료.")
    return app