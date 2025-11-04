# agents/main_agent.py

from langgraph.graph import StateGraph, END
# .state는 같은 폴더 (profit_loss_report) 내 state.py를 가리킵니다.
from ..state import AgentState
from ..nodes.tool_nodes import load_data, aggregate_financial_data
from ..nodes.llm_nodes import generate_visualization_data, analyze_investment_results

import pandas as pd
import json

# --- 1. LangGraph Node 함수 정의 (LangGraph 형식에 맞춰 입력과 출력을 통일) ---

def load_data_node(state: AgentState) -> dict:
    """Node: 데이터 로드 (tool_nodes)"""
    print("🚀 1. Node: 투자 상품 데이터 로드...")
    raw_data = load_data() 
    return {"raw_data": raw_data} 

def calculate_data_node(state: AgentState) -> dict:
    """Node: 금융 계산 및 집계 (tool_nodes)"""
    print("🛠️ 2. Node: 수익/손실 계산 및 금융 데이터 집계...")
    raw_data = state['raw_data']
    df, principal, net_pl = aggregate_financial_data(raw_data)
    print(f"--- [계산 결과] 총 수익률: {net_pl / principal * 100:.2f}% ---")
    return {
        "df_results": df,
        "total_principal": principal,
        "total_net_profit_loss": net_pl,
    }

def generate_vis_node(state: AgentState) -> dict:
    """Node: 시각화 데이터 생성 (llm_nodes)"""
    print("📊 3. Node: 시각화 데이터 생성...")
    chart_data, image_tag = generate_visualization_data(state['df_results'])
    print(image_tag)
    return {"chart_data": chart_data}

def analyze_llm_node(state: AgentState) -> dict:
    """Node: LLM 분석 보고서 작성 (llm_nodes)"""
    print("🧠 4. Node: LLM 기반 투자 결과 분석 보고서 작성...")
    report = analyze_investment_results(
        state['df_results'],
        state['total_principal'],
        state['total_net_profit_loss'],
        state['chart_data']
    )
    print("\n✅ 5. 최종 보고서 출력\n")
    print(report)
    return {"llm_report": report}


# --- 2. LangGraph 워크플로우 빌드 및 실행 ---

def build_graph():
    """LangGraph 워크플로우를 정의하고 컴파일합니다."""
    workflow = StateGraph(AgentState)

    workflow.add_node("load_data", load_data_node)
    workflow.add_node("calculate_data", calculate_data_node)
    workflow.add_node("generate_vis", generate_vis_node)
    workflow.add_node("analyze_llm", analyze_llm_node)

    # 순차 실행 정의
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "calculate_data")
    workflow.add_edge("calculate_data", "generate_vis")
    workflow.add_edge("generate_vis", "analyze_llm")
    workflow.add_edge("analyze_llm", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    
    # 초기 상태 (pandas DataFrame 초기화를 위해 필요)
    initial_state = {
        "raw_data": {},
        "df_results": pd.DataFrame(),
        "total_principal": 0.0,
        "total_net_profit_loss": 0.0,
        "chart_data": {},
        "llm_report": "",
    }
    
    print("\n--- LangGraph 실행 시작 ---\n")
    app.invoke(initial_state)
    print("\n--- LangGraph 실행 완료 ---")