import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from langgraph.graph import StateGraph, END # 랭그래프 핵심 임포트

# 모듈 임포트
from .state import ConsumptionAnalysisState
from .builders.model_builder import load_assets
from .nodes.tool_nodes import get_user_cluster_node, generate_cluster_nickname_node, analyze_user_spending_node
from .nodes.llm_nodes import generate_final_report_node

# (시각화 함수는 로직 변경이 없어 여기에 그대로 유지)
def plot_user_cluster(state: ConsumptionAnalysisState):
    # ... (기존 plot_user_cluster 코드 그대로 복사하여 여기에 붙여넣기)
    # ... (생략)
    pass
# ----------------------------------------------------------------------


if __name__ == "__main__":
    
    # 🌟 경로 설정 (기존과 동일)
    FINAL_DATA_PATH = 'report/comsume_report/data/final_data_k3.csv'
    CLUSTER_PROFILE_PATH = 'report/comsume_report/data/cluster_profile_k3.csv'
    SCALER_MODEL_PATH = 'report/comsume_report/models/scaler.pkl'
    KNN_MODEL_PATH = 'report/comsume_report/models/knn_model.pkl'
    AGENT_OLLAMA_MODEL = "qwen3:8b"
    
    # 1. 모델 자산 로드 (그래프 실행 전 준비)
    assets = load_assets(KNN_MODEL_PATH, SCALER_MODEL_PATH, CLUSTER_PROFILE_PATH, FINAL_DATA_PATH)
    
    # 2. 상태 초기화 및 사용자 ID 설정
    initial_state = ConsumptionAnalysisState(
        assets=assets,
        user_id=None,
        ollama_model_name=AGENT_OLLAMA_MODEL,
        # ... (나머지 필드 None 초기화)
        user_cluster=None, user_data=None, cluster_nickname=None, user_analysis=None, final_report=None
    )

    if not assets['df_data'].empty:
        initial_state['user_id'] = assets['df_data']['user_id'].iloc[500] 
    else:
        print("❌ 오류: 데이터가 없어 분석을 시작할 수 없습니다.")
        sys.exit(1)

    print(f"\n--- 🔎 사용자 ID: {initial_state['user_id']} LangGraph 정의 및 실행 ---")

    # 3. 랭그래프 정의
    graph_builder = StateGraph(ConsumptionAnalysisState)

    # 4. 노드 추가 (Nodes)
    # 각 노드는 tools와 llm_nodes에서 정의된 함수입니다.
    graph_builder.add_node("predict_cluster", get_user_cluster_node)
    graph_builder.add_node("generate_nickname", generate_cluster_nickname_node)
    graph_builder.add_node("analyze_spending", analyze_user_spending_node)
    graph_builder.add_node("generate_report", generate_final_report_node)

    # 5. 노드 연결 (Edges) - 순차적 워크플로우
    # 시작점 설정
    graph_builder.set_entry_point("predict_cluster")
    
    # 순서대로 노드 연결
    graph_builder.add_edge("predict_cluster", "generate_nickname")
    graph_builder.add_edge("generate_nickname", "analyze_spending")
    graph_builder.add_edge("analyze_spending", "generate_report")
    
    # 최종 보고서 생성 후 종료
    graph_builder.add_edge("generate_report", END)

    # 6. 그래프 컴파일
    app = graph_builder.compile()

    # 7. 그래프 실행
    try:
        # 상태를 시작점으로 전달하고 실행합니다.
        final_state = app.invoke(initial_state)

        # 8. 최종 결과 출력 및 시각화 (그래프 외부)
        plot_user_cluster(final_state) # 시각화는 그래프 로직 외부에서 수행

        print("\n" + "="*70)
        print(f"### 🏆 최종 AI Agent 보고서 (Ollama {final_state['ollama_model_name']}) 🏆 ###")
        print("-" * 70)
        print("📌 군집 ID:", final_state['user_cluster'])
        print("📌 군집 별명:", final_state['cluster_nickname'])
        print("📌 소비 TOP 3:", ", ".join(final_state['user_analysis']['top_3_categories']))
        print("-" * 70)
        print("[LLM 생성 보고서]")
        print(final_state['final_report'])
        print("="*70)

    except Exception as e:
        print(f"\n❌ 랭그래프 파이프라인 실행 중 치명적인 오류 발생: {e}")
        # ... (Ollama 오류 안내 등 기존 로직 유지)