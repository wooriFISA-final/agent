# report_project/consume/execute.py

import pandas as pd
import pickle
import sys
from typing import Dict, Any, Optional
from state import AgentState, ModelAssets 
from .builder import build_consume_graph 

# ----------------------------------------------------------------------
# 1. 자산 로드 로직 (TEST MODE만 사용)
# ----------------------------------------------------------------------
def get_dummy_assets(user_id: Optional[int]) -> ModelAssets:
    """파일 로드 대신 더미 ModelAssets를 반환합니다."""
    print("🧪 [TEST MODE] Consume: 더미 자산 생성 중.")
    test_id = user_id if user_id else 500
    # 군집 예측 노드에서 필요한 최소한의 구조를 갖춘 더미 DataFrame
    return ModelAssets(
        knn_model=None, scaler=None, 
        df_profile=pd.DataFrame({'avg_age': [35]}, index=[1]), # 인덱스 1 사용 (오류 방지)
        df_data=pd.DataFrame({'user_id': [test_id], 'CAT2_A': [10.0], 'total_spend': [10.0], 'spend_month': ['2025-01']}, index=[0]),
        cat2_cols=['CAT2_A'], K_CLUSTERS=3
    )

def load_analysis_assets_real(knn_path: str, scaler_path: str, profile_path: str, data_path: str) -> ModelAssets:
    """Agent 구동에 필요한 모든 모델 파일과 데이터를 로드합니다. (실제 데이터 로드, 현재는 주석 처리됨)"""
    print("⚠️ 실제 파일 로드는 주석 처리되었으며, 더미 자산이 반환됩니다. (is_test=False 시)")
    return ModelAssets(
        knn_model=None, scaler=None, 
        df_profile=pd.DataFrame(), df_data=pd.DataFrame(),
        cat2_cols=[], K_CLUSTERS=3
    )

# ----------------------------------------------------------------------
# 2. 그래프 실행 함수
# ----------------------------------------------------------------------
def execute_consume_agent(initial_input: Dict[str, Any]) -> Dict[str, Any]:
    """Consume 에이전트를 실행하고 최종 보고서 결과를 반환합니다."""
    
    is_test = initial_input.get("is_test", True)
    user_id = initial_input.get("user_id")
    
    try:
        if is_test:
            assets = get_dummy_assets(user_id) # TEST MODE
        else:
            # 🚨 is_test=False 시 실제 로드 로직 (현재는 더미 반환)
            assets = load_analysis_assets_real(knn_path="consume/models/knn_model.pkl", scaler_path="consume/models/scaler.pkl", profile_path="consume/data/cluster_profile_k3.csv", data_path="consume/data/final_data_k3.csv")
            
        initial_state = AgentState(assets=assets, **initial_input)
        
        consume_graph = build_consume_graph()
        final_state = consume_graph.invoke(initial_state)

        return {
            "user_id": final_state['user_id'],
            "cluster_nickname": final_state['cluster_nickname'],
            "consumption_report": final_state['final_report'],
            "user_analysis": final_state['user_analysis']
        }

    except Exception as e:
        raise e