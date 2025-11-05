from typing import TypedDict, Optional, Dict, Any
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. 모델과 데이터를 저장할 TypedDict 정의
class ModelAssets(TypedDict):
    knn_model: KNeighborsClassifier
    scaler: StandardScaler
    df_profile: pd.DataFrame
    df_data: pd.DataFrame
    cat2_cols: list[str]
    K_CLUSTERS: int
    
# 2. 분석 결과를 저장하고 노드 간에 전달할 상태 정의
class ConsumptionAnalysisState(TypedDict):
    """소비 리포트 에이전트의 상태를 정의합니다."""
    
    # 모델 자산 (model_builder에서 로드)
    assets: Optional[ModelAssets]
    
    # 입력 정보
    user_id: Optional[int]
    ollama_model_name: str
    
    # 분석 중간 및 최종 결과
    user_cluster: Optional[int]
    user_data: Optional[Dict[str, Any]]
    cluster_nickname: Optional[str]
    user_analysis: Optional[Dict[str, Any]]
    final_report: Optional[str]