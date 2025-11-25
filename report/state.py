# report_project/state.py

from typing import TypedDict, Any, List, Optional, Dict
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 

# 1. ⚙️ Consume Agent의 모델 자산 상태 (ModelAssets) 정의
class ModelAssets(TypedDict):
    knn_model: Optional[KNeighborsClassifier]
    scaler: Optional[StandardScaler]
    df_profile: pd.DataFrame
    df_data: pd.DataFrame
    cat2_cols: List[str]
    K_CLUSTERS: int

# 2. 🚀 LangGraph의 공통 상태 (AgentState) 정의
class AgentState(TypedDict):
    """LangGraph 워크플로우를 위한 공통 상태 정의입니다."""
    
    # ------------------ 공통/입력 필드 ------------------
    member_id: Optional[int]
    user_id: Optional[int]
    is_test: bool
    ollama_model_name: Optional[str]

    # ------------------ Compare 에이전트 필드 ------------------
    report_data: Optional[Any]
    house_info: Optional[Any]
    policy_info: Optional[Any]
    credit_info: Optional[Any]
    comparison_result: Optional[str]

    # ------------------ Consume 에이전트 필드 ------------------
    assets: Optional[ModelAssets]
    user_cluster: Optional[int]
    user_data: Optional[Dict[str, Any]]
    cluster_nickname: Optional[str]
    user_analysis: Optional[Dict[str, Any]]
    final_report: Optional[str]

    # ------------------ Profit 에이전트 필드 ------------------
    raw_data: Optional[Dict[str, Any]]      
    analysis_df: Optional[pd.DataFrame]
    total_principal: Optional[float]
    total_net_profit_loss: Optional[float]
    chart_data: Optional[Dict[str, Any]]
    investment_analysis_result: Optional[str]