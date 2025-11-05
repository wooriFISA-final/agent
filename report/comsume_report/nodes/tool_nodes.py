import pandas as pd
from typing import Dict, Any, Tuple
from ..state import ConsumptionAnalysisState

# Task 2/2-2: 군집 예측 함수
def get_user_cluster_node(state: ConsumptionAnalysisState) -> ConsumptionAnalysisState:
    """사용자 ID를 기반으로 군집을 예측하고 데이터를 추출합니다."""
    assets = state['assets']
    user_id = state['user_id']
    
    user_data_row = assets['df_data'][assets['df_data']['user_id'] == user_id] \
                        .sort_values(by='spend_month', ascending=False).iloc[0]
    
    user_features = user_data_row[assets['cat2_cols']].values.reshape(1, -1)
    user_scaled = assets['scaler'].transform(user_features)
    user_cluster = assets['knn_model'].predict(user_scaled)[0]
    
    state['user_cluster'] = int(user_cluster)
    state['user_data'] = user_data_row.to_dict()
    print(f"✅ Tool Node: 군집 예측 완료 (Cluster: {user_cluster})")
    return state

# Task 4: 군집 별명 생성 함수
def generate_cluster_nickname_node(state: ConsumptionAnalysisState) -> ConsumptionAnalysisState:
    """군집 ID를 기반으로 별명을 생성합니다."""
    assets = state['assets']
    cluster_id = state['user_cluster']
    df_profile = assets['df_profile']

    profile = df_profile.loc[cluster_id]
    cat2_profile = profile.filter(like='CAT2_')
    top3_cats = cat2_profile.sort_values(ascending=False).head(3).index.str.replace('CAT2_', '').tolist()
    avg_age = int(profile.get('avg_age', 35))
    age_str = "중장년층 중심의" if avg_age > 45 else ("청년층 중심의" if avg_age < 30 else "핵심 소비 세대의")
    
    nickname = (f"**[ {age_str} {top3_cats[0]} 및 {top3_cats[1]} 집중형 그룹 ]** "
                f"평균 나이 {avg_age}세")
    
    state['cluster_nickname'] = nickname
    print("✅ Tool Node: 군집 별명 생성 완료")
    return state

# Task 5: 개인 소비 분석 함수
def analyze_user_spending_node(state: ConsumptionAnalysisState) -> ConsumptionAnalysisState:
    """개인 데이터를 기반으로 소비액과 항목 등을 분석합니다."""
    assets = state['assets']
    user_data = state['user_data']
    
    user_spending = pd.Series({k: v for k, v in user_data.items() if k in assets['cat2_cols']}).sort_values(ascending=False)
    top3_cats_str = [f"{c.replace('CAT2_', '')} ({v:.1f}만원)" for c, v in user_spending.head(3).items()]
    
    fixed_cost_cats = ['공과금/통신', '보험/금융']
    fixed_cols = [f'CAT2_{c}' for c in fixed_cost_cats if f'CAT2_{c}' in user_data]
    fixed_cost = sum(user_data.get(c, 0) for c in fixed_cols)
    total_spend = user_data.get('total_spend', 1)
    non_fixed_cost_rate = f"{((total_spend - fixed_cost) / total_spend) * 100:.1f}%" if total_spend > 0 else "0.0%"
    
    analysis_data = {
        'total_spend_amount': f"{total_spend:.1f}만원", 
        'top_3_categories': top3_cats_str, 
        'fixed_cost': f"{fixed_cost:.1f}만원", 
        'non_fixed_cost_rate': non_fixed_cost_rate
    }
    
    state['user_analysis'] = analysis_data
    print("✅ Tool Node: 개인 소비 분석 완료")
    return state