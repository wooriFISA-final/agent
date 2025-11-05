# report_project/nodes/tool_nodes.py

import pandas as pd
import json
import os
from typing import Dict, Any, Tuple

# ----------------------------------------------------------------------
# mcp_nodes.py에서 query_mysql을 import합니다.
# ----------------------------------------------------------------------
try:
    from .mcp_nodes import query_mysql
except ImportError:
    def query_mysql(state: Dict[str, Any], query: str, params=None, key: str = "db_result") -> Dict[str, Any]:
        print(f"🔗 [Tool Node] ERROR: mcp_nodes.py의 query_mysql을 찾을 수 없습니다. 더미 실행.")
        state[key] = None
        return state
# ----------------------------------------------------------------------


# ==============================================================================
# 1. 🔍 compare 에이전트용: 데이터 로드 및 검색 노드 (is_test 지원)
# ==============================================================================
def load_prev_month_report(state: Dict[str, Any]) -> Dict[str, Any]:
    print("🗂️ [Tool Node] 이전 달 레포트 데이터 가져오기...")
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 리포트 데이터를 불러옵니다.")
        state["report_data"] = {
            "month": "2025-10", "income": 5000000, "loan_balance": 20000000, 
            "credit_score": 800, "target_location": "서울 송파구", 
            "avg_house_price": 400000000, "policy_content": "규제지역의 LTV를 40%로 축소..."
        }
        return state
    query = f"SELECT * FROM reports WHERE member_id = {state['member_id']} ORDER BY month DESC LIMIT 1"
    return query_mysql(state, query, key="report_data")

def load_house_info(state: Dict[str, Any]) -> Dict[str, Any]:
    print("🏠 [Tool Node] 주택 정보 검색 중...")
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 주택 정보를 불러옵니다.")
        state["house_info"] = {"price": 420000000, "location": "서울 송파구"}
        return state
    state["house_info"] = {"avg_price": 420000000, "region": "Seoul"}
    return state

def load_policy_info(state: Dict[str, Any]) -> Dict[str, Any]:
    print("📜 [Tool Node] 정책 정보 검색 중...")
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 정책 정보를 불러옵니다.")
        state["policy_info"] = {
            "content": "10월 15일 대책 발표...", "updated_at": "2025-10-15",
        }
        return state
    state["policy_info"] = {"new_policy": "청년 주택 대출 한도 2배 확대"}
    return state

def load_credit_info(state: Dict[str, Any]) -> Dict[str, Any]:
    print("💳 [Tool Node] 개인 신용정보 불러오는 중...")
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 신용 정보를 불러옵니다.")
        state["credit_info"] = {"score": 780, "debt": 1200}
        return state
    state["credit_info"] = {"score": 780, "debt": 1200}
    return state


# ==============================================================================
# 2. 🧾 consume 에이전트용: 모델 활용 및 분석 노드 (더미 assets 사용)
# ==============================================================================
def get_user_cluster_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """사용자 ID를 기반으로 군집을 예측하고 데이터를 추출합니다. (is_test 지원)"""
    assets = state.get('assets', {})
    user_id = state.get('user_id')
    
    # 🟢 [수정] is_test 확인 및 모델 사용 로직 주석 처리
    if state.get("is_test") or not assets.get('knn_model'):
        print("🧪 [TEST MODE] Consume: 모델 건너뛰고 더미 군집 ID 할당.")
        user_cluster = 1 # 더미 군집 ID
        
        # 더미 데이터 생성 시 user_id를 참조할 수 있도록 보강
        user_data_row = assets.get('df_data', pd.DataFrame({'user_id': [user_id if user_id else 1], 'CAT2_A': [10.0], 'total_spend': [10.0], 'spend_month': ['2025-01']}, index=[0])).iloc[0]
        
        state['user_cluster'] = int(user_cluster)
        state['user_data'] = user_data_row.to_dict()
        print(f"✅ [Tool Node] 군집 예측 완료 (TEST Cluster: {user_cluster})")
        return state
    
    # 🚨 [주석 처리] is_test가 False일 때의 실제 로직 (파일 로드 및 모델 사용)
    # try: 
    #     user_data_row = assets['df_data'][assets['df_data']['user_id'] == user_id].sort_values(by='spend_month', ascending=False).iloc[0]
    #     user_features = user_data_row[assets['cat2_cols']].values.reshape(1, -1)
    #     user_scaled = assets['scaler'].transform(user_features)
    #     user_cluster = assets['knn_model'].predict(user_scaled)[0]
    #     state['user_cluster'] = int(user_cluster)
    #     state['user_data'] = user_data_row.to_dict()
    #     print(f"✅ [Tool Node] 군집 예측 완료 (Cluster: {user_cluster})")
    # except Exception as e:
    #     print(f"❌ [Tool Node] 군집 예측 실패: {e}")
    #     state['user_cluster'] = -1
        
    return state

def generate_cluster_nickname_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """군집 ID를 기반으로 별명을 생성합니다."""
    assets = state.get('assets', {})
    cluster_id = state.get('user_cluster')
    df_profile = assets.get('df_profile')

    try:
        if cluster_id == -1:
             nickname = "**[ TEST MODE: 실패 그룹 ]**"
        else:
            profile = df_profile.iloc[0] # 더미 DataFrame의 첫 행 사용
            top3_cats = ['외식', '쇼핑'] 
            avg_age = 35
            age_str = "핵심 소비 세대의" 
            nickname = (f"**[ {age_str} {top3_cats[0]} 및 {top3_cats[1]} 집중형 그룹 ]** "
                        f"평균 나이 {avg_age}세")
        
        state['cluster_nickname'] = nickname
        print("✅ [Tool Node] 군집 별명 생성 완료")
    except Exception as e:
        print(f"❌ [Tool Node] 군집 별명 생성 실패: {e}")
        state['cluster_nickname'] = "분석 실패 그룹"
        
    return state

def analyze_user_spending_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """개인 데이터를 기반으로 소비액과 항목 등을 분석합니다."""
    assets = state.get('assets', {})
    user_data = state.get('user_data', {})
    
    try:
        user_spending = pd.Series({k: v for k, v in user_data.items() if k in assets.get('cat2_cols', [])}).sort_values(ascending=False)
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
        print("✅ [Tool Node] 개인 소비 분석 완료")
    except Exception as e:
        print(f"❌ [Tool Node] 개인 소비 분석 실패: {e}")
        state['user_analysis'] = {}
        
    return state


# ==============================================================================
# 3. 💰 profit 에이전트용: 금융 데이터 처리 및 계산 노드 (is_test 지원)
# ==============================================================================
def load_data():
    """JSON 파일에서 투자 상품 데이터를 로드합니다. (is_test=False 시 호출)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'profit', 'data', 'test_data.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: [Tool Node] 데이터 파일을 찾을 수 없습니다. 시도한 경로: {file_path}")
        raise

def calculate_deposit_profit(deposit: Dict[str, Any]) -> Dict[str, Any]:
    """예금의 만기 이자 수익(세후)을 계산합니다."""
    principal = deposit['principal']
    rate = deposit['interest_rate']
    tax = deposit['tax_rate']
    
    gross_interest = principal * rate * (deposit['total_period_months'] / 12)
    net_interest = gross_interest * (1 - tax)
    return {
        'product_id': deposit['product_id'], 'type': '예금', 'principal': principal,
        'gross_profit': gross_interest, 'net_profit': net_interest, 'net_profit_loss': 0.0
    }

def calculate_savings_profit(savings: Dict[str, Any]) -> Dict[str, Any]:
    """적금의 만기 이자 수익(세후)을 계산합니다. (단리 기준)"""
    monthly_payment = savings['monthly_payment']
    period = savings['total_period_months']
    rate = savings['interest_rate']
    tax = savings['tax_rate']
    
    gross_interest = monthly_payment * (rate / 12) * (period * (period + 1) / 2)
    net_interest = gross_interest * (1 - tax)
    return {
        'product_id': savings['product_id'], 'type': '적금', 'principal': monthly_payment * period,
        'gross_profit': gross_interest, 'net_profit': net_interest, 'net_profit_loss': 0.0
    }

def calculate_fund_loss_profit(fund: Dict[str, Any], report_date: str) -> Dict[str, Any]:
    """펀드의 현재 시점 수익/손실을 계산합니다."""
    purchase_nav = fund['purchase_nav']
    current_nav = fund['current_nav']
    total_shares = fund['total_shares']
    fee_rate = fund['fee_rate']
    
    current_value = total_shares * current_nav
    total_purchase_cost = total_shares * purchase_nav
    profit_loss = current_value - total_purchase_cost
    
    fee = total_purchase_cost * fee_rate 
    net_profit_loss = profit_loss - fee
    
    return {
        'product_id': fund['product_id'], 'type': '펀드', 'principal': total_purchase_cost, 
        'current_value': current_value, 'profit_loss': profit_loss, 'net_profit_loss': net_profit_loss,
        'net_profit': 0.0
    }

def aggregate_financial_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """모든 상품의 수익/손실을 계산하고 집계하여 상태에 저장합니다. (is_test 지원)"""
    try:
        # 🟢 [수정] TEST MODE 확인: 파일 로드 대신 더미 데이터를 사용
        if state.get("is_test"):
            print("🧪 [TEST MODE] Profit: 더미 금융 데이터를 사용합니다.")
            data = {
                "report_date": "2025-11-01",
                "deposits": [{"principal": 5000000, "interest_rate": 0.03, "tax_rate": 0.154, "total_period_months": 12, "product_id": "D001"}],
                "savings": [],
                "funds": [],
            }
        else:
            # is_test=False 시 실제 로직: 파일을 로드 (경로가 올바른지 확인해야 함)
            data = load_data()
            
        all_results = []
        report_date = data['report_date']

        # 계산 로직
        for dep in data.get('deposits', []):
            all_results.append(calculate_deposit_profit(dep))
            
        for sav in data.get('savings', []):
            all_results.append(calculate_savings_profit(sav))
            
        for fun in data.get('funds', []):
            all_results.append(calculate_fund_loss_profit(fun, report_date))

        df = pd.DataFrame(all_results)
        
        # 집계 로직
        total_principal = df['principal'].sum()
        net_p = df['net_profit'].fillna(0).sum()
        net_l = df['net_profit_loss'].fillna(0).sum()
        total_net_profit_loss = net_p + net_l
        
        state['analysis_df'] = df
        state['total_principal'] = total_principal
        state['total_net_profit_loss'] = total_net_profit_loss
        print("✅ [Tool Node] 금융 데이터 집계 완료")

    except Exception as e:
        print(f"❌ [Tool Node] 금융 데이터 집계 실패: {e}")
        state['analysis_df'] = pd.DataFrame()
        state['total_principal'] = 0
        state['total_net_profit_loss'] = 0
        
    return state