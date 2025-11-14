# report_project/nodes/tool_nodes.py

import pandas as pd
import json
import os
from typing import Dict, Any, Tuple, List
import numpy as np 

# 🚨 Compare Agent RAG 실습을 위한 import
from report.compare.policy_retriever import retrieve_policy_changes

# ----------------------------------------------------------------------
# DB 툴 함수를 가져옵니다.
try:
    from report.nodes.db_tools import (
        fetch_user_products, 
        fetch_user_consume_data, 
        fetch_user_id, 
        fetch_recent_report_summary, # reports 테이블 조회 함수
        fetch_house_price # HOUSE_PRICES 테이블 조회 함수
    )
except ImportError:
    print("❌ 오류: 'report/nodes/db_tools.py' 로드 실패. DB 연결 기능을 사용할 수 없습니다.")
    # 실제 연결이 실패했을 때를 대비해 빈 데이터를 반환하는 목업 함수를 정의합니다.
    def fetch_user_id(user_name): return None
    def fetch_user_products(user_id): return []
    def fetch_user_consume_data(user_id, month=None): return []
    def fetch_recent_report_summary(member_id): return None
# ----------------------------------------------------------------------

# 🚨 [경로 정리]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(CURRENT_DIR, '..', 'compare', 'data')

FILE_NAME_OLD = "20241224.pdf"
FILE_NAME_NEW = "20250305.pdf"

POLICY_PATH_OLD = os.path.join(BASE_DATA_DIR, FILE_NAME_OLD)
POLICY_PATH_NEW = os.path.join(BASE_DATA_DIR, FILE_NAME_NEW)
POLICY_FAILURE_MESSAGE = "🚨 정책 파일 로드 실패: PDF 원본 파일을 찾거나 처리할 수 없습니다."


# ==============================================================================
# 1. 🔍 compare 에이전트용: 데이터 로드 및 검색 노드 (DB 연결 반영)
# ==============================================================================
def load_prev_month_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """reports 테이블에서 직전 월 레포트 요약 데이터를 로드합니다."""
    if state.get("is_test"):
        state["report_data"] = {
            "month": "2025-10", "income": 5000000, "loan_balance": 20000000, 
            "credit_score": 800, "target_location": "서울 송파구", 
            "avg_house_price": 400000000, "policy_content": "규제지역의 LTV를 40%로 축소..."
        }
        return state
        
    member_id = state.get('member_id')
    report_record = fetch_recent_report_summary(member_id) 
    
    if report_record:
        state["report_data"] = report_record
    else:
        state["report_data"] = {}
        
    return state

def load_house_info(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("is_test"):
        state["house_info"] = {"price": 420000000, "location": "서울 송파구"}
        return state
    # DB에서 실제 지역의 평균 가격 정보 로직으로 대체 필요 (현재는 Mocked)
    state["house_info"] = {"avg_price": 420000000, "region": "Seoul"}
    return state

def load_policy_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """RAG 실습을 위해, 정책 PDF 파일을 로드/검색하는 노드."""
    print("📜 [Tool Node] FAISS DB에서 정책 변동 청크 검색 중...")
    
    query = state.get('user_query', "2024년 12월 정책과 2025년 3월 정책 사이에서 **가장 중요한 3가지 변동 사항**에 대해 비교 분석하시오.")
    
    old_policies = [] 
    new_policies = [] 
    retrieved_chapters = []

    try:
        retrieved_chapters = retrieve_policy_changes(query, k=10)
        
        for chapter in retrieved_chapters:
            if isinstance(chapter, dict) and 'content' in chapter:
                source_identifier = chapter.get('title', '')
                
                if FILE_NAME_OLD[:-4] in source_identifier: 
                    old_policies.append(chapter['content'])
                elif FILE_NAME_NEW[:-4] in source_identifier: 
                    new_policies.append(chapter['content'])
        
        state['retrieved_documents'] = retrieved_chapters 
             
    except Exception as e:
        print(f"❌ RAG 시스템 오류 발생: {str(e)}")
        
    state["policy_info"] = {
        "old_policy": old_policies, 
        "new_policy": new_policies
    }
    
    return state

def load_credit_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    DB members 테이블에서 사용자의 주요 금융, 부채, 자격 정보를 조회합니다.
    """
    # 🚨 user_id를 1로 고정하여 사용합니다. (load_user_consume_data에서 설정됨)
    user_id = state.get('user_id', 1) 
    
    # 1. DB 연결: fetch_member_details 함수를 통해 실제 상세 정보 로드
    member_record = fetch_member_details(user_id)
    
    if member_record:
        # DB에서 가져온 레코드를 State 형식에 맞게 정리 (요청하신 주요 항목 반영)
        state["member_credit_info"] = {
            # 현재 신용 점수
            "credit_score": member_record.get('credit_score', None),
            # 월 급여
            "monthly_salary": member_record.get('monthly_salary', None),
            # 연봉
            "annual_salary": member_record.get('annual_salary', None),
            # 총 부채액
            "total_debt": member_record.get('total_debt', None),
            # 주택 보유 여부
            "has_house": member_record.get('has_house', False),
            # DSR
            "DSR": member_record.get('DSR', None),
            # 저소득층 여부 (자격 정보 예시)
            "is_low_income_class": member_record.get('is_low_income_class', False)
        }
        print(f"✅ [Tool Node] User ID {user_id}의 멤버 상세 정보 로드 완료.")
    else:
        print(f"⚠️ [Tool Node] User ID {user_id}의 멤버 상세 정보 로드 실패. 빈 Dict 반환.")
        state["member_credit_info"] = {}
        
    return state


# ==============================================================================
# 2. 🧾 consume 에이전트용: 데이터 로드 노드 (유저 ID 1로 고정 수정)
# ==============================================================================
def load_user_consume_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    유저 ID 1의 2023년 2월 및 1월 데이터를 DB에서 로드하여 상태에 저장합니다.
    """
    # 🚨 유저 ID를 1로 고정하여 사용합니다.
    user_id = 1
    
    # DB의 spend_month 컬럼 형식에 맞춰야 합니다.
    target_months = ["2023-02-01", "2023-01-01"] 
    
    # 1. 테스트 모드 처리 (Mockup 제거)
    if state.get("is_test"):
        print("🔎 [DEBUG] 테스트 모드 실행. DB 접근을 건너뛰고 빈 DataFrame을 반환합니다.")
        state['df_consume_data'] = pd.DataFrame()
        return state
    
    # 🚨 user_id가 1로 고정되었으므로, ID 조회 로직을 건너뜁니다.
    state['user_id'] = user_id 
    
    # 2. DB에서 1월과 2월 데이터 조회
    print(f"📜 [Tool Node] User ID {user_id}의 {target_months} 데이터 조회 중...")
    consume_records = fetch_user_consume_data(user_id, target_months) 
    
    if not consume_records:
        print(f"⚠️ [Tool Node] {target_months}에 해당하는 소비 데이터가 없습니다. 분석을 건너뜁니다.")
        state['df_consume_data'] = pd.DataFrame()
        return state
    
    # 3. Pandas DataFrame으로 변환 및 저장
    df = pd.DataFrame(consume_records)
    state['df_consume_data'] = df
    
    print(f"✅ [Tool Node] 소비 데이터 {len(df)}건 로드 완료 (2월 및 1월).")
    
    return state

def get_user_cluster_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자의 소비 패턴 클러스터를 계산하는 노드.
    (클러스터링 모델(knn_model)과 클러스터링을 위한 데이터프레임이 필요합니다.)
    """
    if state.get("is_test"):
        # 테스트 로직 유지 (실제 모델이 없으므로)
        user_cluster = 1
        state['user_cluster'] = int(user_cluster)
        return state
        
    # 🚨 [주석] 여기에 실제 클러스터링 모델을 이용한 계산 로직을 구현해야 합니다.
    print("⚠️ [Tool Node] 클러스터링 로직 구현 필요. 임시 값을 사용합니다.")
    state['user_cluster'] = 1 
    
    return state


def generate_cluster_nickname_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """클러스터 ID를 기반으로 별명을 생성하는 노드."""
    cluster_id = state.get('user_cluster')
    
    if cluster_id is None:
        state['cluster_nickname'] = "데이터 부족 그룹"
        return state

    # 현재는 Mocked Nickname 반환
    state['cluster_nickname'] = f"**[AI 분석]** 소비 패턴 그룹 {cluster_id} 유형"
    
    return state

def analyze_user_spending_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    로드된 소비 데이터를 기반으로 총 지출, Top 3 카테고리 등을 분석하고 1월/2월 비교 아웃풋을 생성합니다.
    """
    df_consume = state.get('df_consume_data')
    
    if df_consume is None or df_consume.empty or len(df_consume) < 2:
        print("⚠️ [Tool Node] 분석을 위한 월별 데이터(2개)가 부족합니다.")
        state['user_analysis'] = {"error": "비교 분석을 위한 최소 2개월의 데이터가 필요합니다."}
        return state
    
    try:
        # 1. 1월과 2월 데이터 분리
        df_consume = df_consume.sort_values(by='spend_month', ascending=False)
        feb_data = df_consume.iloc[0] # 2월 데이터 (가장 최근)
        jan_data = df_consume.iloc[1] # 1월 데이터
        
        # 2. 총 지출 비교 (원 단위 가정)
        total_spend_feb = feb_data.get('total_spend', 0)
        total_spend_jan = jan_data.get('total_spend', 0)
        
        diff = total_spend_feb - total_spend_jan
        change_rate = (diff / total_spend_jan) * 100 if total_spend_jan else 0

        # 3. Top 3 카테고리 비교 (CAT1 기준)
        # CAT1_교통, CAT1_쇼핑 등 DB에서 가져온 컬럼명을 사용해야 합니다.
        cat1_cols = [col for col in feb_data.index if col.startswith('CAT1_')]
        
        feb_cats = feb_data[cat1_cols].sort_values(ascending=False).head(3)
        jan_cats = jan_data[cat1_cols].sort_values(ascending=False).head(3)
        
        # 4. 비교 요약 데이터 생성 (LLM 처리용)
        analysis_data = {
            'feb_total_spend': f"{total_spend_feb / 10000:.0f}만원",
            'jan_total_spend': f"{total_spend_jan / 10000:.0f}만원",
            'total_change_diff': f"{diff / 10000:+.0f}만원",
            'total_change_rate': f"{change_rate:.1f}%",
            'feb_top_3_categories': [f"{col.replace('CAT1_', '')} ({val/10000:.0f}만)" for col, val in feb_cats.items()],
            'jan_top_3_categories': [f"{col.replace('CAT1_', '')} ({val/10000:.0f}만)" for col, val in jan_cats.items()],
            # 🚨 [주석] 여기에 상세 카테고리별 증감 분석 로직이 추가되어야 합니다.
        }
        
        state['user_analysis'] = analysis_data
        state['user_data'] = feb_data.to_dict() # 최신 데이터(2월)를 user_data로 저장
    except Exception as e:
        print(f"❌ 소비 분석 오류: {e}")
        state['user_analysis'] = {"error": f"분석 중 예외 발생: {str(e)}"}
        
    return state


# ==============================================================================
# 3. 💰 profit 에이전트용: 금융 데이터 처리 및 계산 노드 (주석 처리)
# ==============================================================================

# 🚨 [주석] profit Agent의 계산 로직은 DB에서 가져온 my_products 데이터 구조에 맞게 수정해야 합니다.
# 🚨 현재 load_data() 함수는 제거하고 fetch_user_products를 사용해야 합니다.

def load_data():
    """🚨 DB 통합 후 사용하지 않음. aggregate_financial_data_node에서 DB 툴을 직접 호출해야 합니다."""
    # JSON 파일에서 투자 상품 데이터를 로드합니다. (is_test=False 시 호출)
    raise NotImplementedError("DB 통합 후 load_data()는 사용되지 않습니다. aggregate_financial_data_node를 수정하세요.")

def calculate_deposit_profit(deposit: Dict[str, Any]) -> Dict[str, Any]:
    # ... (기존 계산 로직 유지)
    return {}

def calculate_savings_profit(savings: Dict[str, Any]) -> Dict[str, Any]:
    # ... (기존 계산 로직 유지)
    return {}

def calculate_fund_loss_profit(fund: Dict[str, Any], report_date: str) -> Dict[str, Any]:
    # ... (기존 계산 로직 유지)
    return {}

def aggregate_financial_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """모든 상품의 수익/손실을 계산하고 집계하여 상태에 저장합니다. (DB 데이터 사용)"""
    user_id = state.get('user_id')
    
    try:
        if state.get("is_test") or not user_id:
            # 테스트 모드 데이터
            data_list = [
                calculate_deposit_profit({"principal": 5000000, "interest_rate": 0.03, "tax_rate": 0.154, "total_period_months": 12, "product_id": "D001"}),
                calculate_savings_profit({"monthly_payment": 1000000, "interest_rate": 0.05, "tax_rate": 0.154, "total_period_months": 12, "product_id": "S001"}),
                calculate_fund_loss_profit({"purchase_nav": 1000, "current_nav": 1100, "total_shares": 10000, "fee_rate": 0.01, "product_id": "F001", "report_date": "2025-11-01"})
            ]
            df = pd.DataFrame(data_list)
        else:
            # 🚨 DB 연결: fetch_user_products 함수를 통해 my_products 테이블 데이터 로드
            db_products = fetch_user_products(user_id) 
            
            all_results = []
            
            # 🚨 [주석] 여기에 DB 데이터를 활용하여 계산 함수 호출 및 결과 생성 로직을 완성해야 합니다.
            # for prod in db_products:
            #     if prod['product_type'] == '예금':
            #         all_results.append(calculate_deposit_profit(prod))
            
            df = pd.DataFrame(all_results)
            print(f"✅ [Tool Node] 금융 상품 데이터 {len(df)}건 로드 완료.")
            
        # 집계 로직
        df['net_profit'] = df['net_profit'].fillna(0)
        df['net_profit_loss'] = df['net_profit_loss'].fillna(0)
        
        total_principal = df['principal'].sum()
        net_p = df['net_profit'].sum()
        net_l = df['net_profit_loss'].sum()
        total_net_profit_loss = net_p + net_l
        
        state['analysis_df'] = df
        state['total_principal'] = total_principal
        state['total_net_profit_loss'] = total_net_profit_loss
        
    except Exception as e:
        print(f"❌ 금융 데이터 집계 오류: {str(e)}")
        state['analysis_df'] = pd.DataFrame()
        state['total_principal'] = 0
        state['total_net_profit_loss'] = 0
        
    return state