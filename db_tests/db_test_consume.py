# /agent/db_tests/db_test_consume.py

import sys
import os
import pandas as pd
from typing import Dict, Any

# 프로젝트의 루트 디렉토리를 PYTHONPATH에 추가
# (report/nodes/tool_nodes.py를 가져오기 위해 필요)
# 현재 스크립트는 db_tests에 있으므로, 두 단계를 위로 이동하여 agent 폴더를 찾습니다.
# ====================================================================
# 🚨 수정된 경로 설정 🚨
# 현재 db_tests에서 두 단계 위로 올라가 '/agent' 디렉토리를 찾고,
# 이 디렉토리를 시스템 경로에 추가하여 'report' 패키지를 인식시킵니다.
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(PROJECT_ROOT)
# ====================================================================
# tool_nodes를 가져옵니다.
from report.nodes.tool_nodes import load_user_consume_data
from report.nodes.tool_nodes import analyze_user_spending_node 

def run_consume_test():
    """
    load_user_consume_data 노드를 실행하여 DB 연결 및 데이터 로드를 테스트합니다.
    """
    print("--- 💸 Consume Agent DB 연결 테스트 시작 ---")
    
    # 초기 상태 (유진수 님의 user_id는 DB에서 조회되므로, member_id 등은 비워둡니다.)
    initial_state: Dict[str, Any] = {
        "is_test": False,  # 실제 DB 접속을 위해 False 설정
        "user_id": None    # load_user_consume_data 내에서 '유진수' 이름으로 조회됨
    }

    # 1. 데이터 로드 노드 실행
    print("\n[1단계] load_user_consume_data 실행 중...")
    state_after_load = load_user_consume_data(initial_state)

    # 2. 결과 확인 및 검증
    df_consume = state_after_load.get('df_consume_data')
    
    if df_consume is None or df_consume.empty:
        print("\n❌ 테스트 실패: DataFrame이 로드되지 않았거나 비어 있습니다. (DB 연결 및 사용자/데이터 확인 필요)")
    else:
        print("\n✅ 테스트 성공: DB에서 소비 데이터 로드 완료.")
        print(f"   - 로드된 행 수: {len(df_consume)} (기대값: 2)")
        print(f"   - 컬럼 목록: {list(df_consume.columns)[:5]}...")
        print("   - 데이터 미리보기:")
        print(df_consume[['spend_month', 'total_spend', 'CAT1_교통', 'CAT1_식품']].head())
        
        # 3. 분석 노드 실행 (데이터 비교 로직 테스트)
        print("\n[2단계] analyze_user_spending_node 실행 중 (1월/2월 비교)...")
        state_after_analyze = analyze_user_spending_node(state_after_load)
        
        analysis = state_after_analyze.get('user_analysis', {})
        
        if 'error' in analysis:
             print(f"❌ 분석 실패: {analysis['error']}")
        else:
             print("✅ 분석 노드 실행 성공.")
             print("   - 2월 총 지출:", analysis.get('feb_total_spend'))
             print("   - 총 지출 변화율:", analysis.get('total_change_rate'))
             print("   - 2월 Top 3:", analysis.get('feb_top_3_categories'))


if __name__ == '__main__':
    # DB 연결 정보가 없는 경우를 대비해 os.environ에 직접 설정하거나 .env가 로드되어야 합니다.
    # .env 파일은 이미 load_dotenv()를 통해 로드되므로, 별도 설정 없이 실행합니다.
    run_consume_test()