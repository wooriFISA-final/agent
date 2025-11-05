# report_project/main_orchestrator.py

import sys
from typing import Dict, Any, Optional
import pandas as pd 

from consume.execute import execute_consume_agent
from compare.execute import execute_compare_agent
from profit.execute import execute_profit_agent
from state import AgentState 

# ----------------------------------------------------------------------
# 🛠️ 설정 변수 (모두 TEST MODE 사용)
# ----------------------------------------------------------------------
GLOBAL_OLLAMA_MODEL = "qwen3:8b" 
TEST_MEMBER_ID = 1004
TEST_USER_ID = 500 


def run_full_report_pipeline(member_id: int, user_id: int, ollama_model: str) -> Dict[str, Any]:
    # ... (실행 시작 출력 로직 유지)
    print("\n" + "="*80)
    print("      🚀 FINAL REPORT ORCHESTRATOR 실행 시작 🚀")
    print("="*80)
    
    final_output: Dict[str, Any] = {}
    
    # ------------------------------------------------
    # 1. 💰 Profit 에이전트 실행 (TEST MODE)
    # ------------------------------------------------
    try:
        print("\n--- [Phase 1/3: Profit Agent] 투자 분석 실행 ---")
        profit_initial_state = AgentState(
            raw_data={}, analysis_df=pd.DataFrame(), total_principal=0.0,
            total_net_profit_loss=0.0, chart_data={}, investment_analysis_result="",
            member_id=member_id, user_id=user_id, is_test=True, # 🚨 is_test=True
            ollama_model_name=ollama_model, house_info=None, assets=None 
        )
        # execute_profit_agent는 상태 객체를 인수로 받도록 수정한다고 가정하고 호출
        profit_result = execute_profit_agent(profit_initial_state) 
        final_output.update(profit_result)
        print("✅ Profit Agent 실행 완료.")
        
    except Exception as e:
        print(f"❌ Profit Agent 실행 실패: {e}")
        final_output["profit_report"] = "실패: 투자 분석을 수행할 수 없습니다."
        final_output["total_net_profit_loss"] = 0.0


    # ------------------------------------------------
    # 2. 🧾 Consume 에이전트 실행 (TEST MODE)
    # ------------------------------------------------
    try:
        print("\n--- [Phase 2/3: Consume Agent] 소비 분석 실행 ---")
        consume_initial_state = AgentState(
            member_id=member_id, user_id=user_id, is_test=True, # 🚨 is_test=True
            ollama_model_name=ollama_model, raw_data={}, analysis_df=pd.DataFrame() # 기타 필수 필드
        )
        
        consume_result = execute_consume_agent(consume_initial_state) # execute_consume_agent는 상태 객체를 인수로 받음
        final_output.update(consume_result)
        print("✅ Consume Agent 실행 완료.")
        
    except Exception as e:
        print(f"❌ Consume Agent 실행 실패: {e}")
        final_output["consumption_report"] = "실패: 소비 분석을 수행할 수 없습니다."


    # ------------------------------------------------
    # 3. 🔍 Compare 에이전트 실행 (TEST MODE)
    # ------------------------------------------------
    try:
        print("\n--- [Phase 3/3: Compare Agent] 변동 사항 비교 실행 ---")
        compare_initial_state = AgentState(
            member_id=member_id, is_test=True, # 🚨 is_test=True
            user_id=user_id, ollama_model_name=ollama_model, raw_data={}, analysis_df=pd.DataFrame() # 기타 필수 필드
        )
        compare_result = execute_compare_agent(compare_initial_state)
        final_output.update(compare_result)
        print("✅ Compare Agent 실행 완료.")
        
    except Exception as e:
        print(f"❌ Compare Agent 실행 실패: {e}")
        final_output["comparison_result"] = "실패: 변동 사항 비교를 수행할 수 없습니다."
        

    # ------------------------------------------------
    # 4. 최종 통합 보고서 생성
    # ------------------------------------------------
    print("\n" + "="*80)
    print("        🎉 최종 통합 보고서 결과 🎉")
    print("="*80)
    
    final_report_text = f"""
[회원 ID: {member_id} / 통합 분석 보고서]
=======================================

1. 소비 분석 결과 (Consume Agent)
---------------------------------------
- 군집 유형: {final_output.get('cluster_nickname', 'N/A')}
- 상세 분석: {final_output.get('consumption_report', 'N/A')}

2. 투자 분석 결과 (Profit Agent)
---------------------------------------
- 순수익/손실: {final_output.get('total_net_profit_loss', 0):,.0f} 원
- 상세 분석: {final_output.get('profit_report', 'N/A')}

3. 환경 변화 분석 (Compare Agent)
---------------------------------------
- 주요 변동 사항: {final_output.get('comparison_result', 'N/A')}
- 주택 정보: {final_output.get('house_info', {}).get('price', 'N/A')}

======================================="""
    
    final_output["final_integrated_report"] = final_report_text
    print(final_report_text)
    print("\n" + "="*80)

    return final_output


if __name__ == "__main__":
    run_full_report_pipeline(
        member_id=TEST_MEMBER_ID, 
        user_id=TEST_USER_ID,
        ollama_model=GLOBAL_OLLAMA_MODEL
    )