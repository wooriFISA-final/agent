# report_project/main_orchestrator.py (최종 클린 출력 버전)

import sys
from typing import Dict, Any, Optional
import pandas as pd 
import re # Markdown 클리닝을 위해 추가
import os # sys.path.append에 필요

# 🚨 [경로 주입] builder 파일들이 nodes를 찾도록 Python 경로에 report 폴더 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from consume.execute import execute_consume_agent
from compare.execute import execute_compare_agent
from profit.execute import execute_profit_agent
from state import AgentState 

# ----------------------------------------------------------------------
# 🛠️ 유틸리티: LLM 출력 정리 함수 (Markdown 클리닝)
# ----------------------------------------------------------------------
def clean_markdown_output(text: str) -> str:
    """LLM이 생성한 텍스트에서 불필요한 Markdown 문자를 제거하고 정리합니다."""
    if text is None:
        return ""
    
    # 1. Markdown 헤더/굵게 제거 (#, **, etc.)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'(\*{2}|_{2})(.*?)\1', r'\2', text) 
    
    # 2. 연속된 공백 및 줄바꿈을 공백으로 통일
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    
    # 3. 리스트 마커 제거 (- , *)
    text = re.sub(r'^\s*[\-\*]\s*', '', text, flags=re.MULTILINE)
    
    return text.strip()


# ----------------------------------------------------------------------
# 🛠️ 설정 변수
# ----------------------------------------------------------------------
GLOBAL_OLLAMA_MODEL = "qwen3:8b" 
TEST_MEMBER_ID = 1004
TEST_USER_ID = 500 


def run_full_report_pipeline(member_id: int, user_id: int, ollama_model: str) -> Dict[str, Any]:
    """세 에이전트 (Consume, Compare, Profit)를 순서대로 실행하고 최종 결과를 통합합니다."""
    
    final_output: Dict[str, Any] = {}
    
    # ------------------------------------------------
    # 1. 💰 Profit 에이전트 실행 (TEST MODE)
    # ------------------------------------------------
    try:
        profit_initial_state = AgentState(
            raw_data={}, analysis_df=pd.DataFrame(), total_principal=0.0,
            total_net_profit_loss=0.0, chart_data={}, investment_analysis_result="",
            member_id=member_id, user_id=user_id, is_test=True, 
            ollama_model_name=ollama_model, assets=None, report_data=None 
        )
        
        profit_result = execute_profit_agent(profit_initial_state) 
        final_output.update(profit_result)
        
    except Exception as e:
        final_output["profit_report"] = f"실패: 투자 분석을 수행할 수 없습니다. 오류: {e}"
        final_output["total_net_profit_loss"] = 0.0


    # ------------------------------------------------
    # 2. 🧾 Consume 에이전트 실행 (TEST MODE)
    # ------------------------------------------------
    try:
        consume_initial_state = AgentState(
            member_id=member_id, user_id=user_id, is_test=True, 
            ollama_model_name=ollama_model, raw_data={}, analysis_df=pd.DataFrame()
        )
        consume_result = execute_consume_agent(consume_initial_state)
        final_output.update(consume_result)
        
    except Exception as e:
        final_output["consumption_report"] = f"실패: 소비 분석을 수행할 수 없습니다. 오류: {e}"


    # ------------------------------------------------
    # 3. 🔍 Compare 에이전트 실행 (정책 비교 RAG 실습)
    # ------------------------------------------------
    try:
        compare_initial_state = AgentState(
            member_id=member_id, is_test=True, 
            user_id=user_id, ollama_model_name=ollama_model, raw_data={}, analysis_df=pd.DataFrame()
        )
        
        compare_result = execute_compare_agent(compare_initial_state) 
        final_output.update(compare_result)
        
    except Exception as e:
        final_output["comparison_result"] = f"실패: 변동 사항 비교를 수행할 수 없습니다. 오류: {e}"
        final_output["house_info"] = {} # 안전한 빈 딕셔너리로 초기화


    # ------------------------------------------------
    # 4. 최종 통합 보고서 생성 및 출력
    # ------------------------------------------------

    # [AttributeError 해결] house_info 키가 None이거나 없더라도 안전하게 값을 추출
    house_info_data = final_output.get('house_info')
    house_price = house_info_data.get('price', 'N/A') if isinstance(house_info_data, dict) else 'N/A'

    final_report_text = f"""
================================================================================
        🎉 최종 통합 보고서 결과 🎉
================================================================================

[회원 ID: {member_id} / 통합 분석 보고서]
=======================================

1. 소비 분석 결과 (Consume Agent)
---------------------------------------
- 군집 유형: {final_output.get('cluster_nickname', 'N/A')}
- 상세 분석: {clean_markdown_output(final_output.get('consumption_report', 'N/A'))}

2. 투자 분석 결과 (Profit Agent)
---------------------------------------
- 순수익/손실: {final_output.get('total_net_profit_loss', 0):,.0f} 원
- 상세 분석: {clean_markdown_output(final_output.get('profit_report', 'N/A'))}

3. 환경 변화 분석 (Compare Agent)
---------------------------------------
- 주요 변동 사항: {clean_markdown_output(final_output.get('comparison_result', 'N/A'))}
- 주택 정보: {house_price} 
=======================================
"""
    
    final_output["final_integrated_report"] = final_report_text
    print(final_output["final_integrated_report"])
    
    return final_output


if __name__ == "__main__":
    # 🚨 실행 위치: agent 폴더 내에서 실행
    run_full_report_pipeline(
        member_id=TEST_MEMBER_ID, 
        user_id=TEST_USER_ID,
        ollama_model=GLOBAL_OLLAMA_MODEL
    )