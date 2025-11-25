# report_project/report/main_orchestrator.py

import sys
import os
import io
from typing import Dict, Any, Optional
import pandas as pd 
import re 
import datetime 
from fastapi import HTTPException # Exception 처리를 위해 추가

# 🚨 [핵심 수정] 1단계: 현재 폴더 (report) 경로를 PYTHONPATH에 추가
# 이렇게 하면 아래의 'from report.consume...' 임포트가 작동합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 🚨 [수정] 2단계: report 제거
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
        return "분석 결과 없음"
    
    # Markdown 제거 및 공백 정리 로직
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'(\*{2}|_{2})(.*?)\1', r'\2', text) 
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'^\s*[\-\*]\s*', '', text, flags=re.MULTILINE)
    
    return text.strip()

# ----------------------------------------------------------------------
# 🛠️ 유틸리티: 실행 로그 임시 비활성화 함수 (노드 내부 print 억제용)
# ----------------------------------------------------------------------
def suppress_stdout(func, *args, **kwargs):
    """함수를 실행하는 동안 모든 표준 출력(print)을 억제합니다."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        sys.stdout = old_stdout


# ----------------------------------------------------------------------
# 🛠️ 설정 변수
# ----------------------------------------------------------------------
GLOBAL_OLLAMA_MODEL = "qwen3:8b" 
TEST_MEMBER_ID = 1004
TEST_USER_ID = 500 


def run_full_report_pipeline(member_id: int, user_id: int, ollama_model: str) -> Dict[str, Any]:
    """세 에이전트를 실행하고 최종 결과를 통합합니다. (로그 출력 없음)"""
    
    final_output: Dict[str, Any] = {}
    
    # ------------------------------------------------
    # 1, 2, 3. Profit, Consume, Compare 에이전트 실행 (로그 억제)
    # ------------------------------------------------
    # Profit Agent 실행 (TEST MODE)
    try:
        profit_initial_state = AgentState(
            raw_data={}, analysis_df=pd.DataFrame(), total_principal=0.0,
            total_net_profit_loss=0.0, chart_data={}, investment_analysis_result="",
            member_id=member_id, user_id=user_id, is_test=True, 
            ollama_model_name=ollama_model, assets=None, report_data=None 
        )
        # 🚨 suppress_stdout 적용하여 내부 로그 숨김
        profit_result = suppress_stdout(execute_profit_agent, profit_initial_state) 
        final_output.update(profit_result)
        
    except Exception as e:
        final_output["profit_report"] = f"실패: 투자 분석을 수행할 수 없습니다. 오류: {e}"
        final_output["total_net_profit_loss"] = 0.0


    # Consume Agent 실행 (TEST MODE)
    try:
        consume_initial_state = AgentState(
            member_id=member_id, user_id=user_id, is_test=True, 
            ollama_model_name=ollama_model, raw_data={}, analysis_df=pd.DataFrame()
        )
        consume_result = suppress_stdout(execute_consume_agent, consume_initial_state)
        final_output.update(consume_result)
        
    except Exception as e:
        final_output["consumption_report"] = f"실패: 소비 분석을 수행할 수 없습니다. 오류: {e}"


    # Compare Agent 실행 (RAG 실습)
    try:
        compare_initial_state = AgentState(
            member_id=member_id, is_test=True, 
            user_id=user_id, ollama_model_name=ollama_model, raw_data={}, analysis_df=pd.DataFrame()
        )
        compare_result = suppress_stdout(execute_compare_agent, compare_initial_state) 
        final_output.update(compare_result)
        
    except Exception as e:
        final_output["comparison_result"] = f"실패: 변동 사항 비교를 수행할 수 없습니다. 오류: {e}"
        final_output["house_info"] = {}


    # ------------------------------------------------
    # 4. 최종 통합 보고서 생성 및 반환 (JSON 구조화)
    # ------------------------------------------------
    
    house_info_data = final_output.get('house_info')
    house_price = house_info_data.get('price', 'N/A') if isinstance(house_info_data, dict) else 'N/A'

    # 최종 보고서 String (FastAPI의 'report' 필드에 들어갈 내용)
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
    
    # 🚨 DB 저장을 위해 모든 데이터를 구조화하여 반환
    final_json_data = {
        "consume_cluster": final_output.get('cluster_nickname', 'N/A'),
        "consume_analysis": final_output.get('consumption_report', 'N/A'),
        "profit_total_net_profit_loss": final_output.get('total_net_profit_loss', 0),
        "profit_analysis": final_output.get('profit_report', 'N/A'),
        "compare_changes": final_output.get('comparison_result', 'N/A'),
        "compare_house_price": house_price,
        "full_report_string": final_report_text, # FastAPI 'report' 필드에 사용될 통합 텍스트
        "metadata": {
            "member_id": member_id,
            "generated_at": datetime.datetime.now().isoformat(),
        }
    }

    # 최종적으로 FastAPI가 사용할 JSON 객체 반환
    return final_json_data


if __name__ == "__main__":
    # 이 블록은 디버깅용으로, 실행 시 최종 보고서 String을 출력합니다.
    result = run_full_report_pipeline(
        member_id=TEST_MEMBER_ID, 
        user_id=TEST_USER_ID,
        ollama_model=GLOBAL_OLLAMA_MODEL
    )
    print(result.get("full_report_string"))