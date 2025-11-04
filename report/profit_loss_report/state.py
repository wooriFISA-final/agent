# state.py

from typing import TypedDict, List
import pandas as pd

class AgentState(TypedDict):
    """LangGraph 워크플로우의 상태를 정의합니다."""
    raw_data: dict                    # 로드된 원시 데이터
    df_results: pd.DataFrame          # 계산된 결과 DataFrame
    total_principal: float            # 총 투자 원금
    total_net_profit_loss: float      # 총 순수익/손실
    chart_data: dict                  # 시각화 차트 데이터
    llm_report: str                   # LLM이 생성한 최종 보고서