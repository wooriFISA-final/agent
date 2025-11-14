# report_project/report/profit/execute.py

from typing import Dict, Any
import pandas as pd
from report.state import AgentState 
from report.profit.builder import build_profit_graph # 🚨 [수정]


def execute_profit_agent(initial_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profit 에이전트를 실행하고 최종 투자 분석 결과를 반환합니다.
    (main_orchestrator.py로부터 초기 상태를 인수로 받도록 수정)
    """
    print("\n🚀 Profit 에이전트 실행 시작...")
    
    # 1. 그래프 빌드
    profit_graph = build_profit_graph()
    
    # 2. 초기 상태 정의 (main_orchestrator에서 받은 인자를 그대로 사용)
    initial_state = initial_input
    
    # 3. 그래프 실행
    try:
        final_state = profit_graph.invoke(initial_state) 
        
        print("\n✅ Profit 에이전트 실행 완료.")
        
        # 4. 최종 결과 반환
        return {
            "total_principal": final_state.get("total_principal", 0.0),
            "total_net_profit_loss": final_state.get("total_net_profit_loss", 0.0),
            "profit_report": final_state.get("investment_analysis_result", "분석 실패"),
        }
    
    except Exception as e:
        print(f"❌ Profit 에이전트 실행 중 오류 발생: {e}")
        return {
            "profit_report": f"Profit 에이전트 실행 실패: {e}",
            "total_net_profit_loss": 0.0,
        }