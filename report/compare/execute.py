# report_project/compare/execute.py

from typing import Dict, Any
# 같은 폴더의 builder.py에서 그래프 빌드 함수를 import 합니다.
from .builder import build_compare_graph 


def execute_compare_agent(initial_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare 에이전트를 실행하고 최종 결과를 반환합니다.
    
    Args:
        initial_input: member_id, is_test 등을 포함하는 초기 상태 입력.
        
    Returns:
        Dict[str, Any]: 최종 비교 결과를 담은 딕셔너리.
    """
    print("\n🚀 Compare 에이전트 실행 시작...")
    
    # 1. 그래프 빌드
    compare_graph = build_compare_graph()
    
    # 2. 초기 상태 설정 및 그래프 실행
    try:
        # LangGraph invoke를 사용하여 그래프를 실행합니다.
        final_state = compare_graph.invoke(initial_input) 
        
        print("✅ Compare 에이전트 실행 완료.")
        
        # 3. 최종 결과 반환 (main_orchestrator가 사용할 핵심 결과)
        return {
            "comparison_result": final_state["comparison_result"],
            # 필요하다면 로드된 데이터 일부를 통합에 사용하기 위해 반환할 수 있습니다.
            "house_info": final_state["house_info"], 
        }
    
    except Exception as e:
        print(f"❌ Compare 에이전트 실행 중 오류 발생: {e}")
        return {
            "comparison_result": f"Compare 에이전트 실행 실패: {e}",
            "house_info": None,
        }

# ⚠️ execute.py 파일에는 __main__ 로직이 포함되지 않도록 분리합니다.
# 테스트 로직은 main_orchestrator.py에 통합되어야 합니다.