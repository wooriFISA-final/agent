# report_project/report/compare/execute.py (RAG 테스트용)

from typing import Dict, Any
from report.state import AgentState 
from report.compare.builder import build_compare_graph 


def execute_compare_agent(initial_input: Dict[str, Any]) -> Dict[str, Any]:
    print("\n🚀 Compare 에이전트 실행 시작...")
    
    compare_graph = build_compare_graph()
    
    try:
        final_state = compare_graph.invoke(initial_input) 
        
        print("✅ Compare 에이जन트 실행 완료.")
        
        comp_result = final_state.get("comparison_result", "정책 비교 분석 결과를 찾을 수 없습니다.")
        
        return {
            "comparison_result": comp_result,
            "house_info": final_state.get("house_info", None), 
        }
    
    except Exception as e:
        print(f"❌ Compare 에이전트 실행 중 오류 발생: {e}")
        return {
            "comparison_result": f"Compare 에이전트 실행 실패: {type(e).__name__} - {str(e)}",
            "house_info": None,
        }

if __name__ == "__main__":
    
    # 1. Agent Flow 정의
    compare_agent_flow = build_compare_graph() 
    
    # 2. 테스트용 State 초기화 
    test_state = AgentState(
        report_type="compare",
        user_query="2024년 12월 정책과 2025년 3월 정책의 변동 사항을 자세히 비교 분석하고 요약해줘.",
        
        # flow 우회로 인해 필요 없지만, 다른 노드에서 사용할 수 있으므로 유지
        member_id="TEST_001", 
        
        # 🚨 [중요] RAG에 필요한 데이터 경로 설정 (정책 로드를 위한 입력값)
        policy_paths=[
            "report/compare/data/20241224.pdf",
            "report/compare/data/20250305.pdf"
        ],
        policy_info={
            "old_policy_path": "report/compare/data/20241224.pdf",
            "new_policy_path": "report/compare/data/20250305.pdf"
        },
        comparison_result="", 
        
        # RAG 검색 결과를 State에 저장하는 필드
        retrieved_documents=None 
        
        # 🚨 [추가] load_prev_month_report 등이 우회되므로, 혹시 모를 대비책으로 빈 데이터 추가 (선택적)
        # report_data={},
        # house_info={},
        # credit_info={},
    )

    # 3. Agent 실행 및 결과 확인
    print("🚀 Comparing Agent 실행 시작...")
    final_state = compare_agent_flow.invoke(test_state) 
    
    print("\n--- 📄 최종 비교 분석 결과 (LLM Generation) ---")
    print(final_state.get('comparison_result', '결과 필드를 찾을 수 없습니다.')) 
    
    print("\n--- 🔍 RAG 검색 결과 (Retrieval) 확인 ---")
    # 🚨 [수정] final_state에서 'retrieved_documents'를 찾아서 출력합니다.
    retrieved_docs = final_state.get('retrieved_documents', None)
    if retrieved_docs:
        print(f"✅ RAG 검색 성공. 최종 State에 {len(retrieved_docs)}개 청크 확인.")
        # 검색된 청크의 title만 출력하여 확인
        for i, doc in enumerate(retrieved_docs):
            print(f"  [{i+1}] {doc.get('title', '제목 없음')}")
    else:
        # 🚨 [기존 실패 메시지]
        print("검색된 문서 청크(retrieved_documents) 필드를 찾을 수 없거나 비어 있습니다. Policy 로드 및 검색 로직 확인 필요.")