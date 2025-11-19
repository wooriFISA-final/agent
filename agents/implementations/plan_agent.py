import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path 
import operator
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage # ⬅️ 전체 State 호환용

# ----------------------------------------------------------------------
# 1단계: (필수) LangGraph '통합' 상태 정의
# (이 정의는 'plan_graph.py' 파일의 정의와 100% 동일해야 합니다)
# ----------------------------------------------------------------------
class AgentGraphState(TypedDict):
    """
    그래프 전체를 흐르는 공용 메모리
    (이 노드는 'loan/savings/fund' 3개의 키를 읽고 'final_plan' 1개의 키를 씁니다)
    """
    # (Input/Intent)
    user_id: Optional[int]
    messages: Annotated[List[BaseMessage], operator.add] 
    intent: Optional[str]
    
    # (파일 경로)
    fund_data_path: Optional[str]
    savings_data_path: Optional[str]
    
    # (Flags)
    input_completed: Optional[bool]
    validation_passed: Optional[bool]
    
    # (Data)
    plan_input_data: Optional[Dict[str, Any]]
    plan_id: Optional[int]
    user_mydata: Optional[Dict[str, Any]]
    
    # (Worker 노드 출력 - 이 노드의 입력)
    loan_recommendations: Optional[Dict[str, Any]]
    savings_recommendations: Optional[Dict[str, Any]]
    fund_analysis_result: Optional[Dict[str, Any]]
    
    # (이 노드의 출력)
    final_plan: Optional[Dict[str, Any]]
    error_message: Optional[str]

# ----------------------------------------------------------------------
# 2단계: '전역' 프롬프트 템플릿 정의
# (님이 요청한대로, 요약이 아닌 '상품 목록 전체'를 포함)
# ----------------------------------------------------------------------
PLANNER_PROMPT = """
[Persona]
당신은 고객의 모든 금융 데이터를 취합하는 마스터 재무 설계사(PlanAgent)입니다.

[Task]
- 3개의 하위 에이전트가 추천한 JSON 상품 목록을 모두 입력받습니다.
- 이 상품 목록들을 '요약하지 말고' **그대로 최종 결과에 포함**시킵니다.
- 이 상품들을 어떻게 조합하면 좋을지 'final_recommendation' 텍스트를 생성합니다.

[Instructions]
1. 3개의 입력 JSON([Loan Recommendations], [Savings Recommendations], [Fund Recommendations])을 확인합니다.
2. 이 3개의 JSON 객체를 **하나의 새로운 JSON 객체로 통합**합니다.
3. (중요) 'final_recommendation'이라는 'key'를 새로 만들고, 고객을 위한 **종합 추천사**를 텍스트로 작성합니다. (예: "고객님의 목표를 위해, [대출 상품명]으로 1억을 확보하고, '청년희망 적금'에 70%, 'AI 반도체 펀드'에 30%를 투자하는 플랜을 추천합니다.")
4. 모든 결과를 [Output Format]에 맞춰 정확하게 반환합니다.

[Inputs]
Loan Recommendations (JSON): {input_loan_json}
Savings Recommendations (JSON): {input_savings_json}
Fund Recommendations (JSON): {input_fund_json}

[Output Format (Return this)]
<analysis_result>
{{
  "loan_recommendations": {{
    "recommended_loan": {{ "name": "...", "max_amount": 100000000, "summary_for_beginner": "..." }},
    "available_loan_amount": 100000000
  }},
  "savings_recommendations": {{
    "top_deposits": [ {{ "name": "...", "max_rate": 3.5, "summary_for_beginner": "..." }} ],
    "top_savings": [ {{ "name": "...", "max_rate": 6.0, "summary_for_beginner": "..." }} ]
  }},
  "fund_recommendations": {{
    "recommendations": [ {{ "risk_level": "높은 위험", "summary_for_beginner": "..." }} ]
  }},
  "final_recommendation": "고객님의 데이터를 종합 분석한 결과, [대출 상품]으로 1억을 확보하고, 확보된 자금과 월 저축액을 '청년희망 적금'에 70%, 'AI 반도체 펀드'에 30%로 나누어 투자하는 플랜을 추천드립니다."
}}
</analysis_result>
"""

# ----------------------------------------------------------------------
# 3단계: (핵심) LangGraph '노드' 클래스 정의
# ----------------------------------------------------------------------
class PlanAgentNode:

    def __init__(self):
        """
        클래스가 생성될 때 LLM, 프롬프트, 체인을 한 번만 초기화합니다.
        """
        print("--- PlanAgentNode 초기화 ---")
        try:
            # 3-1. LLM 정의
            self.llm = ChatOllama(model="qwen3:8b") 
            print("--- (PlanAgent) 로컬 Ollama (qwen3:8b) 모델 로드 성공 ---")
        except Exception as e:
            print(f"Ollama 모델 로드 중 오류 발생: {e}")
            exit() 

        # 3-2. 프롬프트 템플릿 정의
        self.prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)

        # 3-3. 체인 생성 (파서 함수를 클래스 메서드로 참조)
        self.chain = self.prompt_template | self.llm | StrOutputParser() | self._parse_analysis_result
        
        print("--- (PlanAgent) LLM 체인 구성 완료 ---")

    # --- 3-4. '파서'를 클래스 내부 메서드로 정의 ---
    def _parse_analysis_result(self, llm_output: str):
        """
        LLM의 출력이 <analysis_result>, ```json (백틱),
        '''json (작은따옴표) 등 어떤 형식이든 처리하는 파서
        """
        try:
            if "```json" in llm_output:
                result_str = llm_output.split("```json")[1].split("```")[0].strip()
            elif "'''json" in llm_output:
                result_str = llm_output.split("'''json")[1].split("'''")[0].strip()
            elif "<analysis_result>" in llm_output:
                result_str = llm_output.split("<analysis_result>")[1].split("</analysis_result>")[0].strip()
            elif llm_output.strip().startswith('{') and llm_output.strip().endswith('}'):
                 result_str = llm_output.strip()
            else:
                 raise ValueError("LLM의 출력에서 유효한 JSON 마커(```, ''', <>)를 찾지 못했습니다.")
            return json.loads(result_str)
        except Exception as e:
            print(f"--- (PlanAgent) 파싱 오류 ---")
            print(f"LLM 원본 출력 (파싱 전): {llm_output}")
            print(f"오류 내용: {e}")
            return {"error": "PlanAgent 파싱에 실패했습니다."}

    # --- 3-5. LangGraph '노드' 실행 함수 ---
    def run(self, state: AgentGraphState):
        """
        이 함수가 LangGraph에 '노드'로 등록될 실제 실행 함수입니다.
        (대출, 예/적금, 펀드 노드의 결과를 취합합니다)
        """
        print("--- [노드 시작] '최종 플랜 에이전트' 실행 ---")
        
        # 1. State에서 모든 하위 노드의 결과 입력 받기
        # (만약 이전 노드가 실패했다면, error 객체를 그대로 전달)
        loan_json = state.get('loan_recommendations', {"error": "대출 정보 없음"})
        savings_json = state.get('savings_recommendations', {"error": "예/적금 정보 없음"})
        fund_json = state.get('fund_analysis_result', {"error": "펀드 정보 없음"})

        # 2. LLM 입력을 위한 JSON 문자열로 변환
        loan_str = json.dumps(loan_json, ensure_ascii=False)
        savings_str = json.dumps(savings_json, ensure_ascii=False)
        fund_str = json.dumps(fund_json, ensure_ascii=False)

        print("--- [노드] LLM 호출 (최종 계획 생성 중...) ---")

        # 3. .invoke()를 사용하여 체인 실행 (클래스 내부 체인 호출)
        analysis_result = self.chain.invoke({
            "input_loan_json": loan_str,
            "input_savings_json": savings_str,
            "input_fund_json": fund_str
        })

        print("--- [노드 종료] '최종 플랜 에이전트' 완료 ---")
        
        # 4. State 업데이트 (반환)
        # (이것이 그래프의 최종 출력이 됩니다)
        return {"final_plan": analysis_result}

# ----------------------------------------------------------------------
# 4단계: (테스트) VS Code에서 이 파일만 단독으로 실행
# (python agent/plan_agents/plan_agent.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. 클래스를 인스턴스화
    plan_agent_node = PlanAgentNode()

    # 2. (가상) 이전 노드들이 'state'에 저장했을 데이터 (Mock Data)
    mock_state = {
        "loan_recommendations": {
            "recommended_loan": { "name": "테스트 대출", "max_amount": 10000, "interest_rate": "5.0%" },
            "available_loan_amount": 10000
        },
        "savings_recommendations": {
            "top_deposits": [ { "name": "테스트 예금", "max_rate": 4.0, "summary_for_beginner": "좋은 예금" } ],
            "top_savings": [ { "name": "테스트 적금", "max_rate": 5.0, "summary_for_beginner": "좋은 적금" } ]
        },
        "fund_analysis_result": {
            "recommendations": [ { "risk_level": "높은 위험", "product_name": "테스트 펀드", "summary_for_beginner": "좋은 펀드" } ]
        }
        # (AgentGraphState의 다른 키들은 이 노드가 사용하지 않으므로 생략)
    }

    print("\n--- 🏁 (단독 테스트) PlanAgentNode.run() 실행 시작 🏁 ---")
    
    # 3. 노드의 'run' 메서드 직접 호출
    result_dict = plan_agent_node.run(mock_state)

    # 4. 최종 결과 출력
    print("\n--- 🏁 (단독 테스트) 실행 완료 🏁 ---")
    print("최종 플랜 결과 (JSON):")
    print(json.dumps(result_dict['final_plan'], indent=2, ensure_ascii=False))