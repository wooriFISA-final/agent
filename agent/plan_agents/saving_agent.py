import pandas as pd
import json
import re
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path # ⬅️ VS Code의 경로 처리를 위해 Path 임포트
import operator
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, END

# LangGraph 상태(State) 정의

class SavingsAgentState(TypedDict):
    user_data: Dict[str, Any]
    csv_file_path: str # csv로 받음

    savings_recommendations: dict


# 2단계: 전역 구성 요소 정의 (LLM, 프롬프트, 함수, 체인)

# 도구 함수 정의: Python 필터링 (수정 없음)
def load_and_filter_products(user_data, csv_path):
    """
    (CSV 버전) 'saving_data.csv'를 로드하고,
    가정된 사용자 데이터(MyData)로 '우대 조건'을 '필터링'하여
    최적의 예금/적금 상품 Top 3를 각각 반환하는 '도구'입니다.
    """
    print(f"--- '필터링 도구' 실행: {user_data['user_id']}님 맞춤 상품 필터링 ---")

    try:
        all_products_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"CSV 로드 중 오류 발생: {e}")
        return {"deposits": pd.DataFrame(), "savings": pd.DataFrame()}

    # (이하 님이 작성한 필터링 로직)
    deposits_df = all_products_df[all_products_df['product_type'] == '예금'].copy()
    deposits_df = deposits_df[deposits_df['condition_min_age'] <= user_data['age']]
    if not user_data['is_first_customer']:
        deposits_df = deposits_df[deposits_df['condition_first_customer'] == False]
    period = user_data['period_goal_months']
    deposits_df = deposits_df[
        (deposits_df['min_term'] <= period) &
        (deposits_df['max_term'] >= period)
    ]
    top_3_deposits = deposits_df.sort_values(by='max_rate', ascending=False).head(3)


    savings_df = all_products_df[all_products_df['product_type'] == '적금'].copy()
    savings_df = savings_df[savings_df['condition_min_age'] <= user_data['age']]
    if not user_data['is_first_customer']:
        savings_df = savings_df[savings_df['condition_first_customer'] == False]
    savings_df = savings_df[
        (savings_df['min_term'] <= period) &
        (savings_df['max_term'] >= period)
    ]
    top_3_savings = savings_df.sort_values(by='max_rate', ascending=False).head(3)

    print("--- '필터링 도구' 실행 완료: 최적 상품 선별 완료 ---")

    return {
        "deposits": top_3_deposits,
        "savings": top_3_savings
    }

# 파서 함수 정의: JSON 파싱
def parse_analysis_result(llm_output: str):
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
        print(f"--- 파싱 오류 ---")
        print(f"LLM 원본 출력 (파싱 전): {llm_output}")
        print(f"오류 내용: {e}")
        return {"error": "분석 결과 파싱에 실패했습니다."}

# LLM 정의
try:
    llm = ChatOllama(model="qwen3:8b")
    print("--- 8. 로컬 Ollama (qwen3:8b) 모델 로드 성공 ---") 
except Exception as e:
    print(f"Ollama 모델 로드 중 오류 발생: {e}")
    print("Ollama 데스크탑 앱이 실행 중인지, 'ollama pull qwen3:8b'가 완료되었는지 확인하세요.")
    exit() 

SAVINGS_SUMMARY_PROMPT = """
[Persona]
당신은 최고의 예/적금 상품 분석가(SavingsAnalyst)입니다. 초보자에게 상품의 핵심 특징을 요약하는 데 특화되어 있습니다.

[Task]
- Python이 이미 선별한 [Top 3 예금 목록]과 [Top 3 적금 목록]을 입력받습니다.
- 각 상품의 'description'을 분석하여, **금융 초보자**가 이해하기 쉬운 **"summary_for_beginner" (한 줄 요약)**를 생성합니다.
- (중요) 입력받은 상품 목록 구조에 'summary_for_beginner' 키(key)만 추가하여 전체 JSON을 [Output Format]에 맞춰 반환합니다.

[Instructions]
1. [Top 3 예금 목록]을 확인합니다.
2. 각 예금 상품의 'description'을 읽고, 'summary_for_beginner'를 생성합니다.
3. [Top 3 적금 목록]에 대해 2번 과정을 동일하게 반복합니다.
4. 모든 분석 결과를 지정된 [Output Format]에 맞춰 정확하게 반환합니다.
5. (주의!) 입력받은 데이터(name, max_rate 등)를 절대 변경하지 말고, 'summary_for_beginner' 필드만 추가하세요.

[Inputs]
(Python이 필터링한 JSON 데이터를 받습니다)
Top 3 예금 목록: {input_top_3_deposits}
Top 3 적금 목록: {input_top_3_savings}

[Output Format (Return this)]
<analysis_result>
{{
  "top_deposits": [
    {{
      "product_type": "예금",
      "name": "WON플러스 예금",
      "max_rate": 3.5,
      "description": "비대면 가입시 누구나 우대금리를 받을 수 있습니다.",
      "summary_for_beginner": "비대면으로 쉽게 가입하고 우대금리를 받을 수 있는 상품입니다."
    }}
  ],
  "top_savings": [
    {{
      "product_type": "적금",
      "name": "청년희망 적금",
      "max_rate": 6.0,
      "description": "만 19~34세 청년 대상 정책형 상품입니다.",
      "summary_for_beginner": "만 19세에서 34세 청년이라면 높은 금리를 받을 수 있는 정책 지원 상품입니다."
    }}
  ]
}}
</analysis_result>
"""
prompt_template = ChatPromptTemplate.from_template(SAVINGS_SUMMARY_PROMPT)

# 체인 생성
chain = prompt_template | llm | StrOutputParser() | parse_analysis_result


# LangGraph 노드함수 정의
def run_savings_recommendation_node(state: SavingsAgentState):
    print("--- [노드 시작] '예/적금 추천 노드' 실행 ---")

    # 1. State에서 입력 받기
    user_data = state['user_data']
    csv_path = state['csv_file_path']

    # 2. '도구' 호출 (방식 1: Python 필터링)
    recommendations = load_and_filter_products(user_data, csv_path)

    # 3. LLM 입력을 위한 데이터 가공
    top_3_deposits_str = recommendations['deposits'].to_json(orient='records', force_ascii=False, indent=2)
    top_3_savings_str = recommendations['savings'].to_json(orient='records', force_ascii=False, indent=2)

    print("--- [노드] LLM 호출 (상품 '요약' 생성 중...) ---")

    # 4. LLM 체인 호출 (요약 생성)
    analysis_result = chain.invoke({
        "input_top_3_deposits": top_3_deposits_str,
        "input_top_3_savings": top_3_savings_str
    })

    print("--- [노드 종료] '예/적금 추천 노드' 완료 ---")

    # 5. State 업데이트 (반환)
    return {"savings_recommendations": analysis_result}


# 4단계: (실행) 그래프 정의 및 호출 (VS Code 로컬 실행용)
if __name__ == "__main__":
    
    # 4-1. 그래프 정의
    workflow = StateGraph(SavingsAgentState)

    # 4-2. 노드 추가
    workflow.add_node("recommend_savings", run_savings_recommendation_node)

    # 4-3. 엣지 추가
    workflow.set_entry_point("recommend_savings")
    workflow.add_edge("recommend_savings", END)

    # 4-4. 그래프 컴파일
    app = workflow.compile()

    # 4-5. (입력) 사용자의 MyData 가정
    user_data_input = {
        "user_id": "kim_woori",
        "age": 32,
        "is_first_customer": False,
        "period_goal_months": 12
    }

    # 'agent/plan_agents'에 있다고 가정
    current_script_path = Path(__file__).resolve()
    # agent/plan_agents -> agent -> FINAL_PROJECT
    project_root = current_script_path.parents[2] 
    # 'FINAL_PROJECT/saving_data.csv'
    file_path_to_run = project_root / 'saving_data.csv' 

    # 그래프의 '초기 상태' 정의
    initial_state = {
        "user_data": user_data_input,
        "csv_file_path": str(file_path_to_run), 
        "savings_recommendations": {} 
    }

    print(f"--- 9. 사용자 데이터 정의 완료: {user_data_input['user_id']}님 (나이: 32, 첫 고객 아님, 12개월 희망) ---")
    print(f"--- 9-1. CSV 파일 경로: {file_path_to_run} ---")
    print("\n--- 🏁 (LangGraph) 예/적금 추천 그래프 실행 시작 🏁 ---")

    # 4-8. 그래프 실행
    final_state = app.invoke(initial_state)

    # 4-9. 최종 결과 출력
    print("\n--- 🏁 (LangGraph) 그래프 실행 완료 🏁 ---")
    print("최종 추천 결과 (JSON):")
    print(json.dumps(final_state['savings_recommendations'], indent=2, ensure_ascii=False))