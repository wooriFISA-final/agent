import pandas as pd
import json
import re
import time
import os
from pathlib import Path
from typing import TypedDict, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


# ============================================================
# 1️⃣ LangGraph 상태 정의 (State)
# ============================================================
class SavingsAgentState(TypedDict):
    user_data: Dict[str, Any]
    csv_file_path: str
    savings_recommendations: dict


# ============================================================
# 2️⃣ LLM 프롬프트 템플릿 (전역 상수)
# ============================================================
SAVINGS_SUMMARY_PROMPT = """
[Persona]
당신은 최고의 예/적금 상품 분석가(SavingsAgent)입니다. 초보자에게 상품의 핵심 특징을 요약하는 데 특화되어 있습니다.

[Task]
- Python이 이미 선별한 [Top 3 예금 목록]과 [Top 3 적금 목록]을 입력받습니다.
- 각 상품의 'description'을 분석하여, **금융 초보자**가 이해하기 쉬운 **"summary_for_beginner" (한 줄 요약)**을 생성합니다.
- (중요) 입력받은 상품 목록 구조에 'summary_for_beginner' 키(key)만 추가하여 전체 JSON을 [Output Format]에 맞춰 반환합니다.

[Instructions]
1. [Top 3 예금 목록]을 확인합니다.
2. 각 예금 상품의 'description'을 읽고, 'summary_for_beginner'를 생성합니다.
3. [Top 3 적금 목록]에 대해 2번 과정을 동일하게 반복합니다.
4. 모든 분석 결과를 지정된 [Output Format]에 맞춰 정확하게 반환합니다.
5. (주의!) 입력받은 데이터(name, max_rate 등)를 절대 변경하지 말고, 'summary_for_beginner' 필드만 추가하세요.

[Inputs]
Top 3 예금 목록: {input_top_3_deposits}
Top 3 적금 목록: {input_top_3_savings}

[Output Format]
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


# ============================================================
# 3️⃣ SavingAgentNode 클래스 정의
# ============================================================
class SavingAgentNode:

    def __init__(self):
        print("--- SavingAgentNode 초기화 ---")
        try:
            self.llm = ChatOllama(model="qwen3:8b")
            print("--- ✅ 로컬 Ollama 모델(qwen3:8b) 로드 성공 ---")
        except Exception as e:
            print(f"❌ Ollama 모델 로드 중 오류: {e}")
            print("⚠️ Ollama 앱이 실행 중인지, 'ollama pull qwen3:8b'가 완료되었는지 확인하세요.")
            exit()

        # 프롬프트 체인 구성
        self.prompt_template = ChatPromptTemplate.from_template(SAVINGS_SUMMARY_PROMPT)
        self.chain = self.prompt_template | self.llm | StrOutputParser() | self._parse_analysis_result
        print("--- ✅ LLM 체인 구성 완료 ---")

    # ----------------------------------------------------------
    # 🔹 CSV 로드 & 필터링 로직
    # ----------------------------------------------------------
    def _load_and_filter_products(self, user_data, csv_path):
        print(f"--- [필터링 도구] 실행: {user_data.get('user_id', 'Unknown')}님 맞춤 상품 필터링 ---")

        try:
            all_products_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ CSV 로드 실패 ({csv_path}): {e}")
            return {"deposits": pd.DataFrame(), "savings": pd.DataFrame()}

        # 필터링 로직
        deposits_df = all_products_df[all_products_df['product_type'] == '예금'].copy()
        deposits_df = deposits_df[deposits_df['condition_min_age'] <= user_data.get('age', 0)]
        if not user_data.get('is_first_customer', True):
            deposits_df = deposits_df[deposits_df['condition_first_customer'] == False]
        period = user_data.get('period_goal_months', 12)
        deposits_df = deposits_df[
            (deposits_df['min_term'] <= period) & (deposits_df['max_term'] >= period)
        ]
        top_3_deposits = deposits_df.sort_values(by='max_rate', ascending=False).head(3)

        savings_df = all_products_df[all_products_df['product_type'] == '적금'].copy()
        savings_df = savings_df[savings_df['condition_min_age'] <= user_data.get('age', 0)]
        if not user_data.get('is_first_customer', True):
            savings_df = savings_df[savings_df['condition_first_customer'] == False]
        savings_df = savings_df[
            (savings_df['min_term'] <= period) & (savings_df['max_term'] >= period)
        ]
        top_3_savings = savings_df.sort_values(by='max_rate', ascending=False).head(3)

        print("--- ✅ 상품 필터링 완료 (예금/적금 Top3 선별) ---")

        return {"deposits": top_3_deposits, "savings": top_3_savings}

    # ----------------------------------------------------------
    # 🔹 LLM 분석 결과 파서
    # ----------------------------------------------------------
    def _parse_analysis_result(self, llm_output: str):
        try:
            if "```json" in llm_output:
                result_str = llm_output.split("```json")[1].split("```")[0].strip()
            elif "'''json" in llm_output:
                result_str = llm_output.split("'''json")[1].split("'''")[0].strip()
            elif "<analysis_result>" in llm_output:
                result_str = llm_output.split("<analysis_result>")[1].split("</analysis_result>")[0].strip()
            elif llm_output.strip().startswith("{") and llm_output.strip().endswith("}"):
                result_str = llm_output.strip()
            else:
                raise ValueError("LLM 출력에서 유효한 JSON 구간을 찾지 못했습니다.")
            return json.loads(result_str)
        except Exception as e:
            print(f"⚠️ 파싱 실패: {e}")
            print(f"LLM 원본 출력:\n{llm_output}")
            return {"error": "분석 결과 파싱 실패"}

    # ----------------------------------------------------------
    # 🔹 LangGraph 노드 실행 함수
    # ----------------------------------------------------------
    def run(self, state: SavingsAgentState):
        print("\n--- [노드 시작] 예/적금 추천 노드 실행 ---")

        user_data = state.get("user_data", {})
        csv_path = state.get("csv_file_path")

        # ✅ 안전 처리: csv_file_path가 없으면 기본 경로 사용
        if not csv_path or not os.path.exists(csv_path):
            print("⚠️ csv_file_path가 전달되지 않았거나 존재하지 않습니다. 기본 파일로 대체합니다.")
            default_path = Path(__file__).resolve().parents[2] / "data" / "saving_data.csv"
            csv_path = str(default_path)

        # CSV 기반 상품 필터링
        recommendations = self._load_and_filter_products(user_data, csv_path)

        # JSON 문자열 변환
        top_3_deposits_str = recommendations["deposits"].to_json(orient="records", force_ascii=False, indent=2)
        top_3_savings_str = recommendations["savings"].to_json(orient="records", force_ascii=False, indent=2)

        print("--- [노드] LLM 호출 중... (상품 요약 생성) ---")
        analysis_result = self.chain.invoke({
            "input_top_3_deposits": top_3_deposits_str,
            "input_top_3_savings": top_3_savings_str,
        })

        print("--- [노드 종료] 예/적금 추천 완료 ---")
        return {"savings_recommendations": analysis_result}


# ============================================================
# 4️⃣ VS Code / 로컬 실행 진입점
# ============================================================
if __name__ == "__main__":
    saving_agent_node = SavingAgentNode()

    workflow = StateGraph(SavingsAgentState)
    workflow.add_node("recommend_savings", saving_agent_node.run)
    workflow.set_entry_point("recommend_savings")
    workflow.add_edge("recommend_savings", END)
    app = workflow.compile()

    user_data_input = {
        "user_id": "kim_woori",
        "age": 32,
        "is_first_customer": False,
        "period_goal_months": 12,
    }

    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[2]
    csv_path = "/Users/yoodongseok/Desktop/WooriAgent/saving_data.csv"

    initial_state = {
        "user_data": user_data_input,
        "csv_file_path": str(csv_path),
        "savings_recommendations": {},
    }

    print(f"\n--- 🏁 예/적금 추천 그래프 실행 시작 ---")
    print(f"CSV 경로: {csv_path}")

    final_state = app.invoke(initial_state)

    print("\n--- 🏁 그래프 실행 완료 ---")
    print(json.dumps(final_state["savings_recommendations"], indent=2, ensure_ascii=False))
