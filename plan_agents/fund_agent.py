import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END


# ============================================================
# 1️⃣ LangGraph 상태 정의 (State)
# ============================================================
class FundAgentState(TypedDict):
    fund_data_path: str
    fund_analysis_result: dict


# ============================================================
# 2️⃣ LLM 프롬프트 템플릿
# ============================================================
FUND_ANALYST_PROMPT = """
[Persona]
당신은 최고의 펀드 상품 분석가(FundAnalyst)입니다. 특히 금융 초보자에게 복잡한 상품을 매우 쉽고 명확하게 설명하는 데 특화되어 있습니다.

[Task]
- 입력받은 [Raw Fund Data]를 분석하여, 각 '리스크 레벨'별로 '예상 수익률'이 가장 높은 상품 1개씩을 선별합니다.
- 선별된 각 상품의 설명('description')을 초보자가 즉시 이해할 수 있도록 간결하게 요약합니다.

[Instructions]
1. 입력받은 [Raw Fund Data] 목록 전체를 확인합니다.
2. 펀드 목록을 'risk_level' (예: '높은 위험', '중간 위험', '낮은 위험') 별로 그룹화합니다.
3. 각 리스크_레벨 그룹 내에서 'expected_return'(예상 수익률)이 가장 높은 상품을 **단 하나만** 선정합니다.
4. (중요) 선정된 각 상품의 'description'(설명 원문)을 분석하여, **금융 초보자**가 이해하기 쉬운 단어로 핵심 내용(어디에 투자하는지, 목표는 무엇인지)을 요약합니다.
5. 모든 분석 결과를 지정된 [Output Format]에 맞춰 정확하게 반환합니다.

[Raw Fund Data (Input)]
{input_data}

[Output Format (Return this)]
<analysis_result>
{{
  "recommendations": [
    {{
      "risk_level": "높은 위험",
      "product_name": "예시 펀드 A",
      "expected_return": "12.5%",
      "summary_for_beginner": "AI와 반도체처럼 빠르게 성장하는 기술 기업에 집중 투자합니다."
    }}
  ]
}}
</analysis_result>
"""


# ============================================================
# 3️⃣ FundAgentNode 클래스 정의
# ============================================================
class FundAgentNode:
    def __init__(self):
        print("--- FundAgentNode 초기화 ---")
        try:
            self.llm = ChatOllama(model="qwen3:8b")
            print("--- ✅ 로컬 Ollama 모델(qwen3:8b) 로드 성공 ---")
        except Exception as e:
            print(f"❌ Ollama 모델 로드 실패: {e}")
            print("⚠️ 'ollama pull qwen3:8b' 명령으로 모델을 설치하세요.")
            exit()

        # 프롬프트 템플릿과 체인 구성
        self.prompt_template = ChatPromptTemplate.from_template(FUND_ANALYST_PROMPT)
        self.chain = self.prompt_template | self.llm | StrOutputParser() | self._parse_analysis_result
        print("--- ✅ LLM 체인 구성 완료 ---")

    # ----------------------------------------------------------
    # 🔹 LLM 결과 파싱
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
                raise ValueError("LLM의 출력에서 유효한 JSON 형식을 찾지 못했습니다.")
            return json.loads(result_str)
        except Exception as e:
            print(f"⚠️ 파싱 실패: {e}")
            print(f"LLM 원본 출력:\n{llm_output}")
            return {"error": "분석 결과 파싱 실패"}

    # ----------------------------------------------------------
    # 🔹 LangGraph 노드 실행 함수
    # ----------------------------------------------------------
    def run(self, state: FundAgentState):
        print("\n--- [노드 시작] '펀드 분석 노드' 실행 ---")

        # ✅ 안전하게 파일 경로 확인 및 기본 경로 설정
        file_path = state.get("fund_data_path")
        if not file_path or not os.path.exists(file_path):
            print("⚠️ fund_data_path가 전달되지 않았거나 존재하지 않습니다. 기본 경로를 사용합니다.")
            file_path = "/Users/yoodongseok/Desktop/WooriAgent/agent/fund_data.json"

        # ✅ 파일 로드
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_fund_data = json.load(f)
            print(f"--- ✅ 펀드 데이터 로드 성공: {file_path} ---")
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return {"fund_analysis_result": {"error": f"File not found: {file_path}"}}
        except json.JSONDecodeError:
            print(f"❌ JSON 형식 오류: {file_path}")
            return {"fund_analysis_result": {"error": f"Invalid JSON: {file_path}"}}
        except Exception as e:
            print(f"❌ 파일 로드 중 오류 발생: {e}")
            return {"fund_analysis_result": {"error": str(e)}}

        # ✅ LLM 입력 데이터 준비
        print("--- 펀드 데이터 분석 시작 ---")
        fund_data_str = json.dumps(raw_fund_data, indent=2, ensure_ascii=False)

        # ✅ LLM 체인 실행
        analysis_result = self.chain.invoke({"input_data": fund_data_str})

        print("--- [노드 종료] '펀드 분석 노드' 완료 ---")
        return {"fund_analysis_result": analysis_result}


# ============================================================
# 4️⃣ VS Code 로컬 실행 (단독 테스트용)
# ============================================================
if __name__ == "__main__":
    fund_agent_node = FundAgentNode()

    # 그래프 구성
    workflow = StateGraph(FundAgentState)
    workflow.add_node("analyze_funds", fund_agent_node.run)
    workflow.set_entry_point("analyze_funds")
    workflow.add_edge("analyze_funds", END)
    app = workflow.compile()

    # 절대경로 지정 ✅
    file_path_to_run = "/Users/yoodongseok/Desktop/WooriAgent/fund_data.json"

    # 초기 상태 정의 ✅
    initial_state = {
        "fund_data_path": file_path_to_run,
        "fund_analysis_result": {},
    }

    print("\n--- 🏁 (LangGraph) 펀드 분석 그래프 실행 시작 🏁 ---")
    print(f"🔹 입력 경로: {file_path_to_run}")

    # 그래프 실행
    final_state = app.invoke(initial_state)

    print("\n--- 🏁 (LangGraph) 그래프 실행 완료 🏁 ---")
    print(json.dumps(final_state["fund_analysis_result"], indent=2, ensure_ascii=False))
