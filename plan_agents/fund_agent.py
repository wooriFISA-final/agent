<<<<<<< HEAD
import os
=======
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
<<<<<<< HEAD
from pathlib import Path
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END


# ============================================================
# 1️⃣ LangGraph 상태 정의 (State)
# ============================================================
=======
from pathlib import Path 
import operator
from typing import TypedDict, Annotated, Dict, Any 
from langgraph.graph import StateGraph, END 


# LangGraph 상태 정의
# (클래스 밖에 정의하여 그래프 전체에서 공유)
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
class FundAgentState(TypedDict):
    fund_data_path: str
    fund_analysis_result: dict


<<<<<<< HEAD
# ============================================================
# 2️⃣ LLM 프롬프트 템플릿
# ============================================================
FUND_RECOMMENDATION_PROMPT_V4_KO = """
[Persona]
당신은 '우리은행'의 최고 펀드 전문가(FundAgent)입니다.
금융 초보자에게 객관적인 데이터를 바탕으로 펀드를 추천하는 데 특화되어 있습니다.

[Core Logic]
- '최종_종합품질점수'는 펀드매니저가 과거 1~3년간 얼마나 꾸준히 "위험 대비 자산을 잘 운용했는가"를 평가한 '운용 기술 성적표'입니다.
- 이 점수가 높다는 것은 변동성, 보수, 규모 등을 종합적으로 고려할 때 '현재' 가장 잘 운용되고 있음을 의미합니다.
- 과거 성과가 미래를 보장하진 않지만, 좋은 성적표를 받은 펀드가 앞으로도 잘할 확률이 높다는 논리로 고객을 설득합니다.

[Task]
1. [고객 정보]와 Python이 선별한 [추천 펀드 목록]을 입력받습니다.
   (이 목록은 고객 성향에 맞는 *각 위험 등급별*로 최대 2개씩 선별된 결과입니다.)
2. 각 펀드는 '종합 점수'와 함께 고객이 이해하기 쉬운 4가지 **[핵심_근거_지표]**를 포함하고 있습니다.
3. 각 펀드의 [핵심_근거_지표]를 활용하여, 이 펀드가 왜 '종합품질점수'가 높은지(왜 잘 운용되고 있는지) [Core Logic]에 기반한 **"추천_이유"**를 생성합니다.
4. (중요) '추천_이유'는 반드시 **구체적인 근거(숫자)를 1~2개 포함**하여 초보자가 신뢰할 수 있도록 작성합니다.
5. (중요) 입력받은 상품 목록 구조에 '추천_이유' 키(key)만 추가하여 전체 JSON을 [Output Format]에 맞춰 반환합니다.

[Inputs]
고객 정보:
- user_id: {user_id}
- 리스크 성향: {user_risk_level}

추천 펀드 목록 (Python이 고객 성향 및 등급별 Top 2 기준으로 1차 필터링 완료):
{input_top_n_funds}

[Output Format 예시] (모든 키는 반드시 한국어로 작성)
(예시: '적극투자형' 고객이 1, 2등급에서 2개씩 추천받은 경우)
<analysis_result>
{{
  "추천_펀드_목록": [
    {{
      "펀드명": "(1등급) 삼성 글로벌반도체증권자투자신탁UH[주식]_S-P",
      "위험등급": "매우 높은 위험",
      "상품_설명": "이 투자신탁은 반도체 관련 글로벌 주식에 투자하여...",
      "최종_종합품질점수": 0.95,
      "성과_점수": 1.10,
      "안정성_점수": 0.80,
      "핵심_근거_지표": {{
        "1년_수익률": 15.2,
        "최대_손실_낙폭": -8.5,
        "총보수": 0.45,
        "운용_규모(억)": 1200.0
      }},
      "추천_이유": "고객님의 '적극투자형' 성향에 맞는 '매우 높은 위험(1등급)' 상품 중 종합 품질 1위입니다. **최근 1년 수익률이 15.2%**로 우수하고 **총보수도 0.45%**로 낮아 효율적인 운용 성적표를 보여주고 있습니다."
    }},
    {{
      "펀드명": "(1등급) KB스타 글로벌AI&반도체증권투자신탁(주식) S-P",
      "위험등급": "매우 높은 위험",
      "상품_설명": "AI 및 반도체 관련 글로벌 주식에 투자합니다...",
      "최종_종합품질점수": 0.92,
      "성과_점수": 1.05,
      "안정성_점수": 0.79,
      "핵심_근거_지표": {{
        "1년_수익률": 14.8,
        "최대_손실_낙폭": -9.1,
        "총보수": 0.50,
        "운용_규모(억)": 800.0
      }},
      "추천_이유": "'매우 높은 위험(1등급)' 상품 중 종합 품질 2위 펀드입니다. **14.8%의 높은 1년 수익률** 대비 '최대 손실 낙폭' 관리가 우수하여 좋은 '운용 성적표'를 받았습니다."
    }},
    {{
      "펀드명": "(2등급) 미래에셋 미국S&P500인덱스증권자투자신탁(주식)",
      "위험등급": "높은 위험",
      "상품_설명": "미국 S&P500 지수를 추종하는 인덱스 펀드입니다...",
      "최종_종합품질점수": 0.89,
      "성과_점수": 0.85,
      "안정성_점수": 0.93,
      "핵심_근거_지표": {{
        "1년_수익률": 12.1,
        "최대_손실_낙폭": -7.2,
        "총보수": 0.25,
        "운용_규모(억)": 3500.0
      }},
      "추천_이유": "고객님 성향에 맞는 '높은 위험(2등급)' 상품 중 1위입니다. **총 3,500억 원**의 안정적인 운용 규모와 **0.25%**라는 매우 낮은 보수가 큰 장점입니다."
    }},
    {{
      "펀드명": "(2등급) 한국투자 미국S&P500증권투자신탁(주식)(C-Pe)",
      "위험등급": "높은 위험",
      "상품_설명": "S&P500 지수를 추종하며 환헤지를 실행하는...",
      "최종_종합품질점수": 0.88,
      "성과_점수": 0.84,
      "안정성_점수": 0.92,
      "핵심_근거_지표": {{
        "1년_수익률": 11.9,
        "최대_손실_낙폭": -7.0,
        "총보수": 0.30,
        "운용_규모(억)": 2100.0
      }},
      "추천_이유": "'높은 위험(2등급)' 상품 중 2위 펀드입니다. **연 0.30%의 낮은 보수**와 안정적인 '최대 손실 낙폭' 관리를 바탕으로 꾸준히 우수한 종합 품질 점수를 유지하고 있습니다."
=======
# '전역' 프롬프트 템플릿
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
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
    }}
  ]
}}
</analysis_result>
"""

<<<<<<< HEAD

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
=======
# LangGraph 노드 클래스 정의
class FundAgentNode:

    def __init__(self):
        """
        클래스가 생성될 때 LLM, 프롬프트, 체인을 한 번만 초기화합니다.
        """
        print("--- FundAgentNode 초기화 ---")
        try:
            # LLM 정의
            self.llm = ChatOllama(model="qwen3:8b") 
            print("--- 8. 로컬 Ollama (qwen3:8b) 모델 로드 성공 ---")
        except Exception as e:
            print(f"Ollama 모델 로드 중 오류 발생: {e}")
            print("Ollama 데스크탑 앱이 실행 중인지, 'ollama pull qwen3:8b'가 완료되었는지 확인하세요.")
            exit() 

        # 프롬프트 템플릿 정의
        self.prompt_template = ChatPromptTemplate.from_template(FUND_ANALYST_PROMPT)

        # 체인 생성
        self.chain = self.prompt_template | self.llm | StrOutputParser() | self._parse_analysis_result
        
        print("--- LLM 체인 구성 완료 ---")

    # '파서'를 클래스 내부 메서드로 정의
    def _parse_analysis_result(self, llm_output: str):
        """
        LLM의 출력이 <analysis_result>, ```json (백틱),
        '''json (작은따옴표) 등 어떤 형식이든 처리하는 파서
        """
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
        try:
            if "```json" in llm_output:
                result_str = llm_output.split("```json")[1].split("```")[0].strip()
            elif "'''json" in llm_output:
                result_str = llm_output.split("'''json")[1].split("'''")[0].strip()
            elif "<analysis_result>" in llm_output:
                result_str = llm_output.split("<analysis_result>")[1].split("</analysis_result>")[0].strip()
<<<<<<< HEAD
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
=======
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

    # LangGraph 노드 실행 함수
    def run(self, state: FundAgentState):
        """
        이 함수가 LangGraph에 '노드'로 등록될 실제 실행 함수입니다.
        """
        print("--- [노드 시작] '펀드 분석 노드' 실행 ---")
        
        # State에서 파일 경로 입력 받기 추후 DB끌어오는 걸로 수정
        file_path = state['fund_data_path']

        # 파일 로드
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_fund_data = json.load(f)
            print(f"--- 9. {file_path} 파일 로드 성공 ---")
        except FileNotFoundError:
            print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
            return {"fund_analysis_result": {"error": f"File not found: {file_path}"}}
        except json.JSONDecodeError:
            print(f"오류: {file_path} 파일이 올바른 JSON 형식이 아닙니다.")
            return {"fund_analysis_result": {"error": f"JSON decode error in file: {file_path}"}}
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            return {"fund_analysis_result": {"error": f"File loading error: {e}"}}

        # LLM 입력 데이터 가공
        print("--- 펀드 분석 에이전트 실행 (로컬 PC로 연산 중...) ---")
        fund_data_str = json.dumps(raw_fund_data, indent=2, ensure_ascii=False)

        # .invoke()를 사용하여 체인 실행 (클래스 내부 체인 호출)
        analysis_result = self.chain.invoke({"input_data": fund_data_str})

        print("--- [노드 종료] '펀드 분석 노드' 완료 ---")
        
        # State 업데이트 (반환)
        return {"fund_analysis_result": analysis_result}


# 4단계: (실행) 그래프 정의 및 호출 (VS Code 로컬 실행용)

if __name__ == "__main__":
    
    # 클래스를 인스턴스화
    fund_agent_node = FundAgentNode()

    # 그래프 정의
    workflow = StateGraph(FundAgentState)

    # 노드 추가 (클래스의 run 메서드를 등록)
    workflow.add_node("analyze_funds", fund_agent_node.run)

    # 엣지 추가
    workflow.set_entry_point("analyze_funds")
    workflow.add_edge("analyze_funds", END)

    # 그래프 컴파일
    app = workflow.compile()

    # 파일 경로 설정
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[2] 
    file_path_to_run = project_root / 'fund_data.json'

    # 그래프의 '초기 상태' 정의
    initial_state = {
        "fund_data_path": str(file_path_to_run), 
        "fund_analysis_result": {}
    }

    print("\n--- 🏁 (LangGraph) 펀드 분석 그래프 실행 시작 🏁 ---")
    
    # 4-8. 그래프 실행
    final_state = app.invoke(initial_state)

    # 4-9. 최종 결과 출력
    print("\n--- 🏁 (LangGraph) 그래프 실행 완료 🏁 ---")
    print("최종 분석 결과 (JSON):")
    print(json.dumps(final_state['fund_analysis_result'], indent=2, ensure_ascii=False))
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
