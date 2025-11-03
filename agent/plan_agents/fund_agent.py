import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path 
from typing import TypedDict, Annotated, Dict, Any 
from langgraph.graph import StateGraph, END 

# 펀드 에이전트

# LangGraph 상태(State) 정의
class FundAgentState(TypedDict):
    fund_data_path: str
    
    # 이 노드가 실행된 후 상태에 추가할 데이터
    # (Annotated를 사용하면, 기존 결과에 새로운 결과를 '추가'할 수 있음)
    fund_analysis_result: dict


# 전역 구성 요소 정의

# qwen3:8b 모델 로드
try:
    llm = ChatOllama(model="qwen3:8b") 
    print("--- 8. 로컬 Ollama (qwen3:8b) 모델 로드 성공 ---")
except Exception as e:
    print(f"Ollama 모델 로드 중 오류 발생: {e}")
    print("Ollama 데스크탑 앱이 실행 중인지, 'ollama pull qwen3:8b'가 완료되었는지 확인하세요.")
    exit() 

# 프롬프트 템플릿 (수정 없음)
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
4. (중요) 선정된 각 상품의 'description'(설명 원문)을 분석하여, **금융 초보자**가 이해하기 쉬운 단어로 핵심 내용(어디에 투자하는지, 목표는 무엇인지)을 요약합니다. 전문 용어 사용을 최소화해야 합니다.
5. 모든 분석 결과를 지정된 [Output Format]에 맞춰 정확하게 반환합니다.

[Raw Fund Data (Input)]
{input_data}

[Output Format (Return this)]
<analysis_result>
(JSON 형식의 분석 결과를 여기에 삽입)
</analysis_result>

"""
prompt_template = ChatPromptTemplate.from_template(FUND_ANALYST_PROMPT)

# 파서
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

# 체인 구성
chain = prompt_template | llm | StrOutputParser() | parse_analysis_result


# LangGraph 노드 함수 정의
def run_fund_analysis_node(state: FundAgentState):
    print("--- [노드 시작] '펀드 분석 노드' 실행 ---")
    
    # 1. State에서 파일 경로 입력 받기
    file_path = state['fund_data_path']

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

    # 3. LLM 입력 데이터 가공
    print("--- 펀드 분석 에이전트 실행 (로컬 PC로 연산 중...) ---")
    fund_data_str = json.dumps(raw_fund_data, indent=2, ensure_ascii=False)

    # 4. .invoke()를 사용하여 체인 실행
    analysis_result = chain.invoke({"input_data": fund_data_str})

    print("--- [노드 종료] '펀드 분석 노드' 완료 ---")
    
    # 5. State 업데이트 (반환)
    return {"fund_analysis_result": analysis_result}


# 그래프 정의 및 호출
if __name__ == "__main__":
    
    # 그래프 정의
    workflow = StateGraph(FundAgentState)

    # 노드 추가
    workflow.add_node("analyze_funds", run_fund_analysis_node)

    # 엣지 추가
    workflow.set_entry_point("analyze_funds")
    workflow.add_edge("analyze_funds", END)

    # 그래프 컴파일
    app = workflow.compile()

    # (중요) 초기 상태(Initial State) 정의
    # 상대 경로
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[2] 
    file_path_to_run = project_root / 'fund_data.json'

    initial_state = {
        "fund_data_path": str(file_path_to_run), # 노드에 파일 경로 주입
        "fund_analysis_result": {}
    }

    print("\n--- 🏁 (LangGraph) 펀드 분석 그래프 실행 시작 🏁 ---")
    
    # 4-6. 그래프 실행
    final_state = app.invoke(initial_state)

    # 4-7. 최종 결과 출력
    print("\n--- 🏁 (LangGraph) 그래프 실행 완료 🏁 ---")
    print("최종 분석 결과 (JSON):")
    print(json.dumps(final_state['fund_analysis_result'], indent=2, ensure_ascii=False))