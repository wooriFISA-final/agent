from state import AgentState
from llm.ollama_llm import ollama_llm
from langchain_core.messages import SystemMessage, HumanMessage

##########################################
###  LLM 활용이 필요한 노드들을 정의하는 파일   ###
##########################################

def compare_changes(state: AgentState):
    print("🔍 변동 사항 비교 중...")

    prompt = f"""
    아래는 이전 달 데이터와 새로 불러온 데이터입니다.
    변동 사항을 간결하고 논리적으로 요약해 주세요.

    [이전 달 데이터]: {state.get('report_data')}
    [주택 정보]: {state.get('house_info')}
    [정책 정보]: {state.get('policy_info')}
    [신용 정보]: {state.get('credit_info')}
    """

    response = ollama_llm.invoke([
        SystemMessage(content="너는 데이터 분석과 리포트 요약에 능숙한 한국어 어시스턴트야."),
        HumanMessage(content=prompt)
    ])

    state["comparison_result"] = response.content
    return state

