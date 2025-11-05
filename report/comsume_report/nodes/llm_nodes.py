import requests
import json
from ..state import ConsumptionAnalysisState

# Task 6/7: Ollama LLM 해석 및 보고서 생성 함수
def generate_final_report_node(state: ConsumptionAnalysisState) -> ConsumptionAnalysisState:
    """LLM을 호출하여 군집 정보와 분석 결과를 기반으로 최종 보고서를 생성합니다."""
    
    nickname = state['cluster_nickname']
    analysis_data = state['user_analysis']
    ollama_model_name = state['ollama_model_name']
    
    analysis_text = (
        f"총 지출액: {analysis_data['total_spend_amount']}, "
        f"주 소비 영역: {', '.join(analysis_data['top_3_categories'])}, "
        f"고정비: {analysis_data['fixed_cost']}, "
        f"비고정비 비중: {analysis_data['non_fixed_cost_rate']}"
    )
    
    prompt_template = f"""
    당신은 고객의 소비 분석가입니다. 다음 정보를 기반으로, 고객에게 전달할 4~5줄의 **간결하고 정중한** 소비 분석 보고서를 작성하세요.
    보고서는 한국어로 작성해야 하며, 별도의 머리글이나 꼬리글 없이 바로 본론부터 시작합니다.
    
    [핵심 정보]
    1. 군집 별명: {nickname}
    2. 개인 분석: {analysis_text}
    
    [보고서 포함 요소 및 형식]
    - 고객의 군집 별명을 언급하며 시작
    - 주 소비 영역을 구체적인 금액과 함께 언급
    - 고정비/비고정비 비중을 해석하여 소비 습관에 대한 인사이트 한 줄 포함
    - 최종 아웃풋은 4~5줄의 줄 글 형태여야 함.
    """
    
    payload = {
        "model": ollama_model_name, "prompt": prompt_template, "stream": False,
        "options": {"temperature": 0.5, "num_predict": 1024}
    }
    
    final_report = "❌ Ollama 통신 오류: Ollama 서버 문제 또는 타임아웃." # 기본 오류 메시지
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300) 
        response.raise_for_status() 
        final_report = response.json()['response'].strip()
        print("✅ LLM Node: 최종 보고서 생성 완료")
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM Node: Ollama 통신 오류 발생. 오류: {e}")
        pass # 오류 메시지는 final_report에 남아있음
        
    state['final_report'] = final_report
    return state