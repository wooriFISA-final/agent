import json
import requests
import pandas as pd
import ollama
from typing import Dict, Any

# ⚠️ 주의: AgentState는 최상위 state.py에 정의되어야 하며, 여기서 상대 경로 import는 삭제합니다.
# 실제 사용 시, 각 에이전트의 builder.py에서 state.py의 AgentState 또는 해당 에이전트의 State를 import하여 사용해야 합니다.
# 여기서는 타입 힌트만 제공합니다.
# from state import AgentState, ConsumptionAnalysisState # 이 줄은 통합 파일에서는 제외합니다.

# ==============================================================================
# 🛠️ 공통 Ollama 설정
# ==============================================================================
# 모든 LLM 노드가 Ollama 호출을 위해 공통적으로 사용합니다.
OLLAMA_HOST = 'http://localhost:11434' 
QWEN_MODEL = 'qwen3:8b'
# compare 에이전트에서 사용된 방식:
# from llm.ollama_llm import ollama_llm  # <--- 이 방식은 중앙 집중화 시 경로 문제로 인해 사용하지 않습니다. 
                                        # 대신, 아래 함수들은 requests 또는 ollama.Client를 직접 사용합니다.


# ==============================================================================
# 1. 🔍 compare 에이전트용: 변동 사항 비교 및 요약 노드
# ==============================================================================
def compare_changes_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    이전 데이터와 현재 변동 사항을 비교하고 LLM을 사용하여 결과를 요약합니다.
    Args:
        state: 'report_data', 'house_info', 'policy_info', 'credit_info' 키를 포함하는 상태 딕셔너리.
    """
    print("🔍 [LLM Node] 변동 사항 비교 및 요약 시작...")

    prompt = f"""
    아래는 이전 달 데이터와 새로 불러온 데이터입니다.
    변동 사항을 간결하고 논리적으로 요약해 주세요.

    [이전 달 데이터]: {state.get('report_data', '정보 없음')}
    [주택 정보]: {state.get('house_info', '정보 없음')}
    [정책 정보]: {state.get('policy_info', '정보 없음')}
    [신용 정보]: {state.get('credit_info', '정보 없음')}
    """

    # compare 에이전트에서 사용된 방식(langchain_core/ollama_llm)을 requests 기반으로 통일합니다.
    payload = {
        "model": "qwen3:8b", # 모델은 실제 사용하는 모델로 변경 필요 (예: llama3)
        "prompt": f"[System] 너는 데이터 분석과 리포트 요약에 능숙한 한국어 어시스턴트야.\n\n[Human] {prompt}",
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    response_content = "❌ LLM 호출 실패"
    try:
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
        res.raise_for_status() 
        response_content = res.json()['response'].strip()
        print("✅ [LLM Node] 변동 사항 비교 요약 완료")
    except requests.exceptions.RequestException as e:
        print(f"❌ [LLM Node] Ollama 통신 오류: {e}")
    
    state["comparison_result"] = response_content
    return state


# ==============================================================================
# 2. 🧾 consume 에이전트용: 최종 소비 분석 보고서 생성 노드
# ==============================================================================
def generate_final_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """군집 정보와 분석 결과를 기반으로 최종 보고서를 생성합니다."""
    
    nickname = state.get('cluster_nickname', '미정 군집')
    analysis_data = state.get('user_analysis', {})
    ollama_model_name = state.get('ollama_model_name', 'llama3') # 기본 모델 설정

    # 필요한 데이터 추출 및 형식화
    total_spend_amount = analysis_data.get('total_spend_amount', 'N/A')
    top_3_categories = analysis_data.get('top_3_categories', ['N/A'])
    fixed_cost = analysis_data.get('fixed_cost', 'N/A')
    non_fixed_cost_rate = analysis_data.get('non_fixed_cost_rate', 'N/A')
    
    analysis_text = (
        f"총 지출액: {total_spend_amount}, "
        f"주 소비 영역: {', '.join(top_3_categories)}, "
        f"고정비: {fixed_cost}, "
        f"비고정비 비중: {non_fixed_cost_rate}"
    )
    
    prompt_template = f"""
    [System] 당신은 고객의 소비 분석가입니다. 다음 정보를 기반으로, 고객에게 전달할 4~5줄의 **간결하고 정중한** 소비 분석 보고서를 한국어로 작성하세요. 별도의 머리글이나 꼬리글 없이 본론부터 시작합니다.
    
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
    
    final_report = "❌ Ollama 통신 오류: 서버 문제 또는 타임아웃." 
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=300) 
        response.raise_for_status() 
        final_report = response.json()['response'].strip()
        print("✅ [LLM Node] 최종 보고서 생성 완료")
    except requests.exceptions.RequestException as e:
        print(f"❌ [LLM Node] Ollama 통신 오류 발생. 오류: {e}")
        
    state['final_report'] = final_report
    return state


# ==============================================================================
# 3. 💰 profit 에이전트용: 투자 분석 보고서 생성 노드
# ==============================================================================
def generate_visualization_data(df: pd.DataFrame) -> tuple[Dict[str, float], str]:
    """수익/손실 데이터프레임을 시각화에 적합한 형태로 변환합니다."""
    # Profit 에이전트에서 LLM과 함께 사용된 데이터 전처리 로직입니다.
    # LLM이 필요한 노드는 아니지만, 해당 에이전트에서 함께 쓰였기에 포함합니다.
    vis_data = df.groupby('type').agg({
        'principal': 'sum',
        'net_profit': 'sum',
        'net_profit_loss': 'sum'
    }).fillna(0)
    
    vis_data['total_net_p_l'] = vis_data['net_profit'] + vis_data['net_profit_loss']
    chart_data = vis_data['total_net_p_l'].to_dict()
    
    return chart_data, "" # 차트 데이터와 빈 문자열 반환 (기존 함수의 형태 유지)


def analyze_investment_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ollama에 구동 중인 Qwen LLM을 호출하여 투자 분석 보고서를 생성합니다.
    Args:
        state: 'analysis_df', 'total_principal', 'total_net_profit_loss' 키를 포함하는 상태 딕셔너리.
    """
    df = state.get('analysis_df')
    total_principal = state.get('total_principal', 0)
    total_net_profit_loss = state.get('total_net_profit_loss', 0)

    # 1. 시각화 데이터 준비 (같은 파일 내 함수 사용)
    chart_data, _ = generate_visualization_data(df)
    
    # 2. LLM에게 전달할 입력 데이터 준비
    financial_summary = {
        '총_원금': f"{total_principal:,} 원",
        '총_순수익_손실': f"{total_net_profit_loss:,.0f} 원",
        '수익률': f"{total_net_profit_loss / total_principal * 100:.2f}%" if total_principal > 0 else "0.00%",
        '상품별_상세': df.to_dict('records'),
        '시각화_데이터': chart_data
    }
    
    # 3. Ollama 클라이언트 및 프롬프트 구성
    client = ollama.Client(host=OLLAMA_HOST)

    prompt = f"""
    [System] 당신은 전문 투자 분석가입니다. 아래 JSON 형식의 투자 요약 데이터를 기반으로 사용자에게 한국어 보고서를 작성해 주세요.
    보고서에는 다음 내용이 포함되어야 합니다:
    1. 총 투자 원금 대비 최종 순수익/손실 요약.
    2. 가장 큰 수익을 낸 상품 타입과 가장 큰 손실을 낸 상품 타입 분석.
    3. 전체적인 투자 포트폴리오에 대한 간단한 조언.
    
    [투자 요약 데이터 (JSON)]
    {json.dumps(financial_summary, indent=2, ensure_ascii=False)}
    """
    
    llm_analysis_result = "Ollama Qwen 호출 실패 (Ollama 서버 실행, 모델명, 네트워크 확인 필요)"
    
    try:
        response = client.generate(
            model=QWEN_MODEL,
            prompt=prompt
        )
        llm_analysis_result = response['response'].strip()
        print("✅ [LLM Node] 투자 분석 보고서 생성 완료")
            
    except Exception as e:
        llm_analysis_result = f"❌ [LLM Node] Ollama 호출 중 예외 발생: {e}. Ollama 서버가 {OLLAMA_HOST}에서 실행 중인지 확인하세요."
    
    state['investment_analysis_result'] = llm_analysis_result
    return state

# LLM이 필요한 노드 함수:
# - compare_changes_node
# - generate_final_report_node
# - analyze_investment_results_node
# - generate_visualization_data (profit 에이전트 데이터 전처리용)