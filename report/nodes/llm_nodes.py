# report_project/report/nodes/llm_nodes.py

import json
import requests
import pandas as pd
import ollama
from typing import Dict, Any

# ==============================================================================
# 🛠️ 공통 Ollama 설정
# ==============================================================================
OLLAMA_HOST = 'http://localhost:11434' 
QWEN_MODEL = 'qwen3:8b'


# ==============================================================================
# 1. 🔍 compare 에이전트용: 변동 사항 비교 및 요약 노드 (RAG 기반)
# ==============================================================================
def compare_changes_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("🔍 변동 사항 비교 및 요약 시작...")
    
    # 🚨 수정: policy_info에서 old_policy와 new_policy를 모두 가져옵니다.
    policy_data = state.get('policy_info', {})
    old_policies = policy_data.get('old_policy', [])
    new_policies = policy_data.get('new_policy', [])
    
    # 1. 검색된 청크 내용을 문자열로 포맷팅
    context_text = "\n\n--- [RAG 검색 결과: 정책 변동 컨텍스트] ---\n"
    
    # 🚨 [수정 핵심] old_policies (문자열 리스트) 내용 포맷팅
    if old_policies:
        context_text += "--- 이전 정책 (20241224) 청크 ---\n"
        for i, content in enumerate(old_policies):
            # content는 문자열이므로, .get() 대신 직접 사용합니다.
            context_text += f"[이전 정책 청크 {i+1}]\n내용: {content[:300]}...\n---\n" 
    
    # 🚨 [수정 핵심] new_policies (문자열 리스트) 내용 포맷팅
    if new_policies:
        context_text += "\n--- 신규 정책 (20250305) 청크 ---\n"
        for i, content in enumerate(new_policies):
            # content는 문자열이므로, .get() 대신 직접 사용합니다.
            context_text += f"[신규 정책 청크 {i+1}]\n내용: {content[:300]}...\n---\n"
    
    if not old_policies and not new_policies:
        context_text += "정책 변동 분석을 위한 검색 결과가 없습니다."


    prompt = f"""
    당신은 금융 정책 비교 분석가입니다. 아래 제공된 [RAG 검색 결과 컨텍스트] 텍스트만 사용하여 분석을 수행하십시오. 이 컨텍스트에는 2024년 12월 버전과 2025년 3월 버전의 정책 조항들이 포함되어 있습니다.
    
    [핵심 임무: 오직 변경점만 추출]
    1. **두 정책의 모든 [장 제목]을 대조**하여, **신규 정책(20250305)에서 변경되거나 새롭게 추가된 내용**만을 간결하고 명확하게 요약하여 보고하십시오.
    2. 변경이 없는 내용은 언급하지 않습니다.
    3. 정책 파일 로드에 실패했다면 (내용에 '정책 파일 로드 실패' 포함) 해당 사실을 명시하고 분석을 중단하십시오.
    
    {context_text}
    
    [이전 달 재무/환경 데이터 - 분석 참고용]
    - 이전 달 보고서: {state.get('report_data', 'N/A')}
    - 현재 주택 정보: {state.get('house_info', 'N/A')}
    - 현재 신용 정보: {state.get('credit_info', 'N/A')}
    """
    
    # 🚨 Ollama 호출 로직 (타임아웃 180초로 설정)
    response_content = "❌ LLM 호출 실패"
    payload = {
        "model": QWEN_MODEL, 
        "prompt": f"[System] 너는 데이터 분석과 리포트 요약에 능숙한 한국어 어시스턴트야.\n\n[Human] {prompt}",
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    try:
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
        res.raise_for_status() 
        response_content = res.json()['response'].strip()
        print("✅ [LLM Node] 변동 사항 비교 요약 완료")
    except requests.exceptions.RequestException as e:
        response_content = f"❌ [LLM Node] Ollama 통신 오류: {e}. Ollama 서버(http://localhost:11434)와 모델({QWEN_MODEL}) 상태를 확인하세요."
        print(response_content)
    
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