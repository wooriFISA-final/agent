# nodes/llm_nodes.py

import pandas as pd
import json
import ollama

# 🚨 Ollama 설정 확인 🚨
OLLAMA_HOST = 'http://localhost:11434' 
QWEN_MODEL = 'qwen3:8b'


def generate_visualization_data(df):
    """수익/손실 데이터프레임을 시각화에 적합한 형태로 변환합니다."""
    # ... (이전 코드와 동일, 변경 없음)
    vis_data = df.groupby('type').agg({
        'principal': 'sum',
        'net_profit': 'sum',
        'net_profit_loss': 'sum'
    }).fillna(0)
    
    vis_data['total_net_p_l'] = vis_data['net_profit'] + vis_data['net_profit_loss']
    chart_data = vis_data['total_net_p_l'].to_dict()
    
    return chart_data, ""


def analyze_investment_results(df, total_principal, total_net_profit_loss, chart_data):
    """
    Ollama에 구동 중인 Qwen LLM을 호출하여 분석 보고서를 생성합니다.
    """
    
    # 1. LLM에게 전달할 입력 데이터 준비
    financial_summary = {
        '총_원금': f"{total_principal:,} 원",
        '총_순수익_손실': f"{total_net_profit_loss:,.0f} 원",
        '수익률': f"{total_net_profit_loss / total_principal * 100:.2f}%" if total_principal > 0 else "0.00%",
        '상품별_상세': df.to_dict('records'),
        '시각화_데이터': chart_data
    }
    
    # 2. Ollama 클라이언트 및 프롬프트 구성
    client = ollama.Client(host=OLLAMA_HOST)

    prompt = f"""
    당신은 전문 투자 분석가입니다. 아래 JSON 형식의 투자 요약 데이터를 기반으로 사용자에게 한국어 보고서를 작성해 주세요.
    보고서에는 다음 내용이 포함되어야 합니다:
    1. 총 투자 원금 대비 최종 순수익/손실 요약.
    2. 가장 큰 수익을 낸 상품 타입과 가장 큰 손실을 낸 상품 타입 분석.
    3. 전체적인 투자 포트폴리오에 대한 간단한 조언.
    
    [투자 요약 데이터 (JSON)]
    {json.dumps(financial_summary, indent=2, ensure_ascii=False)}
    """
    
    llm_analysis_result = "Ollama Qwen 호출 실패 (Ollama 서버 실행, 모델명, 네트워크 확인 필요)"
    
    try:
        # 3. Ollama 모델 호출
        response = client.generate(
            model=QWEN_MODEL,
            prompt=prompt
        )

        # 4. 응답 파싱 및 반환
        llm_analysis_result = response['response']
            
    except Exception as e:
        llm_analysis_result = f"Ollama 호출 중 예외 발생: {e}. Ollama 서버가 {OLLAMA_HOST}에서 실행 중인지 확인하세요."
         
    return llm_analysis_result