# nodes/tool_nodes.py

import json
import pandas as pd
import os 
# ... (calculate_deposit_profit, calculate_savings_profit, calculate_fund_loss_profit 함수는 그대로 유지)

# 데이터 로드 함수만 경로 문제 해결을 위해 다시 확인
def load_data():
    """JSON 파일에서 투자 상품 데이터를 로드합니다."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # nodes -> .. (profit_loss_report) -> data -> test_data.json
    file_path = os.path.join(current_dir, '..', 'data', 'test_data.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: 데이터 파일을 찾을 수 없습니다. 시도한 경로: {file_path}")
        raise

def calculate_deposit_profit(deposit):
    """예금의 만기 이자 수익(세후)을 계산합니다."""
    # ... (기존 코드와 동일)
    principal = deposit['principal']
    rate = deposit['interest_rate']
    tax = deposit['tax_rate']
    
    gross_interest = principal * rate * (deposit['total_period_months'] / 12)
    net_interest = gross_interest * (1 - tax)
    return {
        'product_id': deposit['product_id'],
        'type': '예금',
        'principal': principal,
        'gross_profit': gross_interest,
        'net_profit': net_interest
    }

def calculate_savings_profit(savings):
    """적금의 만기 이자 수익(세후)을 계산합니다. (단리 기준)"""
    # ... (기존 코드와 동일)
    monthly_payment = savings['monthly_payment']
    period = savings['total_period_months']
    rate = savings['interest_rate']
    tax = savings['tax_rate']
    
    gross_interest = monthly_payment * (rate / 12) * (period * (period + 1) / 2)
    net_interest = gross_interest * (1 - tax)
    return {
        'product_id': savings['product_id'],
        'type': '적금',
        'principal': monthly_payment * period,
        'gross_profit': gross_interest,
        'net_profit': net_interest
    }

def calculate_fund_loss_profit(fund, report_date):
    """펀드의 현재 시점 수익/손실을 계산합니다."""
    # ... (기존 코드와 동일)
    purchase_nav = fund['purchase_nav']
    current_nav = fund['current_nav']
    total_shares = fund['total_shares']
    fee_rate = fund['fee_rate']
    
    current_value = total_shares * current_nav
    total_purchase_cost = total_shares * purchase_nav
    profit_loss = current_value - total_purchase_cost
    
    fee = total_purchase_cost * fee_rate 
    net_profit_loss = profit_loss - fee
    
    return {
        'product_id': fund['product_id'],
        'type': '펀드',
        'principal': total_purchase_cost,
        'current_value': current_value,
        'profit_loss': profit_loss,
        'net_profit_loss': net_profit_loss,
        'fee': fee
    }

def aggregate_financial_data(data):
    """모든 상품의 수익/손실을 계산하고 집계합니다."""
    # ... (기존 코드와 동일)
    all_results = []
    report_date = data['report_date']

    for dep in data['deposits']:
        all_results.append(calculate_deposit_profit(dep))
        
    for sav in data['savings']:
        all_results.append(calculate_savings_profit(sav))
        
    for fun in data['funds']:
        all_results.append(calculate_fund_loss_profit(fun, report_date))

    df = pd.DataFrame(all_results)
    
    total_principal = df['principal'].sum()
    
    # 순수익/손실 컬럼을 합산할 때 'net_profit' (예적금)과 'net_profit_loss' (펀드)를 모두 고려합니다.
    net_p = df['net_profit'].fillna(0).sum()
    net_l = df['net_profit_loss'].fillna(0).sum()
    total_net_profit_loss = net_p + net_l
    
    return df, total_principal, total_net_profit_loss