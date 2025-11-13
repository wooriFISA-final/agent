import math
import logging
from typing import Dict, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

# ----------------------------------
# 환경 설정 및 로깅
# ----------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")


class SummaryAgent:
    def __init__(self, model_name: str = "qwen3:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0.5)
        self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

    # -----------------------------------------------------------------
    # ① DB 조회 (members + plans + loan_product JOIN)
    # -----------------------------------------------------------------
    def _fetch_user_and_loan_info(self, user_id: int) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            query = text("""
                SELECT 
                    m.user_name, m.salary, m.income_usage_ratio,
                    m.initial_prop, m.hope_price, p.loan_amount,
                    p.product_id,
                    l.product_name, l.summary AS product_summary
                FROM members m
                JOIN plans p ON m.user_id = p.user_id
                LEFT JOIN loan_product l ON p.product_id = l.product_id
                WHERE m.user_id = :uid
                ORDER BY p.plan_id DESC
                LIMIT 1
            """)
            result = conn.execute(query, {"uid": user_id}).mappings().first()

        if not result:
            raise ValueError(f"user_id {user_id}의 정보를 찾을 수 없습니다.")

        # product_id만 있고 이름이 없는 경우 → loan_product에서 다시 조회
        if not result["product_name"] and result.get("product_id"):
            with self.engine.connect() as conn:
                p = conn.execute(
                    text("SELECT product_name, summary FROM loan_product WHERE product_id = :pid LIMIT 1"),
                    {"pid": result["product_id"]}
                ).mappings().first()
                if p:
                    result["product_name"] = p["product_name"]
                    result["product_summary"] = p["summary"]

        return dict(result)

    # -----------------------------------------------------------------
    # ② 부족금 계산 + members 테이블 업데이트 (안정화)
    # -----------------------------------------------------------------
    def _calculate_shortage_and_update(self, user_id, plan_data, loan_data):
        """
        loan_data가 없거나 loan_amount 키가 빠져 있을 때 안전하게 처리.
        """
        if not loan_data:
            logger.warning("⚠️ loan_data가 비어 있음, 기본값 0으로 대체")
            loan_data = {"loan_amount": 0}

        # loan_result가 감싸고 있을 수도 있으므로 fallback 구조 처리
        loan_info = loan_data.get("loan_result", loan_data)
        logger.debug(f"🔍 loan_info 데이터 구조: {loan_info}")

        # loan_amount 또는 last_loan_amount 우선 탐색
        loan_amt = loan_info.get("loan_amount") or loan_info.get("last_loan_amount") or 0
        init_prop = plan_data.get("initial_prop", 0)
        hope_price = plan_data.get("hope_price", 0)

        # 안전하게 숫자형 변환
        try:
            loan_amt = int(loan_amt)
            init_prop = int(init_prop)
            hope_price = int(hope_price)
        except Exception:
            logger.warning("⚠️ 금액 변환 오류 - 기본값 사용")
            loan_amt, init_prop, hope_price = 0, 0, 0

        shortage = max(0, hope_price - (loan_amt + init_prop))

        # DB 업데이트
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE members SET shortage_amount = :shortage WHERE user_id = :uid"),
                {"shortage": shortage, "uid": user_id},
            )

        logger.info(f"✅ shortage_amount({shortage:,}) DB 업데이트 완료 (loan_amount={loan_amt:,}, init_prop={init_prop:,})")
        return shortage

    # -----------------------------------------------------------------
    # ③ LLM 기반 투자 비중 판단
    # -----------------------------------------------------------------
    def _get_optimal_investment_ratio(self, saving_results, fund_results):
        saving_yield = saving_results.get("average_yield", 3.0)
        fund_yield = fund_results.get("average_yield", 6.0)

        prompt = f"""
        당신은 금융 포트폴리오 전문가입니다.
        아래 두 상품의 예상 수익률이 있습니다.

        - 예금/적금 평균 수익률: {saving_yield}%
        - 펀드 평균 수익률: {fund_yield}%

        일반적인 투자자에게 가장 효율적인 비중을 제안하세요.
        JSON으로 출력:
        {{
            "recommended_saving_ratio": 0.35,
            "recommended_fund_ratio": 0.65
        }}
        """
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            raw = response.content.strip()
            data = json.loads(raw.replace("```json", "").replace("```", "").strip())
            return float(data.get("recommended_saving_ratio", 0.35)), float(data.get("recommended_fund_ratio", 0.65))
        except Exception as e:
            logger.warning(f"⚠️ 투자 비율 계산 실패, 기본값 사용: {e}")
            return 0.35, 0.65

    # -----------------------------------------------------------------
    # ④ 복리 기반 투자 시뮬레이션
    # -----------------------------------------------------------------
    def _simulate_combined_investment(self, shortage, available_assets, monthly_income,
                                      income_usage_ratio, saving_yield, fund_yield,
                                      saving_ratio, fund_ratio):
        init_saving = available_assets * saving_ratio
        init_fund = available_assets * fund_ratio
        monthly_invest = monthly_income * (income_usage_ratio / 100)
        saving_monthly = monthly_invest * saving_ratio
        fund_monthly = monthly_invest * fund_ratio

        total_balance = 0
        months = 0
        while total_balance < shortage and months < 600:
            months += 1
            init_saving *= (1 + saving_yield / 100 / 12)
            init_fund *= (1 + fund_yield / 100 / 12)
            total_balance = init_saving + init_fund + (months * (saving_monthly + fund_monthly))

        return {"months_needed": months, "total_balance": int(total_balance), "monthly_invest": int(monthly_invest)}

    # -----------------------------------------------------------------
    # ⑤ 투자 결과 plans 테이블 업데이트
    # -----------------------------------------------------------------
    def _update_plan_targets(self, user_id: int, shortage: int, result: Dict[str, Any],
                             saving_ratio: float, fund_ratio: float):
        target_self_capital = shortage
        target_price_saving = int(result["total_balance"] * saving_ratio)
        target_price_fund = int(result["total_balance"] * fund_ratio)
        target_price_deposit = 0

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE plans
                    SET 
                        target_self_capital = :self_capital,
                        target_price_saving = :saving,
                        target_price_fund = :fund,
                        target_price_deposit = :deposit
                    WHERE user_id = :uid
                    ORDER BY plan_id DESC
                    LIMIT 1
                """),
                {
                    "self_capital": target_self_capital,
                    "saving": target_price_saving,
                    "fund": target_price_fund,
                    "deposit": target_price_deposit,
                    "uid": user_id,
                },
            )
        logger.info(f"✅ plans 테이블 업데이트 완료 (user_id={user_id})")

    # -----------------------------------------------------------------
    # ⑥ 리포트 프롬프트 (결론부는 LLM이 직접 작성)
    # -----------------------------------------------------------------
    def _build_prompt(self, user_data, shortage, result, saving_results, fund_results, saving_ratio, fund_ratio):
        def fmt(v):
            return "정보 없음" if v in (None, "", 0) else f"{int(v):,}원"

        current_balance = user_data.get("initial_prop", 0) + result["total_balance"]
        remaining_gap = max(0, user_data.get("hope_price", 0) - current_balance)

        return f"""
안녕하세요, {user_data.get('user_name', '고객')}님.  
현재 확인된 연소득은 {fmt(user_data.get('salary'))}이며,  
보유 자산은 약 {fmt(user_data.get('initial_prop'))}입니다.  
희망하시는 주택 가격은 {fmt(user_data.get('hope_price'))} 수준으로 확인됩니다.  

추천 대출 상품은 '{user_data.get('product_name', '정보 없음')}'이며,  
{user_data.get('product_summary', '상품 설명이 없습니다.')}  
예상 대출 금액은 약 {fmt(user_data.get('loan_amount'))},  
부족 금액은 약 {fmt(shortage)}로 계산됩니다.

현재 고객님의 월 소득 중 {user_data.get('income_usage_ratio', 30)}%를  
저축과 투자에 활용 중인 것으로 보입니다.  
이 자금을 예금/적금({saving_ratio*100:.1f}%), 펀드({fund_ratio*100:.1f}%)로 분배하면  
약 {result['months_needed']}개월(약 {round(result['months_needed']/12,1)}년) 후  
부족 금액 {fmt(shortage)}를 모두 채우실 수 있습니다.

월 재투자 금액은 {fmt(result['monthly_invest'])},  
총 누적 자금은 {fmt(result['total_balance'])}이며  
전체 자산은 {current_balance:,}원으로 예상됩니다.  
목표 주택 금액까지는 약 {remaining_gap:,}원이 부족합니다.

이 데이터를 기반으로,  
금융 전문가의 시각에서 고객에게 조언을 제시해주세요.  
내용에는 다음이 포함되어야 합니다:
1. 투자 전략 요약 (위험 vs 안정성 균형)
2. 대출/저축 활용 조언
3. 장기 재무 관점에서의 격려 문장
4. 자연스러운 마무리 인사
"""

    # -----------------------------------------------------------------
    # ⑦ 실행 (DB 업데이트 + 리포트 생성 + summary_report 저장)
    # -----------------------------------------------------------------
    def run(self, user_id, plan_data, loan_data, saving_results, fund_results):
        user_data = self._fetch_user_and_loan_info(user_id)

        # ✅ 안전한 부족금 계산
        shortage = self._calculate_shortage_and_update(user_id, plan_data, loan_data)

        monthly_income = (user_data.get("salary", 0) or 0) / 12
        saving_ratio, fund_ratio = self._get_optimal_investment_ratio(saving_results, fund_results)

        result = self._simulate_combined_investment(
            shortage,
            user_data.get("initial_prop", 0),
            monthly_income,
            float(user_data.get("income_usage_ratio", 20)),
            saving_results.get("average_yield", 3.0),
            fund_results.get("average_yield", 6.0),
            saving_ratio,
            fund_ratio,
        )

        self._update_plan_targets(user_id, shortage, result, saving_ratio, fund_ratio)

        prompt = self._build_prompt(user_data, shortage, result, saving_results, fund_results, saving_ratio, fund_ratio)
        response = self.llm.invoke([SystemMessage(content=prompt)])
        summary_text = response.content.strip()

        # ✅ summary_report 저장
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE plans
                    SET summary_report = :report
                    WHERE user_id = :uid
                    ORDER BY plan_id DESC
                    LIMIT 1
                """),
                {"report": summary_text, "uid": user_id},
            )
        logger.info(f"✅ summary_report 컬럼 업데이트 완료 (user_id={user_id})")

        return {
            "shortage_amount": shortage,
            "investment_result": result,
            "summary_text": summary_text
        }
