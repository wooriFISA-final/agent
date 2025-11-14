import math
import logging
from typing import Dict, Any, Tuple
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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


# ============================================================
# 💼 SummaryAgent (그래프의 단일 노드로 사용)
# ============================================================
class SummaryAgent:
    def __init__(self, model_name: str = "qwen3:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0.5)
        self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

        # ✅ 새 SYSTEM PROMPT (자산관리 리포트용)
        self.SYSTEM_PROMPT = SystemMessage(content="""
[페르소나(Persona)]
당신은 '우리은행 프리미엄 자산관리 컨설턴트'입니다.  
고객의 대출, 저축, 투자 데이터를 기반으로 **구체적인 상품 추천 보고서**를 작성합니다.  
전문적이지만 따뜻한 어조로 고객 맞춤형 재무 조언을 제시해야 합니다.

---

[작성 형식]
아래 단계별로 작성하세요. 반드시 각 항목을 포함해야 합니다.

### 1️⃣ 대출 상품 분석 및 추천
- 고객의 소득, 희망 주택 가격, 보유 자산을 고려하여 대출 가능한 상품을 소개합니다.  
- 다음 형식을 사용하세요:
  - 상품명: (예: 스마트징검다리론)
  - 상품 설명: (대출 대상, 특징, 금리, 상환방식 등)
  - 예상 대출금액: (고객 데이터 기반)
  - 이 상품이 고객에게 적합한 이유를 설명하세요.

---

### 2️⃣ 예금 상품 추천
- 예금상품 중 2~3개를 선택하여 소개합니다.
- 각 상품은 다음 형식을 사용하세요:
  - 상품명:
  - 상품 설명:
  - 예상 수익 및 추천 이유:
- 고객의 자금 규모를 고려해 “이 예금을 통해 모을 수 있는 금액”을 구체적으로 언급하세요.

---

### 3️⃣ 적금 상품 및 펀드 추천
- 적금상품 1~2개, 펀드상품 1~2개를 각각 소개하세요.
- 각 상품은 예금 추천과 동일한 형식을 따르세요.
- 펀드상품은 ‘수익률 기대치’나 ‘위험 수준’을 함께 언급하세요.

---

### 4️⃣ 종합 분석 및 예상 소요기간
- 위의 추천 상품을 조합했을 때, 고객이 목표 주택금액을 달성하기까지의 예상 기간을 요약하세요.
- “총 약 X년 (X개월) 정도가 예상됩니다.” 문장을 포함하세요.

---

### 5️⃣ 마무리 인사
- 고객 이름을 포함하여, 따뜻하고 전문적인 어조로 격려하는 마무리 문장을 작성하세요.
- 예: “유진수님, 지금의 계획은 매우 실질적이며 장기적인 재무 안정에 큰 도움이 될 것입니다. 꾸준함이 최고의 자산입니다.”

---

[스타일 가이드]
- 마크다운 형식 사용 (### 제목, **강조**)
- 길이는 800~1200자 내외
- 데이터 수치를 자연스럽게 녹여서 서술
- 모든 금액은 “원” 단위로 표시
""")

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
    # ② 부족금 계산 + members 테이블 업데이트
    # -----------------------------------------------------------------
    def _calculate_shortage_and_update(self, user_id: int, plan_data: Dict[str, Any], loan_data: Dict[str, Any]) -> int:
        if not loan_data:
            logger.warning("⚠️ loan_data가 비어 있음, 기본값 0으로 대체")
            loan_data = {"loan_amount": 0}

        loan_info = loan_data.get("loan_result", loan_data)
        loan_amt = int(loan_info.get("loan_amount") or 0)
        init_prop = int(plan_data.get("initial_prop", 0) or 0)
        hope_price = int(plan_data.get("hope_price", 0) or 0)

        shortage = max(0, hope_price - (loan_amt + init_prop))

        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE members SET shortage_amount = :shortage WHERE user_id = :uid"),
                {"shortage": shortage, "uid": user_id},
            )

        logger.info(f"✅ shortage_amount({shortage:,}) 업데이트 완료 (loan_amount={loan_amt:,}, init_prop={init_prop:,})")
        return shortage

    # -----------------------------------------------------------------
    # ③ 투자 비율 산출 (LLM 기반)
    # -----------------------------------------------------------------
    def _get_optimal_investment_ratio(self, saving_results: Dict[str, Any], fund_results: Dict[str, Any]) -> Tuple[float, float]:
        saving_yield = float(saving_results.get("average_yield", 3.0))
        fund_yield = float(fund_results.get("average_yield", 6.0))

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
            # fence 제거 후 파싱
            payload = response.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(payload)
            return float(data.get("recommended_saving_ratio", 0.35)), float(data.get("recommended_fund_ratio", 0.65))
        except Exception as e:
            logger.warning(f"⚠️ 투자 비율 계산 실패, 기본값 사용: {e}")
            return 0.35, 0.65

    # -----------------------------------------------------------------
    # ④ 복리 기반 투자 시뮬레이션(간단 모델)
    # -----------------------------------------------------------------
    def _simulate_combined_investment(
        self,
        shortage: int,
        available_assets: int,
        monthly_income: float,
        income_usage_ratio: float,
        saving_yield: float,
        fund_yield: float,
        saving_ratio: float,
        fund_ratio: float,
    ) -> Dict[str, Any]:
        init_saving = available_assets * saving_ratio
        init_fund = available_assets * fund_ratio

        monthly_invest = monthly_income * (income_usage_ratio / 100)
        saving_monthly = monthly_invest * saving_ratio
        fund_monthly = monthly_invest * fund_ratio

        total_balance = 0.0
        months = 0
        # 간단 누적 모델(월복리 + 적립식 단순 가산)
        while total_balance < shortage and months < 600:
            months += 1
            init_saving = (init_saving + saving_monthly) * (1 + saving_yield / 100 / 12)
            init_fund = (init_fund + fund_monthly) * (1 + fund_yield / 100 / 12)
            total_balance = init_saving + init_fund

        return {
            "months_needed": months,
            "total_balance": int(total_balance),
            "monthly_invest": int(monthly_invest),
            "saving_ratio": saving_ratio,
            "fund_ratio": fund_ratio,
        }

    # -----------------------------------------------------------------
    # ⑤ 리포트 생성용 사용자 프롬프트
    # -----------------------------------------------------------------
    def _build_prompt(
        self,
        user_data: Dict[str, Any],
        shortage: int,
        result: Dict[str, Any],
        saving_results: Dict[str, Any],
        fund_results: Dict[str, Any],
        saving_ratio: float,
        fund_ratio: float,
    ) -> str:
        def fmt(v):
            return "정보 없음" if v in (None, "", 0) else f"{int(v):,}원"

        product_name = user_data.get("product_name", "정보 없음")
        product_summary = user_data.get("product_summary", "상품 설명이 없습니다.")

        prompt = f"""
고객 요약 데이터:
- 이름: {user_data.get('user_name', '고객')}
- 연소득: {fmt(user_data.get('salary'))}
- 보유 자산: {fmt(user_data.get('initial_prop'))}
- 희망 주택 가격: {fmt(user_data.get('hope_price'))}
- 예상 대출금액: {fmt(user_data.get('loan_amount'))}
- 부족 금액: {fmt(shortage)}
- 월 소득 대비 저축·투자 비율: {user_data.get('income_usage_ratio', 30)}%
- 예금 평균 수익률: {saving_results.get('average_yield', 3.0)}%
- 펀드 평균 수익률: {fund_results.get('average_yield', 6.0)}%
- 추천 비중(예금/펀드): {int(saving_ratio*100)}% / {int(fund_ratio*100)}%
- 목표 달성 예상 기간: 약 {result['months_needed']}개월 (약 {round(result['months_needed']/12,1)}년)
- 추천 대출 상품: {product_name} / {product_summary}

위 정보를 기반으로 **고객 맞춤형 자산관리 보고서**를 작성하세요.
지침을 철저히 따르고 마크다운(###, **강조**)을 사용하세요.
"""
        return prompt

    # -----------------------------------------------------------------
    # ⑥ 실행 (DB 업데이트 + 리포트 생성 + summary_report 저장)
    # -----------------------------------------------------------------
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_id = state.get("user_id")
        plan_data = state.get("validated_plan_input", {}) or {}
        loan_data = state.get("loan_result", {}) or {}
        saving_results = state.get("savings_recommendations", {}) or {}
        fund_results = state.get("fund_analysis_result", {}) or {}

        # 1) 사용자/대출 데이터 로드 + 부족금 계산
        user_data = self._fetch_user_and_loan_info(user_id)
        shortage = self._calculate_shortage_and_update(user_id, plan_data, loan_data)
        monthly_income = (user_data.get("salary", 0) or 0) / 12

        # 2) 투자 비중 추정 + 간단 시뮬레이션
        saving_ratio, fund_ratio = self._get_optimal_investment_ratio(saving_results, fund_results)
        result = self._simulate_combined_investment(
            shortage=shortage,
            available_assets=int(user_data.get("initial_prop", 0) or 0),
            monthly_income=float(monthly_income),
            income_usage_ratio=float(user_data.get("income_usage_ratio", 20)),
            saving_yield=float(saving_results.get("average_yield", 3.0)),
            fund_yield=float(fund_results.get("average_yield", 6.0)),
            saving_ratio=saving_ratio,
            fund_ratio=fund_ratio,
        )

        # 3) 리포트 생성
        prompt = self._build_prompt(user_data, shortage, result, saving_results, fund_results, saving_ratio, fund_ratio)
        response = self.llm.invoke([self.SYSTEM_PROMPT, HumanMessage(content=prompt)])
        summary_text = response.content.strip()

        # 4) 보고서 저장
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
        logger.info(f"✅ summary_report 저장 완료 (user_id={user_id})")

        # 5) 반환 — UI가 바로 렌더할 수 있도록 본문을 messages에 포함
        return {
            "summary_result": {
                "shortage_amount": shortage,
                "investment_result": result,
                "summary_text": summary_text
            },
            # 👉 여기서 실제 보고서 본문을 AIMessage로 넣어줌
            "messages": [AIMessage(content=summary_text)],
            # 👉 토스트/배너 등 별도 알림이 필요하면 notifications로 제공(선택)
            "notifications": ["📊 맞춤형 자산관리 보고서를 생성하고 저장했습니다."]
        }
