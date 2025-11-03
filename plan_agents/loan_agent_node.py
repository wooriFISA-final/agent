import math
import ollama
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")


class LoanAgentNode:
    """
    LoanAgentNode v7
    ---------------------------------------------------------
    사용자(user_info), 계획(plan_input), 지역(state), 상품(loan_product)을 통합하여
    직업유형(직장인/공무원/자영업/프리랜서)에 맞는 대출 상품을 추천하고,
    대출 이후 남은 금액(절댓값), 목표-대출 차액을 계산한다.
    """

    def __init__(self, llm_model="qwen3:8b"):
        self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
        self.llm_model = llm_model

    # ------------------------------------------------
    # 1️⃣ 지역별 시세 조회
    # ------------------------------------------------
    def _get_region_price(self, region_name):
        """
        지역명 매핑 규칙:
        - 서울특별시는 구 단위까지 그대로 검색 (예: '서울특별시 송파구')
        - 그 외 광역시/도는 시 단위까지만 추출 (예: '부산광역시', '경기도 수원시')
        """
        parts = region_name.split()

        if not parts:
            return None

        # ✅ 1️⃣ 서울특별시는 구 단위까지 사용
        if parts[0] == "서울특별시":
            city_name = " ".join(parts[:2]) if len(parts) > 1 else "서울특별시"

        # ✅ 2️⃣ '광역시'는 시까지만 사용 (부산, 대구, 인천, 광주, 대전, 울산)
        elif parts[0].endswith("광역시"):
            city_name = parts[0]

        # ✅ 3️⃣ '특별자치시'(세종)도 그대로 사용
        elif parts[0].endswith("특별자치시"):
            city_name = parts[0]

        # ✅ 4️⃣ '도' 단위 지역은 '도 + 시'까지만 사용 (경기도 수원시)
        elif parts[0].endswith("도"):
            city_name = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]

        else:
            # 예외 처리: 그 외는 첫 단어만 사용
            city_name = parts[0]

        query = text("""
            SELECT apartment_price, multi_price, officetel_price, detached_price
            FROM state
            WHERE region_nm LIKE :region
            LIMIT 1
        """)
        with self.engine.connect() as conn:
            row = conn.execute(query, {"region": f"%{city_name}%"}).fetchone()

        return dict(row._mapping) if row else None


    # ------------------------------------------------
    # 2️⃣ 대출 상품 조회
    # ------------------------------------------------
    def _get_loan_products(self):
        query = text("SELECT * FROM loan_product")
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [dict(row) for row in rows]

    # ------------------------------------------------
    # 3️⃣ 월 상환액 계산 (원리금 균등상환)
    # ------------------------------------------------
    def _calc_monthly_payment(self, principal, annual_rate, years):
        annual_rate = float(annual_rate)  # Decimal → float 변환
        monthly_rate = annual_rate / 12 / 100
        n = years * 12
        if monthly_rate == 0:
            return principal / n
        return principal * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)

    # ------------------------------------------------
    # 4️⃣ 직업유형별 월소득 계산
    # ------------------------------------------------
    def _get_monthly_income(self, user):
        job_type = user["job_type"]
        if job_type in ["직장인", "공무원"]:
            if user["monthly_salary"]:
                return int(user["monthly_salary"])
            elif user["income"]:
                return int(user["income"]) // 12
            else:
                return 0
        elif job_type in ["자영업", "프리랜서"]:
            if user.get("operating_income"):
                return int(user["operating_income"]) // 12
            elif user.get("annual_revenue"):
                return int(user["annual_revenue"] * 0.2 // 12)
            else:
                return 0
        else:
            return 0

    # ------------------------------------------------
    # 5️⃣ 대출 상품 추천 로직
    # ------------------------------------------------
    def _recommend(self, user, plan, region, products):
        target_price = int(plan["target_house_price"])
        available_assets = int(plan["available_assets"])
        credit_score = int(user["credit_score"]) if user["credit_score"] else 700
        monthly_income = self._get_monthly_income(user)
        annual_income = monthly_income * 12

        if monthly_income <= 0:
            return None, 0, 0

        best, best_score = None, float("inf")

        for p in products:
            if not p["max_ltv"] or not p["max_dsr"]:
                continue

            # Decimal → float 변환
            p["interest_rate"] = float(p["interest_rate"])

            # LTV / DSR 기반 대출 가능액
            possible_loan_by_ltv = target_price * (p["max_ltv"] / 100)
            possible_loan_by_dsr = annual_income * (p["max_dsr"] / 100)
            possible_loan = min(possible_loan_by_ltv, possible_loan_by_dsr)

            monthly_payment = self._calc_monthly_payment(possible_loan, p["interest_rate"], p["period_years"])

            # 소득 대비 상환 가능비율 40% 초과 시 제외
            if monthly_payment > monthly_income * 0.4:
                continue

            # 신용점수 가중치 반영
            credit_weight = (900 - credit_score) / 100
            score = p["interest_rate"] + (monthly_payment / 1000000) + credit_weight

            if score < best_score:
                best = {**p, "loan_amount": int(possible_loan), "monthly_payment": monthly_payment}
                best_score = score

        return best, best["loan_amount"], best["monthly_payment"] if best else (None, 0, 0)

    # ------------------------------------------------
    # 6️⃣ LLM 추천 이유 생성
    # ------------------------------------------------
    def _generate_explanation(self, user, plan, loan, shortage):
        prompt = f"""
        사용자의 직업은 {user["job_type"]}, 신용점수는 {user["credit_score"]}점입니다.
        월소득은 약 {self._get_monthly_income(user):,}원이며, 
        목표 주택 가격은 {int(plan['target_house_price']):,}원입니다.
        추천된 상품은 '{loan["loan_name"]}'으로, 금리는 {loan["interest_rate"]}%이고
        월 상환액은 약 {round(loan["monthly_payment"]):,}원입니다.
        대출 이후 남은 금액은 약 {shortage:,}원입니다.
        이 추천이 적합한 이유를 2~3문장으로 간단히 설명해 주세요.
        """
        try:
            response = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]
        except Exception as e:
            return f"(LLM 설명 생성 실패: {str(e)})"

    # ------------------------------------------------
    # 7️⃣ 실행 (핵심)
    # ------------------------------------------------
    def run(self, user_id, plan_id):
        with self.engine.connect() as conn:
            user = conn.execute(
                text("SELECT * FROM user_info WHERE user_id=:id"), {"id": user_id}
            ).mappings().fetchone()
            plan = conn.execute(
                text("SELECT * FROM plan_input WHERE id=:id"), {"id": plan_id}
            ).mappings().fetchone()

        if not user or not plan:
            raise ValueError("User 또는 Plan 데이터가 없습니다.")

        region = self._get_region_price(plan["target_location"])
        products = self._get_loan_products()

        best, loan_amount, monthly_payment = self._recommend(user, plan, region, products)
        if not best:
            return {"message": "적합한 대출 상품을 찾지 못했습니다."}

        # ✅ 변경된 계산 로직
        remaining_after_loan = int(plan["target_house_price"]) - loan_amount
        shortage = abs(int(plan["target_house_price"]) - loan_amount - int(plan["available_assets"]))

        explanation = self._generate_explanation(user, plan, best, shortage)

        # ✅ DB 반영 (plan_input + user_info)
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE plan_input
                SET loan_amount=:loan_amount,
                    remaining_after_loan=:remaining_after_loan,
                    recommended_loan_id=:loan_id,
                    income_usage_ratio=:usage
                WHERE id=:id
            """), {
                "loan_amount": loan_amount,
                "remaining_after_loan": remaining_after_loan,
                "loan_id": best["loan_id"],
                "usage": str(round(monthly_payment / (self._get_monthly_income(user)) * 100, 2)),
                "id": plan_id
            })

            conn.execute(text("""
                UPDATE user_info
                SET last_recommended_loan_id = :loan_id,
                    last_loan_amount = :loan_amount,
                    last_monthly_payment = :monthly_payment,
                    last_shortage_amount = :shortage,
                    last_recommend_date = NOW()
                WHERE user_id = :user_id
            """), {
                "loan_id": best["loan_id"],
                "loan_amount": loan_amount,
                "monthly_payment": monthly_payment,
                "shortage": shortage,
                "user_id": user["user_id"]
            })

        return {
            "user_name": user["name"],
            "job_type": user["job_type"],
            "region": plan["target_location"],
            "loan_name": best["loan_name"],
            "loan_amount": loan_amount,
            "interest_rate": best["interest_rate"],
            "monthly_payment": round(monthly_payment),
            "period_years": best["period_years"],
            "remaining_after_loan": remaining_after_loan,
            "shortage_amount": shortage,
            "credit_score": user["credit_score"],
            "monthly_income": self._get_monthly_income(user),
            "repayment_method": best["repayment_method"],
            "description": best["description"],
            "llm_explanation": explanation
        }
