import os
import re
import json
import math
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
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

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# ============================================================
# 🧠 SYSTEM PROMPT (현실적 대출 규제 반영)
# ============================================================
SYSTEM_PROMPT = SystemMessage(content="""
[페르소나(Persona)]
당신은 '우리은행 대출 컨설턴트 AI(WooriLoanAdvisor)'입니다.  
고객의 소득, 자산, 신용점수, 기존대출, 주택가격을 기반으로  
**현실적인 금융 규제(LTV, DSR, 금리, 지역규제)** 를 모두 고려하여  
대출 가능 여부와 한도를 계산합니다.

---

[TASK]
1️⃣ LTV (Loan To Value)
- 서울/수도권: 최대 40%
- 지방(비규제지역): 최대 60%
- 생애최초/신혼부부 & 주택가 6억 이하: 최대 70%
- 신용점수 750 이상 +5%, 650 미만 -5% (최대 70%)

2️⃣ DSR (Debt Service Ratio)
- DSR = (연간 부채상환액 ÷ 연소득) × 100 ≤ 40%
- 원리금균등상환 공식:
  월이율 r = (연이율 ÷ 12), n = 360개월
  A = P × [r(1+r)^n / ((1+r)^n - 1)]
  → P = A × ((1+r)^n - 1) / [r(1+r)^n]

3️⃣ 금리 4.5%, 기간 30년

4️⃣ 최종 계산
- LTV 기준 대출 = 주택가 × LTV
- DSR 기준 대출 = 연소득 40% 내 감당 가능한 원금
- 실제 대출 = min(LTV, DSR)
- 부족금 = 주택가 - (자산 + 대출액)
- 신용점수 < 600 → 대출 불가

---

[출력 형식(JSON)]
{
  "loan_amount": int,
  "shortage_amount": int,
  "LTV": int,
  "DSR": float,
  "is_loan_possible": bool,
  "reason": "요약 사유"
}
""")

# ============================================================
# 💼 LoanAgent (한 노드 = 한 에이전트)
# ============================================================
class LoanAgent:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3:8b", temperature=0.0)

    # ------------------------------
    # 🔹 DB 조회 함수
    # ------------------------------
    def fetch_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        with engine.connect() as conn:
            q = text("SELECT * FROM members WHERE user_id = :uid LIMIT 1")
            res = conn.execute(q, {"uid": user_id}).mappings().fetchone()
            return dict(res) if res else None

    def fetch_plan_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        with engine.connect() as conn:
            q = text("SELECT * FROM plans WHERE user_id = :uid ORDER BY plan_id DESC LIMIT 1")
            res = conn.execute(q, {"uid": user_id}).mappings().fetchone()
            return dict(res) if res else None

    def fetch_loan_product(self) -> Optional[Dict[str, Any]]:
        with engine.connect() as conn:
            q = text("SELECT * FROM loan_product ORDER BY product_id ASC LIMIT 1")
            res = conn.execute(q).mappings().fetchone()
            return dict(res) if res else None

    # ------------------------------
    # 🧮 LLM 계산
    # ------------------------------
    def calculate_loan_with_llm(self, user: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        아래는 고객의 재무 정보입니다.
        금융 규제 기준(LTV, DSR, 신용점수, 지역규제)을 고려해 현실적인 대출 가능 금액을 계산하세요.
        반드시 JSON 형식으로만 응답하세요.

        {{
            "hope_price": {user.get('hope_price', 0)},
            "initial_prop": {user.get('initial_prop', 0)},
            "salary": {user.get('salary', 0)},
            "credit_score": {user.get('credit_score', 700)},
            "existing_loans": {user.get('existing_loans', 0)},
            "hope_location": "{user.get('hope_location', '서울')}"
        }}
        """
        response = self.llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
        text = response.content.strip()
        logger.info(f"📨 LLM 응답(raw): {text}")

        try:
            match = re.search(r'\{[\s\S]*\}', text)
            return json.loads(match.group(0)) if match else None
        except Exception as e:
            logger.error(f"⚠️ JSON 파싱 실패: {e}")
            return None

    # ------------------------------
    # 💾 DB 반영
    # ------------------------------
    def update_db(self, user_id: int, result: Dict[str, Any], product: Dict[str, Any]):
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE plans 
                    SET loan_amount=:loan_amount, product_id=:pid
                    WHERE user_id=:uid ORDER BY plan_id DESC LIMIT 1
                """),
                {
                    "loan_amount": result.get("loan_amount", 0),
                    "pid": product.get("product_id"),
                    "uid": user_id,
                }
            )
            conn.execute(
                text("UPDATE members SET shortage_amount=:s WHERE user_id=:uid"),
                {"s": result.get("shortage_amount", 0), "uid": user_id}
            )
        logger.info(f"✅ DB 업데이트 완료 — user_id={user_id}")

    # ------------------------------
    # 🧩 LangGraph Node = run()
    # ------------------------------
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_id = state.get("user_id", 1)
        try:
            user = self.fetch_user_data(user_id)
            plan = self.fetch_plan_data(user_id)
            product = self.fetch_loan_product()

            if not user or not plan:
                msg = f"⚠️ 유효한 사용자({user_id}) 데이터가 없습니다."
                return {"messages": [AIMessage(content=msg)]}

            result = self.calculate_loan_with_llm(user)
            if not result:
                msg = "❌ 대출 계산 실패: LLM 응답 오류"
                return {"messages": [AIMessage(content=msg)]}

            self.update_db(user_id, result, product)

            # ✅ user_data 구성 (다음 노드 전달용)
            user_data = {
                "user_name": user.get("user_name"),
                "salary": user.get("salary", 0),
                "assets": user.get("initial_prop", 0),
                "invest_tendency": user.get("invest_tendency"),
                "income_usage_ratio": user.get("income_usage_ratio", 0),
                "credit_score": user.get("credit_score", 700),
            }

            msg = (
                f"💰 {product.get('product_name', '대출상품')} 기준 "
                f"대출금 {result['loan_amount']:,}원 / 부족금 {result['shortage_amount']:,}원"
            )

            return {
                "loan_result": result,
                "product_info": product,
                "user_data": user_data,  # ✅ 다음 노드로 전달
                "messages": [AIMessage(content=msg)],
            }

        except Exception as e:
            logger.error(f"LoanAgent 실행 오류: {e}", exc_info=True)
            return {
                "loan_result": {"status": "error", "message": str(e)},
                "messages": [AIMessage(content=f"❌ LoanAgent 실행 중 오류: {e}")],
            }
