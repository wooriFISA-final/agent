import os
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
당신의 임무는 고객의 소득, 자산, 신용점수, 기존대출, 주택가격을 기반으로  
**현실적인 금융 규제(LTV, DSR, 금리, 지역규제)** 를 모두 고려하여  
대출 가능 여부와 한도를 계산하는 것입니다.

---

[TASK]

1️⃣ **LTV (Loan To Value)** — 담보가치 기준 한도  
- 서울/수도권: 최대 40%  
- 지방(비규제지역): 최대 60%  
- 단, 생애최초 or 신혼부부이고 주택가 6억 이하라면 최대 70%  
- 신용점수 750 이상이면 +5%, 650 미만이면 -5%  
- LTV는 절대 70%를 초과할 수 없습니다.

2️⃣ **DSR (Debt Service Ratio)** — 상환능력 기준 한도  
- DSR = (연간 부채상환액 ÷ 연소득) × 100  
- 규제기준: DSR ≤ 40%  
- 대출금 상환액은 “원리금균등상환” 공식을 이용합니다.  
  월이율 r = (연이율 ÷ 12), 상환개월 n = 30년 × 12 = 360개월  
  월상환액 A = P × [r(1+r)^n / ((1+r)^n - 1)]  
  → P(대출원금) = A × ((1+r)^n - 1) / [r(1+r)^n]

3️⃣ **대출금리 및 기간**
- 기본 금리: 4.5% / 연  
- 상환 기간: 30년

4️⃣ **결정 로직**
- LTV 기준 대출 한도 = 주택가격 × 적용 LTV  
- DSR 기준 대출 한도 = 연소득의 40% 이내에서 감당 가능한 원금 계산  
- 실제 대출 가능액 = min(LTV 기준, DSR 기준)  
- 부족금액 = 주택가격 - (보유자산 + 대출 가능액)
- 만약 신용점수 < 600이면 대출 불가

5️⃣ **출력 형식**
아래 형식의 **JSON만 출력**하세요. (백틱, 설명문 금지)
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
# 💼 LoanAgent
# ============================================================
class LoanAgent:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3:8b", temperature=0.0)

    # ------------------------------
    # 🔹 사용자 및 상품 데이터 조회
    # ------------------------------
    def fetch_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        with engine.connect() as conn:
            query = text("SELECT * FROM members WHERE user_id = :uid LIMIT 1")
            result = conn.execute(query, {"uid": user_id}).mappings().fetchone()
            return dict(result) if result else None

    def fetch_plan_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        with engine.connect() as conn:
            query = text("SELECT * FROM plans WHERE user_id = :uid ORDER BY plan_id DESC LIMIT 1")
            result = conn.execute(query, {"uid": user_id}).mappings().fetchone()
            return dict(result) if result else None

    def fetch_loan_product(self) -> Optional[Dict[str, Any]]:
        with engine.connect() as conn:
            query = text("SELECT * FROM loan_product LIMIT 1")
            result = conn.execute(query).mappings().fetchone()
            return dict(result) if result else None

    # ------------------------------
    # 🧮 LLM 기반 대출 계산 수행
    # ------------------------------
    def calculate_loan_with_llm(self, user: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM에게 현실적 금융 규제(LTV, DSR 등)를 반영한 계산을 직접 맡김
        """
        prompt = f"""
        아래는 고객의 재무 정보입니다.
        한국 금융 규제 기준(LTV, DSR, 신용점수, 지역별 제한)을 적용하여 현실적인 대출 가능 금액과 부족금액을 계산하세요.
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

        messages = [SYSTEM_PROMPT, HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        content = response.content.strip()
        logger.info(f"📨 LLM 응답: {content}")

        # -----------------------------
        # ✅ JSON 파싱 안정화 처리
        # -----------------------------
        def extract_json(text: str) -> Optional[Dict[str, Any]]:
            import re
            try:
                match = re.search(r'\{[\s\S]*\}', text)
                if not match:
                    return None
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"⚠️ JSON 파싱 실패: {e}")
                return None

        parsed = extract_json(content)

        if not parsed:
            logger.error(f"❌ JSON 파싱 오류 - 원문:\n{content}")
            parsed = {
                "loan_amount": 0,
                "shortage_amount": 0,
                "LTV": 0,
                "DSR": 0,
                "is_loan_possible": False,
                "reason": "파싱 오류로 계산 실패"
            }

        return parsed

    # ------------------------------
    # 💾 DB 업데이트
    # ------------------------------
    def update_db(self, user_id: int, loan_result: Dict[str, Any]) -> None:
        with engine.begin() as conn:
            # plans 테이블 업데이트
            conn.execute(
                text("""
                    UPDATE plans 
                    SET loan_amount = :loan_amount 
                    WHERE user_id = :uid 
                    ORDER BY plan_id DESC LIMIT 1
                """),
                {"loan_amount": loan_result["loan_amount"], "uid": user_id}
            )

            # members 테이블 업데이트
            conn.execute(
                text("""
                    UPDATE members 
                    SET shortage_amount = :shortage 
                    WHERE user_id = :uid
                """),
                {"shortage": loan_result["shortage_amount"], "uid": user_id}
            )

        logger.info(f"✅ DB 업데이트 완료 (user_id={user_id})")

    # ------------------------------
    # 🧩 메인 실행 함수
    # ------------------------------
    def run(self, user_id: int) -> Dict[str, Any]:
        user = self.fetch_user_data(user_id)
        plan = self.fetch_plan_data(user_id)
        product = self.fetch_loan_product()

        if not user or not plan:
            logger.warning(f"⚠️ 유효한 사용자({user_id}) 데이터 없음")
            return {"status": "error", "message": "유효한 사용자 데이터가 없습니다."}

        # ✅ LLM 계산
        result = self.calculate_loan_with_llm(user, plan)

        # ✅ DB 반영
        self.update_db(user_id, result)

        # ✅ 결과 요약
        summary = result.get("reason", "대출 계산 완료")
        msg = f"💰 대출 가능 금액: {result['loan_amount']:,}원 / 부족 금액: {result['shortage_amount']:,}원"

        return {
            "status": "success",
            "loan_result": result,
            "summary": summary + "\n" + msg
        }

    # ------------------------------
    # ⚙️ LangGraph용 노드 생성
    # ------------------------------
    def create_recommendation_node(self):
        async def recommendation_node(state):
            user_id = state.get("user_id", 1)
            try:
                result = self.run(user_id)
                if result["status"] == "success":
                    msg = f"💰 대출 계산 완료 — 예상 대출금 {result['loan_result']['loan_amount']:,}원"
                else:
                    msg = f"❌ 대출 계산 실패: {result['message']}"

                # ✅ 여기 수정: loan_result → loan_data
                return {
                    "loan_data": result["loan_result"],
                    "messages": [AIMessage(content=msg)]
                }

            except Exception as e:
                logger.error(f"LoanAgent 노드 오류: {e}", exc_info=True)
                return {
                    "loan_data": {"status": "error", "message": str(e)},
                    "messages": [AIMessage(content=f"❌ LoanAgent 실행 중 오류: {e}")]
                }

        return recommendation_node
