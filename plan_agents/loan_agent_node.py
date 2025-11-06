import math
import re
import os
import json
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------------------------
# (수정) LangChain 및 LangGraph 모듈 임포트
# ------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated 
import operator

# ------------------------------------------------
# (1) DB 설정 (님의 코드와 동일)
# ------------------------------------------------
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

# ------------------------------------------------
# (2) (필수) LangGraph '통합' 상태 정의
# (plan_graph.py의 AgentGraphState와 동일해야 함)
# ------------------------------------------------
class AgentGraphState(TypedDict):
    user_id: int
    plan_id: Optional[int]
    user_mydata: Dict[str, Any]
    plan_input_data: Dict[str, Any]
    loan_recommendations: Dict[str, Any]
    # (messages, fund_recommendations 등 기타 필드들...)

# ------------------------------------------------
# (3) 🟢 (수정) LoanAgentNode 클래스 정의 🟢
# ------------------------------------------------
class LoanAgentNode:
    """
    (수정) LangGraph 'state'와 연동하고 LangChain 'chain'을 사용하는
    표준화된 대출 추천 에이전트 노드입니다.
    """

    def __init__(self, llm_model="qwen3:8b"):
        """
        LoanAgentNode를 초기화합니다.
        - DB 엔진을 생성합니다.
        - (수정) LLM 및 LangChain 'chain'을 초기화합니다.
        """
        try:
            self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
            with self.engine.connect() as conn:
                pass
        except Exception as e:
            print(f"DB 연결 실패: {e}")
            raise
            
        try:
            # --- 3-1. (수정) LLM 체인 정의 ---
            self.llm = ChatOllama(model=llm_model, temperature=0.0)
            
            # (님의 _generate_explanation 프롬프트를 템플릿으로 변경)
            self.explanation_prompt_template = ChatPromptTemplate.from_template(
                """
                [페르소나]
                당신은 친절하고 전문적인 우리은행의 주택담보대출 전문 상담원입니다. 
                고객의 상황을 공감하며 긍정적이고 명확한 어조로 설명해야 합니다.

                [TASK]
                아래 [고객 정보]와 [추천 상품]을 바탕으로, 왜 이 상품이 고객님께 적합한지 2~3문장의 간결한 추천 사유를 작성해 주세요.
                - 고객의 직업, 소득, 목표 주택 가격을 자연스럽게 언급하세요.
                - '월 상환액'과 '대출 실행 후 남은 금액'을 명확히 안내하는 데 집중하세요.
                
                [중요 지시]
                - [추천 상품] 섹션의 '대출 실행 후 남은 금액'({shortage:,}원)을 **반드시 정확하게** 읽어서 말해야 합니다.
                - 이 금액은 고객이 보유 자산({available_assets:,}원)으로 충당해야 할 금액임을 부드럽게 언급해 주세요.

                [고객 정보]
                - 직업: {job_type}
                - 신용점수: {credit_score}점
                - 추정 월소득: {monthly_income:,}원
                - 목표 주택 가격: {target_house_price:,}원
                - 보유 자산: {available_assets:,}원

                [추천 상품]
                - 상품명: {product_name}
                - 추천 대출액: {loan_amount:,}원
                - 금리: {interest_rate:.2f}%
                - 기간: {period_years}년
                - 월 상환액: {monthly_payment:,}원
                - 대출 실행 후 남은 금액 (고객 부담금): {shortage:,}원
                
                [추천 사유 작성]
                (여기에 2-3문장으로 작성)
                """
            )
            
            self.explanation_chain = self.explanation_prompt_template | self.llm | StrOutputParser()
            
        except Exception as e:
            print(f"LLM 로드 중 오류 발생: {e}")
            raise

        print(f"LoanAgentNode 초기화 완료. (LLM: {llm_model})")

    # ------------------------------------------------
    # (4) (수정) 'run' 메서드 - LangGraph의 진입점
    # ------------------------------------------------
    def run(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        [메인 실행] LoanAgentNode의 전체 프로세스를 실행합니다.
        (수정) LangGraph 'state'를 입력받아 'loan_recommendations'를 반환합니다.
        """
        print("\n--- [노드] '대출 추천 노드' 실행 ---")
        
        try:
            # 4-1. State에서 입력 받기
            user_id = state['user_id']
            # (plan_id는 loan_agent_node.py 원본에서 사용하지 않았으므로 user_id로 조회)

            with self.engine.connect() as conn:
                user = conn.execute(
                    text("SELECT * FROM user_info WHERE user_id=:id"), {"id": user_id}
                ).mappings().fetchone()
                
                plan = conn.execute(
                    text("SELECT * FROM plan_input WHERE user_id=:id ORDER BY created_at DESC LIMIT 1"), 
                    {"id": user_id}
                ).mappings().fetchone()

            if not user or not plan:
                if not user: raise ValueError(f"User(ID:{user_id})를 찾을 수 없습니다.")
                if not plan: raise ValueError(f"User(ID:{user_id})에 해당하는 plan_input 데이터를 찾을 수 없습니다.")

            # 4-2. Tool 실행 (상품 조회)
            product = self._get_loan_product() 
            if not product:
                return {"loan_recommendations": {"error": "조회할 대출 상품(ID=1)이 없습니다."}}

            # 4-3. Tool 실행 (핵심 추천 로직)
            best, loan_amount, monthly_payment = self._recommend(user, plan, None, product)
            if not best:
                return {"loan_recommendations": {"error": "고객님의 조건(LTV)으로는 대출이 불가능합니다."}}

            # 4-4. 결과 계산
            remaining_after_loan = int(plan["target_house_price"]) - loan_amount
            shortage = remaining_after_loan
            if shortage < 0: shortage = 0 
            monthly_income_val = self._get_monthly_income(user)
            
            # 4-5. LLM Tool 실행 (설명 생성)
            # (수정: ollama.chat 대신 self.explanation_chain.invoke 사용)
            explanation = self.explanation_chain.invoke({
                "shortage": shortage,
                "available_assets": int(plan['available_assets']),
                "job_type": user.get("job_type", "N/A"),
                "credit_score": user.get("credit_score", "N/A"),
                "monthly_income": monthly_income_val,
                "target_house_price": int(plan['target_house_price']),
                "product_name": best.get("product_name", best.get("loan_name", "N/A")),
                "loan_amount": loan_amount,
                "interest_rate": best.get("interest_rate"),
                "period_years": best.get("period_years"),
                "monthly_payment": round(monthly_payment)
            })

            # 4-6. DB 업데이트 (님의 코드와 동일)
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE user_info
                    SET loan_amount = :loan_amount,
                        last_recommend_date = NOW()
                    WHERE user_id = :user_id
                """), {
                    "loan_amount": loan_amount,
                    "user_id": user["user_id"]
                })

            # 4-7. 최종 결과 반환 (State 업데이트용)
            final_result = {
                "user_name": user.get("name"),
                "job_type": user.get("job_type"),
                "region": plan.get("target_location"),
                "loan_name": best.get("product_name", best.get("loan_name", "N/A")),
                "loan_amount": loan_amount,
                "interest_rate": best.get("interest_rate"),
                "monthly_payment": round(monthly_payment),
                "period_years": best.get("period_years"),
                "shortage_amount": shortage, 
                "credit_score": user.get("credit_score"),
                "monthly_income": monthly_income_val,
                "repayment_method": best.get("repayment_method"),
                "description": best.get("description", best.get("summary")),
                "llm_explanation": explanation
            }
            
            print("--- [노드 종료] '대출 추천 노드' 완료 ---")
            return {"loan_recommendations": final_result}

        except Exception as e:
            print(f"LoanAgentNode 실행 중 심각한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return {"loan_recommendations": {"error": f"알 수 없는 오류가 발생했습니다: {e}"}}

    # ------------------------------------------------
    # (5) 님의 'Tool' 함수들 (클래스 내부 메서드로 변경)
    # (코드는 님의 원본과 100% 동일합니다)
    # ------------------------------------------------
    def _get_region_price(self, region_name: str) -> Optional[Dict[str, Any]]:
        # ... (님의 _get_region_price 코드) ...
        parts = region_name.split()
        if not parts: return None
        if parts[0] == "서울특별시":
            city_name = " ".join(parts[:2]) if len(parts) > 1 else "서울특별시"
        elif parts[0].endswith("광역시"):
            city_name = parts[0]
        elif parts[0].endswith("특별자치시"):
            city_name = parts[0]
        elif parts[0].endswith("도"):
            city_name = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]
        else:
            city_name = parts[0]
        query = text("SELECT apartment_price, multi_price, officetel_price, detached_price FROM state WHERE region_nm LIKE :region LIMIT 1")
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query, {"region": f"%{city_name}%"}).fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            print(f"지역 시세 조회 실패 ({city_name}): {e}")
            return None

    def _get_loan_product(self) -> Optional[Dict[str, Any]]:
        # ... (님의 _get_loan_product 코드) ...
        query = text("SELECT * FROM loan_product WHERE product_id = 1") 
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query).mappings().fetchone()
            if not row:
                print("경고: product_id = 1인 상품을 찾을 수 없습니다.")
                return None
            return dict(row)
        except Exception as e:
            print(f"대출 상품 조회 실패 (product_id=1): {e}")
            return None

    def _calc_monthly_payment(self, principal: float, annual_rate: float, years: int) -> float:
        # ... (님의 _calc_monthly_payment 코드) ...
        monthly_rate = annual_rate / 12 / 100
        n = years * 12
        if n <= 0: return 0
        if monthly_rate == 0: return principal / n
        return principal * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)

    def _get_monthly_income(self, user: Dict[str, Any]) -> int:
        # ... (님의 _get_monthly_income 코드) ...
        job_type = user.get("job_type")
        try:
            if job_type in ["직장인", "공무원"]:
                if user.get("monthly_salary"): return int(user["monthly_salary"])
                elif user.get("income"): return int(user["income"]) // 12
            elif job_type in ["자영업", "프리랜서"]:
                if user.get("operating_income"): return int(user["operating_income"]) // 12
                elif user.get("annual_revenue"): return int(int(user["annual_revenue"]) * 0.2 // 12)
        except Exception as e:
            print(f"소득 계산 중 오류 (사용자: {user.get('user_id')}): {e}")
            pass 
        return 0

    def _recommend(self, user: Dict[str, Any], plan: Dict[str, Any], region: Optional[Dict[str, Any]], product: Dict[str, Any]):
        # ... (님의 _recommend 코드 - '무조건 추천' 로직) ...
        try:
            target_price = int(plan["target_house_price"])
            available_assets = int(plan["available_assets"])
            credit_score = int(user.get("credit_score", 700))
        except Exception as e:
            print(f"추천 로직: 사용자/계획 데이터 변환 실패: {e}")
            return None, 0, 0
        monthly_income = self._get_monthly_income(user)
        annual_income = monthly_income * 12
        if monthly_income <= 0:
            monthly_income = 1
            annual_income = 12
        try:
            max_ltv = float(product.get("max_ltv") or 70.0) 
            max_dsr = float(product.get("max_dsr") or 40.0)
            interest_rate = float(product.get("interest_rate") or 5.0)
            period_years = int(product.get("period_years") or 30) 
            possible_loan_by_ltv = target_price * (max_ltv / 100)
            possible_loan_by_dsr = annual_income * (max_dsr / 100) * (period_years / 2.5)
            possible_loan = min(possible_loan_by_ltv, possible_loan_by_dsr)
            needed_loan = target_price - available_assets
            if needed_loan <= 0: needed_loan = 0
            final_loan_amount = min(possible_loan, needed_loan)
            if final_loan_amount <= 0:
                final_loan_amount = possible_loan_by_ltv
                if final_loan_amount <= 0: return None, 0, 0
            monthly_payment = self._calc_monthly_payment(final_loan_amount, interest_rate, period_years)
            result_product = product.copy()
            result_product.update({
                "loan_amount": int(final_loan_amount),
                "monthly_payment": monthly_payment,
                "interest_rate": interest_rate,
                "period_years": period_years
            })
            return result_product, int(final_loan_amount), monthly_payment
        except Exception as e:
            print(f"상품 추천 계산 중 오류 (상품 ID: {product.get('product_id')}): {e}")
            return None, 0, 0