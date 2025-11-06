import math
import ollama
import logging # [추가]
import asyncio # [추가]
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, TypedDict # [추가]

# LangChain/LangGraph 관련 임포트 [추가]
from langchain_core.messages import AIMessage
from langgraph.graph.message import MessagesState

# --- 로거 설정 [추가] ---
logger = logging.getLogger(__name__)

# --- 환경 변수 로드 (동일) ---
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

# =================================================================
#  GRAPH STATE 정의 [추가]
# =================================================================
class LoanState(MessagesState):
    """
    이 노드가 LangGraph와 주고받을 상태
    """
    # [입력] 워크플로우의 이전 노드에서 전달받을 값
    user_id: str
    plan_id: int # (참고: 코드가 plan_id 대신 user_id를 사용하도록 수정되었음)
    
    # [출력] 이 노드가 실행된 후의 최종 결과
    loan_result: Optional[Dict[str, Any]] = None

# =================================================================
# 🧠 [뇌] LoanAgent (IntentClassifierAgent 형식으로 리팩토링)
# =================================================================

class LoanAgent: # [이름 변경] LoanAgentNode -> LoanAgent
    """
    LoanAgent (대출 추천 에이전트)
    ---------------------------------------------------------
    기존 LoanAgentNode의 'run' 메서드(모든 툴킷 포함)를
    LangGraph의 단일 노드로 래핑(Wrapping)합니다.
    
    '페르소나'와 'TASK'는 _generate_explanation 메서드 내부에 정의됩니다.
    """

    def __init__(self, llm_model="qwen3:8b"):
        """
        LoanAgent를 초기화합니다.
        (기존 __init__과 100% 동일)
        """
        try:
            self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
            with self.engine.connect() as conn:
                pass
        except Exception as e:
            logger.error(f"DB 연결 실패: {e}", exc_info=True)
            raise
            
        self.llm_model = llm_model
        logger.info(f"LoanAgent (Wrapped-Node) 초기화 완료. (LLM: {llm_model})")

    # ------------------------------------------------
    # Tool 1 ~ 6 (기존의 모든 비공개 헬퍼 메서드)
    # (_get_region_price, _get_loan_product, _calc_monthly_payment, 
    #  _get_monthly_income, _recommend, _generate_explanation)
    #
    # [!] 이 메서드들은 단 하나도 수정할 필요 없이 그대로 복사/붙여넣기 합니다.
    # [!] _generate_explanation가 'ollama.chat' (동기)을 사용하는 것이
    #     이 패턴의 핵심입니다.
    # ------------------------------------------------
    def _get_region_price(self, region_name: str) -> Optional[Dict[str, Any]]:
        # (이전 코드와 100% 동일)
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
        query = text("""
            SELECT apartment_price, multi_price, officetel_price, detached_price
            FROM state WHERE region_nm LIKE :region LIMIT 1
        """)
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query, {"region": f"%{city_name}%"}).fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            logger.error(f"지역 시세 조회 실패 ({city_name}): {e}")
            return None

    def _get_loan_product(self) -> Optional[Dict[str, Any]]:
        # (이전 코드와 100% 동일)
        query = text("SELECT * FROM loan_product WHERE product_id = 1") 
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query).mappings().fetchone()
            if not row:
                logger.warning("경고: product_id = 1인 상품을 찾을 수 없습니다.")
                return None
            return dict(row)
        except Exception as e:
            logger.error(f"대출 상품 조회 실패 (product_id=1): {e}")
            return None

    def _calc_monthly_payment(self, principal: float, annual_rate: float, years: int) -> float:
        # (이전 코드와 100% 동일)
        monthly_rate = annual_rate / 12 / 100
        n = years * 12
        if n <= 0: return 0
        if monthly_rate == 0: return principal / n
        return principal * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)

    def _get_monthly_income(self, user: Dict[str, Any]) -> int:
        # (이전 코드와 100% 동일)
        job_type = user.get("job_type")
        try:
            if job_type in ["직장인", "공무원"]:
                if user.get("monthly_salary"): return int(user["monthly_salary"])
                elif user.get("income"): return int(user["income"]) // 12
            elif job_type in ["자영업", "프리랜서"]:
                if user.get("operating_income"): return int(user["operating_income"]) // 12
                elif user.get("annual_revenue"): return int(int(user["annual_revenue"]) * 0.2 // 12)
        except Exception as e:
            logger.error(f"소득 계산 중 오류 (사용자: {user.get('user_id')}): {e}")
            pass
        return 0

    def _recommend(self, user: Dict[str, Any], plan: Dict[str, Any], region: Optional[Dict[str, Any]], product: Dict[str, Any]):
        # (이전 코드와 100% 동일)
        try:
            target_price = int(plan["target_house_price"])
            available_assets = int(plan["available_assets"])
            credit_score = int(user["credit_score"]) if user.get("credit_score") else 700
        except Exception as e:
            logger.error(f"추천 로직: 사용자/계획 데이터 변환 실패: {e}")
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
            logger.error(f"상품 추천 계산 중 오류 (상품 ID: {product.get('product_id')}): {e}")
            return None, 0, 0

    def _generate_explanation(self, user: Dict[str, Any], plan: Dict[str, Any], loan: Dict[str, Any], shortage: int) -> str:
        # (이전 코드와 100% 동일)
        # [!] 이 프롬프트가 이 노드의 "페르소나"와 "TASK" 역할을 합니다.
        prompt = f"""
        [페르소나]
        당신은 친절하고 전문적인 우리은행의 주택담보대출 전문 상담원입니다. 
        고객의 상황을 공감하며 긍정적이고 명확한 어조로 설명해야 합니다.

        [TASK]
        아래 [고객 정보]와 [추천 상품]을 바탕으로, 왜 이 상품이 고객님께 적합한지 2~3문장의 간결한 추천 사유를 작성해 주세요.
        - 고객의 직업, 소득, 목표 주택 가격을 자연스럽게 언급하세요.
        - '월 상환액'과 '대출 실행 후 남은 금액'을 명확히 안내하는 데 집중하세요.
        
        [중요 지시]
        - [추천 상품] 섹션의 '대출 실행 후 남은 금액'({shortage:,}원)을 **반드시 정확하게** 읽어서 말해야 합니다.
        - 이 금액은 고객이 보유 자산({int(plan['available_assets']):,}원)으로 충당해야 할 금액임을 부드럽게 언급해 주세요.
        - 절대 다른 숫자를 지어내지 마세요.

        [고객 정보]
        - 직업: {user.get("job_type", "N/A")}
        - 신용점수: {user.get("credit_score", "N/A")}점
        - 추정 월소득: {self._get_monthly_income(user):,}원
        - 목표 주택 가격: {int(plan['target_house_price']):,}원
        - 보유 자산: {int(plan['available_assets']):,}원

        [추천 상품]
        - 상품명: {loan.get("product_name", loan.get("loan_name", "N/A"))}
        - 추천 대출액: {loan['loan_amount']:,}원
        - 금리: {loan['interest_rate']:.2f}%
        - 기간: {loan['period_years']}년
        - 월 상환액: {round(loan['monthly_payment']):,}원
        - 대출 실행 후 남은 금액 (고객 부담금): {shortage:,}원
        
        [추천 사유 작성]
        (여기에 2-3문장으로 작성)
        """
        try:
            # [!] self.llm_model (모델명)을 사용하고 ollama.chat (동기)을 호출
            response = ollama.chat(
                model=self.llm_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM 설명 생성 실패: {e}")
            return "(추천 사유 생성 중 오류가 발생했습니다. 관리자에게 문의하세요.)"


    # ------------------------------------------------
    # 7️⃣ [엔진] 'run' 메서드를 -> '_run_sync_engine'으로 이름 변경
    # ------------------------------------------------
    def _run_sync_engine(self, user_id: str, plan_id: int) -> Dict[str, Any]:
        """
        [메인 실행 엔진] LoanAgent의 전체 프로세스 (기존 'run' 메서드와 동일)
        
        이 함수는 '동기(Synchronous)'로 실행되며, LangGraph 노드에 의해
        별도의 스레드에서 호출(await asyncio.to_thread)됩니다.
        """
        
        try:
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

            # --- Tool 실행 ---
            product = self._get_loan_product()
            if not product:
                return {"message": "조회할 대출 상품(ID=1)이 없습니다."}

            # --- 핵심 로직 실행 ---
            best, loan_amount, monthly_payment = self._recommend(user, plan, None, product)
            if not best:
                return {"message": "고객님의 조건(LTV)으로는 대출이 불가능합니다."}

            # --- 결과 계산 ---
            remaining_after_loan = int(plan["target_house_price"]) - loan_amount
            shortage = remaining_after_loan
            if shortage < 0: shortage = 0

            # --- LLM Tool 실행 ---
            explanation = self._generate_explanation(user, plan, best, shortage)

            # --- DB 업데이트 ---
            monthly_income_val = self._get_monthly_income(user)
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE user_info
                    SET loan_amount = :loan_amount, last_recommend_date = NOW()
                    WHERE user_id = :user_id
                """), {
                    "loan_amount": loan_amount,
                    "user_id": user["user_id"]
                })

            # --- 최종 결과 반환 ---
            return {
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

        except ValueError as ve:
            logger.error(f"데이터 오류: {ve}")
            return {"message": f"오류: {ve}"}
        except Exception as e:
            logger.error(f"LoanAgent 엔진 실행 중 심각한 오류 발생: {e}", exc_info=True)
            return {"message": f"알 수 없는 오류가 발생했습니다: {e}"}

    # ------------------------------------------------
    # 8️⃣ [신규] LangGraph 노드 팩토리 (IntentClassifierAgent 스타일)
    # ------------------------------------------------
    def create_recommendation_node(self):
        """
        LangGraph에 등록할 '단일 대출 추천 노드'를 생성하여 반환합니다.
        (IntentClassifierAgent.create_intent_node와 동일한 구조)
        """
        
        # [핵심] 이 async 함수가 LangGraph의 '노드'가 됩니다.
        async def loan_recommendation_node(state: LoanState):
            logger.info("🔍 LoanAgent (Wrapped-Node): 노드 실행...")
            
            try:
                # 1. LangGraph State에서 입력 데이터를 가져옵니다.
                user_id = state.get("user_id")
                plan_id = state.get("plan_id") # (현재 로직상 무시됨)
                
                if not user_id:
                    raise ValueError("State에서 'user_id'를 찾을 수 없습니다.")

                # 2. [중요!] 동기(sync) 엔진인 '_run_sync_engine'을
                #    'asyncio.to_thread'를 사용해 별도 스레드에서 비동기 실행합니다.
                final_result = await asyncio.to_thread(
                    self._run_sync_engine, # 호출할 동기 함수
                    user_id=user_id,        # 함수의 인자
                    plan_id=plan_id         # 함수의 인자
                )
                
                # 3. 'run' 메서드의 결과를 LangGraph State에 반영합니다.
                if "message" in final_result: # 'run'이 오류를 반환한 경우
                     logger.warning(f"LoanAgent (Wrapped-Node): 노드 실행 중 오류: {final_result['message']}")
                     return {
                         "loan_result": final_result,
                         "messages": [AIMessage(content=f"[대출 추천 실패] {final_result['message']}")]
                     }

                logger.info(f"✅ LoanAgent (Wrapped-Node): 노드 완료. (추천: {final_result.get('loan_name')})")
                
                # 4. State 업데이트
                return {
                    # 'loan_result' 상태에 최종 딕셔너리를 저장
                    "loan_result": final_result, 
                    # 'messages' 상태에 LLM의 설명을 추가
                    "messages": [AIMessage(content=final_result.get("llm_explanation", "대출 추천이 완료되었습니다."))]
                }

            except Exception as e:
                # 5. 노드 래퍼(Wrapper) 자체의 예외 처리
                logger.error(f"❌ LoanAgent (Wrapped-Node) 래퍼 오류: {e}", exc_info=True)
                error_msg = f"대출 에이전트 래퍼 실행 실패: {e}"
                final_response = {"message": error_msg}
                
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "loan_result": final_response
                }
        
        # 6. '노드' 함수를 반환
        return loan_recommendation_node