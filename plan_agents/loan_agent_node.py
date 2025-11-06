import math
import ollama
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Any

# 환경 변수 로드
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")


class LoanAgentNode:
    """
    LoanAgentNode (대출 추천 에이전트 노드)
    ---------------------------------------------------------
    사용자 정보(user_info), 주택 구매 계획(plan_input), 지역 시세(state),
    그리고 단일 대출 상품(loan_product) 정보를 종합하여
    사용자에게 대출 가능 여부와 금액을 계산하는 노드입니다.

    이 노드는 더 큰 LLM 에이전트 플로우의 일부로 작동하도록 설계되었습니다.
    'run' 메서드를 통해 실행되며, 필요한 모든 하위 작업(Tool)을 순차적으로 호출합니다.
    """

    def __init__(self, llm_model="qwen3:8b"):
        """
        LoanAgentNode를 초기화합니다.
        - DB 엔진을 생성합니다.
        - 내부 LLM 모델(설명 생성용)을 설정합니다.
        """
        try:
            self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
            # 연결 테스트
            with self.engine.connect() as conn:
                pass
        except Exception as e:
            print(f"DB 연결 실패: {e}")
            raise
            
        self.llm_model = llm_model
        print(f"LoanAgentNode 초기화 완료. (LLM: {llm_model})")

    # ------------------------------------------------
    # Tool 1: 지역별 시세 조회
    # ------------------------------------------------
    def _get_region_price(self, region_name: str) -> Optional[Dict[str, Any]]:
        """
        [Tool] 'state' 테이블에서 특정 지역의 평균 시세를 조회합니다.

        지역명 매핑 규칙:
        - 서울특별시는 구 단위까지 그대로 검색 (예: '서울특별시 송파구')
        - 그 외 광역시/도는 시 단위까지만 추출 (예: '부산광역시', '경기도 수원시')
        
        :param region_name: 조회할 지역명 (예: "서울특별시 강남구", "경기도 수원시")
        :return: 시세 정보 딕셔너리 (예: {'apartment_price': 50000, ...}) 또는 None
        """
        parts = region_name.split()

        if not parts:
            return None

        # ✅ 1️⃣ 서울특별시는 구 단위까지 사용
        if parts[0] == "서울특별시":
            city_name = " ".join(parts[:2]) if len(parts) > 1 else "서울특별시"
        # ✅ 2️⃣ '광역시'는 시까지만 사용
        elif parts[0].endswith("광역시"):
            city_name = parts[0]
        # ✅ 3️⃣ '특별자치시'(세종)도 그대로 사용
        elif parts[0].endswith("특별자치시"):
            city_name = parts[0]
        # ✅ 4️⃣ '도' 단위 지역은 '도 + 시'까지만 사용
        elif parts[0].endswith("도"):
            city_name = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]
        else:
            city_name = parts[0]

        query = text("""
            SELECT apartment_price, multi_price, officetel_price, detached_price
            FROM state
            WHERE region_nm LIKE :region
            LIMIT 1
        """)
        
        try:
            with self.engine.connect() as conn:
                row = conn.execute(query, {"region": f"%{city_name}%"}).fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            print(f"지역 시세 조회 실패 ({city_name}): {e}")
            return None

    # ------------------------------------------------
    # Tool 2: 단일 대출 상품 조회 (ID=1 고정)
    # ------------------------------------------------
    def _get_loan_product(self) -> Optional[Dict[str, Any]]:
        """
        [Tool] 'loan_product' 테이블에서 'product_id = 1'인 상품 1개만 조회합니다.
        
        :return: 대출 상품 딕셔너리 또는 None
        """
        # product_id가 1인 상품을 고정적으로 조회
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

    # ------------------------------------------------
    # Tool 3: 월 상환액 계산 (원리금 균등상환)
    # ------------------------------------------------
    def _calc_monthly_payment(self, principal: float, annual_rate: float, years: int) -> float:
        """
        [Tool] 원리금 균등 상환 방식의 월 상환액을 계산합니다.
        
        :param principal: 대출 원금
        :param annual_rate: 연 이자율 (예: 3.5)
        :param years: 대출 기간 (년)
        :return: 월 상환액
        """
        monthly_rate = annual_rate / 12 / 100
        n = years * 12
        
        if n <= 0:
            return 0
        if monthly_rate == 0:
            return principal / n
            
        return principal * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)

    # ------------------------------------------------
    # Tool 4: 직업유형별 월소득 추정
    # ------------------------------------------------
    def _get_monthly_income(self, user: Dict[str, Any]) -> int:
        """
        [Tool] 사용자 정보(user) 딕셔너리를 바탕으로 월 소득을 추정합니다.
        
        :param user: 'user_info' 테이블의 사용자 정보 딕셔너리
        :return: 추정된 월 소득 (정수)
        """
        job_type = user.get("job_type")
        
        try:
            if job_type in ["직장인", "공무원"]:
                if user.get("monthly_salary"):
                    return int(user["monthly_salary"])
                elif user.get("income"):
                    return int(user["income"]) // 12
            elif job_type in ["자영업", "프리랜서"]:
                if user.get("operating_income"):
                    return int(user["operating_income"]) // 12
                elif user.get("annual_revenue"):
                    # 연 매출의 20%를 순소득으로 가정 (임시 추정치)
                    return int(int(user["annual_revenue"]) * 0.2 // 12)
        except Exception as e:
            print(f"소득 계산 중 오류 (사용자: {user.get('user_id')}): {e}")
            pass # 오류 발생 시 0 반환
            
        return 0

    # ------------------------------------------------
    # Tool 5: 단일 대출 상품 평가 로직 (수정됨 - 무조건 추천)
    # ------------------------------------------------
    def _recommend(self, user: Dict[str, Any], plan: Dict[str, Any], region: Optional[Dict[str, Any]], product: Dict[str, Any]):
        """
        [Tool] 제공된 단일 상품을 사용자 조건과 관계없이 계산합니다.
        (LTV, DSR, 신용점수 등을 복합적으로 고려)
        
        :param user: 사용자 정보
        :param plan: 주택 구매 계획
        :param region: 지역 시세 (사용은 안되고 있으나 확장성 위해 유지)
        :param product: 평가할 단일 대출 상품
        :return: (추천 상품 딕셔너리, 추천 대출액, 월 상환액) 튜플. 부적합 시 (None, 0, 0)
        """
        try:
            target_price = int(plan["target_house_price"])
            available_assets = int(plan["available_assets"])
            credit_score = int(user["credit_score"]) if user.get("credit_score") else 700
        except Exception as e:
            print(f"추천 로직: 사용자/계획 데이터 변환 실패: {e}")
            return None, 0, 0

        monthly_income = self._get_monthly_income(user)
        annual_income = monthly_income * 12

        if monthly_income <= 0:
            print("소득 정보가 없어 추천이 불가능합니다.")
            # [수정] 소득이 없어도 LTV 기반으로는 계산을 시도하도록 변경 (월소득 1로 가정)
            monthly_income = 1
            annual_income = 12
            # return None, 0, 0 # 기존: 소득 없으면 탈락

        try:
            # --------------------------------------------------------------
            # ✅ [수정] 상품 정보가 0이거나 NULL일 경우, 계산을 위해 임의의 기본값(70, 40, 5, 30)을 사용합니다.
            # --------------------------------------------------------------
            max_ltv = float(product.get("max_ltv") or 70.0) 
            max_dsr = float(product.get("max_dsr") or 40.0)
            interest_rate = float(product.get("interest_rate") or 5.0)
            period_years = int(product.get("period_years") or 30) # 0년 방지

            # ❌ [삭제] 상품 정보가 부족(0 또는 NULL)해도 탈락시키지 않고 기본값으로 계산을 강행합니다.
            # if max_ltv == 0 or max_dsr == 0 or interest_rate == 0 or period_years <= 0:
            #     print(f"상품 정보 부족 (LTV/DSR/금리/기간): {product.get('product_id')}")
            #     return None, 0, 0 # 계산에 필요한 필수 값이 없으면 부적합

            # LTV / DSR 기반 대출 가능액
            possible_loan_by_ltv = target_price * (max_ltv / 100)
            possible_loan_by_dsr = annual_income * (max_dsr / 100) * (period_years / 2.5) # 임의의 가중치
            
            possible_loan = min(possible_loan_by_ltv, possible_loan_by_dsr)
            
            # 목표가 - 자산 = 필요한 금액
            needed_loan = target_price - available_assets
            if needed_loan <= 0:
                needed_loan = 0 # 자산이 더 많으면 대출 불필요
            
            # 실제 대출액은 (가능한도)와 (필요한 금액) 중 작은 값
            final_loan_amount = min(possible_loan, needed_loan)
            if final_loan_amount <= 0:
                # [수정] LTV/DSR로 계산된 대출액이 0 이하여도, LTV 기준 금액으로 강제 설정
                print("계산된 최종 대출액이 0 이하입니다. LTV 기준액으로 강제 조정합니다.")
                final_loan_amount = possible_loan_by_ltv # DSR 무시하고 LTV 값으로 설정
                if final_loan_amount <= 0:
                     print("LTV 기준액도 0 이하입니다. 추천 실패.")
                     return None, 0, 0 # LTV마저 0이면 실패

            monthly_payment = self._calc_monthly_payment(final_loan_amount, interest_rate, period_years)

            # ❌ [삭제] 소득 대비 상환 가능비율(DSR) 40% 초과 시 탈락시키는 로직을 제거합니다.
            # if monthly_payment > monthly_income * 0.4:
            #     print("월 상환액이 소득 대비 40%를 초과합니다.")
            #     return None, 0, 0 # 상환 부담이 크면 부적합

            # 모든 조건을 통과한 경우
            result_product = product.copy() # 원본 딕셔너리 복사
            result_product.update({
                "loan_amount": int(final_loan_amount),
                "monthly_payment": monthly_payment,
                "interest_rate": interest_rate, # 형 변환된 값으로 덮어쓰기
                "period_years": period_years # 형 변환된 값으로 덮어쓰기
            })
            
            return result_product, int(final_loan_amount), monthly_payment
        
        except Exception as e:
            print(f"상품 추천 계산 중 오류 (상품 ID: {product.get('product_id')}): {e}")
            return None, 0, 0 # 이 상품은 건너뛰고 다음 상품 계속

    # ------------------------------------------------
    # Tool 6: LLM 기반 추천 사유 생성
    # ------------------------------------------------
    def _generate_explanation(self, user: Dict[str, Any], plan: Dict[str, Any], loan: Dict[str, Any], shortage: int) -> str:
        """
        [Tool] LLM을 호출하여 사용자 맞춤형 추천 사유를 생성합니다.
        (이것이 LLM의 "페르소나"와 "TASK"가 정의되는 부분입니다.)
        
        [수정] shortage의 의미가 (목표가 - 대출액)으로 변경되었습니다.
        
        :param user: 사용자 정보
        :param plan: 주택 구매 계획
        :param loan: 추천된 대출 상품 정보
        :param shortage: [수정] 대출 실행 후 남은 금액 (목표가 - 대출액)
        :return: LLM이 생성한 추천 사유 텍스트
        """
        
        # --------------------------------------------------------------
        # ✅ LLM 페르소나 및 TASK 정의 (수정됨)
        # --------------------------------------------------------------
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
            response = ollama.chat(
                model=self.llm_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"LLM 설명 생성 실패: {e}")
            return "(추천 사유 생성 중 오류가 발생했습니다. 관리자에게 문의하세요.)"

    # ------------------------------------------------
    # 7️⃣ 실행 (메인 핸들러)
    # ------------------------------------------------
    def run(self, user_id: str, plan_id: int) -> Dict[str, Any]:
        """
        [메인 실행] LoanAgentNode의 전체 프로세스를 실행합니다.
        
        [수정] plan_id 대신 user_id를 사용하여 최신 계획을 조회합니다.
        
        1. DB에서 사용자(user)와 최신 계획(plan) 정보를 가져옵니다.
        2. "Tool"(_get_loan_product)을 호출하여 단일 상품을 가져옵니다.
        3. 핵심 "Tool"(_recommend)을 호출하여 상품이 적합한지 평가합니다.
        4. "Tool"(_generate_explanation)을 호출하여 LLM 추천 사유를 생성합니다.
        5. 계산된 결과를 DB(user_info)에 업데이트합니다. (plan_input 업데이트 제거)
        6. 최종 결과를 딕셔너리 형태로 반환합니다.
        
        :param user_id: 조회할 사용자 ID
        :param plan_id: (무시됨) LangGraph 호환성을 위해 인자는 유지하지만 사용하지 않음.
        :return: 추천 결과 딕셔너리
        """
        
        try:
            with self.engine.connect() as conn:
                user = conn.execute(
                    text("SELECT * FROM user_info WHERE user_id=:id"), {"id": user_id}
                ).mappings().fetchone()
                
                # [수정] plan_id 대신 user_id로 최신 계획 1개 조회 (created_at 기준)
                plan = conn.execute(
                    text("SELECT * FROM plan_input WHERE user_id=:id ORDER BY created_at DESC LIMIT 1"), 
                    {"id": user_id}
                ).mappings().fetchone()

            if not user or not plan:
                # [수정] 에러 메시지를 더 명확하게 변경
                if not user:
                    raise ValueError(f"User(ID:{user_id})를 찾을 수 없습니다.")
                if not plan:
                    raise ValueError(f"User(ID:{user_id})에 해당하는 plan_input 데이터를 찾을 수 없습니다.")

            # --- Tool 실행 (수정됨) ---
            # region = self._get_region_price(plan["target_location"]) # 현재 추천 로직에서 미사용
            product = self._get_loan_product() # products -> product
            if not product: # products -> product
                return {"message": "조회할 대출 상품(ID=1)이 없습니다."} # [수정] 메시지 명확화

            # --- 핵심 로직 실행 (수정됨) ---
            best, loan_amount, monthly_payment = self._recommend(user, plan, None, product) # products -> product
            if not best:
                # [수정] 이 로직은 이제 거의 발생하지 않음 (LTV마저 0일 때)
                return {"message": "고객님의 조건(LTV)으로는 대출이 불가능합니다."}

            # --- 결과 계산 (수정됨) ---
            # (목표가 - 대출액) = 대출 실행 후 남은 금액 (내가 내야 할 돈)
            remaining_after_loan = int(plan["target_house_price"]) - loan_amount
            
            # [수정] shortage = (목표가 - 대출액)
            shortage = remaining_after_loan
            if shortage < 0:
                shortage = 0 # (혹시 모를 경우 대비)
            
            # [기존 로직]
            # (남은 금액 - 내 자산) = 최종 부족 금액
            # shortage = remaining_after_loan - int(plan["available_assets"])
            # if shortage < 0:
            #     shortage = 0 # 내 자산이 더 많으면 부족분은 0원

            # --- LLM Tool 실행 ---
            # [수정] LLM에게 (목표가 - 대출액) 값을 'shortage'로 전달
            explanation = self._generate_explanation(user, plan, best, shortage)

            # --- DB 업데이트 ---
            monthly_income_val = self._get_monthly_income(user)
            income_usage_ratio = (monthly_payment / monthly_income_val * 100) if monthly_income_val > 0 else 0

            with self.engine.begin() as conn:
                # ❌ [삭제] plan_input 테이블 업데이트 로직 제거
                # conn.execute(text("""
                #     UPDATE plan_input
                #     SET loan_amount=:loan_amount,
                # ...
                # """), { ... })

                # ✅ [수정] user_info 테이블에 'loan_amount'만 업데이트
                conn.execute(text("""
                    UPDATE user_info
                    SET loan_amount = :loan_amount,
                        last_recommend_date = NOW()
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
                # [수정] shortage_amount의 의미가 변경됨 (최종 부족분 -> 대출 후 남은 금액)
                "shortage_amount": shortage, 
                "credit_score": user.get("credit_score"),
                "monthly_income": monthly_income_val,
                "repayment_method": best.get("repayment_method"),
                "description": best.get("description", best.get("summary")),
                "llm_explanation": explanation
            }

        except ValueError as ve:
            print(f"데이터 오류: {ve}")
            return {"message": f"오류: {ve}"}
        except Exception as e:
            print(f"LoanAgentNode 실행 중 심각한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return {"message": f"알 수 없는 오류가 발생했습니다: {e}"}