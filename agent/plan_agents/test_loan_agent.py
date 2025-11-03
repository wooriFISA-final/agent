# test/test_loan_agent_v7.py

from input_loan_agent.loan_agent_node import LoanAgentNode
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")


def test_loan_agent_v7(user_id=2, plan_id=2):
    print("✅ LoanAgentNode v7 테스트 시작")

    # -------------------------------
    # 1️⃣ DB 연결 및 데이터 확인
    # -------------------------------
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

    with engine.connect() as conn:
        user = conn.execute(
            text("SELECT * FROM user_info WHERE user_id=:id"), {"id": user_id}
        ).mappings().fetchone()
        plan = conn.execute(
            text("SELECT * FROM plan_input WHERE id=:id"), {"id": plan_id}
        ).mappings().fetchone()

    if not user:
        print(f"❌ user_info에 user_id={user_id} 데이터가 없습니다.")
        return
    if not plan:
        print(f"❌ plan_input에 id={plan_id} 데이터가 없습니다.")
        return

    print("\n📋 사용자 정보:")
    print({
        "user_id": user["user_id"],
        "name": user["name"],
        "job_type": user["job_type"],
        "credit_score": user["credit_score"],
        "income": user["income"],
        "monthly_salary": user.get("monthly_salary"),
    })

    print("\n📋 계획 정보:")
    print({
        "id": plan["id"],
        "target_house_price": plan["target_house_price"],
        "target_location": plan["target_location"],
        "available_assets": plan["available_assets"],
    })

    # -------------------------------
    # 2️⃣ LoanAgentNode 실행
    # -------------------------------
    agent = LoanAgentNode()
    result = agent.run(user_id=user_id, plan_id=plan_id)

    print("\n✅ LoanAgentNode 결과:")
    for k, v in result.items():
        print(f"{k}: {v}")

    # -------------------------------
    # 3️⃣ DB 반영 결과 확인
    # -------------------------------
    with engine.connect() as conn:
        updated_plan = conn.execute(
            text("""
                SELECT loan_amount, remaining_after_loan, recommended_loan_id, income_usage_ratio
                FROM plan_input WHERE id=:id
            """),
            {"id": plan_id},
        ).mappings().fetchone()

        updated_user = conn.execute(
            text("""
                SELECT last_loan_amount, last_recommended_loan_id, 
                       last_monthly_payment, last_shortage_amount, last_recommend_date
                FROM user_info WHERE user_id=:id
            """),
            {"id": user_id},
        ).mappings().fetchone()

    print("\n💾 DB 반영 결과 (plan_input):")
    print(dict(updated_plan) if updated_plan else "❌ 갱신 실패")

    print("\n💾 DB 반영 결과 (user_info):")
    print(dict(updated_user) if updated_user else "❌ 갱신 실패")

    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    test_loan_agent_v7(user_id=2, plan_id=2)
