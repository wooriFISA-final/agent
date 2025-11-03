import os
import re
import ollama
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# 환경설정 및 ORM 초기화
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

Base = declarative_base()
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}", echo=False)
Session = sessionmaker(bind=engine)
session = Session()

class PlanInput(Base):
    __tablename__ = "plan_input"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_house_price = Column(String(50))
    target_location = Column(String(100))
    housing_type = Column(String(50))
    available_assets = Column(String(50))
    target_period_years = Column(String(50))
    saving_ratio = Column(Integer)
    investment_ratio = Column(Integer)
    income_usage_ratio = Column(String(50))

Base.metadata.create_all(engine)

# PlanAgentNode
class PlanAgentNode:
    """Agentf 전체 기능을 수행하는 단일 노드"""
    def __init__(self, model="qwen3:8b"):
        self.model = model

        # 내부 질문 리스트
        self.questions = [
            ("target_house_price", "목표 주택 가격이 얼마인가요? (원 단위로 입력해주세요)"),
            ("target_location", "주택 위치는 어디인가요? (예: 서울 송파구)"),
            ("housing_type", "주거지 형태를 선택해주세요 (1: 아파트, 2: 연립/다세대, 3: 단독주택)"),
            ("available_assets", "현재 사용 가능한 자산은 얼마인가요? (원 단위로 입력해주세요)"),
            ("target_period_years", "목표 달성 기간은 몇 년인가요?"),
            ("saving_investment_ratio", "예/적금과 투자 비율은 어떻게 나누시겠어요? (예: 60 대 40)"),
            ("income_usage_ratio", "월급에서 저축/투자에 사용할 비율은 몇 퍼센트인가요?")
        ]

    def ask_llm(self, question: str):
        """LLM이 질문을 그대로 출력"""
        system_prompt = (
            "너는 재무 상담 AI이지만, 질문을 절대 바꾸거나 요약하지 말아야 해. "
            "입력받은 문장을 그대로 출력해야 하며, 말투나 어미도 수정하지 마. "
            "출력 언어는 반드시 한국어다."
        )
        res = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return res["message"]["content"].strip()

    def parse_input(self, key, value):
        """예/적금과 투자 비율 처리"""
        if key == "saving_investment_ratio":
            parts = [int(x) for x in re.findall(r"\d+", value)]
            if len(parts) == 2:
                return {"saving_ratio": parts[0], "investment_ratio": parts[1]}
            elif len(parts) == 1:
                return {"saving_ratio": parts[0], "investment_ratio": 100 - parts[0]}
            else:
                return {"saving_ratio": 50, "investment_ratio": 50}
        return value

    def save_to_db(self, data):
        """DB 저장"""
        record = PlanInput(**data)
        session.add(record)
        session.commit()
        print("\n[DB 저장 완료]")

    def summarize(self, responses):
        """입력 요약"""
        summary = f"""
        사용자가 입력한 재무 계획 요약

        - 주택 가격: {responses['target_house_price']}원
        - 위치: {responses['target_location']}
        - 주거지 형태: {responses['housing_type']}
        - 사용 가능 자산: {responses['available_assets']}원
        - 목표 기간: {responses['target_period_years']}년
        - 예금 비율: {responses['saving_ratio']}%
        - 투자 비율: {responses['investment_ratio']}%
        - 소득 활용 비율: {responses['income_usage_ratio']}%
        """
        print(summary)

    def run(self):
        """Agentf 단일 노드 실행"""
        responses = {}

        for key, question in self.questions:
            # LLM 질문 출력
            llm_question = self.ask_llm(question)
            print(f"\n {llm_question}")

            # 사용자 입력 처리
            if key == "housing_type":
                print("1. 아파트\n2. 연립/다세대\n3. 단독주택")
                while True:
                    choice = input("번호를 선택해주세요 (1~3): ")
                    if choice == "1":
                        responses["housing_type"] = "아파트"
                        break
                    elif choice == "2":
                        responses["housing_type"] = "연립/다세대"
                        break
                    elif choice == "3":
                        responses["housing_type"] = "단독주택"
                        break
                    else:
                        print("잘못된 입력입니다. 1~3 중 하나를 입력해주세요.")
                continue

            answer = input("사용자: ")
            parsed = self.parse_input(key, answer)

            if isinstance(parsed, dict):
                responses.update(parsed)
            else:
                responses[key] = parsed

        # DB 저장 및 요약 출력
        self.save_to_db(responses)
        self.summarize(responses)

# 실행
if __name__ == "__main__":
    print("단일 노드 PlanAgentNode 시작 (LLM: qwen3:8b)\n")
    agent_node = PlanAgentNode()
    agent_node.run()
