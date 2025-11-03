import os
import re
import ollama
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from input_loan_agent.validation_agent import ValidationAgent


# ------------------------------------------------
# 환경설정 및 ORM 초기화
# ------------------------------------------------
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

Base = declarative_base()
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}", echo=False)
Session = sessionmaker(bind=engine)
session = Session()


# ------------------------------------------------
# PlanInput 테이블 정의
# ------------------------------------------------
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


# ------------------------------------------------
# PlanAgentNode (사용자 입력 수집 및 검증)
# ------------------------------------------------
class PlanAgentNode:
    """사용자 입력 수집 및 ValidationAgent와 연동"""

    def __init__(self, model="qwen3:8b"):
        self.model = model
        self.validator = ValidationAgent()  # ✅ ValidationAgent 인스턴스화

        self.questions = [
            ("target_house_price", "목표 주택 가격이 얼마인가요? (원 단위로 입력해주세요)"),
            ("target_location", 
             "주택 위치는 어디인가요? (예: 서울 송파구 / 그 외 지역은 시까지만 입력해주세요. 예: 부산광역시, 세종특별자치시)"),
            ("housing_type", "주거지 형태를 선택해주세요 (1: 아파트, 2: 연립/다세대, 3: 단독주택, 4: 오피스텔)"),
            ("available_assets", "현재 사용 가능한 자산은 얼마인가요? (원 단위로 입력해주세요)"),
            ("target_period_years", "목표 달성 기간은 몇 년인가요?"),
            ("saving_investment_ratio", "예/적금과 투자 비율은 어떻게 나누시겠어요? (예: 60 대 40)"),
            ("income_usage_ratio", "월급에서 저축/투자에 사용할 비율은 몇 퍼센트인가요?")
        ]

    # ------------------------------------------------
    def ask_llm(self, question: str):
        """LLM이 질문을 그대로 출력 + 간단 설명 추가"""
        system_prompt = (
            "너는 재무 상담 AI야. "
            "사용자가 입력한 질문을 먼저 그대로 출력한 다음, "
            "간단한 예시나 안내 문장을 한 줄로 덧붙여줘. "
            "예시는 자연스럽게 '~해주세요' 어미로 끝내."
        )
        res = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return res["message"]["content"].strip()

    # ------------------------------------------------
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

    # ------------------------------------------------
    def normalize_location(self, location: str):
        """서울 외 지역은 시 단위까지만 남기고 평균값 기준으로 처리"""
        location = location.strip()

        # 서울특별시는 구 단위 유지
        if location.startswith("서울"):
            return location

        # 나머지는 '시' 또는 '특별자치시'까지만 남김
        match = re.match(r"^(\S+시|\S+특별자치시)", location)
        if match:
            normalized = match.group(1)
            print(f"입력하신 지역 '{location}'은 '{normalized}' 평균 기준으로 처리됩니다.")
            return normalized
        
        # 기본 fallback (입력값 그대로 사용)
        return location

    # ------------------------------------------------
    def save_to_db(self, data):
        """DB 저장"""
        record = PlanInput(**data)
        session.add(record)
        session.commit()
        print("\n[DB 저장 완료]")

    # ------------------------------------------------
    def summarize(self, responses):
        """입력 요약"""
        location_note = " (※ 서울특별시는 구 단위 기준, 그 외 지역은 시 평균 기준)"
        summary = f"""
        [입력 요약]
        ---------------------------------
        - 주택 가격: {responses['target_house_price']}원
        - 위치: {responses['target_location']}{location_note}
        - 주거지 형태: {responses['housing_type']}
        - 사용 가능 자산: {responses['available_assets']}원
        - 목표 기간: {responses['target_period_years']}년
        - 예금 비율: {responses['saving_ratio']}%
        - 투자 비율: {responses['investment_ratio']}%
        - 소득 활용 비율: {responses['income_usage_ratio']}%
        ---------------------------------
        """
        print(summary)

    # ------------------------------------------------
    def run(self):
        """전체 입력 및 검증 파이프라인"""
        responses = {}

        for key, question in self.questions:
            llm_question = self.ask_llm(question)
            print(f"\n{llm_question}")

            # 주거지 형태 선택
            if key == "housing_type":
                print("1. 아파트\n2. 연립/다세대\n3. 단독주택\n4. 오피스텔")
                while True:
                    choice = input("번호를 선택해주세요 (1~4): ")
                    mapping = {"1": "아파트", "2": "연립/다세대", "3": "단독주택", "4": "오피스텔"}
                    if choice in mapping:
                        responses[key] = mapping[choice]
                        break
                    else:
                        print("잘못된 입력입니다. 다시 선택해주세요.")
                continue

            # 사용자 입력
            answer = input("사용자: ")
            parsed = self.parse_input(key, answer)

            # 위치 정규화 처리 (서울 외 지역은 시 단위로 단순화)
            if key == "target_location":
                parsed = self.normalize_location(parsed)

            # 딕셔너리면 unpack, 아니면 그대로 저장
            if isinstance(parsed, dict):
                responses.update(parsed)
            else:
                responses[key] = parsed

        # ✅ 1️⃣ 입력값 검증
        print("\n[입력값 검증 중...]")
        if not self.validator.run(responses):  # DB 기반 검증 수행
            print("입력값 검증 실패. 저장이 취소되었습니다.")
            return

        # ✅ 2️⃣ 검증 통과 시 저장
        self.save_to_db(responses)
        self.summarize(responses)


# # ------------------------------------------------
# # 실행
# # ------------------------------------------------
# if __name__ == "__main__":
#     print("PlanAgentNode 시작 (입력 → 검증 → 저장)\n")
#     agent_node = PlanAgentNode()
#     agent_node.run()
