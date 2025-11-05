import os
import re
import ollama
# ✅ [수정] Enum, BigInteger, DateTime, ForeignKey, func 추가
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Enum, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func # created_at을 위해 추가
from dotenv import load_dotenv
import json

# ... (DB 설정 - 변경 없음) ...
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

Base = declarative_base()
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}", echo=False)
Session = sessionmaker(bind=engine)
session = Session()


# ✅ [수정] 1. user_info 테이블 모델 (새 스키마)
class UserInfo(Base):
    __tablename__ = "user_info"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    age = Column(Integer)
    gender = Column(Enum('M', 'F')) # ENUM 타입 사용
    region = Column(String(100))
    income = Column(BigInteger)
    monthly_salary = Column(BigInteger)
    job_type = Column(String(50))
    employment_years = Column(Integer)
    # (참고: 기존의 username, email, created_at 등은 새 스키마에 없으므로 제거됨)


# (변경 없음) 2. plan_input 테이블 모델 (이전과 동일)
class PlanInput(Base):
    __tablename__ = "plan_input"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 이 FK 정의는 user_info의 PK가 user_id이므로 여전히 유효합니다.
    user_id = Column(Integer, ForeignKey("user_info.user_id", ondelete="CASCADE"), nullable=False)
    
    target_house_price = Column(BigInteger)
    target_location = Column(String(100))
    housing_type = Column(String(50))
    available_assets = Column(BigInteger)
    income_usage_ratio = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# 이 코드는 위의 두 테이블을 생성/업데이트합니다.
Base.metadata.create_all(engine)

# ------------------------------------------------
# PlanAgentNode (이하 모든 코드는 변경할 필요가 없습니다)
# ------------------------------------------------
class PlanAgentNode:
    """사용자 입력 수집 및 ValidationAgent와 연동"""

    def __init__(self, model="qwen3:8b"):
        self.model = model
        
        self.required_info = {
            "target_house_price": "목표 주택 가격 (원 단위, 숫자만)",
            "target_location": "주택 위치 (예: 서울 송파구, 부산광역시)",
            "housing_type": "주거지 형태 (아파트, 연립/다세대, 단독주택, 오피스텔 중 하나)",
            "available_assets": "현재 사용 가능한 자산 (원 단위, 숫자만)",
            "income_usage_ratio": "월급에서 저축/투자에 사용할 비율 (퍼센트, 숫자만)"
        }

    # ------------------------------------------------
    def normalize_location(self, location: str):
        # (변경 없음)
        location = location.strip()
        if location.startswith("서울"):
            return location
        match = re.match(r"^(\S+시|\S+특별자치시)", location)
        if match:
            normalized = match.group(1)
            print(f"입력하신 지역 '{location}'은 '{normalized}' 평균 기준으로 처리됩니다.")
            return normalized
        return location

    # ------------------------------------------------
    def save_to_db(self, data: dict, user_id: int):
        # (변경 없음)
        """
        DB 저장 (user_id를 인자로 받고, 숫자 변환 수행)
        """
        try:
            processed_data = {
                "target_house_price": int(data["target_house_price"]),
                "target_location": data["target_location"],
                "housing_type": data["housing_type"],
                "available_assets": int(data["available_assets"]),
                "income_usage_ratio": int(data["income_usage_ratio"]),
                "user_id": user_id
            }
            
            record = PlanInput(**processed_data)
            session.add(record)
            session.commit()
            print(f"\n[DB 저장 완료] user_id: {user_id}, plan_id: {record.id}")
            return record.id 
            
        except Exception as e:
            session.rollback()
            print(f"\n[DB 저장 오류] 롤백 수행. 오류: {e}")
            print(f"저장 시도 데이터: {data}")
            return None

    # ------------------------------------------------
    def summarize(self, responses):
        # (변경 없음)
        location_note = " (※ 서울특별시는 구 단위 기준, 그 외 지역은 시 평균 기준)"
        
        summary = f"""
        [입력 요약]
        ---------------------------------
        - 주택 가격: {responses['target_house_price']}원
        - 위치: {responses['target_location']}{location_note}
        - 주거지 형태: {responses['housing_type']}
        - 사용 가능 자산: {responses['available_assets']}원
        - 소득 활용 비율: {responses['income_usage_ratio']}%
        ---------------------------------
        """
        print(summary)

    # ------------------------------------------------
    def extract_info_with_llm(self, conversation_history):
        # (변경 없음)
        system_prompt = f"""
        당신은 사용자의 대화 내용을 분석하여 재무 계획에 필요한 정보를 추출하는 AI입니다.
        대화 내용을 바탕으로 다음 항목들을 채워야 합니다.
        
        [추출 항목]
        {json.dumps(self.required_info, indent=2, ensure_ascii=False)}

        [규칙]
        1. 모든 항목을 반드시 채워야 합니다. 만약 정보가 부족하면 "정보 부족"이라고 명확히 표시하세요.
        2. 'housing_type'은 반드시 [아파트, 연립/다세대, 단독주택, 오피스텔] 중 하나여야 합니다.
        3. 사용자가 "10억"이라고 말하면 "1000000000"으로, "2억 5천"이라고 하면 "250000000"으로 변환해야 합니다.
        4. 'income_usage_ratio'는 "50%"라고 하면 "50"으로 숫자만 추출합니다.
        5. 최종 결과는 반드시 JSON 형식으로만 반환해야 합니다. 다른 설명은 붙이지 마세요.
        
        [JSON 출력 예시]
        {{
          "target_house_price": "1000000000",
          "target_location": "서울 송파구",
          "housing_type": "아파트",
          "available_assets": "200000000",
          "income_usage_ratio": "50"
        }}
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history) 

        res = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.0} 
        )
        
        response_text = res["message"]["content"].strip()
        return response_text
    
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted_data = json.loads(json_str)
                return extracted_data, None 
            else:
                raise json.JSONDecodeError("No JSON object found", response_text, 0)
                
        except json.JSONDecodeError as e:
            print(f"[LLM 파싱 오류] LLM 응답: {response_text}")
            return None, f"LLM이 유효한 JSON을 반환하지 못했습니다: {e}"

    # ------------------------------------------------
    def run_conversational(self, conversation_history: list):
        # (변경 없음)
        print("재무 계획을 위한 상담을 시작하겠습니다. 자유롭게 말씀해주세요.")
        print("예: '서울 송파구에 10억짜리 아파트를 사고 싶어요. ...'")
        print("정보가 부족하면 제가 추가로 질문하겠습니다. (종료하시려면 '종료' 입력)\n")
        
        if not conversation_history:
             conversation_history.append({"role": "assistant", "content": "재무 계획 상담을 시작하겠습니다. 필요한 정보를 말씀해주세요."})
        elif len(conversation_history) > 1:
             print(f"AI: {conversation_history[-1]['content']}")

        while True:
            user_input = input("사용자: ")
            if user_input == "종료":
                print("상담을 종료합니다.")
                return None, conversation_history 

            conversation_history.append({"role": "user", "content": user_input})
            
            print("\n[대화 내용 분석 중...]")
            extracted_data, error = self.extract_info_with_llm(conversation_history)
            
            if error:
                print(f"[LLM 파싱 오류] {error}")
                print("다시 말씀해주시겠어요?")
                conversation_history.pop() 
                continue

            missing_info = []
            for key, desc in self.required_info.items():
                if not extracted_data.get(key) or extracted_data.get(key) == "정보 부족":
                    missing_info.append(desc)
            
            if not missing_info:
                print("[모든 정보 수집 완료. 검증 노드로 전달합니다.]")
                return extracted_data, conversation_history 
            
            else:
                missing_str = ", ".join(missing_info)
                ai_question = f"말씀 감사합니다. 추가적으로 {missing_str} 정보가 필요합니다. 알려주시겠어요?"
                print(f"AI: {ai_question}")
                conversation_history.append({"role": "assistant", "content": ai_question})
        
    # ------------------------------------------------
    def run_as_node(self, state: dict) -> dict:
        # (변경 없음)
        print("\n[Node: PlanInputNode 시작]")
        
        conversation_history = state.get("messages", [])
        
        responses, new_history = self.run_conversational(conversation_history) 
        
        state["messages"] = new_history 
        
        if responses:
            print(f"[Node: PlanInputNode] 정보 수집 완료.")
            state["responses"] = responses      
            state["input_completed"] = True     
        else:
            print(f"[Node: PlanInputNode] 사용자가 입력을 중단했습니다.")
            state["input_completed"] = False    
        
        print("[Node: PlanInputNode 완료]")
        return state