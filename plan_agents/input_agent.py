import os
import re
import json
from dotenv import load_dotenv

# ------------------------------------------------
# 1. (수정) 필요한 LangChain 및 Typing 모듈 임포트
# ------------------------------------------------
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Enum, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import operator

# ------------------------------------------------
# (2) DB 설정 및 모델 정의 (님의 코드와 동일)
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

# (UserInfo, PlanInput 테이블 모델 - 님의 코드와 동일)
class UserInfo(Base):
    __tablename__ = "user_info"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False) 
    age = Column(Integer)
    gender = Column(Enum('M', 'F'))
    region = Column(String(100))
    income = Column(BigInteger)
    monthly_salary = Column(BigInteger)
    job_type = Column(String(50))
    employment_years = Column(Integer)

class PlanInput(Base):
    __tablename__ = "plan_input"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_info.user_id", ondelete="CASCADE"), nullable=False)
    target_house_price = Column(BigInteger)
    target_location = Column(String(100))
    housing_type = Column(String(50))
    available_assets = Column(BigInteger)
    income_usage_ratio = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

Base.metadata.create_all(engine)

# ------------------------------------------------
# (3) (필수) LangGraph '통합' 상태 정의
# (이 파일이 'plan_graph.py'에서 import될 때를 대비해, 
#  'plan_graph.py'의 AgentGraphState와 동일한 구조를 정의합니다)
# ------------------------------------------------
class AgentGraphState(TypedDict):
    """
    그래프 전체를 흐르는 공용 메모리
    (이 노드는 'messages'와 'user_id'를 읽고, 
     'input_completed', 'plan_input_data', 'plan_id', 'messages'를 씁니다)
    """
    # (Input)
    user_id: int
    messages: Annotated[List[BaseMessage], operator.add] 
    
    # (파일 경로)
    fund_data_path: Optional[str]
    savings_data_path: Optional[str]
    
    # (Flags)
    input_completed: bool
    validation_passed: bool
    
    # (Data)
    plan_input_data: Dict[str, Any]
    plan_id: Optional[int]
    user_mydata: Optional[Dict[str, Any]]
    loan_recommendations: Optional[Dict[str, Any]]
    savings_recommendations: Optional[Dict[str, Any]]
    fund_analysis_result: Optional[Dict[str, Any]]
    final_plan: Optional[Dict[str, Any]]
    error_message: Optional[str]


# ------------------------------------------------
# (4) 🟢 (수정) InputAgentNode 클래스 정의 🟢
# (PlanAgentNode -> InputAgentNode로 이름 변경)
# ------------------------------------------------
class InputAgentNode:
    """
    (수정) 턴제(Turn-based) 대화를 통해 사용자로부터 재무 계획 입력을 수집합니다.
    FastAPI 서버와 연동되며, 'while True' 루프를 사용하지 않습니다.
    """

    def __init__(self, model="qwen3:8b"):
        
        self.required_info = {
            "target_house_price": "목표 주택 가격 (원 단위, 숫자만)",
            "target_location": "주택 위치 (예: 서울 송파구, 부산광역시)",
            "housing_type": "주거지 형태 (아파트, 연립/다세대, 단독주택, 오피스텔 중 하나)",
            "available_assets": "현재 사용 가능한 자산 (원 단위, 숫자만)",
            "income_usage_ratio": "월급에서 저축/투자에 사용할 비율 (퍼센트, 숫자만)"
        }
        
        # --- (수정) LangChain 'chain'으로 변경 (LangSmith 추적 가능) ---
        try:
            llm = ChatOllama(model=model, temperature=0.0)
            system_prompt = f"""
            당신은 사용자의 대화 내용을 분석하여 재무 계획에 필요한 정보를 추출하는 AI입니다.
            대화 내용을 바탕으로 다음 항목들을 채워야 합니다.
            
            [추출 항목]
            {json.dumps(self.required_info, indent=2, ensure_ascii=False)}

            [규칙]
            1. 모든 항목을 반드시 채워야 합니다. 만약 정보가 부족하면 "정보 부족"이라고 명확히 표시하세요.
            2. 'housing_type'은 반드시 [아파트, 연립/다세대, 단독주택, 오피스텔] 중 하나여야 합니다.
            3. 사용자가 "10억"이라고 말하면 "1000000000"으로 변환해야 합니다.
            4. 'income_usage_ratio'는 "50%"라고 하면 "50"으로 숫자만 추출합니다.
            5. 최종 결과는 반드시 JSON 형식으로만 반환해야 합니다. 다른 설명은 붙이지 마세요.
            """
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "다음은 현재까지의 대화 기록입니다:\n\n{conversation_history_str}\n\n위 대화 기록을 바탕으로 5가지 항목을 JSON으로 추출하세요.")
            ])
            
            # (LangChain 체인 정의)
            self.llm_chain = prompt_template | llm | StrOutputParser() | self._parse_llm_json

        except Exception as e:
            print(f"LLM 로드 중 오류 발생: {e}")
            self.llm_chain = None
    
    # --- (추가) LangChain 체인용 파서 ---
    def _parse_llm_json(self, llm_output: str):
        try:
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted_data = json.loads(json_str)
                return extracted_data, None 
            else:
                raise json.JSONDecodeError("No JSON object found", llm_output, 0)
        except json.JSONDecodeError as e:
            print(f"[LLM 파싱 오류] LLM 응답: {llm_output}")
            return None, f"LLM이 유효한 JSON을 반환하지 못했습니다: {e}"

    # --- (수정) 님의 함수들을 클래스 내부 메서드로 변경 ---
    def _normalize_location(self, location: str):
        # (님의 normalize_location 코드)
        location = location.strip()
        if location.startswith("서울"):
            return location
        match = re.match(r"^(\S+시|\S+특별자치시)", location)
        if match:
            normalized = match.group(1)
            print(f"입력하신 지역 '{location}'은 '{normalized}' 평균 기준으로 처리됩니다.")
            return normalized
        return location

    def _save_to_db(self, data: dict, user_id: int):
        # (님의 save_to_db 코드)
        print(f"\n[DB 저장 시도] user_id: {user_id}")
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
            print(f"[DB 저장 완료] plan_id: {record.id}")
            return record.id 
        except Exception as e:
            session.rollback()
            print(f"[DB 저장 오류] 롤백 수행. 오류: {e}")
            return None
            
    def _summarize(self, responses):
        # (님의 summarize 코드)
        pass

    # ------------------------------------------------
    # (핵심 수정) 🟢 LangGraph 노드 실행 함수 🟢
    # (run_as_node -> run, 'while True' 루프 제거)
    # ------------------------------------------------
    def run(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        (수정) 이 함수가 LangGraph에 '노드'로 등록될 실제 실행 함수입니다.
        'while True' 루프 없이 '단 한 번' 실행됩니다.
        """
        print("\n--- [노드 0] '입력 수집 노드' 실행 ---")
        
        if not self.llm_chain:
            return {
                "input_completed": False, 
                "messages": [AIMessage(content="오류: LLM 모델 로드에 실패했습니다. 관리자에게 문의하세요.")]
            }

        # 1. State에서 현재 대화 기록 가져오기
        messages_list = state["messages"]
        user_id = state["user_id"]

        # 2. 대화 기록을 LLM에 전달할 문자열로 변환
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages_list])
        
        # 3. LLM 체인 '단 한 번' 호출 (정보 추출)
        extracted_data, error = self.llm_chain.invoke({"conversation_history_str": history_str})
        
        if error:
            print(f"[LLM 파싱 오류] {error}")
            return {
                "input_completed": False,
                "messages": [AIMessage(content=f"분석 중 오류가 발생했습니다. {error}. 다시 말씀해주시겠어요?")]
            }

        # 4. 정보가 부족한지 확인
        missing_info = []
        for key, desc in self.required_info.items():
            if not extracted_data.get(key) or extracted_data.get(key) == "정보 부족":
                missing_info.append(desc)
        
        # 5. 분기 처리 (정보 수집 완료 vs 추가 질문)
        if not missing_info:
            # 5-A: 정보 수집 완료
            print("[정보 수집 완료. DB 저장 및 다음 노드로 전달합니다.]")
            
            # (위치 정규화 적용)
            extracted_data["target_location"] = self._normalize_location(extracted_data["target_location"])
            
            # (DB 저장)
            plan_id = self._save_to_db(extracted_data, user_id)
            
            if plan_id:
                return {
                    "plan_input_data": extracted_data, # ⬅️ 다음 노드들이 사용할 데이터
                    "plan_id": plan_id,
                    "input_completed": True, # ⬅️ 다음 노드로 가라는 신호
                    "messages": [AIMessage(content=f"모든 정보(Plan ID: {plan_id})가 수집되었습니다. 검증(Validation)을 시작합니다.")]
                }
            else:
                return {
                    "input_completed": False, # ⬅️ 그래프 종료 신호
                    "messages": [AIMessage(content="정보 수집에 성공했으나, DB 저장에 실패했습니다.")]
                }
        else:
            # 5-B: 정보 부족 (추가 질문)
            missing_str = ", ".join(missing_info)
            ai_question = f"말씀 감사합니다. 추가적으로 {missing_str} 정보가 필요합니다. 알려주시겠어요?"
            print(f"[정보 부족] AI 추가 질문: {ai_question}")
            
            return {
                "input_completed": False, # ⬅️ 그래프 종료 신호 (대기)
                "messages": [AIMessage(content=ai_question)]
            }

# ------------------------------------------------
# (5) (테스트) VS Code에서 이 파일만 단독으로 실행
# (python agent/plan_agents/input_agent.py)
# ------------------------------------------------
if __name__ == "__main__":
    
    # (로깅 설정)
    logging.basicConfig(level=logging.INFO)
    
    # 1. 노드 인스턴스화
    input_node = InputAgentNode(model="qwen3:8b")

    # 2. (가상) 프론트엔드에서 첫 번째 요청이 들어옴
    print("--- 1차 호출 (정보 부족) ---")
    initial_messages = [HumanMessage(content="서울에 10억짜리 아파트 사고 싶어요")]
    initial_state_input = {
        "user_id": 1, # (테스트용 user_id)
        "messages": initial_messages
    }
    
    # 3. 노드 실행
    # (AgentGraphState의 모든 키가 필요하지만, 테스트를 위해 dict로 임시 전달)
    state_after_1st_call = input_node.run(initial_state_input)
    
    # 4. AI의 추가 질문 출력
    print("\n[AI 응답 (프론트엔드로 전달)]")
    # (messages는 BaseMessage 객체 리스트이므로 .content로 접근)
    print(state_after_1st_call["messages"][-1].content)
    
    # 5. (가상) 사용자가 AI의 질문에 답변함
    # (실제로는 state_after_1st_call["messages"]를 누적해야 함)
    messages_2 = [
        HumanMessage(content="서울에 10억짜리 아파트 사고 싶어요"),
        AIMessage(content=state_after_1st_call["messages"][-1].content),
        HumanMessage(content="현재 자산은 2억이고, 월급의 50%를 쓸 수 있어요.")
    ]
    state_input_2 = {
        "user_id": 1,
        "messages": messages_2
    }
    
    # 6. (가상) 프론트엔드에서 두 번째 요청이 들어옴
    print("\n\n--- 2차 호출 (정보 수집 완료) ---")
    state_after_2nd_call = input_node.run(state_input_2)
    
    # 7. 최종 결과 출력
    print("\n[AI 응답 (프론트엔드로 전달)]")
    print(state_after_2nd_call["messages"][-1].content)
    
    print("\n[최종 수집된 데이터]")
    print(json.dumps(state_after_2nd_call.get("plan_input_data"), indent=2, ensure_ascii=False))