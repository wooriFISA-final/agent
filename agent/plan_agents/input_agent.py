import os
import re
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Enum, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from dotenv import load_dotenv

# LangChain / LangGraph 관련
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
# [수정!] MessagesState는 이제 'TYPE_CHECKING'에만 사용됩니다.
# from langgraph.graph import MessagesState 
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field, field_validator

# [신규!] 순환 참조를 피하기 위한 '타입 힌트' 전용 임포트
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # 이 코드는 실행 시에는 무시되지만,
    # VSCode 같은 IDE가 타입을 인식하도록 도와줍니다.
    # [!] plan_graph.py의 위치에 따라 . 또는 ..을 조정해야 합니다.
    from ..plan_graph import GraphState


# --- 로거 설정 ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- DB 설정 ---
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

Base = declarative_base()
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}", echo=False)
Session = sessionmaker(bind=engine)


# --- 1️⃣ user_info 테이블 ---
class UserInfo(Base):
    __tablename__ = "user_info"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    age = Column(Integer)
    job_type = Column(String(50))
    employment_years = Column(Integer)


# --- 2️⃣ plan_input 테이블 ---
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


# --- 유틸리티 함수 ---
def normalize_location(location: str):
    location = location.strip()
    if location.startswith("서울"):
        return location
    match = re.match(r"^(\S+시|\S+특별자치시)", location)
    if match:
        normalized = match.group(1)
        logger.info(f"입력 지역 '{location}'은 '{normalized}' 기준으로 처리됩니다.")
        return normalized
    return location


def summarize_plan(data: dict):
    location_note = " (※ 서울특별시는 구 단위 기준, 그 외 지역은 시 평균 기준)"
    summary = f"""
    [입력 요약]
    ---------------------------------
    - 주택 가격: {data.get('target_house_price', 'N/A')}원
    - 위치: {data.get('target_location', 'N/A')}{location_note}
    - 주거지 형태: {data.get('housing_type', 'N/A')}
    - 사용 가능 자산: {data.get('available_assets', 'N/A')}원
    - 소득 활용 비율: {data.get('income_usage_ratio', 'N/A')}%
    ---------------------------------
    """
    logger.info(summary)
    return summary


# --- Pydantic 모델 정의 ---
class ExtractedInfo(BaseModel):
    target_house_price: Optional[str] = Field(description="목표 주택 가격 (원 단위, 숫자만)")
    target_location: Optional[str] = Field(description="주택 위치 (예: 서울 송파구)")
    housing_type: Optional[str] = Field(description="주거지 형태 (아파트, 오피스텔 등)")
    available_assets: Optional[str] = Field(description="현재 사용 가능한 자산 (원 단위, 숫자만)")
    income_usage_ratio: Optional[str] = Field(description="월급에서 저축/투자에 사용할 비율 (퍼센트, 숫자만)")


class ValidatedPlanInput(BaseModel):
    user_id: int
    target_house_price: int
    target_location: str
    housing_type: str
    available_assets: int
    income_usage_ratio: int

    @field_validator('target_location')
    def validate_location(cls, v):
        return normalize_location(v)


# --- ✅ PlanInputAgent 정의 ---
class PlanInputAgent:
    """
    대화형으로 재무 계획 입력을 수집하고 검증하는 Agent.
    LangGraph에서 사용할 노드 팩토리(create_..._node)를 제공합니다.
    """

    def __init__(self, model="qwen3:8b"):
        self.llm = ChatOllama(model=model, temperature=0.0)
        self.required_info_schema = ExtractedInfo.model_json_schema()["properties"]

        self.system_prompt = SystemMessage(content=f"""
        당신은 사용자의 대화 내용을 분석하여 재무 계획에 필요한 정보를 추출하는 AI입니다.
        다음 항목들을 JSON 형태로 추출하세요:

        {json.dumps(self.required_info_schema, indent=2, ensure_ascii=False)}

        규칙:
        1. 대화 내용에서 알 수 있는 항목만 추출.
        2. 정보가 부족한 항목은 JSON에서 제외.
        3. JSON 형식만 반환 (설명 금지).

        예시:
        {{
          "target_house_price": "1000000000",
          "target_location": "서울 송파구"
        }}
        """)

    # -------------------------------
    # 1️⃣ 정보 추출 노드
    # -------------------------------
    def create_extraction_node(self):
        
        # 'state' 타입은 'Any' (혹은 비워둠)
        async def extraction_node(state): 
            
            # '지연 임포트'
            try:
                from agent.plan_graph import GraphState
            except ImportError:
                from ..plan_graph import GraphState
            
            state: "GraphState" = state 
            
            logger.info("ℹ️ PlanInputAgent: 정보 추출 중...")

            # [!!!] 이 'try' 블록이 중요합니다 [!!!]
            try:
                history_messages = state.get("messages", [])
                llm_messages = [self.system_prompt] + history_messages

                response = await self.llm.ainvoke(llm_messages)
                response_text = response.content.strip()

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                
                # 1. 파싱 실패 시
                if not json_match:
                    raise json.JSONDecodeError("LLM 응답에서 JSON 객체를 찾지 못함", response_text, 0)

                # 2. 'extracted_data' 정의
                extracted_data = json.loads(json_match.group(0))
                
                # (plan_graph.py의 병합 함수가 'None'을 처리해 줌)
                current_info = state.get("extracted_info", {}) 
                
                # 3. 'extracted_data' 사용
                # [!] 2번과 3번은 *반드시* 같은 try 블록 안에 있어야 합니다.
                current_info.update(extracted_data) 
                
                parsed_info = ExtractedInfo(**current_info)

                logger.info(f"✅ 정보 추출/업데이트 완료: {parsed_info.model_dump_json(exclude_unset=True)}")

                # 4. 성공 시 반환
                return {"extracted_info": parsed_info.model_dump(exclude_unset=True)}

            # 5. 1~3번에서 뭐 하나라도 실패하면...
            except Exception as e:
                # 6. 여기가 실행됨 (update는 건너뜀)
                logger.error(f"❌ PlanInputAgent(추출) 오류: {e}", exc_info=True)
                return {"messages": [AIMessage(content=f"정보 추출 중 오류 발생: {e}")]}
        
        return extraction_node

    # -------------------------------
    # 2️⃣ 완전성 검사 노드
    # -------------------------------
    def create_check_completeness_node(self):
        
        # [수정!] state: MessagesState -> state
        async def check_completeness_node(state):
            
            # [신규!] '지연 임포트'
            try:
                from agent.plan_graph import GraphState
            except ImportError:
                from ..plan_graph import GraphState
            
            state: "GraphState" = state
            
            logger.info("ℹ️ PlanInputAgent: 정보 완전성 검사 중...")
            
            # [중요!] state.get("extracted_info")가 이제 정상적으로 데이터를 가져옴
            extracted_info = state.get("extracted_info", {})

            missing_info = []
            for key, desc in self.required_info_schema.items():
                if not extracted_info.get(key):
                    missing_info.append(desc.get("description", key))

            if not missing_info:
                logger.info("✅ 모든 필수 정보 수집 완료.")
                try:
                    user_id = state.get("user_id", 0)

                    validated_data = ValidatedPlanInput(
                        user_id=user_id,
                        **extracted_info
                    )
                    summary = summarize_plan(validated_data.model_dump())
                    
                    # [수정!] 'original_input'을 여기서 반환 (라우터 수정 불필요)
                    return {
                        "input_completed": True,
                        "validated_plan_input": validated_data.model_dump(),
                        "original_input": extracted_info, # 👈 다음 노드(validate)를 위해 추가
                        "messages": [AIMessage(content=f"모든 정보가 수집되었습니다.\n{summary}")]
                    }
                                    
                except Exception as e:
                    logger.warning(f"⚠️ Pydantic 검증 실패: {e}")
                    return {
                        "input_completed": False,
                        "messages": [AIMessage(content=f"입력 정보를 확인하는 중 오류가 발생했습니다: {e}. 다시 말씀해 주시겠어요?")]
                    }
            else:
                missing_str = ", ".join(missing_info)
                logger.info(f"⚠️ 부족한 정보: {missing_str}")
                ai_question = f"말씀 감사합니다. 추가적으로 다음 정보가 필요합니다: **{missing_str}**. 알려주시겠어요?"
                return {
                    "input_completed": False,
                    "messages": [AIMessage(content=ai_question)]
                }
        return check_completeness_node

    # -------------------------------
    # 3️⃣ DB 저장 노드
    # -------------------------------
    def create_save_to_db_node(self):
        def _save_sync(data: dict) -> int:
            db_session = Session()
            try:
                record = PlanInput(**data)
                db_session.add(record)
                db_session.commit()
                plan_id = record.id
                logger.info(f"✅ [DB 저장 완료] plan_id: {plan_id}")
                return plan_id
            except Exception as e:
                db_session.rollback()
                logger.error(f"❌ [DB 저장 오류] {e}")
                raise
            finally:
                db_session.close()

        # [수정!] state: MessagesState -> state
        async def save_to_db_node(state):
            
            # [신규!] '지연 임포트'
            try:
                from agent.plan_graph import GraphState
            except ImportError:
                from ..plan_graph import GraphState
            
            state: "GraphState" = state
            
            logger.info("ℹ️ PlanInputAgent: DB 저장 중...")
            
            # [중요!] state.get("validated_plan_input")을 정상적으로 가져옴
            validated_data = state.get("validated_plan_input")
            if not validated_data:
                return {"messages": [AIMessage(content="저장할 데이터가 없습니다.")]}
            try:
                plan_id = await asyncio.to_thread(_save_sync, validated_data)
                return {"plan_id": plan_id,
                        "messages": [AIMessage(content=f"계획이 저장되었습니다. (Plan ID: {plan_id})")]}
            except Exception as e:
                return {"messages": [AIMessage(content=f"DB 저장 오류: {e}")]}
        return save_to_db_node