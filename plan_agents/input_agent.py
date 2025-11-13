import os
import re
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
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

# ✅ DB 연결은 ValidationAgent에서만 사용함 (이 파일에서는 필요 없음)
# engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# ----------------------------------
# LLM 설정
# ----------------------------------
llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# ----------------------------------
# SYSTEM PROMPT
# ----------------------------------
SYSTEM_PROMPT = SystemMessage(content="""
[페르소나(Persona)]
당신은 '우리은행 부동산 재무 설계 상담사(WooriPlanner)'입니다.  
고객의 재무 상황을 친근하고 따뜻하게 묻되, 불필요한 인사나 자기소개를 반복하지 않습니다.  
모든 질문은 한 번에 하나씩, 자연스럽게 물어봐야 합니다.

[TASK]
1️⃣ 질문은 반드시 한 항목씩만 합니다.  
2️⃣ 다음 다섯 가지 정보를 순서대로 수집합니다:
   - initial_prop : 초기 사용 가능 자산 (예: 3000만원)
   - hope_location : 희망 지역 (예: 서울 마포구)
   - hope_price : 희망 주택 가격 (예: 12억 5천만원)
   - hope_housing_type : 주택 유형 (아파트, 오피스텔, 단독다가구, 연립다세대)
   - income_usage_ratio : 월급 중 주택 자금 사용 비율 (예: 30%)
3️⃣ 금액 단위(억, 천만, 만)는 모두 원 단위 정수로 인식합니다.
4️⃣ 불필요한 감탄사, 인사말, 자기소개를 반복하지 않습니다.
5️⃣ 응답은 오직 자연스러운 한국어 문장으로만 구성합니다.
""")

# ----------------------------------
# 금액 단위 변환 함수
# ----------------------------------
def parse_korean_currency(value: Any) -> int:
    """'3억 5천' 같은 금액 표현을 정수(원)로 변환"""
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(value)

    value = str(value).replace(",", "").replace(" ", "")
    total = 0
    for pattern, multiplier in [
        (r"(\d+(?:\.\d+)?)억", 100000000),
        (r"(\d+(?:\.\d+)?)천만", 10000000),
        (r"(\d+(?:\.\d+)?)백만", 1000000),
        (r"(\d+(?:\.\d+)?)만", 10000),
    ]:
        match = re.search(pattern, value)
        if match:
            total += float(match.group(1)) * multiplier

    if total == 0:
        try:
            total = int(float(re.sub(r"[^0-9]", "", value)))
        except ValueError:
            total = 0
    return int(total)


# ----------------------------------
# PlanInputAgent
# ----------------------------------
class PlanInputAgent:
    def __init__(self):
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT
        self.question_order = [
            ("initial_prop", "초기 사용 가능 자산은 얼마인가요? (예: 3000만원)"),
            ("hope_location", "희망 지역을 알려주세요 (예: 서울 마포구)"),
            ("hope_price", "구매를 희망하는 주택의 가격은 얼마인가요? (예: 12억 5천만원)"),
            ("hope_housing_type", "희망 주택 유형은 무엇인가요? (아파트, 오피스텔, 단독다가구, 연립다세대 중 택1)"),
            ("income_usage_ratio", "월급 중 주택 자금으로 사용할 비율은 몇 퍼센트인가요? (예: 30%)")
        ]

    # ----------------------------------
    # 입력값 파싱
    # ----------------------------------
    def _simple_parse(self, field: str, value: str):
        if field in ["initial_prop", "hope_price"]:
            return parse_korean_currency(value)
        elif field == "income_usage_ratio":
            try:
                return int(str(value).replace("%", "").strip())
            except:
                return 0
        elif field in ["hope_location", "hope_housing_type"]:
            return value.strip()
        return value

    # ----------------------------------
    # 자연스러운 질문 생성
    # ----------------------------------
    def _generate_natural_question(self, field_key: str, base_question: str) -> str:
        messages = [
            self.system_prompt,
            HumanMessage(content=f"다음 문장을 자연스럽게 질문으로 바꿔주세요:\n'{base_question}'")
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    # ----------------------------------
    # LangGraph: 입력 수집 노드
    # ----------------------------------
    def create_extraction_node(self):
        async def extraction_node(state):
            user_id = state.get("user_id") or 1
            collected = state.get("extracted_info", {}) or {}
            pending_fields = [f for f, _ in self.question_order if f not in collected or not collected[f]]

            # 첫 질문
            if not collected and not state.get("messages", []):
                q = self._generate_natural_question("initial_prop", self.question_order[0][1])
                logger.info(f"👤 user_id={user_id} | 첫 질문: {q}")
                return {
                    "user_id": user_id,
                    "extracted_info": {},
                    "input_completed": False,
                    "messages": [AIMessage(content=q)]
                }

            # 사용자 입력
            last_msg = state.get("messages", [])
            user_input = last_msg[-1].content.strip() if last_msg else ""
            current_field = pending_fields[0] if pending_fields else None

            if not user_input:
                q = dict(self.question_order)[current_field]
                natural_q = self._generate_natural_question(current_field, q)
                return {
                    "user_id": user_id,
                    "extracted_info": collected,
                    "input_completed": False,
                    "messages": [AIMessage(content=natural_q)]
                }

            # 입력값 저장
            if current_field:
                collected[current_field] = self._simple_parse(current_field, user_input)

            # 다음 질문 or 완료
            pending_fields = [f for f, _ in self.question_order if f not in collected or not collected[f]]
            if pending_fields:
                next_field = pending_fields[0]
                q = dict(self.question_order)[next_field]
                natural_q = self._generate_natural_question(next_field, q)
                return {
                    "user_id": user_id,
                    "extracted_info": collected,
                    "input_completed": False,
                    "messages": [AIMessage(content=natural_q)]
                }

            # ✅ 모든 입력 완료 시
            logger.info(f"✅ 모든 입력 완료 (user_id={user_id}): {collected}")
            return {
                "user_id": user_id,
                "extracted_info": collected,
                "input_completed": True,
                "messages": [
                    AIMessage(content="✅ 입력이 모두 완료되었습니다. 이제 입력하신 정보를 검증하겠습니다.")
                ],
            }

        return extraction_node

    # ----------------------------------
    # 완전성 검사 노드
    # ----------------------------------
    def create_check_completeness_node(self):
        async def completeness_node(state):
            info = state.get("extracted_info", {}) or {}
            required = [f for f, _ in self.question_order]
            missing = [f for f in required if not info.get(f)]

            if missing:
                missing_field = missing[0]
                base_q = dict(self.question_order)[missing_field]
                messages = [
                    self.system_prompt,
                    HumanMessage(content=f"'{base_q}'에 대해 부드럽고 자연스럽게 물어봐줘.")
                ]
                response = self.llm.invoke(messages)
                natural_q = response.content.strip()
                logger.warning(f"⚠️ {missing_field} 정보 누락 → LLM 질문: {natural_q}")
                return {
                    "input_completed": False,
                    "messages": [AIMessage(content=natural_q)]
                }

            # 모든 입력이 존재 → 검증 단계로 이동
            return {
                "input_completed": True,
                "messages": [AIMessage(content="✅ 모든 정보가 입력되었습니다. 검증을 시작합니다.")]
            }

        return completeness_node
