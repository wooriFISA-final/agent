<<<<<<< HEAD
import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ----------------------------------
# 🌐 환경 설정 및 로깅
# ----------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

=======
"""
validation_agent.py (클래스 노드 수정본)
- (수정) '뇌'(ReAct)와 '손발'(Toolkit)을 ValidationAgentNode 클래스로 통합
- (수정) ollama.chat -> LangChain .invoke()로 변경
- (수정) LLM이 Pydantic 모델을 사용해 ToolCall JSON을 반환하도록 강제
- (수정) run_agent_executor -> run(self, state) 메서드로 변경
"""

import re
import os
import json
from difflib import get_close_matches
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------------------------
# (수정) LangChain 및 LangGraph 모듈 임포트
# ------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated 
import operator

# ------------------------------------------------
# (1) DB 설정 (님의 코드와 동일)
# ------------------------------------------------
load_dotenv()
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")
<<<<<<< HEAD
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# ----------------------------------
# 🤖 LLM 설정
# ----------------------------------
llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# ----------------------------------
# 🧠 SYSTEM PROMPT
# ----------------------------------
SYSTEM_PROMPT = SystemMessage(content="""
[페르소나(Persona)]
당신은 '우리은행 금융 검증 전문가(WooriValidator)'입니다.  
당신의 역할은 고객이 입력한 주택 구매 계획 데이터를 전문적으로 검증하고 정규화하는 것입니다.  
항상 **명확하고 완전한 JSON 형식**으로만 응답해야 합니다.

---

[TASK]
1️⃣ **데이터 검증**
- None, null, "", 0 값은 누락으로 간주합니다.
- 누락된 필드는 missing_fields 배열에 포함시키고 "status": "incomplete"로 표시합니다.

2️⃣ **데이터 정규화**
- 금액(억, 천만, 만) 단위를 원(₩) 단위로 변환합니다.
- income_usage_ratio는 %를 제거하고 정수로 변환합니다.
- hope_housing_type은 ENUM(아파트, 오피스텔, 단독다가구, 연립다세대) 중 가장 가까운 값으로 보정합니다.

3️⃣ **지역명 검증**
- "서울"이 포함되어 있으면 '서울특별시 {구}' 형태로 보정합니다.
- 다른 지역은 광역시나 도 단위까지만 표준화합니다.

4️⃣ **논리적 경고**
- logical_warning 필드에는 다음 중 하나를 포함:
  - "⚠️ 보유 자산이 목표 주택가 대비 매우 낮습니다."
  - null (정상일 경우)

---

[출력 형식(JSON)]
{
  "status": "success" | "incomplete",
  "data": {
    "initial_prop": int,
    "hope_location": str,
    "hope_price": int,
    "hope_housing_type": str,
    "income_usage_ratio": int,
    "price_warning": str | null,
    "logical_warning": str | null,
    "validation_timestamp": "YYYY-MM-DD HH:MM:SS"
  },
  "missing_fields": [optional]
}
""")

# ----------------------------------
# 🧹 JSON 파싱
# ----------------------------------
def clean_json_response(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"^```[a-zA-Z]*|```$", "", text.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON 파싱 실패: {e}\n원문: {text}")
        return None


# ----------------------------------
# 💰 금액 파서
# ----------------------------------
def parse_korean_currency(value: Any) -> int:
    """‘3억 5천만’ 등 한국어 금액을 원 단위로 변환"""
    if value in [None, "", 0]:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).replace(",", "").replace(" ", "")
    total = 0
    for pattern, multiplier in [
        (r"(\d+(?:\.\d+)?)억", 100_000_000),
        (r"(\d+(?:\.\d+)?)천만", 10_000_000),
        (r"(\d+(?:\.\d+)?)백만", 1_000_000),
        (r"(\d+(?:\.\d+)?)만", 10_000),
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
# 📊 state 테이블 시세 조회
# ----------------------------------
def get_market_price(location: str, housing_type: str) -> Optional[int]:
    """state 테이블에서 지역 + 주택유형 평균 시세 조회"""
    with engine.connect() as conn:
        query = text("""
            SELECT 
                CASE 
                    WHEN :housing_type = '아파트' THEN apartment_price
                    WHEN :housing_type = '오피스텔' THEN officetel_price
                    WHEN :housing_type = '연립다세대' THEN multi_price
                    WHEN :housing_type = '단독다가구' THEN detached_price
                    ELSE NULL
                END AS avg_price
            FROM state
            WHERE region_nm = :loc
            LIMIT 1
        """)
        result = conn.execute(query, {"loc": location, "housing_type": housing_type}).scalar()
        return result if result else None


# ----------------------------------
# 💾 members & plans 업데이트
# ----------------------------------
def upsert_member_and_plan(parsed: Dict[str, Any], user_id: Optional[int] = None) -> int:
    """검증된 데이터 members 및 plans 테이블에 저장"""
    if not user_id:
        user_id = 1
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE members
            SET initial_prop=:initial_prop,
                hope_location=:hope_location,
                hope_price=:hope_price,
                hope_housing_type=:hope_housing_type,
                income_usage_ratio=:income_usage_ratio
            WHERE user_id=:user_id
        """), {
            "user_id": user_id,
            "initial_prop": parsed.get("initial_prop", 0),
            "hope_location": parsed.get("hope_location", ""),
            "hope_price": parsed.get("hope_price", 0),
            "hope_housing_type": parsed.get("hope_housing_type", "아파트"),
            "income_usage_ratio": parsed.get("income_usage_ratio", 0)
        })

        existing_plan = conn.execute(
            text("SELECT plan_id FROM plans WHERE user_id=:uid ORDER BY plan_id DESC LIMIT 1"),
            {"uid": user_id}
        ).scalar()

        if existing_plan:
            conn.execute(text("""
                UPDATE plans
                SET target_loc=:target_loc,
                    target_build_type=:target_build_type,
                    create_at=NOW(),
                    plan_status='진행중'
                WHERE plan_id=:pid
            """), {
                "pid": existing_plan,
                "target_loc": parsed.get("hope_location", ""),
                "target_build_type": parsed.get("hope_housing_type", "아파트")
            })
        else:
            conn.execute(text("""
                INSERT INTO plans (user_id, target_loc, target_build_type, create_at, plan_status)
                VALUES (:user_id, :target_loc, :target_build_type, NOW(), '진행중')
            """), {
                "user_id": user_id,
                "target_loc": parsed.get("hope_location", ""),
                "target_build_type": parsed.get("hope_housing_type", "아파트")
            })

        conn.commit()
        logger.info(f"💾 DB 업데이트 완료: user_id={user_id}")
        return user_id


# ----------------------------------
# 🧩 ValidationAgent (LangGraph 노드형)
# ----------------------------------
class ValidationAgent:
    """LangGraph에서 직접 호출 가능한 단일 노드형 Agent"""
    def __init__(self):
        self.llm = llm

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph에서 이 노드를 실행"""
        user_id = state.get("user_id") or 1
        extracted = state.get("extracted_info", {})

        logger.info(f"🔍 Validation 요청 데이터: {extracted}")

        # ① LLM 검증
        messages = [SYSTEM_PROMPT, HumanMessage(content=json.dumps(extracted, ensure_ascii=False))]
        response = self.llm.invoke(messages)
        parsed = clean_json_response(response.content)

        if parsed is None:
            return {
                "final_response": {"status": "error"},
                "messages": [AIMessage(content="❌ LLM 응답이 JSON 형식이 아닙니다.")]
            }

        logger.info(f"🧠 LLM 응답(JSON): {parsed}")

        # ② 정규화
        if parsed.get("status") == "success":
            data = parsed.get("data", {})
            data["hope_price"] = parse_korean_currency(data.get("hope_price", 0))
            data["initial_prop"] = parse_korean_currency(data.get("initial_prop", 0))
            data["income_usage_ratio"] = int(str(data.get("income_usage_ratio", 0)).replace("%", ""))
            data["validation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parsed["data"] = data

            # ③ ✅ state 테이블 기반 시세 검증 추가
            market_price = get_market_price(data["hope_location"], data["hope_housing_type"])
            if market_price:
                ratio = data["hope_price"] / market_price
                if ratio > 1.5 or ratio < 0.5:
                    logger.warning(
                        f"⚠️ 시세 오차 발생: 입력={data['hope_price']:,}, 평균={market_price:,}, 비율={ratio:.2f}"
                    )
                    parsed["status"] = "incomplete"
                    parsed["data"]["price_warning"] = "⚠️ 입력한 가격이 평균 시세 대비 ±1.5배 범위를 벗어났습니다."
                    warning_msg = (
                        f"⚠️ 해당 지역({data['hope_location']})의 평균 {data['hope_housing_type']} 시세는 약 {market_price:,}원입니다.\n"
                        f"입력하신 가격({data['hope_price']:,}원)은 평균 대비 {ratio:.2f}배 차이가 납니다.\n\n"
                        f"❗ 해당 지역의 시세와 크게 다릅니다. 다른 금액이나 지역을 다시 입력해주세요."
                    )
                    return {"final_response": parsed, "messages": [AIMessage(content=warning_msg)]}

            # ✅ 정상 시세 → DB 저장
            upsert_member_and_plan(data, user_id)
            msg = f"✅ [user_id={user_id}] 검증 완료 및 DB 저장이 완료되었습니다."
            return {"final_response": parsed, "messages": [AIMessage(content=msg)]}

        elif parsed.get("status") == "incomplete":
            missing = parsed.get("missing_fields", [])
            msg = f"⚠️ 다음 정보가 누락되었습니다: {', '.join(missing)}. 다시 입력해주세요." if missing else \
                  "⚠️ 입력하신 내용이 시세 기준과 일치하지 않습니다. 다시 입력해주세요."
            return {"final_response": parsed, "messages": [AIMessage(content=msg)]}

        return {
            "final_response": parsed,
            "messages": [AIMessage(content="❌ 검증 중 오류가 발생했습니다.")]
        }
=======

try:
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
    # (툴킷 함수들이 전역에서 사용하므로 engine도 전역 유지)
except Exception as e:
    print(f"DB 연결 실패: {e}")
    engine = None

# ------------------------------------------------
# (2) (필수) LangGraph '통합' 상태 정의
# ------------------------------------------------
class AgentGraphState(TypedDict):
    # (이 노드가 읽을 데이터)
    plan_input_data: Dict[str, Any] 
    
    # (이 노드가 쓸 데이터)
    validation_passed: bool
    error_message: Optional[str]
    # (messages, user_id 등 기타 필드들...)

# ------------------------------------------------
# (3) 🛠️ [손발] VALIDATION TOOLKIT 함수들
# (클래스 외부의 전역 함수로 유지, DB 로딩을 위해)
# ------------------------------------------------
def load_valid_locations_from_db() -> List[str]:
    # ... (님의 load_valid_locations_from_db 코드) ...
    if not engine: return []
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT region_nm FROM state"))
            locations = [row[0] for row in result.fetchall()]
            print(f"[Toolkit] DB에서 {len(locations)}개의 유효한 지역명 로드 완료.")
            return locations
    except Exception as e:
        print(f"DB에서 지역명 로드 실패: {e}")
        return []

# (전역 변수로 DB에서 한 번만 로드)
VALID_LOCATIONS_LIST = load_valid_locations_from_db() 

def tool_sanitize_inputs(responses: Dict[str, Any]) -> Dict[str, Any]:
    # ... (님의 tool_sanitize_inputs 코드) ...
    cleaned_responses = {}
    for key, val in responses.items():
        if isinstance(val, str):
            cleaned_val = re.sub(r"[^\w\s-]", "", val).strip()
            cleaned_val = cleaned_val.replace("원", "").strip()
            cleaned_responses[key] = cleaned_val
        else:
            cleaned_responses[key] = val
    return cleaned_responses

def tool_check_input_format(responses: Dict[str, Any]) -> Dict[str, Any]:
    # ... (님의 tool_check_input_format 코드) ...
    for key, val in responses.items():
        val_str = str(val)
        if not val_str or val_str.strip() == "":
            return {"status": "error", "message": f"'{key}' 값이 비어 있습니다."}
        if re.search(r"-", val_str):
            return {"status": "error", "message": f"'{key}'에는 음수를 입력할 수 없습니다."}
    return {"status": "success", "message": "모든 입력 형식이 유효합니다."}

def _internal_normalize_location(loc: str) -> str:
    # ... (님의 _internal_normalize_location 코드) ...
    loc = loc.strip()
    mapping = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"}
    for short, full in mapping.items():
        if loc.startswith(short): loc = loc.replace(short, full, 1); break
    seoul_districts = ["강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천", "노원", "도봉", "동대문", "동작", "마포", "서대문", "서초", "성동", "성북", "송파", "양천", "영등포", "용산", "은평", "종로", "중", "중랑"]
    for gu in seoul_districts:
        if loc.startswith(gu): loc = f"서울특별시 {gu}구"; break
    return loc

def _internal_simplify_non_seoul(loc: str) -> str:
    # ... (님의 _internal_simplify_non_seoul 코드) ...
    if loc.startswith("서울"): return loc
    match = re.match(r"^(\S+시|\S+특별자치시|\S+도)", loc)
    if match: return match.group(1)
    return loc

def tool_validate_location(location_input: str) -> Dict[str, Any]:
    # (수정) valid_locations_list 인자를 제거 (전역 변수 VALID_LOCATIONS_LIST 사용)
    """
    사용자가 입력한 지역명을 검증하고, DB에 있는 유효한 지역명으로 보정합니다.
    :return: {"status": "success" | "corrected" | "error", ...}
    """
    normalized = _internal_normalize_location(location_input)
    simplified = _internal_simplify_non_seoul(normalized)
    target_to_check = simplified
    
    if target_to_check in VALID_LOCATIONS_LIST:
        return {"status": "success", "validated_location": target_to_check}

    matches = get_close_matches(target_to_check, VALID_LOCATIONS_LIST, n=1, cutoff=0.7)
    if matches:
        corrected = matches[0]
        return {"status": "corrected", "validated_location": corrected, "original_input": location_input}

    message = f"유효하지 않은 지역입니다. (입력: {location_input}). 서울은 '송파구'처럼, 그 외는 '부산광역시'처럼 입력해야 합니다."
    return {"status": "error", "message": message, "original_input": location_input}

# ----------------------------------------------------------------------
# (4) 🟢 (수정) ValidationAgentNode 클래스 정의 🟢
# ----------------------------------------------------------------------

# --- 4-1. (신규) LLM이 반환할 JSON 형식을 Pydantic으로 정의 ---
class ToolCall(BaseModel):
    """LLM이 도구를 호출할 때 사용할 JSON 스키마"""
    name: str = Field(description="호출할 도구의 이름. [tool_sanitize_inputs, tool_check_input_format, tool_validate_location, tool_generate_friendly_error_message] 중 하나")
    args: Dict[str, Any] = Field(description="도구에 전달할 인수 딕셔너리")

class FinalAnswer(BaseModel):
    """LLM이 최종 답변을 반환할 때 사용할 JSON 스키마"""
    status: str = Field(description="'success' 또는 'error'")
    message: str = Field(description="검증 결과에 대한 최종 메시지")
    data: Optional[Dict[str, Any]] = Field(description="검증이 성공한 경우, 보정된 데이터 딕셔너리")

# (LLM이 ToolCall 또는 FinalAnswer 둘 중 하나를 반환하도록 Union 사용)
class ValidationDecision(BaseModel):
    """LLM의 생각과 결정 (도구 호출 또는 최종 답변)"""
    thought: str = Field(description="현재 상황을 분석하고 다음 행동을 결정하는 과정")
    decision: Union[ToolCall, FinalAnswer] = Field(description="도구 호출(ToolCall) 또는 최종 답변(FinalAnswer) 중 하나")


class ValidationAgentNode:
    
    def __init__(self, model="qwen3:8b"):
        """
        ValidationAgentNode를 초기화합니다.
        - LLM 및 LangChain 'chain'을 초기화합니다.
        - '도구' 함수들을 매핑합니다.
        """
        print("--- ValidationAgentNode 초기화 ---")
        try:
            # --- 4-2. (수정) LLM 및 LangChain 체인 정의 ---
            llm = ChatOllama(model=model, temperature=0.0)
            
            # (님의 SYSTEM_PROMPT를 LangChain 프롬프트로 변환)
            system_prompt = f"""
            당신은 '입력 검증 전문 에이전트'입니다.
            당신의 임무는 사용자의 JSON(딕셔너리) 입력을 받아, 사용 가능한 도구(Tool)들을 순서대로 호출하여 입력을 검증하고 최종 결과를 반환하는 것입니다.

            **[검증 절차]**
            1.  먼저 `tool_sanitize_inputs`를 사용해 입력을 정제합니다.
            2.  정제된 결과로 `tool_check_input_format`을 호출해 기본 형식을 검사합니다.
            3.  형식 검사가 통과되면, `target_location` 값으로 `tool_validate_location`을 호출해 지역명을 검증합니다.
            4.  만약 2단계나 3단계에서 'error'가 발생하면, 즉시 검증을 중단하고 `tool_generate_friendly_error_message`를 호출하여 사용자에게 친절한 '오류 메시지'를 반환하세요.
            5.  모든 검증(1, 2, 3)이 성공하면, 최종적으로 "검증 완료" 상태와 "보정된 데이터"를 반환하세요.

            **[Tool 사용 규칙]**
            - Tool을 호출하든 최종 답변을 하든, '반드시' Pydantic 스키마('ValidationDecision')에 맞는 JSON 형식으로만 응답해야 합니다.
            - `thought` 필드에는 당신의 생각을, `decision` 필드에 `ToolCall` 또는 `FinalAnswer` 객체를 명시하세요.

            **[사용 가능한 Tool 목록]**
            ---
            1.  **tool_sanitize_inputs(responses: Dict)**:
                - {tool_sanitize_inputs.__doc__}
            2.  **tool_check_input_format(responses: Dict)**:
                - {tool_check_input_format.__doc__}
            3.  **tool_validate_location(location_input: str)**:
                - {tool_validate_location.__doc__}
            4.  **tool_generate_friendly_error_message(technical_error_message: str)**:
                - (이 함수는 '문자열'을 반환합니다. 이 문자열을 받으면 'FinalAnswer'로 포장해서 반환하세요)
            ---
            """
            
            # (수정) LLM이 Pydantic(ValidationDecision) JSON을 반환하도록 강제
            self.llm_with_tools = llm.with_structured_output(ValidationDecision, method="json")
            
            # (수정) LangChain 체인 정의
            self.chain = (
                ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    # ⬇️ LangGraph는 'messages' 키를 자동으로 처리
                    ("placeholder", "{messages}") 
                ])
                | self.llm_with_tools
            )

        except Exception as e:
            print(f"LLM 로드 중 오류 발생: {e}")
            raise

        # --- 4-3. (수정) '도구' 함수들을 클래스 내부 딕셔너리로 매핑 ---
        self.tools = {
            "tool_sanitize_inputs": tool_sanitize_inputs,
            "tool_check_input_format": tool_check_input_format,
            "tool_validate_location": tool_validate_location,
            # (tool_generate_friendly_error_message는 LLM 체인으로 따로 만듦 - 아래 참조)
        }
        
        # --- 4-4. (신규) 오류 메시지 생성 전용 LLM 체인 ---
        self.error_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "너는 입력 검증 AI야. 시스템 오류 메시지를 받았다. 핵심 원인을 파악해서, 간결하고 공손하게 문제점을 설명하고, 마지막엔 '다시 입력해주세요.'로 끝내라."),
                ("user", "시스템 오류: {technical_error_message}")
            ])
            | self.llm
            | StrOutputParser()
        )
        print("--- ValidationAgentNode LLM 체인 구성 완료 ---")

    # --- 4-5. (수정) LangGraph '노드' 실행 함수 (ReAct 루프) ---
    def run(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        (수정) LangGraph '노드'로 등록될 실제 실행 함수입니다.
        'run_agent_executor'의 ReAct 루프 로직을 포함합니다.
        """
        print("\n--- [노드] '검증 노드' 실행 ---")
        
        # 1. State에서 검증할 데이터(plan_input_data) 가져오기
        user_input_data = state.get("plan_input_data")
        if not user_input_data:
            return {"validation_passed": False, "error_message": "검증할 데이터가 없습니다."}

        # 2. ReAct 루프를 위한 'messages' 리스트 초기화
        messages = [
            HumanMessage(content=f"다음 입력을 검증해주세요: {json.dumps(user_input_data, ensure_ascii=False)}")
        ]

        # 3. ReAct 루프 (최대 5회)
        for i in range(5):
            print(f"\n[ValidationAgent 루프 {i+1}] Thinking... 🧠")
            
            try:
                # 3-1. LLM 호출 (JSON 강제)
                llm_decision: ValidationDecision = self.chain.invoke({"messages": messages})
                
                print(f"[ValidationAgent 생각] {llm_decision.thought}")
                
                # 3-2. LLM의 결정(decision) 분석
                decision = llm_decision.decision

                if isinstance(decision, FinalAnswer):
                    # (A) 최종 답변 반환
                    print(f"\n[ValidationAgent 최종 답변... ✅] {decision.status}")
                    if decision.status == "success":
                        return {
                            "validation_passed": True,
                            "plan_input_data": decision.data # 보정된 데이터로 덮어쓰기
                        }
                    else:
                        return {
                            "validation_passed": False,
                            "error_message": decision.message
                        }
                
                elif isinstance(decision, ToolCall):
                    # (B) 도구 호출
                    tool_name = decision.name
                    tool_args = decision.args
                    
                    if tool_name not in self.tools:
                        raise ValueError(f"LLM이 유효하지 않은 Tool을 호출했습니다: {tool_name}")

                    print(f"[ValidationAgent Action... 🎬] '{tool_name}' Tool 호출")
                    tool_function = self.tools[tool_name]
                    
                    # 3-3. 도구 실행
                    tool_result = tool_function(**tool_args)
                    
                    # 3-4. (특별 처리) 도구 실행 결과가 'error'인 경우
                    if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                        print(f"[ValidationAgent] ❌ 도구 실행 오류: {tool_result['message']}")
                        
                        # (tool_generate_friendly_error_message 실행)
                        friendly_error_msg = self.error_chain.invoke({
                            "technical_error_message": tool_result['message']
                        })
                        
                        return {
                            "validation_passed": False,
                            "error_message": friendly_error_msg
                        }
                    
                    # 3-5. 도구 실행 결과를 'Observation'으로 messages에 추가
                    observation = f"ToolResult for {tool_name}: {json.dumps(tool_result, ensure_ascii=False)}"
                    print(f"[ValidationAgent Observation... 📝] {observation}")
                    messages.append(AIMessage(content=str(llm_decision.dict()))) # LLM의 응답 추가
                    messages.append(HumanMessage(content=observation)) # 도구 결과 추가

            except Exception as e:
                print(f"[ValidationAgent Error... 💥] {e}")
                import traceback
                traceback.print_exc()
                messages.append(HumanMessage(content=f"ToolError: {e}. 규칙(JSON 형식)을 다시 확인하세요."))
        
        # 4. 루프가 5번 다 돌아도 끝나지 않으면 강제 종료
        print("--- [노드 종료] '검증 노드' 작업 시간 초과 ---")
        return {"validation_passed": False, "error_message": "작업 시간(5단계)을 초과했습니다."}

# ------------------------------------------------
# (5) (테스트) VS Code에서 이 파일만 단독으로 실행
# (python agent/plan_agents/validation_agent.py)
# ------------------------------------------------
if __name__ == "__main__":
    
    # (로깅 설정)
    import logging
    logging.basicConfig(level=logging.INFO)

    # 1. 노드 인스턴스화
    validation_node = ValidationAgentNode(model="qwen3:8b")

    # 2. (가상) InputAgent로부터 받은 'plan_input_data'
    test_data_success = {
        "target_house_price": "1000000000",
        "target_location": "서울 송파구", # (정상 데이터)
        "housing_type": "아파트",
        "available_assets": "200000000",
        "income_usage_ratio": "50"
    }
    
    test_data_fail = {
        "target_house_price": "1000000000",
        "target_location": "서울 송파", # (오류 데이터)
        "housing_type": "아파트",
        "available_assets": "200000000",
        "income_usage_ratio": "50"
    }

    # 3. (가상) LangGraph 'state' 생성
    test_state_success = {
        "plan_input_data": test_data_success
    }
    test_state_fail = {
        "plan_input_data": test_data_fail
    }

    # 4. (테스트 1: 성공)
    print("\n\n--- 🏁 테스트 1: 검증 성공 🏁 ---")
    result_success = validation_node.run(test_state_success)
    print("\n[최종 반환 결과 (성공)]")
    print(json.dumps(result_success, indent=2, ensure_ascii=False))

    # 5. (테스트 2: 실패)
    print("\n\n--- 🏁 테스트 2: 검증 실패 🏁 ---")
    result_fail = validation_node.run(test_state_fail)
    print("\n[최종 반환 결과 (실패)]")
    print(json.dumps(result_fail, indent=2, ensure_ascii=False))
>>>>>>> c35374b0f210d38053de68412e5413857b8674da
