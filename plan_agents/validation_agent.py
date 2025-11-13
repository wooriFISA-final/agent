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
# 환경 설정 및 로깅
# ----------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# ----------------------------------
# LLM 설정
# ----------------------------------
llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# ----------------------------------
# SYSTEM PROMPT (LLM 전담 검증 페르소나)
# ----------------------------------
SYSTEM_PROMPT = SystemMessage(content="""
[페르소나(Persona)]
당신은 '우리은행 금융 검증 전문가(WooriValidator)'입니다.  
당신의 역할은 고객이 입력한 주택 구매 계획 데이터를 전문적으로 검증하고 정규화하는 것입니다.  
당신은 친근하지만 전문적인 어조로 생각하며, 항상 **명확하고 완전한 JSON 형식**으로만 응답해야 합니다.

---

[TASK]

1️⃣ **데이터 검증**
- `None`, `null`, `""`, `0` 값은 누락으로 간주합니다.
- 누락된 필드는 `"missing_fields"` 배열에 포함시키고 `"status": "incomplete"`로 표시합니다.

2️⃣ **데이터 정규화**
- 금액(`억`, `천만`, `만`) 단위는 **원(₩)** 단위로 변환합니다.
- `income_usage_ratio`는 `%`를 제거하고 정수로 변환합니다.
- `hope_housing_type`은 ENUM(`아파트`, `오피스텔`, `단독다가구`, `연립다세대`) 중 가장 유사한 값으로 보정합니다.

3️⃣ **지역명 검증**
- "서울"이 포함되어 있으면 `'서울특별시 {구}'` 형태로 보정합니다.
- 다른 지역은 광역시나 도 단위까지만 표준화합니다.

4️⃣ **시세 검증**
- `price_warning` 필드에 다음 중 하나를 반드시 포함하세요:
  - `"정상 범위 내 가격입니다."`
  - `"⚠️ 입력한 가격이 평균 시세 대비 ±1.5배 범위를 벗어났습니다."`

5️⃣ **논리적 경고**
- `logical_warning` 필드에 다음 중 하나를 포함하세요:
  - `"⚠️ 보유 자산이 목표 주택가 대비 매우 낮습니다."`
  - `null` (정상일 경우)

6️⃣ **출력 구조**
항상 아래 형식으로 응답하세요.
```json
{
  "status": "success" | "incomplete",
  "data": {
    "initial_prop": int,
    "hope_location": str,
    "hope_price": int,
    "hope_housing_type": str,
    "income_usage_ratio": int,
    "price_warning": str,
    "logical_warning": str | null,
    "validation_timestamp": "YYYY-MM-DD HH:MM:SS"
  },
  "missing_fields": [optional]
}
""")

# ----------------------------------
# ✅ JSON 응답 정리 함수 (백틱 제거)
# ----------------------------------
def clean_json_response(text: str) -> Dict[str, Any]:
    """
    LLM 응답에서 ```json ... ``` 코드블록을 제거 후 JSON 파싱
    """
    cleaned = re.sub(r"^```[a-zA-Z]*|```$", "", text.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON 파싱 실패: {e}\n원문: {text}")
        return None

# ----------------------------------
# 금액 단위 변환 함수 (보조용)
# ----------------------------------
def parse_korean_currency(value: Any) -> int:
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
# DB 저장 함수
# ----------------------------------
def upsert_member_and_plan(parsed: Dict[str, Any], user_id: Optional[int] = None) -> int:
    """검증 완료된 데이터로 members 및 plans 테이블 업데이트"""
    if not user_id:
        user_id = 1
    with engine.connect() as conn:
        # ✅ members 테이블 업데이트
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

        # ✅ plans 테이블 — 기존 계획 갱신 or 새로 추가
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
# ValidationAgent
# ----------------------------------
class ValidationAgent:
    def __init__(self):
        self.llm = llm

    # ------------------------------
    # 🔍 검증 수행
    # ------------------------------
    def validate_input_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM에게 검증 및 정규화를 맡기고 결과(JSON)를 파싱"""
        logger.info(f"🔍 Validation 요청 데이터: {raw_data}")

        messages = [
            SYSTEM_PROMPT,
            HumanMessage(content=json.dumps(raw_data, ensure_ascii=False))
        ]
        response = self.llm.invoke(messages)

        # ✅ LLM 응답 파싱 (백틱 제거 후 안전 파싱)
        parsed = clean_json_response(response.content)
        if parsed is None:
            return {"status": "error", "message": "LLM 응답이 JSON 형식이 아닙니다."}

        logger.info(f"🧠 LLM 응답(JSON): {parsed}")

        # ✅ 수치 보정 (안전장치)
        if parsed.get("status") == "success":
            data = parsed.get("data", {})
            data["hope_price"] = parse_korean_currency(data.get("hope_price", 0))
            data["initial_prop"] = parse_korean_currency(data.get("initial_prop", 0))
            data["income_usage_ratio"] = int(str(data.get("income_usage_ratio", 0)).replace("%", ""))
            data["validation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parsed["data"] = data

        return parsed

    # ------------------------------
    # 🧩 LangGraph용 검증 + DB저장 노드
    # ------------------------------
    def create_validation_node(self):
        async def validation_node(state):
            """LangGraph 내 검증 단계"""
            user_id = state.get("user_id") or 1
            extracted = state.get("extracted_info", {})
            result = self.validate_input_data(extracted)

            if result.get("status") == "incomplete":
                missing = result.get("missing_fields", [])
                msg = f"⚠️ 다음 정보가 누락되었습니다: {', '.join(missing)}. 다시 입력해주세요."
                return {
                    "final_response": result,
                    "messages": [AIMessage(content=msg)]
                }

            if result.get("status") == "success":
                validated = result["data"]
                upsert_member_and_plan(validated, user_id)
                return {
                    "final_response": result,
                    "messages": [AIMessage(content=f"✅ [user_id={user_id}] 검증 완료 및 DB 저장이 완료되었습니다.")]
                }

            return {
                "final_response": result,
                "messages": [AIMessage(content="❌ 검증 중 오류가 발생했습니다.")]
            }

        return validation_node
