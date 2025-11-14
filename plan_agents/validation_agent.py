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

DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")
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
