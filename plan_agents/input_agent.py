import os
import re
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ----------------------------------
# 환경 설정 및 로깅
# ----------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
# PlanInputAgent (한 노드 = 한 에이전트)
# ----------------------------------
class PlanInputAgent:
    def __init__(self, model_name: str = "qwen3:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0.3)
        self.system_prompt = SystemMessage(content="""
[페르소나(Persona)]
당신은 '우리은행 주택 자금 설계 컨설턴트 AI'입니다.
고객의 대답을 기반으로 다음 질문을 결정하고,
현재까지 확보한 정보를 JSON으로 요약합니다.

---

[TASK]
1. 아래 5가지 핵심 정보를 모두 수집해야 합니다:
   - initial_prop : 초기 자산
   - hope_location : 희망 지역
   - hope_price : 희망 주택 가격
   - hope_housing_type : 주택 유형
   - income_usage_ratio : 월급 사용 비율
2. 이미 확보된 정보는 반복하지 마세요.  
3. 한 번에 하나의 질문만 하세요.  
4. 모든 정보를 확보하면 “is_complete”: true로 설정하고, “next_question”은 빈 문자열로 두세요.
5. 입력값에 '억', '천만', '만' 등의 단위가 이미 숫자로 변환되어 있다면 **추가 곱셈을 하지 마세요**.
6. 예: 사용자가 3억이라고 입력한 경우 → 300000000으로 변환
7. 이미 숫자로 들어온 값(300000000 등)은 그대로 유지하세요.
8. 10배, 100배를 더 곱하지 않습니다.

---

[출력 형식(JSON)]
{
  "next_question": "희망하시는 주택의 위치는 어디인가요?",
  "collected_info": {
    "initial_prop": "3000만원",
    "hope_location": "서울 마포구"
  },
  "is_complete": false
}

⚠️ 절대 한국어 설명문, 코드블록, 백틱, 불필요한 텍스트를 포함하지 마세요.
JSON만 출력하세요.
""")

    # -------------------------------
    # 내부 파서
    # -------------------------------
    def _parse_value(self, field: str, value: str):
        if field in ["initial_prop", "hope_price"]:
            return parse_korean_currency(value)
        elif field == "income_usage_ratio":
            try:
                return int(str(value).replace("%", "").strip())
            except:
                return 0
        return str(value).strip()

    # -------------------------------
    # 메인 실행 함수 (LangGraph Node)
    # -------------------------------
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """한 노드로써 동작"""
        user_id = state.get("user_id", 1)
        conversation = state.get("messages", [])
        collected_info = state.get("extracted_info", {}) or {}

        # LLM 호출
        messages = [self.system_prompt] + conversation
        response = self.llm.invoke(messages)
        raw_output = response.content.strip()
        logger.info(f"📨 LLM 출력(raw): {raw_output}")

        # JSON 파싱
        match = re.search(r"\{[\s\S]*\}", raw_output)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception as e:
                logger.error(f"⚠️ JSON 파싱 실패: {e}")
        if not parsed:
            return {
                "user_id": user_id,
                "extracted_info": collected_info,
                "input_completed": False,
                "messages": [AIMessage(content="죄송합니다. 다시 한 번 말씀해주시겠어요?")]
            }

        # 정보 병합
        for k, v in parsed.get("collected_info", {}).items():
            if v and k not in collected_info:
                collected_info[k] = self._parse_value(k, v)

        is_complete = parsed.get("is_complete", False)
        next_q = parsed.get("next_question", "")

        if is_complete:
            logger.info(f"✅ 입력 완료: {collected_info}")
            return {
                "user_id": user_id,
                "extracted_info": collected_info,
                "input_completed": True,
                "messages": [AIMessage(content="✅ 모든 정보가 입력되었습니다. 검증을 시작하겠습니다.")]
            }

        return {
            "user_id": user_id,
            "extracted_info": collected_info,
            "input_completed": False,
            "messages": [AIMessage(content=next_q)]
        }
