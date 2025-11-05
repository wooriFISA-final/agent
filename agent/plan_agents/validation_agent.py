"""
validation_agent.py (JSON 통합본)
- '뇌' (ReAct 에이전트) + '손발' (툴킷 함수)
- [수정] LLM이 ToolCall 문자열 대신 JSON을 반환하도록 변경
- [수정] eval() 대신 json.loads()를 사용해 100% 안정적인 파싱 수행
"""

import re
import os
import ollama
import json
from difflib import get_close_matches
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------------------------
# 환경 설정 및 DB 연결 (툴킷 코드)
# ------------------------------------------------
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

try:
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
except Exception as e:
    print(f"DB 연결 실패: {e}")
    engine = None

# =================================================================
# 🛠️ [손발] VALIDATION TOOLKIT 함수들 (변경 없음)
# =================================================================

# ------------------------------------------------
# 🛠️ 1. [데이터 로딩 Tool]
# ------------------------------------------------
def load_valid_locations_from_db() -> List[str]:
    """
    DB의 'state' 테이블에서 유효한 지역명(region_nm) 목록 전체를 불러옵니다.
    :return: 유효한 지역명 문자열 리스트
    """
    if not engine:
        print("DB 엔진이 초기화되지 않았습니다. 빈 리스트를 반환합니다.")
        return []
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT region_nm FROM state"))
            locations = [row[0] for row in result.fetchall()]
            print(f"[Toolkit] DB에서 {len(locations)}개의 유효한 지역명 로드 완료.")
            return locations
    except Exception as e:
        print(f"DB에서 지역명 로드 실패: {e}")
        return []

# ------------------------------------------------
# 🛠️ 2. [검증 Tool] - 텍스트 정제
# ------------------------------------------------
def tool_sanitize_inputs(responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 입력값이 담긴 딕셔너리를 받아, 각 값에서 불필요한 기호('원', ',', '.')나 공백을 제거합니다.
    :param responses: 사용자 입력 원본 딕셔너리
    :return: 값이 정제된 새로운 딕셔너리
    """
    cleaned_responses = {}
    for key, val in responses.items():
        if isinstance(val, str):
            cleaned_val = re.sub(r"[^\w\s-]", "", val).strip()
            cleaned_val = cleaned_val.replace("원", "").strip()
            cleaned_responses[key] = cleaned_val
        else:
            cleaned_responses[key] = val
    return cleaned_responses

# ------------------------------------------------
# 🛠️ 3. [검증 Tool] - 기본 형식 검증
# ------------------------------------------------
def tool_check_input_format(responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    입력값 딕셔너리를 검사하여 비어있거나 음수 값이 있는지 확인합니다.
    :return: {"status": "success"} 또는 {"status": "error", "message": "'key' 값이 비어 있습니다."}
    """
    for key, val in responses.items():
        val_str = str(val)
        if not val_str or val_str.strip() == "":
            return {"status": "error", "message": f"'{key}' 값이 비어 있습니다."}
        if re.search(r"-", val_str):
            # 예외 없이 모든 키에 대해 음수 검사
            return {"status": "error", "message": f"'{key}'에는 음수를 입력할 수 없습니다."}
    return {"status": "success", "message": "모든 입력 형식이 유효합니다."}

# ------------------------------------------------
# 🛠️ 4. [검증 Tool] - 지역명 검증 및 보정
# (내부 함수 _internal_normalize_location, _internal_simplify_non_seoul 포함)
# ------------------------------------------------
def _internal_normalize_location(loc: str) -> str:
    loc = loc.strip()
    mapping = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"}
    for short, full in mapping.items():
        if loc.startswith(short): loc = loc.replace(short, full, 1); break
    seoul_districts = ["강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천", "노원", "도봉", "동대문", "동작", "마포", "서대문", "서초", "성동", "성북", "송파", "양천", "영등포", "용산", "은평", "종로", "중", "중랑"]
    for gu in seoul_districts:
        if loc.startswith(gu): loc = f"서울특별시 {gu}구"; break
    return loc

def _internal_simplify_non_seoul(loc: str) -> str:
    if loc.startswith("서울"): return loc
    match = re.match(r"^(\S+시|\S+특별자치시|\S+도)", loc)
    if match: return match.group(1)
    return loc

def tool_validate_location(location_input: str, valid_locations_list: List[str]) -> Dict[str, Any]:
    """
    사용자가 입력한 지역명을 검증하고, DB에 있는 유효한 지역명으로 보정합니다.
    :return: {"status": "success" | "corrected" | "error", ...}
    """
    normalized = _internal_normalize_location(location_input)
    simplified = _internal_simplify_non_seoul(normalized)
    target_to_check = simplified
    
    if target_to_check in valid_locations_list:
        return {"status": "success", "validated_location": target_to_check}

    matches = get_close_matches(target_to_check, valid_locations_list, n=1, cutoff=0.7)
    if matches:
        corrected = matches[0]
        return {"status": "corrected", "validated_location": corrected, "original_input": location_input}

    message = f"유효하지 않은 지역입니다. (입력: {location_input}). 서울은 '송파구'처럼, 그 외는 '부산광역시'처럼 입력해야 합니다."
    return {"status": "error", "message": message, "original_input": location_input}

# ------------------------------------------------
# 🛠️ 5. [LLM Tool] - 사용자 친화적 메시지 생성
# ------------------------------------------------
def tool_generate_friendly_error_message(technical_error_message: str, model="qwen3:8b") -> str:
    """
    '기계적인(technical)' 오류 메시지를 입력받아,
    사용자에게 보여줄 친절하고 간결한 안내 메시지를 생성합니다.
    """
    system_prompt = (
        "너는 입력 검증 AI야. 시스템 오류 메시지를 받았다. "
        "핵심 원인을 파악해서, 간결하고 공손하게 문제점을 설명하고, "
        "마지막엔 '다시 입력해주세요.'로 끝내라."
    )
    try:
        res = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"시스템 오류: {technical_error_message}"}
            ]
        )
        return res["message"]["content"].strip()
    except Exception as e:
        print(f"LLM 오류 메시지 생성 실패: {e}")
        return f"오류: {technical_error_message} 다시 입력해주세요."

# =================================================================
# 🧠 [뇌] ReAct 에이전트 로직 (JSON으로 수정됨)
# =================================================================

# ------------------------------------------------
# 1. 에이전트 초기화 (Tool 준비) - (변경 없음)
# ------------------------------------------------

print("에이전트 실행기(JSON 통합본) 초기화: 유효한 지역 목록 로드 중...")
VALID_LOCATIONS_LIST = load_valid_locations_from_db() 

TOOLS_AVAILABLE = {
    "tool_sanitize_inputs": tool_sanitize_inputs,
    "tool_check_input_format": tool_check_input_format,
    "tool_validate_location": tool_validate_location,
    "tool_generate_friendly_error_message": tool_generate_friendly_error_message,
}

# ------------------------------------------------
# 2. 페르소나 및 태스크 정의 (System Prompt) - ✅ [수정됨]
# ------------------------------------------------
SYSTEM_PROMPT = f"""
당신은 '입력 검증 전문 에이전트'입니다.
당신의 임무는 사용자의 JSON(딕셔너리) 입력을 받아, 사용 가능한 도구(Tool)들을 순서대로 호출하여 입력을 검증하고 최종 결과를 반환하는 것입니다.

**[검증 절차]**
1.  먼저 `tool_sanitize_inputs`를 사용해 입력을 정제합니다.
2.  정제된 결과로 `tool_check_input_format`을 호출해 기본 형식을 검사합니다.
3.  형식 검사가 통과되면, `target_location` 값으로 `tool_validate_location`을 호출해 지역명을 검증합니다.
4.  만약 2단계나 3단계에서 'error'가 발생하면, 즉시 검증을 중단하고 `tool_generate_friendly_error_message`를 호출하여 사용자에게 친절한 '오류 메시지'를 반환하세요.
5.  모든 검증(1, 2, 3)이 성공하면, 최종적으로 "검증 완료" 상태와 "보정된 데이터"를 반환하세요.

**[Tool 사용 규칙]**
- Tool을 호출할 때는 다른 설명 없이 '반드시' 다음 JSON 형식으로만 응답해야 합니다.
- `thought` 필드에는 당신의 생각을, `tool_call` 필드에 호출할 도구를 명시하세요.

[JSON 출력 예시 (Tool 호출 시)]
{{
  "thought": "데이터를 받았으니 먼저 정제해야겠다.",
  "tool_call": {{
    "name": "tool_sanitize_inputs",
    "args": {{"responses": {{"key": "value", ...}} }}
  }}
}}

[JSON 출력 예시 (오류 발생 시)]
{{
  "thought": "지역 검증에 실패했다. 사용자에게 알려줘야겠다.",
  "tool_call": {{
    "name": "tool_generate_friendly_error_message",
    "args": {{"technical_error_message": "유효하지 않은 지역..."}}
  }}
}}

[JSON 출력 예시 (최종 완료 시)]
{{
  "thought": "모든 검증을 통과했다. 최종 데이터를 반환한다.",
  "final_answer": {{
    "status": "success",
    "message": "모든 검증 통과",
    "data": {{ "target_house_price": "1000", ... }}
  }}
}}

[JSON 출력 예시 (최종 실패 시 - tool_generate_...가 반환한 메시지)]
{{
  "thought": "오류 메시지 생성이 완료되었다. 이 메시지를 최종 반환한다.",
  "final_answer": {{
    "status": "error",
    "message": "지역명이 잘못되었습니다. '서울 송파구'처럼 다시 입력해주세요."
  }}
}}

**[사용 가능한 Tool 목록]**
---
1.  **tool_sanitize_inputs(responses: Dict)**:
    - {tool_sanitize_inputs.__doc__}
2.  **tool_check_input_format(responses: Dict)**:
    - {tool_check_input_format.__doc__}
3.  **tool_validate_location(location_input: str, valid_locations_list: List = None)**:
    - {tool_validate_location.__doc__}
4.  **tool_generate_friendly_error_message(technical_error_message: str)**:
    - (이 함수는 '문자열'을 반환합니다. 이 문자열을 받으면 'final_answer'로 포장해서 반환하세요)
---
"""

# ------------------------------------------------
# 3. 에이전트 실행기 (ReAct Loop) - ✅ [수정됨]
# ------------------------------------------------
def run_agent_executor(user_input_data: Dict[str, Any], model="qwen3:8b") -> Dict:
    """
    LLM이 Tool을 스스로 선택하고 호출하게 만드는 실행기 (JSON 모드)
    :return: 최종 결과 딕셔너리 (예: {"status": "success", "data": ...})
    """
    
    print(f"\n--- 🚀 새 작업 시작 (JSON Mode) ---")
    print(f"[입력] {user_input_data}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"다음 입력을 검증해주세요: {json.dumps(user_input_data, ensure_ascii=False)}"}
    ]
    
    # tool_generate_friendly_error_message가 호출되었는지 추적하는 플래그
    error_tool_called = False

    for _ in range(10): # 최대 10번의 ReAct 루프
        print("\n[Thinking... 🧠 (JSON Mode)]")
        
        try:
            # [수정] ollama.chat에 format="json" 추가
            res = ollama.chat(model=model, messages=messages, format="json")
            llm_response_str = res["message"]["content"].strip()
            messages.append({"role": "assistant", "content": llm_response_str})

            # [수정] re.search와 eval() 대신 json.loads() 사용
            response_json = json.loads(llm_response_str)

            if "tool_call" in response_json:
                tool_call = response_json.get("tool_call", {})
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                if not tool_name or tool_name not in TOOLS_AVAILABLE:
                    raise ValueError(f"LLM이 유효하지 않은 Tool을 호출했습니다: {tool_name}")

                print(f"[Action... 🎬] '{tool_name}' Tool 호출 (JSON)")
                print(f"[DEBUG] Tool Args: {tool_args}")
                
                # [수정] tool_args가 이미 딕셔너리이므로 eval 불필요
                tool_function = TOOLS_AVAILABLE[tool_name]
                
                # [특별 규칙] tool_validate_location일 경우, 리스트 자동 주입
                if tool_name == "tool_validate_location":
                    tool_args["valid_locations_list"] = VALID_LOCATIONS_LIST
                    # 이 Tool은 location_input만 받으므로 args를 직접 전달
                    tool_result = tool_function(**tool_args)
                
                # [특별 규칙] 오류 메시지 생성 Tool 추적
                elif tool_name == "tool_generate_friendly_error_message":
                    error_tool_called = True # 이 Tool이 호출되었음을 기억
                    tool_result = tool_function(**tool_args) # 이 Tool은 '문자열'을 반환
                
                else:
                    # tool_sanitize_inputs, tool_check_input_format
                    tool_result = tool_function(**tool_args)
                
                # Tool 실행 결과를 LLM에게 다시 알려줌 (Observation)
                observation = f"ToolResult: {json.dumps(tool_result, ensure_ascii=False)}"
                print(f"[Observation... 📝] {observation}")
                messages.append({"role": "user", "content": observation})
                
                # [특별 규칙] 오류 메시지 생성 Tool이 문자열을 반환했다면,
                # LLM이 이걸 'final_answer'로 포장하도록 유도
                if error_tool_called:
                    messages.append({
                        "role": "user", 
                        "content": "방금 받은 'ToolResult' 문자열을 'final_answer' JSON 형식으로 포장해서 즉시 반환하세요."
                    })

            elif "final_answer" in response_json:
                # LLM이 '최종 답변'을 JSON으로 반환
                print("\n[Final Answer... ✅]")
                final_data = response_json["final_answer"]
                print(f"최종 결과: {final_data}")
                print("--- 🏁 작업 완료 (JSON Mode) ---")
                return final_data # 👈 [수정] 문자열이 아닌 JSON(Dict)을 반환

            else:
                raise ValueError("LLM의 응답에 'tool_call'이나 'final_answer'가 없습니다.")

        except json.JSONDecodeError:
            print(f"[Tool Error... 💥] LLM이 유효한 JSON을 뱉지 않음: {llm_response_str}")
            messages.append({"role": "user", "content": "ToolError: 유효한 JSON이 아닙니다. 규칙(JSON 형식)을 다시 확인하세요."})
        
        except Exception as e:
            print(f"[Tool Error... 💥] {e}")
            observation = f"ToolError: {e}"
            messages.append({"role": "user", "content": observation})

    # 루프가 10번 다 돌아도 끝나지 않으면 강제 종료
    print("--- 🏁 작업 시간 초과 ---")
    return {"status": "error", "message": "작업 시간(10단계)을 초과했습니다."}