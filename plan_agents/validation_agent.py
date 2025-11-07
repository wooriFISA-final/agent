import re
import os
import ollama
import json
import logging
import asyncio # 비동기 노드에서 동기 LLM 호출을 위해 필수
from difflib import get_close_matches # (이 방식에서는 사용되지 않음)
from typing import List, Dict, Any, TypedDict, Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain/LangGraph 관련 임포트
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_models import ChatOllama

# [신규!] 순환 참조를 피하기 위한 '타입 힌트' 전용 임포트
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # 'plan_graph'의 위치에 따라 경로 수정 (. 또는 ..)
    from ..plan_graph import GraphState 

# --- 로거, DB 설정 (동일) ---
# [수정] logging.basicConfig는 main.py나 plan_graph.py에서 한 번만
logger = logging.getLogger(__name__)

load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")

try:
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
except Exception as e:
    logger.error(f"DB 연결 실패: {e}")
    engine = None

# =================================================================
# 🛠️ [손발] VALIDATION TOOLKIT 함수들 (참조용으로 유지)
# =================================================================
# (DB 로드 함수 외에는 아래 'LLM-Only Judge' 방식에서 직접 호출되지 않습니다)

def load_valid_locations_from_db() -> List[str]:
    # (로직 동일 - LLM에게 주입할 목록을 위해 필수)
    if not engine: logger.error("DB 엔진이 초기화되지 않았습니다."); return []
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT region_nm FROM state"))
            locations = [row[0] for row in result.fetchall()]
            logger.info(f"[Toolkit] DB에서 {len(locations)}개의 유효한 지역명 로드 완료.")
            return locations
    except Exception as e:
        logger.error(f"DB에서 지역명 로드 실패: {e}")
        return []

def tool_sanitize_inputs(responses: Dict[str, Any]) -> Dict[str, Any]:
    # (로직 동일 - 참고용)
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
    # (로직 동일 - 참고용)
    for key, val in responses.items():
        val_str = str(val)
        if not val_str or val_str.strip() == "":
            return {"status": "error", "message": f"'{key}' 값이 비어 있습니다."}
        if re.search(r"-", val_str):
            return {"status": "error", "message": f"'{key}'에는 음수를 입력할 수 없습니다."}
    return {"status": "success", "message": "모든 입력 형식이 유효합니다."}

def _internal_normalize_location(loc: str) -> str:
    # (로직 동일 - 참고용)
    loc = loc.strip()
    mapping = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"}
    for short, full in mapping.items():
        if loc.startswith(short): loc = loc.replace(short, full, 1); break
    seoul_districts = ["강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천", "노원", "도봉", "동대문", "동작", "마포", "서대문", "서초", "성동", "성북", "송파", "양천", "영등포", "용산", "은평", "종로", "중", "중랑"]
    for gu in seoul_districts:
        if loc.startswith(gu): loc = f"서울특별시 {gu}구"; break
    return loc

def _internal_simplify_non_seoul(loc: str) -> str:
    # (로직 동일 - 참고용)
    if loc.startswith("서울"): return loc
    match = re.match(r"^(\S+시|\S+특별자치시|\S+도)", loc)
    if match: return match.group(1)
    return loc

def tool_validate_location(location_input: str, valid_locations_list: List[str]) -> Dict[str, Any]:
    # (로직 동일 - 참고용)
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

def tool_generate_friendly_error_message(technical_error_message: str, model="qwen3:8b") -> str:
    # (로직 동일 - 참고용. 이제 LLM이 직접 친절한 메시지를 생성합니다)
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
        logger.error(f"LLM 오류 메시지 생성 실패: {e}")
        return f"오류: {technical_error_message} 다시 입력해주세요."


# =================================================================
#  GRAPH STATE 정의 (삭제)
# =================================================================
# (자체 State 정의 삭제)

# =================================================================
# 🧠 [뇌] ValidationAgent (🆕 LLM-Only Judge)
# =================================================================

class ValidationAgent:
    """
    [수정됨]
    LangGraph 노드에서 단일 LLM 호출을 통해 입력을 검증합니다.
    ReAct 루프 대신, 강력한 페르소나와 작업 지시(Task)가 담긴
    단일 프롬프트를 사용하여 LLM이 직접 모든 검증을 '판단'하도록 합니다.
    """

    def __init__(self, model="qwen3:8b"):
        self.model = model
        logger.info("ValidationAgent (LLM-Only Judge): 에이전트 초기화 중...")
        
        # 유효한 지역 목록은 여전히 DB에서 로드합니다.
        # 이 목록은 LLM의 프롬프트에 주입됩니다.
        self.VALID_LOCATIONS_LIST = load_valid_locations_from_db()
        
        # [신규] ReAct 대신 사용할 단일 프롬프트 (페르소나 + Tasks)
        # {{VALID_LOCATIONS_JSON}} 부분은 나중에 실제 DB 값으로 대체됩니다.
        self.VALIDATION_SYSTEM_PROMPT_TEMPLATE = f"""
        당신은 '입력 검증 전문 에이전트'입니다.
        아래는 입력 가능한 필드 목록입니다:
        ['target_house_price', 'target_location', 'housing_type', 'available_assets', 'income_usage_ratio']

        입력된 필드명이 이 목록에 포함되어 있다면 유효한 필드로 인정하세요.
        필드명이 일치하지 않는다고 경고하지 마세요.
        
        **[검증 절차 (TASKS)]**
        당신은 다음 절차를 *반드시* 순서대로 모두 수행해야 합니다.

        1.  **기본 형식 검증**:
            * 입력된 JSON 객체에 빈 값(e.g., "", null)이 있는지 확인합니다.
            * 숫자가 입력되어야 할 필드(e.g., 'target_house_price')에 음수나 0이 있는지 확인합니다.

        2.  **입력값 정제 (Sanitization)**:
            * 'target_house_price' 같은 숫자 필드에서 '원', ',' 같은 불필요한 문자를 제거하고 숫자로 변환합니다.
            * 모든 문자열 입력의 앞뒤 공백을 제거합니다.

        3.  **핵심 'location' 필드 검증**:
            * 입력된 'location' 값을 정규화합니다. (e.g., "서울" -> "서울특별시", "부산" -> "부산광역시", "송파" -> "서울특별시 송파구")
            * 정규화된 'location'이 아래의 **[유효한 지역 목록]** 중 하나와 정확히 일치하는지 확인합니다.
            * 만약 '서울특별시' 외의 지역(e.g., '부산광역시 금정구')이 입력되면, '부산광역시'처럼 상위 지역명으로 단순화하여 목록과 비교합니다.
            * **[유효한 지역 목록]**:
                ```json
                {{VALID_LOCATIONS_JSON}}
                ```
            * **[위치 교정]**: 목록에 정확히 일치하지는 않지만, 매우 유사한 경우(e.g., "성남시분당구", "성남") '경기도'의 '성남시'로 교정할 수 있습니다. (단, 목록에 '성남시'가 있어야 함)

        **[출력 형식 (OUTPUT FORMAT)]**
        검증 결과를 *반드시* 다음 두 가지 JSON 형식 중 하나로만 반환해야 합니다.
        다른 말은 절대 덧붙이지 마세요.

        1.  **[검증 성공 시]**:
            * 모든 검증을 통과하고, 값(e.g., 'location')이 교정된 경우.
            * 'validated_data'에는 정제되고 교정된 *최종* 데이터를 포함해야 합니다.
            ```json
            {{
                "status": "success",
                "validated_data": {{
                    "location": "서울특별시 송파구", 
                    "target_house_price": 100000000
                }}
            }}
            ```

        2.  **[검증 실패 시]**:
            * 하나라도 검증에 실패한 경우 (e.g., 빈 값, 음수, 알 수 없는 지역).
            * 'message' 필드에는 **[친절한 오류 메시지]**를 담아야 합니다.
            * **[친절한 오류 메시지]**: 사용자에게 "정확한 원인"과 "해결 방법"을 공손하게 설명하고, "다시 입력해주세요."로 끝나는 문장.
            ```json
            {{
                "status": "error",
                "message": "입력하신 '서울시 강남' 지역을 찾을 수 없습니다. '강남구' 또는 '서울특별시 강남구'처럼 입력해주시겠어요? 다시 입력해주세요."
            }}
            ```
        """

    # [신규] ReAct 루프(_run_internal_agent_executor)를 대체하는 단일 LLM 호출 함수
    def _run_single_validation_call(self, user_input_data: Dict[str, Any]) -> Dict:
        """
        페르소나와 Task가 정의된 단일 프롬프트로 LLM을 호출하여
        모든 검증을 한 번에 수행합니다.
        """
        logger.info(f"\n--- 🚀 [Node-Internal] LLM-Only Judge 시작 ---")
        logger.info(f"[입력] {user_input_data}")

        # 1. 프롬프트에 실시간 DB 정보 주입
        locations_json = json.dumps(self.VALID_LOCATIONS_LIST, ensure_ascii=False)
        system_prompt = self.VALIDATION_SYSTEM_PROMPT_TEMPLATE.replace(
            "{{VALID_LOCATIONS_JSON}}", locations_json
        )
        
        user_prompt = f"다음 입력을 검증해주세요: {json.dumps(user_input_data, ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # 2. LLM을 *단 한 번* 호출 (ReAct 루프 없음)
            logger.info("\n[Node-Internal] Judging... 🧠")
            res = ollama.chat(
                model=self.model, 
                messages=messages, 
                format="json"  # JSON 모드 사용
            )
            llm_response_str = res["message"]["content"].strip()
            
            # [!!!] 🔥 디버깅 코드 (LLM의 원본 응답 확인) [!!!]
            print("="*50)
            print(f"[!!!] LLM 원본 응답 (LLM-Only Judge) [!!!]\n{llm_response_str}")
            print("="*50)
            # [!!!] 🔥 여기까지 [!!!]
            
            response_json = json.loads(llm_response_str)

            # 3. LLM의 '판단' 결과 분석
            status = response_json.get("status")
            
            if status == "success":
                validated_data = response_json.get("validated_data")
                if not validated_data:
                    # LLM이 지시를 어기고 success인데 데이터를 안 줌
                    raise ValueError("LLM이 'success'를 반환했지만 'validated_data'가 없습니다.")
                
                logger.info("\n[Node-Internal] Final Answer... ✅ (Success)")
                logger.info(f"최종 결과: {validated_data}")
                # LangGraph 노드가 기대하는 형식으로 반환
                return {"status": "success", "data": validated_data}
            
            elif status == "error":
                error_message = response_json.get("message")
                if not error_message:
                    # LLM이 지시를 어기고 error인데 메시지를 안 줌
                    raise ValueError("LLM이 'error'를 반환했지만 'message'가 없습니다.")

                logger.info(f"\n[Node-Internal] Final Answer... ❌ (Error)")
                logger.info(f"오류 메시지: {error_message}")
                
                # LLM이 직접 생성한 친절한 메시지를 사용
                # LangGraph 노드가 기대하는 형식으로 반환
                return {"status": "error", "message": error_message}

            else:
                # LLM이 status 필드를 빼먹음
                raise ValueError(f"LLM이 'status' 필드(success/error)가 없는 부적절한 JSON을 반환했습니다: {llm_response_str}")

        except json.JSONDecodeError:
            logger.warning(f"[Node-Internal] Judge Error... 💥] LLM이 유효한 JSON을 뱉지 않음: {llm_response_str}")
            return {"status": "error", "message": "시스템 내부 오류가 발생했습니다. (JSON 파싱 실패)"}
        
        except Exception as e:
            logger.error(f"[Node-Internal] Judge Error... 💥] {e}", exc_info=True)
            return {"status": "error", "message": f"검증 중 알 수 없는 오류가 발생했습니다: {e}"}


    # [연결잭] (수정됨: 호출 대상 함수 변경)
    def create_validation_node(self):
        """
        LangGraph에 등록할 '단일 검증 노드'를 생성하여 반환합니다.
        ('지연 임포트' 꼼수 적용됨)
        """
        
        async def validation_node(state):
            
            # [신규!] '지연 임포트'로 순환 참조 회피
            try:
                from agent.plan_graph import GraphState
            except ImportError:
                from ..plan_graph import GraphState
            
            state: "GraphState" = state 
            
            logger.info("🔍 ValidationAgent (LLM-Only Judge): 노드 실행...")
            
            try:
                input_data = state.get("original_input")
                if not input_data:
                    raise ValueError("검증할 'original_input' 데이터가 state에 없습니다.")

                # [수정!] ReAct 루프(_run_internal_agent_executor) 대신
                # '단일 호출' 함수(_run_single_validation_call)를 실행
                final_result = await asyncio.to_thread(
                    self._run_single_validation_call, # 👈 호출 대상 변경
                    input_data
                )
                
                # 결과 반환 로직은 거의 동일 (final_result 형식은 유지됨)
                status = final_result.get("status", "error")
                
                if status == "success":
                    message = f"검증 성공. {final_result.get('data')}"
                    logger.info(f"✅ ValidationAgent (LLM-Only Judge): 노드 완료. (결과: Success)")
                    return {
                        "final_response": final_result, # {"status": "success", "data": ...}
                        "messages": [AIMessage(content=f"[검증 결과: {status}] {message}")]
                    }
                else: # status == "error"
                    message = final_result.get("message", "Unknown error")
                    logger.info(f"✅ ValidationAgent (LLM-Only Judge): 노드 완료. (결과: Error)")
                    return {
                        "final_response": final_result, # {"status": "error", "message": ...}
                        "messages": [AIMessage(content=f"[검증 결과: {status}] {message}")]
                    }

            except Exception as e:
                # 래퍼 예외 처리 (로직 동일)
                logger.error(f"❌ ValidationAgent (LLM-Only Judge) 래퍼 오류: {e}", exc_info=True)
                error_msg = f"검증 에이전트 래퍼(Wrapper) 실행 실패: {e}"
                final_response = {"status": "error", "message": error_msg}
                
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "final_response": final_response
                }
        
        return validation_node