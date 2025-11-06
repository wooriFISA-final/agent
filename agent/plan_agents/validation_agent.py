import re
import os
import ollama
import json
import logging
import asyncio # 비동기 노드에서 동기 ReAct를 돌리기 위해 필수
from difflib import get_close_matches
from typing import List, Dict, Any, TypedDict, Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain/LangGraph 관련 임포트 (IntentClassifierAgent와 동일)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph.message import MessagesState
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field # Pydantic은 이 예제에선 직접 쓰이진 않음

# --- 로거, DB 설정 (validation_agent.py와 동일) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
# 🛠️ [손발] VALIDATION TOOLKIT 함수들
# =================================================================
# (validation_agent.py의 툴킷 함수 5개를 그대로 복사)

def load_valid_locations_from_db() -> List[str]:
    # (이전 코드와 동일)
    if not engine:
        logger.error("DB 엔진이 초기화되지 않았습니다. 빈 리스트를 반환합니다.")
        return []
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
    # (이전 코드와 동일)
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
    # (이전 코드와 동일)
    for key, val in responses.items():
        val_str = str(val)
        if not val_str or val_str.strip() == "":
            return {"status": "error", "message": f"'{key}' 값이 비어 있습니다."}
        if re.search(r"-", val_str):
            return {"status": "error", "message": f"'{key}'에는 음수를 입력할 수 없습니다."}
    return {"status": "success", "message": "모든 입력 형식이 유효합니다."}

def _internal_normalize_location(loc: str) -> str:
    # (이전 코드와 동일)
    loc = loc.strip()
    mapping = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"}
    for short, full in mapping.items():
        if loc.startswith(short): loc = loc.replace(short, full, 1); break
    seoul_districts = ["강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천", "노원", "도봉", "동대문", "동작", "마포", "서대문", "서초", "성동", "성북", "송파", "양천", "영등포", "용산", "은평", "종로", "중", "중랑"]
    for gu in seoul_districts:
        if loc.startswith(gu): loc = f"서울특별시 {gu}구"; break
    return loc

def _internal_simplify_non_seoul(loc: str) -> str:
    # (이전 코드와 동일)
    if loc.startswith("서울"): return loc
    match = re.match(r"^(\S+시|\S+특별자치시|\S+도)", loc)
    if match: return match.group(1)
    return loc

def tool_validate_location(location_input: str, valid_locations_list: List[str]) -> Dict[str, Any]:
    # (이전 코드와 동일)
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
    # (이전 코드와 동일)
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
#  GRAPH STATE 정의 (IntentClassifierAgent와 유사)
# =================================================================
class ValidationState(MessagesState):
    """
    이 노드가 LangGraph와 주고받을 상태
    """
    # [입력] 검증이 필요한 원본 데이터
    original_input: Dict[str, Any]
    
    # [출력] 이 노드가 실행된 후의 최종 결과
    final_response: Optional[Dict[str, Any]]


# =================================================================
# 🧠 [뇌] ValidationAgent (IntentClassifierAgent 형식)
# =================================================================

class ValidationAgent:
    """
    기존 'validation_agent.py'의 ReAct 로직 전체를
    단일 LangGraph 노드로 래핑(Wrapping)하는 에이전트.
    """

    def __init__(self, model="qwen3:8b"):
        # 1. IntentClassifierAgent처럼 LLM 모델명 저장
        self.model = model
        
        # 2. ReAct 에이전트(뇌)에 필요한 모든 '준비물'을 self에 로드
        logger.info("ValidationAgent (ReAct-in-Node): 내부 에이전트 초기화 중...")
        
        self.VALID_LOCATIONS_LIST = load_valid_locations_from_db()
        
        # 3. '손발'이 될 툴킷 정의
        self.TOOLS_AVAILABLE = {
            "tool_sanitize_inputs": tool_sanitize_inputs,
            "tool_check_input_format": tool_check_input_format,
            "tool_validate_location": tool_validate_location,
            "tool_generate_friendly_error_message": tool_generate_friendly_error_message,
        }
        
        # 4. [핵심] ReAct 에이전트가 사용할 페르소나와 TASK (System Prompt)
        # (validation_agent.py의 SYSTEM_PROMPT를 그대로 가져옴)
        self.REACT_SYSTEM_PROMPT = f"""
        당신은 '입력 검증 전문 에이전트'입니다.
        당신의 임무는 사용자의 JSON(딕셔너리) 입력을 받아, 사용 가능한 도구(Tool)들을 순서대로 호출하여 입력을 검증하고 최종 결과를 반환하는 것입니다.
        
        **[검증 절차]**
        1.  먼저 `tool_sanitize_inputs`를 사용해 입력을 정제합니다.
        2.  정제된 결과로 `tool_check_input_format`을 호출해 기본 형식을 검사합니다.
        3.  형식 검사가 통과되면, `target_location` 값으로 `tool_validate_location`을 호출해 지역명을 검증합니다.
        4.  만약 2단계나 3단계에서 'error'가 발생하면, 즉시 검증을 중단하고 `tool_generate_friendly_error_message`를 호출하여 사용자에게 친절한 '오류 메시지'를 반환하세요.
        5.  모든 검증(1, 2, 3)이 성공하면, 최종적으로 "검증 완료" 상태와 "보정된 데이터"를 반환하세요.
        
        **[Tool 사용 규칙]**
        (이하 validation_agent.py의 프롬프트와 동일...)
        
        [JSON 출력 예시 (Tool 호출 시)]
        {{
          "thought": "데이터를 받았으니 먼저 정제해야겠다.",
          "tool_call": {{
            "name": "tool_sanitize_inputs",
            "args": {{"responses": {{"key": "value", ...}} }}
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

        **[사용 가능한 Tool 목록]**
        ---
        1.  **tool_sanitize_inputs(responses: Dict)**:
            - {tool_sanitize_inputs.__doc__}
        2.  **tool_check_input_format(responses: Dict)**:
            - {tool_check_input_format.__doc__}
        3.  **tool_validate_location(location_input: str, valid_locations_list: List = None)**:
            - {tool_validate_location.__doc__}
        4.  **tool_generate_friendly_error_message(technical_error_message: str)**:
            - {tool_generate_friendly_error_message.__doc__}
        ---
        """

    # 5. [엔진] 기존 run_agent_executor 로직을 '비공개 메서드'로 이식
    # (self를 인자로 받도록 수정)
    def _run_internal_agent_executor(self, user_input_data: Dict[str, Any]) -> Dict:
        """
        'validation_agent.py'의 'run_agent_executor' 로직과 100% 동일합니다.
        단, self.model, self.REACT_SYSTEM_PROMPT 등을 사용합니다.
        
        [중요] 이 함수는 '동기(Synchronous)'입니다. (ollama.chat 사용)
        """
        
        logger.info(f"\n--- 🚀 [Node-Internal] ReAct 루프 시작 ---")
        logger.info(f"[입력] {user_input_data}")

        messages = [
            {"role": "system", "content": self.REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 입력을 검증해주세요: {json.dumps(user_input_data, ensure_ascii=False)}"}
        ]
        
        error_tool_called = False

        for _ in range(10): # 최대 10번의 ReAct 루프
            logger.info("\n[Node-Internal] Thinking... 🧠")
            
            try:
                res = ollama.chat(model=self.model, messages=messages, format="json")
                llm_response_str = res["message"]["content"].strip()
                messages.append({"role": "assistant", "content": llm_response_str})

                response_json = json.loads(llm_response_str)

                if "tool_call" in response_json:
                    tool_call = response_json.get("tool_call", {})
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    
                    if not tool_name or tool_name not in self.TOOLS_AVAILABLE:
                        raise ValueError(f"LLM이 유효하지 않은 Tool을 호출했습니다: {tool_name}")

                    logger.info(f"[Node-Internal] Action... 🎬] '{tool_name}' Tool 호출")
                    
                    tool_function = self.TOOLS_AVAILABLE[tool_name]
                    
                    if tool_name == "tool_validate_location":
                        tool_args["valid_locations_list"] = self.VALID_LOCATIONS_LIST
                        tool_result = tool_function(**tool_args)
                    
                    elif tool_name == "tool_generate_friendly_error_message":
                        error_tool_called = True
                        tool_args["model"] = self.model # 모델명 주입
                        tool_result = tool_function(**tool_args)
                    
                    else:
                        tool_result = tool_function(**tool_args)
                    
                    observation = f"ToolResult: {json.dumps(tool_result, ensure_ascii=False)}"
                    logger.info(f"[Node-Internal] Observation... 📝] {observation[:100]}...")
                    messages.append({"role": "user", "content": observation})
                    
                    if error_tool_called:
                        messages.append({
                            "role": "user", 
                            "content": "방금 받은 'ToolResult' 문자열을 'final_answer' JSON 형식으로 포장해서 즉시 반환하세요."
                        })

                elif "final_answer" in response_json:
                    logger.info("\n[Node-Internal] Final Answer... ✅]")
                    final_data = response_json["final_answer"]
                    logger.info(f"최종 결과: {final_data}")
                    logger.info("--- 🏁 [Node-Internal] ReAct 루프 완료 ---")
                    return final_data

                else:
                    raise ValueError("LLM의 응답에 'tool_call'이나 'final_answer'가 없습니다.")

            except json.JSONDecodeError:
                logger.warning(f"[Node-Internal] Tool Error... 💥] LLM이 유효한 JSON을 뱉지 않음: {llm_response_str}")
                messages.append({"role": "user", "content": "ToolError: 유효한 JSON이 아닙니다. 규칙(JSON 형식)을 다시 확인하세요."})
            
            except Exception as e:
                logger.error(f"[Node-Internal] Tool Error... 💥] {e}")
                observation = f"ToolError: {e}"
                messages.append({"role": "user", "content": observation})

        logger.warning("--- 🏁 [Node-Internal] ReAct 루프 시간 초과 ---")
        return {"status": "error", "message": "작업 시간(10단계)을 초과했습니다."}


    # 6. [연결잭] LangGraph에 노드를 제공하는 팩토리 함수
    # (IntentClassifierAgent.create_intent_node와 동일한 구조)
    def create_validation_node(self):
        """
        LangGraph에 등록할 '단일 검증 노드'를 생성하여 반환합니다.
        """
        
        # [핵심] 이 async 함수가 LangGraph의 '노드'가 됩니다.
        async def validation_node(state: ValidationState):
            logger.info("🔍 ValidationAgent (ReAct-in-Node): 노드 실행...")
            
            try:
                # 1. LangGraph State에서 입력 데이터를 가져옵니다.
                input_data = state.get("original_input")
                if not input_data:
                    raise ValueError("검증할 'original_input' 데이터가 state에 없습니다.")

                # 2. [중요!] 동기(sync) ReAct 엔진을
                #    비동기(async) 노드에서 실행하려면 반드시
                #    'asyncio.to_thread'로 감싸서 별도 스레드에서 실행해야 합니다.
                final_result = await asyncio.to_thread(
                    self._run_internal_agent_executor,
                    input_data
                )
                
                # 3. ReAct 엔진의 결과를 LangGraph State에 반영합니다.
                status = final_result.get("status", "error")
                message = final_result.get("message", "Unknown error")
                
                logger.info(f"✅ ValidationAgent (ReAct-in-Node): 노드 완료. (결과: {status})")
                
                return {
                    "final_response": final_result,
                    "messages": [AIMessage(content=f"[검증 결과: {status}] {message}")]
                }

            except Exception as e:
                # 4. 노드 실행 중 발생한 예외 처리
                logger.error(f"❌ ValidationAgent (ReAct-in-Node) 오류: {e}", exc_info=True)
                error_msg = f"검증 에이전트 래퍼(Wrapper) 실행 실패: {e}"
                final_response = {"status": "error", "message": error_msg}
                
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "final_response": final_response
                }
        
        # 5. '노드' 함수를 반환
        return validation_node