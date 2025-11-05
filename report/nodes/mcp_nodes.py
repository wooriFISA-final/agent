import requests
from typing import Dict, Any, Union

# ⚠️ 주의: AgentState는 최상위 state.py에 정의되어야 합니다.
# 여기서는 타입 힌트만 Dict[str, Any]로 대체하여 사용합니다.

# ==============================================================================
# 🛠️ MCP(Microservice Communication Protocol) 설정
# ==============================================================================
# 필요에 따라서 내부 url 을 변경하거나, .env 파일에서 불러와야 합니다.
MCP_BASE_URL = "http://localhost:8001"

def query_mysql(state: Dict[str, Any], query: str, params: Union[list, dict, None] = None, key: str = "db_result") -> Dict[str, Any]:
    """
    param으로 받은 쿼리를 MCP 서버를 이용해서 실행하는 함수.
    
    Args:
        state (Dict[str, Any]): 현재 에이전트 상태 딕셔너리.
        query (str): 실행할 SQL 쿼리 문자열.
        params (Union[list, dict, None]): 쿼리에 바인딩할 파라미터.
        key (str): 쿼리 결과를 저장할 상태 키.

    Returns:
        Dict[str, Any]: 쿼리 결과가 저장된 업데이트된 상태 딕셔너리.
    
    Raises:
        RuntimeError: MCP 서버와 통신 중 오류가 발생할 경우.
    """
    
    log_query = query.replace('\n', ' ').strip()[:70] + ('...' if len(query) > 70 else '')
    print(f"🧭 [MCP Node] MySQL 쿼리 실행 중 (키: {key}, 쿼리: {log_query})")
    
    payload = {"query": query, "params": params}
    
    try:
        response = requests.post(f"{MCP_BASE_URL}/query", json=payload, timeout=30)
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        
        # 쿼리 성공 시, 결과를 지정된 키에 저장
        state[key] = response.json().get("data")
        print(f"✅ [MCP Node] 쿼리 실행 성공. 결과가 state['{key}']에 저장됨.")
        
    except requests.exceptions.RequestException as e:
        # requests 관련 오류 처리 (연결 실패, 타임아웃, HTTP 오류 등)
        raise RuntimeError(f"❌ MCP 서버 통신 오류 (URL: {MCP_BASE_URL}): {e}")
        
    return state

# LLM이 필요한 노드 함수:
# - query_mysql