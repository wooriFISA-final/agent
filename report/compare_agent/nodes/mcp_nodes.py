import requests
from report.compare_agent.state import AgentState

# 필요에 따라서 내부 url 을 변경하거나, .env 파일에서 불러오기
MCP_BASE_URL = "http://localhost:8001"

def query_mysql(state: AgentState, query: str, params=None, key: str = "db_result"):
    """
    param으로 받은 쿼리를 mcp 서버를 이용해서 실행하는 함수

    return
    AgentState
    성공: staet[key]에 쿼리의 결과 저장 후 반환
    실패: RuntimeError 반환
    """
    print(f"🧭 MCP(MySQL) 쿼리 실행 중: {query}")
    response = requests.post(f"{MCP_BASE_URL}/query", json={"query": query, "params": params})
    if response.status_code == 200:
        state[key] = response.json()["data"]
    else:
        raise RuntimeError(f"MCP 서버 오류: {response.text}")
    return state
