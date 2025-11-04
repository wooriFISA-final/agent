from report.compare_agent.state import AgentState
from report.compare_agent.nodes.mcp_nodes import query_mysql

##########################################
###  LLM 활용이 필요없는 노드들을 정의하는 파일  ###
##########################################


def load_prev_month_report(state):
    """
    mcp 서버를 이용해 db에서 이전 달 레포트 데이터를 가져옴
    """
    print("🗂️ 이전 달 레포트 데이터 MCP 서버에서 가져오기...")
    query = f"SELECT * FROM reports WHERE member_id = {state.member_id} ORDER BY month DESC LIMIT 1"
    return query_mysql(state, query, key="report_data")


def load_house_info(state: AgentState) -> AgentState:
    print("🏠 주택 정보 검색 중...")
    # TODO: RAG 검색 로직
    state["house_info"] = {"avg_price": 420000000, "region": "Seoul"}
    return state


def load_policy_info(state: AgentState) -> AgentState:
    print("📜 정책 정보 검색 중...")
    # TODO: RAG 검색 로직
    state["policy_info"] = {"new_policy": "청년 주택 대출 한도 2배 확대"}
    return state


def load_credit_info(state: AgentState) -> AgentState:
    print("💳 개인 신용정보 불러오는 중...")
    # TODO: MCP(MySQL) SELECT 쿼리 실행
    state["credit_info"] = {"score": 780, "debt": 1200}
    return state
