from state import AgentState
from .mcp_nodes import query_mysql

##########################################
###  LLM 활용이 필요없는 노드들을 정의하는 파일  ###
##########################################

def load_prev_month_report(state):
    """
    mcp 서버를 이용해 db에서 이전 달 레포트 데이터를 가져오는 노드
    """
    print("🗂️ 이전 달 레포트 데이터 MCP 서버에서 가져오기...")
    
    # agent 로직 테스트 용 로직
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 리포트 데이터를 불러옵니다.")
        state["report_data"] = {
            "month": "2025-10",
            "income": 5000000,
            "loan_balance": 20000000,
            "credit_score": 800,
            "target_location": "서울 송파구",
            "avg_house_price": 400000000,
            "policy_content": "규제지역의 LTV를 40%로 축소하고, 주택 임대 및 매매사업자 대출을 금지"
        }
        return state

    # test가 아닐 때 로직
    query = f"SELECT * FROM reports WHERE member_id = {state['member_id']} ORDER BY month DESC LIMIT 1"
    return query_mysql(state, query, key="report_data")


def load_house_info(state: AgentState) -> AgentState:
    """
    RAG 에서 주택 정보를 가져오는 노드
    """

    print("🏠 주택 정보 검색 중...")

    # test 용 코드
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 주택 정보를 불러옵니다.")
        state["house_info"] = {
            "price": 420000000,
            "location": "서울 송파구",
        }
        return state
    
    # TODO: RAG 검색 로직
    state["house_info"] = {"avg_price": 420000000, "region": "Seoul"}
    return state


def load_policy_info(state: AgentState) -> AgentState:
    """
    RAG 에서 정책 정보를 가져오는 노드
    """

    print("📜 정책 정보 검색 중...")

    # test 용 코드
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 주택 정보를 불러옵니다.")
        state["policy_info"] = {
            "content": "10월 15일 대책 발표로 서울 전역과 경기도 12개 지역이 토지거래허가구역으로 추가 지정되었습니다. 이는 10월 20일부터 효력이 발생함",
            "updated_at": "2025-10-15",
        }
        return state
    

    # TODO: RAG 검색 로직
    state["policy_info"] = {"new_policy": "청년 주택 대출 한도 2배 확대"}
    return state


def load_credit_info(state: AgentState) -> AgentState:
    print("💳 개인 신용정보 불러오는 중...")

    # test 용 코드
    if state.get("is_test"):
        print("🧪 [TEST MODE] 더미 주택 정보를 불러옵니다.")
        state["credit_info"] = {"score": 780, "debt": 1200}
        return state


    # TODO: MCP(MySQL) SELECT 쿼리 실행
    state["credit_info"] = {"score": 780, "debt": 1200}
    return state
