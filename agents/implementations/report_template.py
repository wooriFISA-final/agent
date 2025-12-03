import logging
from typing import Dict, Any
from agents.base.agent_base import AgentBase, BaseAgentConfig
from agents.registry.agent_registry import AgentRegistry
from agents.config.base_config import AgentState
from datetime import date

logger = logging.getLogger("agent_system")


@AgentRegistry.register("report_agent")
class ReportAgent(AgentBase):

    REPORT_SCHEDULE_DAY = 1

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)

        self.allowed_tools = [
            "get_report_member_details",
            "get_user_consume_data_raw",
            "get_recent_report_summary",
            "analyze_user_profile_changes_tool",
            "analyze_user_spending_tool",
            "analyze_investment_profit_tool",
            "check_and_report_policy_changes_tool",
            "save_report_document",
        ]

        self.allowed_agents = []

    # --------------------------
    # 기본 입력 검증
    # --------------------------
    def validate_input(self, state: Dict[str, Any]) -> bool:
        messages = state.get("messages")
        if not messages or not isinstance(messages, list):
            logger.error(f"[{self.name}] messages must be a non-empty list.")
            return False
        return True

    # --------------------------
    # 사전 실행 처리
    # --------------------------
    def pre_execute(self, state: AgentState) -> AgentState:

        # 1. user_id 설정
        if "user_id" not in state:
            import re
            messages = state.get("messages", []) or state.get("global_messages", [])
            found = None

            for msg in reversed(messages):
                text = msg.content if hasattr(msg, "content") else str(msg)

                m1 = re.search(r"(\d+)번\s*사용자", text)
                if m1:
                    found = int(m1.group(1))
                    break

                m2 = re.search(r"user_id[:\s]+(\d+)", text, re.IGNORECASE)
                if m2:
                    found = int(m2.group(1))
                    break

            if found:
                state["user_id"] = found
                logger.info(f"[{self.name}] Extracted user_id: {found}")
            else:
                input_data = state.get("input", {})
                if isinstance(input_data, dict) and "user_id" in input_data:
                    state["user_id"] = input_data["user_id"]
                else:
                    state["user_id"] = 1
                    logger.info(f"[{self.name}] user_id not found. Default=1")

        # 2. 보고서 기준 월(report_month_str)
        if "report_month_str" not in state:
            import re
            messages = state.get("messages", []) or state.get("global_messages", [])

            found_date = None
            for msg in reversed(messages):
                text = msg.content if hasattr(msg, "content") else str(msg)

                m = re.search(r"(\d{4})년\s*(\d{1,2})월", text)
                if m:
                    year, month = m.groups()
                    found_date = f"{year}-{int(month):02d}-01"
                    break

                m2 = re.search(r"(\d{4})-(\d{1,2})", text)
                if m2:
                    year, month = m2.groups()
                    found_date = f"{year}-{int(month):02d}-01"
                    break

            if found_date:
                state["report_month_str"] = found_date
                logger.info(f"[{self.name}] Extracted report month: {found_date}")
            else:
                today = date.today()
                state["report_month_str"] = today.strftime("%Y-%m-01")
                logger.warning(f"[{self.name}] No report month found. Default=current month")

        return state

    # --------------------------
    # Agent 역할 정의 프롬프트
    # --------------------------
    def get_agent_role_prompt(self) -> str:
        return """
당신은 월간 금융 보고서를 자동 생성하는 Report Agent입니다.

반드시 아래 순서를 지켜 Tool을 호출하십시오.
사용자에게 질문하거나 순서를 건너뛰면 안 됩니다.

[필수 실행 순서]
1) get_report_member_details
2) get_user_consume_data_raw (최근 2개월)
3) get_recent_report_summary
4) analyze_user_profile_changes_tool
5) analyze_user_spending_tool
6) analyze_investment_profit_tool
7) check_and_report_policy_changes_tool
8) save_report_document

[추가 생성해야 할 항목]
- cluster_nickname: 소비 상위 카테고리를 기반으로 창의적으로 생성
- consume_report: 소비 분석 텍스트
- threelines_summary: 전체 분석 3줄 요약

[DB 저장 시 포함할 metadata]
- cluster_nickname
- consume_report
- consume_analysis_summary
- spend_chart_json
- change_analysis_report
- change_raw_changes
- profit_analysis_report (빈 문자열)
- net_profit
- profit_rate
- trend_chart_json
- fund_comparison_json
- policy_analysis_report
- policy_changes
- threelines_summary

모든 Tool 실행이 끝나면 save_report_document를 호출해 보고서를 저장하고 종료하십시오.
"""

    # --------------------------
    # 본문 Prompt 템플릿
    # --------------------------
    def get_prompt_template(self) -> str:
        return """
📌 자동 보고서 생성 시작

user_id: {user_id}
report_month_str: {report_month_str}

아래 순서를 정확히 따르십시오.

1. get_report_member_details(user_id={user_id})
2. get_user_consume_data_raw(user_id={user_id}, dates=["최근2개월"])
3. get_recent_report_summary(member_id={user_id}, report_date_for_comparison="직전월")
4. analyze_user_profile_changes_tool(...)
5. analyze_user_spending_tool(...)
6. analyze_investment_profit_tool(user_id={user_id})
7. check_and_report_policy_changes_tool(report_month_str={report_month_str})

[중요] 5~7번 결과를 활용해:
- cluster_nickname 생성
- consume_report 작성
- threelines_summary 생성

8. save_report_document 호출:
   - member_id: {user_id}
   - report_date: {report_month_str}
   - report_text: 생성한 threelines_summary
   - metadata: 모든 Tool 결과를 포함한 딕셔너리

metadata 필수 필드:
- cluster_nickname (문자열)
- consume_report (문자열)
- consume_analysis_summary (객체)
- spend_chart_json (문자열)
- change_analysis_report (문자열)
- change_raw_changes (리스트)
- profit_analysis_report (빈 문자열 "")
- net_profit (숫자)
- profit_rate (숫자)
- trend_chart_json (문자열)
- fund_comparison_json (문자열)
- policy_analysis_report (문자열)
- policy_changes (리스트)
- threelines_summary (문자열)

저장 성공 후 최종 응답을 보내고 종료하십시오.
"""

