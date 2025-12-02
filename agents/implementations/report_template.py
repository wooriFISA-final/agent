import logging
from typing import Dict, Any, List
# from langchain_core.messages import HumanMessage
# agents.base.agent_base의 AgentBase와 BaseAgentConfig가 있다고 가정
from agents.base.agent_base import AgentBase, BaseAgentConfig
# agents.registry.agent_registry의 AgentRegistry와 AgentState가 있다고 가정
from agents.registry.agent_registry import AgentRegistry
from agents.config.base_config import AgentState 

# 🚨 [추가] 스케줄링 구현을 위한 datetime 임포트
from datetime import datetime, date 
import time

logger = logging.getLogger("agent_system")


@AgentRegistry.register("report_agent")
class ReportAgent(AgentBase):
    """
    Report Agent (보고서 에이전트)
    
    역할:
    - 고객의 금융 데이터를 분석하고, 정책 변동 사항을 확인하여
    - 최종 고객에게 전달할 명확하고 간결하며 전문적인 월간 재무 보고서를 작성합니다.
    
    사용 가능한 도구:
    1. analyze_user_spending_tool: 월별 소비 데이터 비교 분석 및 군집 생성
    2. analyze_investment_profit_tool: 투자 상품 손익/진척도 분석
    3. analyze_user_profile_changes_tool: 사용자 개인 지수 변동 분석 (연봉, 부채, 신용 점수)
    4. check_and_report_policy_changes_tool: 금융 정책 변동 사항 자동 비교 및 보고서 생성
    5. generate_final_summary_llm: 통합 보고서 본문을 받아 핵심 3줄 요약 생성 (deprecated)
    """
    
    # 🎯 [스케줄 설정]: 매월 보고서를 생성할 날짜 (예: 1일)
    REPORT_SCHEDULE_DAY = 1
    
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        
        # 🎯 사용 가능한 Tool 목록을 정의
        self.allowed_tools = [
            # DB 조회 도구
            "get_report_member_details",
            "get_user_consume_data_raw",
            "get_recent_report_summary",
            "get_recent_report_summary",
            # "get_user_products", # Removed
            # "get_monthly_simulation_data", # Removed
            # 분석 도구
            "analyze_user_spending_tool",
            "analyze_investment_profit_tool",
            "analyze_user_profile_changes_tool",
            "check_and_report_policy_changes_tool",
            # 저장 도구
            "save_report_document",
            # Deprecated
            "generate_final_summary_llm",
        ]
        
        # 위임 가능한 Agent는 현재 설정하지 않음
        self.allowed_agents = []

    def validate_input(self, state: Dict[str, Any]) -> bool:
        """state에 messages가 있고, HumanMessage가 포함되어 있는지 확인"""
        messages = state.get("messages")
        
        if not messages or not isinstance(messages, list):
            logger.error(f"[{self.name}] 'messages' must be a non-empty list")
            return False
            
        return True
        
    def pre_execute(self, state: AgentState) -> AgentState:
        """실행 전 전처리 및 월간 스케줄 트리거 확인"""
        
        # 0. user_id 확인 및 설정
        if "user_id" not in state:
            # 메시지에서 user_id 추출 시도
            import re
            messages = state.get("messages", []) or state.get("global_messages", [])
            
            user_id_found = None
            for msg in reversed(messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                # "1번 사용자" 또는 "user_id: 1" 패턴 찾기
                match = re.search(r"(\d+)번\s*사용자", content)
                if match:
                    user_id_found = int(match.group(1))
                    break
                match_id = re.search(r"user_id[:\s]+(\d+)", content, re.IGNORECASE)
                if match_id:
                    user_id_found = int(match_id.group(1))
                    break
            
            if user_id_found:
                state["user_id"] = user_id_found
                logger.info(f"[{self.name}] 메시지에서 user_id 추출 성공: {user_id_found}")
            else:
                # input에서 확인 시도
                input_data = state.get("input", {})
                if isinstance(input_data, dict) and "user_id" in input_data:
                    state["user_id"] = input_data["user_id"]
                else:
                    # 🚨 기본값: 1번 유저로 설정
                    logger.info(f"[{self.name}] user_id를 찾을 수 없어 기본값(1)로 설정합니다.")
                    state["user_id"] = 1

        # 1. report_month_str이 없으면 메시지에서 추출 시도
        if "report_month_str" not in state:
            import re
            messages = state.get("messages", [])
            # global_messages도 확인
            if not messages:
                messages = state.get("global_messages", [])
                
            found_date = None
            for msg in reversed(messages):
                content = msg.content if hasattr(msg, "content") else str(msg)
                # "2025년 1월" 또는 "2025-01" 패턴 찾기
                match = re.search(r"(\d{4})년\s*(\d{1,2})월", content)
                if match:
                    year, month = match.groups()
                    found_date = f"{year}-{int(month):02d}-01"
                    break
                
                match_hyphen = re.search(r"(\d{4})-(\d{1,2})", content)
                if match_hyphen:
                    year, month = match_hyphen.groups()
                    found_date = f"{year}-{int(month):02d}-01"
                    break
            
            if found_date:
                state["report_month_str"] = found_date
                logger.info(f"[{self.name}] 메시지에서 보고서 기준월 추출 성공: {found_date}")
            else:
                # 추출 실패 시 기본값 (현재 월) 또는 에러
                logger.warning(f"[{self.name}] 보고서 기준월을 찾을 수 없습니다. 현재 월로 설정합니다.")
                today = date.today()
                state["report_month_str"] = today.strftime("%Y-%m-01")

        # ----------------------------------------------------------------------
        # 🎯 [주석 처리된 월간 스케줄 트리거]
        # ----------------------------------------------------------------------
        """
        # 🚨 [트리거 로직 시작] 이 주석을 풀면, 보고서 생성일이 아닌 경우 실행이 중단됩니다.
        try:
            # 현재 날짜 및 보고서 월의 시작일 (report_month_str은 YYYY-MM-DD 형태)
            current_date = datetime.now().date()
            report_month_start = datetime.strptime(state["report_month_str"], "%Y-%m-%d").date().replace(day=1)
            
            # 다음 보고서 실행 예상일 (보고서 월의 REPORT_SCHEDULE_DAY)
            if current_date.month == report_month_start.month and current_date.year == report_month_start.year:
                # 현재 월이 보고서 월과 같으면, 해당 월의 스케줄 날짜 확인
                target_report_date = report_month_start.replace(day=self.REPORT_SCHEDULE_DAY)
            else:
                # 보고서 월이 현재 월보다 앞서 있다면(과거 보고 요청), 바로 실행 허용
                if report_month_start < current_date.replace(day=1):
                    logger.info("과거 보고서 생성이 요청되어 스케줄 체크를 건너뜁니다.")
                    return state
                    
                # 보고서 월이 현재 월보다 나중이라면, 스케줄 날짜를 다음 달로 계산
                target_month = (report_month_start.month % 12) + 1
                target_year = report_month_start.year + (1 if report_month_start.month == 12 else 0)
                target_report_date = date(target_year, target_month, self.REPORT_SCHEDULE_DAY)

            # [핵심 체크] 오늘 날짜가 목표 실행일 이전이라면 실행 중단
            if current_date < target_report_date:
                # 💡 [테스트 모드 임시 해제] 테스트를 위해 이 조건문을 주석 처리합니다.
                # error_msg = f"[{self.name}] 월간 보고서 스케줄 실행일({target_report_date.strftime('%Y-%m-%d')})이 아닙니다. 실행을 중단합니다."
                # logger.warning(error_msg)
                # raise ValueError(error_msg)
                pass # 테스트 모드에서는 통과
                
        except Exception as e:
            logger.error(f"[{self.name}] 스케줄 체크 오류: {e}")
            raise e
        # 🚨 [트리거 로직 끝] 이 주석을 풀면, 보고서 생성일이 아닌 경우 실행이 중단됩니다.
        """
        # ----------------------------------------------------------------------
        # 🎯 [테스트 모드] 주석을 풀지 않으면 항상 즉시 실행 가능합니다.
        # ----------------------------------------------------------------------
        
        # 🎯 사용 가능한 Tool 목록을 정의
        self.allowed_tools = [
            "get_report_member_details",
            "get_user_consume_data_raw",
            "get_report_member_details",
            "get_user_consume_data_raw",
            # "get_user_products", # Removed
            "get_recent_report_summary",
            # "get_monthly_simulation_data", # Removed
            # "get_fund_portfolio_data",  # Removed
            "analyze_user_profile_changes_tool",
            "analyze_user_spending_tool",
            "analyze_investment_profit_tool",
            "check_and_report_policy_changes_tool",
            "save_report_document"
        ]
        
        return state
    
    def get_agent_role_prompt(self) -> str:
        """
        Agent 역할 정의 Prompt 반환
        """
        # state에서 user_id와 report_month_str 가져오기 (동적으로 설정됨)
        return """당신은 사용자의 금융 데이터를 종합적으로 분석하여 월간 리포트를 작성하는 '금융 리포트 에이전트'입니다.

🚨 중요: 당신은 자동화된 Agent입니다. 사용자에게 질문하지 말고 즉시 도구를 호출하세요!

📋 작업 순서 (반드시 순서대로 실행):

1️⃣ 데이터 조회 단계:
   - get_report_member_details: 사용자 정보 조회
   - get_user_consume_data_raw: 소비 데이터 조회 (최근 2개월)
   - get_recent_report_summary: 직전 레포트 조회

2️⃣ 데이터 분석 단계:
   - analyze_user_profile_changes_tool: 프로필 변동 분석
   - analyze_user_spending_tool: 소비 패턴 분석
   - analyze_investment_profit_tool: 투자 손익 분석
   - check_and_report_policy_changes_tool: 정책 변동 분석

3️⃣ 레포트 작성 단계:
   - 모든 분석 결과를 종합하여 최종 레포트 텍스트 작성
   - 3줄 요약 생성 (반드시 "1. ... 2. ... 3. ..." 형식)

4️⃣ DB 저장 단계 (🚨 필수):
   - save_report_document: 작성한 레포트를 DB에 저장
   - metadata에 모든 분석 결과 포함

⚠️ 주의사항:
- 첫 번째 액션: 즉시 get_report_member_details 호출
- 사용자에게 질문하거나 응답하지 마세요
- 모든 도구를 순서대로 호출하세요
- DB 저장을 반드시 실행하세요
- 완료 후 respond 액션으로 종료하세요
"""
        
    def get_prompt_template(self) -> str:
        """
        리포트 생성 에이전트의 프롬프트 템플릿 반환
        """
        return """
🚨 당신은 자동화된 월간 금융 리포트 생성 Agent입니다.
절대 사용자에게 응답하지 말고, 즉시 도구를 호출하세요.

📌 현재 상태:
- user_id: {user_id}
- report_month_str: {report_month_str}

 첫 번째 액션: 즉시 get_report_member_details 도구를 호출하세요.
인자: {{"user_id": {user_id}}}

📋 전체 작업 순서:

1️⃣ 데이터 조회 (3개 도구를 순서대로 호출):
   a. get_report_member_details(user_id={user_id})
   b. get_user_consume_data_raw(user_id={user_id}, dates=["이전 2개월"])
   c. get_recent_report_summary(member_id={user_id}, report_date_for_comparison="직전월")
 
2️⃣ 데이터 분석 (4개 도구를 순서대로 호출):
   a. analyze_user_profile_changes_tool(current_data=..., previous_data=...)
   b. analyze_user_spending_tool(consume_records=..., member_data=...)
   c. analyze_investment_profit_tool(user_id={user_id})
   d. check_and_report_policy_changes_tool(report_month_str={report_month_str})

3️⃣ 리포트 작성:
   - 분석 결과를 종합하여 최종 리포트 텍스트 생성
   - 3줄 요약 생성 (반드시 "1. ... 2. ... 3. ..." 형식)

4️⃣ DB 저장 (🚨 필수 🚨):
   save_report_document(
     member_id={user_id},
     report_date={report_month_str},
     report_text="작성한 리포트 전체 텍스트",
     metadata={{
       "consume_report": "소비분석 텍스트",
       "cluster_nickname": "군집별명",
       "spend_chart_json": "소비차트JSON문자열",
       "change_analysis_report": "프로필변동분석 텍스트",
       "profit_analysis_report": "투자분석 텍스트",
       "trend_chart_json": "투자추이JSON문자열",
       "fund_comparison_json": "펀드비교JSON문자열",
       "policy_analysis_report": "정책분석 텍스트",
       "threelines_summary": "1. ... 2. ... 3. ..."
     }}
   )

5️⃣ 종료:
   - save_report_document 성공 확인 후 종료
   - Action: respond

⚠️ 주의사항:
- 즉시 도구를 호출하세요. 사용자에게 응답하지 마세요.
- 4단계(DB 저장)를 반드시 실행하세요.
- 모든 도구 호출 결과를 metadata에 포함하세요.
- JSON 문자열은 반드시 문자열로 변환하세요 (json.dumps 사용).
"""
