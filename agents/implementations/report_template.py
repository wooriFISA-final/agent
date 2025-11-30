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
            "get_user_products",
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
            # input에서 확인 시도
            input_data = state.get("input", {})
            if isinstance(input_data, dict) and "user_id" in input_data:
                state["user_id"] = input_data["user_id"]
            else:
                # 🚨 [임시] 테스트를 위해 무조건 1번 유저로 설정
                logger.info(f"[{self.name}] user_id가 감지되지 않아 테스트용 ID(1)로 설정합니다.")
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
        
        return state
        
    def get_agent_role_prompt(self) -> str:
        """
        Agent의 역할 정의
        
        이 Prompt 하나로 Agent의 모든 행동 원칙이 결정됨
        """
        return """ 당신은 금융 보고서 작성 전문 에이전트입니다.

주된 임무는 사용자의 금융 데이터를 DB에서 조회하고 분석하여, 최종 고객에게 전달할 명확하고 간결하며 전문적인 월간 재무 보고서를 작성하는 것입니다.

**🚨 중요: state에 user_id와 report_month_str이 이미 설정되어 있습니다. 사용자에게 묻지 말고 바로 사용하세요!**

**⚠️ 필수 체크리스트 - 모든 항목이 완료되기 전에는 절대 respond 액션을 선택하지 마세요!**
□ 1단계: state 값 확인 완료
□ 2단계: DB 조회 4개 도구 모두 호출 완료 (get_report_member_details, get_user_consume_data_raw, get_user_products, get_recent_report_summary)
□ 3단계: 분석 4개 도구 모두 호출 완료 (analyze_user_profile_changes_tool, analyze_user_spending_tool, analyze_investment_profit_tool, check_and_report_policy_changes_tool)
□ 4단계: 보고서 작성 완료
□ 5단계: save_report_document 도구 호출 완료 및 성공 확인
□ 6단계: 최종 응답 반환

**작업 흐름 (반드시 순서대로 실행):**

**1단계: state에서 필요한 값 확인**
   - user_id: state["user_id"]에 이미 설정되어 있음 (예: 1)
   - report_month_str: state["report_month_str"]에 이미 설정되어 있음 (예: "2025-01-01")

**2단계: DB에서 데이터 조회 (state의 user_id 사용)**
   a. get_report_member_details 도구 호출:
      - 인자: {"user_id": state의 user_id}
   
   b. get_user_consume_data_raw 도구 호출:
      - report_month_str에서 이전 2개월 날짜 계산 (YYYY-MM 형식으로!)
      - 예: report_month_str이 "2025-01-01"이면 dates=["2024-12", "2024-11"]
      - 인자: {"user_id": state의 user_id, "dates": [이전 2개월]}
   
   c. get_user_products 도구 호출:
      - 인자: {"user_id": state의 user_id}
   
   d. get_recent_report_summary 도구 호출:
      - report_month_str에서 이전 월 계산 (YYYY-MM-DD 형식 유지)
      - 예: report_month_str이 "2025-01-01"이면 report_date_for_comparison="2024-12-01"
      - 인자: {"member_id": state의 user_id, "report_date_for_comparison": "이전 월"}

**3단계: 데이터 분석 (DB 조회 결과를 각 도구에 전달)**
   a. analyze_user_profile_changes_tool:
      - current_data: get_report_member_details의 결과["data"]
      - previous_data: get_recent_report_summary의 결과["data"] (없으면 빈 dict)
   
   b. analyze_user_spending_tool:
      - consume_records: get_user_consume_data_raw의 결과["data"]
      - member_data: get_report_member_details의 결과["data"]
   
   c. analyze_investment_profit_tool:
      - products: get_user_products의 결과["data"]
   
   d. check_and_report_policy_changes_tool:
      - report_month_str: state의 report_month_str

**4단계: 보고서 작성**
   - 각 분석 도구의 결과를 바탕으로 섹션별 보고서 텍스트를 직접 생성
   - 모든 섹션을 통합하여 최종 보고서 작성
   - 핵심 내용 3줄 요약 생성하되, 1번 2번 3번과 같이 인덱싱을 해서 3줄로 작성

**5단계: DB에 저장 (🚨🚨🚨 절대 필수! 이 단계 없이는 작업이 완료되지 않음 🚨🚨🚨)**
   - **경고: save_report_document 도구를 호출하지 않으면 보고서가 DB에 저장되지 않습니다!**
   - **이 단계를 건너뛰면 안 됩니다. 반드시 실행하세요!**
   - save_report_document 도구를 호출하여 보고서를 DB에 저장하세요
   - 인자:
      * member_id: state의 user_id
      * report_date: state의 report_month_str
      * report_text: 작성한 최종 보고서 전체 텍스트
      * metadata: 각 분석 결과의 메타데이터 (JSON 형식)
         - consume_report: 소비 분석 보고서 텍스트
         - cluster_nickname: 군집 별명
         - consume_analysis_summary: 소비 분석 요약 데이터
         - spend_chart_json: 소비 차트 데이터
         - change_analysis_report: 개인 지표 변동 보고서
         - change_raw_changes: 변동 내역 리스트
         - profit_analysis_report: 투자 분석 보고서
         - net_profit: 순손익
         - profit_rate: 수익률
         - policy_analysis_report: 정책 분석 보고서
         - policy_changes: 정책 변동 리스트
         - threelines_summary: 3줄 요약
   
   예시 (JSON 형식 오류 수정):
   {
     "member_id": 1,
     "report_date": "2025-01-01",
     "report_text": "작성한 최종 보고서 전체 내용...",
     "metadata": {
       "consume_report": "소비 분석 텍스트...",
       "cluster_nickname": "균형잡힌 소비형",
       "threelines_summary": "1. 소비자의~ 2. 사용자의 변동사항~ 3. 주택 변동사항은~"
     }
   }

**6단계: 최종 결정 및 종료 (✅ 종료 조건 명확화)**
   - **중요: 5단계에서 save_report_document 도구를 성공적으로 호출한 후에만 이 단계로 진행하세요!**
   - **save_report_document의 응답에서 "success": true를 확인한 후에만 종료하세요!**
   - **저장 없이 종료하면 안 됩니다!**
   - **Action**: respond

   **Final Answer 형식**:
   ```json
   {
     "status": "success",
     "response": "보고서 작성이 완료되었으며, DB에 성공적으로 저장되었습니다. 웹 프론트에서 최신 리포트를 확인해 주십시오.",
     "report_month": "[state의 report_month_str 값]"
   }"""
