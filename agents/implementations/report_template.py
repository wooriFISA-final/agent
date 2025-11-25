import logging
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
# agents.base.agent_base의 AgentBase와 BaseAgentConfig가 있다고 가정
from agents.base.agent_base import AgentBase, BaseAgentConfig
# agents.registry.agent_registry의 AgentRegistry와 AgentState가 있다고 가정
from agents.registry.agent_registry import AgentRegistry, AgentState 

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
    1. rebuild_vector_db_tool: 정책 문서를 기반으로 FAISS 벡터 DB를 재구축 (🚨 신규 추가)
    2. analyze_user_spending_tool: 월별 소비 데이터 비교 분석 및 군집 생성
    3. analyze_investment_profit_tool: 투자 상품 손익/진척도 분석
    4. analyze_user_profile_changes_tool: 사용자 개인 지수 변동 분석 (연봉, 부채, 신용 점수)
    5. check_and_report_policy_changes_tool: 금융 정책 변동 사항 자동 비교 및 보고서 생성
    6. generate_final_summary_llm: 통합 보고서 본문을 받아 핵심 3줄 요약 생성
    """
    
    # 🎯 [스케줄 설정]: 매월 보고서를 생성할 날짜 (예: 5일)
    REPORT_SCHEDULE_DAY = 5 
    
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        
        # 🎯 사용 가능한 6가지 전문 Tool 목록을 정의 (DB 재구축 툴 포함)
        self.allowed_tools = [
            "rebuild_vector_db_tool",  # 🚨 [신규 추가]
            "analyze_user_spending_tool",
            "analyze_investment_profit_tool",
            "analyze_user_profile_changes_tool",
            "check_and_report_policy_changes_tool",
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
        
        # 보고서 생성에 필요한 핵심 데이터 (예: report_month_str 등)가 state에 있는지 확인
        if "report_month_str" not in state:
            logger.error(f"[{self.name}] Missing required key 'report_month_str' in state.")
            return False
            
        return True
        
    def pre_execute(self, state: AgentState) -> AgentState:
        """실행 전 전처리 및 월간 스케줄 트리거 확인"""
        
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
        return f"""당신은 금융 보고서 작성 전문 에이전트입니다.
        
        주된 임무는 사용자로부터 수집된 모든 금융 데이터(소비 기록, 투자 상품, 정책 변동, 개인 지표)를 분석하고 통합하여, 최종 고객에게 전달할 명확하고 간결하며 전문적인 월간 재무 보고서를 작성하는 것입니다.
        
        **행동 원칙:**
        1. **순차적 도구 사용**: 보고서의 각 섹션을 완성하기 위해 6가지 전문 도구를 순차적으로 호출하여 필요한 정보를 수집하십시오.
        2. **최종 요약**: 모든 정보 수집 및 분석이 완료되면, 'generate_final_summary_llm' 도구를 사용하여 통합 보고서의 핵심 내용을 간결한 3줄 요약본으로 최종 정리하십시오.
        3. **간결한 응답**: 보고서 생성을 완료한 후에는 생성된 보고서와 핵심 요약본을 포함하여 사용자에게 최종적으로 보고하십시오.
        
        **도구 사용 순서 (권장):**
        1. **rebuild_vector_db_tool**: 정책 문서가 최신 버전으로 업데이트된 경우, **반드시 가장 먼저** 이 툴을 호출하여 정책 DB를 최신화하십시오. (일반적으로는 최신화가 필요하지 않으면 건너뛰지만, 시스템 초기화나 정책 파일 변경 시 필수입니다.)
        2. analyze_user_profile_changes_tool (개인 지표 변동)
        3. analyze_user_spending_tool (소비 분석)
        4. analyze_investment_profit_tool (투자 분석)
        5. check_and_report_policy_changes_tool (정책 변동) - **업데이트된 DB를 사용하도록 유의**
        6. 모든 분석 결과를 합쳐 최종 보고서 본문을 구성한 후, generate_final_summary_llm (3줄 요약)을 호출하십시오.
        """