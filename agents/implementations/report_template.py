import logging
from typing import Dict, Any, List
# from langchain_core.messages import HumanMessage
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
        
        # 🎯 사용 가능한 5가지 전문 Tool 목록을 정의
        self.allowed_tools = [
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

**🚨 중요: 도구는 데이터만 반환합니다. 당신이 직접 LLM을 사용하여 보고서 텍스트를 생성해야 합니다.**

**작업 흐름:**
1. 각 분석 도구를 순차적으로 호출하여 데이터 수집
2. 수집된 데이터를 바탕으로 각 섹션의 보고서 텍스트 생성
3. 모든 섹션을 통합하여 최종 보고서 작성
4. 핵심 내용 3줄 요약 생성

**도구 사용 순서 및 데이터 처리 방법:**

1. **analyze_user_profile_changes_tool** (개인 지표 변동):
   - 반환 데이터: `change_raw_changes` (변동 내역 리스트), `is_first_report`
   - 생성할 내용:
     * 변동이 없으면: "직전 보고서 대비 주요 개인 지표에 큰 변동 사항이 없습니다."
     * 변동이 있으면: 4줄 이내로 변동 사항 요약 및 재정 조언
     * 첫 보고서인 경우: 현재 상태를 기준으로 분석

2. **analyze_user_spending_tool** (소비 분석):
   - 반환 데이터: `consume_analysis_summary` (총 지출, 변화율, Top 5 카테고리, 금액)
   - 생성할 내용:
     * **군집 별명**: Top 5 소비 카테고리와 재정 건전성을 고려하여 생성 (예: "균형잡힌 소비형", "투자 중심형", "문화생활 애호가형")
     * **소비 분석 보고서**: 4-5줄로 총 지출 변화, 주요 카테고리, 고정비/비고정비 해석, 저축/투자 조언 포함
   - 프롬프트 예시:
     ```
     총 지출: {latest_total_spend}원 (전월 대비 {change_rate}% 변동)
     주요 5대 소비 영역: {top_5_categories} (각각 {top_5_amounts}원)
     
     위 데이터를 바탕으로:
     1. 소비 패턴에 맞는 군집 별명 생성
     2. 지출 변화 해석 및 주요 카테고리 설명
     3. 재정 조언 (4-5줄)
     ```

3. **analyze_investment_profit_tool** (투자 분석):
   - 반환 데이터: `total_principal`, `total_valuation`, `net_profit`, `profit_rate`, `products_count`
   - 생성할 내용:
     * 투자 원금 대비 수익률 평가
     * 투자 진척도 분석 및 다음 단계 전략 조언 (5줄 이내)
   - 프롬프트 예시:
     ```
     총 투자 원금: {total_principal:,}원
     현재 평가액: {total_valuation:,}원
     순손익: {net_profit:+,}원
     수익률: {profit_rate}%
     보유 상품 수: {products_count}개
     
     위 데이터를 바탕으로 투자 진척도를 평가하고 다음 단계 전략 조언 (5줄 이내)
     ```

4. **check_and_report_policy_changes_tool** (정책 변동):
   - 반환 데이터: `policy_changes` (변동 리스트, 각 항목에 `effective_date`와 `policy_text` 포함)
   - 생성할 내용:
     * 변동이 없으면: 도구가 반환한 `message` 사용
     * 변동이 있으면: 간결한 단일 단락 분석 보고서 (5줄 이내)
     * 반드시 '📌 [시행일: {earliest_date}]'로 시작
     * 변동 사항의 핵심 내용과 고객에게 미치는 영향 포함
   - 프롬프트 예시:
     ```
     정책 변동 사항:
     {각 policy_change의 effective_date와 policy_text를 나열}
     
     위 정책 변동을 바탕으로:
     1. '📌 [시행일: {earliest_date}]'로 시작
     2. 핵심 내용과 고객 영향을 5줄 이내로 요약
     3. Markdown 서식 기호 사용 금지 (순수 평문)
     ```

5. **최종 단계**:
   - 모든 섹션을 통합하여 완전한 보고서 작성
   - 통합 보고서에서 가장 핵심적인 3가지 사항을 뽑아 3줄 요약 생성
   - **generate_final_summary_llm 도구는 호출하지 마세요** (deprecated)

**출력 형식:**
- 각 섹션은 명확히 구분
- 간결하고 정중한 한국어 사용
- 전문적이면서도 고객이 이해하기 쉬운 표현
- 불필요한 Markdown 서식 최소화 (보고서 본문은 평문 위주)
"""