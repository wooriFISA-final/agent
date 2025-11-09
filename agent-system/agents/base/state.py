"""
State Management Module
공통 상태 정의 및 타입 안전성 보장
"""
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class AgentState(TypedDict, total=False):
    """
    기본 Agent 상태 스키마
    
    LangGraph에서 사용하는 TypedDict 기반 상태
    모든 Agent가 공통으로 사용하는 필드
    """
    # 메타데이터
    session_id: str
    timestamp: datetime
    current_agent: str
    execution_path: List[str]
    
    # 실행 정보
    status: ExecutionStatus
    iteration: int
    max_iterations: int
    
    # 데이터
    input: Dict[str, Any]
    output: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    
    # 에러 처리
    errors: List[Dict[str, Any]]
    warnings: List[str]
    
    # 메타 정보
    metadata: Dict[str, Any]


class StateBuilder:
    """상태 빌더 헬퍼 클래스"""
    
    @staticmethod
    def create_initial_state(
        session_id: str,
        input_data: Dict[str, Any],
        max_iterations: int = 10
    ) -> AgentState:
        """
        초기 상태 생성
        
        Args:
            session_id: 세션 ID
            input_data: 입력 데이터
            max_iterations: 최대 반복 횟수
            
        Returns:
            초기화된 상태
        """
        return AgentState(
            session_id=session_id,
            timestamp=datetime.now(),
            current_agent="",
            execution_path=[],
            status=ExecutionStatus.PENDING,
            iteration=0,
            max_iterations=max_iterations,
            input=input_data,
            output={},
            intermediate_results={},
            errors=[],
            warnings=[],
            metadata={}
        )
    
    @staticmethod
    def update_agent_context(
        state: AgentState,
        agent_name: str
    ) -> AgentState:
        """
        Agent 컨텍스트 업데이트
        
        Args:
            state: 현재 상태
            agent_name: 실행 중인 Agent 이름
            
        Returns:
            업데이트된 상태
        """
        state["current_agent"] = agent_name
        state["execution_path"].append(agent_name)
        state["timestamp"] = datetime.now()
        return state
    
    @staticmethod
    def add_error(
        state: AgentState,
        error: Exception,
        agent_name: str
    ) -> AgentState:
        """
        에러 추가
        
        Args:
            state: 현재 상태
            error: 발생한 예외
            agent_name: Agent 이름
            
        Returns:
            업데이트된 상태
        """
        state["errors"].append({
            "agent": agent_name,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = ExecutionStatus.FAILED
        return state
    
    @staticmethod
    def add_warning(
        state: AgentState,
        warning: str
    ) -> AgentState:
        """
        경고 추가
        
        Args:
            state: 현재 상태
            warning: 경고 메시지
            
        Returns:
            업데이트된 상태
        """
        state["warnings"].append(warning)
        return state
    
    @staticmethod
    def increment_iteration(state: AgentState) -> AgentState:
        """
        반복 횟수 증가
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        state["iteration"] += 1
        return state
    
    @staticmethod
    def is_max_iterations_reached(state: AgentState) -> bool:
        """
        최대 반복 횟수 도달 여부
        
        Args:
            state: 현재 상태
            
        Returns:
            도달 여부
        """
        return state["iteration"] >= state["max_iterations"]


# 사용자 정의 상태 예시
class ResearchState(AgentState):
    """리서치 워크플로우용 상태"""
    query: str
    search_results: List[Dict[str, Any]]
    analyzed_data: Dict[str, Any]
    final_report: str
    confidence_score: float


class DataProcessingState(AgentState):
    """데이터 처리 워크플로우용 상태"""
    raw_data: List[Dict[str, Any]]
    cleaned_data: List[Dict[str, Any]]
    transformed_data: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    processing_steps: List[str]


class ConversationState(AgentState):
    """대화형 워크플로우용 상태"""
    user_message: str
    conversation_history: List[Dict[str, str]]
    intent: str
    entities: Dict[str, Any]
    response: str
    context: Dict[str, Any]


# State Annotation 헬퍼
def create_state_annotation(base_state: type) -> type:
    """
    LangGraph용 State Annotation 생성
    
    Args:
        base_state: 베이스 상태 클래스
        
    Returns:
        Annotated 상태 타입
    """
    # 여기서는 단순 반환, 실제로는 reducer 등을 추가할 수 있음
    return base_state


# State Validator
class StateValidator:
    """상태 검증 유틸리티"""
    
    @staticmethod
    def validate_required_fields(
        state: Dict[str, Any],
        required_fields: List[str]
    ) -> bool:
        """
        필수 필드 존재 확인
        
        Args:
            state: 검증할 상태
            required_fields: 필수 필드 목록
            
        Returns:
            검증 성공 여부
        """
        return all(field in state for field in required_fields)
    
    @staticmethod
    def validate_field_types(
        state: Dict[str, Any],
        field_types: Dict[str, type]
    ) -> bool:
        """
        필드 타입 검증
        
        Args:
            state: 검증할 상태
            field_types: {필드명: 타입} 딕셔너리
            
        Returns:
            검증 성공 여부
        """
        for field, expected_type in field_types.items():
            if field in state:
                if not isinstance(state[field], expected_type):
                    return False
        return True
    
    @staticmethod
    def sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        상태 정제 (None 제거, 기본값 설정 등)
        
        Args:
            state: 정제할 상태
            
        Returns:
            정제된 상태
        """
        sanitized = {}
        for key, value in state.items():
            if value is not None:
                sanitized[key] = value
        return sanitized