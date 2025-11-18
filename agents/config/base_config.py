"""
Agent State & Config Management Module
Agent의 설정(Config)과 실행 상태(State)를 명확히 분리하여 관리
"""
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages  # ✅ 핵심 추가!


# ============================================================================
# 1. 설정 (Config) - Agent의 불변 설정 정보
# ============================================================================

class BaseAgentConfig(BaseModel):
    """
    모든 Agent의 기본 설정
    
    - Agent의 정적 설정 정보 (실행 중 변하지 않음)
    - Pydantic 기반 타입 검증 및 직렬화
    """
    # 필수 설정
    name: str = Field(..., description="Agent 고유 이름")
    
    # 선택적 설정
    description: Optional[str] = Field(None, description="Agent 역할 설명")
    
    # 실행 제어
    max_retries: int = Field(default=1, ge=0, description="실행 실패 시 재시도 횟수")
    timeout: int = Field(default=1000, gt=0, description="실행 타임아웃(초)")
    max_iterations: int = Field(default=10, ge=1, description="멀티턴 최대 반복 횟수")
    
    # Agent 관리
    enabled: bool = Field(default=True, description="Agent 활성화 여부")
    dependencies: List[str] = Field(default_factory=list, description="의존 Agent 목록")
    
    # 메타데이터
    tags: List[str] = Field(default_factory=list, description="Agent 분류 태그")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    
    # llm 관리
    model_name: str = Field(default="qwen3:8b", description="사용할 LLM 모델명")
    # temperature: 0.7 = Field(default=0.7, ge=0.0, le=1.0, description="LLM Teperature 설정")
    # top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="LLM Top-p 설정")
    # top_k: int = Field(default=40, ge=1, description="LLM Top-k 설정")
    # penalty: float = Field(default=1.0, description="LLM Penalty 설정")
    
# ============================================================================
# 2. 실행 상태 (State) - Agent의 동적 실행 정보
# ============================================================================

class ExecutionStatus(str, Enum):
    """Agent 실행 상태"""
    PENDING = "pending"           # 대기 중
    RUNNING = "running"           # 실행 중
    SUCCESS = "success"           # 성공
    FAILED = "failed"             # 실패
    TIMEOUT = "timeout"           # 타임아웃
    SKIPPED = "skipped"           # 건너뜀
    MAX_ITERATIONS = "max_iterations"  # 최대 반복 도달


class AgentState(TypedDict, total=False):
    """
    Agent 실행 상태 스키마 (LangGraph 호환)
    
    - TypedDict 기반으로 LangGraph State와 직접 호환
    - 실행 중 동적으로 변경되는 정보
    
    ⚠️ 핵심 변경:
    messages 필드에 Annotated[List[BaseMessage], add_messages] 사용
    → LangGraph가 자동으로 메시지를 누적(append)함
    """
    # === 메시지 처리 (Reducer 적용!) ===
    messages: Annotated[List[BaseMessage], add_messages]  # ✅ 핵심 수정!
    
    # === 실행 메타데이터 ===
    session_id: str                          # 세션 ID
    timestamp: datetime                      # 마지막 업데이트 시각
    current_agent: str                       # 현재 실행 중인 Agent
    execution_path: List[str]                # Agent 실행 경로
    
    # === 실행 제어 ===
    status: ExecutionStatus                  # 현재 실행 상태
    iteration: int                           # 현재 반복 횟수
    max_iterations: int                      # 최대 반복 횟수
    
    # === 데이터 ===
    input: Dict[str, Any]                    # 입력 데이터
    output: Dict[str, Any]                   # 출력 데이터
    last_result: Any                         # 마지막 실행 결과
    intermediate_results: Dict[str, Any]     # 중간 결과
    
    # === Tool 실행 추적 ===
    tool_calls: List[Dict[str, Any]]         # Tool 호출 이력
    tool_results: List[Dict[str, Any]]       # Tool 실행 결과
    
    # === 에러 처리 ===
    errors: List[Dict[str, Any]]             # 발생한 에러 목록
    warnings: List[str]                      # 경고 메시지
    
    # === 메타 정보 ===
    metadata: Dict[str, Any]                 # 추가 메타데이터


# ============================================================================
# 3. State 빌더 - 상태 생성 및 관리 헬퍼
# ============================================================================

class StateBuilder:
    """상태 생성 및 업데이트를 위한 헬퍼 클래스"""
    
    @staticmethod
    def create_initial_state(
        messages: List[BaseMessage],
        session_id: Optional[str] = None,
        max_iterations: int = 10,
        **kwargs
    ) -> AgentState:
        """
        초기 상태 생성
        
        Args:
            messages: 초기 메시지 리스트
            session_id: 세션 ID (없으면 자동 생성)
            max_iterations: 최대 반복 횟수
            **kwargs: 추가 상태 필드
            
        Returns:
            초기화된 상태
        """
        from uuid import uuid4
        
        state = AgentState(
            messages=messages,
            session_id=session_id or str(uuid4()),
            timestamp=datetime.now(),
            current_agent="",
            execution_path=[],
            status=ExecutionStatus.PENDING,
            iteration=0,
            max_iterations=max_iterations,
            input={},
            output={},
            last_result=None,
            intermediate_results={},
            tool_calls=[],
            tool_results=[],
            errors=[],
            warnings=[],
            metadata={}
        )
        
        # 추가 필드 업데이트
        state.update(kwargs)
        return state
    
    @staticmethod
    def update_agent_context(
        state: AgentState,
        agent_name: str,
        status: ExecutionStatus = ExecutionStatus.RUNNING
    ) -> AgentState:
        """
        Agent 실행 컨텍스트 업데이트
        
        Args:
            state: 현재 상태
            agent_name: 실행할 Agent 이름
            status: 실행 상태
            
        Returns:
            업데이트된 상태
        """
        state["current_agent"] = agent_name
        state["execution_path"].append(agent_name)
        state["status"] = status
        state["timestamp"] = datetime.now()
        return state
    
    @staticmethod
    def add_tool_call(
        state: AgentState,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any = None
    ) -> AgentState:
        """
        Tool 호출 기록 추가
        
        Args:
            state: 현재 상태
            tool_name: Tool 이름
            arguments: Tool 인자
            result: Tool 실행 결과
            
        Returns:
            업데이트된 상태
        """
        call_record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        }
        state["tool_calls"].append(call_record)
        
        if result is not None:
            result_record = {
                "tool_name": tool_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            state["tool_results"].append(result_record)
        
        return state
    
    @staticmethod
    def add_error(
        state: AgentState,
        error: Exception,
        agent_name: Optional[str] = None
    ) -> AgentState:
        """
        에러 추가
        
        Args:
            state: 현재 상태
            error: 발생한 예외
            agent_name: Agent 이름 (없으면 current_agent 사용)
            
        Returns:
            업데이트된 상태
        """
        state["errors"].append({
            "agent": agent_name or state.get("current_agent", "unknown"),
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = ExecutionStatus.FAILED
        return state
    
    @staticmethod
    def add_warning(state: AgentState, warning: str) -> AgentState:
        """경고 추가"""
        state["warnings"].append(f"[{datetime.now().isoformat()}] {warning}")
        return state
    
    @staticmethod
    def increment_iteration(state: AgentState) -> AgentState:
        """반복 횟수 증가"""
        state["iteration"] += 1
        
        # 최대 반복 도달 체크
        if state["iteration"] >= state["max_iterations"]:
            state["status"] = ExecutionStatus.MAX_ITERATIONS
        
        return state
    
    @staticmethod
    def is_max_iterations_reached(state: AgentState) -> bool:
        """최대 반복 횟수 도달 여부"""
        return state["iteration"] >= state["max_iterations"]
    
    @staticmethod
    def finalize_state(
        state: AgentState,
        status: ExecutionStatus = ExecutionStatus.SUCCESS
    ) -> AgentState:
        """
        상태 완료 처리
        
        Args:
            state: 현재 상태
            status: 최종 상태
            
        Returns:
            완료된 상태
        """
        state["status"] = status
        state["timestamp"] = datetime.now()
        return state


# ============================================================================
# 4. State Validator - 상태 검증
# ============================================================================

class StateValidator:
    """상태 검증 유틸리티"""
    
    @staticmethod
    def validate_required_fields(
        state: Dict[str, Any],
        required_fields: List[str]
    ) -> tuple[bool, List[str]]:
        """
        필수 필드 검증
        
        Returns:
            (검증 성공 여부, 누락된 필드 목록)
        """
        missing = [field for field in required_fields if field not in state]
        return len(missing) == 0, missing
    
    @staticmethod
    def validate_messages(state: AgentState) -> bool:
        """메시지 필드 검증"""
        if "messages" not in state:
            return False
        
        messages = state["messages"]
        if not isinstance(messages, list):
            return False
        
        return all(isinstance(msg, BaseMessage) for msg in messages)
    
    @staticmethod
    def validate_execution_state(state: AgentState) -> tuple[bool, Optional[str]]:
        """
        실행 상태 검증
        
        Returns:
            (검증 성공 여부, 에러 메시지)
        """
        # 반복 횟수 체크
        if state.get("iteration", 0) > state.get("max_iterations", 10):
            return False, "Iteration count exceeds max_iterations"
        
        # 상태 값 체크
        status = state.get("status")
        if status and status not in ExecutionStatus.__members__.values():
            return False, f"Invalid status: {status}"
        
        return True, None