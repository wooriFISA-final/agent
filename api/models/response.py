"""
Response 모델

API 응답에 사용되는 Pydantic 모델을 정의합니다.
"""
from pydantic import BaseModel
from typing import Optional


class ChatResponse(BaseModel):
    """채팅 응답 모델
    
    Attributes:
        response: AI 응답 메시지
        status: 응답 상태 (기본값: "success")
        metadata: 추가 메타데이터
    """
    response: str
    status: str = "success"
    metadata: dict = {}


class HealthResponse(BaseModel):
    """헬스체크 응답 모델
    
    Attributes:
        status: 시스템 상태
        mcp_connected: MCP 연결 상태
        available_tools: 사용 가능한 도구 수
        registered_agents: 등록된 에이전트 목록
        error: 에러 메시지 (선택적)
    """
    status: str
    mcp_connected: bool
    available_tools: int
    registered_agents: list
    error: Optional[str] = None
