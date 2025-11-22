"""
Request 모델

API 요청에 사용되는 Pydantic 모델을 정의합니다.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """채팅 요청 모델
    
    Attributes:
        message: 사용자 메시지
        session_id: 세션 ID (기본값: "default-session")
    """
    message: str
    session_id: str = "default-session"
