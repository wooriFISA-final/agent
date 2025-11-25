from typing import Optional,Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl
from pathlib import Path

class AgentSystemConfig(BaseSettings):
    """
    Assistant Agent System Configuration
    .env 파일과 환경 변수에서 설정을 로드합니다.
    중요 : 모든 필수 필드는 .env 파일에 정의되어야 합니다.
    """
    model_config = SettingsConfigDict(
        env_prefix='AGENT_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Environment
    ENVIRONMENT: str = Field(..., description="운영 환경 (development, staging, production)")
    DEBUG: bool = Field(..., description="디버그 모드 활성화 여부")

    # FastAPI Server
    API_HOST: str = Field(..., description="FastAPI 서버 호스트 주소")
    API_PORT: int = Field(..., description="FastAPI 서버 포트 번호")
    API_VERSION: str = Field(..., description="API 버전")

    # Logging
    LOG_LEVEL: str = Field(..., description="Logging level")
    LOG_FILE: Optional[str] = Field(None, description="Log 파일 경로")

    # MCP (Mission Control Protocol)
    MCP_URL: HttpUrl = Field(..., description="URL for the MCP server")
    MCP_CONNECTION_RETRIES: int = Field(..., description="MCP 연결 재시도 횟수")
    MCP_CONNECTION_TIMEOUT: int = Field(..., description="Timeout for MCP 연결 (초)")

    # LLM Provider
    LLM_PROVIDER: str = Field(..., description="LLM 제공자 이름(예: openai, ollama 등)") # 제거 예정
    LLM_MODEL: str = Field(..., description="기본 LLM 모델 이름")
    LLM_API_BASE_URL: Optional[HttpUrl] = Field(None, description="LLM API URL")
    LLM_TEMPERATURE: float = Field(..., ge=0.0, le=2.0, description="LLM temperature setting")
    LLM_TIMEOUT: int = Field(..., ge=1, description="LLM 요청 타임아웃 (초)")
    LLM_TOP_P: float = Field(..., ge=0.0, le=1.0, description="LLM top-p sampling value.")
    LLM_TOP_K: int = Field(..., ge=0, description="Default LLM top-k sampling value.")
    LLM_NUM_CTX: int = Field(..., ge=4096, description="LLM context length (토큰 수).")
    # 새로 추가할 필드
    LLM_STREAM: bool = Field(default=False, description="스트리밍 응답 활성화 여부")
    LLM_FORMAT: Literal["", "json"] = Field(default="", description='응답 형식 지정 ("", "json")')
    
    # Agent Registry
    AGENTS_MODULE_PATH: str = Field(..., description="Agent 구현 모듈 경로 (예: agents.implementations)")
    
settings = AgentSystemConfig()
