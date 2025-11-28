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
    
    # Langgraph 실행 관련
    MAX_GRAPH_ITERATIONS: int = Field(
        20,  # 기본값은 원하는 값으로 조정
        ge=1,
        description="LangGraph 전체 최대 반복 횟수 (chat.py에서 사용)",
    )

    # MCP (Mission Control Protocol)
    MCP_URL: HttpUrl = Field(..., description="URL for the MCP server")
    MCP_CONNECTION_RETRIES: int = Field(..., description="MCP 연결 재시도 횟수")
    MCP_CONNECTION_TIMEOUT: int = Field(..., description="Timeout for MCP 연결 (초)")

    # AWS Bedrock Configuration
    AWS_REGION: str = Field(..., description="AWS 리전 (예: us-east-1)")
    AWS_BEARER_TOKEN_BEDROCK: Optional[str] = Field(None, description="AWS Bedrock 인증 토큰")
    BEDROCK_MODEL_ID: str = Field(..., description="Bedrock 모델 ID (예: openai.gpt-oss-20b-1:0)")
    
    # LLM Parameters (Bedrock 호환)
    LLM_TEMPERATURE: float = Field(..., ge=0.0, le=2.0, description="LLM temperature setting")
    LLM_TOP_P: float = Field(..., ge=0.0, le=1.0, description="LLM top-p sampling value")
    LLM_MAX_TOKENS: int = Field(default=120000, ge=1, description="최대 토큰 수 (maxTokens)")
    LLM_TIMEOUT: int = Field(..., ge=1, description="LLM 요청 타임아웃 (초)")
    LLM_STREAM: bool = Field(default=False, description="스트리밍 응답 활성화 여부 (현재 미지원)")
    
    # Agent Registry
    AGENTS_MODULE_PATH: str = Field(..., description="Agent 구현 모듈 경로 (예: agents.implementations)")
    
settings = AgentSystemConfig()
