"""
System Configuration Module
전역 설정 관리
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import os
from pathlib import Path


class Environment(str, Enum):
    """실행 환경"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AgentSystemConfig(BaseModel):
    """Agent 시스템 전역 설정"""
    
    # 환경 설정
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="실행 환경"
    )
    debug: bool = Field(default=True, description="디버그 모드")
    
    # 로깅 설정
    log_level: LogLevel = Field(default=LogLevel.INFO, description="로그 레벨")
    log_file: Optional[str] = Field(default="logs/agent_system.log", description="로그 파일 경로")
    log_rotation: bool = Field(default=True, description="로그 로테이션 활성화")
    log_max_size: int = Field(default=10 * 1024 * 1024, description="로그 파일 최대 크기 (bytes)")
    
    # Agent 기본 설정
    default_agent_timeout: int = Field(default=30, ge=1, description="기본 Agent 타임아웃 (초)")
    default_agent_retries: int = Field(default=3, ge=0, description="기본 재시도 횟수")
    max_concurrent_agents: int = Field(default=10, ge=1, description="최대 동시 실행 Agent 수")
    
    # Graph 설정
    max_graph_iterations: int = Field(default=100, ge=1, description="그래프 최대 반복 횟수")
    graph_execution_timeout: int = Field(default=300, ge=1, description="그래프 실행 타임아웃 (초)")
    
    # 성능 설정
    enable_caching: bool = Field(default=True, description="캐싱 활성화")
    cache_ttl: int = Field(default=3600, ge=0, description="캐시 TTL (초)")
    enable_metrics: bool = Field(default=True, description="메트릭 수집 활성화")
    
    # 보안 설정
    enable_rate_limiting: bool = Field(default=True, description="Rate limiting 활성화")
    rate_limit_requests: int = Field(default=100, description="시간당 최대 요청 수")
    enable_authentication: bool = Field(default=False, description="인증 활성화")
    
    # MCP 설정
    mcp_enabled: bool = Field(default=True, description="MCP 프로토콜 활성화")
    mcp_server_host: str = Field(default="localhost", description="MCP 서버 호스트")
    mcp_server_port: int = Field(default=8000, ge=1, le=65535, description="MCP 서버 포트")
    
    # 데이터베이스 설정 (선택적)
    database_url: Optional[str] = Field(default=None, description="데이터베이스 URL")
    enable_persistence: bool = Field(default=False, description="영속성 활성화")
    
    # 외부 서비스 설정
    llm_provider: str = Field(default="anthropic", description="LLM 제공자")
    llm_model: str = Field(default="claude-3-sonnet-20240229", description="LLM 모델")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API 키")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM Temperature")
    llm_max_tokens: int = Field(default=4096, ge=1, description="LLM 최대 토큰")
    
    # 모니터링 설정
    enable_tracing: bool = Field(default=False, description="분산 추적 활성화")
    tracing_endpoint: Optional[str] = Field(default=None, description="추적 엔드포인트")
    enable_profiling: bool = Field(default=False, description="프로파일링 활성화")
    
    # 경로 설정
    agents_module_path: str = Field(default="agents.implementations", description="Agent 모듈 경로")
    tools_module_path: str = Field(default="tools.implementations", description="Tool 모듈 경로")
    workflows_dir: Path = Field(default=Path("workflows"), description="워크플로우 디렉토리")
    
    class Config:
        env_prefix = "AGENT_"
        case_sensitive = False


class Settings:
    """
    설정 싱글톤
    
    환경 변수 및 설정 파일에서 설정을 로드하고 관리
    """
    
    _instance: Optional['Settings'] = None
    _config: Optional[AgentSystemConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config(cls) -> AgentSystemConfig:
        """
        설정 가져오기
        
        Returns:
            AgentSystemConfig 인스턴스
        """
        if cls._config is None:
            cls._config = cls._load_config()
        return cls._config
    
    @classmethod
    def _load_config(cls) -> AgentSystemConfig:
        """
        설정 로드
        
        우선순위:
        1. 환경 변수
        2. 설정 파일 (config.yaml)
        3. 기본값
        """
        config_dict = {}
        
        # 환경 변수에서 로드
        for field in AgentSystemConfig.__fields__.keys():
            env_key = f"AGENT_{field.upper()}"
            if env_key in os.environ:
                config_dict[field] = os.environ[env_key]
        
        # 설정 파일에서 로드 (선택적)
        config_file = Path("config.yaml")
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                config_dict.update(file_config)
        
        # Pydantic 모델 생성
        return AgentSystemConfig(**config_dict)
    
    @classmethod
    def reload_config(cls):
        """설정 재로드"""
        cls._config = cls._load_config()
    
    @classmethod
    def update_config(cls, **kwargs):
        """
        설정 업데이트
        
        Args:
            **kwargs: 업데이트할 설정
        """
        if cls._config is None:
            cls._config = cls._load_config()
        
        # 기존 설정 dict로 변환
        config_dict = cls._config.dict()
        config_dict.update(kwargs)
        
        # 새 설정 생성
        cls._config = AgentSystemConfig(**config_dict)
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        특정 설정값 가져오기
        
        Args:
            key: 설정 키
            default: 기본값
            
        Returns:
            설정값
        """
        config = cls.get_config()
        return getattr(config, key, default)
    
    @classmethod
    def is_development(cls) -> bool:
        """개발 환경 여부"""
        return cls.get("environment") == Environment.DEVELOPMENT
    
    @classmethod
    def is_production(cls) -> bool:
        """프로덕션 환경 여부"""
        return cls.get("environment") == Environment.PRODUCTION
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return cls.get_config().dict()
    
    @classmethod
    def to_yaml(cls, output_path: str):
        """설정을 YAML 파일로 저장"""
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(cls.to_dict(), f, default_flow_style=False)


# 전역 설정 인스턴스
settings = Settings.get_config()


# 환경별 설정 프리셋
DEVELOPMENT_CONFIG = {
    "environment": Environment.DEVELOPMENT,
    "debug": True,
    "log_level": LogLevel.DEBUG,
    "enable_caching": False,
    "enable_metrics": True,
    "enable_rate_limiting": False,
}

PRODUCTION_CONFIG = {
    "environment": Environment.PRODUCTION,
    "debug": False,
    "log_level": LogLevel.INFO,
    "enable_caching": True,
    "enable_metrics": True,
    "enable_rate_limiting": True,
    "enable_authentication": True,
}

TEST_CONFIG = {
    "environment": Environment.TEST,
    "debug": True,
    "log_level": LogLevel.WARNING,
    "enable_caching": False,
    "enable_metrics": False,
    "enable_rate_limiting": False,
}


def configure_for_environment(env: Environment):
    """
    환경에 맞게 설정
    
    Args:
        env: 환경
    """
    if env == Environment.DEVELOPMENT:
        Settings.update_config(**DEVELOPMENT_CONFIG)
    elif env == Environment.PRODUCTION:
        Settings.update_config(**PRODUCTION_CONFIG)
    elif env == Environment.TEST:
        Settings.update_config(**TEST_CONFIG)