"""
Agent Configuration Loader
agents.yaml 파일을 로드하고 파싱하여 Agent별 설정을 제공합니다.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from core.logging.logger import setup_logger

logger = setup_logger()

class AgentYamlConfig(BaseModel):
    """
    agents.yaml의 Agent별 설정 스키마
    """
    # 필수 설정
    name: str = Field(..., description="Agent 고유 이름")
    description: Optional[str] = Field(None, description="Agent 역할 설명")
    enabled: bool = Field(default=True, description="Agent 활성화 여부")

    # 실행 제어
    max_retries: int = Field(default=1, ge=0, description="실행 실패 시 재시도 횟수")
    timeout: int = Field(default=1000, gt=0, description="실행 타임아웃(초)")
    max_iterations: int = Field(default=10, ge=1, description="멀티턴 최대 반복 횟수")

    # 분류
    tags: list[str] = Field(default_factory=list, description="Agent 분류 태그")

    # LLM 설정 (Agent별 커스터마이징)
    llm_config: Optional[Dict[str, Any]] = Field(None, description="Agent별 LLM 설정")
    
class AgentConfigLoader:
    """
    agents.yaml 파일을 로드하고 관리하는 클래스
    사용 예:
        loader = AgentConfigLoader("configs/agents.yaml")
        config = loader.get_agent_config("user_create_agent")
    """

    _instance = None
    _configs: Dict[str, AgentYamlConfig] = {}

    def __new__(cls, yaml_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        
        # 인스턴스가 이미 존재하더라도 yaml_path가 제공되면 설정을 다시 로드합니다.
        # 이를 통해 그래프별로 다른 agent 설정을 적용할 수 있습니다.
        if yaml_path:
            cls._instance._load_configs(yaml_path)
            
        return cls._instance

    def _load_configs(self, yaml_path: str):
        """agents.yaml 파일 로드"""
        path = Path(yaml_path)
        
        if not path.exists():
            logger.error(f"❌ agents.yaml not found: {yaml_path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_configs = yaml.safe_load(f)
            
            agents_dict = raw_configs.get("agents", {})
            
            for agent_name, agent_data in agents_dict.items():
                # name 필드가 없으면 키를 name으로 사용
                if "name" not in agent_data:
                    agent_data["name"] = agent_name
                
                try:
                    config = AgentYamlConfig(**agent_data)
                    
                    # enabled=false인 Agent는 경고만 출력하고 저장은 함 (나중에 필터링)
                    if not config.enabled:
                        logger.warning(f"⚠️  Agent '{agent_name}' is disabled (enabled: false)")
                    
                    self._configs[agent_name] = config
                    logger.info(f"✅ Loaded config for: {agent_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to parse config for '{agent_name}': {e}")
            
            logger.info(f"📦 Total {len(self._configs)} agent configs loaded from {yaml_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load agents.yaml: {e}")

    @classmethod
    def get_agent_config(cls, agent_name: str) -> Optional[AgentYamlConfig]:
        """Agent 설정 조회"""
        return cls._configs.get(agent_name)

    @classmethod
    def get_enabled_agents(cls) -> list[str]:
        """활성화된(enabled=true) Agent 목록 반환"""
        return [
            name for name, config in cls._configs.items()
            if config.enabled
        ]

    @classmethod
    def get_agents_by_tag(cls, tag: str) -> list[str]:
        """특정 태그를 가진 Agent 목록"""
        return [
            name for name, config in cls._configs.items()
            if tag in config.tags and config.enabled
        ]

    @classmethod
    def list_all_configs(cls) -> Dict[str, AgentYamlConfig]:
        """모든 설정 반환 (디버깅용)"""
        return cls._configs.copy()
