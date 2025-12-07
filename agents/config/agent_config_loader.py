"""
Agent Configuration Loader
agents.yaml 파일을 로드하고 파싱하여 Agent별 설정을 제공합니다.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from contextvars import ContextVar
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
    Agent Configuration Loader
    
    각 그래프가 독립적인 인스턴스를 생성하여 사용합니다.
    싱글톤 패턴이 제거되어 여러 설정 파일을 동시에 로드할 수 있습니다.
    
    Context Variable을 사용하여 현재 그래프의 config loader에 접근할 수 있습니다.
    """
    
    # Context variable to store current loader instance
    _current_loader: ContextVar[Optional['AgentConfigLoader']] = ContextVar('current_loader', default=None)

    def __init__(self, yaml_path: str):
        """
        Args:
            yaml_path: agents.yaml 파일 경로
        """
        self._configs: Dict[str, AgentYamlConfig] = {}
        self._load_configs(yaml_path)
    
    @classmethod
    def set_current(cls, loader: 'AgentConfigLoader'):
        """현재 컨텍스트의 config loader 설정
        
        Args:
            loader: 설정할 AgentConfigLoader 인스턴스
        """
        cls._current_loader.set(loader)
        logger.debug(f"Set current AgentConfigLoader context")
    
    @classmethod
    def get_current(cls) -> Optional['AgentConfigLoader']:
        """현재 컨텍스트의 config loader 가져오기
        
        Returns:
            현재 컨텍스트의 AgentConfigLoader 인스턴스, 없으면 None
        """
        return cls._current_loader.get()

    def _load_configs(self, yaml_path: str):
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

    def get_agent_config(self, agent_name: str) -> Optional[AgentYamlConfig]:
        """Agent 설정 조회 (인스턴스 메서드)"""
        return self._configs.get(agent_name)
    
    @classmethod
    def get_agent_config_from_current(cls, agent_name: str) -> Optional[AgentYamlConfig]:
        """현재 컨텍스트에서 Agent 설정 조회 (클래스 메서드, 하위 호환성)
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            Agent 설정, 없으면 None
        """
        current = cls.get_current()
        if current:
            return current.get_agent_config(agent_name)
        # 컨텍스트가 없는 것은 정상 (agent discovery 단계)
        return None

    def get_enabled_agents(self) -> list[str]:
        """활성화된(enabled=true) Agent 목록 반환 (인스턴스 메서드)"""
        return [
            name for name, config in self._configs.items()
            if config.enabled
        ]
    
    @classmethod
    def get_enabled_agents_from_current(cls) -> list[str]:
        """현재 컨텍스트에서 활성화된 Agent 목록 반환 (클래스 메서드, 하위 호환성)"""
        current = cls.get_current()
        if current:
            return current.get_enabled_agents()
        logger.warning("No current AgentConfigLoader context set")
        return []

    def get_agents_by_tag(self, tag: str) -> list[str]:
        """특정 태그를 가진 Agent 목록 (인스턴스 메서드)"""
        return [
            name for name, config in self._configs.items()
            if tag in config.tags and config.enabled
        ]
    
    @classmethod
    def get_agents_by_tag_from_current(cls, tag: str) -> list[str]:
        """현재 컨텍스트에서 특정 태그를 가진 Agent 목록 (클래스 메서드, 하위 호환성)"""
        current = cls.get_current()
        if current:
            return current.get_agents_by_tag(tag)
        logger.warning("No current AgentConfigLoader context set")
        return []

    def list_all_configs(self) -> Dict[str, AgentYamlConfig]:
        """모든 설정 반환 (디버깅용, 인스턴스 메서드)"""
        return self._configs.copy()
    
    @classmethod
    def list_all_configs_from_current(cls) -> Dict[str, AgentYamlConfig]:
        """현재 컨텍스트에서 모든 설정 반환 (클래스 메서드, 하위 호환성)"""
        current = cls.get_current()
        if current:
            return current.list_all_configs()
        logger.warning("No current AgentConfigLoader context set")
        return {}
