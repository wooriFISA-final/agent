"""
Agent Config Loader
YAML 설정 파일을 로드하고 Agent 설정을 제공
"""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from agent.config.base_config import BaseAgentConfig

logger = logging.getLogger(__name__)


class AgentConfigLoader:
    """
    Agent 설정 로더 (싱글톤)
    
    YAML 파일에서 Agent 설정을 로드하고 캐싱
    """
    
    _instance: Optional['AgentConfigLoader'] = None
    _configs: Dict[str, Dict[str, Any]] = {}
    _loaded: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load_configs(cls, config_path: Optional[str] = None) -> Dict[str, Dict]:
        """
        설정 파일 로드
        
        Args:
            config_path: YAML 파일 경로 (기본값: agents/config/agent_configs.yaml)
            
        Returns:
            전체 설정 딕셔너리
        """
        if cls._loaded and not config_path:
            return cls._configs
        
        # 기본 경로
        if config_path is None:
            config_path = Path(__file__).parent / "agent_configs.yaml"
        else:
            config_path = Path(config_path)
        
        # 파일 존재 확인
        if not config_path.exists():
            logger.warning(f"⚠️ Config file not found: {config_path}")
            logger.info("Using default configurations")
            cls._configs = cls._get_default_configs()
            cls._loaded = True
            return cls._configs
        
        # YAML 로드
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
            
            cls._configs = configs or {}
            cls._loaded = True
            
            logger.info(f"✅ Loaded {len(cls._configs)} agent configurations from {config_path}")
            return cls._configs
            
        except Exception as e:
            logger.error(f"❌ Failed to load config file: {e}")
            cls._configs = cls._get_default_configs()
            cls._loaded = True
            return cls._configs
    
    @classmethod
    def get_config(cls, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        특정 Agent의 설정 가져오기
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            Agent 설정 딕셔너리 또는 None
        """
        if not cls._loaded:
            cls.load_configs()
        
        config = cls._configs.get(agent_name)
        
        if config is None:
            logger.debug(f"No config found for '{agent_name}', using default")
            return cls._configs.get("default")
        
        # default 설정과 병합
        default_config = cls._configs.get("default", {})
        merged_config = {**default_config, **config}
        
        return merged_config
    
    @classmethod
    def get_agent_config(cls, agent_name: str) -> BaseAgentConfig:
        """
        AgentConfig 객체 생성
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            AgentConfig 인스턴스
        """
        config_dict = cls.get_config(agent_name)
        
        if config_dict is None:
            # 최소 설정
            return BaseAgentConfig(name=agent_name)
        
        # AgentConfig에 맞는 필드만 추출
        agent_config_fields = {
            "name": config_dict.get("name", agent_name),
            "description": config_dict.get("description"),
            "timeout": config_dict.get("timeout", 30),
            "max_retries": config_dict.get("max_retries", 3),
            "enabled": config_dict.get("enabled", True),
            "dependencies": config_dict.get("dependencies", []),
        }
        
        return BaseAgentConfig(**agent_config_fields)
    
    @classmethod
    def get_custom_config(cls, agent_name: str, key: str, default: Any = None) -> Any:
        """
        커스텀 설정값 가져오기
        
        Args:
            agent_name: Agent 이름
            key: 설정 키
            default: 기본값
            
        Returns:
            설정값
        """
        config = cls.get_config(agent_name)
        
        if config is None:
            return default
        
        return config.get(key, default)
    
    @classmethod
    def get_llm_config(cls, agent_name: str) -> Dict[str, Any]:
        """
        LLM 관련 설정 가져오기
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            LLM 설정 딕셔너리
        """
        return cls.get_custom_config(agent_name, "llm", {})
    
    @classmethod
    def list_agents(cls) -> list[str]:
        """
        설정된 모든 Agent 이름 목록
        
        Returns:
            Agent 이름 리스트
        """
        if not cls._loaded:
            cls.load_configs()
        
        # 'default' 제외
        return [name for name in cls._configs.keys() if name != "default"]
    
    @classmethod
    def is_enabled(cls, agent_name: str) -> bool:
        """
        Agent 활성화 여부 확인
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            활성화 여부
        """
        config = cls.get_config(agent_name)
        if config is None:
            return True  # 설정 없으면 기본적으로 활성화
        
        return config.get("enabled", True)
    
    @classmethod
    def get_priority(cls, agent_name: str) -> int:
        """
        Agent 우선순위 가져오기
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            우선순위 (낮을수록 먼저 실행)
        """
        config = cls.get_config(agent_name)
        if config is None:
            return 50  # 기본 우선순위
        
        return config.get("priority", 50)
    
    @classmethod
    def get_dependencies(cls, agent_name: str) -> list[str]:
        """
        Agent 의존성 목록
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            의존성 Agent 이름 리스트
        """
        config = cls.get_config(agent_name)
        if config is None:
            return []
        
        return config.get("dependencies", [])
    
    @classmethod
    def reload(cls):
        """설정 재로드"""
        cls._loaded = False
        cls._configs = {}
        logger.info("🔄 Config reloaded")
    
    @classmethod
    def _get_default_configs(cls) -> Dict[str, Dict]:
        """기본 설정 반환"""
        return {
            "default": {
                "timeout": 30,
                "max_retries": 3,
                "enabled": True,
                "dependencies": [],
                "priority": 50
            }
        }
    
    @classmethod
    def validate_config(cls, agent_name: str) -> tuple[bool, list[str]]:
        """
        설정 검증
        
        Args:
            agent_name: Agent 이름
            
        Returns:
            (검증 성공 여부, 에러 메시지 리스트)
        """
        config = cls.get_config(agent_name)
        
        if config is None:
            return True, []  # 설정 없어도 OK (default 사용)
        
        errors = []
        
        # 필수 필드 확인
        if "name" not in config:
            errors.append("'name' field is required")
        
        # 타입 검증
        if "timeout" in config and not isinstance(config["timeout"], int):
            errors.append("'timeout' must be an integer")
        
        if "max_retries" in config and not isinstance(config["max_retries"], int):
            errors.append("'max_retries' must be an integer")
        
        if "enabled" in config and not isinstance(config["enabled"], bool):
            errors.append("'enabled' must be a boolean")
        
        if "dependencies" in config and not isinstance(config["dependencies"], list):
            errors.append("'dependencies' must be a list")
        
        # 값 범위 검증
        if "timeout" in config and config["timeout"] <= 0:
            errors.append("'timeout' must be positive")
        
        if "max_retries" in config and config["max_retries"] < 0:
            errors.append("'max_retries' cannot be negative")
        
        return len(errors) == 0, errors
    
    @classmethod
    def export_config(cls, agent_name: str, output_path: str):
        """
        특정 Agent 설정을 파일로 내보내기
        
        Args:
            agent_name: Agent 이름
            output_path: 출력 파일 경로
        """
        config = cls.get_config(agent_name)
        
        if config is None:
            logger.error(f"No config found for '{agent_name}'")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump({agent_name: config}, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Config exported to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export config: {e}")
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """
        전체 설정 요약
        
        Returns:
            요약 정보 딕셔너리
        """
        if not cls._loaded:
            cls.load_configs()
        
        enabled_count = sum(
            1 for config in cls._configs.values() 
            if config.get("enabled", True)
        )
        
        return {
            "total_agents": len(cls._configs) - 1,  # 'default' 제외
            "enabled_agents": enabled_count,
            "disabled_agents": len(cls._configs) - 1 - enabled_count,
            "loaded": cls._loaded,
            "agents": cls.list_agents()
        }


# 편의 함수들
def load_agent_config(agent_name: str) -> BaseAgentConfig:
    """Agent 설정 로드 (간편 함수)"""
    return AgentConfigLoader.get_agent_config(agent_name)


def get_llm_settings(agent_name: str) -> Dict[str, Any]:
    """LLM 설정 가져오기 (간편 함수)"""
    return AgentConfigLoader.get_llm_config(agent_name)


def is_agent_enabled(agent_name: str) -> bool:
    """Agent 활성화 여부 (간편 함수)"""
    return AgentConfigLoader.is_enabled(agent_name)