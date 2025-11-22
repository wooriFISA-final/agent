"""
Agent Registry Module

Agent 자동 등록 및 관리를 담당하는 레지스트리
"""
from typing import Dict, Type, Optional, List
from agent.base.agent_base import AgentBase
import importlib
import inspect
import pkgutil
import logging

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Agent 자동 등록 및 관리"""
    
    _instance = None
    _agents: Dict[str, Type[AgentBase]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # ------------------------------------
    # 1️⃣ 데코레이터 기반 등록
    # ------------------------------------
    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        데코레이터로 Agent 자동 등록
        사용 예:
            @AgentRegistry.register("research")
            class ResearchAgent(AgentBase):
                ...
        """
        def decorator(agent_class: Type[AgentBase]):
            agent_name = name or agent_class.__name__
            
            # ✅ enabled 체크 추가
            from agent.config.agent_config_loader import AgentConfigLoader
            yaml_config = AgentConfigLoader.get_agent_config(agent_name)
            
            if yaml_config and not yaml_config.enabled:
                logger.warning(
                    f"⚠️  Agent '{agent_name}' is disabled in agents.yaml. "
                    f"Skipping registration."
                )
                return agent_class
            
            if agent_name in cls._agents:
                logger.warning(f"⚠️ Agent '{agent_name}' 이미 등록되어 있음. 기존 항목을 덮어씁니다.")
            
            cls._agents[agent_name] = agent_class
            logger.info(f"✅ Agent 등록됨: {agent_name}")
            return agent_class
        return decorator
    
    # ------------------------------------
    # 2️⃣ 조회 및 목록 기능
    # ------------------------------------
    @classmethod
    def get(cls, name: str) -> Type[AgentBase]:
        """이름으로 Agent 클래스 가져오기"""
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not found in registry")
        return cls._agents[name]
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """등록된 모든 Agent 목록"""
        return list(cls._agents.keys())
    
    @classmethod
    def list_enabled_agents(cls) -> List[str]:
        """활성화된(enabled=true) Agent 목록만 반환"""
        from agent.config.agent_config_loader import AgentConfigLoader
        
        enabled = []
        for agent_name in cls._agents.keys():
            yaml_config = AgentConfigLoader.get_agent_config(agent_name)
            if not yaml_config or yaml_config.enabled:
                # yaml_config가 없거나 enabled=true인 경우
                enabled.append(agent_name)
        
        return enabled
    
    @classmethod
    def get_agents_by_tag(cls, tag: str) -> List[str]:
        """특정 태그를 가진 활성화된 Agent 목록"""
        from agent.config.agent_config_loader import AgentConfigLoader
        
        result = []
        for agent_name in cls._agents.keys():
            yaml_config = AgentConfigLoader.get_agent_config(agent_name)
            if yaml_config and yaml_config.enabled and tag in yaml_config.tags:
                result.append(agent_name)
        
        return result
    
    # ------------------------------------
    # 3️⃣ 패키지 자동 탐색 기능 개선
    # ------------------------------------
    @classmethod
    def auto_discover(cls, module_path: str = "agent.implementations"):
        """
        지정된 패키지 내 모든 서브모듈에서 Agent 클래스 자동 등록
        ex) agent/implementations/research_agent.py 등
        """
        try:
            package = importlib.import_module(module_path)
        except ModuleNotFoundError:
            logger.error(f"❌ 패키지 '{module_path}'를 찾을 수 없습니다.")
            return

        # ✅ agents.yaml 로드 확인
        from agent.config.agent_config_loader import AgentConfigLoader
        
        # 서브모듈 재귀 탐색
        for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, AgentBase) and obj is not AgentBase:
                        agent_name = getattr(obj, "__agent_name__", name)
                        
                        # ✅ enabled 체크
                        yaml_config = AgentConfigLoader.get_agent_config(agent_name)
                        
                        if yaml_config and not yaml_config.enabled:
                            logger.warning(
                                f"⚠️  Skipping disabled agent: {agent_name} "
                                f"(enabled: false in agents.yaml)"
                            )
                            continue
                        
                        cls._agents[agent_name] = obj
                        logger.info(f"🔍 자동 등록됨: {agent_name} ({module_name})")
            except Exception as e:
                logger.warning(f"⚠️ {module_name} 모듈 로드 실패: {e}")