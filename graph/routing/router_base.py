from abc import ABC, abstractmethod
from typing import Any, Dict

class RouterBase(ABC):
    """라우팅 베이스 클래스"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    @abstractmethod
    def route(self, state: Dict[str, Any]) -> str:
        """
        상태를 기반으로 다음 노드 결정
        Returns: 다음 노드의 이름
        """
        pass  # ✅ 구현 제거, 순수 추상 메서드