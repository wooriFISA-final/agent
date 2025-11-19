from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class CheckpointerAdapter(ABC):
    """
    다양한 Checkpointer 타입을 지원하는 어댑터 패턴
    
    MemorySaver, RedisSaver, PostgresSaver 등을 통일된 인터페이스로 관리
    """
    
    @abstractmethod
    def list_sessions(self) -> List[str]:
        """활성 세션 목록 조회"""
        pass
    
    @abstractmethod
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """특정 세션의 상세 정보"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """세션 삭제 (삭제된 체크포인트 수 반환)"""
        pass
    
    @abstractmethod
    def get_checkpoint_count(self, session_id: str) -> int:
        """세션의 체크포인트 개수"""
        pass