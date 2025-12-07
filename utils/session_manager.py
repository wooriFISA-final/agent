# ============================================================================
# 세션 관리 모듈
# ============================================================================

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


# ============================================================================
# CheckpointerAdapter - 추상 클래스 (인라인으로 포함)
# ============================================================================

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


# ============================================================================
# MemorySaver 전용 어댑터
# ============================================================================

class MemorySaverAdapter(CheckpointerAdapter):
    """MemorySaver 전용 어댑터"""
    
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
    
    def list_sessions(self) -> List[str]:
        """
        활성 세션 목록 조회
        
        MemorySaver의 storage를 분석하여 고유한 thread_id 추출
        """
        if not hasattr(self.checkpointer, 'storage'):
            return []
        
        # 모든 키에서 thread_id 추출
        session_ids = set()
        
        for key in self.checkpointer.storage.keys():
            # key는 (config_dict, checkpoint_id) 형태의 튜플
            config_dict = key[0]
            
            # config_dict에서 thread_id 추출
            if isinstance(config_dict, dict):
                thread_id = config_dict.get("thread_id")
                if thread_id:
                    session_ids.add(thread_id)
        
        return sorted(list(session_ids))
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션의 상세 정보 조회
        
        Returns:
            {
                "session_id": str,
                "checkpoint_count": int,
                "first_checkpoint": datetime,
                "last_checkpoint": datetime,
                "message_count": int
            }
        """
        if not hasattr(self.checkpointer, 'storage'):
            return None
        
        checkpoints = []
        
        # 해당 세션의 모든 체크포인트 찾기
        for key, checkpoint in self.checkpointer.storage.items():
            config_dict = key[0]
            
            if isinstance(config_dict, dict) and config_dict.get("thread_id") == session_id:
                checkpoints.append(checkpoint)
        
        if not checkpoints:
            return None
        
        # 체크포인트에서 정보 추출
        timestamps = []
        total_messages = 0
        
        for cp in checkpoints:
            # 타임스탬프 수집
            if isinstance(cp, dict) and 'ts' in cp:
                timestamps.append(cp['ts'])
            
            # 메시지 개수 계산
            if isinstance(cp, dict) and 'channel_values' in cp:
                messages = cp.get('channel_values', {}).get('messages', [])
                total_messages = max(total_messages, len(messages))
        
        return {
            "session_id": session_id,
            "checkpoint_count": len(checkpoints),
            "first_checkpoint": min(timestamps) if timestamps else None,
            "last_checkpoint": max(timestamps) if timestamps else None,
            "message_count": total_messages
        }
    
    def delete_session(self, session_id: str) -> int:
        """
        세션의 모든 체크포인트 삭제
        
        Returns:
            삭제된 체크포인트 수
        """
        if not hasattr(self.checkpointer, 'storage'):
            return 0
        
        # 삭제할 키 목록 수집
        keys_to_delete = []
        
        for key in self.checkpointer.storage.keys():
            config_dict = key[0]
            
            if isinstance(config_dict, dict) and config_dict.get("thread_id") == session_id:
                keys_to_delete.append(key)
        
        # 삭제 실행
        for key in keys_to_delete:
            del self.checkpointer.storage[key]
        
        return len(keys_to_delete)
    
    def get_checkpoint_count(self, session_id: str) -> int:
        """세션의 체크포인트 개수"""
        if not hasattr(self.checkpointer, 'storage'):
            return 0
        
        count = 0
        for key in self.checkpointer.storage.keys():
            config_dict = key[0]
            if isinstance(config_dict, dict) and config_dict.get("thread_id") == session_id:
                count += 1
        
        return count


# ============================================================================
# SessionManager - 세션 관리를 위한 헬퍼 클래스
# ============================================================================

class SessionManager:
    """
    세션 관리를 위한 통합 인터페이스
    
    다양한 Checkpointer를 지원하고 세션 관련 작업을 단순화
    """
    
    def __init__(self, checkpointer):
        """
        Args:
            checkpointer: LangGraph Checkpointer 인스턴스
        """
        self.checkpointer = checkpointer
        self.adapter = self._create_adapter()
    
    def _create_adapter(self) -> CheckpointerAdapter:
        """Checkpointer 타입에 따라 적절한 어댑터 생성"""
        checkpointer_type = type(self.checkpointer).__name__
        
        if checkpointer_type == "MemorySaver":
            return MemorySaverAdapter(self.checkpointer)
        # 향후 다른 타입 추가 가능
        # elif checkpointer_type == "RedisSaver":
        #     return RedisSaverAdapter(self.checkpointer)
        else:
            # 기본 어댑터 (최소 기능)
            return MemorySaverAdapter(self.checkpointer)
    
    def list_all_sessions(self) -> List[str]:
        """모든 활성 세션 ID 목록"""
        return self.adapter.list_sessions()
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 상세 정보"""
        return self.adapter.get_session_info(session_id)
    
    def list_sessions_with_details(self) -> List[Dict[str, Any]]:
        """모든 세션의 상세 정보 목록"""
        session_ids = self.list_all_sessions()
        
        sessions = []
        for sid in session_ids:
            info = self.get_session_details(sid)
            if info:
                sessions.append(info)
        
        return sessions
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        세션 삭제
        
        Returns:
            {
                "session_id": str,
                "deleted": bool,
                "checkpoints_deleted": int
            }
        """
        deleted_count = self.adapter.delete_session(session_id)
        
        return {
            "session_id": session_id,
            "deleted": deleted_count > 0,
            "checkpoints_deleted": deleted_count
        }
    
    def cleanup_empty_sessions(self) -> Dict[str, Any]:
        """빈 세션 정리 (체크포인트가 없는 세션)"""
        all_sessions = self.list_all_sessions()
        deleted_sessions = []
        
        for session_id in all_sessions:
            count = self.adapter.get_checkpoint_count(session_id)
            if count == 0:
                self.delete_session(session_id)
                deleted_sessions.append(session_id)
        
        return {
            "deleted_sessions": deleted_sessions,
            "count": len(deleted_sessions)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계 정보"""
        sessions = self.list_sessions_with_details()
        
        if not sessions:
            return {
                "total_sessions": 0,
                "total_checkpoints": 0,
                "total_messages": 0,
                "avg_checkpoints_per_session": 0,
                "avg_messages_per_session": 0
            }
        
        total_checkpoints = sum(s.get("checkpoint_count", 0) for s in sessions)
        total_messages = sum(s.get("message_count", 0) for s in sessions)
        
        return {
            "total_sessions": len(sessions),
            "total_checkpoints": total_checkpoints,
            "total_messages": total_messages,
            "avg_checkpoints_per_session": total_checkpoints / len(sessions),
            "avg_messages_per_session": total_messages / len(sessions)
        }