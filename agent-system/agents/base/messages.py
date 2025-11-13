from langchain_core.messages import AIMessage
from datetime import datetime
from typing import Optional, Dict

class HumanMessage(Message):
    pass
class AIMessage(Message):
    pass

class ToolMessage(Message):
    pass
class ThinkMessage(AIMessage):
    """Agent 내부 사고 메시지"""
    def __init__(self, content: str, metadata: Optional[Dict] = None, timestamp: Optional[datetime] = None):
        super().__init__(content=content, metadata=metadata or {})
        self.timestamp = timestamp or datetime.now()
        self.type = "think"  # 메시지 타입 구분용

class ResultMessage(AIMessage):
    """최종 결과 메시지"""
    def __init__(self, content: str, metadata: Optional[Dict] = None, timestamp: Optional[datetime] = None):
        super().__init__(content=content, metadata=metadata or {})
        self.timestamp = timestamp or datetime.now()
        self.type = "result"  # 메시지 타입 구분용
