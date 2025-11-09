from typing import TypedDict, Optional

class MyStateSchema(TypedDict, total=False):
    """테스트용 단순 state schema"""
    query: str
    research_result: Optional[str]
    analysis_result: Optional[str]
