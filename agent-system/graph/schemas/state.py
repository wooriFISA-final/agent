from typing import TypedDict, Optional

class LLMStateSchema(TypedDict, total=False):
    query: str
    research_result: Optional[str]
    analysis_result: Optional[str]
    final_report: Optional[str]
    intent_result: Optional[str]
    messages: Optional[str]
