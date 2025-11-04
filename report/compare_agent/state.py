from typing import TypedDict, Any

class AgentState(TypedDict):
    member_id: int = 1
    report_data: Any
    house_info: Any
    policy_info: Any
    credit_info: Any
    comparison_result: Any
