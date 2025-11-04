from typing import TypedDict, Any

class AgentState(TypedDict):
    member_id: int
    is_test: bool
    report_data: Any
    house_info: Any
    policy_info: Any
    credit_info: Any
    comparison_result: Any
