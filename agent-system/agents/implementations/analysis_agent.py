from agents.base.agent_base import AgentBase, AgentConfig
from agents.registry.agent_registry import AgentRegistry
from typing import Dict, Any

@AgentRegistry.register("analysis")
class AnalysisAgent(AgentBase):
    """분석 Agent"""
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = state.get("research_result", "")
        state["analysis_result"] = f"'{result}'를 분석 완료"
        return state
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        return "research_result" in state