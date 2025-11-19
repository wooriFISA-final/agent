from .router_base import RouterBase # 같은 폴더에 있는 router_base.py를 import
from typing import Dict, Any
import logging

logger = logging.getLogger("agent_system")

class PlanInputRouter(RouterBase):
    """
    PlanInputAgent가 설정한 완료 플래그(is_input_complete)를 기반으로 라우팅합니다.
    """
    
    def route(self, state: Dict[str, Any]) -> str:
        """
        state를 확인하여 'NEXT' (완료) 또는 'CONTINUE' (반복)를 반환합니다.
        """
        
        # 1. PlanInputAgent가 설정한 플래그 키를 가져옵니다.
        is_complete = state.get("is_input_complete", False)
        
        # 2. 완료 상태에 따라 YAML 경로 키를 반환합니다.
        if is_complete:
            # graph.yaml: paths: NEXT: validation
            next_step = "NEXT"
        else:
            # graph.yaml: paths: CONTINUE: plan_input
            next_step = "CONTINUE"
            
        logger.info(f"[PlanInputRouter] is_input_complete: {is_complete}, routing to: {next_step}")
        
        return next_step