from routing.router_base import RouterBase
from typing import Dict, Any

class SavingRouter(RouterBase):
    def route(self, state: Dict[str, Any]) -> str:
        """
        적금/예금 추천 완료 플래그를 확인하여 'NEXT' 또는 'CONTINUE'를 반환합니다.
        """
        # SavingAgent가 설정할 플래그 키를 사용해야 합니다.
        is_done = state.get("is_saving_done", False) 
        
        if is_done:
            # YAML: NEXT: fund_recommend
            return "NEXT"
        else:
            # YAML: CONTINUE: saving_recommend
            return "CONTINUE"