from routing.router_base import RouterBase
from typing import Dict, Any

class FundRouter(RouterBase):
    def route(self, state: Dict[str, Any]) -> str:
        """
        펀드 추천 완료 플래그를 확인하여 'NEXT' 또는 'CONTINUE'를 반환합니다.
        """
        # FundAgent가 설정할 플래그 키를 사용해야 합니다.
        is_done = state.get("is_fund_done", False) 
        
        if is_done:
            # YAML: NEXT: summary_node
            return "NEXT"
        else:
            # YAML: CONTINUE: fund_recommend
            return "CONTINUE"