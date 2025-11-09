from graph.routing.router_base import RouterBase  # ✅ 경로 수정
from typing import Dict, Any, Optional

class ConditionalRouter(RouterBase):
    """조건 기반 라우팅"""
    
    def __init__(
        self, 
        condition_key: str, 
        thresholds: Optional[Dict[str, float]] = None,
        default: str = "default"
    ):
        super().__init__()
        self.condition_key = condition_key
        self.thresholds = thresholds or {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.0
        }
        self.default = default
    
    def route(self, state: Dict[str, Any]) -> str:
        value = state.get(self.condition_key)
        
        if value is None:
            return self.default
        
        # 임계값에 따라 라우팅 (내림차순)
        for path, threshold in sorted(
            self.thresholds.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if value >= threshold:
                return path
        
        return self.default