from pydantic import BaseModel
from typing import List, Optional

class BaseAgentConfig(BaseModel):
    """모든 Agent 공통 설정"""
    """Agent 설정 정보를 담는 모델"""
    name: str                            # Agent 이름 (필수)
    model_name: str = "qwen3:8b"         # 모델 명
    description: Optional[str] = None    # 설명 (선택)
    max_retries: int = 3                  # 실행 실패 시 재시도 횟수
    timeout: int = 30                     # 실행 타임아웃(초)
    enabled: bool = True                  # Agent 활성화 여부
    dependencies: List[str] = []          # 다른 Agent 의존성 목록