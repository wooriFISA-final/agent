"""
Multi-Agent System 서버 실행 스크립트

이 스크립트는 Multi-Agent System의 FastAPI 서버를 시작합니다.

사용법:
    uv run main.py

서버는 자동으로 다음 작업을 수행합니다:
- MCP 서버 연결
- Agent 로드 및 등록
- Router 등록
- Graph 빌드
- API 서버 시작
"""
import uvicorn

from core.config.setting import settings
from core.logging.logger import setup_logger

logger = setup_logger()


if __name__ == "__main__":
    logger.info(f"🚀 Starting API Server on http://{settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )