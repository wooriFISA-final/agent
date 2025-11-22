"""
FastAPI 애플리케이션 설정

FastAPI 앱 인스턴스를 생성하고 라우터를 등록합니다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config.setting import settings
from api.lifespan import lifespan
from api.routes import health, chat, session


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 생성 및 설정
    
    Returns:
        FastAPI: 설정된 FastAPI 애플리케이션 인스턴스
    """
    app = FastAPI(
        title="Multi-Agent Planner",
        version=settings.API_VERSION,
        description="Multi-Agent system with conversation history",
        lifespan=lifespan
    )

    # CORS 미들웨어 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(session.router, tags=["Session"])

    return app


# FastAPI 앱 인스턴스 생성
app = create_app()
