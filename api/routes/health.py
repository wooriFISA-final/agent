"""
헬스체크 및 루트 엔드포인트

시스템 상태 확인 및 기본 정보 제공 엔드포인트를 정의합니다.
"""
from fastapi import APIRouter, Request

from core.logging.logger import setup_logger
from agents.registry.agent_registry import AgentRegistry
from api.models import HealthResponse
from core.config.setting import settings

logger = setup_logger()

router = APIRouter()


@router.get("/")
async def root():
    """루트 엔드포인트
    
    API 기본 정보를 반환합니다.
    
    Returns:
        dict: API 상태 및 정보
    """
    return {
        "status": "ok",
        "message": "AI Agent API is running 🚀",
        "version": settings.API_VERSION,
        "agents": AgentRegistry.list_agents(),
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """헬스체크 엔드포인트
    
    시스템 상태를 확인하고 MCP 연결 상태, 사용 가능한 도구 수 등을 반환합니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        HealthResponse: 시스템 상태 정보
    """
    mcp_manager = request.app.state.mcp_manager
    try:
        await mcp_manager.ensure_connected()
        tools = await mcp_manager.list_tools()
        
        return HealthResponse(
            status="healthy",
            mcp_connected=True,
            available_tools=len(tools),
            registered_agents=AgentRegistry.list_agents()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            mcp_connected=False,
            available_tools=0,
            registered_agents=AgentRegistry.list_agents(),
            error=str(e)
        )
