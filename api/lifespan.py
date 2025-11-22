"""
애플리케이션 라이프사이클 관리

FastAPI 앱의 시작(startup)과 종료(shutdown) 로직을 관리합니다.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional
import asyncio

from core.config.setting import settings
from core.logging.logger import setup_logger
from core.mcp.mcp_manager import MCPManager
from utils.session_manager import SessionManager
from agents.registry.agent_registry import AgentRegistry
from agents.config.agent_config_loader import AgentConfigLoader
from graph.factory import mk_graph
from graph.routing.router_registry import RouterRegistry

logger = setup_logger()


class AppState:
    """애플리케이션 상태를 관리하는 클래스
    
    Attributes:
        graph: LangGraph 인스턴스
        checkpointer: 메모리 체크포인터
        session_manager: 세션 관리자
        mcp_manager: MCP 관리자
    """
    def __init__(self):
        self.graph = None
        self.checkpointer: Optional[MemorySaver] = None
        self.session_manager: Optional[SessionManager] = None
        self.mcp_manager: Optional[MCPManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱 라이프사이클 관리
    
    시작 시:
    - Checkpointer 초기화
    - SessionManager 초기화
    - MCP 연결
    - Agent 로드 및 등록
    - Router 등록
    - Graph 빌드
    
    종료 시:
    - MCP 연결 종료
    
    Args:
        app: FastAPI 애플리케이션 인스턴스
    """
    logger.info(f"🚀 Starting Multi-Agent System (v{settings.API_VERSION}) in {settings.ENVIRONMENT} mode...")
    
    app.state = AppState()

    # 1. Initialize Checkpointer
    app.state.checkpointer = MemorySaver()
    logger.info("✅ Global MemorySaver initialized")

    # 2. Initialize SessionManager
    app.state.session_manager = SessionManager(app.state.checkpointer)
    logger.info("✅ SessionManager initialized")

    # 3. Initialize and connect to MCP
    app.state.mcp_manager = MCPManager()
    app.state.mcp_manager.initialize(str(settings.MCP_URL))

    for attempt in range(1, settings.MCP_CONNECTION_RETRIES + 1):
        try:
            await app.state.mcp_manager.connect()
            logger.info("✅ MCP connected successfully!")
            break
        except Exception as e:
            logger.warning(f"⚠️  MCP connection attempt {attempt}/{settings.MCP_CONNECTION_RETRIES} failed: {e}")
            if attempt < settings.MCP_CONNECTION_RETRIES:
                await asyncio.sleep(settings.MCP_CONNECTION_TIMEOUT)
            else:
                logger.error(f"❌ Failed to connect to MCP after {settings.MCP_CONNECTION_RETRIES} attempts")
                raise

    # 4. Load agents.yaml configuration
    logger.info("📋 Loading agents.yaml configuration...")
    AgentConfigLoader(yaml_path=str(settings.AGENTS_CONFIG_PATH))
    enabled_agents = AgentConfigLoader.get_enabled_agents()
    logger.info(f"✅ Loaded {len(enabled_agents)} enabled agents from agents.yaml")
    
    # 5. Discover and register agents
    logger.info("📦 Discovering agents...")
    AgentRegistry.auto_discover(module_path=settings.AGENTS_MODULE_PATH)

    # 6. Discover and register routers
    logger.info("🔍 Discovering routers...")
    RouterRegistry.auto_discover()
    
    # 7. Build the main agent graph
    logger.info(f"🔧 Building agent graph from '{settings.GRAPH_YAML_PATH}'...")
    app.state.graph = mk_graph(
        yaml_path=str(settings.GRAPH_YAML_PATH),
        checkpointer=app.state.checkpointer
    )
    if not app.state.graph:
        logger.error("❌ Agent graph could not be built. Shutting down.")
        return

    logger.info("✅ Agent graph built successfully!")
    logger.info("🎉 Application startup complete.")

    yield

    # --- Shutdown Logic ---
    logger.info("🧹 Shutting down Multi-Agent System...")
    if app.state.mcp_manager:
        await app.state.mcp_manager.close()
        logger.info("✅ MCP connection closed.")
    logger.info("👋 Application shutdown complete.")
