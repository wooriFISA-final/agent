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
        graphs: 여러 그래프를 관리하는 딕셔너리
                {graph_name: {"graph": CompiledGraph, "checkpointer": MemorySaver, "config_loader": AgentConfigLoader}}
        session_manager: 세션 관리자 (더 이상 사용하지 않을 수 있음)
        mcp_manager: MCP 관리자
    """
    def __init__(self):
        self.graphs: dict = {}  # {name: {"graph": ..., "checkpointer": ..., "config_loader": ...}}
        self.session_manager: Optional[SessionManager] = None
        self.mcp_manager: Optional[MCPManager] = None
    
    def get_graph(self, graph_name: str = "default"):
        """그래프 이름으로 그래프 가져오기
        
        Args:
            graph_name: 그래프 이름 (기본값: "default")
            
        Returns:
            해당 이름의 그래프, 없으면 None
        """
        graph_data = self.graphs.get(graph_name)
        if graph_data:
            return graph_data.get("graph")
        return None
    
    def get_graph_checkpointer(self, graph_name: str):
        """그래프별 checkpointer 가져오기
        
        Args:
            graph_name: 그래프 이름
            
        Returns:
            해당 그래프의 checkpointer, 없으면 None
        """
        graph_data = self.graphs.get(graph_name)
        if graph_data:
            return graph_data.get("checkpointer")
        return None
    
    def get_graph_config_loader(self, graph_name: str):
        """그래프별 config_loader 가져오기
        
        Args:
            graph_name: 그래프 이름
            
        Returns:
            해당 그래프의 config_loader, 없으면 None
        """
        graph_data = self.graphs.get(graph_name)
        if graph_data:
            return graph_data.get("config_loader")
        return None
    
    def add_graph(self, name: str, graph, checkpointer=None, config_loader=None):
        """그래프 추가
        
        Args:
            name: 그래프 이름
            graph: 컴파일된 그래프 인스턴스
            checkpointer: 그래프 전용 checkpointer (선택)
            config_loader: 그래프 전용 config_loader (선택)
        """
        self.graphs[name] = {
            "graph": graph,
            "checkpointer": checkpointer,
            "config_loader": config_loader
        }
        logger.info(f"✅ Graph '{name}' added to AppState")
    
    def list_graphs(self):
        """등록된 모든 그래프 이름 반환"""
        return list(self.graphs.keys())


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

    # 0. Setup AWS Bedrock Authentication
    import os
    logger.info("🔑 Setting up AWS Bedrock authentication...")
    api_key = settings.AWS_BEARER_TOKEN_BEDROCK
    if api_key:
        os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key
        logger.info("✅ AWS_BEARER_TOKEN_BEDROCK environment variable set")
    else:
        logger.warning("⚠️ AWS_BEARER_TOKEN_BEDROCK not configured in settings")

    # 1. Global checkpointer removed - each graph will have its own
    # (Keeping this comment for reference)
    logger.info("✅ Skipping global MemorySaver (using graph-specific instances)")

    # 2. Initialize SessionManager (deprecated - each graph has its own checkpointer now)
    # Keeping for backward compatibility if needed
    app.state.session_manager = None
    logger.info("✅ SessionManager skipped (using graph-specific checkpointers)")

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


    # 4. Discover and register agents (모든 Agent 클래스 발견)
    logger.info("📦 Discovering agents...")
    AgentRegistry.auto_discover(module_path=settings.AGENTS_MODULE_PATH)

    # 5. Discover and register routers
    logger.info("🔍 Discovering routers...")
    RouterRegistry.auto_discover()
    
    # 6. Build multiple graphs with their own agent configurations
    from pathlib import Path
    base_path = Path(__file__).parent.parent  # agent/ 디렉토리
    
    graph_configs = {
        "plan": {
            "graph_yaml": "graph/config/plan_graph.yaml",
            "agents_yaml": str(base_path / "agents/config/plan_agents.yaml")
        },
        "report": {
            "graph_yaml": "graph/config/report_graph.yaml",
            "agents_yaml": str(base_path / "agents/config/report_agents.yaml")
        }
    }
    
    for graph_name, config in graph_configs.items():
        logger.info(f"🔧 Building '{graph_name}' graph...")
        
        # Create graph-specific MemorySaver
        graph_checkpointer = MemorySaver()
        logger.info(f"✅ Created independent MemorySaver for '{graph_name}' graph")
        
        # Load agent configuration for this graph
        try:
            logger.info(f"📋 Loading agents from '{config['agents_yaml']}'...")
            config_loader = AgentConfigLoader(yaml_path=config['agents_yaml'])
            enabled_agents = config_loader.get_enabled_agents()
            logger.info(f"✅ Loaded {len(enabled_agents)} enabled agents for '{graph_name}'")
        except FileNotFoundError:
            logger.warning(f"⚠️ Agent config file not found: {config['agents_yaml']}")
            logger.info(f"ℹ️  Skipping '{graph_name}' graph")
            continue
        except Exception as e:
            logger.error(f"❌ Error loading agent config for '{graph_name}': {e}")
            continue
        
        # Build graph with loaded agent configuration and graph-specific checkpointer
        try:
            graph = mk_graph(
                yaml_path=str(config['graph_yaml']),
                checkpointer=graph_checkpointer,
                config_loader=config_loader
            )
            if graph:
                app.state.add_graph(
                    name=graph_name,
                    graph=graph,
                    checkpointer=graph_checkpointer,
                    config_loader=config_loader
                )
                logger.info(f"✅ '{graph_name}' graph built successfully with independent memory!")
            else:
                logger.warning(f"⚠️ Failed to build '{graph_name}' graph from '{config['graph_yaml']}'")
        except FileNotFoundError:
            logger.warning(f"⚠️ Graph config file not found: {config['graph_yaml']}")
        except Exception as e:
            logger.error(f"❌ Error building '{graph_name}' graph: {e}")
    
    if not app.state.graphs:
        logger.error("❌ No graphs could be built. Shutting down.")
        return
    
    logger.info(f"✅ Total {len(app.state.graphs)} graph(s) built: {app.state.list_graphs()}")
    logger.info("🎉 Application startup complete.")

    yield

    # --- Shutdown Logic ---
    logger.info("🧹 Shutting down Multi-Agent System...")
    if app.state.mcp_manager:
        await app.state.mcp_manager.close()
        logger.info("✅ MCP connection closed.")
    logger.info("👋 Application shutdown complete.")
