from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from typing import Optional

# Centralized settings
from core.config.setting import settings
from agent.registry.agent_registry import AgentRegistry
from agent.config.base_config import StateBuilder
from core.logging.logger import setup_logger
from graph.factory import mk_graph
from core.mcp.mcp_manager import MCPManager
from utils.session_manager import SessionManager

logger = setup_logger()

# =============================
# Application State
# =============================
# Use a class to hold application state, attached to the FastAPI app instance.
class AppState:
    def __init__(self):
        self.graph = None
        self.checkpointer: Optional[MemorySaver] = None
        self.session_manager: Optional[SessionManager] = None
        self.mcp_manager: Optional[MCPManager] = None

# =============================
# Lifespan Event Handler
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI app startup and shutdown logic."""
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
    from agent.config.agent_config_loader import AgentConfigLoader
    
    AgentConfigLoader(yaml_path=str(settings.AGENTS_CONFIG_PATH))
    enabled_agents = AgentConfigLoader.get_enabled_agents()
    logger.info(f"✅ Loaded {len(enabled_agents)} enabled agents from agents.yaml")
    
    # 5. Discover and register agents
    logger.info("📦 Discovering agents...")
    AgentRegistry.auto_discover(module_path=settings.AGENTS_MODULE_PATH)

    # 6. Discover and register routers
    logger.info("🔍 Discovering routers...")
    from graph.routing.router_registry import RouterRegistry
    RouterRegistry.auto_discover()
    
    # 7. Build the main agent graph
    logger.info(f"🔧 Building agent graph from '{settings.GRAPH_YAML_PATH}'...")
    app.state.graph = mk_graph(
        yaml_path=str(settings.GRAPH_YAML_PATH),
        checkpointer=app.state.checkpointer
    )
    if not app.state.graph:
        logger.error("❌ Agent graph could not be built. Shutting down.")
        # In a real scenario, you might want to prevent the app from starting
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


app = FastAPI(
    title="Multi-Agent Planner",
    version=settings.API_VERSION,
    description="Multi-Agent system with conversation history",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    mcp_connected: bool
    available_tools: int
    registered_agents: list
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AI Agent API is running 🚀",
        "version": settings.API_VERSION,
        "agents": AgentRegistry.list_agents(),
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Provides a health check of the system, including MCP connection."""
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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Handles chat requests, managing conversation history and invoking the agent graph.
    """
    graph = request.app.state.graph
    if not graph:
        logger.error("❌ Agent graph not initialized")
        return ChatResponse(
            response="System is not initialized.",
            status="error",
            metadata={"error": "graph_not_initialized"}
        )

    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"📩 NEW REQUEST | Session: {chat_request.session_id}")
        logger.info(f"   Message: {chat_request.message}")
        logger.info(f"{'='*80}")

        graph_config = {"configurable": {"thread_id": chat_request.session_id}}

        # Check for existing conversation state
        try:
            existing_state = await graph.aget_state(graph_config)
            has_history = existing_state and existing_state.values.get('messages')
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing state for session '{chat_request.session_id}': {e}")
            has_history = False

        if has_history:
            logger.info(f"📚 Continuing conversation for session '{chat_request.session_id}'")
            input_state = {"messages": [HumanMessage(content=chat_request.message)]}
        else:
            logger.info(f"🆕 Starting new conversation for session '{chat_request.session_id}'")
            input_state = StateBuilder.create_initial_state(
                messages=[HumanMessage(content=chat_request.message)],
                session_id=chat_request.session_id,
                max_iterations=settings.MAX_GRAPH_ITERATIONS
            )

        # Execute the agent graph
        logger.info("🚀 Executing agent graph...")
        result_state = await graph.ainvoke(input_state, config=graph_config)
        logger.info("✅ Graph execution completed.")

        # Extract the final response
        all_messages = result_state.get("messages", [])
        ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]

        if not ai_messages:
            logger.warning("⚠️ No AI messages found in the final state.")
            return ChatResponse(response="AI did not generate a response.", status="warning")

        final_response = ai_messages[-1].content
        logger.info(f"💬 Returning response for session '{chat_request.session_id}'.")

        return ChatResponse(
            response=final_response,
            status="success",
            metadata={"session_id": chat_request.session_id}
        )

    except asyncio.TimeoutError:
        logger.error(f"❌ Request timeout for session '{chat_request.session_id}'")
        return ChatResponse(
            response="Request timed out.",
            status="error",
            metadata={"error": "timeout", "session_id": chat_request.session_id}
        )
    
    except Exception as e:
        logger.error(f"❌ Chat processing failed for session '{chat_request.session_id}': {e}", exc_info=True)
        return ChatResponse(
            response=f"An internal error occurred: {str(e)}",
            status="error",
            metadata={"error": "processing_error", "detail": str(e), "session_id": chat_request.session_id}
        )

# =============================
# Session Management API
# =============================

@app.get("/chat/sessions")
async def list_sessions(request: Request):
    """Lists all active session IDs."""
    session_manager = request.app.state.session_manager
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    sessions = session_manager.list_all_sessions()
    return {"status": "success", "sessions": sessions, "count": len(sessions)}


@app.get("/chat/sessions/detailed")
async def list_sessions_detailed(request: Request):
    """Lists detailed information for all active sessions."""
    session_manager = request.app.state.session_manager
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
        
    sessions = session_manager.list_sessions_with_details()
    return {"status": "success", "sessions": sessions, "count": len(sessions)}


@app.get("/chat/session/{session_id}/history")
async def get_conversation_history(session_id: str, request: Request):
    """Retrieves the conversation history for a specific session."""
    graph = request.app.state.graph
    if not graph:
        return {"status": "error", "message": "Graph not initialized"}
    
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)
        
        if not state or not state.values:
            return {"status": "not_found", "message": f"Session {session_id} not found", "messages": []}
        
        messages = state.values.get('messages', [])
        message_list = [
            {"type": type(msg).__name__, "content": msg.content} for msg in messages
        ]
        
        return {
            "status": "success",
            "session_id": session_id,
            "message_count": len(messages),
            "messages": message_list
        }
    except Exception as e:
        logger.error(f"Failed to get conversation history for '{session_id}': {e}")
        return {"status": "error", "message": str(e)}


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Deletes a session and its history."""
    session_manager = request.app.state.session_manager
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    result = session_manager.delete_session(session_id)
    if result["deleted"]:
        logger.info(f"🗑️ Session {session_id} deleted")
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        return {"status": "not_found", "message": f"Session {session_id} not found"}


# =============================
# Server Execution
# =============================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting API Server on http://{settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )