# core/mcp/mcp_manager.py

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

class MCPManager:
    """MCP 클라이언트 싱글톤 매니저"""
    
    _instance: Optional['MCPManager'] = None
    _client: Optional[Client] = None
    _transport: Optional[StreamableHttpTransport] = None
    _connected: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, url: str, headers: Optional[Dict[str, str]] = None):
        """MCP 클라이언트 초기화"""
        if self._client is not None:
            logger.warning("MCP client already initialized")
            return

        self._transport = StreamableHttpTransport(url=url, headers=headers or {})
        self._client = Client(self._transport)
        logger.info(f"MCP client initialized with URL: {url}")

    async def connect(self):
        """MCP 서버에 연결"""
        if self._client is None:
            raise RuntimeError("MCP client not initialized. Call initialize() first.")
        if self._connected:
            return  # 이미 연결됨

        try:
            await self._client.__aenter__()
            self._connected = True
            logger.info("✅ MCP client connected successfully")
        except Exception as e:
            self._connected = False
            logger.error(f"❌ Failed to connect MCP client: {e}")
            raise

    async def ensure_connected(self):
        """세션이 닫혀 있으면 자동 복구"""
        if not self._connected:
            logger.warning("⚠️ MCP session closed — reconnecting...")
            await self.connect()

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("MCP client not initialized.")
        return self._client

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """MCP 도구 호출"""
        await self.ensure_connected()
        try:
            result = await self.client.call_tool(name, args)
            return result
        except Exception as e:
            logger.error(f"MCP tool '{name}' failed: {e}")
            # 💡 연결이 닫혔을 가능성 → 다시 시도
            self._connected = False
            await self.ensure_connected()
            result = await self.client.call_tool(name, args)
            return result

    async def list_tools(self) -> list:
        """도구 목록 조회"""
        await self.ensure_connected()
        try:
            tools = await self.client.list_tools()
            return tools
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            self._connected = False
            await self.ensure_connected()
            return await self.client.list_tools()

    async def close(self):
        """연결 종료"""
        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
                logger.info("MCP client disconnected")
            except Exception as e:
                logger.warning(f"Error during MCP disconnect: {e}")
        self._client = None
        self._transport = None
        self._connected = False
