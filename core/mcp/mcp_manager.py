# core/mcp/mcp_manager.py

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from typing import Optional, Any, Dict
import logging
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class MCPManager:
    """MCP 클라이언트 싱글톤 매니저 (강화된 연결 복구)"""
    
    _instance: Optional['MCPManager'] = None
    _client: Optional[Client] = None
    _transport: Optional[StreamableHttpTransport] = None
    _connected: bool = False
    _url: Optional[str] = None
    _headers: Optional[Dict[str, str]] = None
    _connection_lock: Optional[asyncio.Lock] = None

    # ---------------------------
    # 🔥 싱글톤 생성
    # ---------------------------
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connection_lock = asyncio.Lock()  # 🔥 여기서 Lock 생성
        return cls._instance

    @classmethod
    def get_instance(cls):
        return cls()

    # ---------------------------
    # 설정
    # ---------------------------
    def initialize(self, url: str, headers: Optional[Dict[str, str]] = None):
        """MCP 클라이언트 초기화"""
        self._url = url
        self._headers = headers or {}

        logger.info(f"MCP client configured with URL: {url}")

    # ---------------------------
    # 연결
    # ---------------------------
    async def connect(self):
        """MCP 서버에 연결 (멱등성 보장)"""
        # Lock 없을 가능성 대비 안전장치
        if self._connection_lock is None:
            self._connection_lock = asyncio.Lock()

        async with self._connection_lock:
            # 이미 연결되어 있고 정상 작동?
            if self._connected and self._client is not None:
                try:
                    await self._client.list_tools()
                    logger.debug("MCP connection already active and healthy")
                    return
                except Exception:
                    logger.warning("MCP connection stale — reconnecting...")
                    await self._force_disconnect()

            if self._url is None:
                raise RuntimeError("MCP client not initialized. Call initialize() first.")

            try:
                # transport 생성
                self._transport = StreamableHttpTransport(
                    url=self._url,
                    headers=self._headers
                )

                # Client 생성
                self._client = Client(self._transport)

                # 연결 시작
                await self._client.__aenter__()
                self._connected = True

                logger.info("✅ MCP client connected successfully")

            except Exception as e:
                self._connected = False
                self._client = None
                self._transport = None
                logger.error(f"❌ Failed to connect MCP client: {e}")
                raise

    # ---------------------------
    # 강제 종료
    # ---------------------------
    async def _force_disconnect(self):
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error during force disconnect: {e}")

        self._client = None
        self._transport = None
        self._connected = False

    # ---------------------------
    # 상태 확인
    # ---------------------------
    async def ensure_connected(self):
        if not self._connected or self._client is None:
            logger.warning("⚠️ MCP session not active — reconnecting...")
            await self.connect()

    # ---------------------------
    # property
    # ---------------------------
    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("MCP client not initialized or disconnected.")
        return self._client

    # ---------------------------
    # 도구 호출 (자동 재시도)
    # ---------------------------
    async def call_tool(self, name: str, args: Dict[str, Any], max_retries: int = 3) -> Any:
        for attempt in range(max_retries):
            try:
                await self.ensure_connected()
                return await self.client.call_tool(name, args)

            except Exception as e:
                error_msg = str(e).lower()

                if any(x in error_msg for x in ['closed', 'connection', 'timeout', 'session']):
                    logger.warning(f"MCP tool '{name}' failed (attempt {attempt+1}/{max_retries}): {e}")
                    self._connected = False  # 연결 상태 초기화

                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        raise
                else:
                    logger.error(f"MCP tool '{name}' execution error: {e}")
                    raise

    # ---------------------------
    # 도구 목록
    # ---------------------------
    async def list_tools(self, max_retries: int = 3) -> list:
        for attempt in range(max_retries):
            try:
                await self.ensure_connected()
                return await self.client.list_tools()

            except Exception as e:
                self._connected = False
                logger.warning(f"Failed to list MCP tools (attempt {attempt+1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise

    # ---------------------------
    # 종료
    # ---------------------------
    async def close(self):
        if self._connection_lock is None:
            self._connection_lock = asyncio.Lock()

        async with self._connection_lock:
            if self._client and self._connected:
                try:
                    await self._client.__aexit__(None, None, None)
                    logger.info("MCP client disconnected")
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")

            self._client = None
            self._transport = None
            self._connected = False

    # ---------------------------
    # 세션 매니저
    # ---------------------------
    @asynccontextmanager
    async def session(self):
        await self.connect()
        try:
            yield self
        finally:
            pass