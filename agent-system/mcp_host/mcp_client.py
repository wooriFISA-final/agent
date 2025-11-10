import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MCPHTTPClient:
    """HTTP 기반 MCP 클라이언트 (분리 실행된 MCP 서버에 연결)"""
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        logger.info(f"🔗 Connected to MCP server at {self.base_url}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        logger.info("🔌 Disconnected from MCP server")

    async def call_tool(self, tool_name: str, params: Dict[str, Any]):
        """MCP 서버의 특정 tool 엔드포인트 호출"""
        if self.session is None:
            raise RuntimeError("MCPHTTPClient not initialized. Use 'async with'.")
        
        url = f"{self.base_url}/tools/{tool_name}"
        async with self.session.post(url, json=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"❌ MCP tool call failed: {resp.status}, {text}")
            data = await resp.json()
            return data
