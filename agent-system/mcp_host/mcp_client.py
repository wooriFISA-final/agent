import aiohttp
import logging
from typing import Optional, Dict, Any
from fastmcp.client import Client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MCPHTTPClient:
    """
    FastMCP (Streamable-HTTP) 기반 MCP 클라이언트
    (단일 /mcp 엔드포인트로 통신)
    """
    def __init__(self, base_url: str = "http://localhost:8000/mcp"):  # ✅ trailing slash 제거
        self.base_url = base_url
        # FastMCPClient 인스턴스 생성
        self.client: Client = Client(self.base_url)

    async def __aenter__(self):
        """컨텍스트 매니저 시작 시, FastMCPClient의 세션을 시작합니다."""
        await self.client.__aenter__()
        logger.info(f"🔗 Connected to FastMCP server at {self.base_url}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 시, FastMCPClient의 세션을 종료합니다."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
        logger.info("🔌 Disconnected from FastMCP server")

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """MCP 서버의 tool 호출"""
        if not self.client.is_connected:
            raise RuntimeError("MCPHTTPClient not connected. Use 'async with'.")
        
        try:
            logger.debug(f"Calling tool: {tool_name} with params: {params}")
            result = await self.client.tool.call(tool_name, **params) 
            return result
        except Exception as e:
            logger.error(f"❌ MCP tool call failed for '{tool_name}': {e}")
            raise RuntimeError(f"MCP tool call failed: {e}") from e

    async def get_resource(self, resource_uri: str) -> Any:
        """MCP 서버의 resource 호출"""
        if not self.client.is_connected:
            raise RuntimeError("MCPHTTPClient not connected. Use 'async with'.")
        
        try:
            logger.debug(f"Getting resource: {resource_uri}")
            result = await self.client.resource.get(resource_uri)
            return result
        except Exception as e:
            logger.error(f"❌ MCP resource get failed for '{resource_uri}': {e}")
            raise RuntimeError(f"MCP resource get failed: {e}") from e

    async def call_prompt(self, prompt_name: str, params: Dict[str, Any]) -> Any:
        """MCP 서버의 prompt 호출"""
        if not self.client.is_connected:
            raise RuntimeError("MCPHTTPClient not connected. Use 'async with'.")
        
        try:
            logger.debug(f"Calling prompt: {prompt_name} with params: {params}")
            result = await self.client.prompt.run(prompt_name, **params)
            return result
        except Exception as e:
            logger.error(f"❌ MCP prompt call failed for '{prompt_name}': {e}")
            raise RuntimeError(f"MCP prompt call failed: {e}") from e