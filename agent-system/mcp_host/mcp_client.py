# mcp_host/mcp_client.py
import logging
from typing import Dict, Any
from fastmcp import Client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MCPHTTPClient:
    """FastMCP (Streamable-HTTP) 기반 MCP 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000/mcp", transport: str = 'http'):
        self.base_url = base_url
        self.transport = transport
        self.client: Client = Client(self.base_url, self.transport)

    async def __aenter__(self):
        await self.client.__aenter__()
        logger.info(f"🔗 Connected to FastMCP server at {self.base_url}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
        logger.info("🔌 Disconnected from FastMCP server")

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """MCP 서버의 tool 호출"""
        if not self.client.is_connected:
            raise RuntimeError("MCPHTTPClient not connected. Use 'async with'.")
        
        try:
            logger.info(f"🔧 Calling tool: {tool_name}")
            logger.debug(f"Parameters: {params}")
            
            # ✅ FastMCP Client의 call_tool 메서드 사용
            result = await self.client.call_tool(
                name=tool_name, 
                arguments=params)
            
            # ✅ CallToolResult 객체에서 content 추출
            if hasattr(result, 'content') and result.content:
                # content는 리스트이고, 각 항목은 TextContent 또는 ImageContent
                if isinstance(result.content, list) and len(result.content) > 0:
                    import json
                    content_item = result.content[0]
                    
                    # TextContent의 경우 text 속성 접근
                    if hasattr(content_item, 'text'):
                        try:
                            # JSON 문자열을 파싱
                            return json.loads(content_item.text)
                        except json.JSONDecodeError:
                            # JSON이 아니면 텍스트 그대로 반환
                            return content_item.text
            
            return {"success": False, "error": "Invalid response format"}
            
        except Exception as e:
            logger.error(f"❌ Tool call failed for '{tool_name}': {e}", exc_info=True)
            raise

    async def get_resource(self, resource_uri: str) -> Any:
        """MCP 서버의 resource 조회"""
        if not self.client.is_connected:
            raise RuntimeError("MCPHTTPClient not connected. Use 'async with'.")
        
        try:
            logger.info(f"📦 Getting resource: {resource_uri}")
            
            # ✅ FastMCP Client의 read_resource 메서드 사용
            result = await self.client.read_resource(resource_uri)
            
            # ✅ ReadResourceResult 파싱
            if hasattr(result, 'contents') and result.contents:
                if isinstance(result.contents, list) and len(result.contents) > 0:
                    content_item = result.contents[0]
                    
                    # TextResourceContents의 경우
                    if hasattr(content_item, 'text'):
                        return content_item.text
                    # BlobResourceContents의 경우
                    elif hasattr(content_item, 'blob'):
                        return content_item.blob
            
            return "No content available"
            
        except Exception as e:
            logger.error(f"❌ Resource get failed for '{resource_uri}': {e}", exc_info=True)
            raise

    async def call_prompt(self, prompt_name: str, params: Dict[str, Any]) -> Any:
        """MCP 서버의 prompt 호출"""
        if not self.client.is_connected:
            raise RuntimeError("MCPHTTPClient not connected. Use 'async with'.")
        
        try:
            logger.info(f"💬 Calling prompt: {prompt_name}")
            logger.debug(f"Parameters: {params}")
            
            # ✅ FastMCP Client의 get_prompt 메서드 사용
            result = await self.client.get_prompt(
                name=prompt_name, 
                arguments=params)
            
            # ✅ GetPromptResult 파싱
            if hasattr(result, 'messages') and result.messages:
                # 메시지 내용 추출
                messages = []
                for msg in result.messages:
                    if hasattr(msg, 'content'):
                        # content가 문자열인 경우
                        if isinstance(msg.content, str):
                            messages.append(msg.content)
                        # content가 리스트인 경우 (TextContent 객체들)
                        elif isinstance(msg.content, list):
                            for content_item in msg.content:
                                if hasattr(content_item, 'text'):
                                    messages.append(content_item.text)
                
                return "\n".join(messages) if messages else "No prompt content"
            
            return "No prompt content available"
            
        except Exception as e:
            logger.error(f"❌ Prompt call failed for '{prompt_name}': {e}", exc_info=True)
            raise

    async def list_tools(self) -> list:
        """사용 가능한 도구 목록 조회"""
        try:
            tools = await self.client.list_tools()
            return tools.tools if hasattr(tools, 'tools') else []
        except Exception as e:
            logger.error(f"❌ Failed to list tools: {e}", exc_info=True)
            raise

    async def list_resources(self) -> list:
        """사용 가능한 리소스 목록 조회"""
        try:
            resources = await self.client.list_resources()
            return resources.resources if hasattr(resources, 'resources') else []
        except Exception as e:
            logger.error(f"❌ Failed to list resources: {e}", exc_info=True)
            raise

    async def list_prompts(self) -> list:
        """사용 가능한 프롬프트 목록 조회"""
        try:
            prompts = await self.client.list_prompts()
            return prompts.prompts if hasattr(prompts, 'prompts') else []
        except Exception as e:
            logger.error(f"❌ Failed to list prompts: {e}", exc_info=True)
            raise