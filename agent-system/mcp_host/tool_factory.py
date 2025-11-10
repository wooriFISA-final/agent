import json
import logging
from typing import Any, Dict
from langchain_core.tools import StructuredTool
from .mcp_client import call_mcp_tool

logger = logging.getLogger(__name__)

def create_tool(name: str, description: str) -> StructuredTool:
    async def _wrapper(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        result = await call_mcp_tool(name, kwargs)
        return result
    return StructuredTool.from_function(
        coroutine=_wrapper,
        name=name,
        description=description,
    )
