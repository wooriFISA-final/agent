# mcp_client.py
import asyncio
import sys
import os
import logging
import json
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from langchain_core.tools import StructuredTool
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

class MCPClientManager:
    """
    MCP 서버와의 연결을 관리하는 싱글톤 클래스.
    비동기 컨텍스트 매니저로 안전한 리소스 관리를 제공합니다.
    """
    
    _instance: Optional['MCPClientManager'] = None
    _session: Optional[ClientSession] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, server_script: str = "mcp_server.py"):
        if hasattr(self, '_initialized'):
            return
            
        self.server_script = server_script
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
            env=dict(os.environ, PYTHONUNBUFFERED="1")
        )
        self._client = None
        self._read = None
        self._write = None
        self._session_cm = None
        self._client_cm = None
        self._initialized = True

    async def __aenter__(self):
        """컨텍스트 진입 시 서버 시작 및 클라이언트 세션 초기화"""
        logger.info(f"🔗 Starting MCP server: {self.server_script}")
        
        try:
            # stdio 클라이언트 시작
            self._session_cm = stdio_client(self.server_params)
            self._read, self._write = await self._session_cm.__aenter__()
            
            # 클라이언트 세션 초기화
            self._client_cm = ClientSession(self._read, self._write)
            self._client = await self._client_cm.__aenter__()
            
            # MCP 프로토콜 초기화
            await self._client.initialize()
            
            # 클래스 변수에 세션 저장
            MCPClientManager._session = self._client
            
            logger.info("✅ MCP client session initialized successfully")
            return self._client
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP client: {e}")
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 리소스 정리"""
        logger.info("🔌 Shutting down MCP client...")
        
        # 세션 정리
        MCPClientManager._session = None
        
        # 클라이언트 세션 종료
        if self._client_cm:
            try:
                await self._client_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error closing client session: {e}")
        
        # stdio 연결 종료
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error closing stdio session: {e}")
        
        logger.info("✅ MCP client shutdown completed")

    @classmethod
    def get_session(cls) -> Optional[ClientSession]:
        """현재 활성화된 세션 반환"""
        return cls._session

# --- MCP 도구 래퍼 함수들 ---

async def _call_mcp_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 도구 호출을 위한 공통 함수"""
    session = MCPClientManager.get_session()
    if not session:
        raise ConnectionError("MCP client session is not available. Please initialize MCPClientManager first.")
    
    logger.debug(f"🔧 Calling MCP tool '{tool_name}' with parameters: {parameters}")
    
    try:
        result = await session.call_tool(tool_name, parameters)
        
        if result.content and hasattr(result.content[0], 'text'):
            data = json.loads(result.content[0].text)
            logger.debug(f"✅ Tool '{tool_name}' executed successfully")
            return data
        else:
            logger.warning(f"⚠️  Tool '{tool_name}' returned empty result")
            return {"error": "도구 실행 결과가 없습니다"}
            
    except Exception as e:
        logger.error(f"❌ Error calling tool '{tool_name}': {e}")
        return {"error": f"도구 실행 중 오류가 발생했습니다: {str(e)}"}

async def list_applicants() -> str:
    """전체 신청자 목록을 조회합니다."""
    result = await _call_mcp_tool("list_applicants", {})
    return json.dumps(result, indent=2, ensure_ascii=False)

async def get_applicant_information(applicant_id: str) -> str:
    """
    주어진 지원자 ID에 해당하는 지원자의 상세 정보를 조회합니다.
    (소득, 근무 연수, 신용 점수, 기존 부채, 요청 금액 포함)
    """
    result = await _call_mcp_tool("get_applicant_information", {"applicant_id": applicant_id})
    return json.dumps(result, indent=2, ensure_ascii=False)

async def evaluate_loan_application(applicant_id: str) -> str:
    """
    주어진 지원자 ID에 대해 대출 신청 심사를 수행하고,
    그 결과(결정, 점수, 사유)를 반환합니다.
    """
    result = await _call_mcp_tool("evaluate_loan_application", {"applicant_id": applicant_id})
    return json.dumps(result, indent=2, ensure_ascii=False)

async def report_email(applicant_id: str) -> str:
    """
    주어진 지원자 ID에 대해 대출 신청 심사를 수행하고,
    그 결과(결정, 점수, 사유)를 반환합니다.
    """
    result = await _call_mcp_tool("report_email", {"applicant_id": applicant_id})
    return json.dumps(result, indent=2, ensure_ascii=False)

# --- LangChain 도구 생성 ---

list_applicants_tool = StructuredTool.from_function(
    coroutine=list_applicants,
    name="list_applicants",
    description="전체 신청자 목록을 조회합니다."
)

get_applicant_information_tool = StructuredTool.from_function(
    coroutine=get_applicant_information,
    name="get_applicant_information",
    description="주어진 지원자 ID(applicant_id)에 해당하는 지원자의 상세 정보를 조회합니다."
)

evaluate_loan_application_tool = StructuredTool.from_function(
    coroutine=evaluate_loan_application,
    name="evaluate_loan_application", 
    description="주어진 지원자 ID(applicant_id)에 대해 대출 신청 심사를 수행하고 결과를 반환합니다."
)

report_email_tool = StructuredTool.from_function(
    coroutine=report_email,
    name="report_email",
    description="주어진 지원자 ID(applicant_id)에 대해 대출 신청 심사 결과에 대한 정보를 요약하고 이메일 알림 결과를 반환합니다."
)
# 도구 목록 (편의를 위한 그룹화)
ALL_TOOLS = [
    #list_applicants_tool,
    get_applicant_information_tool,
    evaluate_loan_application_tool,
    report_email_tool
]

DATA_TOOL = [get_applicant_information_tool]
EVALUATION_TOOL = [evaluate_loan_application_tool]
SEND_EMAIL_TOOL = [report_email_tool]