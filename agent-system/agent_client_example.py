import asyncio
import logging
from mcp_host.mcp_client import MCPHTTPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp_client():
    """FastMCP 클라이언트 기능 테스트"""
    
    logger.info("🚀 FastMCP 클라이언트 테스트를 시작합니다...")
    
    try:
        async with MCPHTTPClient(base_url="http://localhost:8000/mcp") as client:
            
            # 1. Tool 테스트: create_user
            logger.info("\n📝 Test 1: create_user 호출")
            result = await client.call_tool(
                "create_user",
                {
                    "name": "홍길동",
                    "email": "hong@example.com",
                    "age": 30,
                    "phone": "010-1234-5678"
                }
            )
            logger.info(f"결과: {result}")
            
            # 2. Tool 테스트: get_user
            if result.get("success"):
                user_id = result["user"]["id"]
                logger.info(f"\n🔍 Test 2: get_user 호출 (user_id: {user_id})")
                get_result = await client.call_tool("get_user", {"user_id": user_id})
                logger.info(f"결과: {get_result}")
            
            # 3. Tool 테스트: list_users
            logger.info("\n📋 Test 3: list_users 호출")
            list_result = await client.call_tool("list_users", {"limit": 10, "offset": 0})
            logger.info(f"결과: {list_result}")
            
            # 4. Resource 테스트: user_stats
            logger.info("\n📊 Test 4: user_stats 리소스 조회")
            stats = await client.get_resource("user://database/stats")
            logger.info(f"결과: {stats}")
            
            # 5. Prompt 테스트: user_greeting
            logger.info("\n👋 Test 5: user_greeting 프롬프트 호출")
            greeting = await client.call_prompt("user_greeting", {"user_name": "홍길동"})
            logger.info(f"결과: {greeting}")
            
            logger.info("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
            
    except RuntimeError as e:
        logger.error(f"❌ 클라이언트 런타임 에러: {e}")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 에러: {e}", exc_info=True)
    
    logger.info("\n🏁 클라이언트 테스트를 종료합니다.")


if __name__ == "__main__":
    asyncio.run(test_mcp_client())