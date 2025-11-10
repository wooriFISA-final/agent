"""
Agent 서버에서 MCP 서버에 연결하는 Client 예시
Streamable HTTP 방식으로 MCP 서버에 연결
"""
import asyncio
import json
from fastmcp.client import Client  # FastMCPClient / Client 사용


MCP_SERVER_URL = "http://127.0.0.1:8000/"  # streamable-http 서버 URL


async def test_mcp_connection():
    """MCP 서버에 연결하고 tool들을 테스트합니다."""

    # ❗ 컨텍스트 매니저 사용
    async with Client(MCP_SERVER_URL) as client:
        await client.ping()
        print("✅ MCP 서버 연결 성공!")

        # ==========================================
        # 1. 사용자 생성
        # ==========================================
        print("\n📝 사용자 생성 테스트...")
        create_result = await client.call_tool(
            "create_user",
            arguments={
                "name": "김성욱",
                "email": "sungwook@example.com",
                "age": 25,
                "phone": "010-1234-5678"
            }
        )
        print(f"결과: {create_result.content[0].text}")

        # ==========================================
        # 2. 또 다른 사용자 생성
        # ==========================================
        print("\n📝 두 번째 사용자 생성...")
        create_result2 = await client.call_tool(
            "create_user",
            arguments={
                "name": "홍길동",
                "email": "hong@example.com",
                "age": 30
            }
        )
        print(f"결과: {create_result2.content[0].text}")

        # ==========================================
        # 3. 사용자 목록 조회
        # ==========================================
        print("\n📋 사용자 목록 조회...")
        list_result = await client.call_tool(
            "list_users",
            arguments={"limit": 10}
        )
        print(f"결과: {list_result.content[0].text}")

        # ==========================================
        # 4. 특정 사용자 조회
        # ==========================================
        print("\n🔍 특정 사용자 조회...")
        get_result = await client.call_tool(
            "get_user",
            arguments={"user_id": "user_1"}
        )
        print(f"결과: {get_result.content[0].text}")

        # ==========================================
        # 5. 사용자 검색
        # ==========================================
        print("\n🔎 사용자 검색 (이름으로)...")
        search_result = await client.call_tool(
            "search_users",
            arguments={"query": "김", "field": "name"}
        )
        print(f"결과: {search_result.content[0].text}")

        # ==========================================
        # 6. 사용자 수정
        # ==========================================
        print("\n✏️ 사용자 정보 수정...")
        update_result = await client.call_tool(
            "update_user",
            arguments={
                "user_id": "user_1",
                "age": 26,
                "phone": "010-9999-8888"
            }
        )
        print(f"결과: {update_result.content[0].text}")

        # ==========================================
        # 7. Resource 읽기 (통계)
        # ==========================================
        print("\n📊 사용자 통계 리소스 읽기...")
        stats_resource = await client.read_resource("user://database/stats")
        print(f"통계:\n{stats_resource}")

        # ==========================================
        # 8. Prompt 사용
        # ==========================================
        print("\n💬 Prompt 가져오기...")
        greeting_prompt = await client.get_prompt(
            "user_greeting",
            arguments={"user_name": "김성욱"}
        )
        print(f"프롬프트:\n{greeting_prompt.messages[0].content.text}")

        # ==========================================
        # 9. 사용자 삭제
        # ==========================================
        print("\n🗑️ 사용자 삭제...")
        delete_result = await client.call_tool(
            "delete_user",
            arguments={"user_id": "user_2"}
        )
        print(f"결과: {delete_result.content[0].text}")

        print("\n✨ 모든 테스트 완료!")

        await client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MCP Client 테스트 시작")
    print("=" * 60)
    asyncio.run(test_mcp_connection())
