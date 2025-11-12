from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import uuid
import json
import asyncio
import requests
from typing import Any, Dict, List

# === MCP 클라이언트 객체 정의 ===
transport = StreamableHttpTransport(
    url="http://localhost:8888/mcp/"
    #headers={"X-Account-Password": "1234"}
)
mcp_client = Client(transport)

# === MCP에서 도구 스펙 받아와서 Function calling 포맷으로 변환 ===
async def load_tools(client: Client) -> List[Dict[str, Any]]:
    tools = await client.list_tools()
    tools_spec = []
    for tool in tools:
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        if not props:
            continue
        tools_spec.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": schema.get("type", "object"),
                    "properties": {
                        k: {
                            "type": p.get("type", "string"),
                            "description": p.get("description", "")
                        } for k, p in props.items()
                    },
                    "required": schema.get("required", [])
                },
            },
        })
    return tools_spec


# === MCP 도구 실행 ===
async def call_mcp_tool(client: Client, name: str, args: Dict[str, Any]) -> Any:
    return await client.call_tool(name, args)

# # === 모델 호출 ===
# def call_llm(messages: List[Dict[str, Any]], tools_spec=None, tool_choice=None) -> Dict[str, Any]:
#     data = {"messages": messages, "maxTokens": 1024, "seed": 0}
#     if tools_spec:
#         data["tools"] = tools_spec
#     if tool_choice:
#         data["toolChoice"] = tool_choice
#     headers = MODEL_HEADERS | {"X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4())}
#     resp = requests.post(MODEL_URL, headers=headers, json=data)
#     resp.raise_for_status()
#     return resp.json()

# === main loop ===
# async def main():
#     async with mcp_client as client:
#         tools_spec = await load_tools(client)
#         system_prompt = {
#             "role": "system",
#             "content": (
#                 "당신은 사용자 주식 거래를 돕는 AI 어시스턴트입니다. "
#                 "매수·매도, 잔고 조회, 거래 내역 조회, 주가 조회를 처리하고 결과를 수치로 명확히 안내하세요. "
#                 "잔고·수량 부족 등 거래가 불가능하면 이유를 숫자와 함께 설명하세요."
#             ),
#         }

#         while True:
#             user_input = input("\n사용자 요청을 입력하세요: ")
#             if user_input.lower() in {"exit", "quit", "종료"}:
#                 print("\n대화를 종료합니다.")
#                 break

#             user_msg = {"role": "user", "content": user_input}
#             first_resp = call_llm([system_prompt, user_msg], tools_spec=tools_spec, tool_choice="auto")
#             if first_resp.get("status", {}).get("code") != "20000":
#                 print("\nLLM 호출 실패:", first_resp.get("status"))
#                 continue

#             assistant_msg = first_resp["result"]["message"]
#             tool_calls = assistant_msg.get("toolCalls", [])

#             if not tool_calls:
#                 print("\n모델 답변:", assistant_msg.get("content", ""))
#                 continue

#             tool_call = tool_calls[0]
#             func_name = tool_call["function"]["name"]
#             func_args = tool_call["function"]["arguments"]
#             call_id = tool_call["id"]

#             try:
#                 tool_result = await call_mcp_tool(client, func_name, func_args)
#             except Exception as err:
#                 print("\nMCP 도구 실행 실패:", err)
#                 continue


#             tool_response_prompt = {
#                 "role": "system",
#                 "content": (
#                     "아래 tool 결과를 기반으로 간결하게 최종 답변을 작성하세요. "
#                     "'available_cash'는 현재 남은 현금 잔고, 'portfolio'는 종목별 보유 수량과 평균 단가입니다. "
#                     "수치는 단위와 함께 명확하게 표현하세요. (예: 3주, 1,000원)\n"
#                     "금액 해석 시 숫자의 자릿수를 기준으로 정확히 구분하세요."
#                 ),
#             }

#             second_resp = call_llm(
#                 [
#                     tool_response_prompt,
#                     user_msg,
#                     {"role": "assistant", "content": "", "toolCalls": [tool_call]},
#                     {
#                         "role": "tool",
#                         "toolCallId": call_id,
#                         "name": func_name,
#                         "content": json.dumps(tool_result.structured_content, ensure_ascii=False),
#                     },
#                 ]
#             )

#             if second_resp.get("status", {}).get("code") == "20000":
#                 print("\n모델 답변:", second_resp["result"]["message"]["content"])
#             else:
#                 print("\nLLM 호출 실패:", second_resp.get("status"))