import ast
import logging
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base.agent_base import AgentBase, BaseAgentConfig
from agents.registry.agent_registry import AgentRegistry
from core.llm.llm_manger import LLMManager

from mcp_host.mcp_client import mcp_client 

logger = logging.getLogger("agent_system")


def remove_think_tags(text: str) -> str:
    """<think> 태그 제거"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@AgentRegistry.register("user_registration")
class UserRegistrationAgent(AgentBase):
    """
    사용자 등록 자동화 Agent

    입력:
        - query: str (사용자 입력)
    
    출력:
        - tool_result: MCP tool 호출 결과
    """

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        self.client = mcp_client # 임시 테스트용
        self.llm = LLMManager.get_llm(provider=getattr(config, "provider", "ollama"),
                                      model=config.model_name)

    def validate_input(self, state: Dict[str, Any]) -> bool:
        """state에 'messages' 존재 확인"""
        messages = state.get("messages")
        if not messages or not isinstance(messages, list):
            logger.error(f"[{self.name}] ❌ 'messages' must be a non-empty list")
            return False
        if not any(isinstance(m, HumanMessage) for m in messages):
            logger.error(f"[{self.name}] ❌ No HumanMessage in messages")
            return False
        return True


    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        logger.info(f"[{self.name}] Step 1 - incoming messages: {messages}")

        user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if user_message is None:
            logger.warning(f"[{self.name}] No HumanMessage found in messages")
            return {"messages": messages, "tool_result": "No human message found."}

        logger.info(f"[{self.name}] Step 2 - user_message content: {user_message.content}")

        async with self.client:
            # 1️⃣ MCP 서버 툴 조회
            try:
                tools = await self.client.list_tools()
                logger.info(f"mcp tools 목록 : {tools}")
                tool_names = [t.name for t in tools]
                logger.info(f"[{self.name}] Step 3 - available tools: {tool_names}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Failed to list tools: {e}")
                return {"messages": messages, "tool_result": f"Client error: {e}"}

            # 2️⃣ LLM 호출
            system_prompt = SystemMessage(content=f"""
    당신은 사용자 등록을 담당하는 MCP 에이전트입니다.
    사용자 메시지를 분석하여 적합한 MCP 툴을 선택하고 필요한 argument를 JSON 형식으로 생성하세요.
    툴 목록: {tool_names}
    """)
            llm_messages = [system_prompt, user_message]
            logger.info(f"[{self.name}] Step 4 - sending to LLM: {[m.content for m in llm_messages]}")

            response = await self.llm.ainvoke(llm_messages)
            logger.info(f"[{self.name}] Step 5 - raw LLM response: {response.content}")

            # 3️⃣ LLM 출력 파싱
            import json
            try:
                llm_output = json.loads(response.content)
                tool_name = llm_output["tool_name"]
                tool_args = llm_output.get("arguments", {})
                logger.info(f"[{self.name}] Step 6 - parsed LLM output: tool_name={tool_name}, tool_args={tool_args}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ LLM parse error: {e}")
                return {"messages": messages, "tool_result": f"LLM parse error: {e}"}

            # 4️⃣ MCP tool 호출
            try:
                result = await self.client.call_tool(tool_name, tool_args)
                logger.info(f"[{self.name}] Step 7 - tool result: {result}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Tool call error: {e}")
                return {"messages": messages, "tool_result": f"Tool call error: {e}"}

        state["last_tool_result"] = result
        # 5️⃣ 최종 messages에 tool 결과 추가
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=f"Tool Result: {result}"))
        logger.info(f"[{self.name}] Step 8 - final messages: {messages}")

        return {"messages": messages, "tool_result": result}




