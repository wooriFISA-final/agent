from abc import ABC, abstractmethod
import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum

from agents.config.base_config import (
    BaseAgentConfig,
    AgentState,
    StateBuilder,
    StateValidator,
    ExecutionStatus
)

from agents.base.agent_base_prompts import DECISION_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from core.mcp.mcp_manager import MCPManager
from core.logging.logger import setup_logger
from core.llm.llm_manger import LLMHelper

logger = setup_logger()


# =============================
# Agent 관련 클래스
# =============================

class AgentAction(Enum):
    """Agent가 취할 수 있는 행동 타입"""
    USE_TOOL = "use_tool"
    RESPOND = "respond"
    DELEGATE = "delegate"


class AgentDecision:
    """Agent의 의사결정 결과"""
    def __init__(
        self,
        action: AgentAction,
        reasoning: str,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[Dict] = None,
        tool_use_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        next_agent: Optional[str] = None,
        response_text: Optional[str] = None,
        requires_post_processing: bool = False
    ):
        self.action = action
        self.reasoning = reasoning
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {}
        self.tool_use_id = tool_use_id
        self.tool_calls = tool_calls or []
        self.next_agent = next_agent
        self.response_text = response_text
        self.requires_post_processing = requires_post_processing


class AgentBase(ABC):
    """
    멀티턴 Tool 호출을 지원하는 Agent 베이스 클래스
    
    핵심 설계:
    - LLMHelper를 통한 Bedrock Converse API 직접 호출
    - LangChain 메시지는 LangGraph 호환을 위해 유지
    - LLM 호출 시에만 딕셔너리로 변환하여 사용
    - Agent별 LLM 설정 지원
    - DELEGATE 액션으로 다른 Agent에게 작업 위임 가능
    """

    def __init__(self, config: BaseAgentConfig):
        self.name = config.name
        self.config = config
        self.mcp = MCPManager().get_instance()
        
        # ✅ agents.yaml 설정 우선 적용
        from agents.config.agent_config_loader import AgentConfigLoader
        
        yaml_config = AgentConfigLoader.get_agent_config_from_current(self.name)
        
        if yaml_config:
            self.max_iterations = yaml_config.max_iterations
            self.config.max_retries = yaml_config.max_retries
            self.config.timeout = yaml_config.timeout
            self.config.tags = yaml_config.tags
            
            if yaml_config.llm_config:
                merged_llm = {**config.get_llm_config_dict(), **yaml_config.llm_config}
                self.llm_config = merged_llm
            else:
                self.llm_config = config.get_llm_config_dict()
            
            logger.info(f"[{self.name}] ✅ Applied agents.yaml config:")
            logger.info(f"   max_retries: {self.config.max_retries}")
            logger.info(f"   timeout: {self.config.timeout}")
            logger.info(f"   max_iterations: {self.max_iterations}")
            logger.info(f"   tags: {self.config.tags}")
        else:
            self.max_iterations = config.max_iterations
            self.llm_config = config.get_llm_config_dict()
            logger.info(f"[{self.name}] Using BaseAgentConfig defaults")
        
        logger.info(f"[{self.name}] Agent initialized")
        logger.info(f"[{self.name}] LLM config: {self.llm_config if self.llm_config else 'Using global settings'}")
        
        self._validate_config()

    # =============================
    # LLM 호출 헬퍼 메서드
    # =============================
    
    def _langchain_to_dict(self, message) -> Dict[str, Any]:
        """LangChain 메시지를 Bedrock 딕셔너리로 변환"""
        if isinstance(message, HumanMessage):
            if isinstance(message.content, list):
                return {"role": "user", "content": message.content}
            else:
                return {"role": "user", "content": [{"text": message.content}]}
        
        elif isinstance(message, AIMessage):
            from core.llm.llm_manger import _sanitize_extended_thinking_tokens
            
            if isinstance(message.content, list):
                sanitized_content = []
                for block in message.content:
                    if isinstance(block, dict) and "text" in block:
                        sanitized_block = block.copy()
                        sanitized_block["text"] = _sanitize_extended_thinking_tokens(block["text"])
                        sanitized_content.append(sanitized_block)
                    else:
                        sanitized_content.append(block)
                return {"role": "assistant", "content": sanitized_content}
            else:
                sanitized_text = _sanitize_extended_thinking_tokens(message.content)
                return {"role": "assistant", "content": [{"text": sanitized_text}]}
        
        elif isinstance(message, SystemMessage):
            return {"role": "user", "content": [{"text": f"[System] {message.content}"}]}
        
        elif isinstance(message, ToolMessage):
            logger.warning(f"[{self.name}] ToolMessage deprecated, use HumanMessage with toolResult")
            return {"role": "user", "content": [{"text": message.content}]}
        
        else:
            msg_type = type(message).__name__
            msg_attrs = {k: v for k, v in message.__dict__.items() if not k.startswith('_')}
            logger.warning(f"[{self.name}] ⚠️ Unknown message type: {msg_type}")
            logger.warning(f"[{self.name}]    Message attributes: {msg_attrs}")
            
            if hasattr(message, 'type'):
                logger.warning(f"[{self.name}]    Message.type: {message.type}")
            
            return {"role": "user", "content": [{"text": str(message)}]}
    
    def _convert_messages_to_dict(self, messages: List) -> List[Dict[str, str]]:
        """메시지 리스트를 딕셔너리 리스트로 일괄 변환"""
        return [self._langchain_to_dict(msg) for msg in messages]
        
    # =============================
    # Message 포맷팅 및 LLM 호출 (Debug용)
    # =============================
    def _pretty_messages(self, messages: List) -> str:
        """LangChain 메시지 리스트를 JSON 문자열로 예쁘게 변환"""
        converted = self._convert_messages_to_dict(messages)
        return json.dumps(converted, ensure_ascii=False, indent=2)

    def _prepare_llm_params(
        self,
        use_agent_config: bool = True,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """LLM 호출 파라미터 준비"""
        if use_agent_config:
            llm_params = {**self.llm_config, **kwargs}
        else:
            llm_params = {**kwargs}
        
        if stream is not None:
            llm_params["stream"] = stream
            
        return llm_params
    
    def _call_llm(
        self,
        messages: List,
        stream: Optional[bool] = None,
        **kwargs
    ) -> str:
        """LLM 호출 (동기 방식)"""
        llm_params = self._prepare_llm_params(
            use_agent_config=True,
            stream=stream,
            **kwargs
        )
        
        logger.debug(f"[{self.name}] LLM Call Parameters: {llm_params}")
        
        formatted_messages = self._convert_messages_to_dict(messages)
        
        return LLMHelper.invoke_with_history(
            history=formatted_messages,
            **llm_params
        )

    # =============================
    # 상태 관리 헬퍼 메서드
    # =============================
    
    def _add_message_to_state(self, state: AgentState, message) -> AgentState:
        """상태에 메시지를 추가하고 global_messages 업데이트"""
        global_messages = state.get("global_messages", [])
        global_messages.append(message)
        state["global_messages"] = global_messages
        return state

    # =============================
    # 멀티턴 실행 파이프라인
    # =============================
    
    async def run(self, state: AgentState) -> AgentState:
        """Agent 실행 메인 플로우"""
        self._log_start(state)

        if not self.validate_input(state):
            error = ValueError(f"Invalid input for {self.name}")
            state = StateBuilder.add_error(state, error, self.name)
            state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
            return state
        
        state["iteration"] = 0
        logger.info(f"[{self.name}] Iteration reset to 0 for this agent")

        state = self.pre_execute(state)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with asyncio.timeout(self.config.timeout):
                    result = await self.execute_multi_turn(state)
                break
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout} seconds"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                
                if attempt == self.config.max_retries:
                    error = TimeoutError(f"{self.name} execution timed out")
                    state = StateBuilder.add_error(state, error, self.name)
                    state = StateBuilder.finalize_state(state, ExecutionStatus.TIMEOUT)
                    return state
                
                await asyncio.sleep(1.5 * attempt)
                
            except Exception as e:
                logger.warning(f"[{self.name}] attempt {attempt} failed: {e}")
                
                if attempt == self.config.max_retries:
                    state = StateBuilder.add_error(state, e, self.name)
                    state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
                    return state
                
                await asyncio.sleep(1.5 * attempt)

        self._log_end(result)
        return result

    # =============================
    # 멀티턴 실행 로직 (ReAct Loop)
    # =============================
    
    async def execute_multi_turn(self, state: AgentState) -> AgentState:
        """멀티턴 실행 플로우 - global_messages 사용"""
        
        if state.get("status") == ExecutionStatus.RESPONDING:
            logger.info(f"[{self.name}] ⚙️ Re-entered for post-processing (status: RESPONDING)")
            state.pop("requires_post_processing", None)
            state["status"] = ExecutionStatus.RUNNING
            logger.info(f"[{self.name}] Status changed to RUNNING for post-processing")
        
        global_messages = state.get("global_messages", [])
        if not global_messages:
            global_messages = state.get("messages", [])
            state["global_messages"] = global_messages
        
        logger.info(f"[{self.name}] Global messages count: {len(global_messages)}")
        
        agent_role = self.get_agent_role_prompt()
        system_msg = SystemMessage(content=agent_role)
        
        global_messages = [system_msg] + global_messages
        state["global_messages"] = global_messages
        
        logger.info(f"[{self.name}] ✅ Added agent role as SystemMessage at the beginning")
        
        available_tools = await self._list_mcp_tools()
        logger.info(f"[{self.name}] MCP tools available: {len(available_tools)}")
                
        bedrock_tool_config = self._convert_mcp_to_bedrock_toolspec(available_tools)
        if bedrock_tool_config:
            state["bedrock_tool_config"] = bedrock_tool_config
            logger.info(f"[{self.name}] ✅ Bedrock toolConfig created with {len(bedrock_tool_config['tools'])} tools")
            tool_names = [t["toolSpec"]["name"] for t in bedrock_tool_config["tools"]]
        else:
            logger.warning(f"[{self.name}] ⚠️ No Bedrock toolConfig created")
            tool_names = []
        
        # ReAct Loop
        while not StateBuilder.is_max_iterations_reached(state):
            state = StateBuilder.increment_iteration(state)
            current_iteration = state.get("iteration", 0)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{self.name}] Iteration {current_iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            global_messages = state.get("global_messages", [])
            
            # ✅ 메시지 구조 검증 추가
            if not self._validate_message_structure(global_messages):
                logger.error(f"[{self.name}] ❌ Message structure validation failed")
                # 메시지 정규화 시도
                global_messages = self._normalize_messages(global_messages)
                state["global_messages"] = global_messages
                logger.info(f"[{self.name}] ✅ Messages normalized")
            
            try:
                logger.info("🤔 Making Decision (Bedrock native tool calling)\n")
                
                decision = await self._make_decision(state, global_messages, tool_names)
                
                logger.info(f"🤔 Decision: {decision.action.value}")
                logger.info(f"   Reasoning: {decision.reasoning}")
                
            except Exception as e:
                logger.error(f"[{self.name}] Decision making failed: {e}")
                state = StateBuilder.add_error(state, e, self.name)
                break
            
            if decision.action == AgentAction.USE_TOOL:
                state = await self._execute_tool_action(state, decision)
                continue
            
            elif decision.action == AgentAction.DELEGATE:
                return await self._execute_delegate_action(state, decision)
                
            elif decision.action == AgentAction.RESPOND:
                return await self._execute_respond_action(state, global_messages, available_tools, decision)
        
        return await self._handle_max_iterations(state, global_messages)
    
    # =============================
    # 액션 실행 메서드
    # =============================
    
    async def _execute_tool_action(
        self,
        state: AgentState,
        decision: AgentDecision
    ) -> AgentState:
        """Tool 실행 액션 처리 - 첫 번째만 실행하고 나머지는 메시지에서 제거"""
        
        tool_calls = decision.tool_calls if decision.tool_calls else [{
            "name": decision.tool_name,
            "arguments": decision.tool_arguments,
            "tool_use_id": decision.tool_use_id
        }]
        
        total_tools = len(tool_calls)
        logger.info(f"🔧 Total {total_tools} tool(s) requested")
        
        first_tool = tool_calls[0]
        logger.info(f"🔧 Executing tool 1/{total_tools}: {first_tool['name']}")
        logger.info(f"   Arguments: {first_tool['arguments']}")
        
        tool_results = []
        
        # 첫 번째 tool 실행
        try:
            tool_result = await self._execute_mcp_tool(
                first_tool["name"],
                first_tool["arguments"]
            )
            
            state = StateBuilder.add_tool_call(
                state,
                tool_name=first_tool["name"],
                arguments=first_tool["arguments"],
                result=tool_result
            )
            
            import json
            if isinstance(tool_result, dict):
                result_content = json.dumps(tool_result, ensure_ascii=False)
            else:
                result_content = str(tool_result)
            
            tool_results.append({
                "toolResult": {
                    "toolUseId": first_tool["tool_use_id"],
                    "content": [{"text": result_content}]
                }
            })
            
            logger.info(f"✅ Tool 1/{total_tools} executed successfully")
            
        except Exception as e:
            logger.error(f"[{self.name}] Tool execution failed: {e}")
            state = StateBuilder.add_error(state, e, self.name)
            
            tool_results.append({
                "toolResult": {
                    "toolUseId": first_tool["tool_use_id"],
                    "content": [{"text": f"Error: {str(e)}"}],
                    "status": "error"
                }
            })
        
        # ✅ 나머지 tool은 assistant 메시지에서 제거 (재구성)
        if total_tools > 1:
            logger.warning(f"⚠️ Removing {total_tools - 1} unused tool(s) from message history")
            
            global_messages = state.get("global_messages", [])
            if global_messages and isinstance(global_messages[-1].content, list):
                last_msg = global_messages[-1]
                
                # 첫 번째 toolUse만 남기기
                new_content = []
                tool_count = 0
                for block in last_msg.content:
                    if isinstance(block, dict) and "toolUse" in block:
                        tool_count += 1
                        if tool_count == 1:
                            new_content.append(block)
                    else:
                        new_content.append(block)
                
                # 메시지 업데이트
                last_msg.content = new_content
                global_messages[-1] = last_msg
                state["global_messages"] = global_messages
                
                logger.info(f"   Kept first toolUse, removed {total_tools - 1} toolUse block(s)")
        
        # ✅ 첫 번째 tool의 결과만 추가
        tool_result_message = HumanMessage(content=tool_results)
        state = self._add_message_to_state(state, tool_result_message)
        
        logger.info(f"✅ Tool execution completed: 1 executed, {total_tools - 1} removed")
        
        return state
    
    async def _execute_delegate_action(
        self,
        state: AgentState,
        decision: AgentDecision
    ) -> AgentState:
        """Delegate 액션 처리 - toolResult 추가"""
        logger.info(f"🔀 Delegating to agent: {decision.next_agent}")
        logger.info(f"   Reason: {decision.reasoning}")
        
        # ✅ delegate toolResult 추가 (Bedrock API 요구사항)
        tool_result = {
            "toolResult": {
                "toolUseId": decision.tool_use_id,
                "content": [{
                    "text": json.dumps({
                        "status": "delegated",
                        "next_agent": decision.next_agent,
                        "reason": decision.reasoning
                    }, ensure_ascii=False)
                }]
            }
        }
        
        # global_messages에 toolResult 추가
        tool_result_message = HumanMessage(content=[tool_result])
        state = self._add_message_to_state(state, tool_result_message)
        
        # delegation 메타데이터 설정
        state["previous_agent"] = self.name
        state["next_agent"] = decision.next_agent
        state["delegation_reason"] = decision.reasoning
        state["status"] = ExecutionStatus.RUNNING
        state["timestamp"] = datetime.now()
        
        global_messages = state.get("global_messages", [])
        logger.info(f"[{self.name}] Delegation: next_agent={state.get('next_agent')}, status={state.get('status')}")
        logger.info(f"[{self.name}] ✅ Full conversation history preserved ({len(global_messages)} messages)")
        
        return state

    async def _execute_respond_action(
        self,
        state: AgentState,
        global_messages: List,
        available_tools: List[str],
        decision: AgentDecision
    ) -> AgentState:
        """Respond 액션 처리"""
        logger.info("✅ Processing response action")
        
        try:
            if decision.requires_post_processing:
                # ✅ respond_intermediate toolResult 추가
                tool_result = {
                    "toolResult": {
                        "toolUseId": decision.tool_use_id,
                        "content": [{
                            "text": json.dumps({
                                "status": "intermediate",
                                "reason": decision.reasoning,
                                "message": "중간 단계 - 추가 작업 필요"
                            }, ensure_ascii=False)
                        }]
                    }
                }
                
                # global_messages에 toolResult 추가
                tool_result_message = HumanMessage(content=[tool_result])
                state = self._add_message_to_state(state, tool_result_message)
                
                state["status"] = ExecutionStatus.RESPONDING
                state["requires_post_processing"] = True
                logger.info(f"[{self.name}] ⚙️ Intermediate stage - RESPONDING (toolResult added)")
                logger.info(f"[{self.name}] Router will re-enter this agent for post-processing")
                logger.info(f"[{self.name}] Reason: {decision.reasoning}")
            else:
                final_response = decision.response_text
                
                if not final_response:
                    logger.error(f"[{self.name}] No response_text in decision")
                    raise ValueError("response_text is required for final RESPOND action")
                
                logger.info(f"[{self.name}] Response ready ({len(final_response)} chars)")
                
                state["last_result"] = final_response
                
                state = self._add_message_to_state(state, AIMessage(content=final_response))
                
                usage = state.get("usage", {})
                total_tokens = usage.get("totalTokens", 0)
                
                if total_tokens > 50000:
                    logger.warning(f"⚠️ Token limit approaching: {total_tokens}/128000 - Compressing history...")
                    state = await self._compress_conversation_history(state)
                else:
                    logger.info(f"📊 Token usage OK: {total_tokens}/128000")
                
                state = StateBuilder.finalize_state(state, ExecutionStatus.SUCCESS)
                logger.info(f"[{self.name}] ✅ Final response saved and finalized with SUCCESS")
            
            logger.info(f"[{self.name}] Total messages: {len(state.get('global_messages', []))}")
            logger.info(f"💬 Response action processed")
            
        except Exception as e:
            logger.error(f"[{self.name}] Response processing failed: {e}")
            state = StateBuilder.add_error(state, e, self.name)
            state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
        
        return state
    
    async def _handle_max_iterations(
        self,
        state: AgentState,
        global_messages: List
    ) -> AgentState:
        """최대 반복 횟수 도달 시 처리"""
        logger.warning(f"⚠️ Max iterations ({self.max_iterations}) reached")
        
        try:
            fallback_response = await self._generate_fallback_response(global_messages)
            state = self._add_message_to_state(state, AIMessage(content=fallback_response))
            state["last_result"] = fallback_response
        except Exception as e:
            logger.error(f"[{self.name}] Fallback response generation failed: {e}")
            state = StateBuilder.add_error(state, e, self.name)
        
        state = StateBuilder.finalize_state(state, ExecutionStatus.MAX_ITERATIONS)
        return state

    # =============================
    # Agent React Function 단계별 메서드
    # =============================
    async def _make_decision(
        self,
        state: AgentState,
        messages: List,
        available_tools: List[str],
    ) -> AgentDecision:
        available_agents = self._get_available_agents_list()
        user_id = state.get("user_id", "test_user_1")
        
        if available_tools:
            tools_formatted = "\n".join([f"     - {tool}" for tool in available_tools])
        else:
            tools_formatted = "     - (없음)"
        
        system_prompt = DECISION_PROMPT.format(
            name=self.name,
            user_id=user_id,
            available_agents=available_agents,
            available_tools=tools_formatted
        )
        
        try:
            logger.info(f"[{self.name}] 🤔 Making decision with Bedrock Native Tool Calling")
        
            messages.append(HumanMessage(content=system_prompt))
            state["global_messages"] = messages
            
            bedrock_tool_config = state.get("bedrock_tool_config")
            if not bedrock_tool_config:
                raise Exception("bedrock_tool_config not found in state")
            
            formatted_messages = self._convert_messages_to_dict(messages)
            
            from core.llm.llm_manger import LLMHelper
            response = await asyncio.to_thread(
                LLMHelper.invoke_with_history,
                history=formatted_messages,
                tool_config=bedrock_tool_config,
                tool_choice={"auto": {}},
                return_full_response=True,
                temperature=0.01,
                top_p=0.01
            )
            
            stop_reason = response.get("stopReason")
            logger.info(f"[{self.name}] stopReason: {stop_reason}")
            
            usage = response.get("usage", {})
            state["usage"] = usage
            logger.info(f"📊 Token usage - Input: {usage.get('inputTokens', 0)}, Output: {usage.get('outputTokens', 0)}, Total: {usage.get('totalTokens', 0)}")
            
            # end_turn 처리
            if stop_reason == "end_turn":
                message = response.get("output", {}).get("message", {})
                content_blocks = message.get("content", [])
                
                response_text = ""
                for block in content_blocks:
                    if "text" in block:
                        response_text = block["text"]
                        break
                
                logger.info(f"[{self.name}] ✅ Final response via end_turn")
                
                messages.append(AIMessage(content=response_text))
                state["global_messages"] = messages
                
                return AgentDecision(
                    action=AgentAction.RESPOND,
                    reasoning="Final response without post-processing",
                    response_text=response_text,
                    requires_post_processing=False
                )
            
            if stop_reason != "tool_use":
                logger.error(f"[{self.name}] Unexpected stopReason: {stop_reason}")
                raise Exception(f"Unexpected stopReason: '{stop_reason}'")
            
            message = response["output"]["message"]
            content = message.get("content", [])

            # reasoningContent 필터링
            filtered_content = [
                block for block in content 
                if not isinstance(block, dict) or "reasoningContent" not in block
            ]
            
            # ✅ 빈 경우 빈 텍스트 블록 추가 (원본 복원 금지)
            if not filtered_content:
                logger.warning(f"[{self.name}] ⚠️ All content filtered out, adding empty text block")
                filtered_content = [{"text": ""}]

            # ✅ toolUse.name sanitize
            for block in filtered_content:
                if isinstance(block, dict) and "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name_raw = tool_use.get("name", "")
                    
                    tool_name_clean = tool_name_raw.split('<')[0].split('|')[0].strip()
                    tool_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '', tool_name_clean)
                    
                    if tool_name_clean != tool_name_raw:
                        logger.warning(f"[{self.name}] ⚠️ Sanitized toolUse.name in message: '{tool_name_raw}' → '{tool_name_clean}'")
                        tool_use["name"] = tool_name_clean

            messages.append(AIMessage(content=filtered_content))
            state["global_messages"] = messages
            
            # ✅ 모든 toolUse 블록 수집
            tool_calls = []
            for block in filtered_content:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name_raw = tool_use["name"]
                    tool_input = tool_use.get("input", {})
                    tool_use_id = tool_use["toolUseId"]
                    
                    tool_name = tool_name_raw.split('<')[0].split('|')[0].strip()
                    
                    if tool_name != tool_name_raw:
                        logger.warning(f"[{self.name}] ⚠️ Tool name sanitized: '{tool_name_raw}' → '{tool_name}'")
                    
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": tool_input,
                        "tool_use_id": tool_use_id
                    })
            
            if not tool_calls:
                logger.error(f"[{self.name}] No toolUse block found")
                raise Exception("No toolUse block found despite stopReason='tool_use'")
            
            logger.info(f"[{self.name}] Found {len(tool_calls)} tool call(s)")
            
            first_tool = tool_calls[0]
            
            logger.info(f"[{self.name}] 🔧 Primary tool: {first_tool['name']}")
            logger.info(f"[{self.name}] 📋 Tool input: {first_tool['arguments']}")
            
            # respond_intermediate
            if first_tool["name"] == "respond_intermediate":
                reason = first_tool["arguments"].get("reason", "Additional work required")
                logger.info(f"[{self.name}] ⚙️ Intermediate stage")
                
                return AgentDecision(
                    action=AgentAction.RESPOND,
                    reasoning=reason,
                    response_text="",
                    requires_post_processing=True,
                    tool_use_id=first_tool["tool_use_id"],
                    tool_calls=tool_calls
                )
            
            # delegate
            elif first_tool["name"] == "delegate":
                agent_name = first_tool["arguments"].get("agent_name")
                reason = first_tool["arguments"].get("reason", "")
                
                logger.info(f"[{self.name}] 🔀 Delegating to: {agent_name}")
                
                return AgentDecision(
                    action=AgentAction.DELEGATE,
                    reasoning=reason,
                    next_agent=agent_name,
                    tool_use_id=first_tool["tool_use_id"],
                    tool_calls=tool_calls
                )
            
            # 일반 MCP Tool
            else:
                return AgentDecision(
                    action=AgentAction.USE_TOOL,
                    reasoning="Bedrock native tool calling",
                    tool_name=first_tool["name"],
                    tool_arguments=first_tool["arguments"],
                    tool_use_id=first_tool["tool_use_id"],
                    tool_calls=tool_calls
                )
                
        except Exception as e:
            logger.error(f"[{self.name}] Decision making failed: {e}")
            raise
        
    async def _generate_fallback_response(self, messages: List) -> str:
        """최대 반복 횟수 도달 시 폴백 응답 생성"""
        return f"""처리 과정이 예상보다 복잡하여 {self.max_iterations}회 반복 제한에 도달했습니다.
지금까지 수집한 정보를 바탕으로 답변드리겠습니다.

추가로 필요한 정보가 있다면 질문을 더 구체적으로 다시 해주시면 감사하겠습니다."""

    async def _compress_conversation_history(self, state: AgentState) -> AgentState:
        """대화 히스토리 자동 압축 - toolUse/toolResult 쌍 보존"""
        messages = state.get("global_messages", [])
        
        if len(messages) <= 12:
            logger.info(f"[{self.name}] History short enough ({len(messages)} messages), skipping compression")
            return state
        
        logger.info(f"[{self.name}] 🗜️ Compressing conversation history...")
        logger.info(f"   Before: {len(messages)} messages")
        
        try:
            compressed_messages = self._compress_history_safely(messages)
            state["global_messages"] = compressed_messages
            
            logger.info(f"   After: {len(compressed_messages)} messages")
            logger.info(f"[{self.name}] ✅ History compressed successfully")
            
        except Exception as e:
            logger.error(f"[{self.name}] ❌ History compression failed: {e}")
        
        return state
    
    def _compress_history_safely(self, messages: List) -> List:
        """히스토리 압축 - toolUse/toolResult 쌍 보존"""
        if len(messages) <= 12:
            return messages
        
        compressed = []
        i = 0
        
        # 첫 메시지 보존
        compressed.append(messages[0])
        i = 1
        
        # 중간 부분 요약 (쌍을 유지하면서)
        middle_end = len(messages) - 10
        pairs_to_summarize = []
        
        while i < middle_end:
            msg = messages[i]
            
            # assistant + user (toolUse/toolResult) 쌍 감지
            if (isinstance(msg, AIMessage) and 
                i + 1 < len(messages) and
                isinstance(messages[i + 1], HumanMessage)):
                
                # toolUse 확인
                has_tool_use = any(
                    isinstance(block, dict) and "toolUse" in block
                    for block in (msg.content if isinstance(msg.content, list) else [])
                )
                
                if has_tool_use:
                    # 쌍으로 요약 대상에 추가
                    pairs_to_summarize.append((msg, messages[i + 1]))
                    i += 2
                    continue
            
            pairs_to_summarize.append((msg,))
            i += 1
        
        # 요약 생성
        summary_text = self._summarize_message_pairs(pairs_to_summarize)
        compressed.append(SystemMessage(content=f"[이전 대화 요약]\n{summary_text}"))
        
        # 최근 10개 보존
        compressed.extend(messages[-10:])
        
        return compressed
    
    def _summarize_message_pairs(self, pairs: List) -> str:
        """메시지 쌍 요약"""
        if not pairs:
            return "이전 대화 내용 없음"
        
        conversation_parts = []
        for pair in pairs:
            if len(pair) == 2:
                # toolUse/toolResult 쌍
                ai_msg, user_msg = pair
                conversation_parts.append(f"Tool 호출: {self._extract_tool_names(ai_msg)}")
            else:
                # 단일 메시지
                msg = pair[0]
                msg_type = msg.__class__.__name__
                content = str(msg.content)[:200] if not isinstance(msg.content, list) else "[구조화된 메시지]"
                conversation_parts.append(f"{msg_type}: {content}...")
        
        return "\n".join(conversation_parts[:20])  # 최대 20개만
    
    def _extract_tool_names(self, ai_message: AIMessage) -> str:
        """AIMessage에서 tool 이름 추출"""
        if not isinstance(ai_message.content, list):
            return "unknown"
        
        tool_names = []
        for block in ai_message.content:
            if isinstance(block, dict) and "toolUse" in block:
                tool_names.append(block["toolUse"].get("name", "unknown"))
        
        return ", ".join(tool_names) if tool_names else "unknown"
    
    async def _summarize_messages(self, messages: List) -> str:
        """메시지 목록을 LLM으로 요약"""
        if not messages:
            return "이전 대화 내용 없음"
        
        conversation_parts = []
        for msg in messages:
            msg_type = msg.__class__.__name__
            content = str(msg.content)
            
            if isinstance(msg.content, list):
                text_parts = []
                for block in msg.content:
                    if isinstance(block, dict):
                        if "text" in block:
                            text_parts.append(block["text"][:200])
                        elif "toolUse" in block:
                            tool_name = block["toolUse"].get("name", "unknown")
                            text_parts.append(f"[Tool: {tool_name}]")
                        elif "toolResult" in block:
                            text_parts.append("[Tool Result]")
                content = " ".join(text_parts)
            else:
                content = content[:200]
            
            conversation_parts.append(f"{msg_type}: {content}...")
        
        conversation_text = "\n".join(conversation_parts)
        
        prompt = f"""다음은 사용자와 AI 에이전트 간의 대화 내용입니다. 핵심 정보만 간결하게 요약해주세요.

{conversation_text}

요약 시 반드시 포함할 내용:
- 사용자가 요청한 주요 정보나 작업, 사용자가 선택한 상품 정보, 금액 등
- 에이전트가 수행한 주요 작업 (Tool 호출, 계산 등)
- 중요한 숫자나 데이터, 사용자 정보 (금액, 비율, 상품명 등)
- 현재까지의 진행 상황

300자 이내로 간결하게 요약:"""
        
        try:
            from core.llm.llm_manger import LLMHelper
            summary = await asyncio.to_thread(
                LLMHelper.invoke,
                prompt=prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Summarization failed: {e}")
            return f"이전 대화: {len(messages)}개 메시지 (사용자 요청 및 에이전트 응답 포함)"

    def _validate_message_structure(self, messages: List) -> bool:
        """메시지 구조 검증 - toolUse/toolResult 쌍 확인"""
        for i in range(len(messages) - 1):
            if not isinstance(messages[i], AIMessage):
                continue
            
            content = messages[i].content
            if not isinstance(content, list):
                continue
            
            # toolUse 개수 확인
            tool_uses = [
                block for block in content
                if isinstance(block, dict) and "toolUse" in block
            ]
            
            if not tool_uses:
                continue
            
            # 다음 메시지가 user인지 확인
            if i + 1 >= len(messages) or not isinstance(messages[i + 1], HumanMessage):
                logger.error(f"⚠️ toolUse without following user message at index {i}")
                return False
            
            # toolResult 개수 확인
            next_content = messages[i + 1].content
            if not isinstance(next_content, list):
                logger.error(f"⚠️ Invalid user message content at index {i + 1}")
                return False
            
            tool_results = [
                block for block in next_content
                if isinstance(block, dict) and "toolResult" in block
            ]
            
            if len(tool_uses) != len(tool_results):
                logger.error(
                    f"⚠️ Mismatch at index {i}: "
                    f"{len(tool_uses)} toolUse vs {len(tool_results)} toolResult"
                )
                return False
        
        return True
    
    def _normalize_messages(self, messages: List) -> List:
        normalized = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            
            # SystemMessage와 HumanMessage(일반)는 그대로 추가
            if isinstance(msg, SystemMessage):
                normalized.append(msg)
                i += 1
                continue
            
            if isinstance(msg, HumanMessage):
                # toolResult가 없는 일반 HumanMessage
                if not isinstance(msg.content, list):
                    normalized.append(msg)
                    i += 1
                    continue
                
                # toolResult 확인
                has_tool_result = any(
                    isinstance(block, dict) and "toolResult" in block
                    for block in msg.content
                )
                
                if not has_tool_result:
                    normalized.append(msg)
                    i += 1
                    continue
                
                # toolResult가 있는데 이전 메시지가 없거나 AIMessage가 아님
                if not normalized or not isinstance(normalized[-1], AIMessage):
                    logger.warning(f"⚠️ Orphaned toolResult at index {i} - removing")
                    i += 1
                    continue
                
                # 이전 AIMessage의 toolUse 확인
                prev_ai = normalized[-1]
                if not isinstance(prev_ai.content, list):
                    # 이전 AIMessage에 toolUse가 없음 - toolResult 제거
                    logger.warning(f"⚠️ toolResult without toolUse at index {i} - removing")
                    i += 1
                    continue
                
                tool_uses = [
                    block for block in prev_ai.content
                    if isinstance(block, dict) and "toolUse" in block
                ]
                
                if not tool_uses:
                    # 이전 AIMessage에 toolUse가 없음 - toolResult 제거
                    logger.warning(f"⚠️ toolResult without toolUse at index {i} - removing")
                    i += 1
                    continue
                
                # toolResult 개수 확인 및 조정
                tool_results = [
                    block for block in msg.content
                    if isinstance(block, dict) and "toolResult" in block
                ]
                
                if len(tool_uses) == len(tool_results):
                    # 정상 - 그대로 추가
                    normalized.append(msg)
                else:
                    # 불일치 - 조정
                    logger.warning(
                        f"⚠️ Adjusting toolResult count at index {i}: "
                        f"{len(tool_uses)} toolUse vs {len(tool_results)} toolResult"
                    )
                    
                    # toolUse 개수만큼 toolResult 유지
                    adjusted_results = tool_results[:len(tool_uses)]
                    
                    # 부족하면 빈 결과 추가
                    while len(adjusted_results) < len(tool_uses):
                        adjusted_results.append({
                            "toolResult": {
                                "toolUseId": tool_uses[len(adjusted_results)]["toolUse"]["toolUseId"],
                                "content": [{"text": "Normalized: Missing result"}]
                            }
                        })
                    
                    normalized.append(HumanMessage(content=adjusted_results))
                
                i += 1
                continue
            
            # AIMessage with toolUse 처리
            if isinstance(msg, AIMessage) and isinstance(msg.content, list):
                tool_uses = [
                    block for block in msg.content
                    if isinstance(block, dict) and "toolUse" in block
                ]
                
                if tool_uses:
                    # 다음 메시지 확인
                    if i + 1 < len(messages) and isinstance(messages[i + 1], HumanMessage):
                        next_content = messages[i + 1].content
                        
                        if isinstance(next_content, list):
                            tool_results = [
                                block for block in next_content
                                if isinstance(block, dict) and "toolResult" in block
                            ]
                            
                            # 쌍이 일치하면 그대로 추가
                            if len(tool_uses) == len(tool_results):
                                normalized.append(msg)
                                normalized.append(messages[i + 1])
                                i += 2
                                continue
                            else:
                                # 불일치 - toolUse 개수만큼 toolResult 조정
                                logger.warning(
                                    f"⚠️ Normalizing mismatch at index {i}: "
                                    f"{len(tool_uses)} toolUse vs {len(tool_results)} toolResult"
                                )
                                
                                # toolUse 개수만큼 toolResult 유지
                                adjusted_results = tool_results[:len(tool_uses)]
                                
                                # 부족하면 빈 결과 추가
                                while len(adjusted_results) < len(tool_uses):
                                    adjusted_results.append({
                                        "toolResult": {
                                            "toolUseId": tool_uses[len(adjusted_results)]["toolUse"]["toolUseId"],
                                            "content": [{"text": "Normalized: Missing result"}]
                                        }
                                    })
                                
                                normalized.append(msg)
                                normalized.append(HumanMessage(content=adjusted_results))
                                i += 2
                                continue
                    else:
                        # 다음 메시지가 없거나 HumanMessage가 아님 - toolUse 제거
                        logger.warning(f"⚠️ Removing orphaned toolUse at index {i}")
                        msg_copy = AIMessage(content=[
                            block for block in msg.content
                            if not (isinstance(block, dict) and "toolUse" in block)
                        ])
                        
                        if msg_copy.content:
                            normalized.append(msg_copy)
                        
                        i += 1
                        continue
            
            # 일반 AIMessage (toolUse 없음)는 그대로 추가
            normalized.append(msg)
            i += 1
        
        return normalized

    # =============================
    # 구체적인 Agent가 구현해야 할 메서드
    # =============================
    
    @abstractmethod
    def get_agent_role_prompt(self) -> str:
        """Agent 역할 정의 Prompt 반환"""
        pass

    # =============================
    # 공통 헬퍼 메서드
    # =============================
    
    def _get_available_agents(self) -> str:
        """현재 Agent에서 위임 가능한 다른 Agent 목록 반환"""
        if hasattr(self, "allowed_agents"):
            agents = [name for name in self.allowed_agents if name != self.name]
        else:
            from agents.registry.agent_registry import AgentRegistry
            all_agents = AgentRegistry.list_agents()
            agents = [name for name in all_agents if name != self.name]
            
        logger.info(f"{agents} available for delegation from {self.name}")
        
        if not agents:
            return f"""없음 (이 에이전트가 모든 작업을 직접 처리해야 함)

**당신의 정체성: {self.name}**
**위임 불가: 자기 자신({self.name})에게는 절대 위임할 수 없습니다.**"""
        
        agent_list = "\n".join([f"- {agent}" for agent in agents])
        
        return f"""
[위임 가능한 다른 Agent 목록]
{agent_list}

**당신의 정체성: {self.name}**
**주의:** 위 목록에 없는 Agent(특히 자기 자신인 {self.name})에게는 절대 위임할 수 없습니다.
"""
    
    def _get_available_agents_list(self) -> List[str]:
        """현재 Agent에서 위임 가능한 다른 Agent 목록을 리스트로 반환"""
        if hasattr(self, "allowed_agents"):
            agents = [name for name in self.allowed_agents if name != self.name]
        else:
            from agents.registry.agent_registry import AgentRegistry
            all_agents = AgentRegistry.list_agents()
            agents = [name for name in all_agents if name != self.name]
        
        return agents
    
    async def _list_mcp_tools(self) -> List[Dict[str, Any]]:
        """MCP 도구 목록 조회 및 필터링"""
        try:
            tools = await self.mcp.list_tools()
            tools_spec = []
            
            if hasattr(self, "allowed_tools"):
                if self.allowed_tools == 'ALL':
                    pass
                elif len(self.allowed_tools) == 0:
                    tools = []
                else:
                    tools = [t for t in tools if t.name in self.allowed_tools]

            for tool in tools:
                schema = tool.inputSchema or {}
                props = schema.get("properties", {})
                if not props:
                    continue
                tools_spec.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
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
            logger.debug(f"[{self.name}] Retrieved {len(tools_spec)} tools")
            return tools_spec
        except Exception as e:
            logger.error(f"[{self.name}] Failed to list MCP tools: {e}")
            return []
    
    def _convert_mcp_to_bedrock_toolspec(
        self,
        mcp_tools: List[Dict[str, Any]]
    ) -> Optional[Dict]:
        """MCP tool spec을 Bedrock toolConfig 형식으로 변환"""
        bedrock_tools = []
        
        from core.llm.llm_manger import _sanitize_extended_thinking_tokens
        
        # 1. MCP Tools 변환
        if mcp_tools:
            for tool in mcp_tools:
                func = tool.get("function", {})
                params = func.get("parameters", {})
                
                tool_name = _sanitize_extended_thinking_tokens(func.get("name", "")).strip()
                
                description = func.get("description", "").strip()
                description = _sanitize_extended_thinking_tokens(description)
                
                if not description:
                    description = f"MCP tool: {tool_name}"
                
                bedrock_tools.append({
                    "toolSpec": {
                        "name": tool_name,
                        "description": description,
                        "inputSchema": {
                            "json": params
                        }
                    }
                })
        
        # 2. respond_intermediate Tool 추가
        bedrock_tools.append({
            "toolSpec": {
                "name": "respond_intermediate",
                "description": """중간 단계 신호. 사용자에게 최종 응답을 제공하기 전에 추가 작업(DB 저장, 데이터 처리 등)이 더 필요한 경우 사용합니다.

사용 시나리오:
- 정보 수집 완료 → DB 저장 필요 → 저장 후 최종 응답
- 데이터 조회 완료 → 추가 계산 필요 → 계산 후 최종 응답
- 중간 결과 확인 → 검증 필요 → 검증 후 최종 응답

**중요**: 
- 이 Tool 사용 후 필요한 MCP Tool을 호출하여 작업을 완료하세요
- 모든 작업 완료 후 최종 응답은 별도로 생성해야 합니다 (end_turn)
- 이 Tool은 "아직 작업이 더 필요함"을 알리는 신호일 뿐, 사용자에게 직접 표시되지 않습니다""",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "추가 작업이 필요한 이유 (예: 'DB에 상담 내용 저장 필요', '포트폴리오 계산 후 응답 생성 필요')"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }
        })
        
        # 3. delegate Tool 추가
        available_agents = self._get_available_agents_list()
        if available_agents:
            bedrock_tools.append({
                "toolSpec": {
                    "name": "delegate",
                    "description": """
                    다른 에이전트에게 작업을 위임합니다. 현재 에이전트가 처리할 수 없거나 다른 에이전트의 전문성이 필요한 경우 사용합니다.
                    반드시, 현재 에이전트가 위임 가능한 agent를 delegate해야 합니다. 
[delegate agents]
1. plan_input_agent
   - 역할: 기본 정보 8가지 수집 및 검증
     * 초기 자본, 희망 지역, 희망 주택 가격, 희망 주택 유형, 소득 대비 사용 비율
     * 이름, 나이, 투자성향 (Tool로 조회)
   - 위임 시점:
     * 사용자 입력 정보가 들어온 경우
     * 8가지 정보 중 하나라도 없는 경우
     * 검증 실패한 정보가 있는 경우
     * 이름/나이/투자성향 정보 없는 경우

2. validation_agent
   - 역할: 기본 정보 6가지 검증
     * initial_prop, hope_location, hope_price, hope_housing_type, income_usage_ratio, ratio_str
   - 위임 시점:
     * 정보가 모였으나 검증 미완료
     * 검증 실패 후 재입력된 경우
     * 이미 검증이 되었으나 새로운 입력이 들어와 검증이 필요한 경우
     * 평균 시세 비교 및 포트폴리오 저장 필요시

3. loan_agent
   - 역할: 대출 한도, DSR/LTV, 상환 구조 계산
   - 위임 시점:
     * plan_input_agent 완료 후
     * 기본 정보 6가지 검증 완료
     * 대출 결과 없는 경우

4. saving_agent
   - 역할: 예·적금 저축 전략 설계
   - 위임 시점:
     * 사용자가 예금/적금 전략 요청
     * 대출 후 자기자본 부족
     * 예금/적금 상품 입력/선택/추천 요청

5. fund_agent
   - 역할: 펀드/투자 전략 제안
   - 위임 시점:
     * 추가 투자 수익 언급
     * '펀드', 'ETF', '투자', '수익률' 키워드 사용

6. summary_agent
   - 역할: 최종 주택 자금 계획 리포트 작성
   - 위임 시점:
     * 주요 단계 대부분 완료
     * '전체 요약', '최종 계획', '리포트', '정리' 요청
                    """,
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "agent_name": {
                                    "type": "string",
                                    "description": f"위임할 에이전트 이름. 가능한 에이전트: {', '.join(available_agents)}",
                                    "enum": available_agents
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "위임 이유 및 전달할 컨텍스트"
                                }
                            },
                            "required": ["agent_name", "reason"]
                        }
                    }
                }
            })
        
        logger.info(f"[{self.name}] ✅ Created Bedrock toolConfig: {len(bedrock_tools)} tools (MCP: {len(mcp_tools) if mcp_tools else 0}, respond_intermediate: 1, delegate: {1 if available_agents else 0})")
        
        return {
            "tools": bedrock_tools
        }

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Any:
        """MCP 도구 실행"""
        try:
            result = await self.mcp.call_tool(tool_name, tool_args)
            logger.info(f"[{self.name}] Tool '{tool_name}' Result : {result}")
            logger.info(f"[{self.name}] Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Tool '{tool_name}' execution failed: {e}")
            raise
    
    def _remove_think_tag(self, text: str) -> str:
        """</think> 태그 제거 및 JSON 추출"""
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        elif "<think>" in text:
            text = text.rsplit("<think>", 1)[-1]

        text = text.strip()
        
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx : end_idx + 1]
        
        return text

    # =============================
    # 기타 공통 메서드
    # =============================
    
    def validate_input(self, state: AgentState) -> bool:
        """입력 상태 검증"""
        if "messages" not in state or not isinstance(state["messages"], list):
            logger.error(f"[{self.name}] Invalid messages field")
            return False
        
        is_valid, error_msg = StateValidator.validate_execution_state(state)
        if not is_valid:
            logger.error(f"[{self.name}] Invalid execution state: {error_msg}")
            return False
        
        return True

    def pre_execute(self, state: AgentState) -> AgentState:
        """실행 전 전처리"""
        return state

    def _validate_config(self):
        """설정 검증"""
        if not self.config.name:
            raise ValueError("Agent name is required")

    def _log_start(self, state: AgentState):
        """실행 시작 로깅"""
        logger.info(f"[{self.name}] Starting execution")
        logger.info(f"   Session ID: {state.get('session_id', 'unknown')}")
        logger.info(f"   Messages: {len(state.get('messages', []))}")

    def _log_end(self, state: AgentState):
        """실행 완료 로깅"""
        logger.info(f"[{self.name}] Execution completed")
        logger.info(f"   Final Status: {state.get('status', 'unknown')}")
        logger.info(f"   Iterations: {state.get('iteration', 0)}")
        logger.info(f"   Tool Calls: {len(state.get('tool_calls', []))}")