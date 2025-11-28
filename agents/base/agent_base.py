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
    DELEGATE = "delegate"  # ✅ 새로 추가: 다른 Agent로 위임


class AgentDecision:
    """Agent의 의사결정 결과"""
    def __init__(
        self,
        action: AgentAction,
        reasoning: str,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[Dict] = None,
        tool_use_id: Optional[str] = None,  # ⭐ Bedrock toolUseId
        next_agent: Optional[str] = None,
        response_text: Optional[str] = None  # ⭐ end_turn 시 LLM 응답
    ):
        self.action = action
        self.reasoning = reasoning
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {}
        self.tool_use_id = tool_use_id  # ⭐ 추가
        self.next_agent = next_agent
        self.response_text = response_text  # ⭐ 추가


class AgentBase(ABC):
    """
    멀티턴 Tool 호출을 지원하는 Agent 베이스 클래스
    
    핵심 설계:
    - LLMHelper를 통한 Ollama Chat API 직접 호출
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
            # agents.yaml 설정이 있으면 우선 적용
            self.max_iterations = yaml_config.max_iterations
            self.config.max_retries = yaml_config.max_retries
            self.config.timeout = yaml_config.timeout
            self.config.tags = yaml_config.tags
            
            # LLM 설정 병합 (agents.yaml > BaseAgentConfig > 전역 설정)
            if yaml_config.llm_config:
                # agents.yaml의 llm_config를 우선 적용
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
            # agents.yaml 설정이 없으면 BaseAgentConfig 사용
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
        """LangChain 메시지를 딕셔너리로 변환
        
        Args:
            message: LangChain 메시지 객체
            
        Returns:
            Dict[str, Any]: {"role": role, "content": content, ...} 형태의 딕셔너리
        """
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, ToolMessage):
            # ToolMessage는 tool_call_id도 함께 전달해야 함
            result = {"role": "tool", "content": message.content}
            if hasattr(message, 'tool_call_id') and message.tool_call_id:
                result["tool_call_id"] = message.tool_call_id
            return result
        else:
            return {"role": "user", "content": str(message)}
    
    def _convert_messages_to_dict(self, messages: List) -> List[Dict[str, str]]:
        """메시지 리스트를 딕셔너리 리스트로 일괄 변환
        
        Args:
            messages: LangChain 메시지 리스트
            
        Returns:
            List[Dict[str, str]]: 변환된 딕셔너리 리스트
        """
        return [self._langchain_to_dict(msg) for msg in messages]
        
    # =============================
    # Message 포맷팅 및 LLM 호출 (Debug용)
    # =============================
    def _pretty_messages(self, messages: List) -> str:
        """LangChain 메시지 리스트를 JSON 문자열로 예쁘게 변환
        
        Args:
            messages: LangChain 메시지 리스트
            
        Returns:
            str: JSON 형태의 문자열
        """
        converted = self._convert_messages_to_dict(messages)
        return json.dumps(converted, ensure_ascii=False, indent=2)

    def _prepare_llm_params(
        self,
        use_agent_config: bool = True,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """LLM 호출 파라미터 준비
        
        Args:
            use_agent_config: Agent 설정 사용 여부
            stream: 스트리밍 모드
            format: 응답 포맷
            **kwargs: 추가 파라미터
            
        Returns:
            Dict[str, Any]: 준비된 LLM 파라미터
        """
        # Agent 설정 사용 여부에 따라 기본값 설정
        if use_agent_config:
            llm_params = {**self.llm_config, **kwargs}
        else:
            llm_params = {**kwargs}
        
        # stream 명시적 처리
        if stream is not None:
            llm_params["stream"] = stream
            
        return llm_params
    
    def _call_llm(
        self,
        messages: List,
        stream: Optional[bool] = None,
        **kwargs
    ) -> str:
        """LLM 호출 (동기 방식)
        
        우선순위:
        1. 메서드 호출 시 전달된 kwargs
        2. Agent별 llm_config
        3. 전역 설정 (LLMHelper 기본값)
        
        Args:
            messages: LangChain 메시지 리스트
            stream: 스트리밍 모드
            **kwargs: 추가 LLM 파라미터
            
        Returns:
            str: LLM 응답
        """
        llm_params = self._prepare_llm_params(
            use_agent_config=True,
            stream=stream,
            **kwargs
        )
        
        logger.debug(f"[{self.name}] LLM Call Parameters: {llm_params}")
        
        # LangChain 메시지를 딕셔너리로 변환
        formatted_messages = self._convert_messages_to_dict(messages)
        
        return LLMHelper.invoke_with_history(
            history=formatted_messages,
            **llm_params
        )
        


    # =============================
    # 상태 관리 헬퍼 메서드
    # =============================
    
    def _add_message_to_state(self, state: AgentState, message) -> AgentState:
        """상태에 메시지를 추가하고 global_messages 업데이트
        
        Args:
            state: 현재 상태
            message: 추가할 LangChain 메시지
            
        Returns:
            AgentState: 업데이트된 상태
        """
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
        
        # Agent 진입 시 iteration 초기화
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
        
        # global_messages 사용 (없으면 messages로 폴백)
        global_messages = state.get("global_messages", [])
        if not global_messages:
            global_messages = state.get("messages", [])
            state["global_messages"] = global_messages
        
        logger.info(f"[{self.name}] Global messages count: {len(global_messages)}")
        
        # 현재 에이전트의 역할을 SystemMessage로 맨 앞에 추가
        agent_role = self.get_agent_role_prompt()
        system_msg = SystemMessage(content=agent_role)
        
        # 맨 앞에 SystemMessage 삽입
        global_messages = [system_msg] + global_messages
        state["global_messages"] = global_messages
        
        logger.info(f"[{self.name}] ✅ Added agent role as SystemMessage at the beginning")
        
        # MCP 도구 목록 조회
        available_tools = await self._list_mcp_tools()
        logger.info(f"[{self.name}] MCP tools available: {len(available_tools)}")
                
        # Bedrock toolConfig로 변환
        bedrock_tool_config = self._convert_mcp_to_bedrock_toolspec(available_tools)
        if bedrock_tool_config:
            state["bedrock_tool_config"] = bedrock_tool_config
            logger.info(f"[{self.name}] ✅ Bedrock toolConfig created with {len(bedrock_tool_config['tools'])} tools")
        else:
            logger.warning(f"[{self.name}] ⚠️ No Bedrock toolConfig created")
        
        # ReAct Loop
        while not StateBuilder.is_max_iterations_reached(state):
            state = StateBuilder.increment_iteration(state)
            current_iteration = state.get("iteration", 0)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{self.name}] Iteration {current_iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # global_messages를 사용
            global_messages = state.get("global_messages", [])
            
            # Bedrock native tool calling: 1단계로 통합
            # _analyze_request 제거 → _make_decision에서 stopReason으로 판단
            try:
                logger.info("🤔 Making Decision (Bedrock native tool calling)\n" + self._pretty_messages(global_messages))
                decision = await self._make_decision(state, global_messages, available_tools)
                
                logger.info(f"🤔 Decision: {decision.action.value}")
                logger.info(f"   Reasoning: {decision.reasoning}")
                
            except Exception as e:
                logger.error(f"[{self.name}] Decision making failed: {e}")

                state = StateBuilder.add_error(state, e, self.name)
                break
            
            # Step 3: 액션 실행
            if decision.action == AgentAction.USE_TOOL:
                state = await self._execute_tool_action(state, decision)
                continue
            
            elif decision.action == AgentAction.DELEGATE:
                return await self._execute_delegate_action(state, decision)
                
            elif decision.action == AgentAction.RESPOND:
                return await self._execute_respond_action(state, global_messages, available_tools, decision)
        
        # 최대 반복 횟수 도달
        return await self._handle_max_iterations(state, global_messages)
    
    # =============================
    # 액션 실행 메서드
    # =============================
    
    async def _execute_tool_action(
        self,
        state: AgentState,
        decision: AgentDecision
    ) -> AgentState:
        """Tool 실행 액션 처리
        
        Args:
            state: 현재 상태
            decision: Agent 의사결정 결과
            
        Returns:
            AgentState: 업데이트된 상태
        """
        logger.info(f"🔧 Executing tool: {decision.tool_name}")
        logger.info(f"   Arguments: {decision.tool_arguments}")
        
        try:
            tool_result = await self._execute_mcp_tool(
                decision.tool_name,
                decision.tool_arguments
            )
            
            state = StateBuilder.add_tool_call(
                state,
                tool_name=decision.tool_name,
                arguments=decision.tool_arguments,
                result=tool_result
            )
            
            # Tool 결과를 JSON으로 직렬화
            try:
                import json
                if isinstance(tool_result, dict):
                    result_content = json.dumps(tool_result)
                else:
                    result_content = str(tool_result)
            except:
                result_content = str(tool_result)
            
            # Bedrock native tool calling인지 확인
            if decision.tool_use_id:
                # Bedrock native: ToolMessage (toolResult 블록)
                tool_message = ToolMessage(
                    content=result_content,
                    tool_call_id=decision.tool_use_id
                )
                logger.debug(f"[{self.name}] Using Bedrock toolResult format (toolUseId: {decision.tool_use_id})")
            else:
                # JSON fallback: 일반 AIMessage
                tool_message = AIMessage(
                    content=f"Tool: {decision.tool_name}\nResult: {result_content}"
                )
                logger.debug(f"[{self.name}] Using text format (no toolUseId)")
            
            state = self._add_message_to_state(state, tool_message)
            logger.info(f"✅ Tool executed successfully")
            
        except Exception as e:
            logger.error(f"[{self.name}] Tool execution failed: {e}")
            state = StateBuilder.add_error(state, e, self.name)
            
            # 에러 처리도 동일하게
            if decision.tool_use_id:
                error_message = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=decision.tool_use_id
                )
            else:
                error_message = AIMessage(
                    content=f"Tool: {decision.tool_name}\nError: {str(e)}"
                )
            state = self._add_message_to_state(state, error_message)
        
        return state
    
    async def _execute_delegate_action(
        self,
        state: AgentState,
        decision: AgentDecision
    ) -> AgentState:
        """Delegate 액션 처리
        
        Args:
            state: 현재 상태
            decision: Agent 의사결정 결과
            
        Returns:
            AgentState: 업데이트된 상태
        """
        logger.info(f"🔀 Delegating to agent: {decision.next_agent}")
        logger.info(f"   Reason: {decision.reasoning}")
        
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
        """Respond 액션 처리
        
        Args:
            state: 현재 상태
            global_messages: 전역 메시지 리스트
            available_tools: 사용 가능한 도구 목록
            decision: Agent 의사결정 결과
            
        Returns:
            AgentState: 업데이트된 상태
        """
        logger.info("✅ Generating final response")
        
        try:
            # end_turn으로 응답이 전달됨
            final_response = decision.response_text
            
            if not final_response:
                logger.error(f"[{self.name}] No response_text in decision")
                raise ValueError("response_text is required for RESPOND action")
            
            logger.info(f"[{self.name}] Using response from end_turn (no additional LLM call)")
    
            state["last_result"] = final_response
            
            # 최종 응답도 global_messages에 추가
            state = self._add_message_to_state(state, AIMessage(content=final_response))
            
            state = StateBuilder.finalize_state(state, ExecutionStatus.SUCCESS)
            logger.info(f"[{self.name}] Total messages: {len(state['global_messages'])}")
            logger.info(f"💬 Final response generated ({len(final_response)} chars)")
            
        except Exception as e:
            logger.error(f"[{self.name}] Final response generation failed: {e}")
            state = StateBuilder.add_error(state, e, self.name)
            state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
        
        return state
    
    async def _handle_max_iterations(
        self,
        state: AgentState,
        global_messages: List
    ) -> AgentState:
        """최대 반복 횟수 도달 시 처리
        
        Args:
            state: 현재 상태
            global_messages: 전역 메시지 리스트
            
        Returns:
            AgentState: 업데이트된 상태
        """
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
        """Agent 의사결정 (Bedrock Native Tool Calling 전용)
        
        toolChoice="auto"를 사용하여 LLM이 필요할 때만 Tool을 선택합니다.
        JSON 파싱을 사용하지 않으므로 100% 안정적입니다.
        
        사용 가능한 Tool:
        - MCP Tools: 실제 데이터 조회
        - delegate_to_agent: 다른 에이전트에게 위임
        
        stopReason 처리:
        - tool_use: Tool 실행
        - end_turn: 최종 응답 생성
        
        Args:
            state: 현재 상태
            messages: LangChain 메시지 리스트
            available_tools: 사용 가능한 도구 목록
            
        Returns:
            AgentDecision: Agent 의사결정 결과
            
        Raises:
            Exception: 의사결정 실패 시
        """
        available_agents = self._get_available_agents()
        
        system_prompt = DECISION_PROMPT.format(
            name=self.name,
            available_agents=available_agents,
            available_tools=available_tools
        )
        
        try:
            logger.info(f"[{self.name}] 🤔 Making decision with Bedrock Native Tool Calling")
        
            messages.append(HumanMessage(content=system_prompt))
            state["global_messages"] = messages
            
            # State에서 bedrock_tool_config 가져오기
            bedrock_tool_config = state.get("bedrock_tool_config")
            
            if not bedrock_tool_config:
                raise Exception("bedrock_tool_config not found in state")
            
            # LLM 호출 - toolConfig 전달, toolChoice="auto"로 자동 선택
            formatted_messages = self._convert_messages_to_dict(messages)
            
            from core.llm.llm_manger import LLMHelper
            response = await asyncio.to_thread(
                LLMHelper.invoke_with_history,
                history=formatted_messages,
                tool_config=bedrock_tool_config,
                tool_choice={"auto": {}},  
                return_full_response=True,
                temperature=0.1,
                top_p=0.1
            )
            
            # stopReason 확인
            stop_reason = response.get("stopReason")
            logger.info(f"[{self.name}] stopReason: {stop_reason}")
            
            # end_turn: 최종 응답 준비 완료
            if stop_reason == "end_turn":
                # 응답 텍스트 추출
                message = response.get("output", {}).get("message", {})
                content_blocks = message.get("content", [])
                
                response_text = ""
                for block in content_blocks:
                    if "text" in block:
                        response_text = block["text"]
                        break
                
                logger.info(f"[{self.name}] ✅ LLM ready to provide final response (end_turn)")
                logger.info(f"[{self.name}] Response preview: {response_text[:100]}...")
                
                return AgentDecision(
                    action=AgentAction.RESPOND,
                    reasoning="LLM decided to respond directly without using tools",
                    response_text=response_text
                )
            
            # tool_use가 아니면 에러
            if stop_reason != "tool_use":
                logger.error(f"[{self.name}] Unexpected stopReason: {stop_reason}")
                raise Exception(f"Unexpected stopReason: '{stop_reason}'")
            
            # Tool 사용 정보 추출
            message = response["output"]["message"]
            content = message.get("content", [])

            # reasoningContent 블록 필터링 (Extended Thinking 모델용)
            # toolUse, text, image 등만 유지
            filtered_content = [
                block for block in content 
                if not isinstance(block, dict) or "reasoningContent" not in block
            ]
            
            # 필터링 후 content가 비어있으면 원본 사용
            if not filtered_content:
                filtered_content = content

            # assistant 응답을 히스토리에 추가
            # Bedrock native tool calling에서는 content를 그대로 유지해야 함
            # _prepare_bedrock_messages에서 이를 적절히 변환함
            messages.append(AIMessage(content=filtered_content))

            state["global_messages"] = messages
            
            # toolUse 블록 찾기 (필터링된 content 사용)
            for block in filtered_content:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name = tool_use["name"]
                    tool_input = tool_use.get("input", {})
                    tool_use_id = tool_use["toolUseId"]
                    
                    logger.info(f"[{self.name}] 🔧 Tool selected: {tool_name}")
                    logger.info(f"[{self.name}] 📋 Tool input: {tool_input}")
                    
                    # delegate_to_agent Tool 감지
                    if tool_name == "delegate_to_agent":
                        agent_name = tool_input.get("agent_name")
                        reason = tool_input.get("reason", "")
                        
                        logger.info(f"[{self.name}] 🔀 Delegating to: {agent_name}")
                        
                        return AgentDecision(
                            action=AgentAction.DELEGATE,
                            reasoning=reason,
                            next_agent=agent_name
                        )
                    
                    # finish_conversation Tool은 제거됨 (end_turn으로 대체)
                    
                    # 일반 MCP Tool
                    else:
                        return AgentDecision(
                            action=AgentAction.USE_TOOL,
                            reasoning="Bedrock native tool calling",
                            tool_name=tool_name,
                            tool_arguments=tool_input,
                            tool_use_id=tool_use_id
                        )
            
            # toolUse 블록을 찾지 못한 경우 (발생하지 않아야 함)
            logger.error(f"[{self.name}] No toolUse block found in response")
            raise Exception("No toolUse block found despite stopReason='tool_use'")
                
        except Exception as e:
            logger.error(f"[{self.name}] Decision making failed: {e}")
            raise
    
    async def _generate_fallback_response(self, messages: List) -> str:
        """최대 반복 횟수 도달 시 폴백 응답 생성
        
        Args:
            messages: LangChain 메시지 리스트
            
        Returns:
            str: 폴백 응답 텍스트
        """
        return f"""처리 과정이 예상보다 복잡하여 {self.max_iterations}회 반복 제한에 도달했습니다.
지금까지 수집한 정보를 바탕으로 답변드리겠습니다.

추가로 필요한 정보가 있다면 질문을 더 구체적으로 다시 해주시면 감사하겠습니다."""

    # =============================
    # 구체적인 Agent가 구현해야 할 메서드
    # =============================
    
    @abstractmethod
    def get_agent_role_prompt(self) -> str:
        """Agent 역할 정의 Prompt 반환
        
        각 Agent는 이 메서드를 구현하여 자신의 역할을 정의해야 합니다.
        
        Returns:
            str: Agent의 역할을 설명하는 프롬프트 텍스트
        """
        pass

    # =============================
    # 공통 헬퍼 메서드
    # =============================
    
    def _get_available_agents(self) -> str:
        """현재 Agent에서 위임 가능한 다른 Agent 목록 반환
        
        allowed_agents 속성이 있으면 해당 목록을 사용하고,
        없으면 등록된 모든 Agent를 사용합니다.
        자기 자신은 항상 목록에서 제외됩니다.
        
        Returns:
            str: 위임 가능한 Agent 목록을 포함하는 포맷팅된 텍스트
        """
        if hasattr(self, "allowed_agents"):
            # allowed_agents가 있어도 자기 자신은 무조건 제외해야 함
            agents = [name for name in self.allowed_agents if name != self.name]
        else:
            # 기본: 모든 등록된 Agent (자신 제외)
            from agents.registry.agent_registry import AgentRegistry
            all_agents = AgentRegistry.list_agents()
            agents = [name for name in all_agents if name != self.name]
            
        logger.info(f"{agents} available for delegation from {self.name}")
        
        if not agents:
            return f"""없음 (이 에이전트가 모든 작업을 직접 처리해야 함)

**당신의 정체성: {self.name}**
**위임 불가: 자기 자신({self.name})에게는 절대 위임할 수 없습니다.**"""
        
        # 포맷팅
        agent_list = "\n".join([f"- {agent}" for agent in agents])
        
        return f"""
[위임 가능한 다른 Agent 목록]
{agent_list}

**당신의 정체성: {self.name}**
**주의:** 위 목록에 없는 Agent(특히 자기 자신인 {self.name})에게는 절대 위임할 수 없습니다.
"""
    
    def _get_available_agents_list(self) -> List[str]:
        """현재 Agent에서 위임 가능한 다른 Agent 목록을 리스트로 반환
        
        Bedrock toolSpec의 enum에 사용하기 위한 헬퍼 메서드
        
        Returns:
            List[str]: 위임 가능한 Agent 이름 리스트
        """
        if hasattr(self, "allowed_agents"):
            agents = [name for name in self.allowed_agents if name != self.name]
        else:
            from agents.registry.agent_registry import AgentRegistry
            all_agents = AgentRegistry.list_agents()
            agents = [name for name in all_agents if name != self.name]
        
        return agents
    
    async def _list_mcp_tools(self) -> List[Dict[str, Any]]:
        """MCP 도구 목록 조회 및 필터링
        
        allowed_tools 속성을 확인하여 허용된 도구만 반환합니다.
        - 'ALL': 모든 도구 허용
        - []: 도구 없음
        - [도구명 목록]: 해당 도구만 허용
        
        Returns:
            List[Dict[str, Any]]: 도구 명세 리스트 (function call 형식)
        """
        try:
            tools = await self.mcp.list_tools()
            tools_spec = []
            
            if hasattr(self, "allowed_tools"):
                if self.allowed_tools == 'ALL':
                    pass  # 전체 툴 허용
                elif len(self.allowed_tools) == 0:
                    tools = []  # 툴 없음
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
        """
        MCP tool spec을 Bedrock toolConfig 형식으로 변환하고,
        delegate_to_agent와 finish_conversation Tool 추가
        
        MCP는 OpenAI function call 형식:
        {
            "type": "function",
            "function": {
                "name": "get_user",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        
        Bedrock는 toolSpec 형식:
        {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_user",
                        "description": "...",
                        "inputSchema": {
                            "json": {...}  # MCP parameters 그대로
                        }
                    }
                }
            ]
        }
        
        Args:
            mcp_tools: _list_mcp_tools()에서 반환된 tool 목록
            
        Returns:
            Bedrock toolConfig 딕셔너리
        """
        bedrock_tools = []
        
        # 1. MCP Tools 변환
        if mcp_tools:
            for tool in mcp_tools:
                func = tool.get("function", {})
                params = func.get("parameters", {})
                
                # description이 비어있으면 안 되므로 기본값 제공
                description = func.get("description", "").strip()
                if not description:
                    description = f"MCP tool: {func.get('name', 'unknown')}"
                
                bedrock_tools.append({
                    "toolSpec": {
                        "name": func.get("name"),
                        "description": description,
                        "inputSchema": {
                            "json": params
                        }
                    }
                })
        
        # 2. delegate_to_agent Tool 추가
        available_agents = self._get_available_agents_list()
        if available_agents:
            bedrock_tools.append({
                "toolSpec": {
                    "name": "delegate_to_agent",
                    "description": "다른 에이전트에게 작업을 위임합니다. 현재 에이전트가 처리할 수 없거나 다른 에이전트의 전문성이 필요한 경우 사용합니다.",
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
        
        logger.info(f"[{self.name}] ✅ Created Bedrock toolConfig: {len(bedrock_tools)} tools (MCP: {len(mcp_tools) if mcp_tools else 0}, delegate: {1 if available_agents else 0})")
        
        return {
            "tools": bedrock_tools
        }

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Any:
        """MCP 도구 실행
        
        Args:
            tool_name: 실행할 도구 이름
            tool_args: 도구 인자 딕셔너리
            
        Returns:
            Any: 도구 실행 결과
            
        Raises:
            Exception: 도구 실행 실패 시
        """
        try:
            result = await self.mcp.call_tool(tool_name, tool_args)
            logger.info(f"[{self.name}] Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Tool '{tool_name}' execution failed: {e}")
            raise
    
    def _remove_think_tag(self, text: str) -> str:
        """</think> 태그 제거 및 JSON 추출
        
        LLM 응답에서 <think> 태그를 제거하고 순수한 JSON만 추출합니다.
        
        Args:
            text: 원본 텍스트
            
        Returns:
            str: 태그가 제거된 깨끗한 텍스트
        """
        # 1. </think>가 있다면, 그 뒤의 내용만 취합니다.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        
        # 2. 혹시라도 <think>만 있고 닫는 태그가 없는 경우를 대비해 안전장치로 시작 태그 처리
        elif "<think>" in text:
            text = text.rsplit("<think>", 1)[-1]

        # 3. 앞뒤 공백 제거
        text = text.strip()
        
        # 4. 순수한 JSON 객체만 추출 (첫 '{' 부터 마지막 '}' 까지)
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