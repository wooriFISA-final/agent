from abc import ABC, abstractmethod
import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum

from agent.config.base_config import (
    BaseAgentConfig,
    AgentState,
    StateBuilder,
    StateValidator,
    ExecutionStatus
)

from agent.base.agent_base_prompts import ANALYSIS_PROMPT, DECISION_PROMPT, FINAL_PROMPT

# ✅ LangGraph 호환을 위해 LangChain 메시지는 유지
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
        next_agent: Optional[str] = None  # ✅ 새로 추가: 위임할 Agent 이름
    ):
        self.action = action
        self.reasoning = reasoning
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {}
        self.next_agent = next_agent


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
        from agent.config.agent_config_loader import AgentConfigLoader
        
        yaml_config = AgentConfigLoader.get_agent_config(self.name)
        
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
            logger.warning(f"[{self.name}] ⚠️  No agents.yaml config found, using BaseAgentConfig defaults")
        
        logger.info(f"[{self.name}] Agent initialized")
        logger.info(f"[{self.name}] LLM config: {self.llm_config if self.llm_config else 'Using global settings'}")
        
        self._validate_config()

    # =============================
    # LLM 호출 헬퍼 메서드
    # =============================
    
    def _langchain_to_dict(self, message) -> Dict[str, str]:
        """LangChain 메시지를 딕셔너리로 변환"""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, ToolMessage):
            return {"role": "tool", "content": message.content}
        else:
            return {"role": "user", "content": str(message)}    
        
    # =============================
    # Message 포맷팅 및 LLM 호출 (Debug용)
    # =============================
    def _pretty_messages(self, messages: List) -> str:
        """LangChain 메시지 리스트를 JSON 문자열로 예쁘게 변환"""
        converted = []
        for msg in messages:
            converted.append(self._langchain_to_dict(msg))
        return json.dumps(converted, ensure_ascii=False, indent=2)

    def _call_llm(
        self,
        messages: List,
        system_prompt: Optional[str] = None,
        stream: Optional[bool] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        LLM 호출 (동기 방식)
        
        우선순위:
        1. 메서드 호출 시 전달된 kwargs
        2. Agent별 llm_config
        3. 전역 설정 (LLMHelper 기본값)
        """
        # Agent 설정과 kwargs 병합 (kwargs가 우선)
        llm_params = {**self.llm_config, **kwargs}
        
        # stream, format 명시적 처리
        if stream is not None:
            llm_params["stream"] = stream
        if format is not None:
            llm_params["format"] = format
        
        logger.debug(f"[{self.name}] LLM Call Parameters: {llm_params}")
        
        # LangChain 메시지를 딕셔너리로 변환
        formatted_messages = [self._langchain_to_dict(msg) for msg in messages]
        
        # system_prompt가 있으면 맨 앞에 추가
        if system_prompt:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 마지막 user 메시지를 prompt로, 나머지를 history로
        if not formatted_messages:
            return ""
        
        last_msg = formatted_messages[-1]
        history = formatted_messages[:-1] if len(formatted_messages) > 1 else []
        
        if last_msg["role"] == "user":
            return LLMHelper.invoke_with_history(
                prompt=last_msg["content"],
                history=history,
                system_prompt=None,  # 이미 history에 포함됨
                **llm_params
            )
        else:
            # 마지막이 user가 아니면 전체를 history로
            return LLMHelper.invoke_with_history(
                prompt="",
                history=formatted_messages,
                system_prompt=None,
                **llm_params
            )
    
    def _call_llm_with_fixed_params(
        self,
        messages: List,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        format: str = "",
        **fixed_kwargs
    ) -> str:
        """
        LLM 호출 (고정 파라미터)
        
        ⭐ 핵심: Agent llm_config를 무시하고 고정값만 사용
        
        이 메서드는 분석/의사결정 같이 정확성이 중요한 작업에 사용
        Agent별 설정을 무시하고 기본값만 따름
        
        우선순위:
        1. 이 메서드의 파라미터 (stream, format 고정)
        2. fixed_kwargs (기본값)
        3. 전역 설정 (LLMHelper 기본값)
        
        Args:
            messages: 메시지 리스트
            system_prompt: 시스템 프롬프트
            stream: 스트리밍 (기본: False=전체 응답)
            format: 포맷 (기본: ""=텍스트, "json"=JSON 강제)
            **fixed_kwargs: 고정 파라미터 (temperature 등)
        """
        # ⭐ Agent llm_config를 무시하고 fixed_kwargs만 사용
        llm_params = {**fixed_kwargs}  # Agent 설정 무시!
        
        # stream, format은 이 메서드의 파라미터 사용
        llm_params["stream"] = stream
        llm_params["format"] = format
        
        logger.debug(f"[{self.name}] LLM Call (FIXED PARAMS): {llm_params}")
        logger.info(f"[{self.name}] Using fixed parameters (ignoring Agent config)")
        
        # LangChain 메시지를 딕셔너리로 변환
        formatted_messages = [self._langchain_to_dict(msg) for msg in messages]
        
        if system_prompt:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})
        
        if not formatted_messages:
            return ""
        
        last_msg = formatted_messages[-1]
        history = formatted_messages[:-1] if len(formatted_messages) > 1 else []
        
        if last_msg["role"] == "user":
            return LLMHelper.invoke_with_history(
                prompt=last_msg["content"],
                history=history,
                system_prompt=None,
                **llm_params
            )
        else:
            return LLMHelper.invoke_with_history(
                prompt="",
                history=formatted_messages,
                system_prompt=None,
                **llm_params
            )

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
        
        # ✅ Agent 진입 시 iteration 초기화
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
        """멀티턴 실행 플로우"""
        messages = state.get("messages", [])
        
        logger.info(f"[{self.name}] Messages count: {len(messages)}")
        
        # ✅ 매번 Agent 진입 시 역할 정의 추가 (전체 히스토리 유지)
        agent_role = self.get_agent_role_prompt()
        system_msg = SystemMessage(content=agent_role)
        
        # 맨 앞에 추가
        state["messages"] = [system_msg] + messages
        messages = state["messages"]
        
        logger.info(f"[{self.name}] ✅ Added agent role as system message")
        
        # MCP 도구 목록 조회
        available_tools = await self._list_mcp_tools()
        logger.info(f"[{self.name}] MCP tools available: {len(available_tools)}")

        if not available_tools:
            error_msg = "No MCP tools available"
            logger.error(f"[{self.name}] {error_msg}")
            state = StateBuilder.add_warning(state, error_msg)
            state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
            return state
        
        logger.info(f"[{self.name}] Available tools: {available_tools}")
        
        # ReAct Loop
        while not StateBuilder.is_max_iterations_reached(state):
            state = StateBuilder.increment_iteration(state)
            current_iteration = state.get("iteration", 0)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{self.name}] Iteration {current_iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            # Step 1: 요구사항 분석
            try:
                logger.info("📋 Analyzing Input Message\n" + self._pretty_messages(messages))
                analyzed_request = await self._analyze_request(messages, available_tools)
                analyzed_request = self._remove_think_tag(analyzed_request)
                
                logger.info(f"📋 Analyzed Request: {analyzed_request}")
            except Exception as e:
                logger.error(f"[{self.name}] Request analysis failed: {e}")
                state = StateBuilder.add_error(state, e, self.name)
                break
            
            # Step 2: Agent 의사결정
            try:
                logger.info("📋 MakeDecision Input Message\n" + self._pretty_messages(messages))
                decision = await self._make_decision(messages, available_tools)
                
                logger.info(f"🤔 Decision: {decision.action.value}")
                logger.info(f"   Reasoning: {decision.reasoning}")
            except Exception as e:
                logger.error(f"[{self.name}] Decision making failed: {e}")
                state = StateBuilder.add_error(state, e, self.name)
                break
            
            # Step 2: 액션 실행
            if decision.action == AgentAction.USE_TOOL:
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
                    
                    tool_message = ToolMessage(
                        content=f"Tool: {decision.tool_name}\nResult: {tool_result}",
                        tool_call_id=decision.tool_name
                    )
                    messages.append(tool_message)
                    state["messages"] = messages
                    
                    logger.info(f"✅ Tool executed successfully")
                    
                except Exception as e:
                    logger.error(f"[{self.name}] Tool execution failed: {e}")
                    state = StateBuilder.add_error(state, e, self.name)
                    
                    error_message = ToolMessage(
                        content=f"Tool: {decision.tool_name}\nError: {str(e)}",
                        tool_call_id=decision.tool_name
                    )
                    messages.append(error_message)
                    state["messages"] = messages
                
                continue
            
            elif decision.action == AgentAction.DELEGATE:
                logger.info(f"🔀 Delegating to agent: {decision.next_agent}")
                logger.info(f"   Reason: {decision.reasoning}")
                
                # ✅ 메시지 초기화하지 않고 그대로 유지!
                delegation_msg = AIMessage(
                    content=f"[내부 위임] {decision.next_agent}에게 작업을 위임합니다.\n이유: {decision.reasoning}"
                )
                messages.append(delegation_msg)
                state["messages"] = messages
                
                # ✅ delegation 메타데이터 설정
                state["previous_agent"] = self.name
                state["next_agent"] = decision.next_agent
                state["delegation_reason"] = decision.reasoning
                state["status"] = ExecutionStatus.RUNNING
                state["timestamp"] = datetime.now()
                
                logger.info(f"[{self.name}] Delegation: next_agent={state.get('next_agent')}, status={state.get('status')}")
                logger.info(f"[{self.name}] ✅ Full conversation history preserved ({len(messages)} messages)")
                return state
                
            elif decision.action == AgentAction.RESPOND:
                logger.info("✅ Generating final response")
                
                try:
                    final_response = await self._generate_final_response(messages, available_tools)
                    
                    messages.append(AIMessage(content=final_response))
                    state["messages"] = messages
                    state["last_result"] = final_response
                    
                    state = StateBuilder.finalize_state(state, ExecutionStatus.SUCCESS)
                    logger.info(f"[{self.name}] Total messages: {len(state['messages'])}")
                    logger.info(f"💬 Final response generated ({len(final_response)} chars)")
                    return state
                    
                except Exception as e:
                    logger.error(f"[{self.name}] Final response generation failed: {e}")
                    state = StateBuilder.add_error(state, e, self.name)
                    state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
                    return state
        
        # 최대 반복 횟수 도달
        logger.warning(f"⚠️ Max iterations ({self.max_iterations}) reached")
        
        try:
            fallback_response = await self._generate_fallback_response(messages)
            messages.append(AIMessage(content=fallback_response))
            state["messages"] = messages
            state["last_result"] = fallback_response
        except Exception as e:
            logger.error(f"[{self.name}] Fallback response generation failed: {e}")
            state = StateBuilder.add_error(state, e, self.name)
        
        state = StateBuilder.finalize_state(state, ExecutionStatus.MAX_ITERATIONS)
        return state

    # =============================
    # Agent React Function 단계별 메서드
    # =============================
    
    async def _analyze_request(
        self,
        messages: List,
        available_tools: List[str]
    ) -> str:
        """
        요구사항 분석 (기본값 고정)
        
        ⭐ Agent 설정 무시, 항상 기본값 사용
        - temperature: 0.1 (매우 일관적)
        - format: "" (텍스트)
        - stream: False (전체 응답)
        """
        agent_role = self.get_agent_role_prompt()
        
        system_prompt = f"""{agent_role}

---
[현재 실행 중인 에이전트 ID]
**{self.name}** (당신입니다)

[현재 단계: 요구사항 분석]

당신의 현재 에이전트의 역할을 바탕으로, 사용자의 메시지를 분석하여 다음을 파악하세요:

1. 사용자가 원하는 것이 무엇인가?
2. 이전 대화 맥락이 있다면 무엇인가?
3. 현재 해결해야 할 구체적인 작업은 무엇인가?
                                      
출력 형식 (JSON):
{{
  "user_intent": "사용자가 원하는 것에 대한 명확한 설명",
  "context_summary": "이전 대화에서 이미 수행된 작업 요약",
  "next_task": "지금 수행해야 할 구체적인 작업"
}}

**중요:** 
- 반드시 JSON 형식으로만 응답하세요. Markdown 백틱(```)은 사용하지 마세요. 
- 절대 JSON 이외에 어떠한 정보, 텍스트는 포함하지 마세요.
- JSON 출력은 1개의 객체여야 합니다.
"""
        
        try:
            logger.info(f"[{self.name}] 📋 Analyzing request with FIXED parameters")
            
            # ✅ 고정된 파라미터 사용
            response = await asyncio.to_thread(
                self._call_llm_with_fixed_params,
                messages,
                system_prompt,
                False,      # stream=False (전체 응답)
                "json",         # format="" (텍스트, JSON 아님!)
                temperature=0.1  # 기본값 고정
            )
            
            content = self._remove_think_tag(response)
            logger.info(f"[{self.name}] ✅ Request analysis completed")
            
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[{self.name}] Request analysis failed: {e}")
            raise
    
    async def _make_decision(
        self,
        messages: List,
        available_tools: List[str],
        analyzed_request: str = ""
    ) -> "AgentDecision":
        """
        Agent 의사결정 (기본값 고정)
        
        ⭐ Agent 설정 무시, 항상 기본값 사용
        - temperature: 0.1 (매우 일관적)
        - format: "json" (JSON 강제)
        - stream: False (전체 응답)
        """
        available_agents = self._get_available_agents()
        
        system_prompt = DECISION_PROMPT.format(
            name=self.name,
            available_agents=available_agents,
            available_tools=available_tools
        )
        
        try:
            logger.info(f"[{self.name}] 🤔 Making decision with FIXED parameters")
            
            # ✅ 고정된 파라미터 사용
            response = await asyncio.to_thread(
                self._call_llm_with_fixed_params,
                messages,
                system_prompt,
                False,       # stream=False (전체 응답)
                "json",      # format="json" (JSON 강제)
                temperature=0.1  # 기본값 고정
            )
            
            content = self._remove_think_tag(response)
            logger.info(f"[{self.name}] ✅ Decision made successfully")
            logger.info(f"📋 Decision Request: {content}")
            
            decision_json = json.loads(content)
            
            action_str = decision_json.get("action")
            reasoning = decision_json.get("reasoning", "")
            
            if action_str == "use_tool":
                return AgentDecision(
                    action=AgentAction.USE_TOOL,
                    reasoning=reasoning,
                    tool_name=decision_json.get("tool_name"),
                    tool_arguments=decision_json.get("tool_arguments", {})
                )
            elif action_str == "delegate":
                return AgentDecision(
                    action=AgentAction.DELEGATE,
                    reasoning=reasoning,
                    next_agent=decision_json.get("next_agent")
                )
            else:
                return AgentDecision(
                    action=AgentAction.RESPOND,
                    reasoning=reasoning
                )
                
        except Exception as e:
            logger.error(f"[{self.name}] Decision making failed: {e}")
            raise
    
    async def _generate_final_response(
        self,
        messages: List,
        tool_names: List[str]
    ) -> str:
        """
        최종 답변 생성 (Agent 설정 따름)
        
        ⭐ Agent의 llm_config 사용
        - 창의성 조정 가능
        - 포맷 설정 가능
        - Agent별로 다른 스타일 가능
        
        각 Agent에서 llm_config를 다르게 설정하면
        이 메서드가 그에 따라 답변을 생성함
        """
        agent_role = self.get_agent_role_prompt()
        
        system_prompt = f"""{agent_role}

---

**[현재 단계: 최종 답변 생성]**

당신의 역할을 바탕으로, 지금까지 수행한 작업의 결과를 사용자에게 전달하세요.

**출력:** 순수 텍스트 응답
"""
        
        try:
            logger.info(f"[{self.name}] 💬 Generating final response with Agent config")
            logger.info(f"[{self.name}] Using Agent's LLM settings: {self.llm_config}")
            
            # ✅ Agent 설정을 따름 (_call_llm 사용)
            response = await asyncio.to_thread(
                self._call_llm,
                messages,
                system_prompt,
                None,   # stream: Agent 설정 따름
                ""      # format: 텍스트 응답
            )
            
            logger.info(f"[{self.name}] ✅ Final response generated")
            return self._remove_think_tag(response)
        except Exception as e:
            logger.error(f"[{self.name}] Final response generation failed: {e}")
            raise
    
    async def _generate_fallback_response(self, messages: List) -> str:
        """최대 반복 횟수 도달 시 폴백 응답"""
        return f"""처리 과정이 예상보다 복잡하여 {self.max_iterations}회 반복 제한에 도달했습니다.
지금까지 수집한 정보를 바탕으로 답변드리겠습니다.

추가로 필요한 정보가 있다면 질문을 더 구체적으로 다시 해주시면 감사하겠습니다."""

    # =============================
    # 구체적인 Agent가 구현해야 할 메서드
    # =============================
    
    @abstractmethod
    def get_agent_role_prompt(self) -> str:
        """Agent 역할 정의 Prompt"""
        pass

    # =============================
    # 공통 헬퍼 메서드
    # =============================
    
    def _get_available_agents(self) -> str:
        """
        현재 Agent에서 위임 가능한 다른 Agent 목록
        """
        if hasattr(self, "allowed_agents"):
            # allowed_agents가 있어도 자기 자신은 무조건 제외해야 함
            agents = [name for name in self.allowed_agents if name != self.name]
        else:
            # 기본: 모든 등록된 Agent (자신 제외)
            from agent.registry.agent_registry import AgentRegistry
            all_agents = AgentRegistry.list_agents()
            agents = [name for name in all_agents if name != self.name]
            
        logger.info(f"{agents} available for delegation from {self.name}")
        
        if not agents:
            return "없음 (이 에이전트가 모든 작업을 직접 처리해야 함)"
        
        # 포맷팅
        agent_list = "\n".join([f"- {agent}" for agent in agents])
        
        return f"""
[위임 가능한 다른 Agent 목록]
{agent_list}

**주의:** 위 목록에 없는 Agent(특히 자기 자신)에게는 절대 위임할 수 없습니다.
"""
    
    async def _list_mcp_tools(self) -> List[Dict[str, Any]]:
        """MCP 도구 목록 조회"""
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

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Any:
        """MCP 도구 실행"""
        try:
            result = await self.mcp.call_tool(tool_name, tool_args)
            logger.info(f"[{self.name}] Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Tool '{tool_name}' execution failed: {e}")
            raise
    
    def _remove_think_tag(self, text: str) -> str:
        """
        </think> 종료 태그를 기준으로 그 뒤의 텍스트(진짜 결과)만 추출합니다.
        그 후 JSON 형식('{ ... }')만 정확히 발라냅니다.
        """
        # 1. </think>가 있다면, 그 뒤의 내용만 취합니다.
        #    (앞에 있는 <think> 블록이나 중복된 JSON은 모두 무시됨)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        
        # 2. 혹시라도 <think>만 있고 닫는 태그가 없는 경우를 대비해 안전장치로 시작 태그 처리
        elif "<think>" in text:
            text = text.rsplit("<think>", 1)[-1]

        # 3. 앞뒤 공백 제거
        text = text.strip()
        
        # 4. 순수한 JSON 객체만 추출 (첫 '{' 부터 마지막 '}' 까지)
        #    이렇게 하면 "Here is the JSON:" 같은 군더더기 텍스트가 붙어도 제거됩니다.
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