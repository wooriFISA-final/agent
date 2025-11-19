from abc import ABC, abstractmethod
import asyncio
import json
import re
from typing import Any, Dict, Optional, List
from enum import Enum

from agent.config.base_config import (
    BaseAgentConfig,
    AgentState,
    StateBuilder,
    StateValidator,
    ExecutionStatus
)

# ✅ LangGraph 호환을 위해 LangChain 메시지는 유지
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from core.mcp.mcp_manager import MCPManager
from core.logging.logger import setup_logger
from core.llm.llm_manger import LLMManager, LLMHelper

logger = setup_logger()


# =============================
# Agent 관련 클래스
# =============================

class AgentAction(Enum):
    """Agent가 취할 수 있는 행동 타입"""
    USE_TOOL = "use_tool"
    RESPOND = "respond"


class AgentDecision:
    """Agent의 의사결정 결과"""
    def __init__(
        self,
        action: AgentAction,
        reasoning: str,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[Dict] = None
    ):
        self.action = action
        self.reasoning = reasoning
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {}


class AgentBase(ABC):
    """
    멀티턴 Tool 호출을 지원하는 Agent 베이스 클래스
    
    핵심 설계:
    - LLMManager를 통한 Ollama Chat API 직접 호출
    - LangChain 메시지는 LangGraph 호환을 위해 유지
    - LLM 호출 시에만 딕셔너리로 변환하여 사용
    - Agent별 LLM 설정 지원
    """

    def __init__(self, config: BaseAgentConfig):
        self.name = config.name
        self.config = config
        self.mcp = MCPManager().get_instance()
        self.max_iterations = config.max_iterations
        
        # Agent별 LLM 설정 병합 (전역 설정 + Agent별 오버라이드)
        self.llm_config = config.get_llm_config_dict()
        
        logger.info(f"[{self.name}] Agent initialized")
        logger.info(f"[{self.name}] LLM overrides: {self.llm_config if self.llm_config else 'None (using global settings)'}")
        
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
            return {"role": "user", "content": message.content}
        else:
            return {"role": "user", "content": str(message)}
    
    def _call_llm(
        self,
        messages: List,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        LLM 호출 (동기 방식)
        
        우선순위:
        1. 메서드 호출 시 전달된 kwargs
        2. Agent별 llm_config
        3. 전역 설정 (LLMManager 기본값)
        
        Args:
            messages: LangChain 메시지 리스트
            system_prompt: 시스템 프롬프트
            **kwargs: 추가 LLM 설정 (최우선)
            
        Returns:
            LLM 응답 텍스트
        """
        # Agent 설정과 kwargs 병합 (kwargs가 우선)
        llm_params = {**self.llm_config, **kwargs}
        
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
                system_prompt=None,  # 이미 history에 포함
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
        
        # MCP 도구 목록 조회
        available_tools = await self._list_mcp_tools()
        logger.info(f"[{self.name}] MCP tools available: {len(available_tools)}")

        if not available_tools:
            error_msg = "No MCP tools available"
            logger.error(f"[{self.name}] {error_msg}")
            state = StateBuilder.add_warning(state, error_msg)
            state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
            return state
        
        logger.debug(f"[{self.name}] Available tools: {available_tools}")
        
        # ReAct Loop
        while not StateBuilder.is_max_iterations_reached(state):
            state = StateBuilder.increment_iteration(state)
            current_iteration = state.get("iteration", 0)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{self.name}] Iteration {current_iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # Step 1: 요구사항 분석
            try:
                analyzed_request = await self._analyze_request(messages, available_tools)
                logger.info(f"📋 Analyzed Request: {analyzed_request}")
            except Exception as e:
                logger.error(f"[{self.name}] Request analysis failed: {e}")
                state = StateBuilder.add_error(state, e, self.name)
                break
            
            # Step 2: Agent 의사결정
            try:
                decision = await self._make_decision(messages, available_tools, analyzed_request)
                logger.info(f"🤔 Decision: {decision.action.value}")
                logger.info(f"   Reasoning: {decision.reasoning}")
            except Exception as e:
                logger.error(f"[{self.name}] Decision making failed: {e}")
                state = StateBuilder.add_error(state, e, self.name)
                break
            
            # Step 3: 액션 실행
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
    # 범용 Prompt 체계
    # =============================
    
    async def _analyze_request(self, messages: List, available_tools: List[str]) -> str:
        """요구사항 분석"""
        agent_role = self.get_agent_role_prompt()
        
        system_prompt = f"""{agent_role}

---

[현재 단계: 요구사항 분석]

당신의 역할을 바탕으로, 사용자의 메시지를 분석하여 다음을 파악하세요:

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
- JSON 이외에 다른 텍스트는 포함하지 마세요.
"""
        
        try:
            # asyncio로 동기 함수를 비동기 실행
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._call_llm,
                messages,
                system_prompt
            )
            
            content = self._remove_think_tag(response)
            logger.debug(f"[{self.name}] Request analysis raw response: {content}")
            
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[{self.name}] Request analysis failed: {e}")
            raise
    
    async def _make_decision(
        self,
        messages: List,
        available_tools: List[str],
        analyzed_request: str
    ) -> AgentDecision:
        """Tool 필요 여부 판단 + Tool 선택"""
        agent_role = self.get_agent_role_prompt()
        
        system_prompt = f"""{agent_role}

---

**현재 단계: 의사결정**

[분석된 요구사항:]
{analyzed_request}

[사용 가능한 MCP Tools]
{available_tools}

[의사결정 규칙]

1. 현재 수행해야 할 작업(next_task)을 해결하기 위해 Tool이 필요한가?
   - Tool 필요 → "use_tool" 선택, 어떤 Tool을 사용할지 결정
   - Tool 불필요 → "respond" 선택

2. **Tool 선택 시 주의사항:**
   - 반드시 하나의 Tool만 선택
   - Tool 이름은 정확히 위 목록에서 선택
   - Tool 실행에 필요한 모든 arguments를 제공

3. **이전 Tool 실행 결과 확인:**
   - 이전 대화에 Tool 실행 결과가 있다면 이를 고려
   - 다음 단계로 넘어갈지, 추가 Tool이 필요한지 판단

[출력 형식(JSON)]
{{
  "action": "use_tool | respond",
  "reasoning": "의사결정 이유",
  "tool_name": "사용할 Tool 이름",
  "tool_arguments": {{"arg1": "value1"}}
}}

**중요:** 
- 반드시 [출력 형식(JSON)]에 맞게 응답하세요. Markdown 백틱(```)은 사용하지 마세요. 
- JSON 이외에 다른 텍스트는 포함하지 마세요.
"""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_llm,
                messages,
                system_prompt
            )
            
            content = self._remove_think_tag(response)
            logger.debug(f"[{self.name}] Decision making raw response: {content}")
            
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
        """최종 답변 생성"""
        agent_role = self.get_agent_role_prompt()
        
        system_prompt = f"""{agent_role}

---

**[현재 단계: 최종 답변 생성]**

당신의 역할을 바탕으로, 지금까지 수행한 작업의 결과를 사용자에게 전달하세요.

**출력:** 순수 텍스트 응답 (JSON 아님)
"""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_llm,
                messages,
                system_prompt
            )
            
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
        """<think> 태그 제거"""
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

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