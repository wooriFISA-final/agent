from abc import ABC, abstractmethod
import asyncio
import json
import re
from typing import Any, Dict, Optional, List, Tuple
from enum import Enum

from agents.config.base_config import (
    BaseAgentConfig,
    AgentState,
    StateBuilder,
    StateValidator,
    ExecutionStatus
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agents.base.messages import ThinkMessage
from core.mcp.mcp_manager import MCPManager
from core.logging.logger import setup_logger


logger = setup_logger()


class AgentAction(Enum):
    """Agent가 취할 수 있는 행동 타입"""
    USE_TOOL = "use_tool"      # Tool 사용
    RESPOND = "respond"         # 최종 답변 생성


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
    멀티턴 Tool 호출을 지원하는 Agent 베이스 클래스 (AgentState 통합)
    
    핵심 설계:
    - AgentBase: 모든 Agent의 공통 동작 로직 + 범용 Prompt 템플릿
    - AgentState: 통합된 상태 관리 (StateBuilder, StateValidator)
    - 구체적인 Agent: 단 1개의 "역할 정의 Prompt"만 구현
    
    동작 흐름 (ReAct Loop):
    1. 요구사항 분석
    2. Tool 필요 여부 판단 + Tool 선택
    3. Tool 실행 → 다시 1로 (반복)
    4. 최종 답변 생성
    """

    def __init__(self, config: BaseAgentConfig):
        self.name = config.name
        self.config = config
        self.mcp = MCPManager().get_instance()
        self.max_iterations = getattr(config, 'max_iterations', 10)
        self._validate_config()

    # =============================
    # 멀티턴 실행 파이프라인 (AgentState 통합)
    # =============================
    async def run(self, state: AgentState) -> AgentState:
        """
        Agent 실행 메인 플로우 (AgentState 사용)
        
        개선사항:
        - Dict → AgentState 타입 사용
        - StateBuilder를 통한 상태 관리
        - 에러 시 자동으로 상태에 기록
        - 실행 상태 추적 (RUNNING → SUCCESS/FAILED)
        """
        self._log_start(state)

        # 1. 입력 검증 (StateValidator 활용)
        if not self.validate_input(state):
            error = ValueError(f"Invalid input for {self.name}")
            state = StateBuilder.add_error(state, error, self.name)
            state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
            return state

        # 2. 전처리
        state = self.pre_execute(state)

        # 3. 재시도 로직
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with asyncio.timeout(self.config.timeout):
                    result = await self.execute_multi_turn(state)
                
                # 성공 시 루프 종료
                break
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout} seconds"
                logger.warning(f"[{self.name}] attempt {attempt} failed: {error_msg}")
                
                if attempt == self.config.max_retries:
                    # 최종 실패
                    error = TimeoutError(f"{self.name} execution timed out")
                    state = StateBuilder.add_error(state, error, self.name)
                    state = StateBuilder.finalize_state(state, ExecutionStatus.TIMEOUT)
                    return state
                
                await asyncio.sleep(1.5 * attempt)
                
            except Exception as e:
                logger.warning(f"[{self.name}] attempt {attempt} failed: {e}")
                
                if attempt == self.config.max_retries:
                    # 최종 실패
                    state = StateBuilder.add_error(state, e, self.name)
                    state = StateBuilder.finalize_state(state, ExecutionStatus.FAILED)
                    return state
                
                await asyncio.sleep(1.5 * attempt)

        # 4. 후처리 및 로깅
        self._log_end(result)
        return result

    # =============================
    # 멀티턴 실행 로직 (ReAct Loop with AgentState)
    # =============================
    async def execute_multi_turn(self, state: AgentState) -> AgentState:
        """
        멀티턴 실행 플로우 (AgentState 완전 통합)
        
        개선사항:
        - messages는 state에서 직접 관리
        - Tool 호출 시 StateBuilder.add_tool_call() 사용
        - 반복마다 StateBuilder.increment_iteration() 호출
        - 상태 추적 및 검증 강화
        
        Loop:
          1. 요구사항 분석
          2. Tool 필요 여부 판단 + Tool 선택
          3-a. Tool 필요 → Tool 실행 → Loop 재진입
          3-b. Tool 불필요 → 최종 답변 생성 → 종료
        """
        messages = state.get("messages", [])
        
        # MCP 도구 목록 조회 (최초 1회)
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
            # 반복 증가
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
            
            # Step 2: Agent 의사결정 (Tool 필요 여부 + Tool 선택)
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
                # Tool 실행
                logger.info(f"🔧 Executing tool: {decision.tool_name}")
                logger.info(f"   Arguments: {decision.tool_arguments}")
                
                try:
                    tool_result = await self._execute_mcp_tool(
                        decision.tool_name,
                        decision.tool_arguments
                    )
                    
                    # Tool 결과를 상태에 기록
                    state = StateBuilder.add_tool_call(
                        state,
                        tool_name=decision.tool_name,
                        arguments=decision.tool_arguments,
                        result=tool_result
                    )
                    
                    # Tool 결과를 메시지에 추가
                    tool_message = ToolMessage(
                        content=f"Tool: {decision.tool_name}\nResult: {tool_result}",
                        tool_call_id=decision.tool_name
                    )
                    messages.append(tool_message)
                    
                    # 상태 업데이트
                    state["messages"] = messages
                    
                    logger.info(f"✅ Tool executed successfully")
                    
                except Exception as e:
                    logger.error(f"[{self.name}] Tool execution failed: {e}")
                    state = StateBuilder.add_error(state, e, self.name)
                    
                    # 에러를 메시지에도 추가
                    error_message = ToolMessage(
                        content=f"Tool: {decision.tool_name}\nError: {str(e)}",
                        tool_call_id=decision.tool_name
                    )
                    messages.append(error_message)
                    state["messages"] = messages
                    
                    # 에러 발생 시에도 계속 진행 (Agent가 판단)
                
                # 다음 iteration으로 계속
                continue
                
            elif decision.action == AgentAction.RESPOND:
                # 최종 답변 생성
                logger.info("✅ Generating final response")
                
                try:
                    final_response = await self._generate_final_response(messages, available_tools)
                    
                    # 답변 메시지 추가
                    messages.append(AIMessage(content=final_response))
                    state["messages"] = messages
                    state["last_result"] = final_response
                    
                    # 성공 상태로 완료
                    state = StateBuilder.finalize_state(state, ExecutionStatus.SUCCESS)
                    logger.info(f"[{self.name}]의 전체 메시지: {state['messages']}")
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
        
        # 최대 반복 상태로 완료
        state = StateBuilder.finalize_state(state, ExecutionStatus.MAX_ITERATIONS)
        return state

    # =============================
    # 범용 Prompt 체계 (AgentBase가 관리)
    # =============================
    
    async def _analyze_request(self, messages: List, available_tools: List[str]) -> str:
        """
        1️⃣ 요구사항 분석 Prompt (범용)
        
        - 구체적인 Agent의 역할 정의를 주입
        - 사용자 요청을 분석
        """
        agent_role = self.get_agent_role_prompt()  # 구체적인 Agent에서 구현
        
        system_prompt = SystemMessage(content=f"""{agent_role}

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

예시:
사용자: "김철수 조회하고 그 사람 이메일로 메일 보내줘"
{{
  "user_intent": "김철수 조회 후 이메일 발송",
  "context_summary": "아직 작업한 내용이 없음",
  "next_task": "김철수 사용자 정보 조회"
}}

**중요:** 
- 반드시 JSON 형식으로만 응답하세요. Markdown 백틱(```)은 사용하지 마세요. 
- JSON 이외에 다른 텍스트는 포함하지 마세요.
""")
        
        try:
            response = await self.llm.ainvoke([system_prompt, *messages])

            content = self._remove_think_tag(response.content)

            logger.info(f"[{self.name}] Request analysis raw response: {content}")

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
        """
        2️⃣ Tool 필요 여부 판단 + 3️⃣ Tool 선택 Prompt (범용)
        
        - 구체적인 Agent의 역할 정의를 주입
        - Tool이 필요한지 판단하고, 필요하면 선택
        """
        agent_role = self.get_agent_role_prompt()

        
        system_prompt = SystemMessage(content=f"""{agent_role}

---

**현재 단계: 의사결정**

[분석된 요구사항:]
{analyzed_request}

[사용 가능한 MCP Tools]
{available_tools}

[의사결정 규칙]

1. 현재 수행해야 할 작업(next_task)을 해결하기 위해 Tool이 필요한가?
   - Tool 필요 → "use_tool" 선택, 어떤 Tool을 사용할지 결정
   - Tool 불필요 → "respond" 선택 (이미 충분한 정보가 있어 답변 가능)

2. **Tool 선택 시 주의사항:**
   - 반드시 하나의 Tool만 선택
   - Tool 이름은 정확히 위 목록에서 선택
   - Tool 실행에 필요한 모든 arguments를 제공

3. **이전 Tool 실행 결과 확인:**
   - 이전 대화에 Tool 실행 결과가 있다면 이를 고려
   - Tool의 결과에서 success true/false 여부를 반드시 확인
   - 다음 단계로 넘어갈지, 추가 Tool이 필요한지 판단

[출력 형식 (JSON)]
{{
  "action": "use_tool | respond",
  "reasoning": "의사결정 이유",
  "tool_name": "사용할 Tool 이름 (action=use_tool인 경우에만)",
  "tool_arguments": {{"arg1": "value1", "arg2": "value2"}}
}}

**예시 1 - Tool 사용:**
{{
  "action": "use_tool",
  "reasoning": "김철수 사용자 정보를 생성하기 위해 create_user Tool이 필요",
  "tool_name": "create_user",
  "tool_arguments": {{"name": "김철수", "age": "25"}}
}}

**예시 2 - 답변 생성:**
{{
  "action": "respond",
  "reasoning": "모든 필요한 정보가 수집되었고, 이제 사용자에게 결과를 전달할 수 있음"
}}

**중요**
- 반드시 JSON 형식으로만 응답하세요. Markdown 백틱(```)은 사용하지 마세요.
- JSON 이외에 다른 텍스트는 포함하지 마세요.
""")
        
        try:
            response = await self.llm.ainvoke([system_prompt, *messages])
            content = self._remove_think_tag(response.content)

            logger.info(f"[{self.name}] Decision making raw response: {content}")

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
        """
        4️⃣ 최종 답변 생성 Prompt (범용)
        
        - 구체적인 Agent의 역할 정의를 주입
        - Tool 실행 결과를 바탕으로 최종 답변 생성
        """
        agent_role = self.get_agent_role_prompt()
        
        system_prompt = SystemMessage(content=f"""{agent_role}

---

**[현재 단계: 최종 답변 생성]**

당신의 역할을 바탕으로, 지금까지 수행한 작업의 결과를 사용자에게 전달하세요.

**답변 작성 가이드:**

1. **작업 결과 요약:**
   - 무엇을 수행했는지 명확히 전달
   - Tool 실행 결과의 핵심 정보만 포함

2. **사용자 친화적 표현:**
   - 기술적인 세부사항은 생략
   - 친근하고 자연스러운 톤 유지
   - 필요시 Markdown 포맷 사용 가능

3. **성공/실패 명확히 구분:**
   - 작업이 성공했는지, 실패했는지 명확히
   - 실패 시 이유와 다음 단계 제안

**출력:** 순수 텍스트 응답 (JSON 아님)
""")
        
        try:
            logger.info(f"[{self.name}] Generating final response with messages: {messages}")
            response = await self.llm.ainvoke([system_prompt, *messages])
            logger.info(f"[{self.name}] Final response raw content: {response.content}")
            return self._remove_think_tag(response.content)
        except Exception as e:
            logger.error(f"[{self.name}] Final response generation failed: {e}")
            raise
    
    async def _generate_fallback_response(self, messages: List) -> str:
        """최대 반복 횟수 도달 시 폴백 응답"""
        return f"""처리 과정이 예상보다 복잡하여 {self.max_iterations}회 반복 제한에 도달했습니다.
지금까지 수집한 정보를 바탕으로 답변드리겠습니다.

추가로 필요한 정보가 있다면 질문을 더 구체적으로 다시 해주시면 감사하겠습니다."""

    # =============================
    # 구체적인 Agent가 구현해야 할 메서드 (단 1개!)
    # =============================
    
    @abstractmethod
    def get_agent_role_prompt(self) -> str:
        """
        구체적인 Agent의 역할 정의 Prompt
        
        이 Agent가:
        - 누구인지 (정체성)
        - 무엇을 하는지 (담당 업무)
        - 어떻게 동작하는지 (행동 원칙)
        
        를 명확히 정의하세요.
        
        Returns:
            str: Agent 역할 정의 텍스트
        """
        pass

    # =============================
    # 공통 헬퍼 메서드
    # =============================
    async def _list_mcp_tools(self) -> List[Dict[str, Any]]:
        """MCP 서버의 사용 가능한 도구 목록 조회 + Agent 허용 필터링"""
        try:
            tools = await self.mcp.list_tools()
            tools_spec = []
            
            # === MCP에서 도구 스펙 받아와서 Function calling 포맷으로 변환 ===
            # === Agent의 allowed_tools에 따라 필터링 ===
            if hasattr(self, "allowed_tools") and self.allowed_tools:
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
            logger.info(f"[{self.name}] Retrieved {json.dumps(tools_spec, indent=2, ensure_ascii=False, default=str)}")
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
    
    
    def _remove_think_tag(self,text: str) -> Tuple[str, List[ThinkMessage]]:
        """
        <think>...</think> 내용을 제거하고, 나머지 텍스트를 반환합니다.
        """
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
    # =============================
    # 기타 공통 메서드 (선택적 오버라이드, AgentState 사용)
    # =============================
    
    def validate_input(self, state: AgentState) -> bool:
        """
        입력 상태 검증 (StateValidator 활용)
        
        기본 검증:
        - messages 필드 존재 및 타입
        - 실행 상태 유효성
        
        Override 가능: 구체적인 Agent에서 추가 검증 구현
        """
        # messages 필드 검증
        if not StateValidator.validate_messages(state):
            logger.error(f"[{self.name}] Invalid messages field")
            return False
        
        # 실행 상태 검증
        is_valid, error_msg = StateValidator.validate_execution_state(state)
        if not is_valid:
            logger.error(f"[{self.name}] Invalid execution state: {error_msg}")
            return False
        
        return True

    def pre_execute(self, state: AgentState) -> AgentState:
        """
        실행 전 전처리
        
        Override 가능: 구체적인 Agent에서 추가 전처리 구현
        """
        # 기본: 아무것도 하지 않음
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
        logger.info(f"   Status: {state.get('status', 'unknown')}")

    def _log_end(self, state: AgentState):
        """실행 완료 로깅"""
        logger.info(f"[{self.name}] Execution completed")
        logger.info(f"   Final Status: {state.get('status', 'unknown')}")
        logger.info(f"   Iterations: {state.get('iteration', 0)}")
        logger.info(f"   Tool Calls: {len(state.get('tool_calls', []))}")
        logger.info(f"   Errors: {len(state.get('errors', []))}")
        logger.info(f"   Warnings: {len(state.get('warnings', []))}")