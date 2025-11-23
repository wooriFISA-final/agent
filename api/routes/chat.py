"""
채팅 엔드포인트

사용자와 AI 간의 대화를 처리하는 엔드포인트를 정의합니다.
"""
from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
from typing import Dict

from core.logging.logger import setup_logger
from core.config.setting import settings
from agents.config.base_config import StateBuilder
from api.models import ChatRequest, ChatResponse

logger = setup_logger()

router = APIRouter()

# 세션별 잠금 저장소 (동일 세션의 동시 요청 방지)
_session_locks: Dict[str, asyncio.Lock] = {}




async def _execute_graph(
    request: Request,
    chat_request: ChatRequest,
    graph_name: str = "default"
) -> ChatResponse:
    """그래프 실행 공통 로직
    
    Args:
        request: FastAPI Request 객체
        chat_request: 채팅 요청 데이터
        graph_name: 사용할 그래프 이름
        
    Returns:
        ChatResponse: AI 응답 데이터
    """
    session_id = chat_request.session_id
    
    # 세션별 잠금 생성 (없으면)
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    
    # 세션 잠금 획득 (동일 세션의 다른 요청은 대기)
    async with _session_locks[session_id]:
        logger.info(f"세션 잠금 획득: '{session_id}'")
        
        try:
            graph = request.app.state.get_graph(graph_name)
            if not graph:
                logger.error(f"그래프 '{graph_name}'가 초기화되지 않았습니다.")
                available_graphs = request.app.state.list_graphs()
                return ChatResponse(
                    response=f"Graph '{graph_name}' is not available. Available graphs: {available_graphs}",
                    status="error",
                    metadata={
                        "error": "graph_not_found",
                        "graph": graph_name,
                        "available_graphs": available_graphs
                    }
                )

            logger.info(f"\n{'='*80}")
            logger.info(f"새로운 요청 | 그래프: {graph_name} | 세션: {chat_request.session_id}")
            logger.info(f"   메시지: {chat_request.message}")
            logger.info(f"{'='*80}")

            graph_config = {"configurable": {"thread_id": chat_request.session_id}}

            # Check for existing conversation state
            try:
                existing_state = await graph.aget_state(graph_config)
                has_history = existing_state and existing_state.values.get('global_messages')
            except Exception as e:
                logger.warning(f"세션 '{chat_request.session_id}'의 기존 상태를 로드할 수 없습니다: {e}")
                has_history = False

            if has_history:
                logger.info(f"세션 '{chat_request.session_id}'의 대화를 이어갑니다.")
                input_state = {"global_messages": [HumanMessage(content=chat_request.message)]}
            else:
                logger.info(f"세션 '{chat_request.session_id}'의 새로운 대화를 시작합니다.")
                input_state = StateBuilder.create_initial_state(
                    messages=[HumanMessage(content=chat_request.message)],
                    session_id=chat_request.session_id,
                )

            # Execute the agent graph
            logger.info(f"'{graph_name}' 그래프 실행 중...")
            result_state = await graph.ainvoke(input_state, config=graph_config)
            logger.info("그래프 실행 완료.")

            # Extract the final response from global_messages
            all_messages = result_state.get("global_messages", [])
            ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]

            if not ai_messages:
                logger.warning("최종 상태에서 AI 메시지를 찾을 수 없습니다.")
                # 폴백: last_result 확인
                last_result = result_state.get("last_result")
                if last_result:
                    logger.info("last_result를 대체 응답으로 사용합니다.")
                    return ChatResponse(
                        response=last_result,
                        status="success",
                        metadata={
                            "session_id": chat_request.session_id,
                            "graph": graph_name,
                            "source": "last_result"
                        }
                    )
                return ChatResponse(
                    response="AI did not generate a response.",
                    status="warning",
                    metadata={"graph": graph_name}
                )

            final_response = ai_messages[-1].content
            logger.info(f"세션 '{chat_request.session_id}'에 대한 응답을 반환합니다.")
            
            return ChatResponse(
                response=final_response,
                status="success",
                metadata={
                    "session_id": chat_request.session_id,
                    "graph": graph_name
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"세션 '{chat_request.session_id}' 요청 시간 초과")
            return ChatResponse(
                response="Request timed out.",
                status="error",
                metadata={
                    "error": "timeout",
                    "session_id": chat_request.session_id,
                    "graph": graph_name
                }
            )
        
        except Exception as e:
            logger.error(f"세션 '{chat_request.session_id}' 채팅 처리 실패: {e}", exc_info=True)
            return ChatResponse(
                response=f"An internal error occurred: {str(e)}",
                status="error",
                metadata={
                    "error": "processing_error",
                    "detail": str(e),
                    "session_id": chat_request.session_id,
                    "graph": graph_name
                }
            )
        finally:
            logger.info(f"세션 잠금 해제: '{session_id}'")


@router.post("/chat/plan", response_model=ChatResponse)
async def chat_plan_endpoint(request: Request, chat_request: ChatRequest):
    """Plan 그래프 전용 채팅 엔드포인트
    
    재무 계획 관련 그래프를 사용하여 채팅을 처리합니다.
    
    Args:
        request: FastAPI Request 객체
        chat_request: 채팅 요청 데이터
        
    Returns:
        ChatResponse: AI 응답 데이터
    """
    return await _execute_graph(request, chat_request, "plan")


@router.post("/chat/report", response_model=ChatResponse)
async def chat_report_endpoint(request: Request, chat_request: ChatRequest):
    """Report 그래프 전용 채팅 엔드포인트
    
    리포트 생성 관련 그래프를 사용하여 채팅을 처리합니다.
    
    Args:
        request: FastAPI Request 객체
        chat_request: 채팅 요청 데이터
        
    Returns:
        ChatResponse: AI 응답 데이터
    """
    return await _execute_graph(request, chat_request, "report")
