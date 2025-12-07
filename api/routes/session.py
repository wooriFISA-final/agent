"""
세션 관리 엔드포인트

대화 세션의 조회, 관리, 삭제 기능을 제공하는 엔드포인트를 정의합니다.
"""
from fastapi import APIRouter, Request

from core.logging.logger import setup_logger

logger = setup_logger()

router = APIRouter(prefix="/chat")


@router.get("/sessions")
async def list_sessions(request: Request):
    """세션 목록 조회
    
    모든 활성 세션 ID를 반환합니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        dict: 세션 목록 및 개수
    """
    session_manager = request.app.state.session_manager
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    sessions = session_manager.list_all_sessions()
    return {"status": "success", "sessions": sessions, "count": len(sessions)}


@router.get("/sessions/detailed")
async def list_sessions_detailed(request: Request):
    """세션 상세 정보 조회
    
    모든 활성 세션의 상세 정보를 반환합니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        dict: 세션 상세 정보 및 개수
    """
    session_manager = request.app.state.session_manager
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
        
    sessions = session_manager.list_sessions_with_details()
    return {"status": "success", "sessions": sessions, "count": len(sessions)}


@router.get("/session/{session_id}/history")
async def get_conversation_history(session_id: str, request: Request):
    """대화 히스토리 조회
    
    특정 세션의 대화 히스토리를 반환합니다.
    
    Args:
        session_id: 세션 ID
        request: FastAPI Request 객체
        
    Returns:
        dict: 대화 히스토리 정보
    """
    graph = request.app.state.graph
    if not graph:
        return {"status": "error", "message": "Graph not initialized"}
    
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)
        
        if not state or not state.values:
            return {"status": "not_found", "message": f"Session {session_id} not found", "messages": []}
        
        messages = state.values.get('global_messages', [])
        message_list = [
            {"type": type(msg).__name__, "content": msg.content} for msg in messages
        ]
        
        return {
            "status": "success",
            "session_id": session_id,
            "message_count": len(messages),
            "messages": message_list
        }
    except Exception as e:
        logger.error(f"Failed to get conversation history for '{session_id}': {e}")
        return {"status": "error", "message": str(e)}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """세션 삭제
    
    특정 세션과 그 히스토리를 삭제합니다.
    
    Args:
        session_id: 세션 ID
        request: FastAPI Request 객체
        
    Returns:
        dict: 삭제 결과
    """
    session_manager = request.app.state.session_manager
    if not session_manager:
        return {"status": "error", "message": "SessionManager not initialized"}
    
    result = session_manager.delete_session(session_id)
    if result["deleted"]:
        logger.info(f"🗑️ Session {session_id} deleted")
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        return {"status": "not_found", "message": f"Session {session_id} not found"}
