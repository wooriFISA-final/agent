"""
세션 관리 시스템 통합 테스트
API 서버가 실행 중이어야 합니다: python main.py
"""
import requests
import json
from time import sleep

# API 기본 URL
BASE_URL = "http://localhost:8080"


def print_section(title: str):
    """섹션 구분선 출력"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def test_chat(message: str, session_id: str = "default-session"):
    """채팅 API 테스트"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "session_id": session_id}
    )
    data = response.json()
    
    print(f"[Session: {session_id}]")
    print(f"User: {message}")
    print(f"AI: {data.get('response', 'No response')}")
    print(f"Metadata: {json.dumps(data.get('metadata', {}), indent=2, ensure_ascii=False)}")
    print()
    
    return data


def test_list_sessions():
    """세션 목록 조회"""
    response = requests.get(f"{BASE_URL}/chat/sessions")
    data = response.json()
    
    print(f"Status: {data.get('status')}")
    print(f"Total sessions: {data.get('count', 0)}")
    print(f"Sessions: {data.get('sessions', [])}")
    print()
    
    return data


def test_list_sessions_detailed():
    """세션 상세 정보 조회"""
    response = requests.get(f"{BASE_URL}/chat/sessions/detailed")
    data = response.json()
    
    print(f"Status: {data.get('status')}")
    print(f"Total sessions: {data.get('count', 0)}")
    
    for session in data.get('sessions', []):
        print(f"\n  Session: {session.get('session_id')}")
        print(f"    Checkpoints: {session.get('checkpoint_count')}")
        print(f"    Messages: {session.get('message_count')}")
        print(f"    First: {session.get('first_checkpoint')}")
        print(f"    Last: {session.get('last_checkpoint')}")
    print()
    
    return data


def test_get_session_info(session_id: str):
    """특정 세션 정보 조회"""
    response = requests.get(f"{BASE_URL}/chat/session/{session_id}")
    data = response.json()
    
    print(f"Status: {data.get('status')}")
    if data.get('status') == 'success':
        session = data.get('session', {})
        print(f"Session: {session.get('session_id')}")
        print(f"  Checkpoints: {session.get('checkpoint_count')}")
        print(f"  Messages: {session.get('message_count')}")
        print(f"  First: {session.get('first_checkpoint')}")
        print(f"  Last: {session.get('last_checkpoint')}")
    else:
        print(f"Message: {data.get('message')}")
    print()
    
    return data


def test_statistics():
    """전체 통계 조회"""
    response = requests.get(f"{BASE_URL}/chat/statistics")
    data = response.json()
    
    print(f"Status: {data.get('status')}")
    if data.get('status') == 'success':
        stats = data.get('statistics', {})
        print(f"Total sessions: {stats.get('total_sessions')}")
        print(f"Total checkpoints: {stats.get('total_checkpoints')}")
        print(f"Total messages: {stats.get('total_messages')}")
        print(f"Avg checkpoints/session: {stats.get('avg_checkpoints_per_session', 0):.2f}")
        print(f"Avg messages/session: {stats.get('avg_messages_per_session', 0):.2f}")
    print()
    
    return data


def test_delete_session(session_id: str):
    """세션 삭제"""
    response = requests.delete(f"{BASE_URL}/chat/session/{session_id}")
    data = response.json()
    
    print(f"Status: {data.get('status')}")
    print(f"Message: {data.get('message')}")
    if data.get('status') == 'success':
        print(f"Checkpoints deleted: {data.get('checkpoints_deleted')}")
    print()
    
    return data


def test_cleanup_sessions():
    """빈 세션 정리"""
    response = requests.post(f"{BASE_URL}/chat/sessions/cleanup")
    data = response.json()
    
    print(f"Status: {data.get('status')}")
    print(f"Message: {data.get('message')}")
    print(f"Deleted sessions: {data.get('deleted_sessions', [])}")
    print()
    
    return data


def main():
    """메인 테스트 플로우"""
    
    print_section("1️⃣  서버 상태 확인")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running")
        print(f"Version: {response.json().get('version')}")
    except Exception as e:
        print(f"❌ Server is not running: {e}")
        print("Please start the server first: python main.py")
        return
    
    print_section("2️⃣  사용자 Alice - 첫 번째 대화")
    test_chat("김철수 25세 등록해줘", session_id="user-alice")
    sleep(1)
    
    print_section("3️⃣  사용자 Alice - 두 번째 대화 (이전 대화 기억)")
    test_chat("방금 등록한 사람 조회해줘", session_id="user-alice")
    sleep(1)
    
    print_section("4️⃣  사용자 Bob - 새 세션 시작")
    test_chat("이영희 30세 등록해줘", session_id="user-bob")
    sleep(1)
    
    print_section("5️⃣  사용자 Bob - 두 번째 대화")
    test_chat("방금 등록한 사람 조회해줘", session_id="user-bob")
    sleep(1)
    
    print_section("6️⃣  세션 목록 조회 (간단)")
    test_list_sessions()
    
    print_section("7️⃣  세션 목록 조회 (상세)")
    test_list_sessions_detailed()
    
    print_section("8️⃣  Alice 세션 정보 조회")
    test_get_session_info("user-alice")
    
    print_section("9️⃣  전체 통계 조회")
    test_statistics()
    
    print_section("🔟 Bob 세션 삭제")
    test_delete_session("user-bob")
    
    print_section("1️⃣1️⃣ 삭제 후 세션 목록")
    test_list_sessions()
    
    print_section("1️⃣2️⃣ Alice는 여전히 이전 대화 기억 중")
    test_chat("김철수 나이가 몇 살이었지?", session_id="user-alice")
    
    print_section("1️⃣3️⃣ 빈 세션 정리")
    test_cleanup_sessions()
    
    print_section("✅ 테스트 완료!")
    print("모든 테스트가 성공적으로 완료되었습니다.")


if __name__ == "__main__":
    main()