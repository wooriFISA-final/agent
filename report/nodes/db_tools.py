# report/nodes/db_tools.py (상단 수정)

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# /agent/report/nodes 에서 /agent 로 이동 (두 단계)
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..', '..') 
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
USER_CONFIG_PATH = "/Users/lyra8/lyra8_files/pythonProject/fisa_final_project/agent/db_tests/user_config.json"

# .env 파일에서 환경 변수를 로드합니다. (경로 명시)
print(f"🔎 DEBUG: ENV 파일 경로 시도: {ENV_PATH}") # 경로 확인

# load_dotenv는 파일 로드 성공 여부를 불리언 값으로 반환합니다.
load_result = load_dotenv(dotenv_path=ENV_PATH) 
print(f"🔎 DEBUG: .env 파일 로드 성공 여부: {load_result}")

# =======================================================================
# 1. DB 연결 설정: 환경 변수에서 값 가져오기
# =======================================================================
db_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), 
    "database": os.getenv("DB_NAME", "woorifisa6") 
}

print(f"🔎 DEBUG: 로드된 DB_HOST: {db_CONFIG['host']}")
print(f"🔎 DEBUG: 로드된 DB_USER: {db_CONFIG['user']}")
print(f"🔎 DEBUG: 로드된 DB_PASSWORD: {'***' if db_CONFIG['password'] else 'None/Empty'}") # 보안을 위해 마스킹

# ... (get_db_connection 함수는 그대로 유지)

def get_db_connection():
    """DB 연결 객체를 반환합니다."""
    conn = None
    if not all([db_CONFIG['host'], db_CONFIG['user'], db_CONFIG['password']]):
        print("❌ 오류: DB 연결 정보(HOST, USER, PASSWORD)가 환경 변수에 설정되지 않았습니다.")
        return None
        
    try:
        conn = mysql.connector.connect(**db_CONFIG)
        return conn
    except Error as e:
        # DB 연결 실패 시 에러 코드 출력
        print(f"❌ Error connecting to MySQL: {e}")
        return None


def fetch_user_id(user_name: str) -> int | None:
    """사용자 이름으로 user_id를 조회합니다."""
    conn = get_db_connection()
    if conn is None:
        return None
    
    query = "SELECT user_id FROM members WHERE user_name = %s"
    user_id = None
    try:
        cursor = conn.cursor()
        cursor.execute(query, (user_name,))
        result = cursor.fetchone()
        if result:
            user_id = result[0] # user_id (BIGINT) 반환
    except Error as e:
        print(f"Error fetching user ID: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return user_id


def fetch_user_consume_data(user_id: int, dates: List[str]) -> List[Dict[str, Any]]:
    """
    특정 사용자 ID의 여러 날짜에 해당하는 지출 내역을 조회합니다.
    dates 예: ['2023-01-01', '2023-02-01']
    """
    conn = get_db_connection()
    if conn is None:
        return []

    # IN 쿼리를 사용하여 여러 날짜의 데이터를 한 번에 가져옵니다.
    placeholders = ', '.join(['%s'] * len(dates))
    query = f"SELECT * FROM user_consume WHERE user_id = %s AND spend_month IN ({placeholders})"
    params = [user_id] + dates

    results = []
    try:
        # 딕셔너리 형태로 결과를 받기 위해 설정
        cursor = conn.cursor(dictionary=True) 
        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
    except Error as e:
        print(f"Error fetching consume data: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return results

# fetch_user_products 등 필요한 다른 DB 함수도 여기에 추가되어야 합니다.
# 예시:
def fetch_user_products(user_id: int) -> List[Dict[str, Any]]:
    # my_products 테이블에서 user_id 기반으로 데이터를 가져오는 로직 구현...
    return []

def fetch_recent_report_summary(member_id: int) -> dict | None:
    # reports 테이블에서 가장 최근의 요약 데이터를 가져옵니다.
    return []

def fetch_house_price(region_name: str) -> dict | None:
    # HOUSE_PRICES 테이블에서 지역 기반 가격을 가져옵니다.
    return None

def fetch_member_details(user_id: int) -> dict | None:
    # members 테이블에서 user_id를 기준으로 모든 상세 정보를 가져옵니다.
    # (credit_score, monthly_salary, total_debt, has_house 등 모든 컬럼 조회)
    return None