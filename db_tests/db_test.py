import json
import mysql.connector
from datetime import date
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
# 이 스크립트를 실행하는 환경에서 .env 파일이 올바르게 로드되는지 확인해야 합니다.
load_dotenv()

# =======================================================================
# 1. DB 연결 설정: 환경 변수에서 값 가져오기
# =======================================================================
# os.getenv()는 해당 환경 변수가 없으면 None을 반환합니다.
db_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), 
    "database": os.getenv("DB_NAME", "woorifisa6") 
}

# =======================================================================
# 2. JSON 파일 경로 설정
# =======================================================================
JSON_FILE_PATH = "/Users/lyra8/lyra8_files/pythonProject/fisa_final_project/agent/db_tests/jinsoo_data.json"

# JSON 키와 user_consume SQL 칼럼 이름의 매핑 순서 정의 (순서가 DB와 일치해야 함)
# 테이블 생성 시 칼럼 이름을 언더바(_)로 수정했다고 가정하고, JSON 키를 매핑합니다.
CONSUME_MAPPING = [
    # (JSON KEY, SQL COLUMN NAME)
    ('spend_month', 'spend_month'), ('total_spend', 'total_spend'),
    ('CAT1_교통', 'CAT1_교통'), ('CAT1_쇼핑', 'CAT1_쇼핑'),
    ('CAT1_식품', 'CAT1_식품'), ('CAT1_교육/문화', 'CAT1_교육_문화'),
    ('CAT1_생활/주거', 'CAT1_생활_주거'), ('CAT1_레저/여행', 'CAT1_레저_여행'),
    ('CAT1_자기계발', 'CAT1_자기계발'), ('CAT1_기타 지출', 'CAT1_기타_지출'),
    
    ('CAT2_대중교통', 'CAT2_대중교통'), ('CAT2_자가용/연료', 'CAT2_자가용_연료'),
    ('CAT2_택시/대리', 'CAT2_택시_대리'), ('CAT2_항공/기차', 'CAT2_항공_기차'),
    ('CAT2_의류', 'CAT2_의류'), ('CAT2_잡화/뷰티', 'CAT2_잡화_뷰티'),
    ('CAT2_명품/쥬얼리', 'CAT2_명품_쥬얼리'), ('CAT2_전자제품', 'CAT2_전자제품'),
    ('CAT2_외식/배달', 'CAT2_외식_배달'), ('CAT2_가정식/식재료', 'CAT2_가정식_식재료'),
    ('CAT2_주점/유흥', 'CAT2_주점_유흥'), ('CAT2_커피/음료', 'CAT2_커피_음료'),
    ('CAT2_사교육/학원', 'CAT2_사교육_학원'), ('CAT2_도서/음반', 'CAT2_도서_음반'),
    ('CAT2_문화생활/취미', 'CAT2_문화생활_취미'), ('CAT2_온라인강의', 'CAT2_온라인강의'),
    ('CAT2_공과금/통신', 'CAT2_공과금_통신'), ('CAT2_병원/약국', 'CAT2_병원_약국'),
    ('CAT2_인테리어/가구', 'CAT2_인테리어_가구'), ('CAT2_보험/금융', 'CAT2_보험_금융'),
    ('CAT2_국내여행/숙박', 'CAT2_국내여행_숙박'), ('CAT2_해외여행/항공', 'CAT2_해외여행_항공'),
    ('CAT2_레포츠/취미', 'CAT2_레포츠_취미'), ('CAT2_기타 여가', 'CAT2_기타_여가'),
    ('CAT2_자격증/어학', 'CAT2_자격증_어학'), ('CAT2_운동/피트니스', 'CAT2_운동_피트니스'),
    ('CAT2_온라인 구독', 'CAT2_온라인_구독'), ('CAT2_도구/재료 구매', 'CAT2_도구_재료_구매'),
    ('CAT2_현금서비스', 'CAT2_현금서비스'), ('CAT2_경조사/기부', 'CAT2_경조사_기부'),
    ('CAT2_해외 직구', 'CAT2_해외_직구'), ('CAT2_금융 수수료', 'CAT2_금융_수수료'),
]


# 3. 데이터 삽입/업데이트 함수
def process_json_data():
    # 3-1. 파일 로드
    print(f"📄 파일 로드 중: {JSON_FILE_PATH}")
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ 오류: 지정된 경로에 파일이 존재하지 않습니다. 경로를 확인해 주세요: {JSON_FILE_PATH}")
        return

    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 오류: JSON 파일 디코딩 오류가 발생했습니다. 파일 내용을 확인하세요. ({e})")
        return
    except Exception as e:
        print(f"❌ 오류: 파일을 읽는 중 예외 발생: {e}")
        return

    # 3-2. DB 연결
    conn = None
    if not all([db_CONFIG['host'], db_CONFIG['user'], db_CONFIG['password']]):
        print("❌ 오류: DB 연결 정보(HOST, USER, PASSWORD)가 환경 변수에 설정되지 않았습니다.")
        return

    try:
        conn = mysql.connector.connect(**db_CONFIG)
        cursor = conn.cursor()

        def safe_get(key, value, default=None):
            """value 딕셔너리에서 key를 가져옵니다."""
            return value.get(key, default)

        # members 테이블 업데이트용 칼럼 이름 리스트
        member_cols = [
            'age', 'gender', 'region', 'residence', 'internet_banking', 'sms_agree', 
            'has_woori_account', 'has_saving_product', 'open_banking', 'is_corporate_employee', 
            'monthly_salary', 'annual_salary', 'credit_score', 'has_house', 'first_home_buyer', 
            'house_value_million_krw', 'DTI', 'DSR', 'total_debt', 'has_npay_account', 
            'has_business_registration', 'has_pre_agreement', 'is_basic_living_recipient', 
            'is_low_income_class', 'is_orphan', 'is_marriage_immigrant', 'is_north_defector', 
            'is_earned_income_beneficiary', 'is_smile_finance_recommended', 
            'has_military_saving_eligibility', 'is_below_250pct_median_income'
        ]

        # members 테이블 업데이트 쿼리 (WHERE user_name 기준)
        member_update_sql = f"""
        UPDATE members SET {', '.join([f"{col} = %s" for col in member_cols])} 
        WHERE user_name = %s
        """

        # user_consume 테이블 삽입 쿼리 구성 (DB 칼럼 이름 사용)
        consume_db_cols = [sql_col for json_key, sql_col in CONSUME_MAPPING]
        consume_insert_sql = f"""
        INSERT INTO user_consume (user_id, {', '.join(consume_db_cols)})
        VALUES ( (SELECT user_id FROM members WHERE user_name = %s), 
                 {', '.join(['%s'] * len(consume_db_cols))} )
        """
        
        processed_users = set()
        insert_count = 0
        update_count = 0

        # 3-3. 데이터 순회 및 삽입
        for key, value in data.items():
            user_name = safe_get('name', value)
            
            # 1. members 테이블 업데이트 (사용자 정보 - 중복 업데이트 방지)
            if user_name and user_name not in processed_users:
                member_data_values = [
                    safe_get('age', value), safe_get('gender', value), safe_get('region', value), safe_get('residence', value),
                    safe_get('internet_banking', value) == 'True', safe_get('sms_agree', value) == 'True', 
                    safe_get('has_woori_account', value) == 'True', safe_get('has_saving_product', value) == 'True', 
                    safe_get('open_banking', value) == 'True', safe_get('is_corporate_employee', value) == 'True', 
                    safe_get('monthly_salary', value), safe_get('annual_salary', value), safe_get('credit_score', value), 
                    safe_get('has_house', value) == 'True', safe_get('first_home_buyer', value) == 'True', 
                    safe_get('house_value_million_krw', value), safe_get('DTI', value), safe_get('DSR', value), safe_get('total_debt', value), 
                    safe_get('has_npay_account', value) == 'True', safe_get('has_business_registration', value) == 'True', 
                    safe_get('has_pre_agreement', value) == 'True', safe_get('is_basic_living_recipient', value) == 'True', 
                    safe_get('is_low_income_class', value) == 'True', safe_get('is_orphan', value) == 'True', 
                    safe_get('is_marriage_immigrant', value) == 'True', safe_get('is_north_defector', value) == 'True', 
                    safe_get('is_earned_income_beneficiary', value) == 'True', safe_get('is_smile_finance_recommended', value) == 'True', 
                    safe_get('has_military_saving_eligibility', value) == 'True', 
                    safe_get('is_below_250pct_median_income', value) == 'True',
                    user_name # WHERE 절에 사용될 user_name
                ]
                cursor.execute(member_update_sql, tuple(member_data_values))
                processed_users.add(user_name)
                update_count += cursor.rowcount

            # 2. user_consume 테이블 삽입 (지출 정보)
            
            # user_id 조회를 위해 user_name을 첫 번째 인자로 사용
            consume_data_values = [user_name] 

            for json_key, sql_col in CONSUME_MAPPING:
                val = safe_get(json_key, value)
                
                if json_key == 'spend_month':
                    # spend_month는 date 객체로 변환
                    try:
                        year, month = map(int, val.split('-'))
                        consume_data_values.append(date(year, month, 1))
                    except:
                        # 데이터 형식 오류 시 None 처리
                        consume_data_values.append(None) 
                else:
                    # 나머지 데이터는 JSON 키로 가져온 값을 그대로 추가
                    consume_data_values.append(val)
            
            cursor.execute(consume_insert_sql, tuple(consume_data_values))
            insert_count += 1


        conn.commit()
        print("---")
        print(f"✅ JSON 데이터 DB 삽입 완료!")
        print(f"   - members 테이블 업데이트: {len(processed_users)}명 ({update_count}건 반영)")
        print(f"   - user_consume 테이블 삽입: {insert_count}건")

    except mysql.connector.Error as err:
        print(f"❌ 데이터베이스 오류 발생: {err}")
        print(f"   - 오류 코드: {err.errno}")
        print(f"   - 오류 메시지: {err.msg}")
        if conn:
            conn.rollback()
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# 4. 함수 실행
if __name__ == '__main__':
    process_json_data()