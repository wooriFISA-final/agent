from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# ============================================================
# [설정] 스크립트 경로 (환경에 맞게 수정 필수)
# Docker 환경이라면 보통 /opt/airflow/scripts 입니다.
# ============================================================

SCRIPTS_PATH = "/opt/airflow/scripts"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'fund_ranking_pipeline',
    default_args=default_args,
    description='Fund Crawling -> Preprocess -> PCA -> DB Snapshot -> User Balance Update',
    schedule_interval='0 2 * * *',  # 매일 새벽 2시 실행
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['fund', 'ml', 'automation', 'fisa'],
) as dag:

    # --------------------------------------------------------
    # 1단계: 펀드 데이터 수집 (Crawling)
    # 실행파일: fund_collect.py
    # --------------------------------------------------------
    t1_collect = BashOperator(
        task_id='fund_collect',
        bash_command=f'python {SCRIPTS_PATH}/fund_collect.py',
        cwd=SCRIPTS_PATH, # 스크립트 실행 위치를 scripts 폴더로 고정
    )

    # --------------------------------------------------------
    # 2단계: 데이터 전처리 (Preprocessing)
    # 실행파일: fund_preprocess.py
    # --------------------------------------------------------
    t2_preprocess = BashOperator(
        task_id='fund_preprocess',
        bash_command=f'python {SCRIPTS_PATH}/fund_preprocess.py',
        cwd=SCRIPTS_PATH,
    )

    # --------------------------------------------------------
    # 3단계: PCA 분석 및 랭킹 산출 (Analysis)
    # 실행파일: fund_pca.py
    # --------------------------------------------------------
    t3_pca = BashOperator(
        task_id='fund_pca',
        bash_command=f'python {SCRIPTS_PATH}/fund_pca.py',
        cwd=SCRIPTS_PATH,
    )

    # --------------------------------------------------------
    # 4단계: 분석 결과 DB 적재 (Snapshot Insert)
    # 실행파일: tmp_db_insert.py
    # --------------------------------------------------------
    t4_insert = BashOperator(
        task_id='db_snapshot_insert',
        bash_command=f'python {SCRIPTS_PATH}/tmp_db_insert.py',
        cwd=SCRIPTS_PATH,
    )

    # --------------------------------------------------------
    # 5단계: 사용자 펀드 평가금액 갱신 (Balance Update)
    # 실행파일: fund_daily_update.py (새로 만든 파일)
    # --------------------------------------------------------
    t5_update = BashOperator(
        task_id='daily_balance_update',
        bash_command=f'python {SCRIPTS_PATH}/fund_daily_update.py',
        cwd=SCRIPTS_PATH,
    )

    # ========================================================
    # [실행 순서 정의]
    # 수집 -> 전처리 -> 분석 -> 스냅샷저장 -> 잔고갱신
    # ========================================================
    t1_collect >> t2_preprocess >> t3_pca >> t4_insert >> t5_update