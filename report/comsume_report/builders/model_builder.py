import pandas as pd
import pickle
import sys
from typing import Dict, Any
from ..state import ModelAssets # 상태 파일에서 ModelAssets 타입 임포트

def load_assets(knn_path: str, scaler_path: str, profile_path: str, data_path: str) -> ModelAssets:
    """
    Agent 구동에 필요한 모든 모델 파일과 데이터를 로드하고 메타 정보를 설정합니다.
    """
    try:
        # 1. 자산 로드 (경로: /models 및 /data)
        with open(knn_path, 'rb') as f:
            knn_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        df_profile = pd.read_csv(profile_path, index_col='cluster')
        df_data = pd.read_csv(data_path)
        
        # 2. 메타 정보 설정
        cat2_cols = [col for col in df_data.columns if col.startswith('CAT2_')]
        K_CLUSTERS = df_profile.shape[0]
        
        print("✅ Agent 자산 로드 완료: 모델, 스케일러, 프로파일 로드.")
        
        return ModelAssets(
            knn_model=knn_model,
            scaler=scaler,
            df_profile=df_profile,
            df_data=df_data,
            cat2_cols=cat2_cols,
            K_CLUSTERS=K_CLUSTERS
        )
    except Exception as e:
        print(f"\n❌ 오류: 필수 파일 로드 실패. 경로를 확인하세요. 오류: {e}")
        sys.exit(1)