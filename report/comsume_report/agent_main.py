import pandas as pd
import numpy as np
import pickle
import requests
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os
import sys

# ----------------------------------------------------
# 1. Agent 클래스 정의 (모든 Task 로직 포함)
# ----------------------------------------------------
class ConsumptionAgent:
    def __init__(self, knn_path, scaler_path, profile_path, data_path, ollama_model_name="qwen3:8b"):
        try:
            # 1. 자산 로드 (경로: /models 및 /data)
            with open(knn_path, 'rb') as f:
                self.knn_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.df_profile = pd.read_csv(profile_path, index_col='cluster')
            self.df_data = pd.read_csv(data_path)
            
            # 2. 메타 정보 설정
            self.cat2_cols = [col for col in self.df_data.columns if col.startswith('CAT2_')]
            self.ollama_model_name = ollama_model_name
            self.K_CLUSTERS = self.df_profile.shape[0]
            
            # 3. 한글 폰트 설정 (Windows 환경을 고려하여 Malgun Gothic 선호)
            plt.rcParams['font.family'] = 'Malgun Gothic' if sys.platform.startswith('win') else 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False 
            
            print("✅ Agent 초기화: 모델, 스케일러, 프로파일 로드 완료.")
        except Exception as e:
            raise FileNotFoundError(f"필수 파일 로드 실패: {e}. 경로(/data, /models)와 파일명을 확인하세요.")

    # Task 2/2-2: 군집 예측 함수
    def get_user_cluster(self, user_id):
        user_data_row = self.df_data[self.df_data['user_id'] == user_id] \
                            .sort_values(by='spend_month', ascending=False).iloc[0]
        user_features = user_data_row[self.cat2_cols].values.reshape(1, -1)
        user_scaled = self.scaler.transform(user_features)
        user_cluster = self.knn_model.predict(user_scaled)[0]
        return user_cluster, user_data_row.to_dict()

    # Task 5: 개인 소비 분석 함수
    def analyze_user_spending(self, user_data):
        user_spending = pd.Series({k: v for k, v in user_data.items() if k in self.cat2_cols}).sort_values(ascending=False)
        top3_cats_str = [f"{c.replace('CAT2_', '')} ({v:.1f}만원)" for c, v in user_spending.head(3).items()]
        fixed_cost_cats = ['공과금/통신', '보험/금융']
        fixed_cols = [f'CAT2_{c}' for c in fixed_cost_cats if f'CAT2_{c}' in user_data]
        fixed_cost = sum(user_data.get(c, 0) for c in fixed_cols)
        total_spend = user_data.get('total_spend', 1)
        non_fixed_cost_rate = f"{((total_spend - fixed_cost) / total_spend) * 100:.1f}%" if total_spend > 0 else "0.0%"
        
        return {
            'total_spend_amount': f"{total_spend:.1f}만원", 'top_3_categories': top3_cats_str, 
            'fixed_cost': f"{fixed_cost:.1f}만원", 'non_fixed_cost_rate': non_fixed_cost_rate
        }

    # Task 4: 군집 별명 생성 함수
    def generate_cluster_nickname(self, cluster_id):
        profile = self.df_profile.loc[cluster_id]
        cat2_profile = profile.filter(like='CAT2_')
        top3_cats = cat2_profile.sort_values(ascending=False).head(3).index.str.replace('CAT2_', '').tolist()
        avg_age = int(profile.get('avg_age', 35))
        age_str = "중장년층 중심의" if avg_age > 45 else ("청년층 중심의" if avg_age < 30 else "핵심 소비 세대의")
        nickname = (f"**[ {age_str} {top3_cats[0]} 및 {top3_cats[1]} 집중형 그룹 ]** "
                    f"평균 나이 {avg_age}세")
        return nickname

    # Task 6/7: Ollama LLM 해석 및 보고서 생성 함수
    def generate_final_report(self, nickname, analysis_data):
        analysis_text = (
            f"총 지출액: {analysis_data['total_spend_amount']}, "
            f"주 소비 영역: {', '.join(analysis_data['top_3_categories'])}, "
            f"고정비: {analysis_data['fixed_cost']}, "
            f"비고정비 비중: {analysis_data['non_fixed_cost_rate']}"
        )
        prompt_template = f"""
        당신은 고객의 소비 분석가입니다. 다음 정보를 기반으로, 고객에게 전달할 4~5줄의 **간결하고 정중한** 소비 분석 보고서를 작성하세요.
        보고서는 한국어로 작성해야 하며, 별도의 머리글이나 꼬리글 없이 바로 본론부터 시작합니다.
        
        [핵심 정보]
        1. 군집 별명: {nickname}
        2. 개인 분석: {analysis_text}
        
        [보고서 포함 요소 및 형식]
        - 고객의 군집 별명을 언급하며 시작
        - 주 소비 영역을 구체적인 금액과 함께 언급
        - 고정비/비고정비 비중을 해석하여 소비 습관에 대한 인사이트 한 줄 포함
        - 최종 아웃풋은 4~5줄의 줄 글 형태여야 함.
        """
        
        payload = {
            "model": self.ollama_model_name, "prompt": prompt_template, "stream": False,
            "options": {"temperature": 0.5, "num_predict": 1024}
        }
        
        try:
            # 💡 Ollama 통신 오류 해결 방안 1: 타임아웃을 180초(3분)로 대폭 증가
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180) 
            response.raise_for_status() 
            final_report = response.json()['response'].strip()
            return final_report
        except requests.exceptions.RequestException as e:
            return f"❌ Ollama 통신 오류: Ollama 서버 문제 또는 타임아웃. 오류: {e}"

    # Task 3: 군집 시각화 함수 (PCA 기반 산점도)
    def plot_user_cluster(self, user_cluster, user_data):
        # 1. 데이터 준비 및 PCA
        # 군집 예측에 사용된 스케일러를 사용하여 전체 데이터 스케일링
        X_all_scaled = self.scaler.transform(self.df_data[self.cat2_cols].values) 
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_all_scaled)
        
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        df_pca['cluster'] = self.df_data['cluster']
        
        # 2. 사용자 위치 찾기
        user_id = user_data['user_id']
        # 가장 최근 데이터 행의 인덱스를 찾고 PCA 변환 결과에서 해당 위치 추출
        user_index = self.df_data[self.df_data['user_id'] == user_id].sort_values(by='spend_month', ascending=False).index[0]
        user_pc = df_pca.loc[user_index]

        # 3. 시각화 (산점도)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="PC1", y="PC2",
            hue="cluster",
            data=df_pca,
            palette=sns.color_palette("hsv", self.K_CLUSTERS),
            legend="full", alpha=0.6, s=20
        )
        # 사용자 위치를 빨간색 별표로 강조
        plt.scatter(user_pc['PC1'], user_pc['PC2'], color='red', marker='*', s=300, label='현재 사용자')
        
        plt.title(f'군집 시각화 및 사용자 위치 (Cluster {user_cluster})', fontsize=16)
        plt.xlabel(f'주성분 1 (Variance: {pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'주성분 2 (Variance: {pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend(title="Cluster ID", loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show() # 그래프 출력
        print(f"✅ Task 3: 사용자 군집 시각화 완료 (Cluster ID: {user_cluster})")

# ----------------------------------------------------
# 2. 메인 실행 블록 (Agent 구동)
# ----------------------------------------------------
if __name__ == "__main__":
    
    # 🌟🌟🌟 폴더 구조에 맞게 경로 설정 🌟🌟🌟
    FINAL_DATA_PATH = 'data/final_data_k3.csv'
    CLUSTER_PROFILE_PATH = 'data/cluster_profile_k3.csv'
    SCALER_MODEL_PATH = 'models/scaler.pkl'
    KNN_MODEL_PATH = 'models/knn_model.pkl'

    # 🌟 Ollama 모델 설정
    AGENT_OLLAMA_MODEL = "qwen3:8b" 
    
    # 1. Agent 객체 생성 및 초기화
    try:
        agent = ConsumptionAgent(
            KNN_MODEL_PATH, 
            SCALER_MODEL_PATH, 
            CLUSTER_PROFILE_PATH, 
            FINAL_DATA_PATH,
            ollama_model_name=AGENT_OLLAMA_MODEL 
        )
    except FileNotFoundError as e:
        print(f"\n❌ 오류: 필수 파일 경로를 찾을 수 없습니다. {e.filename}을 확인하세요.")
        print("💡 /data와 /models 폴더 안에 모든 파일이 있는지 확인해 주세요.")
        sys.exit(1)
        
    # 2. 사용자 ID 설정 및 분석 파이프라인 실행
    # 데이터프레임이 비어있지 않은 경우에만 실행
    if not agent.df_data.empty:
        EXAMPLE_USER_ID = agent.df_data['user_id'].iloc[500] 
    else:
        print("❌ 오류: 로드된 데이터가 비어있어 분석을 시작할 수 없습니다.")
        sys.exit(1)

    print(f"\n--- 🔎 사용자 ID: {EXAMPLE_USER_ID} 분석 시작 ---")

    # 3. 분석 파이프라인 실행
    try:
        # Task 2/2-2: 군집 예측 및 데이터 추출
        user_cluster, user_data = agent.get_user_cluster(EXAMPLE_USER_ID)
        
        # Task 3: 군집 시각화
        agent.plot_user_cluster(user_cluster, user_data)
        
        # Task 4: 군집 별명 생성
        cluster_nickname = agent.generate_cluster_nickname(user_cluster)
        
        # Task 5: 개인 소비 분석
        user_analysis = agent.analyze_user_spending(user_data)
        
        # Task 6 & 7: LLM 해석 및 최종 보고서 생성
        final_report = agent.generate_final_report(cluster_nickname, user_analysis)

        # 4. 최종 결과 출력
        print("\n" + "="*70)
        print(f"### 🏆 최종 AI Agent 보고서 (Ollama {agent.ollama_model_name}) 🏆 ###")
        print("-" * 70)
        print("📌 군집 ID:", user_cluster)
        print("📌 군집 별명:", cluster_nickname)
        print("📌 소비 TOP 3:", ", ".join(user_analysis['top_3_categories']))
        print("-" * 70)
        print("[LLM 생성 보고서]")
        print(final_report)
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 파이프라인 실행 중 치명적인 오류 발생: {e}")
        # Ollama 오류 발생 시, 서버 확인 안내 재강조
        if "Ollama 통신 오류" in str(e):
             print("💡 **Ollama 서버**가 'ollama run qwen3:8b' 상태로 **정상 실행 중인지** 다시 확인해 주세요.")