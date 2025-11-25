# report_project/compare/policy_indexer.py

import os
import faiss
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# 🚨 Policy 텍스트 분할 함수 임포트
from .rag_search_engine import get_policy_chapters 

# --- 설정 ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CACHE_FILE_NAME = 'policy_cache.json'
FAISS_INDEX_FILE = 'policy_faiss.index'

# --- 경로 설정 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
CACHE_PATH = os.path.join(DATA_DIR, CACHE_FILE_NAME)
FAISS_PATH = os.path.join(DATA_DIR, FAISS_INDEX_FILE)

# 🚨 [수정 반영] 파일명 단순화 (사용자님의 시스템에 맞춤)
POLICY_PATH_OLD = os.path.join(DATA_DIR, "20241224.pdf")
POLICY_PATH_NEW = os.path.join(DATA_DIR, "20250305.pdf")

def create_policy_index():
    """
    두 정책 PDF를 읽어 청크를 분할하고, FAISS 인덱스를 생성 후 저장합니다.
    """
    
    # 1. 모델 로드
    print("⏳ 1. 임베딩 모델 로드 중...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"❌ 오류: SentenceTransformer 로드 실패. 라이브러리 설치 또는 인터넷 연결 확인 필요. 오류: {e}")
        return

    # 2. 텍스트 추출 및 분할
    print("⏳ 2. 정책 파일 텍스트 추출 및 청크 분할...")
    
    # 두 파일에서 청크 추출
    chapters_old = get_policy_chapters(POLICY_PATH_OLD)
    chapters_new = get_policy_chapters(POLICY_PATH_NEW)

    # 🚨 DB 저장을 위한 통합 리스트 구성
    all_chunks = []
    
    for i, chap in enumerate(chapters_old):
        all_chunks.append({
            "id": f"OLD_{i}",
            "version": "20241224",
            "title": chap['title'],
            "content": chap['content']
        })

    for i, chap in enumerate(chapters_new):
        all_chunks.append({
            "id": f"NEW_{i}",
            "version": "20250305",
            "title": chap['title'],
            "content": chap['content']
        })
    
    if not all_chunks:
        print("❌ 오류: PDF에서 유효한 청크 텍스트를 추출하지 못했습니다. (PDF 파일 또는 rag_search_engine.py 확인 필요)")
        return

    # 3. 임베딩 벡터 생성
    documents = [c['content'] for c in all_chunks]
    print(f"⏳ 3. 총 {len(documents)}개 청크 임베딩 생성 중...")
    
    # 🚨 sentences_transformers를 사용하여 벡터 생성
    embeddings = model.encode(documents, convert_to_numpy=True)
    d = embeddings.shape[1] # 벡터 차원 수 (예: 384)
    
    # 4. FAISS 인덱스 생성 및 벡터 저장
    index = faiss.IndexFlatL2(d) # L2 거리 기반의 평면 인덱스 생성
    index.add(embeddings.astype('float32')) # FAISS는 float32를 기대합니다.
    
    # 5. FAISS 인덱스 파일 저장
    faiss.write_index(index, FAISS_PATH)
    print(f"✅ 4. FAISS 인덱스 저장 완료. ({FAISS_PATH})")

    # 6. 원본 텍스트 및 메타데이터 캐시 파일 저장 (FAISS는 텍스트를 저장하지 않으므로 필수)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ 5. 원본 텍스트 캐시 저장 완료. ({CACHE_PATH})")
    
    print("\n--- 인덱싱 완료: RAG 검색 준비 완료 ---")


if __name__ == '__main__':
    create_policy_index()