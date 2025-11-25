# report_project/compare/policy_retriever.py

import os
import faiss
import json
import numpy as np
from typing import List, Dict, Any, Optional 
from sentence_transformers import SentenceTransformer
import traceback

# --- 설정 ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CACHE_FILE_NAME = 'policy_cache.json'
FAISS_INDEX_FILE = 'policy_faiss.index'

# --- 경로 설정 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data') # compare/data/
CACHE_PATH = os.path.join(DATA_DIR, CACHE_FILE_NAME)
FAISS_PATH = os.path.join(DATA_DIR, FAISS_INDEX_FILE)

# 🚨 전역 변수: 모델과 인덱스를 메모리에 캐시합니다.
MODEL = None
INDEX = None
CACHE = None

def load_rag_assets():
    """모델과 FAISS 인덱스, 캐시 파일을 메모리에 로드합니다."""
    global MODEL, INDEX, CACHE
    if INDEX is None:
        print("⏳ [RAG DEBUG] RAG Assets 로드 시작...")
        try:
            # 1. 모델 로드 (가장 무거운 작업)
            print("⏳ [RAG DEBUG] 1. SentenceTransformer 로드 중...")
            if MODEL is None:
                MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
            
            # 2. FAISS 인덱스 로드
            print(f"⏳ [RAG DEBUG] 2. FAISS 인덱스 로드 시도: {FAISS_PATH}")
            INDEX = faiss.read_index(FAISS_PATH)
            
            # 3. 캐시 로드
            print(f"⏳ [RAG DEBUG] 3. 원본 캐시 파일 로드 시도: {CACHE_PATH}")
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                CACHE = json.load(f)
            
            print(f"✅ RAG Assets 로드 완료. (총 {len(CACHE)} 청크)")
        except Exception as e:
            print(f"\n\n--- ❌ CRITICAL RAG LOAD FAILURE ---")
            print(f"FATAL ERROR: {type(e).__name__} - {e}")
            print("--------------------------------------\n")
            # 🚨 오류를 다시 던져서 Compare Agent 실행을 멈춥니다.
            raise RuntimeError(f"RAG Assets 로드 실패: {e}")

def retrieve_policy_changes(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    쿼리를 임베딩하고 FAISS에서 가장 유사한 정책 청크를 검색합니다.
    (k의 기본값을 10으로 늘려, 더 많은 컨텍스트를 LLM에 전달합니다.)
    """
    load_rag_assets() # 자산 로드 실행
    
    if INDEX is None or MODEL is None or CACHE is None:
        return [{"title": "ERROR", "content": "RAG 시스템 로드 실패."}]

    try:
        # 1. 쿼리 임베딩
        query_vector = MODEL.encode([query], convert_to_numpy=True)
        
        # 2. FAISS 검색 (k=10 사용)
        D, I = INDEX.search(query_vector.astype('float32'), k) 
        
        # 3. 검색된 ID를 기반으로 원본 텍스트(캐시) 추출
        results = []
        for rank, index_id in enumerate(I[0]):
            # FAISS 인덱스 위치(index_id)는 CACHE 리스트의 인덱스와 동일합니다.
            chunk = CACHE[index_id] 
            
            # LLM에 전달할 결과 포맷 구성
            results.append({
                "title": f"[{chunk['version']} | {chunk['title']}]",
                "content": chunk['content'],
                "score": float(D[0][rank]) # 유사도 점수
            })
            
        return results
    
    except Exception as e:
        print(f"❌ RAG 검색 중 오류 발생: {e}")
        return [{"title": "ERROR", "content": f"검색 오류: {e}"}]


if __name__ == '__main__':
    # 🚨 테스트 쿼리
    test_query = "2024년 12월 정책과 2025년 3월 정책 사이의 LTV 규정의 변경 사항은 무엇인가?"
    
    print(f"--- RAG 검색 테스트 시작 ---\n쿼리: {test_query}")
    
    try:
        # k=8로 호출하여 검색 결과를 확인
        retrieved_chunks = retrieve_policy_changes(test_query, k=8) 
        
        if not retrieved_chunks:
             print("\n❌ RAG 검색 결과가 없습니다. 인덱싱 파일을 확인하세요.")
             
        for chunk in retrieved_chunks:
            print(f"\n[검색 결과] Score: {chunk['score']:.4f} Title: {chunk['title']}")
            print(f"내용: {chunk['content'][:150]}...")
    except RuntimeError as e:
        print(f"\n❌ RAG 테스트 실패: {e}")