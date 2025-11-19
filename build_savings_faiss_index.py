import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# 1️⃣ Qwen 임베딩 래퍼 (saving_agent.py 와 동일하게 맞추기)
# ============================================================
class QwenHFEmbeddings(Embeddings):
    """
    HuggingFace InferenceClient + Qwen/Qwen3-Embedding-8B 를
    LangChain Embeddings 인터페이스로 감싼 래퍼.
    """

    def __init__(self, api_key: str, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        self.client = InferenceClient(provider="nebius", api_key=api_key)
        self.model_name = model_name

    def _embed(self, text: str) -> List[float]:
        """
        HF InferenceClient.feature_extraction 결과를
        항상 1차원 list[float] (dim,) 로 변환.
        """
        out = self.client.feature_extraction(text, model=self.model_name)

        # case 1: numpy array
        if isinstance(out, np.ndarray):
            if out.ndim == 2:      # (1, dim) → 첫 row만 사용
                out = out[0]
            return out.astype(float).tolist()

        # case 2: list of lists or list of arrays
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], (list, np.ndarray)):
            first = out[0]
            if isinstance(first, np.ndarray):
                return first.astype(float).tolist()
            return [float(x) for x in first]

        # case 3: 이미 1차원 리스트
        return [float(x) for x in out]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]


# ============================================================
# 2️⃣ JSON 로드 & 예금/적금 분리
#    - 상품명 + 개요 둘 다에서 '예금' / '적금' 검색
# ============================================================
def load_products(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    하나의 JSON 안에 예금/적금이 섞여 있고,
    상품명 또는 '개요'에 '예금', '적금' 이 들어 있다고 가정.
    예:
      - 상품명: 'WON플러스 예금'
      - 개요: '청년을 위한 고금리 적금 상품'
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    deposits: List[Dict[str, Any]] = []
    savings: List[Dict[str, Any]] = []

    for item in data:
        name = item.get("상품명") or item.get("name") or ""
        overview = item.get("개요") or item.get("overview") or ""
        # 혹시 product_type이 이미 있다면 활용
        ptype = item.get("product_type", "")

        # 🔹 예금/적금 판별용 텍스트: 상품명 + 개요
        type_text = f"{name} {overview}"

        # 1) product_type이 명시돼 있으면 그걸 최우선으로 사용
        if ptype == "예금":
            item["product_type"] = "예금"
            deposits.append(item)
            continue
        elif ptype == "적금":
            item["product_type"] = "적금"
            savings.append(item)
            continue

        # 2) product_type이 없으면, 상품명 + 개요에서 '예금' / '적금' 검색
        #    - 둘 다 있는 경우는 애매하니 경고 출력하고 스킵
        has_deposit = "예금" in type_text
        has_saving = "적금" in type_text

        if has_deposit and not has_saving:
            item["product_type"] = "예금"
            deposits.append(item)
        elif has_saving and not has_deposit:
            item["product_type"] = "적금"
            savings.append(item)
        elif has_deposit and has_saving:
            # 예: "예금/적금 겸용 상품" 같은 애매한 케이스
            print(f"[WARN] 예금/적금 키워드가 둘 다 포함되어 있어 스킵: {name}")
        else:
            # 둘 다 안 걸리면 일단 스킵 (필요하면 기본값 규칙 추가 가능)
            print(f"[WARN] 예금/적금 구분 키워드 없음, 스킵: {name}")

    print(f"✅ JSON 로드 완료: 총 {len(data)}개 (예금 {len(deposits)}개, 적금 {len(savings)}개)")
    return {"deposits": deposits, "savings": savings}


# ============================================================
# 3️⃣ Document 생성 유틸
# ============================================================
def build_documents(items: List[Dict[str, Any]]) -> List[Document]:
    """
    각 상품 dict → LangChain Document
    page_content: 검색용으로 적당한 텍스트 (이름 + 개요 + 특징 등)
    metadata: 원본 dict 전체 + name/product_type/max_rate 등 정리
    """
    docs: List[Document] = []

    for item in items:
        name = item.get("상품명") or item.get("name") or ""
        overview = item.get("개요") or item.get("overview") or ""
        feature = item.get("특징") or item.get("feature") or ""
        etc = item.get("기타") or ""

        # 검색용 컨텐츠 (필요하면 원하는 필드 더 이어붙여도 됨)
        content_parts = [str(name), str(overview), str(feature), str(etc)]
        page_content = "\n".join([p for p in content_parts if p])

        # max_rate 필드 이름이 다르면 여기서 맞춰줌
        max_rate = (
            item.get("최고우대금리") 
            or item.get("최고금리") 
            or item.get("max_rate")
        )

        metadata = dict(item)  # 원본 전체 메타 복사
        metadata.setdefault("name", name)
        metadata.setdefault("max_rate", max_rate)

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


# ============================================================
# 4️⃣ 메인: FAISS 인덱스 생성 & 저장
# ============================================================
def main():
    load_dotenv()

    # 1) HF 토큰 로드
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN 환경 변수가 없습니다. .env에 HF_TOKEN=... 추가해주세요.")

    embedding = QwenHFEmbeddings(api_key=hf_token)

    # 2) 프로젝트 루트 기준 경로 설정
    #   - 이 스크립트를 /agent 또는 /agent/plan_agents 에 두는 위치에 따라 조정
    project_root = Path(__file__).resolve().parents[1]  # 필요하면 1,2,3 바꿔쓰기
    print(f"🔹 project_root: {project_root}")

    # 👉 여기 JSON 경로를 네 실제 파일명에 맞게 수정하면 됨
    PRODUCT_JSON_PATH = project_root / "data" / "우리은행_저축상품.json"

    if not PRODUCT_JSON_PATH.exists():
        raise FileNotFoundError(f"상품 JSON 파일을 찾을 수 없습니다: {PRODUCT_JSON_PATH}")

    # 3) JSON 로드 후 예금/적금 분리
    products = load_products(PRODUCT_JSON_PATH)
    deposit_items = products["deposits"]
    saving_items = products["savings"]

    deposit_docs = build_documents(deposit_items)
    saving_docs = build_documents(saving_items)

    print(f"🔹 예금 Document 수: {len(deposit_docs)}")
    print(f"🔹 적금 Document 수: {len(saving_docs)}")

    # 4) FAISS 인덱스 생성
    print("⏳ 예금 FAISS 인덱스 생성 중...")
    deposit_vs = FAISS.from_documents(deposit_docs, embedding)
    print("⏳ 적금 FAISS 인덱스 생성 중...")
    saving_vs = FAISS.from_documents(saving_docs, embedding)

    # 5) 인덱스 차원 로그 (추후 saving_agent와 맞는지 확인용)
    print(f"✅ 예금 index dim: {deposit_vs.index.d}")
    print(f"✅ 적금 index dim: {saving_vs.index.d}")

    # 6) 저장 경로 설정 (saving_agent.py에서 로드하는 경로와 동일하게!)
    deposit_index_dir = project_root / "faiss_deposit_products"
    saving_index_dir = project_root / "faiss_saving_products"

    print(f"💾 예금 인덱스 저장: {deposit_index_dir}")
    deposit_vs.save_local(str(deposit_index_dir))
    print(f"💾 적금 인덱스 저장: {saving_index_dir}")
    saving_vs.save_local(str(saving_index_dir))

    print("🎉 예/적금 FAISS 인덱스 생성 및 저장 완료!")


if __name__ == "__main__":
    main()
