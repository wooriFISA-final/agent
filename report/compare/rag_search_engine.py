# report_project/compare/rag_search_engine.py

import fitz # PyMuPDF
import re
import os
from typing import List, Dict, Any, Optional
import traceback

# [패턴] '제'로 시작하고 '장' 또는 '조'로 끝나는 패턴
# 🚨 [수정된 패턴] 모든 소항목 시작점을 분할 기준으로 잡습니다.
SECTION_PATTERN = re.compile(
    r'^(제\s*\d+\s*[장조]\s*.*)$'      # 제N장/제N조
    r'|^(\d+\.\s*[\(]?[가-힣]*.*)$'   # N. (용어의 정의)
    r'|^([가-힣]\.\s*.*)$'            # 가. 나. 다.
    r'|^(\(\d+\)\s*.*)$'              # (1) (2) (3)
    r'|^(\s*\-\s*.*)$',               # - 마커까지 분할
    re.MULTILINE | re.IGNORECASE
)

# 🚨 [신규 함수] 공백 없이 붙은 한글 텍스트에 띄어쓰기를 복원하는 함수
def restore_spacing(text: str) -> str:
    """한글, 숫자, 영문이 붙어 있을 때 경계에 공백을 삽입하여 가독성을 높입니다."""
    if not text:
        return ""
    
    # 예: '은행은신규주택담보대출' -> '은행은 신규 주택담보대출'
    # 1. 한글/숫자/알파벳이 경계 없이 붙어있을 경우 사이에 공백 삽입
    text = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', text)
    text = re.sub(r'([가-힣])([가-힣])', r'\1 \2', text) # 한글끼리 붙은 경우 (필요 시 주석 처리)
    
    # 2. 쉼표, 마침표, 괄호 뒤에 공백이 없으면 삽입 (가독성 향상)
    text = re.sub(r'([.,])([가-힣A-Za-z0-9])', r'\1 \2', text)
    
    # 3. 연속된 공백 제거 및 정리
    return ' '.join(text.split())

def get_policy_chapters(pdf_path: str) -> List[Dict[str, str]]:
    """
    정책 PDF 파일에서 텍스트를 추출하고, '제1장'을 건너뛴 후 장별 내용을 분할하여 반환합니다.
    """
    print(f"📖 PDF 파일에서 정책 장 분할 시작: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ DEBUG: 파일 존재하지 않음!")
        return []
    
    try:
        doc = fitz.open(pdf_path) 
        full_text = ""
        
        # 1. 텍스트 추출 및 메타데이터 제거
        for page in doc:
            text = page.get_text()
            lines = []
            for line in text.split('\n'):
                # 메타데이터 제거 로직 유지
                if line.strip().lower().startswith('http') or '별표·서식' in line:
                    continue
                lines.append(line.strip())
            full_text += "\n".join(lines) + "\n"

        if len(full_text.strip()) < 100: 
             print(f"⚠️ 경고: 텍스트가 너무 짧거나 깨졌을 수 있습니다.")
             raise Exception("PDF 텍스트 추출 내용 부족.")
        
        # 2. '제1장' 패턴을 찾아 클린 텍스트 확보
        first_chapter_match = re.search(r'제\s*1\s*장', full_text)
        
        if first_chapter_match:
            clean_text = full_text[first_chapter_match.start():].strip()
        else:
            clean_text = full_text.strip()
            print("❌ DEBUG: '제1장' 패턴을 찾을 수 없습니다.")

        # 3. 정규식 매칭 및 제1장 건너뛰기 로직
        matches = list(SECTION_PATTERN.finditer(clean_text))
        
        if len(matches) < 2:
            print(f"❌ DEBUG: 제2장 이후의 제목을 찾을 수 없습니다. (매치 개수: {len(matches)})")
            return []

        chapters = []
        for i in range(1, len(matches)):
            title = matches[i].group(0).strip()
            start_pos = matches[i].end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(clean_text)
            
            # 🚨 [핵심 수정] 추출된 content에 띄어쓰기 복원 함수 적용
            content_raw = clean_text[start_pos: end_pos].strip()
            content_restored = restore_spacing(content_raw)
            
            chapters.append({"title": title, "content": content_restored})

        print(f"✅ 총 {len(chapters)}개의 장/조항 덩어리 분할 완료 (제1장 건너뜀).")
        return chapters
        
    except Exception as e:
        print(f"❌ PDF 처리 중 치명적 오류 발생: {e}")
        traceback.print_exc()
        return []

if __name__ == '__main__':
    # 🚨 테스트를 위해 compare 폴더 내의 data 폴더에 PDF 파일이 있어야 합니다.
    DUMMY_PDF_PATH = os.path.join("data", "20250305.pdf") 
    
    print("\n=========================================")
    print("STARTING PDF EXTRACTION AND SPACING TEST")
    print("=========================================")
    
    chapters = get_policy_chapters(DUMMY_PDF_PATH)
    
    if not chapters:
        print("\n❌ 심각한 오류: 최종 청크 분할에 실패했습니다.")
        
    for i, chap in enumerate(chapters):
        print(f"\n--- 청크 {i+1} ---")
        print(f"제목: {chap['title']}")
        print(f"내용:\n{chap['content'][:500]}...")
        print("-" * 20)