import re
import ollama
from difflib import get_close_matches
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


# ------------------------------------------------
# 환경 설정
# ------------------------------------------------
load_dotenv()
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_NAME = os.getenv("database")


class ValidationAgent:
    """
    ValidationAgent — DB 기반 지역 검증 + 입력값 정제 + 현실성 검증
    (서울은 구 단위까지, 그 외 지역은 시 단위 평균 기준)
    """

    def __init__(self, llm_model="qwen3:8b"):
        self.llm_model = llm_model

        # DB 연결 초기화
        self.engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
        self.valid_locations = self._load_valid_locations()

    # ------------------------------------------------
    # DB에서 지역명 로드
    # ------------------------------------------------
    def _load_valid_locations(self):
        """state 테이블에서 region_nm 목록 불러오기"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT region_nm FROM state"))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            print(f"DB에서 지역명 로드 실패: {e}")
            return []

    # ------------------------------------------------
    # LLM 피드백
    # ------------------------------------------------
    def _ask_llm(self, message: str):
        """잘못된 입력일 때 피드백"""
        try:
            res = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "너는 입력 검증 AI야. "
                            "간결하고 공손하게 문제점을 설명하고, 마지막엔 '다시 입력해주세요.'로 끝내."
                        ),
                    },
                    {"role": "user", "content": message},
                ],
            )
            print(res["message"]["content"].strip())
        except Exception as e:
            print(f"LLM 오류: {e}")

    # ------------------------------------------------
    # 행정명 정규화 (예: '서울 송파구' → '서울특별시 송파구')
    # ------------------------------------------------
    def _normalize_location(self, loc: str) -> str:
        loc = loc.strip()

        # 1️⃣ 기본 시·도 매핑
        mapping = {
            "서울": "서울특별시",
            "부산": "부산광역시",
            "대구": "대구광역시",
            "인천": "인천광역시",
            "광주": "광주광역시",
            "대전": "대전광역시",
            "울산": "울산광역시",
            "세종": "세종특별자치시",
            "경기": "경기도",
            "강원": "강원특별자치도",
            "충북": "충청북도",
            "충남": "충청남도",
            "전북": "전북특별자치도",
            "전남": "전라남도",
            "경북": "경상북도",
            "경남": "경상남도",
            "제주": "제주특별자치도",
        }

        for short, full in mapping.items():
            if loc.startswith(short):
                loc = loc.replace(short, full, 1)
                break

        # 2️⃣ 서울 자치구 단독 입력 자동 보정
        seoul_districts = [
            "강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천", "노원", "도봉", "동대문",
            "동작", "마포", "서대문", "서초", "성동", "성북", "송파", "양천", "영등포",
            "용산", "은평", "종로", "중", "중랑"
        ]
        for gu in seoul_districts:
            if loc.startswith(gu):  # 예: "송파"
                loc = f"서울특별시 {gu}구"
                print(f"입력하신 '{gu}'은(는) '서울특별시 {gu}구'로 인식되었습니다.")
                break

        return loc

    # ------------------------------------------------
    # 서울 외 지역은 시 단위까지만 남기기
    # ------------------------------------------------
    def _simplify_non_seoul(self, loc: str) -> str:
        """서울특별시 제외, 시 단위까지만 남김"""
        if loc.startswith("서울"):
            return loc  # 서울은 구 단위 유지
        match = re.match(r"^(\S+시|\S+특별자치시|\S+도)", loc)
        if match:
            return match.group(1)
        return loc

    # ------------------------------------------------
    # 지역 검증 (서울=구 단위 / 타지역=시 단위)
    # ------------------------------------------------
    def _check_location(self, responses: dict) -> bool:
        loc = responses.get("target_location", "")
        normalized = self._normalize_location(loc)
        simplified = self._simplify_non_seoul(normalized)

        # ✅ 서울특별시는 구 단위로 검증
        if normalized.startswith("서울"):
            target_to_check = normalized
        else:
            target_to_check = simplified

        # 완전 일치 시 통과
        if target_to_check in self.valid_locations:
            responses["target_location"] = target_to_check
            return True

        # 부분 일치 / 유사도 기반 보정
        matches = get_close_matches(target_to_check, self.valid_locations, n=1, cutoff=0.6)
        if matches:
            corrected = matches[0]
            print(f"입력한 '{loc}'을(를) '{corrected}'로 인식했습니다.")
            responses["target_location"] = corrected
            return True

        # 검증 실패 시 LLM 피드백
        if normalized.startswith("서울"):
            self._ask_llm(f"지역명을 ‘서울 송파구’처럼 정확히 입력해주세요. 입력값: {loc}")
        else:
            self._ask_llm(f"서울 외 지역은 ‘부산광역시’, ‘세종특별자치시’처럼 시 단위까지만 입력해주세요. 입력값: {loc}")
        return False

    # ------------------------------------------------
    # 입력 정제
    # ------------------------------------------------
    def _sanitize_text(self, text: str) -> str:
        """불필요한 기호, 단위 제거"""
        return re.sub(r"[^\w\s]", "", str(text)).replace("원", "").strip()

    # ------------------------------------------------
    # 형식 검증
    # ------------------------------------------------
    def _check_format(self, responses: dict) -> bool:
        for key, val in responses.items():
            if not val or str(val).strip() == "":
                self._ask_llm(f"{key} 값이 비어 있습니다. 다시 입력해주세요.")
                return False
            if re.search(r"-", str(val)):
                self._ask_llm(f"{key}에는 음수를 입력할 수 없습니다. 다시 입력해주세요.")
                return False
        return True

    # ------------------------------------------------
    # 전체 실행
    # ------------------------------------------------
    def run(self, responses: dict) -> bool:
        """전체 검증 실행"""
        # 텍스트 정제
        responses = {k: self._sanitize_text(v) for k, v in responses.items()}

        # 형식 검증
        if not self._check_format(responses):
            return False

        # 지역 검증
        if not self._check_location(responses):
            return False

        print("모든 입력 검증 통과")
        return True