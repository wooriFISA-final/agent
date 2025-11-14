import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# 🚨 [수정] run_full_report_pipeline 임포트를 복구/활성화합니다.
from report.main_orchestrator import run_full_report_pipeline

# 🚨 [경로 설정] 파이프라인 호출에 필요할 수 있으므로 유지합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 


app = FastAPI()

# ----------------------------------------------------
# CORS 미들웨어 설정 (유지)
# ----------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.0.184:3000", 
    "http://localhost:5174",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 프론트엔드 요청 본문을 정의
class UserInput(BaseModel):
    member_id: int = 1004
    user_id: int = 500

# OPTIONS 요청 라우터 (유지)
@app.options("/api/v1/generate-report")
async def options_report():
    return {"status": "ok"}


@app.post("/api/v1/generate-report")
def generate_report(input_data: UserInput):
    """
    🚨 [LLM 호출 복구] 에이전트 실행 파이프라인을 호출하여 분석 결과를 반환합니다.
    """
    try:
        # 🚨 [핵심 수정] run_full_report_pipeline 호출 로직 복구
        print("\n--- Starting LLM Agent Pipeline ---")
        final_result_dict = run_full_report_pipeline(
            member_id=input_data.member_id,
            user_id=input_data.user_id,
            ollama_model="qwen3:8b" 
        )
        print("--- LLM Agent Pipeline Finished ---\n")

        # 2. 최종 결과 반환
       # 🚨 [핵심 수정] 최종 결과 반환 필드를 구조화된 필드로 조정
        return {
            "status": "success",
            "report_data": final_result_dict, # ⬅️ 구조화된 모든 데이터를 report_data 필드에 담아 전송
            "summary": final_result_dict.get("compare_changes") # 요약 필드 (옵션)
        }

    except Exception as e:
        # LLM 실행 및 직렬화 오류 포착
        print("\n!!! CRITICAL AGENT EXECUTION ERROR !!!")
        print(f"Error Type: {type(e).__name__}, Detail: {str(e)}")
        print("!!! CRITICAL AGENT EXECUTION ERROR !!!\n")
        
        # 프론트엔드에는 500 오류를 반환합니다.
        # 프론트엔드에서 이 detail 메시지를 볼 수 있도록 수정 (Reports.tsx의 catch 블록에서 처리됨)
        raise HTTPException(status_code=500, detail=f"에이전트 실행 중 서버 내부 오류 발생: {type(e).__name__}")