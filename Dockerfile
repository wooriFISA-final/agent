# ========================================
# Stage 1: Builder
# ========================================
FROM python:3.11-slim AS builder

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY pyproject.toml uv.lock ./

# 의존성 설치 (uv 사용, 개발 의존성 제외)
RUN uv sync --frozen --no-dev

# ========================================
# Stage 2: Runtime
# ========================================
FROM python:3.11-slim

# 비root 사용자 생성
RUN useradd -m -u 1000 appuser

# 작업 디렉토리 설정
WORKDIR /app

# Builder 스테이지에서 가상환경 복사
COPY --from=builder /app/.venv /app/.venv

# 애플리케이션 코드 복사
COPY --chown=appuser:appuser . .

# 로그 디렉토리 생성
RUN mkdir -p logs && chown appuser:appuser logs

# 비root 사용자로 전환
USER appuser

# 환경 변수 설정
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 헬스체크 (컨테이너 내부에서만 실행, 보안상 안전)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost',8080)); s.close()" || exit 1

# 포트 노출
EXPOSE 8080

# 실행 명령
CMD ["python", "main.py"]
