# Multi-Agent System

AI 기반 멀티 에이전트 시스템으로 LangGraph와 FastAPI를 활용한 대화형 AI 서비스입니다.

## 🚀 빠른 시작

### 개발 환경

```bash
# 1. 의존성 설치
make install

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 설정 수정

# 3. 개발 서버 실행
make dev
```

### 프로덕션 배포 (Docker)

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 프로덕션 설정 수정

# 2. 배포
make deploy

# 또는
./deploy.sh
```

## 📦 요구사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (권장)
- Docker & Docker Compose (배포 시)

## 🛠️ 개발

### 사용 가능한 명령어

```bash
make help           # 모든 명령어 보기
make install        # 의존성 설치
make dev            # 개발 서버 실행
make test           # 테스트 실행
make lint           # 코드 린트
make format         # 코드 포맷팅
make clean          # 캐시 정리
```

### Docker 명령어

```bash
make docker-build   # Docker 이미지 빌드
make docker-run     # 컨테이너 실행
make docker-stop    # 컨테이너 중지
make docker-logs    # 로그 확인
make docker-clean   # Docker 리소스 정리
```

## 🌐 API 엔드포인트

서버 실행 후 다음 엔드포인트를 사용할 수 있습니다:

### 기본 엔드포인트
- `GET /` - API 정보
- `GET /health` - 헬스체크

### 채팅 엔드포인트 (멀티 그래프 지원)
- `POST /chat` - 기본 그래프 사용 (하위 호환성)
- `POST /chat/plan` - Plan 그래프 사용 (재무 계획)
- `POST /chat/report` - Report 그래프 사용 (리포트 생성)

### 세션 관리
- `GET /chat/sessions` - 세션 목록
- `GET /chat/session/{id}/history` - 대화 히스토리
- `DELETE /chat/session/{id}` - 세션 삭제

기본 포트: `http://localhost:8080`

## 📁 프로젝트 구조

```
agent/
├── main.py                 # 서버 실행
├── agents/                 # Agent 구현
│   ├── base/              # 베이스 클래스
│   ├── config/            # Agent 설정
│   ├── implementations/   # 구체적인 Agent
│   └── registry/          # Agent 등록
├── api/                   # FastAPI 관련
│   ├── app.py            # 앱 설정
│   ├── lifespan.py       # 라이프사이클
│   ├── models/           # Pydantic 모델
│   └── routes/           # 라우트 핸들러
├── core/                  # 핵심 기능
│   ├── config/           # 전역 설정
│   ├── llm/              # LLM 관련
│   ├── logging/          # 로깅
│   └── mcp/              # MCP 관련
├── graph/                 # LangGraph 관련
├── utils/                 # 유틸리티
└── config/                # 설정 파일 (YAML)
```

## ⚙️ 환경 변수

주요 환경 변수는 `.env` 파일에서 설정합니다:

```bash
# 환경
AGENT_ENVIRONMENT=production
AGENT_DEBUG=false

# API 서버
AGENT_API_HOST=0.0.0.0
AGENT_API_PORT=8080

# LLM 설정 (AWS Bedrock)
AGENT_AWS_REGION=us-east-1
AGENT_AWS_BEARER_TOKEN_BEDROCK=your-token-here
AGENT_BEDROCK_MODEL_ID=openai.gpt-oss-20b-1:0

# MCP 서버
AGENT_MCP_URL=http://localhost:3000

# Agent 모듈
AGENT_AGENTS_MODULE_PATH=agents.implementations
```

자세한 설정은 `.env.example` 파일을 참고하세요.

## 🐳 Docker 배포

### 빌드 및 실행

```bash
# 이미지 빌드
docker build -t multi-agent-system:latest .

# 컨테이너 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down

# make 
make deploy
```

### 포트 변경

`.env` 파일에서 포트를 변경할 수 있습니다:

```bash
AGENT_API_PORT=9000  # 외부 포트를 9000으로 변경
```

## 🔒 보안

- 비root 사용자로 컨테이너 실행
- 환경 변수로 민감한 정보 관리
- 설정 파일은 읽기 전용으로 마운트
- 헬스체크로 컨테이너 상태 모니터링

## 📝 라이선스

MIT License
