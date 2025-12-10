# 🏦 WooriZip Agent - Multi-Agent Financial Planning System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.121+-green?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-1.0+-orange?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/AWS-Bedrock-orange?logo=amazon-aws&logoColor=white" alt="AWS Bedrock">
</p>

<p align="center">
  AI 기반 멀티 에이전트 시스템으로, LangGraph와 FastAPI를 활용한<br/>
  <strong>재무 계획 및 금융 상품 추천</strong> 대화형 AI 서비스입니다.
</p>

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#%EF%B8%8F-configuration)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🤖 Multi-Agent System
- **Supervisor Agent** - 전체 워크플로우 조율 및 에이전트 라우팅
- **Input Agent** - 사용자 입력 분석 및 의도 파악
- **Saving Agent** - 예금/적금 상품 추천
- **Loan Agent** - 대출 상품 분석 및 추천
- **Fund Agent** - 펀드 투자 상품 추천
- **Summary Agent** - 종합 재무 리포트 생성
- **Validation Agent** - 입력 검증 및 품질 보증

### 🔧 Core Capabilities
- 📊 **Plan Graph** - 개인 맞춤형 재무 계획 수립
- 📝 **Report Graph** - 상세 금융 리포트 생성
- 💬 **Real-time Chat** - 스트리밍 대화 지원
- 🔄 **Session Management** - 대화 히스토리 관리
- 🔌 **MCP Integration** - Model Context Protocol 기반 도구 연동

---

## 🏗 Architecture
<img width="20446" height="15864" alt="Image" src="https://github.com/user-attachments/assets/adc46a8c-c3e4-4ba5-a905-fec58e085255" />

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip
- Docker & Docker Compose (배포 시)
- AWS 계정 (Bedrock 사용 시)

### 30초 시작하기

```bash
# 1. 저장소 클론
git clone https://github.com/your-org/woorizip-agent.git
cd agent

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일 편집하여 AWS 토큰 등 설정

# 3. 의존성 설치 및 실행
make install
make dev
```

서버가 시작되면 `http://localhost:8080/docs`에서 API 문서를 확인할 수 있습니다.

---

## 📦 Installation

### Using uv (권장)

```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv sync

# 개발 서버 실행
uv run python main.py
```

### Using pip

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

---

## ⚙️ Configuration

### 환경별 설정

| 환경 | AGENT_ENVIRONMENT | AGENT_DEBUG | 설명 |
|------|-------------------|-------------|------|
| 개발 | `development` | `True` | 상세 로깅, 핫 리로드 |
| 스테이징 | `staging` | `False` | 프로덕션 유사 환경 |
| 프로덕션 | `production` | `False` | 최적화된 설정 |

---

## 📖 API Reference

### Base URL
```
http://localhost:8080
```

### Endpoints

#### 🔹 기본 엔드포인트

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API 정보 |
| `GET` | `/health` | 헬스체크 |

#### 🔹 채팅 엔드포인트

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | 기본 그래프 사용 (하위 호환성) |
| `POST` | `/chat/plan` | Plan 그래프 - 재무 계획 수립 |
| `POST` | `/chat/report` | Report 그래프 - 리포트 생성 |

##### 요청 예시


##### 응답 예시


#### 🔹 세션 관리

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/chat/sessions` | 활성 세션 목록 |
| `GET` | `/chat/session/{id}/history` | 대화 히스토리 조회 |
| `DELETE` | `/chat/session/{id}` | 세션 삭제 |

---

## 🛠 Development

### 사용 가능한 Make 명령어

```bash
# 도움말
make help

# 개발
make install       # 의존성 설치
make dev           # 개발 서버 (핫 리로드)
make run           # 서버 실행

# 코드 품질
make lint          # 린트 검사
make format        # 코드 포맷팅
make type-check    # 타입 검사

# 테스트
make test          # 전체 테스트
make test-cov      # 커버리지 포함

# 정리
make clean         # 캐시 정리
```

### 코드 스타일

```bash
# Ruff (린터 & 포맷터)
ruff check .
ruff format .

# 타입 검사
mypy .
```

---

## 🐳 Docker Deployment

### 빠른 배포

```bash
# 환경 변수 설정
cp .env.example .env
# .env 파일 수정

# 배포 (빌드 + 실행)
make deploy
# 또는
./deploy.sh
```

### 수동 Docker 명령어

```bash
# 이미지 빌드
docker build -t woorizip-agent:latest .

# 컨테이너 실행
docker run -d \
  --name agent \
  -p 8080:8080 \
  --env-file .env \
  woorizip-agent:latest

# 로그 확인
docker logs -f agent

# 컨테이너 중지 및 삭제
docker stop agent && docker rm agent
```

### Docker Compose

```bash
# 시작
docker-compose up -d

# 로그
docker-compose logs -f

# 중지
docker-compose down

# 재빌드 후 시작
docker-compose up -d --build
```

### Make Docker 명령어

```bash
make docker-build   # 이미지 빌드
make docker-run     # 컨테이너 실행
make docker-stop    # 컨테이너 중지
make docker-logs    # 로그 확인
make docker-clean   # 리소스 정리
```

---

## 📁 Project Structure

```
agent/
├── main.py                     # 🚀 서버 엔트리포인트
│
├── agents/                     # 🤖 Agent 구현
│   ├── base/                   # 베이스 클래스
│   │   └── base_agent.py       # AbstractAgent
│   ├── config/                 # Agent 설정
│   │   ├── agent_config.py     # 설정 클래스
│   │   └── prompts/            # 프롬프트 템플릿
│   ├── implementations/        # 구체적인 Agent 구현
│   │   ├── input_agent.py      # 사용자 정보 입력
│   │   ├── saving_agent.py     # 예적금 추천
│   │   ├── loan_agent.py       # 대출 추천
│   │   ├── fund_agent.py       # 펀드 추천
│   │   ├── summary_agent.py    # 요약 생성
│   │   ├── supervisor_agent.py # 워크플로우 관리
│   │   └── validation_agent.py # 응답 검증
│   └── registry/               # Agent 레지스트리
│
├── api/                        # 🌐 FastAPI 관련
│   ├── app.py                  # FastAPI 앱 설정
│   ├── lifespan.py             # 앱 라이프사이클
│   ├── models/                 # Pydantic 모델
│   │   ├── request.py          # 요청 모델
│   │   └── response.py         # 응답 모델
│   └── routes/                 # API 라우트
│       ├── chat.py             # 채팅 엔드포인트
│       └── health.py           # 헬스체크
│
├── core/                       # ⚙️ 핵심 기능
│   ├── config/                 # 전역 설정
│   │   └── settings.py         # Pydantic Settings
│   ├── llm/                    # LLM 관련
│   │   └── bedrock.py          # AWS Bedrock 클라이언트
│   ├── logging/                # 로깅
│   │   └── logger.py           # 커스텀 로거
│   └── mcp/                    # MCP 연동
│       └── client.py           # MCP 클라이언트
│
├── graph/                      # 📊 LangGraph 관련
│   ├── builder/                # 그래프 빌더
│   ├── config/                 # 그래프 설정
│   ├── factory.py              # 그래프 팩토리
│   └── routing/                # 라우팅 로직
│
├── utils/                      # 🔧 유틸리티
├── tests/                      # 🧪 테스트
├── logs/                       # 📝 로그 파일
│
├── Dockerfile                  # Docker 빌드 설정
├── docker-compose.yml          # Docker Compose 설정
├── pyproject.toml              # 프로젝트 메타데이터
├── requirements.txt            # 의존성 목록
├── Makefile                    # Make 명령어
└── .env.example                # 환경 변수 템플릿
```

---

## 🧪 Testing

### 테스트 실행

```bash
# 전체 테스트
make test

# 특정 테스트 파일
pytest tests/test_agents.py -v

# 커버리지 리포트
make test-cov
pytest --cov=. --cov-report=html
```

### 테스트 구조

```
tests/
├── test_agents/           # Agent 단위 테스트
├── test_api/              # API 엔드포인트 테스트
├── test_graph/            # Graph 통합 테스트
└── conftest.py            # 공통 fixtures
```

---

## 🔒 Security

- ✅ 비root 사용자로 컨테이너 실행
- ✅ 환경 변수로 민감한 정보 관리
- ✅ 설정 파일은 읽기 전용으로 마운트
- ✅ 헬스체크로 컨테이너 상태 모니터링
- ✅ AWS IAM 기반 인증

---
### Commit Convention

```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅
refactor: 코드 리팩토링
test: 테스트 코드
chore: 빌드, 설정 변경
```

---
<p align="center">
  Made with by WooriFisa Team 6
</p>
