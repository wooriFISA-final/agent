.PHONY: help setup install dev test lint format clean build run docker-build docker-run docker-stop docker-logs docker-clean deploy

# 기본 타겟
help:
	@echo "🚀 Multi-Agent System - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make setup        - 개발 환경 설정 (uv 설치)"
	@echo "  make install      - 의존성 설치"
	@echo "  make dev          - 개발 서버 실행"
	@echo "  make test         - 테스트 실행"
	@echo "  make lint         - 코드 린트"
	@echo "  make format       - 코드 포맷팅"
	@echo "  make clean        - 캐시 파일 정리"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Docker 이미지 빌드"
	@echo "  make docker-run   - Docker 컨테이너 실행"
	@echo "  make docker-stop  - Docker 컨테이너 중지"
	@echo "  make docker-logs  - Docker 로그 확인"
	@echo "  make docker-clean - Docker 리소스 정리"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy       - 프로덕션 배포 (빌드 + 실행)"
	@echo ""

# ========================================
# Development
# ========================================

# 개발 환경 설정
setup:
	@echo "📦 Installing uv..."
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ Setup complete!"

# 의존성 설치
install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Dependencies installed!"

# 개발 서버 실행
dev:
	@echo "🚀 Starting development server..."
	uv run main.py

# 테스트
test:
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v --cov=agents --cov=graph

# 린트
lint:
	@echo "🔍 Linting code..."
	uv run ruff check agents/ graph/ core/ api/ main.py

# 포맷팅
format:
	@echo "✨ Formatting code..."
	uv run ruff format agents/ graph/ core/ api/ main.py

# 정리
clean:
	@echo "🧹 Cleaning cache files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache .coverage htmlcov/ .ruff_cache/
	@echo "✅ Clean complete!"

# ========================================
# Docker
# ========================================

# Docker 이미지 빌드
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t multi-agent-system:latest .
	@echo "✅ Docker image built!"

# Docker 실행 (docker-compose)
docker-run:
	@echo "▶️  Starting Docker containers..."
	docker-compose up -d
	@echo "✅ Containers started!"
	@echo "📊 Check status: make docker-logs"

# Docker 중지
docker-stop:
	@echo "🛑 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Containers stopped!"

# Docker 로그
docker-logs:
	@echo "📝 Showing Docker logs..."
	docker-compose logs -f

# Docker 리소스 정리
docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Docker resources cleaned!"

# ========================================
# Deployment
# ========================================

# 프로덕션 배포
deploy:
	@echo "🚀 Starting deployment..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found!"; \
		echo "Please copy .env.example to .env and configure it."; \
		exit 1; \
	fi
	@echo "📦 Building Docker image..."
	@docker build -t multi-agent-system:latest .
	@echo "🛑 Stopping existing containers..."
	@docker-compose down || true
	@echo "▶️  Starting new containers..."
	@docker-compose up -d
	@echo "⏳ Waiting for health check..."
	@sleep 10
	@if docker-compose ps | grep -q "Up"; then \
		echo "✅ Deployment successful!"; \
		echo "📊 Container status:"; \
		docker-compose ps; \
		echo ""; \
		echo "📝 View logs: make docker-logs"; \
		echo "🌐 API: http://localhost:$${AGENT_API_PORT:-8000}"; \
	else \
		echo "❌ Deployment failed!"; \
		docker-compose logs; \
		exit 1; \
	fi
