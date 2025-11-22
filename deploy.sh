#!/bin/bash
set -e

echo "🚀 Multi-Agent System Deployment Script"
echo "========================================"
echo ""

# 환경 변수 확인
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Copy .env.example to .env:"
    echo "   cp .env.example .env"
    echo ""
    echo "2. Edit .env and configure your settings"
    echo ""
    exit 1
fi

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed!"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# docker-compose 설치 확인
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed!"
    echo "Please install docker-compose first: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Docker 이미지 빌드
echo "📦 Building Docker image..."
docker build -t multi-agent-system:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully"
echo ""

# 기존 컨테이너 중지
echo "🛑 Stopping existing containers..."
docker-compose down || true
echo ""

# 새 컨테이너 시작
echo "▶️  Starting new containers..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start containers!"
    exit 1
fi

echo "✅ Containers started"
echo ""

# 헬스체크 대기
echo "⏳ Waiting for health check..."
sleep 10

# 상태 확인
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "📊 Container Status:"
    docker-compose ps
    echo ""
    echo "📝 Useful commands:"
    echo "  - View logs:        docker-compose logs -f"
    echo "  - Stop containers:  docker-compose down"
    echo "  - Restart:          docker-compose restart"
    echo ""
    
    # .env에서 포트 읽기
    PORT=$(grep AGENT_API_PORT .env | cut -d '=' -f2 | tr -d '"' | tr -d ' ')
    PORT=${PORT:-8080}
    
    echo "🌐 API Endpoints:"
    echo "  - Health:  http://localhost:${PORT}/health"
    echo "  - API:     http://localhost:${PORT}/"
    echo ""
else
    echo ""
    echo "❌ Deployment failed!"
    echo ""
    echo "📝 Container logs:"
    docker-compose logs
    exit 1
fi
