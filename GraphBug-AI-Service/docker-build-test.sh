#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           Docker Build & Test Script                              ║"
echo "║           Tests locally before HF Spaces deployment               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please copy .env.docker to .env and fill in your values:"
    echo "  cp .env.docker .env"
    echo "  nano .env  # Edit and add your GITHUB_APP_ID and GITHUB_PRIVATE_KEY"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Step 1: Build Docker image
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Step 1: Building Docker image..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker build -t graph-bug-ai-service:test .
echo -e "${GREEN}✅ Docker image built successfully${NC}"
echo ""

# Step 2: Start services
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Step 2: Starting services with docker-compose..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker-compose up -d
echo -e "${GREEN}✅ Services started${NC}"
echo ""

# Step 3: Wait for services to be healthy
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ Step 3: Waiting for services to be healthy..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    HEALTH=$(docker-compose ps --format json | jq -r '.[].Health' 2>/dev/null || echo "starting")
    
    if echo "$HEALTH" | grep -q "healthy"; then
        echo -e "${GREEN}✅ All services are healthy${NC}"
        break
    fi
    
    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo -e "${RED}❌ Services did not become healthy in time${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi
echo ""

# Step 4: Check service logs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 4: Checking service initialization..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait a bit for startup logs
sleep 5

echo ""
echo "AI Service logs (last 20 lines):"
echo "─────────────────────────────────────────────────────────────────────"
docker-compose logs --tail=20 ai-service

# Check for GitHub client initialization
if docker-compose logs ai-service | grep -q "GitHub Client Initialized Successfully"; then
    echo -e "${GREEN}✅ GitHub Client initialized successfully${NC}"
else
    echo -e "${YELLOW}⚠️  GitHub Client may not be initialized - check logs${NC}"
fi

echo ""

# Step 5: Test health endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 Step 5: Testing health endpoint..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

HEALTH_RESPONSE=$(curl -s http://localhost:7860/health || echo "failed")
if echo "$HEALTH_RESPONSE" | grep -q "status"; then
    echo -e "${GREEN}✅ Health endpoint responding${NC}"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Health endpoint not responding${NC}"
    exit 1
fi
echo ""

# Step 6: Show service URLs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Services are ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AI Service API:      http://localhost:7860"
echo "API Docs:            http://localhost:7860/docs"
echo "Neo4j Browser:       http://localhost:7474 (neo4j/graphbug123)"
echo "Qdrant Dashboard:    http://localhost:6333/dashboard"
echo ""
echo "To view logs:        docker-compose logs -f ai-service"
echo "To stop services:    docker-compose down"
echo "To restart:          docker-compose restart ai-service"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ All tests passed! Your Docker setup is working.${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Test ingestion: curl -X POST http://localhost:7860/ingest -H 'Content-Type: application/json' -d '{...}'"
echo "  2. If everything works, deploy to HF Spaces: ./deploy-to-hf.sh"
echo ""
