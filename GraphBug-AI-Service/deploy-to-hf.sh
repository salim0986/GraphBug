#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           Hugging Face Spaces Deployment                          ║"
echo "║           Deploy tested Docker image to HF Spaces                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HF_SPACE_NAME="your-username/graph-bug-ai-service"  # Update this!

echo -e "${BLUE}ℹ️  Deployment Configuration:${NC}"
echo "   HF Space: $HF_SPACE_NAME"
echo ""

# Check if HF CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}⚠️  Hugging Face CLI not found. Installing...${NC}"
    pip install --upgrade huggingface_hub
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged in to Hugging Face${NC}"
    echo "Please login first:"
    echo "  huggingface-cli login"
    exit 1
fi

echo -e "${GREEN}✅ Logged in to Hugging Face${NC}"
echo ""

# Create HF Spaces configuration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Creating HF Spaces configuration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create README for HF Spaces
cat > README_HF_DEPLOY.md << 'EOF'
---
title: Graph Bug AI Service
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Graph Bug - AI Code Review Service

AI-powered code review system using GraphRAG (Neo4j + Qdrant) and Google Gemini.

## Features
- Automatic PR code reviews
- Repository ingestion and analysis
- Context-aware recommendations
- Neo4j graph database for code relationships
- Qdrant vector database for semantic search

## Environment Variables Required

Set these in your HF Space settings:

### Required
- `GITHUB_APP_ID`: Your GitHub App ID
- `GITHUB_PRIVATE_KEY`: Your GitHub App private key (PEM format with \n)
- `NEO4J_URI`: Neo4j Aura connection string
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `QDRANT_URL`: Qdrant Cloud URL
- `QDRANT_API_KEY`: Qdrant Cloud API key

### Optional
- `ENVIRONMENT=production` (auto-detected via SPACE_ID)
- `LOG_LEVEL=INFO`
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins

## API Endpoints

- `GET /health` - Health check
- `POST /ingest` - Ingest repository
- `POST /review` - Trigger PR review
- `POST /query` - Query repository code

## Documentation

Full documentation: [GitHub Repository](https://github.com/your-username/graph-bug)
EOF

echo -e "${GREEN}✅ HF Spaces README created${NC}"
echo ""

# Build Docker image for HF Spaces
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔨 Building Docker image for HF Spaces..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker build -t graph-bug-ai-service:hf-deploy .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi
echo ""

# Prepare deployment files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Preparing deployment files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create temp deployment directory
DEPLOY_DIR=$(mktemp -d)
echo "Using temp directory: $DEPLOY_DIR"

# Copy necessary files
cp Dockerfile "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"
cp -r src "$DEPLOY_DIR/"
cp README_HF_DEPLOY.md "$DEPLOY_DIR/README.md"
cp .dockerignore "$DEPLOY_DIR/" 2>/dev/null || true

echo -e "${GREEN}✅ Files prepared${NC}"
echo ""

# Instructions for deployment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}📋 Deployment Instructions:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option 1: Deploy via Git (Recommended)"
echo "───────────────────────────────────────"
echo "1. Create a new HF Space:"
echo "   https://huggingface.co/new-space"
echo "   - Choose SDK: Docker"
echo "   - Choose Hardware: CPU Basic (free tier)"
echo ""
echo "2. Push your code to the Space:"
echo "   cd $DEPLOY_DIR"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial deployment'"
echo "   git remote add space https://huggingface.co/spaces/$HF_SPACE_NAME"
echo "   git push space main"
echo ""
echo "3. Set environment variables in HF Space settings:"
echo "   - Go to: https://huggingface.co/spaces/$HF_SPACE_NAME/settings"
echo "   - Add all required variables (see README_HF_DEPLOY.md)"
echo ""
echo "4. Wait for build to complete (5-10 minutes)"
echo "   - Monitor at: https://huggingface.co/spaces/$HF_SPACE_NAME"
echo ""
echo ""
echo "Option 2: Deploy via HF CLI (Alternative)"
echo "──────────────────────────────────────────"
echo "huggingface-cli repo create $HF_SPACE_NAME --type space --space_sdk docker"
echo "cd $DEPLOY_DIR"
echo "huggingface-cli upload $HF_SPACE_NAME . . --repo-type space"
echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Deployment files ready at: $DEPLOY_DIR${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "After deployment, test your space at:"
echo "  https://huggingface.co/spaces/$HF_SPACE_NAME"
echo ""
