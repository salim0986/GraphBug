#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     Sync Latest Code to Hugging Face Spaces                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get your HF username
read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter your HF Space name (default: graph-bug-ai-service): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-graph-bug-ai-service}
SPACE_FULL="$HF_USERNAME/$SPACE_NAME"

echo -e "${BLUE}ℹ️  Target Space: $SPACE_FULL${NC}"
echo ""

# Check if space exists
echo "Checking if space exists..."
if ! huggingface-cli repo info --repo-type space "$SPACE_FULL" &> /dev/null; then
    echo -e "${RED}❌ Space not found: $SPACE_FULL${NC}"
    echo ""
    echo "Create it first at: https://huggingface.co/new-space"
    echo "  - Name: $SPACE_NAME"
    echo "  - SDK: Docker"
    echo "  - Hardware: CPU Basic"
    exit 1
fi

echo -e "${GREEN}✅ Space exists${NC}"
echo ""

# Create temp directory for deployment
TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

# Clone HF Space
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Cloning HF Space..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git clone https://huggingface.co/spaces/$SPACE_FULL "$TEMP_DIR"

# Copy latest files
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Copying latest files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Remove old src directory
rm -rf "$TEMP_DIR/src"

# Copy essential files
cp Dockerfile "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"
cp -r src "$TEMP_DIR/"
cp .dockerignore "$TEMP_DIR/" 2>/dev/null || true

# Create/update README for HF Space
cat > "$TEMP_DIR/README.md" << 'EOF'
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

## Environment Variables

Set in Space Settings → Repository secrets:

### Required
- `GITHUB_APP_ID` - Your GitHub App ID
- `GITHUB_PRIVATE_KEY` - GitHub App private key (use \n for newlines)
- `NEO4J_URI` - Neo4j Aura connection (neo4j+s://...)
- `NEO4J_USER` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password  
- `QDRANT_URL` - Qdrant Cloud URL
- `QDRANT_API_KEY` - Qdrant API key

### Optional
- `ALLOWED_ORIGINS` - CORS origins (comma-separated)
- `LOG_LEVEL` - INFO (default)

## API Endpoints

- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /ingest` - Ingest repository
- `POST /review` - Trigger PR review

## Last Updated

$(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF

echo -e "${GREEN}✅ Files copied${NC}"
echo ""

# Show what changed
cd "$TEMP_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Changes to be deployed:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git status --short
echo ""

# Commit and push
if git diff --quiet && git diff --staged --quiet; then
    echo -e "${GREEN}✅ No changes to deploy - Space is up to date${NC}"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Deploying to HF Spaces..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    git add .
    git commit -m "Sync latest code - $(date +%Y-%m-%d)"
    git push
    
    echo ""
    echo -e "${GREEN}✅ Deployment complete!${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 Your space: https://huggingface.co/spaces/$SPACE_FULL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⏳ Build status:"
    echo "   Monitor at: https://huggingface.co/spaces/$SPACE_FULL"
    echo "   Build takes: ~5-10 minutes"
    echo ""
    echo "✅ After build completes, test:"
    echo "   curl https://$HF_USERNAME-$SPACE_NAME.hf.space/health"
    echo ""
fi

# Cleanup
cd -
rm -rf "$TEMP_DIR"

echo "🎉 Done!"
