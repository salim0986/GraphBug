#!/bin/bash
# Deploy to Hugging Face Spaces

set -e

echo "🚀 Deploying Graph Bug AI Service to Hugging Face Spaces"
echo "=========================================================="

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found. Are you in the ai-service directory?"
    exit 1
fi

# Check if HF CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Error: huggingface-cli not found!"
    echo "Install it with: pip install huggingface_hub"
    exit 1
fi

# Check if logged in (try new command first, fallback to old)
if command -v hf &> /dev/null; then
    if ! hf auth whoami &> /dev/null; then
        echo "❌ Error: Not logged in to Hugging Face"
        echo "Run: huggingface-cli login"
        exit 1
    fi
    HF_USERNAME=$(hf auth whoami 2>/dev/null | head -n 1)
else
    if ! huggingface-cli whoami &> /dev/null 2>&1; then
        echo "❌ Error: Not logged in to Hugging Face"
        echo "Run: huggingface-cli login"
        exit 1
    fi
    HF_USERNAME=$(huggingface-cli whoami 2>/dev/null | head -n 1)
fi
echo "✅ Logged in as: $HF_USERNAME"

# Space name
SPACE_NAME="graph-bug-ai-service"
SPACE_FULL="$HF_USERNAME/$SPACE_NAME"

echo ""
echo "📦 Space: $SPACE_FULL"
echo ""

# Check if space exists
echo "Checking if space exists..."
if huggingface-cli repo info --repo-type space "$SPACE_FULL" &> /dev/null 2>&1; then
    echo "✅ Space already exists"
else
    echo "Creating new space..."
    echo "⚠️  Note: You need to create the space manually via the web UI"
    echo "   Go to: https://huggingface.co/new-space"
    echo "   - Name: $SPACE_NAME"
    echo "   - SDK: Docker"
    echo "   - Hardware: CPU Basic (free)"
    echo ""
    read -p "Press Enter once you've created the space..."
fi

# Initialize git if not already
if [ ! -d .git ]; then
    echo "📝 Initializing git repository..."
    git init
    git remote add space "https://huggingface.co/spaces/$SPACE_FULL"
else
    # Check if remote exists
    if ! git remote | grep -q "^space$"; then
        echo "📝 Adding space remote..."
        git remote add space "https://huggingface.co/spaces/$SPACE_FULL"
    fi
fi

# Create .gitattributes for LFS (if needed)
if [ ! -f .gitattributes ]; then
    echo "*.bin filter=lfs diff=lfs merge=lfs -text" > .gitattributes
    echo "*.safetensors filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
fi

# Stage files
echo "📦 Staging files..."
git add Dockerfile README_HF.md requirements.txt src/ .dockerignore .gitattributes

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to deploy"
else
    echo "💾 Committing changes..."
    git commit -m "Deploy to Hugging Face Spaces - $(date +%Y-%m-%d)"
    
    echo "🚀 Pushing to Hugging Face Spaces..."
    git push space main -f
    
    echo ""
    echo "✅ Deployment initiated!"
    echo ""
    echo "🌐 Your space URL: https://huggingface.co/spaces/$SPACE_FULL"
    echo ""
    echo "📝 Next steps:"
    echo "1. Go to your space settings: https://huggingface.co/spaces/$SPACE_FULL/settings"
    echo "2. Add environment variables (Repository secrets):"
    echo "   - NEO4J_URI"
    echo "   - NEO4J_USER"
    echo "   - NEO4J_PASSWORD"
    echo "   - QDRANT_URL"
    echo "   - QDRANT_API_KEY"
    echo "   - GITHUB_APP_ID"
    echo "   - GITHUB_PRIVATE_KEY"
    echo "   - GITHUB_WEBHOOK_SECRET"
    echo "   - ALLOWED_ORIGINS (optional)"
    echo "3. Wait for the build to complete (~5-10 minutes)"
    echo "4. Test: curl https://$HF_USERNAME-$SPACE_NAME.hf.space/health"
    echo ""
fi

echo "🎉 Done!"
