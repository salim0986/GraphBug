---
title: Graph Bug AI Service
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Graph Bug AI Service

AI-powered code review service using GraphRAG (Graph + Vector RAG) for intelligent PR analysis.

## Features

- 🧠 **GraphRAG**: Combines knowledge graphs (Neo4j) with vector search (Qdrant)
- 🔍 **Smart Analysis**: Context-aware code reviews using Gemini AI
- 🤖 **GitHub Integration**: Automatic PR reviews via GitHub App
- 📊 **Multi-language**: Supports Python, TypeScript, JavaScript, Go, Rust, Java, C++

## API Endpoints

### Health Check
```bash
GET /health
```

### Ingest Repository
```bash
POST /ingest
Content-Type: application/json

{
  "repo_url": "https://github.com/owner/repo",
  "gemini_api_key": "your-api-key"
}
```

### Analyze Pull Request
```bash
POST /analyze/pr
Content-Type: application/json

{
  "repo_id": "owner/repo",
  "pr_number": 123,
  "gemini_api_key": "your-api-key"
}
```

### Webhook (GitHub App)
```bash
POST /webhook/github
X-Hub-Signature-256: sha256=...

{GitHub webhook payload}
```

## Architecture

```
┌─────────────────────────┐
│   FastAPI Service       │
│   (This Space)          │
└──────────┬──────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│ Neo4j   │  │ Qdrant  │
│ Aura    │  │ Cloud   │
└─────────┘  └─────────┘
```

## Setup Required

This service requires external database instances:

1. **Neo4j Aura** (Graph Database)
   - Create free instance at: https://neo4j.com/cloud/aura/
   - Add `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` to Spaces secrets

2. **Qdrant Cloud** (Vector Database)
   - Create free cluster at: https://cloud.qdrant.io/
   - Add `QDRANT_URL`, `QDRANT_API_KEY` to Spaces secrets

3. **GitHub App Credentials**
   - Add `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET`

See [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md) for detailed setup instructions.

## Environment Variables

Set these in Spaces Settings → Repository secrets:

| Variable | Description | Required |
|----------|-------------|----------|
| `NEO4J_URI` | Neo4j connection string | ✅ |
| `NEO4J_USER` | Neo4j username (usually "neo4j") | ✅ |
| `NEO4J_PASSWORD` | Neo4j password | ✅ |
| `QDRANT_URL` | Qdrant cluster URL | ✅ |
| `QDRANT_API_KEY` | Qdrant API key | ✅ |
| `GITHUB_APP_ID` | GitHub App ID | ✅ |
| `GITHUB_PRIVATE_KEY` | GitHub App private key | ✅ |
| `GITHUB_WEBHOOK_SECRET` | Webhook secret | ✅ |
| `ALLOWED_ORIGINS` | CORS allowed origins | ⚠️ |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | ❌ |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up .env file
cp .env.example .env
# Edit .env with your credentials

# Start services (Neo4j + Qdrant)
docker-compose up -d

# Run the API
uvicorn src.api:app --reload --port 8000
```

## Links

- [Documentation](https://github.com/yourusername/graph-bug)
- [Frontend App](https://your-app.vercel.app)
- [GitHub App](https://github.com/apps/graph-bug-ai)

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/yourusername/graph-bug/issues).
