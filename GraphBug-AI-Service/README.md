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

AI-powered code intelligence service that builds knowledge graphs and semantic search for GitHub repositories.

## Features

- 🌳 **Multi-language parsing** - Supports 35+ programming languages using tree-sitter
- 🕸️ **Knowledge graphs** - Creates structured graphs of code relationships in Neo4j
- 🔍 **Semantic search** - Vector-based similarity search using Qdrant
- 🔄 **Multi-tenant** - Isolated by repository ID
- ⚡ **Background processing** - Async repository ingestion

## Architecture

- **Parser** (`src/parser.py`) - Universal code parser using tree-sitter
- **Graph Builder** (`src/graph_builder.py`) - Neo4j knowledge graph construction
- **Vector Builder** (`src/vector_builder.py`) - Qdrant vector embeddings
- **API** (`src/api.py`) - FastAPI REST endpoints

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to match your setup. For local development with Docker, the defaults should work.

### 3. Start Infrastructure

Start Neo4j and Qdrant using Docker Compose:

```bash
docker-compose up -d
```

This starts:
- Neo4j on ports 7474 (HTTP) and 7687 (Bolt)
- Qdrant on port 6333

### 4. Run the Service

```bash
# Development
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Verify Setup

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

## API Endpoints

### Health Check
```
GET /health
```

Returns service status and connectivity to Neo4j and Qdrant.

### Ingest Repository
```
POST /ingest
Content-Type: application/json

{
  "repo_url": "https://github.com/user/repo.git",
  "repo_id": "unique-repo-id",
  "installation_id": "github-app-installation-id"
}
```

Queues a repository for background processing. Returns immediately.

### Search Repository
```
POST /query
Content-Type: application/json

{
  "repo_id": "unique-repo-id",
  "query": "function that calculates tax"
}
```

Returns semantically similar code snippets.

## Configuration

All configuration is done via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `neo4j://neo4j:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `graphbug123` |
| `QDRANT_URL` | Qdrant connection URL | `http://qdrant:6333` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `TEMP_REPOS_DIR` | Temporary clone directory | `./temp_repos` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Testing

Run the pipeline test to verify everything works:

```bash
python pipeline_test.py
```

Test the search functionality:

```bash
python verify_search.py
```

## Supported Languages

JavaScript, TypeScript, Python, Go, Rust, Java, Ruby, PHP, C#, C++, C, Bash, Lua, YAML, TOML, Markdown, and 20+ more.

## Development

### Project Structure

```
ai-service/
├── src/
│   ├── api.py           # FastAPI application
│   ├── parser.py        # Tree-sitter parser
│   ├── graph_builder.py # Neo4j integration
│   ├── vector_builder.py # Qdrant integration
│   ├── config.py        # Configuration management
│   ├── logger.py        # Logging setup
│   └── queries/         # Tree-sitter query files
├── data/                # Persistent data (gitignored)
├── temp_repos/          # Temporary clones (gitignored)
├── docker-compose.yml   # Infrastructure setup
├── requirements.txt     # Python dependencies
└── .env                 # Environment config (gitignored)
```

### Adding Language Support

1. Check if tree-sitter-languages supports it
2. Add the extension mapping to `src/parser.py`
3. Ensure the query file exists in `src/queries/{language}/tags.scm`

## Troubleshooting

### Import Errors

Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Connection Errors

Check that Docker services are running:
```bash
docker-compose ps
```

Check service logs:
```bash
docker-compose logs neo4j
docker-compose logs qdrant
```

### Neo4j Authentication

If you changed the password, update both:
- `docker-compose.yml` (NEO4J_AUTH environment variable)
- `.env` file (NEO4J_PASSWORD variable)

### Permission Errors

Ensure data directories are writable:
```bash
chmod -R 755 data/
```

## License

MIT
