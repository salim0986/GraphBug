# GraphBug 🐛

**GraphBug** is an AI-powered code intelligence and automated code review platform. It uses Graph RAG (Retrieval-Augmented Generation) technology to understand your codebase's architecture deeply, building a knowledge graph of relationships to provide context-aware reviews on GitHub Pull Requests.

[![Watch the video](https://github.com/salim0986/GraphBug/blob/65bc024e7caa6ca9f37637369ee3f8075d82d9a8/graphbug-thumbnail.png)](https://www.youtube.com/watch?v=Po2i08VHay8)


---

## 🌟 Key Features

- 🤖 **Automated PR Reviews**: Automatically comments on new pull requests with intelligent feedback.
- 🌳 **Multi-language Knowledge Graphs**: Parses 35+ languages using tree-sitter to build code relationship graphs in Neo4j.
- 🔍 **Semantic Code Search**: Uses Qdrant vector database for high-accuracy semantic retrieval of relevant code snippets.
- 🛠️ **GitHub App Integration**: Seamless integration with GitHub workflows via webhooks and official App APIs.
- 🔐 **Secure & Private (BYOK)**: "Bring Your Own Key" model ensures you control your AI costs and data privacy.
- 📊 **Unified Dashboard**: Manage repository ingestions, view analysis status, and configure API keys.

---

## 📁 Project Structure

The project is divided into two main components:

- **[GraphBug-AI-Service](GraphBug-AI-Service)**: The backend "brain". A Python FastAPI service responsible for parsing code, building knowledge graphs, and handling semantic queries.
- **[GraphBug-Web](GraphBug-Web)**: The frontend "hub". A Next.js application that handles user authentication, GitHub App interactions, and provides the dashboard UI.

---

## 🛠️ Tech Stack

### AI Service (Backend)
- **Framework**: FastAPI (Python)
- **Parser**: Tree-sitter (Multi-language support)
- **Graph Database**: Neo4j
- **Vector Database**: Qdrant
- **ML Models**: Sentence Transformers (`all-MiniLM-L6-v2`)

### Web Platform (Frontend)
- **Framework**: Next.js 16 (App Router)
- **UI**: React 19, Tailwind CSS 4
- **ORM**: Drizzle ORM
- **Database**: PostgreSQL
- **Auth**: NextAuth.js 5.0 (Beta)
- **Communication**: Octokit (GitHub API)

---

## 🏗️ System Architecture

1.  **Ingestion**: When a repository is connected, the **Web Platform** triggers the **AI Service**.
2.  **Parsing**: The AI Service clones the repo, parses it using tree-sitter, and extracts symbols, calls, and dependencies.
3.  **Graph & Vector Building**: Metadata is stored in **Neo4j** (relationships) and code embeddings are stored in **Qdrant** (semantics).
4.  **PR Event**: A `pull_request.opened` webhook is sent from GitHub to the Web Platform.
5.  **Context Retrieval**: The Web Platform queries the AI Service for context relevant to the PR diff.
6.  **AI Review**: Using the retrieved context and the user's Gemini API key, a review is generated and posted back to the PR.

---

## 🚀 Getting Started

### 1. Prerequisites
- Docker & Docker Compose
- Node.js 18+ & pnpm
- Python 3.10+
- PostgreSQL database
- GitHub App credentials (see [Frontend Setup](GraphBug-Web/README.md#3-create-github-app))

### 2. Infrastructure Setup
Both services rely on shared infrastructure. Start them using Docker:

```bash
cd GraphBug-AI-Service
docker-compose up -d
```
This starts **Neo4j** and **Qdrant**.

### 3. AI Service Setup
1.  Install Python dependencies:
    ```bash
    cd GraphBug-AI-Service
    pip install -r requirements.txt
    ```
2.  Configure environment:
    ```bash
    cp .env.example .env
    # Edit .env with your Neo4j/Qdrant credentials
    ```
3.  Run the service:
    ```bash
    uvicorn src.api:app --reload --port 8000
    ```

### 4. Web Platform Setup
1.  Install Node dependencies:
    ```bash
    cd GraphBug-Web
    pnpm install
    ```
2.  Configure environment:
    ```bash
    cp .env.example .env
    # Add your DATABASE_URL, GitHub App credentials, and AUTH_SECRET
    ```
3.  Setup Database:
    ```bash
    pnpm drizzle-kit push
    ```
4.  Run the development server:
    ```bash
    pnpm dev
    ```

---

## 🔑 Bring Your Own Key (BYOK)

GraphBug requires a **Google Gemini API Key** to perform the actual code analysis.
1.  Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  In the GraphBug Dashboard, navigate to **Settings > API Keys**.
3.  Enter your key to enable automated reviews.

---

## 📜 License

This project is licensed under the MIT License - see the individual service directories for details.

## 🙏 Acknowledgements

- Built with [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for robust code parsing.
- Powered by [Neo4j](https://neo4j.com/) and [Qdrant](https://qdrant.tech/) for Graph RAG capabilities.
