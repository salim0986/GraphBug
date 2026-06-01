# GraphBug — AI-Powered Code Review via Graph RAG

A full-stack, service-oriented platform that automatically reviews GitHub Pull Requests by building a living knowledge graph of your codebase — understanding not just the changed lines, but every function they call, every module they depend on, and every pattern they resemble across the entire repository.

[![Watch the video](https://github.com/salim0986/GraphBug/blob/65bc024e7caa6ca9f37637369ee3f8075d82d9a8/graphbug-thumbnail.png)](https://www.youtube.com/watch?v=Po2i08VHay8)

## Project Overview

GraphBug is an automated code intelligence system I built for engineering teams that want review-quality feedback on every PR without the review queue bottleneck. The platform installs as a GitHub App, listens for pull request events, parses the affected repository using tree-sitter, builds a bi-layer knowledge graph in Neo4j and a semantic search index in Qdrant, and then routes the PR through a LangGraph-orchestrated review pipeline that uses those graphs to ground Gemini's analysis in real codebase context — not just the diff in isolation.

The result is a review comment posted directly to the PR that flags security issues, performance problems, bugs, and code quality concerns with file-by-file specificity and cross-codebase awareness that flat diff-based reviewers cannot produce.

The system follows a strict service-oriented architecture: a Next.js frontend and API layer handles authentication, webhooks, dashboard, and analytics; a separate Python FastAPI microservice handles all parsing, graph construction, and AI pipeline execution. The two services communicate over HTTP, while both read and write to the same PostgreSQL database.

The platform operates on a Bring Your Own Key (BYOK) model. Users supply their own Google Gemini API key, stored in the database and passed per-request to the AI pipeline. This gives users complete cost visibility and eliminates shared quota limits.

## Problem Statement

Automated code review has historically failed for one reason: tools that only read the diff cannot see what the diff breaks. A function rename, a changed interface, a subtle state mutation — these only become bugs in the context of everything that calls them. Most static analysis and AI-powered review tools have the same blind spot.

Beyond that, code review presents compounding operational challenges:

- **Review queue depth** — senior engineers become the bottleneck; PRs sit unreviewed for days while the author context fades
- **Inconsistent standards** — what gets caught depends on which reviewer picks up the PR; subtle security and performance issues slip through on busy weeks
- **Diff-only blindness** — reviewers reading a diff don't have the full codebase in their head; edge cases that are obvious with context are invisible without it
- **No cross-PR pattern awareness** — a reviewer on one PR can't see that the same anti-pattern appeared in three other PRs this week
- **Zero feedback loop on review quality** — most tools post a comment and disappear; there is no tracking of which issues were fixed, which were dismissed, or what the cost of each review was

GraphBug addresses each of these through a Graph RAG approach that builds codebase understanding before it reads the PR, then routes the diff through that pre-built context at review time.

## Core Features

### 1. GitHub App Integration

GraphBug installs as a native GitHub App, not a third-party webhook consumer. Installation grants the app access to selected repositories with pull request read/write permissions. GitHub sends a `pull_request.opened` or `pull_request.synchronize` webhook to the Next.js API layer on every qualifying event.

The app authenticates to GitHub using RSA-signed JWTs (GitHub App JWTs) and short-lived installation access tokens. Each clone, API call, and review post is authenticated against the specific installation that owns the repository — private repositories are fully supported. Installation metadata (permissions, repository selection, account type) is persisted in the `github_installations` table; individual repository records go into `github_repositories` with ingestion status tracking.

Users install the app directly from the dashboard's "Manage GitHub App" link. A setup callback at `/api/github/setup` links the GitHub installation ID to the authenticated user's account after the GitHub redirect.

### 2. Repository Ingestion and Knowledge Graph Construction

Before any review can run, the repository must be ingested. Ingestion clones the repo, parses every supported source file, builds a Neo4j knowledge graph of code relationships, and indexes semantic embeddings in Qdrant.

**Two ingestion modes:**

**Full ingestion** — triggered when a repository is first connected. The Python worker shallow-clones the repository (depth=1 for speed), walks the file tree with a configurable ignore list (`node_modules`, `venv`, `.git`, `dist`, `build`, etc.), and processes all valid source files in parallel using a 20-worker `ThreadPoolExecutor`. Each batch of 20 files is parsed concurrently, with individual file timeouts (10s) and per-batch timeouts (120s) to prevent hangs on pathological input. After parsing, Neo4j dependency edges are built from call relationships found across the entire repository.

**Incremental ingestion** — triggered on `pull_request.synchronize` events and any re-ingest request that supplies a `last_commit` SHA. The worker fetches only the files that changed between the base commit and HEAD using git diff, processes them in isolation, and updates only the affected Neo4j nodes and Qdrant vectors. Full re-parse is skipped. Incremental runs complete in 2–10 seconds versus 10–60 seconds for a full ingest.

**Tree-sitter parsing** — a universal parser built on `tree-sitter-languages` handles 35+ languages from a single code path. Rather than maintaining per-language parsing logic, each language is registered with its file extensions and a `.scm` query file that captures definitions and relationships. The HYBRID_MAP system covers three query styles: Zed-style grammar node type names (e.g., `function_definition`, `class_declaration`), standard tree-sitter capture names (e.g., `definition.function`, `definition.class`), and infrastructure-specific captures (e.g., `definition.resource` for Terraform HCL, `definition.base_image` for Dockerfiles, `definition.target` for Makefiles). This covers application code, configuration, and infrastructure as code in a single pipeline.

**Neo4j graph construction** — each parsed symbol (Function, Class, Interface, Module, Struct) becomes a labeled node. Call relationships and imports become typed edges. The `build_dependencies` pass traverses the parsed tree a second time to add `CALLS` edges between function nodes. Queries are scoped by `repo_id` predicate, providing hard isolation between repositories sharing the same Neo4j instance.

**Qdrant vector indexing** — functions, methods, and classes above a minimum line threshold are embedded using `all-MiniLM-L6-v2` from Sentence Transformers and upserted into a Qdrant collection. Only semantic units (not single-line declarations or config keys) are vectorized. Config files in `.yaml`, `.yml`, `.toml`, `.tf`, and Dockerfile format are included as whole-block embeddings when they contain meaningful content. The collection is filtered by `repo_id` at query time.

### 3. GraphRAG Context Retrieval

When a PR arrives for review, the pipeline does not feed the raw diff to Gemini. It first builds rich context using the pre-indexed knowledge graph.

**Vector search** — the PR diff is used to generate search queries against the Qdrant collection. Top matching code snippets (by cosine similarity) serve as entry points into the graph.

**Graph expansion** — for each vector match, the Neo4j graph is traversed to fetch its dependencies: the functions it calls, the modules it imports, and the data structures it manipulates. This expansion surfaces context that is semantically related but textually distant from the diff.

**Temporary GraphRAG** — in addition to the permanent repository graph, the workflow builds a temporary in-memory graph and vector index scoped to just the PR's changed files. This captures intra-PR relationships (e.g., a helper function introduced in the same PR as the function that calls it) that the permanent graph does not yet contain because ingestion for the new commit hasn't run. The temporary and permanent contexts are merged by the `ContextMerger` before being passed to Gemini.

The combined context gives the AI reviewer three things a diff-only tool cannot have: what the changed code depends on, what depends on the changed code, and what similar code in the repository already looks like.

### 4. LangGraph Review Workflow

All review orchestration runs through a LangGraph `StateGraph`. The workflow state is a `ReviewState` TypedDict that carries PR metadata, file change data, retrieved context, review strategy, generated file reviews, the overall summary, timing, and an append-only error list.

**Five nodes:**

**Analyze node** — builds the `PRContext` object by running the GraphRAG context retrieval pipeline. Orchestrates vector search against Qdrant, graph traversal against Neo4j for direct dependencies of changed symbols, file-level complexity scoring, and construction of the temporary in-memory GraphRAG from the PR diff. Merges the two context sources via `ContextMerger`. Writes the result to `state["pr_context"]` and `state["_merged_contexts"]`. Also assembles high/medium/low priority file lists based on complexity scores and issue history.

**Route node** — reads file count, total additions, and complexity scores from state. Applies tier thresholds from `WorkflowConfig` to select a Gemini model. Three tiers:
- *Quick review*: ≤3 changed files and ≤100 additions → `gemini-2.5-flash-lite` (cheapest, fastest)
- *Standard review*: ≤10 files or ≤500 additions → `gemini-2.5-flash` (balanced)
- *Deep review*: everything else, or PRs touching security-sensitive paths → `gemini-2.5-pro` (most capable reasoning)

The routing decision is stored as `review_strategy` and `selected_model` in the state and is visible in the analytics dashboard.

**Review nodes (quick / standard / deep)** — all three share the same review loop structure; they differ only in which Gemini model is instantiated and the depth of context included in each prompt. For each file in priority order, a prompt is constructed from: the file diff, the retrieved graph context for that file, cross-file patterns identified in the analyze phase, and the PR description. The response is parsed into a structured `FileReview` object containing a summary, a list of issues with severity and category tags, and an `issues_found` count. Files are processed serially within each node to stay within context window limits. Gemini calls include retry logic with exponential backoff (up to 3 retries).

**Aggregate node** — collects all file reviews, constructs a cross-file synthesis prompt, and calls Gemini a final time to generate the overall PR summary. This summary addresses the PR holistically — architectural implications, cross-file patterns, overall risk assessment — rather than repeating file-level findings. Formats the final GitHub comment using rich markdown: shields.io severity badges, collapsible sections for long reviews, syntax-highlighted code blocks, and a file-by-file analysis table sorted by issue count. Posts the comment to the PR via `github_client.post_review_comment`. Stores the structured result to PostgreSQL.

**Error handler node** — reads `state["retry_count"]` and `state["errors"]`. Routes back to `analyze` for transient failures (network, rate limit) up to the configured retry ceiling. Routes to END for terminal failures (authentication error, repository not found). The PR author never receives an unformatted error message; the workflow either posts a valid review or posts nothing and records the failure internally.

### 5. BYOK Model

GraphBug does not proxy Gemini calls through a shared platform key. Each user supplies their own Google Gemini API key from Google AI Studio, stored in the `users.gemini_api_key` column. When the review workflow executes, the key is retrieved and passed to a per-request `GeminiClient` instance. Users pay Google directly; GraphBug adds no markup and holds no shared quota.

The Gemini client is instantiated lazily inside `execute_review_task` — if a user key is present, a user-scoped workflow is created for that run; otherwise the service falls back to the server-side key for development environments. Revoking or rotating the key in the dashboard takes effect immediately on the next review.

### 6. Analytics and Cost Tracking

Every review generates a complete analytics record. The `code_reviews` table stores per-review: execution time in milliseconds, total tokens consumed (input and output), total cost in USD, model tier used, an overall score, and a structured breakdown of issues by severity (critical / high / medium / low / info) and category (security / performance / bug / code_quality / best_practice / documentation / testing / accessibility / maintainability).

Individual issues are stored in `review_comments` with file path, line range, diff side (LEFT or RIGHT), severity, category, the suggested fix, and an AI confidence score. Each comment tracks whether it was posted to GitHub, how many post attempts were made, and whether the developer resolved the issue — enabling resolution rate tracking across teams and repositories.

The `review_insights` table provides time-series aggregations (daily, weekly, monthly) across three scopes: per-user, per-repository, and per-installation. Snapshots capture review volume, PR throughput, issue detection rates, resolution rates, average cost per review, and model usage distribution. The analytics pages surface these as trend charts and KPI cards.

### 7. Dashboard

The Next.js dashboard surfaces four primary views:

**Repositories** — lists all connected repositories with ingestion status (pending / processing / completed / failed), last sync timestamp, and an "Ingest" trigger button that fires a full re-ingest.

**Reviews** — a paginated table of all PR reviews with severity badges, model tier, execution time, and a direct link to the GitHub PR comment. A detail view shows the full file-by-file breakdown, per-issue list with resolution status, and the AI reasoning behind each finding.

**Analytics** — cost trends by model tier, review volume over time, issue category distribution, and per-repository deep-dives. Cost projections extrapolate from the current period's usage.

**Settings** — Gemini API key configuration and GitHub App installation management. The key is masked in the UI after entry; the raw value is never surfaced after save.

### 8. Authentication

Authentication runs through NextAuth.js 5 (beta) with three configured providers:

- **GitHub OAuth** — primary sign-in method; the access token is stored in the `accounts` table and used for GitHub API calls scoped to the user
- **Google OAuth** — alternative sign-in for users who prefer to keep GitHub access separate from their GraphBug account
- **Magic links via Resend** — passwordless email sign-in using `verificationTokens`; no password storage required

All API routes call `auth()` and check for a valid session before processing. Internal webhook calls (GitHub → Next.js) use an `x-internal-call: true` header to bypass session checks for events that arrive before any user session is established.

## Technical Architecture

### System Design

The platform separates into two independently deployed services:

**Web + API Layer (Next.js)**: Handles all user-facing concerns — authentication via NextAuth.js, the React dashboard, GitHub App webhook receipt and signature verification, REST API routes consumed by the frontend, and PostgreSQL persistence via Drizzle ORM. When a PR webhook arrives, this layer validates the signature, creates the database records, and dispatches the review job to the Python service fire-and-forget. It does not block on the AI pipeline.

**AI Microservice (Python FastAPI + LangGraph)**: Receives ingestion and review requests from the Next.js layer, runs all compute-heavy operations — cloning, parsing, graph construction, vector indexing, LangGraph workflow execution, Gemini calls — and writes results back to the shared database. Also calls the GitHub API directly to post review comments.

**Database (PostgreSQL)**: A single shared database written to by both services. The Next.js layer uses Drizzle ORM for type-safe access. The Python service uses the Supabase Python client with the service role key.

**Knowledge Stores**: Neo4j and Qdrant run as Docker Compose services alongside the Python API. Neo4j stores the code relationship graph; Qdrant stores the vector embeddings. Both are scoped by `repo_id` rather than tenant — a single collection and graph instance serves all repositories with predicate-based isolation.

**Communication pattern**: The Next.js API creates a PR record and calls `POST /review` on the Python service (fire-and-forget), then returns 200 to GitHub immediately. The Python worker runs the LangGraph pipeline asynchronously as a FastAPI `BackgroundTask`, posts the GitHub comment, and writes the review record back to PostgreSQL. The dashboard reads review status from the database on next load — no WebSocket or polling infrastructure is required.

### Technology Stack

**Frontend & API Layer:**
- Next.js 16.1 with App Router and React Server Components
- React 19 with concurrent features
- TypeScript 5 across the entire frontend
- Tailwind CSS 4 for utility-first styling
- NextAuth.js 5.0 (beta) for multi-provider authentication
- Drizzle ORM 0.45 for type-safe PostgreSQL access
- Octokit REST + `@octokit/auth-app` for GitHub App authentication
- `@octokit/plugin-throttling` + `@octokit/plugin-retry` for production-grade rate limit handling
- `@octokit/webhooks` for webhook payload validation and type-safe event handling
- Zod 4 for runtime request validation
- TanStack Table 8 for the reviews data table
- Recharts for analytics charts
- Framer Motion for UI transitions
- Lucide React for iconography
- Jest + React Testing Library for unit and integration tests

**Python AI Microservice:**
- FastAPI with Uvicorn for the HTTP API layer
- LangGraph for multi-node workflow orchestration with typed state
- LangChain Core for agent primitives
- Google Generative AI SDK — Gemini 2.5 Flash-Lite, Flash, and Pro for all LLM calls
- `tree-sitter-languages` with 35+ language grammars for universal code parsing
- Neo4j Python driver for graph database interaction
- Qdrant Python client for vector database interaction
- Sentence Transformers (`all-MiniLM-L6-v2`) for local code embedding generation
- `slowapi` + custom `RateLimitMiddleware` for layered rate limiting
- Supabase Python client (`asyncpg`-backed) for async PostgreSQL writes
- `GitPython` for repository cloning and incremental diff extraction
- `PyJWT` for GitHub App JWT generation
- `httpx` for async HTTP calls back to the Next.js API
- Pydantic 2 for request/response validation with field-level constraints

**Infrastructure:**
- PostgreSQL for the application database
- Neo4j (Docker Compose) for the code knowledge graph
- Qdrant (Docker Compose) for vector embeddings
- Vercel for Next.js deployment
- Docker Compose for the Python service and knowledge stores

### Database Schema

Eleven tables organized across three functional domains:

**Authentication (NextAuth.js):**
`user` — User accounts. Stores name, email, avatar, and `gemini_api_key` (the BYOK Gemini key in plain text; the field is never exposed in API responses after write).
`account` — OAuth provider links. Stores access and refresh tokens for GitHub and Google. Compound primary key on `(provider, providerAccountId)`.
`session` — Active session tokens with expiry timestamps. Cleaned up by NextAuth on expiry.
`verificationToken` — Magic link tokens. Compound primary key; deleted on first use.
`authenticator` — WebAuthn passkey records (future-ready, currently unused).

**GitHub Integration:**
`github_installation` — GitHub App installation records. Stores GitHub's integer `installationId`, `accountLogin`, `accountType` (User or Organization), permission JSON, and repository selection mode. FK to `user` is nullable — an installation row is created by the webhook before the user links their account, then updated when linkage completes.
`github_repository` — One row per repository. Stores GitHub's integer `repoId`, `fullName`, `private` flag, and a full ingestion lifecycle: `ingestionStatus` (pending / processing / completed / failed), `ingestionStartedAt`, `ingestionCompletedAt`, `ingestionError`, and `lastSyncedAt`. Unique constraint on `(installationId, repoId)` prevents duplicate rows.

**Code Review:**
`pull_request` — Tracks every PR GraphBug has processed. Stores GitHub PR number and ID, metadata (title, description, author with avatar URL, branch refs, commit SHAs), diff statistics (filesChanged, additions, deletions), status enum (open / closed / merged / draft), and review lifecycle timestamps. Indexed on repository, status, review status, author, and creation date. Unique constraint on `(repositoryId, prNumber)`.
`code_review` — One row per AI review execution. Stores the primary model tier (`flash` / `pro` / `thinking`), a JSON `summary` (overallScore, issuesBySeverity), `keyChanges` and `recommendations` arrays, all token and cost fields, `executionTimeMs`, the GitHub comment ID and URL, inline comment count, and a `workflowState` JSON field for debugging LangGraph runs. FK to `pull_request`. Supports a `reviewVersion` counter for re-reviews of the same PR.
`review_comment` — One row per issue identified in a review. Stores file path, start/end line, diff side (LEFT or RIGHT), severity and category enums, issue title and message, suggested fix text, and code snippet for context. Tracks GitHub posting status (`isPosted`, `githubCommentId`, `postAttempts`, `postError`). Tracks developer resolution (`isResolved`, `resolvedAt`, `resolvedBy`). Stores AI confidence score (0–1) and the model's reasoning for the finding. Indexed on review, severity, category, file, and resolution/posting status.
`review_insight` — Time-series analytics aggregations scoped by `(scope, scopeId)` where scope is `user`, `repository`, or `installation`. Stores period type and boundaries, review and PR volume metrics, issue severity and category breakdowns, resolution rate, average execution time and score, total cost, and a `modelUsageStats` JSON field with per-tier call counts. Unique constraint on `(scope, scopeId, periodType, periodStart)`.

### AI Agent Architecture

All review orchestration is expressed as a LangGraph `StateGraph`. Each node is an async Python function that receives the shared `ReviewState` TypedDict and returns a partial state delta. Conditional edges are plain Python functions that read state values and return string route keys — the routing logic is auditable without reading LangGraph internals.

**ReviewState** carries: PR identifiers, raw file changes from GitHub, the built `pr_context`, merged temporary and permanent GraphRAG contexts, the routing decision, priority-sorted file lists (high/medium/low), per-file review dicts keyed by filename, the final summary string, model selection, progress counters (`processed_files`, `total_files`), an `errors` list (append-only, never raises), retry count, timing fields, and LangGraph message history.

**Node: `analyze`** — Calls `ContextBuilder.build_pr_context()`, which orchestrates: vector search against Qdrant for semantically similar existing code, graph traversal against Neo4j for direct dependencies of changed symbols, file-level complexity scoring, and construction of the temporary in-memory GraphRAG from the PR diff. Merges both context sources via `ContextMerger`. Also pre-computes file priority order based on complexity and prior issue density.

**Node: `route`** — Applies `WorkflowConfig` thresholds to select model tier. Sets `requires_deep_review = True` for PRs touching authentication, cryptography, or database schema files regardless of size, overriding the normal thresholds. Emits `review_strategy` and `selected_model` into state.

**Nodes: `review_quick`, `review_standard`, `review_deep`** — Each processes files in priority order within a per-batch limit. For each file: constructs a grounded prompt combining the diff, the retrieved graph context, cross-file patterns, and PR description; calls Gemini with the tier's selected model; parses and validates the structured response into a `FileReview` object; stores it in `state["file_reviews"]`. Files are processed serially within each node to stay within the context window limit. Gemini calls include retry with exponential backoff.

**Node: `aggregate`** — Synthesizes all file reviews into an overall PR summary via a final Gemini call. Formats the GitHub comment markdown with shields.io severity badges, collapsible sections, and file-by-file table sorted by issue count. Posts the comment. Stores the review record and all individual comments to PostgreSQL.

**Node: `handle_error`** — Reads `state["retry_count"]` and routes back to `analyze` for transient failures (up to `max_retries`) or to END for terminal failures. The full error list is written to `code_reviews.workflowState` for post-mortem debugging.

### Authentication & Security

**GitHub App JWT flow** — Every repository clone and API call uses a short-lived installation access token obtained by signing a JWT with the App's RSA private key and exchanging it for an installation token via the GitHub Apps API. The token is embedded in the clone URL as `x-access-token:{token}` and discarded after use. Unauthenticated fallback is explicitly disabled; if token generation fails, the operation fails rather than silently attempting a public-URL clone.

**Webhook signature verification** — All incoming GitHub webhooks are verified against `GITHUB_WEBHOOK_SECRET` using HMAC-SHA256 before any processing occurs. Requests with invalid or missing signatures return 401.

**Input sanitization** — Repository IDs are validated against a strict `^[a-zA-Z0-9_-]+$` pattern before use in any filesystem path. `safe_join()` validates that the constructed path does not escape `temp_repos/`. Error messages returned to callers pass through `sanitize_error_message()` to prevent stack traces and internal paths from appearing in responses.

**Pydantic and Zod validation** — FastAPI endpoints use Pydantic 2 models with field-level regex constraints: `repo_url` must match `^https://github\.com/[\w-]+/[\w.-]+(.git)?$`; `installation_id` must be a numeric string; `query` is capped at 1000 characters. Next.js routes use Zod schemas with identical constraints. Validation failures return structured error responses, not raw exceptions.

**Rate limiting** — Two layers: `slowapi` applies per-IP limits at the FastAPI middleware level; a custom `RateLimitMiddleware` applies per-installation limits to prevent a single GitHub App installation from monopolizing the worker. The Next.js layer checks GitHub API `rateLimit.remaining` before making API calls and returns 429 early if critically low.

**CORS** — The FastAPI service allows origins from the `ALLOWED_ORIGINS` environment variable (defaulting to `http://localhost:3000`). Only GET, POST, and DELETE methods are permitted. Credentials are allowed for authenticated routes.

### Key Design Decisions

**Fire-and-forget webhook processing**: GitHub's webhook delivery system requires a 200 response within 10 seconds. The review pipeline takes 20–120 seconds. The Next.js handler creates records and dispatches the request to the Python service (which queues it as a `BackgroundTask`), then returns 200 to GitHub immediately. The review continues even if the developer closes their browser. Webhook retries are idempotent because the PR record is upserted, not blindly inserted.

**Temporary GraphRAG for intra-PR context**: The permanent repository graph is authoritative but stale by one ingestion cycle. A PR that introduces a new helper function and calls it in the same commit creates a forward reference the permanent graph cannot resolve. The temporary in-memory graph captures intra-PR relationships; the `ContextMerger` merges both with the temporary graph taking precedence for symbols found in both. These in-memory structures are garbage-collected when the workflow completes.

**Three-tier Gemini model routing**: Sending every PR through Gemini 2.5 Pro maximizes quality but wastes cost on one-line hotfixes. The router applies dual thresholds (file count and additions) to select Flash-Lite, Flash, or Pro, with a security override that forces Pro for authentication, cryptography, and schema-touching PRs regardless of size. The selected model is stored per-review and visible in cost analytics.

**Incremental ingestion via git diff**: A full re-parse on every PR event would make the platform unusable for large codebases. Incremental ingestion uses `git diff` between the stored commit SHA and HEAD to identify only changed files, processes additions/modifications with the same parser, and explicitly deletes Neo4j nodes and Qdrant vectors for deleted and renamed files. The graph converges to post-merge state rather than accumulating stale history.

**Append-only error list in workflow state**: LangGraph nodes never raise exceptions. Any failure appends a structured error object to `state["errors"]` and allows the workflow to continue producing a partial review. The aggregate node skips files with no entry in `state["file_reviews"]` and lists them explicitly in the GitHub comment footer. The developer receives a partial review with an honest accounting of what was skipped, not a silent all-clear.

**Per-repo isolation without per-tenant databases**: Neo4j and Qdrant are single-instance services. Isolation is enforced by including `repo_id` as a mandatory predicate on every read and write operation. Deleting a repository's data on app uninstall is a single predicate-filtered deletion in both stores. This eliminates the operational overhead of per-tenant database instances while maintaining hard data separation between repositories.

**Parallel file parsing in ThreadPoolExecutor**: Tree-sitter's C extension releases the GIL, making file parsing CPU-bound but parallelizable from Python threads. A 20-worker thread pool processes files in batches with individual (10s) and per-batch (120s) timeouts. Progress is logged every batch as a percentage with files/sec throughput, making long ingestion runs observable without log flooding.

**Structured TypedDict workflow state**: Using a `ReviewState` TypedDict with `NotRequired` annotations for optional fields makes every field's presence and type explicit at the type-checker level. This eliminates an entire class of `KeyError` bugs in multi-node workflows and makes the state contract readable — any engineer can understand what information flows between nodes by reading the TypedDict definition.

## Implementation Highlights

### 1. GraphRAG: Vector Entry Points Into a Dependency Graph

A naive vector search returns the N most semantically similar code snippets to a query — useful for finding examples but unable to reveal what a function *depends on*. GraphBug uses vector search only as an entry point into the Neo4j graph. For each vector match, a Cypher query fetches the matched node's `CALLS` and `IMPORTS` edges, returning the dependency subgraph. This graph expansion is what tells the reviewer: "this changed function calls `validateToken`, which calls `parseJWT`, which reads from a hard-coded config key" — information that lives nowhere in the diff but is critical to the review.

### 2. Tree-sitter Hybrid Query System for 35+ Languages

Supporting 35+ languages without per-language business logic requires a unified abstraction. Tree-sitter's `.scm` query files capture named nodes from each language's grammar, but different grammars use different node type names for the same conceptual construct (`function_definition` in Python vs. `function_declaration` in C++). The `HYBRID_MAP` in `graph_builder.py` maps all variant names — Zed-style grammar node types, standard tree-sitter capture names, and infrastructure-specific captures — to a canonical symbol type (`Function`, `Class`, `Interface`, `Module`, `Resource`). The graph builder reads this canonical type when creating Neo4j nodes, producing a uniform graph schema regardless of source language. A Dockerfile's `FROM` instruction, a Terraform `resource` block, and a Python class definition all become first-class graph nodes with consistent labels and queryable relationships.

### 3. Incremental Ingestion Converges on Post-Merge State

Re-parsing an entire 10,000-file repository on every PR is a non-starter for response time. Incremental ingestion uses `git diff` between the stored `last_commit` SHA and the current HEAD to isolate changed files. Deletions are processed first — `GraphBuilder.delete_file_nodes()` removes all nodes with a matching `file_path` and their edges from Neo4j; Qdrant vectors are deleted by payload filter — before additions and modifications are processed normally. This explicit deletion pass is what makes the graph converge to post-merge state rather than accumulating stale nodes from deleted files. A 10-file PR in a large repository completes incremental ingestion in 2–5 seconds.

### 4. Complexity-Based Model Selection With Security Override

The routing node implements a two-threshold decision tree: PRs under both the file count and additions limits get Flash-Lite; PRs under the larger limits get Flash; everything else gets Pro. An override path checks changed file paths against a pattern list of security-sensitive locations (authentication modules, cryptographic utilities, database migration files) and forces Pro regardless of size — the cost of a Pro review on a small but security-critical change is always justified. The routing decision, selected model, and override reason are all stored in `code_reviews` and surface in the analytics dashboard, making model distribution visible to operators who want to tune thresholds.

### 5. Rich GitHub Comment Formatting

The raw output of the aggregate node is markdown, but formatting decisions for GitHub comments matter. Reviews above 3000 characters collapse behind a `<details>` element. Per-file reviews beyond the top eight are paginated into a second collapsible. Severity is surfaced immediately via shields.io badge images (no JavaScript needed). Files are sorted by `issues_found` descending so the most critical file appears first. A footer identifies the review as AI-generated, names the Gemini model version, and frames the review appropriately. All of this is assembled by `format_review_for_github()` as pure string concatenation — no templating library required.

### 6. Octokit Plugin Stack for Production GitHub Integration

GitHub's rate limiting is asymmetric: primary limits are per-hour, secondary limits (abuse detection) fire unpredictably on burst requests. The Next.js GitHub client uses `@octokit/plugin-throttling` to handle both: primary rate limit responses are retried up to twice after the specified wait; secondary rate limits are retried once. Client errors (400, 401, 403, 404, 422) are never retried — they indicate request problems that a retry cannot fix. `@octokit/plugin-retry` handles transient 5xx errors with exponential backoff. The combination makes the GitHub client resilient to transient throttling without risking infinite retry loops.

## Challenges & Solutions

### Challenge: Private Repository Access Across Two Services

**Problem**: The Python service needs to clone repositories. Private repositories require authentication. The authentication credentials (GitHub App private key) live in the Python service environment; the repository URL and installation ID arrive from the Next.js layer. A naive implementation would either expose the private key over the internal API or fail on private repos entirely.

**Solution**: The Python service initializes its own `GitHubClient` on startup using `GITHUB_APP_ID` and `GITHUB_PRIVATE_KEY` from environment variables. At ingest time, it generates a short-lived installation access token via the JWT + token exchange flow, embeds it as `x-access-token:{token}` in the clone URL, clones, and discards the token. No private key material ever leaves the Python service. If token generation fails, ingestion fails explicitly with a structured error message — there is no silent fallback to an unauthenticated clone that would produce a confusing "not found" error for private repositories.

### Challenge: Stale Graph State for Intra-PR Symbols

**Problem**: The knowledge graph reflects the repository state at last ingestion. A PR that introduces a new module and immediately imports it creates a forward reference that the permanent graph cannot resolve. Reviewing the import with no knowledge of what it exports produces shallow or incorrect analysis.

**Solution**: The workflow builds a temporary in-memory graph from the PR diff using the same `UniversalParser` and a lightweight `TemporaryGraphBuilder` that stores nodes in Python dicts rather than Neo4j. Embedding runs against the `all-MiniLM-L6-v2` model loaded lazily in the workflow. The `ContextMerger` queries both graphs for each changed file and merges results, with the temporary graph taking precedence for symbols that appear in both. The temporary structures are garbage-collected when the workflow exits — no cleanup required.

### Challenge: LangGraph Node Failures Producing Silent All-Clears

**Problem**: If the analyze node fails to retrieve graph context for two out of ten files due to a Neo4j timeout, a workflow that crashes gives the developer nothing. A workflow that silently omits those files gives them a review that looks complete but is missing analysis for potentially the most significant changed files.

**Solution**: Every node wraps per-file operations in individual try/except blocks, appending structured error objects to `state["errors"]` on failure. The aggregate node checks whether each file has an entry in `state["file_reviews"]`; files without entries are listed explicitly in the GitHub comment footer with a one-line explanation of why they were skipped. The developer sees a partial review with an honest accounting of gaps, not a false all-clear. The full error list is persisted to `code_reviews.workflowState` for debugging.

### Challenge: Incremental Ingestion Leaving Stale Graph Nodes

**Problem**: When a file is deleted or renamed in a PR, the old Neo4j nodes and Qdrant vectors remain. Future reviews for code that used to call the deleted function still surface those nodes as dependencies, producing phantom references to symbols that no longer exist.

**Solution**: Incremental ingestion processes deletions as a first pass before additions. For each file reported as `deleted` or `renamed` in the git diff, `GraphBuilder.delete_file_nodes()` removes all Neo4j nodes with a matching `file_path` property and all their incident relationships via Cypher. Qdrant vectors are deleted by filtering on the `file` payload field. After the deletion pass, additions and modifications proceed normally. The graph converges to the post-merge repository state rather than accumulating stale history from deleted files.

### Challenge: GitHub Webhook Delivery Timeout

**Problem**: GitHub requires a 200 response within 10 seconds. The review pipeline takes 20–120 seconds. A synchronous handler times out, causing GitHub to retry the webhook and producing duplicate reviews.

**Solution**: The Next.js webhook handler performs only synchronous work synchronously: validates the signature, creates the PR record, dispatches the review request as an HTTP POST to the Python service, and returns 200 immediately. The Python service queues the actual workflow execution as a FastAPI `BackgroundTask`, decoupling the HTTP response from pipeline execution. Webhook retries are idempotent: the PR record is upserted by GitHub ID rather than blindly inserted, so a duplicate delivery creates a re-review version rather than a duplicate record.

### Challenge: Gemini Context Window Management Across File Batches

**Problem**: A large PR with 50 changed files and rich graph context for each file would exceed Gemini's context window if processed in a single prompt. Truncating context silently produces incomplete reviews.

**Solution**: The review nodes process files serially in priority order, one at a time (or in small batches), each with its own Gemini call. Each call stays within the `context_window_tokens` budget from `WorkflowConfig` (25,000 tokens). The high/medium/low priority ordering from the route node ensures that if the PR is very large and time-boxed, the highest-complexity files receive full analysis before lower-priority files are reviewed with reduced context. The aggregate node then synthesizes across all file reviews in a final, shorter call.

## Results & Impact

- **Review coverage**: Every PR in a connected repository receives a structured review comment automatically, within minutes of being opened, without requiring reviewer assignment or availability
- **Cross-codebase awareness**: Reviews identify dependency-chain issues — data flow vulnerabilities, broken interface contracts, cascading performance implications — that are invisible to tools limited to the diff
- **Language breadth**: A single platform installation reviews JavaScript, TypeScript, Python, Go, Rust, Java, Ruby, C#, PHP, Bash, Terraform, and 25+ additional languages from the same tree-sitter pipeline
- **Cost transparency**: Per-review cost breakdown by model tier and token count gives teams full visibility into AI spend with no shared quota and no surprise bills
- **Developer-native output**: Structured GitHub comments with severity badges, collapsible file sections, and inline issue descriptions integrate into the existing PR workflow without context switching or new tools

## Future Enhancements

The platform is production-ready and actively reviewing PRs. Planned additions include:

- **Inline GitHub review comments** — post issues as line-level review comments using the Pull Request Review API, surfacing findings exactly where they occur in the diff view
- **PR re-review with delta** — allow developers to request a fresh review after pushing fixes, with a delta showing which previously flagged issues were addressed
- **SARIF output** — export results in SARIF format for GitHub Code Scanning and security dashboards
- **Slack and email notifications** — push review completion summaries to configured channels
- **Repository health score** — aggregate issue resolution rate, critical issue frequency, and review coverage into a per-repository health score
- **Author context** — track each developer's historical patterns (frequent security oversights, preferred idioms) to personalize review emphasis per PR author
- **Webhook replay** — re-trigger a review from any stored webhook payload for debugging without requiring a new PR event

## Getting Started

### Prerequisites

- Docker and Docker Compose (for Neo4j and Qdrant)
- Node.js 18+ and pnpm
- Python 3.10+
- PostgreSQL database
- GitHub App credentials (App ID, private key, webhook secret, OAuth client ID/secret)
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Start Infrastructure

```bash
cd GraphBug-AI-Service
docker-compose up -d
# Starts Neo4j on :7474/:7687 and Qdrant on :6333
```

### 2. AI Service Setup

```bash
cd GraphBug-AI-Service
pip install -r requirements.txt
cp .env.example .env
# Edit .env: NEO4J_URI, NEO4J_PASSWORD, QDRANT_URL,
#   GITHUB_APP_ID, GITHUB_PRIVATE_KEY, FRONTEND_URL
uvicorn src.api:app --reload --port 8000
```

### 3. Web Platform Setup

```bash
cd GraphBug-Web
pnpm install
cp .env.example .env
# Edit .env: DATABASE_URL, AUTH_SECRET,
#   NEXT_PUBLIC_GITHUB_APP_ID, GITHUB_PRIVATE_KEY,
#   GITHUB_WEBHOOK_SECRET, AI_SERVICE_URL,
#   AUTH_GITHUB_ID, AUTH_GITHUB_SECRET
pnpm db:push
pnpm dev
```

### 4. Create a GitHub App

In GitHub Settings → Developer Settings → GitHub Apps, create a new app with:

- **Webhook URL**: `https://your-domain/api/webhooks/github`
- **Callback URL**: `https://your-domain/api/auth/callback/github`
- **Setup URL**: `https://your-domain/api/github/setup`
- **Permissions**: Pull requests (Read & Write), Repository contents (Read)
- **Events**: Pull request

Generate and download the RSA private key. For local development, proxy GitHub webhooks to localhost using smee.io:

```bash
npx smee-client --url https://smee.io/YOUR_CHANNEL --path /api/webhooks/github --port 3000
```

---

**Note**: This case study documents a production platform. Implementation details reflect the actual system architecture. Source code is available upon request.
