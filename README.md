# LocalRAG

**100% local, private document Q&A — powered by Ollama · ChromaDB · FastAPI · Next.js**

No cloud. No API keys. No data leaving your machine.

---

## What It Does

LocalRAG lets you upload documents (PDF, DOCX, Markdown, TXT) and ask questions about them in a chat interface. Every part of the pipeline runs locally:

- **Embeddings** — `nomic-embed-text` via Ollama
- **LLM** — `llama3` via Ollama
- **Vector store** — ChromaDB (on-disk)
- **Database** — SQLite (conversation history)
- **Backend** — FastAPI + MCP server (stdio)
- **Frontend** — Next.js chat UI

---

## Quick Start (Docker — Recommended)

### Prerequisites
- Docker + Docker Compose v2
- 8 GB RAM minimum (16 GB recommended for llama3)
- 10 GB free disk space (for models)

```bash
# 1. Clone the repo
git clone https://github.com/yourname/localrag.git
cd localrag

# 2. Copy env config
cp .env.example .env

# 3. Start everything (pulls models on first run — takes ~5 min)
docker compose up -d

# 4. Open the UI
open http://localhost:3000
```

Services:
| Service     | URL                        |
|-------------|----------------------------|
| Chat UI     | http://localhost:3000      |
| API         | http://localhost:8000      |
| API Docs    | http://localhost:8000/docs |
| Prometheus  | http://localhost:9090      |
| Grafana     | http://localhost:3001      |

---

## Quick Start (Bare Metal)

### Prerequisites
- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com) installed

```bash
# 1. Install Ollama models
ollama pull llama3
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Copy config
cp .env.example .env

# 4. Start backend
python -m uvicorn backend.main:app --reload --port 8000

# 5. Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
localrag/
├── config/                  # Configuration (settings.yaml, config.py)
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── api/                 # Route handlers (chat, sessions, documents, health)
│   ├── services/            # Business logic (ingestion, chat, vector store, watcher)
│   ├── models/              # Pydantic schemas
│   └── middleware/          # Logging, API key, metrics middleware
├── mcp_server/
│   └── server.py            # MCP server (tools, resources, prompts)
├── database/
│   └── models.py            # SQLAlchemy models + session factory
├── monitoring/
│   ├── metrics.py           # Prometheus metric definitions
│   ├── prometheus.yml       # Prometheus scrape config
│   └── grafana-datasources.yml
├── frontend/
│   └── src/
│       ├── app/             # Next.js App Router pages
│       ├── lib/api.ts       # TypeScript API client
│       └── components/      # React components
├── tests/
│   ├── unit/                # Unit tests (mocked deps)
│   ├── integration/         # API integration tests
│   └── e2e/                 # End-to-end tests (requires Ollama)
├── deployment/
│   └── docker/              # Dockerfiles
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Using the MCP Server

The MCP server runs over stdio and exposes these tools:

| Tool                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `search_documents`    | Semantic search over ingested documents          |
| `ingest_document`     | Ingest a file from a local path                  |
| `list_documents`      | List all documents with metadata                 |
| `get_document`        | Get a single document by ID                      |
| `delete_document`     | Remove a document and all its chunks             |
| `ask_question`        | Full RAG Q&A with citations                      |
| `get_collection_stats`| Vector store statistics                          |

### Run the MCP server directly

```bash
python mcp_server/server.py
```

### Connect from Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "localrag": {
      "command": "python",
      "args": ["/path/to/localrag/mcp_server/server.py"]
    }
  }
}
```

---

## Ingesting Documents

### Method 1: Drop files in the watch folder
```bash
cp my-report.pdf ./data/watch_folder/
# Auto-ingested within seconds
```

### Method 2: Upload via API
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@my-report.pdf"
```

### Method 3: Upload via UI
Click the **Documents** button in the sidebar → Upload document.

---

## API Reference

See full interactive docs at: **http://localhost:8000/docs**

### Key endpoints

```
POST   /api/sessions                      Create chat session
GET    /api/sessions                      List sessions
GET    /api/sessions/{id}                 Get session + messages
DELETE /api/sessions/{id}                 Delete session

POST   /api/sessions/{id}/chat            Send message (blocking)
GET    /api/sessions/{id}/chat/stream     Send message (SSE streaming)

POST   /api/documents/upload              Upload + ingest file
GET    /api/documents                     List documents
GET    /api/documents/{id}                Get document
DELETE /api/documents/{id}                Delete document
GET    /api/documents/stats               Vector store stats

GET    /health                            Health check
GET    /metrics                           Prometheus metrics
```

---

## Configuration

All settings live in `config/settings.yaml`. Key options:

```yaml
ollama:
  chat_model: "llama3"        # Change to mistral, phi3, etc.
  embed_model: "nomic-embed-text"

ingestion:
  chunk_size: 512             # Characters per chunk
  chunk_overlap: 64           # Overlap between chunks

retrieval:
  top_k: 5                    # Chunks to retrieve per query
  score_threshold: 0.3        # Min similarity score (0–1)

security:
  api_key_enabled: false      # Set true + provide API_KEY to enable auth
```

Override any value with environment variables (see `.env.example`).

---

## Running Tests

```bash
# Unit + integration (fast, no Ollama needed)
pytest

# With coverage report
pytest --cov-report=html

# Include E2E tests (requires running Ollama)
pytest -m e2e
```

---

## Observability

- **Logs** — JSON structured logs at `./data/logs/app.log` and stdout
- **Metrics** — Prometheus at `/metrics`, scraped every 10s
- **Grafana** — http://localhost:3001 (user: `admin`, pass: `localrag`)

Key metrics:
- `localrag_queries_total` — total queries by session
- `localrag_query_duration_seconds` — end-to-end latency
- `localrag_llm_duration_seconds` — LLM generation time
- `localrag_documents_ingested_total` — ingestion count
- `localrag_chunks_total` — current chunk count

---

## Changing the LLM

```bash
# Pull a different model
ollama pull mistral

# Edit config/settings.yaml
ollama:
  chat_model: "mistral"

# Restart the backend
docker compose restart backend
```

Popular options: `llama3`, `mistral`, `phi3`, `gemma2`, `qwen2`.

---

## Security Notes

- All processing is local — no data ever leaves your machine
- Enable API key auth for multi-user setups: set `API_KEY_ENABLED=true` and `API_KEY=your-secret`
- CORS is restricted to `localhost:3000` by default
- The non-root Docker user prevents container privilege escalation

---

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues.

---

## License

MIT
