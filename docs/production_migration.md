# Production Migration Notes

This project now has a migration path from the Streamlit MVP to a production-style split app:

```text
frontend/            Next.js UI
backend/             FastAPI API layer
architectures/       Existing RAG pipeline implementations
services/            Shared LLM, embedding, vector store, loader, search services
```

The Streamlit app is intentionally still present. It remains useful as a working MVP while the API and Next.js frontend mature.

## Local Development

Start the API:

```powershell
venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Start the Next.js frontend:

```powershell
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Then open:

```text
http://127.0.0.1:3000
```

API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Current API Surface

```text
GET  /api/health
GET  /api/architectures
GET  /api/sessions/{session_id}
POST /api/ingest
POST /api/query
POST /api/compare
```

The API currently keeps pipeline sessions in memory. That is enough for a local production migration pass, but not enough for a multi-server deployment.

## Next Production Hardening Steps

1. Add authentication and per-user session ownership.
2. Move ingestion and graph extraction into a worker queue.
3. Persist session/job metadata in PostgreSQL.
4. Replace in-memory API session state with Redis or database-backed state.
5. Add request limits, file size limits, MIME validation, and upload quarantine.
6. Add structured logging and request IDs.
7. Add integration tests for ingest, query, compare, and frontend API clients.
8. Decide whether Chroma remains the production vector store or move to Qdrant/pgvector.
9. Decide whether NetworkX remains enough or Graph RAG needs Neo4j.
10. Add Docker Compose for API, frontend, worker, database, and vector store.
