# Multiple RAG System

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) system supporting multiple state-of-the-art architectures.

## Features

- **Multiple RAG Architectures:**
  - **Hybrid RAG:** Combines dense vector search (Sentence Transformers) with sparse lexical search (BM25) using Reciprocal Rank Fusion.
  - **Corrective RAG:** Implements document grading and falls back to Web Search (DuckDuckGo/Tavily) when uploaded context is irrelevant.
- **Asynchronous Ingestion:** Powered by Celery and Redis to handle large document processing without blocking the API.
- **Production Backend:** FastAPI with robust PostgreSQL database models, JWT authentication, and SlowAPI rate limiting.
- **Modern Frontend:** Next.js application with real-time ingestion polling, rich UI components, and side-by-side RAG comparison.
- **Google Gemini Integration:** Native support for `gemini-3.1-flash-lite` for high-quality, fast generation.

## Tech Stack

- **Backend:** Python, FastAPI, SQLAlchemy, Alembic, Celery
- **Frontend:** TypeScript, Next.js, React
- **Infrastructure:** Docker, Docker Compose, PostgreSQL, Redis, ChromaDB
- **AI / LLMs:** Google GenAI (Gemini), Sentence-Transformers

## Getting Started (Local Development)

### Prerequisites
- Docker & Docker Compose
- Node.js 20+
- Python 3.11+
- Google Gemini API Key

### 1. Environment Setup
Copy the example environment files:
```bash
cp .env.example .env
cp frontend/.env.local.example frontend/.env.local
```
Add your `GOOGLE_API_KEY` to the `.env` file.

### 2. Boot Infrastructure
Start PostgreSQL and Redis locally:
```bash
docker-compose up -d
```

### 3. Setup Backend
Activate your Python virtual environment and run migrations:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
```

Boot the API server and Celery worker:
```bash
# Terminal 1: FastAPI
uvicorn app:app --reload

# Terminal 2: Celery Worker
celery -A backend.worker.celery worker --loglevel=info
```

### 4. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```
Access the application at `http://localhost:3000`.

## Production Deployment

This project includes a fully containerized production setup using `docker-compose.prod.yml` and GitHub Actions CI.

To deploy in production:
1. Ensure `.env` is populated with secure keys (`POSTGRES_PASSWORD`, `SECRET_KEY`, `GOOGLE_API_KEY`).
2. Run the production build:
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```
This will build optimized containers for the API, Celery workers, and the standalone Next.js build.

## Testing

The backend includes a comprehensive `pytest` test suite:
```bash
pytest tests/ -v
```

## Architecture Overview

- `/core` - Base RAG pipeline interfaces and schemas
- `/architectures` - Specific implementations (Hybrid, Corrective, etc.)
- `/services` - LLM providers, embedding models, and vector stores
- `/backend` - FastAPI routers, database models, and Celery workers
- `/frontend` - Next.js React application
