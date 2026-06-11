# RAG Studio — 10-Architecture RAG Explorer

> Compare **10 state-of-the-art RAG architectures** side-by-side on your own documents.  
> Built with **Next.js + FastAPI** — real-time streaming, adaptive learning, and a full evaluation suite.

Upload PDFs, DOCX, TXT, CSV, Excel, images, or URLs — then ask questions and watch each pipeline think in real time. Switch between architectures, run all 10 simultaneously in Compare mode, score answers with a RAGAS-inspired evaluator, and let the system learn from your feedback.

---

## Architectures

| # | Architecture | Core Idea | Best For |
|---|---|---|---|
| 01 | **Hybrid RAG** | Dense vectors + BM25 sparse search fused via Reciprocal Rank Fusion + cross-encoder re-ranking | General-purpose documents, mixed query types |
| 02 | **Graph RAG** | LLM-extracted entity/relationship triples → NetworkX knowledge graph + vector fallback | Documents rich in named entities and relationships |
| 03 | **Agentic RAG** | LangGraph planner routes to VECTOR_SEARCH, WEB_SEARCH, or direct answer; multi-hop decomposition | Queries needing web context or multi-step reasoning |
| 04 | **Corrective RAG (CRAG)** | Evaluator grades retrieved docs CORRECT / AMBIGUOUS / INCORRECT; rewrites query and falls back to web | When retrieval quality is uncertain |
| 05 | **Multimodal RAG** | Llama 4 Scout vision summarises uploaded images; base64 stored in metadata; image + text sent at query time | Documents with figures, charts, screenshots |
| 06 | **Multilingual RAG** | Cross-lingual embedding space; cross-encoder re-ranking; answers in the query's language | Multilingual documents or cross-language queries |
| 07 | **RAG-Fusion** | Expands query into 4 sub-queries, retrieves separately, merges all ranked lists with RRF | Ambiguous or broad queries |
| 08 | **HyDE RAG** | Generates a hypothetical ideal answer first, embeds it, uses it as the search vector | Short or keyword-style queries worded differently from source text |
| 09 | **Structured RAG** | CSV/Excel → pandas DataFrame; LLM generates pandas code from NL query and executes it | Spreadsheets, datasets, numerical analysis, filtering, aggregation |
| 10 | **Self-RAG** | Simulates Self-RAG paper tokens — grades relevance `[IsRel]`, critiques faithfulness `[IsSup]` and completeness `[IsUse]`; refines and re-retrieves if scores below threshold | When answer quality and factual grounding matter most |

---

## Features

### Core
- **Live "brain working" view** — real-time step indicators + token streaming with blinking cursor as each pipeline executes
- **Compare mode** — run all 10 architectures simultaneously; results in a card grid with expand-to-full-answer modal
- **Per-architecture chat threads** — independent conversation history per architecture with message count badges
- **Source citations** — collapsible panel showing retrieved chunks, source filenames, and relevance scores
- **Multi-document support** — additive ingestion; upload multiple files at once, mix PDFs, CSVs, URLs in any order
- **Multi-document comparison** — every context chunk is labelled `[Source: filename]`; the LLM can compare content across different documents
- **URL ingestion** — paste a webpage URL; it's scraped, chunked, and ingested automatically
- **Document management** — view all ingested files with chunk counts; delete individual files from all architectures
- **API key management** — set your Groq API key from the UI without restarting the server; auto-prompts on first launch
- **Chat export** — download current chat, all architecture chats, or compare results as Markdown files (3 export options)
- **LocalStorage persistence** — chat history survives page refreshes
- **Persistent ChromaDB** — vector store survives server restarts; documents don't need to be re-ingested

### Adaptive RAG
- **Semantic query cache** — queries with cosine similarity ≥ 0.95 (+ length-ratio guard) return instantly (⚡ badge)
- **Clear cache** — one-click cache clear from the sidebar action buttons
- **Context quality self-evaluation** — after retrieval, each pipeline evaluates its own context: CORRECT → generate, AMBIGUOUS → supplement with web search, INCORRECT → web search fallback
- **Parent-child chunking** — small child chunks (300 chars) embedded for precise retrieval; full parent chunk (1000 chars) used for richer generation context
- **Feedback-driven retrieval** — thumbs up/down on any answer; re-click to unmark; positively-rated chunks surface first in future queries; negatively-rated chunks are demoted
- **Multi-hop decomposition** — Agentic RAG splits complex multi-part queries into sub-questions, retrieves each independently, then merges context

### Evaluation
- **RAGAS-inspired 4-metric scorecard** — Faithfulness, Answer Relevance, Context Precision, Context Recall (all 0–10)
- **Analytics dashboard** — per-architecture query count, average latency, eval scores (bar chart), cache hit rate, feedback ratio
- **Architecture explainer** — pipeline flow diagram, how-it-works text, best-use cases, and adaptive features per architecture

### UI
- **OLED dark theme** — deep black (`#07070f`) base with violet (`#7c3aed`) accent throughout
- **Glass surface cards** — layered surface tokens, glow shadows, slide-up animations
- **Real-time streaming** — bouncing dot animation while the LLM generates; emerald checkmark on completion
- **Horizontal eval score bars** — color-coded fills (green ≥ 7, amber ≥ 4, red < 4) with average badge

---

## Tech Stack

| Layer | Choice |
|---|---|
| **LLM** | Meta Llama `meta-llama/llama-4-scout-17b-16e-instruct` via `langchain-groq` (vision + text) |
| **Embeddings** | `all-MiniLM-L6-v2` via `langchain-huggingface` (local, no API key) |
| **Vector DB** | ChromaDB `PersistentClient` — persists to `./chroma_db` across restarts |
| **Sparse Search** | `rank-bm25` — BM25Okapi |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| **Knowledge Graph** | NetworkX (in-memory) + PyVis (interactive visualisation) |
| **Agentic Workflows** | LangGraph — Agentic RAG and CRAG state machines |
| **Web Search Fallback** | DuckDuckGo (`duckduckgo-search`) — no API key required |
| **Structured Data** | pandas + openpyxl — CSV/Excel Text-to-Pandas |
| **Adaptive Storage** | SQLite via `adaptive_db.py` — feedback, semantic cache, analytics |
| **Document Parsing** | PyPDF2, python-docx, pandas (CSV/Excel), Pillow (images) |
| **Backend API** | FastAPI + Uvicorn, SSE streaming via `threading.Queue` |
| **Frontend** | Next.js 14, React 18, Tailwind CSS, TypeScript |
| **Containerisation** | Docker + Docker Compose (backend + frontend + persistent volume) |
| **Python** | 3.11+ |

---

## Project Structure

```
multiple-rag-system/
│
├── requirements.txt
├── Dockerfile                      # FastAPI backend Docker image
├── docker-compose.yml              # backend + frontend + chroma_data volume
├── .env                            # GROQ_API_KEY (gitignored)
├── .env.example                    # Template — copy to .env and fill in
├── adaptive.db                     # SQLite — feedback, semantic cache, analytics (auto-created, gitignored)
├── chroma_db/                      # Persistent ChromaDB vector store (auto-created, gitignored)
│
├── core/
│   ├── shared_services.py          # LLM (ChatGroq), embeddings, PersistentChromaDB, reranker,
│   │                               # parent-child chunking, build_sourced_context,
│   │                               # evaluate_context, web_search_fallback,
│   │                               # chroma_query() — HNSW-safe retry wrapper used by all archs
│   └── adaptive_db.py              # AdaptiveDB — feedback boost/unmark, semantic cache, analytics
│
├── architectures/
│   ├── hybrid_rag.py               # Dense + BM25 + RRF + cross-encoder + feedback boost
│   ├── graph_rag.py                # Entity extraction → NetworkX graph + feedback boost
│   ├── agentic_rag.py              # LangGraph: Planner → Tool Executor → Reasoner + multi-hop
│   ├── corrective_rag.py           # LangGraph: Retrieve → Evaluate → Route → Generate
│   ├── multimodal_rag.py           # Llama 4 Scout vision + base64 image metadata
│   ├── multilingual_rag.py         # Cross-lingual embeddings + cross-encoder + feedback boost
│   ├── rag_fusion.py               # 4 sub-queries + RRF + feedback boost
│   ├── hyde_rag.py                 # Hypothetical document embeddings + feedback boost
│   ├── structured_rag.py           # Text-to-Pandas (CSV/Excel) + vector fallback
│   └── self_rag.py                 # [IsRel] / [IsSup] / [IsUse] critique loop — max 2 iterations
│
├── backend/
│   ├── __init__.py
│   ├── session_manager.py          # GlobalSession — initialises all 10 pipelines + ARCH_INFO
│   └── api.py                      # FastAPI — 18 endpoints, SSE streaming, RAGAS eval
│
├── legacy/
│   └── app.py                      # Original Streamlit prototype (reference only)
│
└── frontend/
    ├── Dockerfile                  # Next.js multi-stage Docker image
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx                # Root — all state, streaming, feedback, analytics, exports
    │   └── globals.css             # CSS variables, glass utilities, animations
    ├── components/
    │   ├── Sidebar.tsx             # Arch list, toggles, action grid (clear/reset/export/cache/stats/api key)
    │   ├── ArchCard.tsx            # Architecture info card with expandable how-it-works
    │   ├── ChatMessage.tsx         # Chat bubbles with thumbs feedback (toggle to unmark) + ⚡ cached badge
    │   ├── MarkdownContent.tsx     # Inline markdown renderer
    │   ├── BrainWorking.tsx        # Live step indicators + bouncing dots + emerald done state
    │   ├── SourcePanel.tsx         # Collapsible source citations with chunk counts and scores
    │   ├── EvalScorecard.tsx       # Horizontal score bars: Faithfulness / Relevance / Precision / Recall
    │   ├── DocumentManager.tsx     # Drag-drop upload + URL ingestion + per-file delete
    │   ├── ApiKeyModal.tsx         # Groq API key modal — auto-opens if no key set
    │   ├── CompareGrid.tsx         # 10-card grid with ping-ring loading + expand modal
    │   ├── AnalyticsDashboard.tsx  # Per-arch stats table with bar charts + recent queries
    │   └── ArchExplainer.tsx       # Pipeline flow, how-it-works, adaptive features per arch
    ├── tailwind.config.ts          # Custom tokens: surface colors, glow shadows, slide-up animation
    └── lib/
        ├── api.ts                  # All API calls + EventSource streamQuery()
        └── types.ts                # TypeScript interfaces
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the Next.js frontend)
- A free [Groq API key](https://console.groq.com/keys)

### 1 — Clone and install

```bash
git clone https://github.com/pranay9981/Multiple-Rag-System.git
cd Multiple-Rag-System

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2 — Add your API key

Copy the example env file and add your Groq key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your-groq-api-key-here
```

> Get a free key at [console.groq.com/keys](https://console.groq.com/keys).  
> Alternatively, leave `.env` empty and enter the key in the UI on first launch.

### 3 — Run the app

**Terminal 1 — FastAPI backend**

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Next.js frontend**

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Docker (one command)

```bash
cp .env.example .env   # add GROQ_API_KEY to .env first
docker compose up --build
```

Both services start automatically. ChromaDB data persists in a named Docker volume.

---

## How to Use

1. **Upload documents** — drag-drop one or more files (PDF, DOCX, TXT, CSV, XLSX, PNG, JPG) or paste a URL. All formats are ingested into all 10 architectures automatically.
2. **Ask questions** — press Enter to send; watch the brain-working panel stream live pipeline steps and tokens in real time.
3. **Compare mode** — toggle "Compare All (10)" to run every architecture simultaneously; results appear in a card grid with per-card expand modal.
4. **Multi-document queries** — upload multiple files then ask comparison questions like "What are the differences between document A and document B?" — each chunk is labelled with its source filename.
5. **RAG Evaluation** — toggle "RAG Evaluation" to get RAGAS-inspired scores after each answer (Faithfulness, Relevance, Context Precision, Context Recall).
6. **Thumbs feedback** — rate any answer up or down; click the active button again to unmark. The system surfaces positively-rated chunks first in future queries.
7. **Export** — click "Export" in the sidebar action grid to download the current chat, all architecture chats, or compare results as `.md` files.
8. **Clear Cache** — click "Cache" in the sidebar to wipe the semantic query cache; forces all architectures to re-run their full retrieval pipelines.
9. **Stats** — click "Stats" in the sidebar to see per-architecture analytics: query counts, latencies, eval scores, cache hits, and feedback ratios.
10. **How it works** — click "How it works" in the header for a detailed explainer of the current architecture's pipeline.
11. **Document management** — view and delete ingested files from the sidebar Documents panel.
12. **API Key** — click the "API Key" button in the sidebar to update your Groq key at any time without restarting the server.

---

## FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/architectures` | All 10 architecture metadata |
| `GET` | `/api/sessions/{id}` | Session info, ingested archs, doc library |
| `POST` | `/api/ingest` | File or URL ingest with parent-child chunking |
| `GET` | `/api/query` | SSE stream — `step` / `token` / `sources` / `done` / `error` events |
| `POST` | `/api/compare` | Run all 10 architectures concurrently (150ms stagger, HNSW-safe) |
| `POST` | `/api/evaluate` | RAGAS-inspired 4-metric scoring |
| `POST` | `/api/feedback` | Store rating (1 = up, -1 = down, 0 = unmark/delete) |
| `GET` | `/api/analytics` | Per-architecture aggregated stats from SQLite |
| `GET` | `/api/graph` | PyVis HTML for Graph RAG knowledge graph visualisation |
| `GET` | `/api/history` | Last 20 queries |
| `DELETE` | `/api/sessions/{id}` | Reset all pipelines and session history |
| `GET` | `/api/documents` | List ingested files + chunk counts for an architecture |
| `DELETE` | `/api/documents` | Delete a file from all architectures |
| `DELETE` | `/api/cache` | Clear the semantic query cache |
| `GET` | `/api/config/status` | Returns `{has_key: bool}` — whether GROQ_API_KEY is set |
| `POST` | `/api/config/apikey` | Set Groq API key at runtime and re-initialise LLM |

---

## Adaptive RAG: How It Learns

```
Query
  │
  ├─► Semantic Cache (cosine sim ≥ 0.95 + length-ratio guard) ──► ⚡ Instant cached answer
  │
  ▼
Retrieve chunks
  │
  ├─► Feedback Boost — positively-rated chunks move up, negative move down
  │
  ▼
Self-Evaluation (evaluate_context)
  ├─► CORRECT  ──────────────────► Generate answer
  ├─► AMBIGUOUS ─► + web search ──► Generate answer  (CRAG: always generate, no web)
  └─► INCORRECT ─► web search ────► Generate answer  (Graph RAG: always use ingested docs)
  │
  ▼
Store in semantic cache + analytics
```

### HNSW Concurrency Protection

All 10 architectures use `services.chroma_query()` — a thread-safe wrapper around ChromaDB's `collection.query()` that retries up to 3× on HNSW segment reader failures (common under concurrent compare-mode load), refreshing the collection handle between attempts. Never call `collection.query()` directly in new architectures.

---

## License

MIT
