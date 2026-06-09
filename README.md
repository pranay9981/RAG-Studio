# Multiple RAG System

> Explore and compare **6 state-of-the-art RAG architectures** side-by-side on your own documents — all in a single Streamlit app.

Upload a PDF, image, DOCX, or TXT file, pick an architecture (or run all 6 at once), and ask questions. Each pipeline retrieves and generates answers using a different strategy so you can see exactly how they differ.

---

## Architectures

| # | Architecture | Core Idea | Key Differentiator |
|---|---|---|---|
| 1 | **Hybrid RAG** | Dense vectors + sparse BM25, merged via Reciprocal Rank Fusion | Best general-purpose retrieval accuracy |
| 2 | **Graph RAG** | LLM-extracted entity/relationship graph + vector fallback | Captures relationships standard RAG misses |
| 3 | **Agentic RAG** | LangGraph planner routes queries to vector search, web search, or direct answer | Adaptive multi-step reasoning |
| 4 | **Corrective RAG (CRAG)** | Evaluator grades retrieved docs; rewrites query and falls back to web search when irrelevant | Self-correcting retrieval |
| 5 | **Multimodal RAG** | Stores base64 images in metadata; sends text + images to Gemini vision | Understands visual content natively |
| 6 | **Multilingual RAG** | Cross-lingual embedding space; answers in the query's language | Query in any language, retrieve from any language |

---

## Tech Stack

| Layer | Choice |
|---|---|
| **LLM** | Google Gemini (`gemini-3.1-flash-lite`) via `langchain-google-genai` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| **Vector DB** | ChromaDB (local persistent, zero-config) |
| **Sparse Search** | `rank-bm25` — BM25Okapi |
| **Knowledge Graph** | NetworkX — in-memory, pure Python |
| **Agentic Workflows** | LangGraph (Agentic RAG + CRAG) |
| **Web Search Fallback** | DuckDuckGo (`duckduckgo-search`) — no API key required |
| **Document Parsing** | PyPDF2, python-docx, Pillow |
| **UI** | Streamlit |
| **Language** | Python 3.11+ |

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/your-username/multiple-rag-system.git
cd multiple-rag-system

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

```bash
cp .env.example .env
```

Edit `.env` and set your key:
```
GOOGLE_API_KEY=your-gemini-api-key-here
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

1. **Upload a document** — PDF, TXT, DOCX, PNG, or JPG in the sidebar
2. **Click "Ingest Document"** — chunks and embeds it into the selected pipeline (or all 6 if Compare mode is on)
3. **Ask a question** in the chat box
4. **Toggle "Compare All Architectures"** to run all 6 pipelines on the same query and see results side-by-side

---

## Project Structure

```
multiple-rag-system/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env.example                  # API key template
│
├── core/
│   └── shared_services.py        # Shared LLM, embeddings, ChromaDB, text splitter
│
├── architectures/
│   ├── hybrid_rag.py             # Dense + Sparse with RRF
│   ├── graph_rag.py              # Knowledge graph extraction + retrieval
│   ├── agentic_rag.py            # LangGraph planning agent
│   ├── corrective_rag.py         # CRAG with doc grading + web search fallback
│   ├── multimodal_rag.py         # Vision + text via Gemini
│   └── multilingual_rag.py       # Cross-lingual retrieval
│
├── data/                         # Auto-created at runtime (gitignored)
│   ├── chroma_db/                # ChromaDB persistent store
│   ├── uploads/                  # Temporary uploaded files
│   └── graphs/                   # Serialized knowledge graphs
│
└── docs/
    └── production_migration.md   # Roadmap: FastAPI + Next.js + PostgreSQL migration path
```

---

## How Each Architecture Works

### Hybrid RAG
Runs two retrievers in parallel — ChromaDB (dense semantic search) and BM25 (sparse keyword search). Their ranked result lists are merged using **Reciprocal Rank Fusion** so documents that rank highly in both methods float to the top.

### Graph RAG
Uses the Gemini LLM to extract `(entity, relationship, entity)` triples from every document chunk and builds a **NetworkX knowledge graph**. At query time, it finds matching entities in the graph, walks their neighbors to collect relationship context, and combines that with dense vector retrieval.

### Agentic RAG
A **LangGraph state machine** with three nodes: Planner → Tool Executor → Reasoner. The Planner decides whether to use VECTOR_SEARCH (internal docs), WEB_SEARCH (DuckDuckGo), or answer directly. The Reasoner synthesises the final answer from whatever the tool returned.

### Corrective RAG (CRAG)
A **LangGraph workflow** with five nodes: Retrieve → Evaluate → Route → (Rewrite + Web Search | Generate). The Evaluator grades retrieved docs as CORRECT, AMBIGUOUS, or INCORRECT. INCORRECT triggers a web search fallback; AMBIGUOUS triggers query rewriting before web search.

### Multimodal RAG
Stores the full base64-encoded image in ChromaDB metadata alongside a text summary. At query time, retrieved chunks that have an attached image include both the text context and the raw image in the Gemini multimodal message.

### Multilingual RAG
Uses the same embedding space for all languages (multilingual sentence-transformers). Retrieval is language-agnostic and the generation prompt explicitly asks Gemini to answer in the same language as the query.

---

## Roadmap

The `docs/production_migration.md` file outlines the path to a production API:

- [ ] FastAPI backend with per-user session management
- [ ] Async document ingestion via Celery + Redis worker queue
- [ ] PostgreSQL for session and document metadata persistence
- [ ] Qdrant or pgvector for production-scale vector storage
- [ ] Neo4j for Graph RAG at scale
- [ ] Next.js frontend with real-time status polling
- [ ] JWT authentication + rate limiting
- [ ] Full Docker Compose orchestration

---

## License

MIT
