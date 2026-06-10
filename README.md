# RAG Studio — Multiple RAG Architecture Explorer

> Compare **8 state-of-the-art RAG architectures** side-by-side on your own documents.  
> Two interfaces: a **Next.js + FastAPI** app for portfolio demos, and the original **Streamlit** app for quick testing.

Upload a PDF, DOCX, TXT, or image — then ask questions and watch each pipeline think in real time. Switch between architectures, run all 8 simultaneously, and score answers with an LLM-as-judge evaluator.

---

## Architectures

| # | Architecture | Core Idea | Best For |
|---|---|---|---|
| 01 | **Hybrid RAG** | Dense vectors + BM25 sparse search fused via Reciprocal Rank Fusion + cross-encoder re-ranking | General-purpose documents, mixed query types |
| 02 | **Graph RAG** | LLM-extracted entity/relationship triples → NetworkX knowledge graph + vector fallback | Documents rich in named entities and relationships |
| 03 | **Agentic RAG** | LangGraph planner routes to VECTOR_SEARCH, WEB_SEARCH (DuckDuckGo), or direct answer | Queries that may need web context or multi-step reasoning |
| 04 | **Corrective RAG (CRAG)** | Evaluator grades retrieved docs as CORRECT / AMBIGUOUS / INCORRECT; rewrites query and falls back to web search | When retrieval quality is uncertain |
| 05 | **Multimodal RAG** | Gemini vision summarises uploaded images; base64 stored in metadata; image + text sent at query time | Documents with figures, charts, screenshots |
| 06 | **Multilingual RAG** | Cross-lingual embedding space; cross-encoder re-ranking; answers in the query's language | Multilingual documents or cross-language queries |
| 07 | **RAG-Fusion** | Expands query into 4 sub-queries, retrieves separately for each, merges all ranked lists with RRF | Ambiguous or broad queries |
| 08 | **HyDE RAG** | Generates a hypothetical ideal answer first, embeds it, uses it as the search vector | Short or keyword-style queries worded differently from the source text |

---

## Features

- **Live "brain working" view** — real-time step indicators + token streaming with a blinking cursor as each pipeline executes
- **Compare mode** — run all 8 architectures simultaneously and see results in a card grid; expand any card for the full formatted answer
- **RAG Evaluation scorecard** — LLM-as-judge scores every answer on Faithfulness, Relevance, and Context Precision (0–10)
- **Source citations** — collapsible panel showing retrieved chunks, source filenames, and relevance scores
- **Multi-document support** — additive ingestion; upload as many files as you like without wiping previous documents
- **URL ingestion** — paste a webpage URL and it's scraped, chunked, and ingested automatically
- **Knowledge graph visualisation** — interactive PyVis graph for Graph RAG (Streamlit UI)
- **Chat export** — download the conversation as a Markdown file
- **Query history** — last 15–20 queries with timing in the sidebar
- **Architecture info cards** — expandable how-it-works descriptions for every pipeline

---

## Tech Stack

| Layer | Choice |
|---|---|
| **LLM** | Google Gemini `gemini-3.1-flash-lite` via `langchain-google-genai` |
| **Embeddings** | `all-MiniLM-L6-v2` via `langchain-huggingface` |
| **Vector DB** | ChromaDB (in-memory `EphemeralClient` — no disk setup required) |
| **Sparse Search** | `rank-bm25` — BM25Okapi |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| **Knowledge Graph** | NetworkX (in-memory) + PyVis (interactive visualisation) |
| **Agentic Workflows** | LangGraph — Agentic RAG and CRAG state machines |
| **Web Search Fallback** | DuckDuckGo (`duckduckgo-search`) — no API key required |
| **Document Parsing** | PyPDF2, python-docx |
| **Backend API** | FastAPI + Uvicorn, SSE streaming via `threading.Queue` |
| **Frontend** | Next.js 14, React 18, Tailwind CSS, TypeScript |
| **Original UI** | Streamlit |
| **Python** | 3.11+ |

---

## Project Structure

```
multiple-rag-system/
│
├── app.py                          # Streamlit UI — all 8 architectures, full features
├── requirements.txt
├── .env / .env.example             # GOOGLE_API_KEY
│
├── core/
│   └── shared_services.py          # Singleton: LLM, embeddings, ChromaDB, reranker, text splitter
│
├── architectures/
│   ├── hybrid_rag.py               # Dense + BM25 + RRF + cross-encoder
│   ├── graph_rag.py                # Entity extraction → NetworkX graph + vector fallback
│   ├── agentic_rag.py              # LangGraph: Planner → Tool Executor → Reasoner
│   ├── corrective_rag.py           # LangGraph: Retrieve → Evaluate → Route → Generate
│   ├── multimodal_rag.py           # Gemini vision + base64 image metadata
│   ├── multilingual_rag.py         # Cross-lingual embeddings + cross-encoder
│   ├── rag_fusion.py               # 4 sub-queries + RRF fusion
│   └── hyde_rag.py                 # Hypothetical document embeddings
│
├── backend/
│   ├── __init__.py
│   ├── session_manager.py          # GlobalSession singleton — initialises all 8 pipelines
│   └── api.py                      # FastAPI: SSE /api/query, /api/ingest, /api/compare, /api/evaluate
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # Root — all state + streaming logic
│   │   └── globals.css             # Dark theme, Inter font, animations
│   ├── components/
│   │   ├── Sidebar.tsx             # Arch list, toggles, doc library, query history
│   │   ├── ArchCard.tsx            # Collapsible architecture info card
│   │   ├── ChatMessage.tsx         # User / assistant chat bubbles
│   │   ├── MarkdownContent.tsx     # Lightweight inline markdown renderer
│   │   ├── BrainWorking.tsx        # Live step indicators + streaming tokens + cursor
│   │   ├── SourcePanel.tsx         # Collapsible source citations
│   │   ├── EvalScorecard.tsx       # Faithfulness / Relevance / Precision pills
│   │   ├── DocumentManager.tsx     # Drag-drop file + URL ingestion
│   │   └── CompareGrid.tsx         # 8-card grid with scrollable cards + expand modal
│   ├── lib/
│   │   ├── api.ts                  # All API calls + EventSource streamQuery()
│   │   └── types.ts                # TypeScript interfaces
│   ├── package.json
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── docs/
│   └── production_migration.md
│
└── .github/
    └── workflows/ci.yml
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the Next.js frontend only)
- A free [Google AI Studio API key](https://aistudio.google.com/apikey)

---

### 1 — Clone and set up Python

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

```bash
cp .env.example .env
```

Open `.env` and set:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

---

### Option A — Next.js + FastAPI (recommended for demos)

Run the FastAPI backend and Next.js frontend in two separate terminals.

**Terminal 1 — FastAPI backend**

```bash
# Windows (venv already activated)
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Next.js frontend**

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

> **Note:** Do NOT run FastAPI and Streamlit at the same time — they both initialise the same ChromaDB collections on startup and will conflict.

---

### Option B — Streamlit (quick testing)

```bash
# venv activated
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## How to Use

1. **Upload a document** — drag and drop a PDF, TXT, DOCX, PNG, or JPG, or paste a URL
2. **Ingest** — chunks and embeds the document into the selected architecture (or all 8 if Compare mode is on)
3. **Ask a question** — press Enter to send; watch the brain-working panel stream live steps and tokens
4. **Compare mode** — toggle "Compare all 8" to run every architecture simultaneously; results appear in a card grid with expand buttons for full answers
5. **RAG Evaluation** — toggle "RAG Evaluation" to get an LLM-as-judge score after each answer

---

## Architecture Deep Dives

### 01 Hybrid RAG
Two retrievers run in parallel: ChromaDB (dense semantic vectors) and BM25 (sparse keyword matching). Their ranked lists are merged with **Reciprocal Rank Fusion** — documents appearing high in both lists float to the top. A `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker then scores the fused candidates for final precision.

### 02 Graph RAG
Gemini extracts `(source, relationship, target)` triples from every chunk at ingest time and builds an in-memory **NetworkX** graph. At query time, the query's entities are matched against graph nodes, their neighbours are walked to collect relationship context, and that graph text is combined with dense vector results before generation.

### 03 Agentic RAG
A three-node **LangGraph** state machine: **Planner** decides whether to use `VECTOR_SEARCH`, `WEB_SEARCH` (DuckDuckGo), or answer directly. The **Tool Executor** runs the chosen tool. The **Reasoner** synthesises the final answer from the tool's output.

### 04 Corrective RAG (CRAG)
A five-node **LangGraph** workflow: **Retrieve → Evaluate → Route → Generate**. The Evaluator grades each retrieved document as `CORRECT`, `AMBIGUOUS`, or `INCORRECT`. `INCORRECT` triggers a DuckDuckGo web search fallback; `AMBIGUOUS` triggers a query rewrite before web search.

### 05 Multimodal RAG
When an image is uploaded, Gemini generates a detailed text description for embedding. The raw base64 image is stored alongside the text in ChromaDB metadata. At query time, both the text context and the original image are included in the Gemini multimodal message, so visual content is understood natively.

### 06 Multilingual RAG
Uses a multilingual sentence-transformer so all languages share the same embedding space — no translation needed. A cross-encoder re-ranks results, and the generation prompt instructs Gemini to respond in the same language as the query.

### 07 RAG-Fusion
Gemini generates **4 different phrasings** of the original query. Each sub-query retrieves its own ranked list. All four lists are merged with Reciprocal Rank Fusion — documents that appear across multiple sub-query results are boosted, capturing a broader semantic net than any single query alone.

### 08 HyDE RAG
Gemini generates a **hypothetical ideal answer** as if it were already in the document. That hypothetical is embedded and used as the search vector instead of the raw query — bridging the vocabulary gap between how questions are asked and how answers are written in source documents.

---

## FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/architectures` | All 8 architecture metadata |
| `GET` | `/api/sessions/{id}` | Session info, ingested archs, doc library |
| `POST` | `/api/ingest` | File or URL ingest (multipart/form-data) |
| `GET` | `/api/query` | SSE stream — `step` / `token` / `sources` / `done` events |
| `POST` | `/api/compare` | Run all 8 architectures concurrently, return JSON |
| `POST` | `/api/evaluate` | LLM-as-judge scoring (faithfulness, relevance, context precision) |
| `GET` | `/api/history` | Last 20 queries |
| `DELETE` | `/api/sessions/{id}` | Reset session — clears all pipelines and history |

---

## License

MIT
