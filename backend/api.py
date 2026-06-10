import os
import re
import json
import time
import base64
import asyncio
import tempfile
import threading
import urllib.request
from queue import Queue, Empty
from typing import Optional, List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.session_manager import session, ARCH_KEYS, STATE_KEY_MAP, ARCH_INFO
from core.shared_services import services
from core.adaptive_db import adaptive_db
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG System API", version="3.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    query: str
    session_id: str = "default"

class EvalRequest(BaseModel):
    query: str
    answer: str
    arch_key: str = ""
    sources: List[dict] = []

class FeedbackRequest(BaseModel):
    query: str
    arch_key: str
    chunk_ids: List[str] = []
    rating: int  # 1 = thumbs up, -1 = thumbs down
    message_id: str = ""

# ── Demo document ─────────────────────────────────────────────────────────────

DEMO_TEXT = """# RAG System Architectures: A Comprehensive Guide

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge bases before generating responses. Instead of relying solely on the model's training data, RAG systems dynamically fetch contextual information, making answers more accurate, up-to-date, and grounded in specific documents.

The core RAG pipeline consists of three stages: Indexing (documents are chunked and converted to embeddings stored in a vector database), Retrieval (user queries are embedded and matched against stored vectors to find relevant chunks), and Generation (retrieved context is combined with the query and sent to an LLM for response).

## The Eight RAG Architectures

### 1. Hybrid RAG (Dense + Sparse + Re-ranking)

Hybrid RAG combines two complementary retrieval strategies: dense vector search and sparse BM25 keyword matching. Dense retrieval captures semantic meaning — it finds documents conceptually related to the query even when exact words do not match. BM25 excels at exact keyword matching, ensuring specific terms are not missed. The two ranked result lists are merged using Reciprocal Rank Fusion (RRF) with a parameter k=60. A cross-encoder model (ms-marco-MiniLM-L-6-v2) then re-ranks the fused candidates by scoring each query-document pair directly. Hybrid RAG is best for general-purpose document retrieval.

### 2. Graph RAG (Knowledge Graphs)

Graph RAG builds a NetworkX knowledge graph by extracting entity-relationship triples from ingested documents using Gemini. Each extraction produces triples of the form: source entity, relationship, target entity. At query time, Graph RAG extracts entities from the query, traverses the knowledge graph to find connected entities and relationships, and combines this graph context with dense vector retrieval results. Graph RAG is best for documents rich in named entities and interconnected concepts such as research papers and technical reports.

### 3. Agentic RAG (LangGraph Planner)

Agentic RAG uses a LangGraph state machine with three nodes: Planner, Tool Executor, and Reasoner. The Planner node is a decision-making agent that analyzes the query to choose between VECTOR_SEARCH, WEB_SEARCH via DuckDuckGo, or ANSWER directly. A multi-hop extension allows Agentic RAG to decompose complex queries into sub-questions, retrieve context for each sub-question separately, and synthesize a comprehensive answer. Agentic RAG is best for queries that may need web context or multi-step reasoning.

### 4. Corrective RAG (CRAG)

Corrective RAG implements a 5-node LangGraph workflow: Retrieve, Evaluate, Route, optional Rewrite plus Web Search, and Generate. The key innovation is the Evaluate node which uses the LLM to judge whether retrieved documents contain sufficient information. The Evaluator classifies retrieval quality as CORRECT (sufficient context), AMBIGUOUS (partial context), or INCORRECT (insufficient context). CORRECT routes directly to generation. AMBIGUOUS triggers query rewrite followed by web search. INCORRECT bypasses original documents entirely and fetches from DuckDuckGo. CRAG is best when document coverage is uncertain.

### 5. Multimodal RAG (Vision + Text)

Multimodal RAG handles both text and image inputs. When an image is uploaded, Gemini Vision generates a detailed text description which is embedded and stored in ChromaDB along with the base64-encoded image in metadata. At query time it retrieves both text chunks and image chunks and sends them together to Gemini Vision. Multimodal RAG is best for documents containing charts, diagrams, screenshots, or mixed image-text content.

### 6. Multilingual RAG (Cross-lingual)

Multilingual RAG uses a multilingual sentence transformer that maps text from over 100 languages into a shared vector space. A query in French can retrieve documents written in English or any other supported language without explicit translation. A cross-encoder re-ranks results and Gemini generates a response in the same language as the user query. Multilingual RAG is best for multilingual document collections or when users query in different languages.

### 7. RAG-Fusion (Multi-Query + RRF)

RAG-Fusion generates 4 different phrasings of the original query using Gemini. Each phrasing retrieves its own ranked list from ChromaDB independently. The 4 ranked lists are merged using Reciprocal Rank Fusion — documents appearing in multiple sub-query result lists receive significantly boosted scores. RAG-Fusion is best for ambiguous, broad, or compound queries.

### 8. HyDE RAG (Hypothetical Document Embeddings)

HyDE RAG generates a hypothetical ideal answer first. That hypothetical is embedded, and real document chunks closest to this hypothetical embedding are retrieved. The key insight is that the vocabulary gap between questions and answers is large. By generating a hypothetical answer first, HyDE bridges this gap by searching in the answer space rather than the question space. HyDE is best for short or keyword-style queries worded very differently from the source text.

## Evaluation Metrics

RAG systems are evaluated on four dimensions. Faithfulness measures whether claims in the generated answer are grounded in the retrieved context, scored 0 to 10. Answer Relevance measures whether the answer directly addresses the user question, scored 0 to 10. Context Precision measures whether the retrieved chunks were relevant to the query, scored 0 to 10. Context Recall measures whether the retrieval captured all information needed to fully answer the query, scored 0 to 10.

## Adaptive RAG: Learning From Interactions

This system implements several adaptive mechanisms. The Semantic Query Cache stores answered queries with their embeddings. When a new query is semantically similar (cosine similarity above 0.92) to a previously answered query for the same architecture, the cached answer is returned instantly. Feedback-Driven Retrieval allows users to rate answers with thumbs up or thumbs down, which is stored for analytics and future improvements. Context Quality Evaluation checks retrieved context quality before generating any answer — if context quality is INCORRECT the pipeline falls back to web search, if AMBIGUOUS web search supplements the original context. The Analytics Dashboard tracks query counts, average latencies, eval scores, and feedback ratios per architecture.

## Choosing the Right Architecture

For general questions about a document use Hybrid RAG for best precision. For relationship and entity questions use Graph RAG for network context. For questions that may need web data use Agentic RAG or CRAG. For images and charts use Multimodal RAG. For non-English documents or queries use Multilingual RAG. For broad or ambiguous questions use RAG-Fusion. For short keyword queries against long documents use HyDE RAG."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_url_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>",   "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


async def process_file(file: UploadFile) -> List[Document]:
    content = await file.read()
    name = (file.filename or "").lower()
    source = file.filename or "upload"

    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(content)
            tmp = f.name
        docs = services.load_pdf(tmp)
        os.remove(tmp)
        return docs

    if name.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
        chunks = services.text_splitter.split_text(text)
        return [Document(page_content=c, metadata={"source": source, "type": "txt"}) for c in chunks]

    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(content)
            tmp = f.name
        from docx import Document as DocxDocument
        docx = DocxDocument(tmp)
        text = "\n".join(p.text for p in docx.paragraphs if p.text.strip())
        os.remove(tmp)
        chunks = services.text_splitter.split_text(text)
        return [Document(page_content=c, metadata={"source": source, "type": "docx"}) for c in chunks]

    if name.endswith(".csv"):
        import pandas as pd
        import io as _io
        df = pd.read_csv(_io.BytesIO(content))
        schema = (
            f"Columns: {list(df.columns)}\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Dtypes:\n" + "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
        )
        preview = df.head(500).to_csv(index=False)
        text = f"TABLE SCHEMA:\n{schema}\n\nDATA:\n{preview}"
        return [Document(
            page_content=text,
            metadata={
                "source": source, "type": "csv",
                "columns": json.dumps(list(df.columns)),
                "rows": str(df.shape[0]),
            },
        )]

    if name.endswith((".xlsx", ".xls")):
        import pandas as pd
        import io as _io
        df = pd.read_excel(_io.BytesIO(content))
        schema = (
            f"Columns: {list(df.columns)}\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
        )
        preview = df.head(500).to_csv(index=False)
        text = f"TABLE SCHEMA:\n{schema}\n\nDATA:\n{preview}"
        return [Document(
            page_content=text,
            metadata={
                "source": source, "type": "excel",
                "columns": json.dumps(list(df.columns)),
                "rows": str(df.shape[0]),
            },
        )]

    b64 = base64.b64encode(content).decode("utf-8")
    msg = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    summary = services.extract_response_text(services.llm.invoke([msg]))
    return [Document(
        page_content=f"Image Description: {summary}",
        metadata={"source": source, "type": "image", "image_base64": b64},
    )]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "architectures": len(ARCH_KEYS)}


@app.get("/api/architectures")
async def get_architectures():
    return list(ARCH_INFO.values())


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    return {
        "session_id": "default",
        "ingested_archs": list(session.ingested_archs),
        "doc_library": session.doc_library,
        "history_count": len(session.history),
    }


@app.post("/api/ingest")
async def ingest_document(
    session_id: str = Form(default="default"),
    arch_keys: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
):
    if arch_keys.strip() == "all":
        target_keys = ARCH_KEYS
    else:
        try:
            parsed = json.loads(arch_keys)
            target_keys = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            target_keys = [arch_keys.strip()]

    docs: List[Document] = []
    source_name = ""

    if file is not None and file.filename:
        source_name = file.filename
        docs = await process_file(file)
    elif url and url.strip():
        source_name = url.strip()
        try:
            raw = fetch_url_text(source_name)
            if len(raw) < 100:
                raise HTTPException(status_code=422, detail="Could not extract meaningful text from URL")
            chunks = services.text_splitter.split_text(raw)
            docs = [Document(page_content=c, metadata={"source": source_name, "type": "url"}) for c in chunks]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"URL fetch failed: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL")

    if not docs:
        raise HTTPException(status_code=422, detail="No content could be extracted")

    # Parent-child chunking: small children for retrieval, parent text stored in metadata for generation
    docs = services.create_parent_child_documents(docs)

    ingested: List[str] = []
    for arch_key in target_keys:
        state_key = STATE_KEY_MAP.get(arch_key)
        if not state_key:
            continue
        pipeline = session.get_pipeline(state_key)
        if pipeline:
            try:
                pipeline.ingest(docs)
                session.ingested_archs.add(arch_key)
                ingested.append(arch_key)
            except Exception as exc:
                print(f"[ingest] {arch_key} failed: {exc}")

    session.doc_library.append({"name": source_name, "chunks": len(docs)})
    return {"chunks": len(docs), "source": source_name, "architectures": ingested}


@app.post("/api/demo/load")
async def load_demo():
    """Ingest the built-in demo document into all 8 architectures."""
    chunks = services.text_splitter.split_text(DEMO_TEXT)
    docs = [
        Document(page_content=c, metadata={"source": "RAG System Guide (Demo)", "type": "demo"})
        for c in chunks
    ]
    docs = services.create_parent_child_documents(docs)

    ingested: List[str] = []
    for arch_key in ARCH_KEYS:
        state_key = STATE_KEY_MAP.get(arch_key)
        pipeline = session.get_pipeline(state_key)
        if pipeline:
            try:
                pipeline.ingest(docs)
                session.ingested_archs.add(arch_key)
                ingested.append(arch_key)
            except Exception as exc:
                print(f"[demo] {arch_key} failed: {exc}")

    session.doc_library.append({"name": "RAG System Guide (Demo)", "chunks": len(docs)})
    return {"chunks": len(docs), "source": "RAG System Guide (Demo)", "architectures": ingested}


@app.get("/api/query")
async def query_stream(
    session_id: str = Query(default="default"),
    query: str = Query(...),
    arch_key: str = Query(...),
):
    state_key = STATE_KEY_MAP.get(arch_key)
    if not state_key:
        raise HTTPException(status_code=400, detail=f"Unknown architecture: {arch_key}")

    pipeline = session.get_pipeline(state_key)
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not found")

    # Check semantic cache
    try:
        query_embedding = services.embeddings.embed_query(query)
        cached = adaptive_db.find_similar_query(query_embedding, arch_key)
    except Exception:
        cached = None
        query_embedding = None

    if cached:
        async def cached_generator():
            sim = cached["similarity"]
            yield f'data: {json.dumps({"type": "step", "content": f"⚡ Semantic cache hit (similarity {sim}) — reusing previous answer"})}\n\n'
            if cached["sources"]:
                yield f'data: {json.dumps({"type": "sources", "content": cached["sources"]})}\n\n'
            # Stream cached answer token by token for visual continuity
            for word in cached["answer"].split(" "):
                yield f'data: {json.dumps({"type": "token", "content": word + " "})}\n\n'
            yield f'data: {json.dumps({"type": "done", "answer": cached["answer"], "elapsed": 0.001, "cached": True})}\n\n'
            adaptive_db.store_query_analytics(arch_key, query, 0.001, cached=True)

        return StreamingResponse(
            cached_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    q: "Queue[tuple]" = Queue()
    t0 = time.time()
    collected_sources: List[dict] = []

    def on_step(event):
        q.put(event)

    def run_pipeline():
        try:
            result = pipeline.query(query, on_step=on_step)
            elapsed = time.time() - t0
            q.put(("done", {"answer": result, "elapsed": round(elapsed, 3)}))
        except Exception as e:
            q.put(("error", {"message": str(e)}))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    async def event_generator():
        collected_answer = ""
        elapsed = 0.0

        while True:
            try:
                event = q.get_nowait()
                kind, content = event

                if kind == "sources":
                    collected_sources.extend(content)
                    payload = json.dumps({"type": "sources", "content": content})
                elif kind == "token":
                    payload = json.dumps({"type": "token", "content": content})
                elif kind == "step":
                    payload = json.dumps({"type": "step", "content": content})
                elif kind == "done":
                    collected_answer = content.get("answer", "")
                    elapsed = content.get("elapsed", 0)
                    payload = json.dumps({
                        "type": "done",
                        "answer": collected_answer,
                        "elapsed": elapsed,
                        "cached": False,
                    })
                elif kind == "error":
                    payload = json.dumps({"type": "error", "content": content.get("message", "Unknown error")})
                else:
                    payload = json.dumps({"type": kind, "content": str(content)})

                yield f"data: {payload}\n\n"

                if kind in ("done", "error"):
                    if kind == "done":
                        session.history.append({
                            "query":   query,
                            "arch":    arch_key,
                            "elapsed": elapsed,
                            "answer":  collected_answer[:120],
                        })
                        adaptive_db.store_query_analytics(arch_key, query, elapsed)
                        # Store in semantic cache for future similar queries
                        if collected_answer and query_embedding:
                            try:
                                adaptive_db.store_query_cache(
                                    arch_key, query, query_embedding,
                                    collected_answer, collected_sources[:5],
                                )
                            except Exception as e:
                                print(f"[cache] store_query_cache failed: {e}")
                    break

            except Empty:
                if not thread.is_alive() and q.empty():
                    break
                await asyncio.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/compare")
async def compare(request: CompareRequest):
    result_list = [None] * len(ARCH_KEYS)

    def run_one(i: int, arch_key: str, state_key: str):
        pipeline = session.get_pipeline(state_key)
        start = time.time()
        try:
            answer = pipeline.query(request.query)
            result_list[i] = {
                "arch_key": arch_key,
                "answer": answer,
                "elapsed": round(time.time() - start, 3),
            }
        except Exception as e:
            result_list[i] = {
                "arch_key": arch_key,
                "answer": "",
                "elapsed": round(time.time() - start, 3),
                "error": str(e),
            }

    threads = []
    for i, arch_key in enumerate(ARCH_KEYS):
        t = threading.Thread(
            target=run_one, args=(i, arch_key, STATE_KEY_MAP[arch_key]), daemon=True
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=120)

    return {"results": [r for r in result_list if r is not None]}


@app.post("/api/evaluate")
async def evaluate_answer(request: EvalRequest):
    context = "\n".join(s.get("text", "") for s in request.sources) if request.sources else ""
    scores = {"faithfulness": 0, "relevance": 0, "context_precision": 0, "context_recall": 0}
    eval_ok = False

    # ── Step 1: RAGAS-style faithfulness — extract claims then verify each ────
    faithfulness_score = 0
    try:
        claims_prompt = f"""Break this answer into individual factual claims. Each claim = one verifiable statement.
Answer: {request.answer[:600]}
Output ONLY a JSON array of strings: ["claim1", "claim2", ...]"""
        claims_resp = services.llm.invoke(claims_prompt)
        claims_text = services.extract_response_text(claims_resp)
        cm = re.search(r"\[.*?\]", claims_text, re.DOTALL)
        claims = json.loads(cm.group()) if cm else []

        if claims and context:
            verify_prompt = f"""For each claim, output true if it is SUPPORTED by the context, false otherwise.
Context: {context[:1500]}
Claims: {json.dumps(claims[:10])}
Output ONLY a JSON array of booleans (one per claim): [true, false, ...]"""
            verify_resp = services.llm.invoke(verify_prompt)
            verify_text = services.extract_response_text(verify_resp)
            vm = re.search(r"\[.*?\]", verify_text, re.DOTALL)
            supported = json.loads(vm.group()) if vm else []
            if supported and claims:
                faithfulness_score = round(sum(1 for s in supported if s) / len(claims) * 10)
    except Exception as e:
        print(f"[evaluate] faithfulness step failed: {e}")

    # ── Step 2: relevance, context_precision, context_recall in one call ─────
    try:
        other_prompt = f"""You are an expert RAG evaluator. Score on three dimensions.

Question: {request.query}
Retrieved Context: {context[:1200] if context else "N/A"}
Generated Answer: {request.answer[:600]}

Score each 0-10:
1. Relevance: Does the answer directly address the question? (0=off-topic, 10=perfect)
2. Context Precision: Were the retrieved chunks relevant to the query? (0=all irrelevant, 10=all relevant)
3. Context Recall: Did the context contain all information needed to fully answer? (0=missing key info, 10=complete)

Output ONLY valid JSON: {{"relevance": X, "context_precision": X, "context_recall": X}}"""

        resp = services.llm.invoke(other_prompt)
        text = services.extract_response_text(resp)
        m = re.search(r"\{[^}]+\}", text)
        if m:
            raw = json.loads(m.group())
            scores = {
                "faithfulness": faithfulness_score,
                "relevance": max(0, min(10, int(raw.get("relevance", 0)))),
                "context_precision": max(0, min(10, int(raw.get("context_precision", 0)))),
                "context_recall": max(0, min(10, int(raw.get("context_recall", 0)))),
            }
            eval_ok = True
    except Exception as e:
        print(f"[evaluate] other metrics step failed: {e}")
        if faithfulness_score:
            scores["faithfulness"] = faithfulness_score
            eval_ok = True

    # Only store analytics when at least one LLM step succeeded
    if eval_ok and request.arch_key:
        try:
            adaptive_db.store_eval_analytics(request.arch_key, request.query, scores)
        except Exception as e:
            print(f"[evaluate] store_eval_analytics failed: {e}")

    return scores


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 (up) or -1 (down)")
    adaptive_db.store_feedback(
        request.query, request.arch_key, request.chunk_ids, request.rating
    )
    return {"status": "ok"}


@app.get("/api/analytics")
async def get_analytics():
    return {
        "data": adaptive_db.get_analytics(),
        "recent": adaptive_db.get_recent_queries(20),
    }


@app.get("/api/graph")
async def get_graph():
    pipeline = session.get_pipeline("graph_pipeline")
    if not pipeline:
        return {"html": ""}
    try:
        html = pipeline.render_graph_html()
        return {"html": html or ""}
    except Exception as e:
        return {"html": "", "error": str(e)}


@app.get("/api/history")
async def get_history(session_id: str = Query(default="default")):
    return {"history": session.history[-20:]}


@app.delete("/api/sessions/{session_id}")
async def reset_session(session_id: str):
    session.reset()
    return {"status": "reset", "session_id": session_id}
