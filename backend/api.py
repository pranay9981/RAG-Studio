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
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG System API", version="2.0.0", docs_url="/docs")

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
    sources: List[dict] = []

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

    # Image
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
    # Parse target architectures
    if arch_keys.strip() == "all":
        target_keys = ARCH_KEYS
    else:
        try:
            parsed = json.loads(arch_keys)
            target_keys = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            target_keys = [arch_keys.strip()]

    # Build documents
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

    # Ingest into each target pipeline (continue on individual failures)
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

    q: "Queue[tuple]" = Queue()
    t0 = time.time()

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
        while True:
            try:
                event = q.get_nowait()
                kind, content = event

                if kind == "sources":
                    payload = json.dumps({"type": "sources", "content": content})
                elif kind == "token":
                    payload = json.dumps({"type": "token", "content": content})
                elif kind == "step":
                    payload = json.dumps({"type": "step", "content": content})
                elif kind == "done":
                    collected_answer = content.get("answer", "")
                    payload = json.dumps({"type": "done", "answer": collected_answer, "elapsed": content.get("elapsed", 0)})
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
                            "elapsed": content.get("elapsed", 0),
                            "answer":  collected_answer[:120],
                        })
                    break

            except Empty:
                if not thread.is_alive() and q.empty():
                    break
                await asyncio.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/compare")
async def compare(request: CompareRequest):
    results = []

    def run_one(arch_key: str, state_key: str):
        pipeline = session.get_pipeline(state_key)
        start = time.time()
        try:
            answer = pipeline.query(request.query)
            return {"arch_key": arch_key, "answer": answer, "elapsed": round(time.time() - start, 3)}
        except Exception as e:
            return {"arch_key": arch_key, "answer": "", "elapsed": round(time.time() - start, 3), "error": str(e)}

    threads = []
    result_list = [None] * len(ARCH_KEYS)

    def worker(i, arch_key, state_key):
        result_list[i] = run_one(arch_key, state_key)

    for i, arch_key in enumerate(ARCH_KEYS):
        state_key = STATE_KEY_MAP[arch_key]
        t = threading.Thread(target=worker, args=(i, arch_key, state_key), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=120)

    return {"results": [r for r in result_list if r is not None]}


@app.post("/api/evaluate")
async def evaluate_answer(request: EvalRequest):
    context = "\n".join(s.get("text", "") for s in request.sources) if request.sources else ""
    prompt = f"""You are an expert RAG system evaluator. Score the answer on three dimensions.

Question: {request.query}
Retrieved Context (first 1500 chars): {context[:1500] if context else "N/A"}
Generated Answer: {request.answer[:800]}

Score each 0-10:
1. Faithfulness: Is the answer grounded in the context? (0=hallucinated, 10=fully supported)
2. Relevance: Does the answer directly address the question? (0=off-topic, 10=perfectly on-point)
3. Context Precision: Were the right chunks retrieved? (0=irrelevant, 10=perfect)

Output ONLY valid JSON: {{"faithfulness": X, "relevance": X, "context_precision": X}}"""

    try:
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        m = re.search(r"\{[^}]+\}", text)
        if m:
            raw = json.loads(m.group())
            return {k: max(0, min(10, int(raw.get(k, 5)))) for k in ("faithfulness", "relevance", "context_precision")}
    except Exception:
        pass
    return {"faithfulness": 5, "relevance": 5, "context_precision": 5}


@app.get("/api/history")
async def get_history(session_id: str = Query(default="default")):
    return {"history": session.history[-20:]}


@app.delete("/api/sessions/{session_id}")
async def reset_session(session_id: str):
    session.reset()
    return {"status": "reset", "session_id": session_id}
