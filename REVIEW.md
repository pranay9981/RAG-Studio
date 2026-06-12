# RAG Studio — Full Re-Audit Report
**Date:** 2026-06-12 (Session 14)
**Scope:** All 10 architectures, backend core + API, frontend (8 components)
**Auditors:** 4 parallel specialist agents (backend, arch 1-5, arch 6-10, frontend)
**Status:** 35 findings — 12 CRITICAL · 13 WARNING · 10 INFO

---

## CRITICAL

### CR-01 — `df.query()` in structured_rag allowlist: confirmed RCE via pandas frame locals
**File:** `architectures/structured_rag.py:23`
`df.query("@pd.read_csv('/any/path')")` passes `_ast_safe_eval` fully. Pandas' query engine resolves `@pd` by walking the **calling frame's locals** — `pd` is a local in `_run_pandas_query`, so it is reachable. Prompt-injected LLM output can trigger arbitrary file reads or other `pd.*` calls.
**Fix:** Remove `'query'` from `_ALLOWED_ATTRS`.

---

### CR-02 — `to_csv` in structured_rag allowlist: arbitrary filesystem write
**File:** `architectures/structured_rag.py:24`
`df.to_csv("/any/server/path")` passes all AST checks and silently writes the DataFrame to disk. `to_csv()` with no argument is safe (returns a string), but the allowlist cannot distinguish calling modes.
**Fix:** Remove `'to_csv'` from `_ALLOWED_ATTRS`.

---

### CR-03 — `apply` in structured_rag allowlist: arbitrary callable execution
**File:** `architectures/structured_rag.py:25`
`df['col'].apply(pd.sum)` — `sum` is allowed, `pd` is an allowed name. `apply` paired with allowed callables expands the attack surface significantly.
**Fix:** Remove `'apply'` from `_ALLOWED_ATTRS`.

---

### CR-04 — multimodal_rag opens image file in text mode — base64 corrupted on Windows
**File:** `architectures/multimodal_rag.py:105`
`open(img_path)` (no `"rb"` flag) returns a `str` of raw characters on Windows (CRLF translation, encoding errors). The result is used as `data:image/jpeg;base64,{b64_data}` — the model receives corrupted or invalid data on every image query.
**Fix:**
```python
import base64
with open(img_path, "rb") as fh:
    b64_data = base64.b64encode(fh.read()).decode("utf-8")
```

---

### CR-05 — Double `collection.count()` race → `n_results=0` crash in compare mode
**Files:** `architectures/agentic_rag.py:54-59`, `architectures/corrective_rag.py:55,58`, `architectures/multimodal_rag.py:50,59`
Each file calls `collection.count()` for a guard check, then calls it again (unprotected) to compute `n_results`. A concurrent `reset()` between the two calls produces `n_results = min(4, 0) = 0`, which ChromaDB rejects with `ValueError`.
**Fix:** Cache the count from the first call and reuse it:
```python
count = self.collection.count()
if not count:
    return ...
n = min(4, count)
```

---

### CR-06 — `nx.Graph` in graph_rag mutated across threads without any lock
**File:** `architectures/graph_rag.py:73,124,209`
`ingest()` calls `self.graph.add_edge(...)`, `query()` iterates `self.graph.nodes()`, and `reset()` calls `self.graph.clear()` — all concurrently in compare mode. NetworkX graphs are not thread-safe. Produces `RuntimeError: dictionary changed size during iteration` or silent corruption.
**Fix:** Add `self._graph_lock = threading.Lock()` in `__init__`, wrap all graph reads and writes with it.

---

### CR-07 — corrective_rag: LLM failure + empty docs routes to `generate_node` (hallucination)
**File:** `architectures/corrective_rag.py:55-56,126-129`
When no docs are ingested and the LLM is unreachable, `evaluate_context()` returns `"AMBIGUOUS"` (its failure fallback). `route_evaluation` maps `AMBIGUOUS → generate_node`, which generates from an empty context string — the LLM hallucinates freely.
**Fix:**
```python
# in generate_node, add early exit:
if not state.get("documents"):
    return {"answer": "No context available. Please ingest a document or check your API key."}
```

---

### CR-08 — multilingual_rag `delete_collection()` in `ingest()` exception handler has no lock
**File:** `architectures/multilingual_rag.py:57`
The dimension-mismatch handler calls `services.chroma_client.delete_collection(...)` without `services._chroma_lock`. A concurrent `chroma_query()` holding the lock alongside an unprotected `delete_collection` produces undefined state or a crash.
**Fix:** Wrap the delete+recreate in `with services._chroma_lock:`.

---

### CR-09 — `chroma_query` rebuild has TOCTOU gap between `get()` and `delete_collection()`
**File:** `core/shared_services.py:79-98`
`collection.get(...)` is called under one lock acquisition, then the lock is released. Before the next acquisition (delete+recreate+add), a concurrent ingest thread could add documents. When the rebuild thread deletes the collection those additions are silently lost.
**Fix:** Re-read the collection inside the final lock before deleting:
```python
with self._chroma_lock:
    existing2 = collection.get(include=["documents", "metadatas"])
    self.chroma_client.delete_collection(collection_name)
    new_col = self.chroma_client.get_or_create_collection(collection_name)
    new_col.add(ids=existing2["ids"], documents=existing2["documents"],
                embeddings=new_embeddings, metadatas=existing2["metadatas"])
```

---

### CR-10 — SSE `JSON.parse` in frontend is unguarded — malformed frame crashes the stream
**File:** `frontend/lib/api.ts:58`
`JSON.parse(e.data)` has no try/catch. Any malformed SSE frame (keep-alive ping, network corruption) throws, fires `onerror`, and terminates the stream irreversibly.
**Fix:** Wrap in `try { d = JSON.parse(e.data) } catch { return }`.

---

### CR-11 — `compareAll` swallows all errors — user sees empty grid with no feedback
**File:** `frontend/lib/api.ts:82-96`, `frontend/app/page.tsx:171`
`compareAll` catches all exceptions and returns `[]`. The empty array is written to state with no error message shown. A backend failure is completely invisible.
**Fix:** Re-throw from `compareAll`; add a `catch` in `handleSend` that appends an error message to chat.

---

### CR-12 — API key endpoint: no format validation; error may echo the key
**File:** `backend/api.py:723-739`
Any string is accepted. If the Groq SDK rejects it, the exception message (which may contain the key) is returned verbatim in the 400 response.
**Fix:** Add `re.match(r'^gsk_[A-Za-z0-9]{40,}$', key)` validation. Return a generic error, not the raw exception.

---

## WARNING

### WR-01 — `multilingual_rag` source labels wrong after cross-encoder rerank
**File:** `architectures/multilingual_rag.py:85-109`
`zip(reranked_texts, metas)` pairs by position — but reranking changes order. Every source label shown to the user is wrong.
**Fix:** Build `doc_meta_map = {text: meta ...}` before reranking; look up by text after.

---

### WR-02 — `self_rag` short grader response silently drops documents
**File:** `architectures/self_rag.py:88-91`
If the LLM returns fewer grades than docs, `zip` stops early and those docs are dropped. The fallback only catches an empty result, not a partial one.
**Fix:** Pad grades to `len(docs)` with `True`.

---

### WR-03 — `self_rag` post-loop fallback block is unreachable dead code
**File:** `architectures/self_rag.py:253-258`
The `is_final = True` loop always returns inside the loop body. Lines 253-258 are never reached.
**Fix:** Remove the dead block.

---

### WR-04 — `structured_rag._table_store` dict mutated and iterated without a lock
**File:** `architectures/structured_rag.py:112,165`
`ingest()` writes and `query()` iterates `_table_store` concurrently in compare mode → `RuntimeError: dictionary changed size during iteration`.
**Fix:** Add `self._table_lock = threading.Lock()` and protect both `ingest()` and `query()` accesses.

---

### WR-05 — `hybrid_rag` sources event reflects pre-fallback chunks; context uses web results
**File:** `architectures/hybrid_rag.py:171-188`
`on_step(("sources", ...))` is called using `scored` (vector results). If web fallback replaces `reranked_texts`, the UI shows vector-retrieved sources while the LLM answers from web content.
**Fix:** Move the sources event emission to after the web-fallback block.

---

### WR-06 — `hybrid_rag` in-memory state extended before ChromaDB write
**File:** `architectures/hybrid_rag.py:66-79`
`self.chunks` and `self.chunk_ids` are extended before `collection.add()`. On failure, BM25 has ghost entries not in ChromaDB.
**Fix:** Extend in-memory state only after a successful `collection.add()`.

---

### WR-07 — All `reset()` methods call `delete_collection` without `services._chroma_lock`
**Files:** `architectures/corrective_rag.py:36-41`, `architectures/agentic_rag.py:36-41`, `architectures/graph_rag.py:67-78`, `architectures/multimodal_rag.py`, `architectures/multilingual_rag.py`, `architectures/self_rag.py`, `architectures/rag_fusion.py`, `architectures/hyde_rag.py`, `architectures/structured_rag.py`
None of the `reset()` methods hold `services._chroma_lock` around delete+recreate. A concurrent ingest or query can receive a handle to a deleted collection.
**Fix:** Wrap the delete+recreate pair in `with services._chroma_lock:` in ALL `reset()` methods.

---

### WR-08 — `multilingual_embeddings` lazy-load is not thread-safe
**File:** `core/shared_services.py:123-128`
No lock around the `if self._multilingual_embeddings is None` check. Two concurrent first-accesses both load the 570 MB model.
**Fix:** Double-checked locking with a dedicated `threading.Lock()`.

---

### WR-09 — `rerank` cross-encoder lazy-load is not thread-safe (same pattern)
**File:** `core/shared_services.py:177-188`
**Fix:** Same double-checked locking pattern.

---

### WR-10 — Compare timeout produces silent partial result with no error to client
**File:** `backend/api.py:540-542`
`t.join(timeout=120)` does not detect timed-out threads. Timed-out archs silently disappear from results.
**Fix:** After join, mark timed-out threads as `"error": "timeout"` in `result_list`.

---

### WR-11 — `EvalScorecard` excludes genuine score of `0` from average
**File:** `frontend/components/EvalScorecard.tsx:23-25`
`.filter(v => v > 0)` silently excludes `0/10` from the average — a complete failure has no effect on the displayed score.
**Fix:** Change to `.filter((v): v is number => v != null)`.

---

### WR-12 — `handleDeleteDoc` can remove two docs sharing a filename from different dirs
**File:** `frontend/app/page.tsx:250-255`
The basename fallback check `label !== source` can match multiple docs if they share a filename.
**Fix:** Filter exclusively by `d.name !== source` (full path only).

---

### WR-13 — `ApiKeyModal` info text says key is memory-only; backend writes it to `.env`
**File:** `frontend/components/ApiKeyModal.tsx:100-103`
Text reads "stored in the server process memory only — restart the server and re-enter to change it." The key actually persists in `.env` across restarts.
**Fix:** Update text to reflect actual behavior.

---

## INFO

### IN-01 — `process_file` temp files leak on exception paths
**File:** `backend/api.py:124-148`
`os.remove(tmp)` is only in the happy path. PDF/DOCX processing errors leak temp files.
**Fix:** Use `try/finally` around `os.remove(tmp)`.

### IN-02 — Image `.b64` files in `adaptive_data/images/` never deleted
**File:** `backend/api.py:203-211`
No cleanup on document delete or reset. Directory grows unboundedly.
**Fix:** Delete matching `.b64` files in the `delete_document` handler.

### IN-03 — `_init_chroma_client` silently falls back to `EphemeralClient`
**File:** `core/shared_services.py:119-121`
Double startup failure makes all data ephemeral with no warning anywhere.
**Fix:** Set `self._ephemeral_fallback = True`; expose on `GET /api/health`.

### IN-04 — `find_similar_query` holds `_lock` across numpy CPU loop
**File:** `core/adaptive_db.py:86-126`
Blocks all DB operations (analytics writes, feedback) during cosine-sim computation.
**Fix:** Fetch rows under lock, release, then do numpy math outside.

### IN-05 — `rag_fusion` sub-query `lstrip` fails for numbered lists `10.` and above
**File:** `architectures/rag_fusion.py:54`
`lstrip("123456789.")` is character-set based. `"10. query"` → `"0. query"`.
**Fix:** `re.sub(r'^[\d\-•.]+\s*', '', q.strip())`

### IN-06 — `DocumentManager` uses array index as React key
**File:** `frontend/components/DocumentManager.tsx:151`
Delete causes spinner to appear on wrong row. **Fix:** Use `d.name` as key.

### IN-07 — `AnalyticsDashboard` uses array index as React key
**File:** `frontend/components/AnalyticsDashboard.tsx:134`
**Fix:** Use compound key `${r.arch_key}-${r.ts}-${i}`.

### IN-08 — `AnalyticsDashboard` renders raw float `avg_elapsed` without `.toFixed(2)`
**File:** `frontend/components/AnalyticsDashboard.tsx:100`

### IN-09 — SSE stream has no client-side timeout
**File:** `frontend/lib/api.ts:54`
Hung backend stalls UI in `isStreaming` forever. **Fix:** `setTimeout` at 120s to close + call `onError`.

### IN-10 — `isCached` heuristic in CompareGrid (`elapsed < 0.05`) has no backing data
**File:** `frontend/components/CompareGrid.tsx:80`
Fast archs get a false cached indicator. **Fix:** Remove heuristic or add explicit `cached` field from backend.

---

## Totals

| Session | Findings fixed |
|---|---|
| Sessions 1–6 | Initial builds |
| Sessions 7–8 | 47 findings |
| Session 9 | 48 findings |
| Session 10 | ~10 live bugs |
| Session 11 | UI + HNSW centralization |
| Session 12 | 14 findings |
| Session 13 | 13 findings |
| **Session 14 audit** | **35 new findings (pending fix)** |
| **All-time fixed** | **~132** |
