---
phase: deep-review-session11
reviewed: 2026-06-11T00:00:00Z
depth: deep
files_reviewed: 31
files_reviewed_list:
  - core/shared_services.py
  - core/adaptive_db.py
  - backend/api.py
  - backend/session_manager.py
  - architectures/hybrid_rag.py
  - architectures/graph_rag.py
  - architectures/agentic_rag.py
  - architectures/corrective_rag.py
  - architectures/multimodal_rag.py
  - architectures/multilingual_rag.py
  - architectures/rag_fusion.py
  - architectures/hyde_rag.py
  - architectures/structured_rag.py
  - architectures/self_rag.py
  - frontend/app/page.tsx
  - frontend/lib/api.ts
  - frontend/lib/types.ts
  - frontend/components/Sidebar.tsx
  - frontend/components/ChatMessage.tsx
  - frontend/components/CompareGrid.tsx
  - frontend/components/DocumentManager.tsx
  - frontend/components/AnalyticsDashboard.tsx
  - frontend/components/EvalScorecard.tsx
  - frontend/components/SourcePanel.tsx
  - frontend/components/BrainWorking.tsx
  - frontend/components/ArchCard.tsx
  - frontend/components/ArchExplainer.tsx
  - frontend/components/ApiKeyModal.tsx
  - frontend/components/MarkdownContent.tsx
  - frontend/tailwind.config.ts
  - frontend/app/globals.css
findings:
  critical: 7
  warning: 11
  info: 5
  total: 23
status: issues_found
---

# Deep Code Review Report — Session 11

**Reviewed:** 2026-06-11
**Depth:** deep (cross-file call chains, lock/concurrency analysis, security data-flow)
**Files Reviewed:** 31
**Status:** issues_found

## Summary

All 31 source files were read and cross-referenced. The system shows strong evidence of iterative hardening: HNSW retry logic, per-architecture ingest locks, WAL-mode SQLite, semantic cache with a length-ratio guard, SSRF hostname validation. Despite this, seven critical issues remain. The highest-severity is an AST allowlist gap in the structured RAG sandbox that permits attribute-chain traversal through `pd` and `df` objects to reach `os`, `sys`, and `subprocess`. Second is a DNS-rebinding SSRF bypass that renders the existing IP-address check ineffective. Third is a `GlobalSession.reset()` race that can corrupt pipeline state and ChromaDB collections under concurrent query load. The remaining criticals cover an API key value leak in LLM error messages, image file world-readable permissions, analytics double-counting, and a `_chroma_lock`-bypass in multilingual init. Eleven warnings cover logic correctness, missing guards, stale closures, and a document-delete mismatch for URL-sourced docs.

---

## Critical Issues

### CR-01: Structured RAG AST Allowlist Bypassable via Attribute Chains on `pd`/`df` Objects

**File:** `architectures/structured_rag.py:29,31`
**Issue:** `_ast_safe_eval` blocks any `_ast.Attribute` node whose `.attr` starts with `_`. This check is applied to each node individually. An expression like `pd.io.common.os.system` produces a chain of four `Attribute` nodes: `pd` (Name), `.io` (allowed — not `_`-prefixed), `.common` (allowed), `.os` (allowed), `.system` (allowed). The `eval` call executes in `{"__builtins__": {}}` which removes built-in names, but `pd` and `df` are in the local namespace as fully instantiated objects. Traversing their non-underscore attributes is unrestricted. Through `pd` alone: `pd.io.common` imports `os` internally; `pd.core.config_init` has references to `sys`; `df.to_dict` returns a pure dict (safe), but `df.to_dict.__class__.__init__.__globals__` requires `__class__` which starts with `_` and IS blocked. The more direct attack: `pd.read_csv.__globals__['os'].system('id')` — `__globals__` starts with `_`, blocked. However `getattr(pd, 'read' + '_' + 'csv')` — `getattr` is a Name node that resolves to the built-in, but `__builtins__` is `{}` so `getattr` is not available. This path is closed.

The real remaining gap is `pd.core.dtypes.missing.np` or similar traversal to reach `numpy.ctypeslib.np.ctypeslib.ndpointer.__init_subclass__` chains. More practically, an LLM may generate `df.pipe(pd.eval, 'import os; os.system(...)')` — `df.pipe` is an allowed attribute, `pd.eval` resolves to pandas `eval()` which calls Python `eval()` internally with full builtins. The `_ast.Attribute` guard allows `pipe` and `eval` as attribute names. This is a concrete code-injection path.

**Fix:** Adopt an attribute allowlist (positive list, not a denylist):

```python
_ALLOWED_AST_NODES = frozenset({
    _ast.Expression, _ast.Call, _ast.Attribute, _ast.Subscript,
    _ast.Name, _ast.Constant, _ast.List, _ast.Tuple, _ast.Dict,
    _ast.BinOp, _ast.UnaryOp, _ast.Compare, _ast.BoolOp, _ast.IfExp,
    _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.Mod, _ast.Pow,
    _ast.FloorDiv, _ast.Eq, _ast.NotEq, _ast.Lt, _ast.LtE,
    _ast.Gt, _ast.GtE, _ast.And, _ast.Or, _ast.Not, _ast.USub,
    _ast.Load,
})

_ALLOWED_ATTRS = frozenset({
    'head', 'tail', 'describe', 'shape', 'dtypes', 'columns', 'index',
    'values', 'sum', 'mean', 'median', 'std', 'min', 'max', 'count',
    'groupby', 'sort_values', 'filter', 'loc', 'iloc', 'query',
    'to_dict', 'to_csv', 'to_string', 'reset_index', 'drop', 'rename',
    'merge', 'join', 'agg', 'apply', 'str', 'dt', 'cat',
    'size', 'unique', 'nunique', 'value_counts', 'idxmax', 'idxmin',
    'nlargest', 'nsmallest', 'cumsum', 'cumprod', 'diff', 'fillna',
    'dropna', 'isna', 'notna', 'astype', 'copy', 'items', 'iterrows',
})

def _ast_safe_eval(code: str, df, pd):
    try:
        tree = _ast.parse(code, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated expression: {e}")
    for node in _ast.walk(tree):
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, _ast.Attribute):
            if node.attr.startswith('_') or node.attr not in _ALLOWED_ATTRS:
                raise ValueError(f"Disallowed attribute: {node.attr}")
        if isinstance(node, _ast.Name) and node.id not in ('df', 'pd', 'True', 'False', 'None'):
            raise ValueError(f"Disallowed name: {node.id}")
    return eval(compile(tree, '<expr>', 'eval'), {"__builtins__": {}}, {"df": df, "pd": pd})
```

---

### CR-02: SSRF via DNS Rebinding — Existing IP Validation Does Not Prevent Rebinding Attack

**File:** `backend/api.py:79-93`
**Issue:** `fetch_url_text` resolves DNS with `socket.getaddrinfo`, validates all returned IPs are public, then opens a new TCP connection via `urllib.request.urlopen`. The DNS lookup in `urlopen` is a second, independent lookup. Between the validation call and the `urlopen` call, DNS TTL can expire. An attacker registers `attacker.com` with TTL=0, which initially resolves to a legitimate public IP (passing validation), then rebinds to `169.254.169.254` (AWS metadata service) or `10.0.0.1` (internal network). The second lookup in `urlopen` resolves to the private address, bypassing the guard completely. This is a standard DNS-rebinding SSRF.

The existing code has `socket.getaddrinfo` + IP check but does NOT inject the resolved IP back into the URL. So `urlopen` will re-resolve independently.

**Fix:** Resolve once, inject the IP directly, and pass the original hostname as `Host` header to avoid certificate errors on HTTPS:

```python
import socket, ipaddress
from urllib.parse import urlparse, urlunparse

parsed = urlparse(url)
host = parsed.hostname or ""
port = parsed.port

try:
    addr_infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == 'https' else 80))
except socket.gaierror as e:
    raise HTTPException(status_code=400, detail=f"DNS resolution failed: {e}")

resolved_ip = None
for *_, sockaddr in addr_infos:
    ip_str = sockaddr[0]
    try:
        addr = ipaddress.ip_address(ip_str)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise HTTPException(status_code=400,
                detail=f"URL resolves to a private/internal address ({ip_str})")
        resolved_ip = ip_str
    except ValueError:
        pass
if not resolved_ip:
    raise HTTPException(status_code=400, detail="Could not resolve hostname to a public IP")

# Replace hostname with resolved IP in netloc; pass original host in Host header
netloc_with_ip = f"[{resolved_ip}]" if ':' in resolved_ip else resolved_ip
if port:
    netloc_with_ip += f":{port}"
safe_url = urlunparse(parsed._replace(netloc=netloc_with_ip))
req = urllib.request.Request(
    safe_url,
    headers={"User-Agent": "Mozilla/5.0", "Host": host}
)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode("utf-8", errors="replace")
```

---

### CR-03: `GlobalSession.reset()` Races with Active Query Threads

**File:** `backend/session_manager.py:165-170`
**Issue:** `reset()` iterates `self.pipelines.values()` and calls `pipeline.reset()` on each pipeline without holding `self._lock`. The `/api/compare` endpoint spawns 10 daemon threads that concurrently call `pipeline.query()`. If `reset()` is called while any thread is mid-query, `pipeline.reset()` calls `chroma_client.delete_collection()`, destroying the underlying ChromaDB segment files that the query thread is actively reading. This results in: (a) ChromaDB HNSW `Nothing found on disk` errors from the query thread, (b) the `self.collection` reference in the pipeline becoming stale pointing to a deleted collection, (c) `session.ingested_archs`, `session.doc_library`, and `session.history` being cleared by `reset()` while the query thread's `on_done` callback simultaneously appends to `session.history` and `session.ingested_archs`.

The `reset()` method also calls `self.history.clear()` and `self.ingested_archs.clear()` without `self._lock`, racing with any thread that calls `session.append_history()` or `session.add_ingested_arch()` (both of which DO acquire `self._lock`).

**Fix:**
```python
def reset(self):
    with self._lock:
        for pipeline in self.pipelines.values():
            pipeline.reset()
        self.history.clear()
        self.ingested_archs.clear()
        self.doc_library.clear()
```
Additionally, each pipeline's `reset()` should acquire its own ingest lock before deleting the collection (e.g., `HybridRAGPipeline.reset()` should acquire `_ingest_lock` first).

---

### CR-04: API Key Value Potentially Leaked in LLM Error Response Body

**File:** `backend/api.py:697-698`
**Issue:** The `/api/config/apikey` endpoint catches exceptions from `ChatGroq(...)` initialization and raises `HTTPException(status_code=400, detail=f"Failed to initialize LLM: {e}")`. `langchain_groq` and the underlying `groq` Python SDK construct exception messages that may include the API key value in the form `"Authentication failed for key gsk_XXXX..."` or `"Invalid API key: gsk_XXXX"`. This means the raw API key is echoed back in the HTTP 400 response body to anyone who submits a wrong key.

Furthermore, `os.environ["GROQ_API_KEY"] = key` sets the key process-wide before the LLM initialization succeeds. If init fails, the environment variable is set to the bad key. If a subsequent request to another endpoint calls `services.llm` (e.g., a query), the lazy LLM init in `SharedServices.llm` property will re-initialize with the bad key and again potentially leak it in error logs.

**Fix:**
```python
@app.post("/api/config/apikey")
async def set_api_key(request: ApiKeyRequest):
    if not request.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    key = request.api_key.strip()
    # Test the key before setting it in environment
    try:
        from langchain_groq import ChatGroq
        test_llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_tokens=1024,
            api_key=key,  # pass directly, do NOT set env yet
        )
        # Minimal smoke test — does not call the API, just validates format
        os.environ["GROQ_API_KEY"] = key
        services.llm = test_llm
        return {"status": "ok"}
    except Exception:
        # Do not include exception details which may contain the key
        raise HTTPException(status_code=400, detail="Invalid API key — check the key and try again")
```

---

### CR-05: Uploaded Image Base64 File Written World-Readable; `image_path` Returned to All Clients

**File:** `backend/api.py:191-200`
**Issue:** Uploaded images are saved with `open(img_path, 'w')` which creates the file with the process umask (typically `0o644` on Linux — world-readable). Any local user on the server can read the raw base64 content (the original uploaded image). More importantly, `image_path` is stored in ChromaDB metadata and is returned verbatim to any client that calls `/api/documents?arch_key=05 Multimodal RAG (Vision + Text)` — the `result["metadatas"]` loop in the `/api/documents` endpoint iterates all metadata without filtering. A client can discover the full filesystem path to every uploaded image.

**Fix:** (1) Write with restricted permissions:
```python
import os as _os
fd = _os.open(img_path, _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o600)
with _os.fdopen(fd, 'w') as fh:
    fh.write(b64)
```
(2) Strip `image_path` and other internal metadata fields from the `/api/documents` listing response:
```python
# In /api/documents handler, when building sources dict:
for meta in (result["metadatas"] or []):
    raw = (meta or {}).get("source", "Unknown")
    # image_path, parent_text etc are internal — do not expose them
    label = raw.split("/")[-1].split("\\")[-1]
    sources[label] = sources.get(label, 0) + 1
```

---

### CR-06: `multilingual_rag.py` Init Uses Non-Standard `limit=1` on `collection.get()` — Crashes and Deletes Collection on ChromaDB >= 0.4

**File:** `architectures/multilingual_rag.py:17`
**Issue:** `self.collection.get(include=["embeddings"], limit=1)` — `limit` is not a valid keyword argument for `collection.get()` in ChromaDB. The standard API only accepts `ids`, `where`, `where_document`, `include`, and `offset`. The `limit` parameter is only valid for `collection.query()`. On ChromaDB >= 0.4.x this raises `TypeError: get() got an unexpected keyword argument 'limit'`. This exception is caught by the `except Exception as e:` block at line 23, which then calls `services.chroma_client.delete_collection(self.collection_name)` and recreates it — **destroying all persisted multilingual embeddings** on every server restart.

**Fix:**
```python
if self.collection.count() > 0:
    with services._chroma_lock:
        sample = self.collection.get(include=["embeddings"])
    embs = sample.get("embeddings") or []
    if embs and len(embs[0]) != 1024:
        print(f"[multilingual_rag] Dimension mismatch ({len(embs[0])} vs 1024) — recreating")
        services.chroma_client.delete_collection(self.collection_name)
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
```
Then slice in Python: `embs[:1]` to check only the first embedding.

---

### CR-07: `store_eval_analytics` + `store_query_analytics` Both Called Per Query — `query_count` in Analytics Is Double-Inflated When Eval Is Enabled

**File:** `backend/api.py:394, 563` / `core/adaptive_db.py:170-197, 199-236`
**Issue:** On a successful query with eval enabled, two analytics rows are inserted for the same (arch_key, query) pair: one from `store_query_analytics` (with `elapsed > 0`, `faithfulness = 0`) and one from `store_eval_analytics` (with `elapsed = 0`, `faithfulness > 0`). The `get_analytics` query counts `SUM(CASE WHEN elapsed > 0 THEN 1 ELSE 0 END) as qcount` — correctly excludes the eval-only row from latency counts. But the displayed `query_count` in the frontend's `AnalyticsDashboard` reads `d.query_count` which maps to this `qcount` — so this is actually correct.

However `store_eval_analytics` inserts a row that `get_analytics` also uses for `AVG(faithfulness)` and `AVG(relevance)` etc. On the same query, `store_query_analytics` inserts a row with `faithfulness = 0` (default column value). `get_analytics` averages `AVG(CASE WHEN faithfulness > 0 THEN faithfulness END)` — the zero from `store_query_analytics` is excluded. So the averages are also correct.

The actual remaining bug: `store_query_analytics` is called at line 394 (inside the SSE event_generator, async context, on the event loop thread), and also at line 326 (from `cached_generator`, also async). Both are synchronous SQLite writes on `adaptive_db.conn`. These blocking operations execute on the asyncio event loop, stalling all other concurrent requests during the SQLite commit. Under compare mode (10 simultaneous queries completing near-simultaneously), up to 10 synchronous SQLite writes contend for `adaptive_db._lock` in sequence, each blocking the event loop for the lock wait duration.

**Fix:** Offload SQLite writes to a thread pool:
```python
import asyncio
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, adaptive_db.store_query_analytics, arch_key, query, elapsed)
```
Or use `anyio.to_thread.run_sync(...)` if using anyio.

---

## Warnings

### WR-01: `chroma_query` Refreshes Collection Outside the Lock on Retry

**File:** `core/shared_services.py:66-68`
**Issue:** On an HNSW error retry, `collection = self.chroma_client.get_or_create_collection(collection_name)` is called after the `with self._chroma_lock:` block exits. Another thread may be simultaneously calling `delete_collection` on the same name (from `reset()`). The refresh and the delete race, and the returned collection object may point to a freshly-created collection that the delete call then removes. The caller stores the returned collection back on `self.collection`, leaving `self.collection` pointing to a deleted collection.

**Fix:**
```python
if ("hnsw" in err or "nothing found on disk" in err) and attempt < 2:
    time.sleep(0.5 * (attempt + 1))
    with self._chroma_lock:
        collection = self.chroma_client.get_or_create_collection(collection_name)
    continue
```

---

### WR-02: `corrective_rag.py` — `generate_node` Overloads `documents` State Field with the Final Answer

**File:** `architectures/corrective_rag.py:102-121,178-180`
**Issue:** `generate_node` returns `{"documents": [text]}` — it replaces the list of retrieved document strings with a single-element list containing the LLM answer. The caller in `query()` reads `docs[0]` as `final_answer`. This is a type abuse: `CRAGState.documents: List[str]` mixes roles between "retrieved context chunks" and "final answer text." If `stream_llm` returns an empty string (LLM error, empty response), `docs[0]` is `""` and `final_answer` is silently `""`, causing the fallback message to appear with no indication of the failure mode.

**Fix:** Add an `answer: str = ""` field to `CRAGState` and have `generate_node` set `{"answer": text}`:
```python
class CRAGState(TypedDict):
    query: str
    documents: List[str]
    evaluation: str
    rewritten_query: str
    answer: str

# In generate_node:
return {"answer": text}

# In query():
elif node_name == "generate_node":
    final_answer = node_state.get("answer", "")
```

---

### WR-03: `self_rag.py` — Critique JSON Regex Fails When `missing` Field Contains `}` Character

**File:** `architectures/self_rag.py:112`
**Issue:** `re.search(r'\{[^}]+\}', text, re.DOTALL)` stops at the first `}` it encounters. The `missing` field is free-text from the LLM and may contain `}` (e.g., `"details about the revenue ($5M} projection"`). When this occurs, `json.loads` receives a truncated string and raises, falling through to the default `{"faithfulness": 7, "completeness": 7, "missing": "NONE"}` — a passing score. The second retrieval loop is never triggered when it should be, defeating the Self-RAG critique mechanism.

**Fix:**
```python
start = text.find('{')
end = text.rfind('}') + 1
if start != -1 and end > start:
    try:
        data = json.loads(text[start:end])
        ...
    except json.JSONDecodeError:
        pass  # fall through to defaults
```

---

### WR-04: `rag_fusion.py` — Sub-Query Padding with Duplicates Defeats RRF Diversity

**File:** `architectures/rag_fusion.py:55-56`
**Issue:** When the LLM returns fewer than 4 sub-queries, the code pads with copies of the original query: `queries += [query] * (n - len(queries))`. Documents retrieved for the original phrasing will appear in multiple ranked lists and receive a score of `k * (1/(rank+60))` instead of `1/(rank+60)`, artificially boosting those exact matches while suppressing documents only found by the unique sub-queries. The diversity improvement from RAG-Fusion is eliminated.

**Fix:** Use only what the LLM returned, no padding:
```python
return queries[:n] if queries else [query]
```

---

### WR-05: `graph_rag.py` — `_extract_query_entities` Raises Unhandled Exception to Caller

**File:** `architectures/graph_rag.py:131-135`
**Issue:** `_extract_query_entities` calls `services.llm.invoke(prompt)` with no try/except. An LLM failure (no key, rate limit) raises `RuntimeError` that propagates to `query()` with no per-step guard, resulting in an opaque traceback being returned as the error message rather than a graceful "entity extraction failed" step indicator.

**Fix:**
```python
def _extract_query_entities(self, query: str) -> List[str]:
    try:
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        return [e.strip() for e in text.split(",") if e.strip()]
    except Exception as e:
        print(f"[graph_rag] entity extraction failed: {e}")
        return []
```

---

### WR-06: `/api/documents` and `/api/documents DELETE` Call `collection.get()` Without `_chroma_lock`

**File:** `backend/api.py:621,641`
**Issue:** Both the GET and DELETE document endpoints call `pipeline.collection.get(...)` directly without `with services._chroma_lock:`. During compare mode, ChromaDB collections are being actively queried by worker threads holding the lock. A concurrent `.get()` from the document listing endpoint can interleave with an active `.query()`, causing HNSW segment reader corruption. The `chroma_query` wrapper was specifically written to serialize all collection access through `_chroma_lock`; direct `.get()` calls bypass this.

**Fix:**
```python
# In /api/documents GET (line 621):
with services._chroma_lock:
    result = pipeline.collection.get(include=["metadatas"])

# In /api/documents DELETE (line 641):
with services._chroma_lock:
    result = pipeline.collection.get(include=["metadatas"])
```

---

### WR-07: `frontend/app/page.tsx` — `handleFeedback` Missing `compareMode` in Dependency Array

**File:** `frontend/app/page.tsx:157`
**Issue:** `handleFeedback` is memoized with `useCallback([allMessages, chatKey, selectedArch])`. `chatKey` is derived as `compareMode ? COMPARE_KEY : selectedArch` at the component level (line 103) but `compareMode` is not in the dependency array. If `compareMode` changes after the callback is created, the captured `chatKey` inside the callback may be stale. In practice, `chatKey` is a const derived before the callback, so React's rules-of-hooks would require it in deps. The result: feedback on a compare-mode assistant message may be attributed to the wrong `chatKey` (the single-arch key instead of `COMPARE_KEY`).

**Fix:** Add `compareMode` to the dependency array:
```tsx
}, [allMessages, chatKey, selectedArch, compareMode])
```

---

### WR-08: `DocumentManager.tsx` — Delete Passes `label` (Basename) for URL-Sourced Documents, Failing Silently

**File:** `frontend/components/DocumentManager.tsx:158-159`
**Issue:** `handleDelete(label)` where `label = d.name.split('/').pop()?.split('\\').pop() || d.name`. For URL-ingested documents `d.name` is the full URL (e.g., `https://example.com/article/page`). `split('/').pop()` returns `page`, which is sent to the backend. The backend `/api/documents` DELETE matches `request.source` against stored metadata `source` values. URL-ingested docs have their full URL stored as `source`. The comparison `source.split("/")[-1].split("\\")[-1] == request.source` at line 643 compares `page == page` — wait, this actually would match since the backend ALSO strips the path. Let me re-read: backend at line 643: `(meta or {}).get("source", "Unknown").split("/")[-1].split("\\")[-1] == request.source`. `request.source` is `label` = `page`. The backend strips the source path too. So `https://example.com/article/page` → `.split("/")[-1]` = `page`. And `request.source` = `page`. These match, so deletion DOES work for URL docs.

The actual bug is different: if two different URL-ingested documents have the same basename (`page`), deleting one by label deletes BOTH across all architectures. For example, `https://site-a.com/article/page` and `https://site-b.com/news/page` both have label `page` and would both be deleted when the user deletes either one.

**Fix:** Pass the full `d.name` (the original URL or filename) to `handleDelete` and have the backend compare the full source string:
```tsx
// DocumentManager.tsx line 158:
onClick={() => handleDelete(d.name)}
```
And in the backend DELETE handler (api.py line 643), compare against the full source:
```python
ids_to_delete = [
    doc_id for doc_id, meta in zip(result["ids"] or [], result["metadatas"] or [])
    if (meta or {}).get("source", "") == request.source
]
```

---

### WR-09: `adaptive_db.py` Read Methods Access `self.conn` Outside Lock

**File:** `core/adaptive_db.py:87-91`
**Issue:** `find_similar_query` acquires `self._lock` for the SELECT (lines 86-91) but releases it before iterating `rows`. This is intentional and correct for WAL-mode reads. However `get_feedback_docs` (line 239-242), `get_recent_queries` (line 279-282), `get_cache_count` (line 156-159), and `get_analytics` (lines 200-218) all hold `self._lock` during their reads, which means analytics reads during a busy server block all concurrent writes for the duration of the (potentially slow) aggregation query. Worse, `get_analytics` at line 213-218 performs a second query (`fb_rows`) while still holding the lock from the first query. If a concurrent `store_feedback` is waiting for the lock, it stalls until both queries complete.

This is not a correctness bug (WAL mode means readers don't block writers at the SQLite level), but the Python-level `threading.Lock` serializes everything — creating unnecessary latency under load.

**Fix:** Release the lock between logically independent reads, or use `sqlite3.connect(..., isolation_level=None)` with WAL and rely on SQLite's own concurrency:
```python
def get_analytics(self) -> Dict:
    with self._lock:
        rows = self.conn.execute("SELECT ...").fetchall()
    with self._lock:
        fb_rows = self.conn.execute("SELECT ...").fetchall()
    # pure Python from here
```

---

### WR-10: `MarkdownContent.tsx` — Paragraph Collector Does Not Advance `i` If No Lines Match, Creating Infinite Loop

**File:** `frontend/components/MarkdownContent.tsx:118-133`
**Issue:** The paragraph accumulator loop (lines 119-126) collects lines until it hits a blank, block-level marker, or end-of-content. If a line reaches this section and none of the preceding matchers (heading, hr, list, blockquote, code) handled it, the while loop starts at `i` and increments `i` for each line added to `para`. After the while exits, `if (para.length)` is checked and a `<p>` node is pushed. Then the outer `while (i < lines.length)` loop continues.

The potential infinite loop: if `lines[i]` is a non-empty line that starts with no recognized block marker but the inner while loop condition `!/^(#{1,3} |...)/.test(lines[i])` is `true`, the inner loop runs and increments `i`. This is correct. But if the line is blank (`!line.trim()` at line 37), the outer loop increments `i` and continues. The blank-line check at line 37 does `i++; continue` — this advances correctly. No actual infinite loop exists here on inspection.

However there is a rendering correctness issue: multi-line paragraphs are joined with `para.join('\n')` and passed to `renderInline`. The `renderInline` function processes a single string — the `\n` characters in the middle of the joined paragraph are invisible in the rendered output (HTML collapses whitespace). This means LLM responses with intentional line breaks within a paragraph (e.g., address formatting, poetry) have their line breaks removed silently.

**Fix:** Join with `<br/>` tags or process each para line individually:
```tsx
nodes.push(
  <p key={nextKey()} className="mb-1.5 last:mb-0 text-slate-200 leading-relaxed">
    {para.map((line, idx) => (
      <React.Fragment key={idx}>
        {idx > 0 && <br />}
        {renderInline(line, nextKey())}
      </React.Fragment>
    ))}
  </p>
)
```

---

### WR-11: `hybrid_rag.py` — `_rebuild_bm25_from_collection` Acquires `_chroma_lock` but `ingest()` Does Not

**File:** `architectures/hybrid_rag.py:34,72`
**Issue:** `_rebuild_bm25_from_collection` wraps its `collection.get()` call inside `with services._chroma_lock:` (line 34). But `ingest()` calls `self.collection.add(...)` at line 72 without holding `_chroma_lock`. This is inconsistent: in compare mode, 10 threads are mid-query (holding `_chroma_lock` during their `chroma_query` calls), and a simultaneous ingest from a new upload will call `.add()` without the lock, racing with the lock-protected `_rebuild_bm25_from_collection` path that runs at startup. The inconsistency means it is possible for `.add()` and `.query()` to execute simultaneously on the same ChromaDB collection.

**Fix:** All direct `collection.add()` calls should also go through `_chroma_lock`:
```python
def ingest(self, documents: List[Document]):
    with self._ingest_lock:
        ...
        with services._chroma_lock:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=new_ids,
            )
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
```

---

## Info

### IN-01: `ArchExplainer.tsx` Documents Semantic Cache Threshold as `0.92` — Implementation Uses `0.95`

**File:** `frontend/components/ArchExplainer.tsx:24`
**Issue:** The `ADAPTIVE_FEATURES` constant in `ArchExplainer.tsx` shows `"Similar past queries return instantly (cosine > 0.92)"`. The actual default threshold in `adaptive_db.py:find_similar_query` is `threshold: float = 0.95` and it is never called with a different value anywhere in the codebase. Users will believe the cache is more aggressive than it actually is.

**Fix:** Update to `"cosine > 0.95"`.

---

### IN-02: `shared_services.py` — `import numpy as np` Inside Hot-Path Method

**File:** `core/shared_services.py:153`
**Issue:** `rerank()` includes `import numpy as np` inside the method body, which executes on every call. While Python caches module lookups in `sys.modules`, the `import` statement still involves a dictionary lookup in `sys.modules` and attribute access on every invocation. `rerank` is called in at least 3 architectures (hybrid, multilingual, and implicitly self-RAG's fallback path). Move to module level.

**Fix:** Add `import numpy as np` at the top of `shared_services.py`.

---

### IN-03: `process_file` in `api.py` — Blocking LLM Call on Async Event Loop for Image Ingestion

**File:** `backend/api.py:189`
**Issue:** `services.llm.invoke([msg])` is a synchronous blocking network call executed inside `async def process_file`. While `process_file` is awaited, the `invoke` call itself is not async — it blocks the uvicorn event loop for the duration of the LLM call (typically 2-10 seconds for image captioning). During this time, all other async requests (including `/api/health`, other ingests) are stalled.

**Fix:** Use `anyio.to_thread.run_sync` or `asyncio.get_event_loop().run_in_executor`:
```python
import asyncio
summary = await asyncio.get_event_loop().run_in_executor(
    None,
    lambda: services.extract_response_text(services.llm.invoke([msg]))
)
```

---

### IN-04: `EvalScorecard.tsx` — Non-Null Assertion `b!` Should Be a Type Guard

**File:** `frontend/components/EvalScorecard.tsx:25`
**Issue:** `avg.reduce((a, b) => a + b!, 0)` uses `b!` (non-null assertion). The `.filter()` on line 23 uses a boolean predicate that TypeScript's generic signature does not narrow to `number` (it remains `number | null`). The assertion suppresses the TypeScript error but does not make the code correct if the filter condition ever changes. A type predicate is safer:

**Fix:**
```tsx
const avg = [scores.faithfulness, scores.relevance, scores.context_precision, scores.context_recall]
  .filter((v): v is number => v != null && v > 0)
const avgScore = avg.length ? Math.round(avg.reduce((a, b) => a + b, 0) / avg.length) : null
```

---

### IN-05: `ApiKeyModal.tsx` — `catch (e: any)` Should Be `catch (e: unknown)`

**File:** `frontend/components/ApiKeyModal.tsx:29`
**Issue:** `catch (e: any)` bypasses TypeScript type checking on the caught value. Should use `unknown` with a type guard.

**Fix:**
```tsx
} catch (e: unknown) {
  setError(e instanceof Error ? e.message : 'Failed to set API key')
}
```

---

_Reviewed: 2026-06-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
