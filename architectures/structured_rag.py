import uuid
import json
import re
from typing import List
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class StructuredRAGPipeline:
    """
    Structured Data RAG — Text-to-Pandas + Vector fallback.
    CSV/Excel files are parsed into DataFrames stored in memory.
    At query time the LLM generates pandas code to answer structured queries.
    Falls back to vector search for narrative/unstructured questions.
    """

    def __init__(self):
        self.arch_key = "09 Structured RAG (CSV/Excel)"
        self.collection_name = "structured_rag_collection"
        self._table_store: dict = {}
        try:
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as e:
            print(f"[{self.collection_name}] init failed ({e}) — recreating")
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self._rebuild_table_store()

    def _rebuild_table_store(self):
        """Rebuild in-memory table store from persisted ChromaDB data after server restart."""
        try:
            if not self.collection.count():
                return
            result = self.collection.get(include=["documents", "metadatas"])
            for text, meta in zip(result["documents"] or [], result["metadatas"] or []):
                meta = meta or {}
                if meta.get("type") in ("csv", "excel"):
                    self._table_store[meta.get("source", "unknown")] = {
                        "columns": json.loads(meta.get("columns", "[]")),
                        "csv_text": text,
                    }
        except Exception as e:
            print(f"[structured_rag] table store rebuild failed: {e}")

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self._table_store = {}

    def ingest(self, documents: List[Document]):
        if not documents:
            return
        existing = self.collection.count()
        texts = [doc.page_content for doc in documents]
        ids = [f"struct_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        metadatas = [
            {k: v for k, v in doc.metadata.items()
             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 8192}
            for doc in documents
        ]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

        for doc in documents:
            if doc.metadata.get("type") in ("csv", "excel"):
                source = doc.metadata.get("source", "unknown")
                self._table_store[source] = {
                    "columns": json.loads(doc.metadata.get("columns", "[]")),
                    "csv_text": doc.page_content,
                }

    def _run_pandas_query(self, csv_text: str, columns: list, query: str) -> str | None:
        prompt = f"""You are a data analyst. Given a CSV table, write Python/pandas code to answer the query.

CSV Data:
{csv_text[:3000]}

Columns: {columns}

Query: {query}

Write ONLY a single Python expression using the variable `df` (a pandas DataFrame already loaded from the CSV).
Import nothing — pandas is available as `pd` and the dataframe is `df`.
Output ONLY the expression, no explanation.
Examples: df['sales'].sum()  |  df[df['region']=='North']['revenue'].mean()  |  df.groupby('category')['sales'].sum().to_dict()"""

        try:
            response = services.llm.invoke(prompt)
            code = services.extract_response_text(response).strip()
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            code = code.split("\n")[0].strip()  # single expression only

            import pandas as pd
            import io
            data_part = csv_text
            if "DATA:\n" in csv_text:
                data_part = csv_text.split("DATA:\n", 1)[1]
            df = pd.read_csv(io.StringIO(data_part))
            _FORBIDDEN = (
                "import", "exec", "eval", "__", "open(", "os.", "sys.",
                "subprocess", "shutil", "globals", "locals", "getattr",
                "setattr", "delattr", "compile", "input", "print(",
                "breakpoint", "vars(", "dir(",
            )
            if any(p in code for p in _FORBIDDEN):
                raise ValueError("Unsafe pattern detected in generated pandas expression")
            if "__" in code:
                raise ValueError("Dunder access forbidden in pandas expression")
            result = eval(code, {"__builtins__": None}, {"df": df, "pd": pd})
            return f"**Pandas result:** `{code}`\n\nResult:\n```\n{str(result)[:1200]}\n```"
        except Exception as e:
            print(f"[structured_rag] pandas query failed: {e}")
            return None

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "Please ingest a CSV, Excel, or text document first!"

        structured_result = None
        if self._table_store:
            step("Attempting structured query (Text-to-Pandas)…")
            all_results = []
            for source, table in self._table_store.items():
                result = self._run_pandas_query(table["csv_text"], table["columns"], query)
                if result:
                    label = source.split("/")[-1].split("\\")[-1]
                    all_results.append(f"**{label}:**\n{result}")
            if all_results:
                structured_result = "\n\n".join(all_results)
                step("✅ Pandas query succeeded across all tables")
            if not structured_result:
                step("⚠️ Pandas query failed — falling back to vector search…")

        step("Vector retrieval from ChromaDB…")
        query_embedding = services.embeddings.embed_query(query)
        n = min(5, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)

        docs, metas = adaptive_db.apply_feedback_boost(docs, metas, self.arch_key)

        if on_step and docs:
            sources = [
                {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
                for text, meta in zip(docs, metas)
            ]
            on_step(("sources", sources))

        doc_context = services.build_sourced_context(docs, metas)

        prompt = f"""You are a data analyst assistant. Answer the query using the structured results and document context below.

{("Structured Pandas Result:\n" + structured_result + "\n\n") if structured_result else ""}Document Context:
{doc_context}

Query: {query}

Provide a clear, data-driven answer. If structured results are available, reference the exact numbers."""

        step("Generating with Llama 4 Scout…")
        return services.stream_llm(
            prompt, on_token=lambda t: on_step and on_step(("token", t))
        )
