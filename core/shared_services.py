import os
import time
import shutil
from typing import List, Any, Optional
import threading
import chromadb
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()


class SharedServices:
    def __init__(self):
        self._llm = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._multilingual_embeddings = None  # lazy-loaded on first use (BAAI/bge-m3)
        self._multilingual_lock = threading.Lock()
        self._cross_encoder_lock = threading.Lock()
        self._ephemeral_fallback = False
        self.chroma_client = self._init_chroma_client()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        )
        # Try to init LLM eagerly if key already present
        if os.environ.get("GROQ_API_KEY", "").strip():
            self._init_llm()
        self._chroma_lock = threading.Lock()

    def _init_llm(self):
        self._llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_tokens=1024,
        )

    @property
    def llm(self):
        if self._llm is None:
            key = os.environ.get("GROQ_API_KEY", "").strip()
            if not key:
                raise RuntimeError("GROQ_API_KEY is not set. Configure it via the API key settings in the UI.")
            self._init_llm()
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    def chroma_query(self, collection, collection_name: str, **kwargs):
        """Thread-safe ChromaDB query with HNSW-error retry, collection refresh, and rebuild.

        Returns (results, collection) — caller should update self.collection with
        the returned collection in case it was refreshed or rebuilt.
        """
        for attempt in range(3):
            try:
                with self._chroma_lock:
                    return collection.query(**kwargs), collection
            except Exception as e:
                err = str(e).lower()
                is_hnsw = "hnsw" in err or "nothing found on disk" in err
                if is_hnsw and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    with self._chroma_lock:
                        collection = self.chroma_client.get_or_create_collection(collection_name)
                    continue
                if is_hnsw:
                    # All handle-refresh retries exhausted — rebuild from SQLite docs + re-embed.
                    # Embeddings are stored inside HNSW segment files, so include=["embeddings"]
                    # also fails when HNSW is broken. Read only documents/metadata (SQLite-only),
                    # re-embed locally, then delete-and-recreate to build a fresh HNSW index.
                    try:
                        with self._chroma_lock:
                            existing = collection.get(include=["documents", "metadatas"])
                        if existing and existing.get("ids"):
                            print(f"[chroma] HNSW unrecoverable for {collection_name}, rebuilding {len(existing['ids'])} docs...")
                            emb_fn = (
                                self.multilingual_embeddings
                                if "multilingual" in collection_name
                                else self.embeddings
                            )
                            new_embeddings = emb_fn.embed_documents(existing["documents"])
                            with self._chroma_lock:
                                # Re-read under lock to capture any concurrent adds before deleting
                                existing2 = collection.get(include=["documents", "metadatas"])
                                self.chroma_client.delete_collection(collection_name)
                                new_col = self.chroma_client.get_or_create_collection(collection_name)
                                new_col.add(
                                    ids=existing2["ids"],
                                    documents=existing2["documents"],
                                    embeddings=emb_fn.embed_documents(existing2["documents"]),
                                    metadatas=existing2["metadatas"],
                                )
                            with self._chroma_lock:
                                return new_col.query(**kwargs), new_col
                    except Exception as rebuild_err:
                        print(f"[chroma] Rebuild failed for {collection_name}: {rebuild_err}")
                raise
        raise RuntimeError("ChromaDB HNSW error after 3 attempts")

    def _init_chroma_client(self):
        """PersistentClient with safe migration — wipes directory on HNSW corruption and retries."""
        chroma_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        )
        for attempt in range(2):
            try:
                client = chromadb.PersistentClient(path=chroma_path)
                client.list_collections()  # smoke-test to catch stale segment files
                return client
            except Exception as e:
                if attempt == 0:
                    print(f"[chroma] PersistentClient failed ({e}) — resetting directory")
                    shutil.rmtree(chroma_path, ignore_errors=True)
                else:
                    print(f"[chroma] retry failed ({e}) — falling back to EphemeralClient")
                    self._ephemeral_fallback = True
                    return chromadb.EphemeralClient()
        self._ephemeral_fallback = True
        return chromadb.EphemeralClient()

    @property
    def multilingual_embeddings(self):
        if self._multilingual_embeddings is None:
            with self._multilingual_lock:
                if self._multilingual_embeddings is None:
                    print("[shared_services] Loading BAAI/bge-m3 multilingual embeddings (first use)...")
                    self._multilingual_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        return self._multilingual_embeddings

    # ── Document loading ──────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> List[Document]:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        chunks = self.text_splitter.split_text(text)
        return [
            Document(page_content=c, metadata={"source": file_path, "type": "pdf"})
            for c in chunks
        ]

    # ── Parent-child / windowed chunking ──────────────────────────────────────

    def create_parent_child_documents(self, documents: List[Document]) -> List[Document]:
        """Creates small child chunks (300 chars) for retrieval with the full parent (1000 chars) in metadata.
        Children are embedded for precise matching; parent text is used at generation time."""
        result = []
        for doc in documents:
            children = self.child_text_splitter.split_text(doc.page_content)
            parent_text = doc.page_content[:4000]
            for child_text in children:
                new_meta = {**doc.metadata, "parent_text": parent_text}
                result.append(Document(page_content=child_text, metadata=new_meta))
        return result

    def get_context_text(self, doc_text: str, meta: Optional[dict]) -> str:
        """Returns the richest available context: parent_text > window_text > raw chunk."""
        if meta and meta.get("parent_text"):
            return meta["parent_text"]
        if meta and meta.get("window_text"):
            return meta["window_text"]
        return doc_text

    def build_sourced_context(self, texts: list, metas: list) -> str:
        """Builds context string with [Source: filename] labels so the LLM can compare documents."""
        parts = []
        for text, meta in zip(texts, metas):
            source = (meta or {}).get("source", "Unknown")
            label = source.split("/")[-1].split("\\")[-1]
            content = self.get_context_text(text, meta)
            parts.append(f"[Source: {label}]\n{content}")
        return "\n\n---\n\n".join(parts)

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def rerank(self, query: str, texts: list, top_n: int = 5) -> list:
        if not hasattr(self, "_cross_encoder") or self._cross_encoder is None:
            with self._cross_encoder_lock:
                if not hasattr(self, "_cross_encoder") or self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder
                    self._cross_encoder = CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
                    )
        pairs = [[query, t] for t in texts]
        import numpy as np
        scores = self._cross_encoder.predict(pairs)
        scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        scored = sorted(zip(scores, texts), key=lambda x: x[0], reverse=True)
        return scored[:top_n]

    def stream_llm(self, prompt: str, on_token=None) -> str:
        full_text = ""
        for chunk in self.llm.stream(prompt):
            token = self.extract_response_text(chunk)
            if token:
                if on_token:
                    on_token(token)
                full_text += token
        return full_text

    def extract_response_text(self, response: Any) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            return " ".join(text_parts) if text_parts else ""
        return str(content)

    # ── Context quality evaluation ────────────────────────────────────────────

    def evaluate_context(self, query: str, docs: List[str]) -> str:
        """Judges whether retrieved docs are sufficient to answer the query.
        Returns 'CORRECT', 'AMBIGUOUS', or 'INCORRECT'."""
        if not docs or not any(d.strip() for d in docs):
            return "INCORRECT"
        context = "\n".join(docs[:3])[:2000]
        prompt = f"""Does the following context contain sufficient information to answer this query?
Context: {context}
Query: {query}
Output exactly one word: CORRECT, AMBIGUOUS, or INCORRECT"""
        try:
            response = self.llm.invoke(prompt)
            text = self.extract_response_text(response).strip().upper()
            if "CORRECT" in text and "INCORRECT" not in text:
                return "CORRECT"
            elif "AMBIGUOUS" in text:
                return "AMBIGUOUS"
            return "INCORRECT"
        except Exception as e:
            print(f"[evaluate_context] LLM call failed: {e}")
            return "AMBIGUOUS"

    # ── Web search fallback ───────────────────────────────────────────────────

    def web_search_fallback(self, query: str, n: int = 3) -> List[str]:
        """DuckDuckGo web search fallback. Returns list of text snippets."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))
                return [f"{r['title']}: {r['body']}" for r in results]
        except Exception as e:
            print(f"[web_search] DuckDuckGo fallback failed: {e}")
            return []


services = SharedServices()
