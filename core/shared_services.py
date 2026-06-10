import os
from typing import List, Any, Optional
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()


class SharedServices:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.2,
            max_tokens=1024,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.multilingual_embeddings = self.embeddings
        self.chroma_client = chromadb.EphemeralClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        )

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

    def create_windowed_documents(
        self, documents: List[Document], window: int = 1
    ) -> List[Document]:
        """Adds `window_text` metadata to each doc: the chunk plus its ±window neighbours.
        This gives the LLM wider context without polluting retrieval embeddings."""
        result = []
        for i, doc in enumerate(documents):
            start = max(0, i - window)
            end = min(len(documents), i + window + 1)
            window_text = " ".join(
                documents[j].page_content for j in range(start, end)
            )
            new_meta = {**doc.metadata, "window_text": window_text[:4000]}
            result.append(Document(page_content=doc.page_content, metadata=new_meta))
        return result

    def get_context_text(self, doc_text: str, meta: Optional[dict]) -> str:
        """Returns window_text from metadata when available, otherwise the raw chunk."""
        if meta and "window_text" in meta:
            return meta["window_text"]
        return doc_text

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def rerank(self, query: str, texts: list, top_n: int = 5) -> list:
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
        except Exception:
            return "AMBIGUOUS"

    # ── Web search fallback ───────────────────────────────────────────────────

    def web_search_fallback(self, query: str, n: int = 3) -> List[str]:
        """DuckDuckGo web search fallback. Returns list of text snippets."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))
                return [f"{r['title']}: {r['body']}" for r in results]
        except Exception:
            return []


services = SharedServices()
