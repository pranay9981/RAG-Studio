import uuid
import threading
from typing import List, Dict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class HybridRAGPipeline:
    def __init__(self):
        self.arch_key = "01 Hybrid RAG (Dense + Sparse)"
        self.collection_name = "hybrid_rag_collection"
        self._ingest_lock = threading.Lock()
        self.bm25 = None
        self.chunks: List[Document] = []
        self.chunk_ids: List[str] = []
        try:
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as e:
            print(f"[{self.collection_name}] init failed ({e}) — recreating")
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self._rebuild_bm25_from_collection()

    def _rebuild_bm25_from_collection(self):
        """Rebuild in-memory BM25 index from persisted ChromaDB data after server restart."""
        try:
            if not self.collection.count():
                return
            with services._chroma_lock:
                result = self.collection.get(include=["documents", "metadatas"])
            docs = result["documents"] or []
            metas = result["metadatas"] or [{}] * len(docs)
            ids = result["ids"] or []
            self.chunks = [Document(page_content=t, metadata=m or {}) for t, m in zip(docs, metas)]
            self.chunk_ids = list(ids)
            if self.chunks:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi([doc.page_content.lower().split() for doc in self.chunks])
        except Exception as e:
            print(f"[hybrid_rag] BM25 rebuild failed: {e}")

    def reset(self):
        with self._ingest_lock:
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
            self.bm25 = None
            self.chunks = []
            self.chunk_ids = []

    def ingest(self, documents: List[Document]):
        if not documents:
            return

        with self._ingest_lock:
            start = len(self.chunk_ids)
            new_ids = [f"hybrid_{uuid.uuid4().hex[:8]}_{start + i}" for i in range(len(documents))]

            self.chunks.extend(documents)
            self.chunk_ids.extend(new_ids)

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = services.embeddings.embed_documents(texts)

            with services._chroma_lock:
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=new_ids,
                )

            tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def _reciprocal_rank_fusion(
        self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60
    ) -> List[Dict]:
        fused_scores: Dict[str, Dict] = {}
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (rank + k)
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (rank + k)
        reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in reranked]

    def query(self, query: str, top_k: int = 5, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.bm25 or not self.chunks:
            return "Please ingest a document first!"

        step("Embedding query with sentence-transformers…")
        query_embedding = services.embeddings.embed_query(query)

        step(f"Dense retrieval from ChromaDB (top {top_k})…")
        n = min(top_k * 2, self.collection.count())
        dense_response, self.collection = services.chroma_query(
            self.collection, self.collection_name,
            query_embeddings=[query_embedding], n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        dense_results = [
            {
                "id": dense_response["ids"][0][i],
                "text": dense_response["documents"][0][i],
                "source": (dense_response["metadatas"][0][i] or {}).get("source", "Unknown"),
                "window_text": (dense_response["metadatas"][0][i] or {}).get("window_text"),
            }
            for i in range(len(dense_response["ids"][0]))
        ]

        step(f"Sparse retrieval via BM25 (top {top_k})…")
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = bm25_scores.argsort()[-top_k:][::-1]
        sparse_results = [
            {
                "id": self.chunk_ids[idx],
                "text": self.chunks[idx].page_content,
                "source": self.chunks[idx].metadata.get("source", "Unknown"),
                "window_text": self.chunks[idx].metadata.get("window_text"),
            }
            for idx in top_sparse_indices
        ]

        step("Applying Reciprocal Rank Fusion (RRF k=60)…")
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        candidates = fused[: top_k * 2]

        step("Re-ranking with cross-encoder…")
        scored = services.rerank(query, [c["text"] for c in candidates], top_n=top_k)

        # Self-evaluation on re-ranked results
        reranked_texts = [text for _, text in scored]

        # Feedback boost — surface positively-rated chunks, demote negatives
        dummy_metas = [{} for _ in reranked_texts]
        reranked_texts, _ = adaptive_db.apply_feedback_boost(reranked_texts, dummy_metas, self.arch_key)
        quality = services.evaluate_context(query, reranked_texts)
        _icons = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}
        step(f"Context quality: {_icons.get(quality, '')} {quality}")

        if quality == "INCORRECT":
            step("Insufficient context — web search fallback…")
            web = services.web_search_fallback(query)
            if web:
                reranked_texts = web
        elif quality == "AMBIGUOUS":
            web = services.web_search_fallback(query, n=2)
            if web:
                reranked_texts = reranked_texts + web

        window_map = {c["text"]: c.get("window_text") for c in candidates}
        if on_step:
            sources = [
                {
                    "text": text[:300],
                    "source": next(
                        (c.get("source", "Unknown") for c in candidates if c["text"] == text),
                        "Unknown",
                    ),
                    "score": round(score, 3),
                }
                for score, text in scored
            ]
            on_step(("sources", sources))

        meta_list = [
            {"source": next((c.get("source") for c in candidates if c["text"] == t), "Unknown"),
             "parent_text": window_map.get(t)}
            for t in reranked_texts
        ]
        context = services.build_sourced_context(reranked_texts, meta_list)

        step("Generating with Llama 4 Scout…")
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.
When the query asks to compare or contrast documents, use the [Source: ...] labels to distinguish between them.

Context:
{context}

Query: {query}

Answer:"""
        return services.stream_llm(
            prompt, on_token=lambda t: on_step and on_step(("token", t))
        )
