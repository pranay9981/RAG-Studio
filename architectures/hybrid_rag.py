import uuid
from typing import List, Dict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from core.shared_services import services

class HybridRAGPipeline:
    def __init__(self):
        self.collection_name = "hybrid_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.bm25 = None
        self.chunks: List[Document] = []
        self.chunk_ids: List[str] = []

    def reset(self):
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

        # Clear stale data from any previous ingest
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

        self.chunks = documents
        self.chunk_ids = [f"hybrid_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = services.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=self.chunk_ids,
        )

        # Build BM25 sparse index over the same set of chunks
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
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
            if on_step: on_step(("step", msg))

        if not self.bm25 or not self.chunks:
            return "Please ingest a document first!"

        step("Embedding query with sentence-transformers…")
        query_embedding = services.embeddings.embed_query(query)

        step(f"Dense retrieval from ChromaDB (top {top_k})…")
        n = min(top_k, self.collection.count())
        dense_response = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        dense_results = [
            {"id": dense_response["ids"][0][i], "text": dense_response["documents"][0][i]}
            for i in range(len(dense_response["ids"][0]))
        ]

        step(f"Sparse retrieval via BM25 (top {top_k})…")
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = bm25_scores.argsort()[-top_k:][::-1]
        sparse_results = [
            {"id": self.chunk_ids[idx], "text": self.chunks[idx].page_content}
            for idx in top_sparse_indices
        ]

        step("Applying Reciprocal Rank Fusion (RRF k=60)…")
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        context = "\n\n".join(doc["text"] for doc in fused[:top_k])

        step("Generating answer with Gemini…")
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.

Context:
{context}

Query: {query}

Answer:"""
        return services.stream_llm(prompt, on_token=lambda t: on_step and on_step(("token", t)))
