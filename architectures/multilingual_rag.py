import uuid
from typing import List
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class MultilingualRAGPipeline:
    def __init__(self):
        self.arch_key = "06 Multilingual RAG (BGE-M3)"
        self.collection_name = "multilingual_rag_collection"
        try:
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
            # Detect stale collection created with MiniLM (384-dim) instead of BGE-M3 (1024-dim)
            if self.collection.count() > 0:
                with services._chroma_lock:
                    sample = self.collection.get(include=["embeddings"])
                embs = (sample.get("embeddings") or [])[:1]
                if embs and len(embs[0]) != 1024:
                    print(f"[multilingual_rag] Dimension mismatch ({len(embs[0])} vs 1024) — recreating collection")
                    services.chroma_client.delete_collection(self.collection_name)
                    self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as e:
            print(f"[{self.collection_name}] init failed ({e}) — recreating")
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def ingest(self, documents: List[Document]):
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        ids = [f"multi_{uuid.uuid4().hex}" for _ in range(len(documents))]
        metadatas = [
            {k: v for k, v in doc.metadata.items()
             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 8192}
            for doc in documents
        ]
        embeddings = services.multilingual_embeddings.embed_documents(texts)
        try:
            with services._chroma_lock:
                self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        except Exception as e:
            if "dimension" in str(e).lower():
                # Stale collection with wrong embedding dim — recreate and retry
                print(f"[multilingual_rag] Dimension mismatch on add — recreating collection and retrying")
                with services._chroma_lock:
                    services.chroma_client.delete_collection(self.collection_name)
                    self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
                with services._chroma_lock:
                    self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
            else:
                raise

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "No documents ingested for Multilingual RAG. Please re-ingest your document (requires BGE-M3 model — ensure ~6 GB RAM is free)."

        if services._multilingual_embeddings is None:
            step("⏳ BGE-M3 model still loading in background — please wait a moment…")
        step("Embedding query with multilingual model…")
        try:
            query_embedding = services.multilingual_embeddings.embed_query(query)
        except Exception as e:
            return f"Multilingual RAG unavailable: BGE-M3 model failed to load ({e}). Free ~6 GB RAM and restart the backend, then re-ingest."

        step("Cross-lingual retrieval from ChromaDB…")
        n = min(8, self.collection.count())
        results, self.collection = services.chroma_query(
            self.collection, self.collection_name,
            query_embeddings=[query_embedding], n_results=n, include=["documents", "metadatas"],
        )
        docs = results["documents"][0] if results.get("documents") and results["documents"][0] else []
        metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)

        # Build text→meta map before reranking so positional alignment is preserved
        doc_meta_map = {text: meta for text, meta in zip(docs, metas)}

        step("Re-ranking with cross-encoder…")
        scored = services.rerank(query, docs, top_n=4)
        reranked_texts = [text for _, text in scored]

        # Self-evaluation
        quality = services.evaluate_context(query, reranked_texts)
        _icons = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}
        step(f"Context quality: {_icons.get(quality, '')} {quality}")

        if quality == "INCORRECT":
            step("Insufficient context — web search fallback…")
            web = services.web_search_fallback(query)
            if web:
                reranked_texts = web
                metas = [{}] * len(reranked_texts)
        elif quality == "AMBIGUOUS":
            web = services.web_search_fallback(query, n=2)
            if web:
                reranked_texts = reranked_texts + web
                metas = metas + [{}] * len(web)

        # Build unified lookup BEFORE boost (covers both ChromaDB + web results)
        combined_meta_map = {text: doc_meta_map.get(text, {}) for text in reranked_texts}

        # Feedback boost — surface positively-rated chunks, demote negatives
        dummy_metas = [{} for _ in reranked_texts]
        reranked_texts, _ = adaptive_db.apply_feedback_boost(reranked_texts, dummy_metas, self.arch_key)

        # Rebuild metas following boosted order using the unified map
        reranked_metas = [combined_meta_map.get(t, {}) for t in reranked_texts]

        if on_step:
            sources = [
                {"text": text[:300], "source": (combined_meta_map.get(text, {}) or {}).get("source", "Unknown")}
                for text in reranked_texts[:4]
            ]
            on_step(("sources", sources))

        context = services.build_sourced_context(reranked_texts, reranked_metas)

        prompt = f"""You are a helpful Multilingual Assistant.
Answer the user's query in the same language as the query, using ONLY the following context.
When the query asks to compare documents, use the [Source: ...] labels to distinguish between them.
If the context does not contain the answer, say "I cannot answer this based on the provided documents."

Context:
{context}

Query: {query}
Answer:"""

        step("Generating answer in query language…")
        return services.stream_llm(
            prompt, on_token=lambda t: on_step and on_step(("token", t))
        )
