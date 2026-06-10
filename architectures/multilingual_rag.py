import uuid
from typing import List
from langchain_core.documents import Document
from core.shared_services import services


class MultilingualRAGPipeline:
    def __init__(self):
        self.collection_name = "multilingual_rag_collection"
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

        existing = self.collection.count()
        texts = [doc.page_content for doc in documents]
        ids = [f"multi_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        embeddings = services.multilingual_embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "No documents ingested yet."

        step("Embedding query with multilingual model…")
        query_embedding = services.multilingual_embeddings.embed_query(query)

        step("Cross-lingual retrieval from ChromaDB…")
        n = min(8, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)

        step("Re-ranking with cross-encoder…")
        scored = services.rerank(query, docs, top_n=4)
        reranked_texts = [text for _, text in scored]

        src_map = {}
        for doc, meta in zip(docs, metas):
            src_map[doc] = (meta or {}).get("source", "Unknown")

        if on_step:
            sources = [
                {"text": text[:300], "source": src_map.get(text, "Unknown"), "score": round(score, 3)}
                for score, text in scored
            ]
            on_step(("sources", sources))

        context = "\n\n".join(reranked_texts)

        prompt = f"""You are a helpful Multilingual Assistant.
Answer the user's query in the same language as the query, using ONLY the following context.
If the context does not contain the answer, say "I cannot answer this based on the provided documents."

Context:
{context}

Query: {query}
Answer:"""

        step("Generating answer in query language…")
        return services.stream_llm(prompt, on_token=lambda t: on_step and on_step(("token", t)))
