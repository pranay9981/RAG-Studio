import uuid
from typing import List
from langchain_core.documents import Document
from core.shared_services import services


class HyDERAGPipeline:
    """
    Hypothetical Document Embeddings (HyDE) RAG.
    Generates a hypothetical answer first, embeds it, then retrieves real
    documents whose embeddings are closest to that hypothetical — bridging
    the vocabulary gap between short queries and long documents.
    """

    def __init__(self):
        self.collection_name = "hyde_rag_collection"
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
        metadatas = [doc.metadata for doc in documents]
        ids = [f"hyde_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def _generate_hypothetical_document(self, query: str) -> str:
        prompt = f"""Write a detailed, factual passage that would perfectly answer the following question.
Write it as if it were an excerpt from a relevant document, article, or knowledge base entry.
Be specific, informative, and use terminology that would appear in real source material.
Output ONLY the passage — no preamble, no labels, no explanation.

Question: {query}

Passage:"""
        response = services.llm.invoke(prompt)
        return services.extract_response_text(response)

    def query(self, query: str, top_k: int = 5, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "Please ingest a document first!"

        step("Generating hypothetical answer document with Gemini…")
        hypothetical_doc = self._generate_hypothetical_document(query)

        step("Embedding hypothetical document into vector space…")
        hyde_embedding = services.embeddings.embed_query(hypothetical_doc)

        step("Retrieving real documents closest to hypothetical embedding…")
        n = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[hyde_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)

        if on_step and docs:
            sources = [
                {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
                for text, meta in zip(docs, metas)
            ]
            on_step(("sources", sources))

        context = "\n\n".join(docs)

        step("Generating final answer from retrieved real context…")
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.

Context:
{context}

Query: {query}

Answer:"""
        return services.stream_llm(prompt, on_token=lambda t: on_step and on_step(("token", t)))
