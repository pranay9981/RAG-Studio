import os
import uuid
from typing import List, Dict, Any
from langchain_core.documents import Document
from core.shared_services import services

class MultilingualRAGPipeline:
    def __init__(self):
        self.collection_name = "multilingual_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def ingest(self, documents: List[Document]):
        """Ingests chunks into ChromaDB using multilingual embeddings."""
        if not documents:
            return

        # Clear stale data from any previous ingest
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

        texts = [doc.page_content for doc in documents]
        ids = [f"multi_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]

        embeddings = services.multilingual_embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
        )

    def query(self, query: str, on_step=None) -> str:
        """Retrieves using multilingual search and generates an answer."""
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "No documents ingested yet."

        step("Embedding query with multilingual model…")
        query_embedding = services.multilingual_embeddings.embed_query(query)

        step("Cross-lingual retrieval from ChromaDB…")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )

        docs = results['documents'][0]
        context = "\n\n".join(docs)

        prompt = f"""You are a helpful Multilingual Assistant.
        Answer the user's query in the same language as the query, using ONLY the following context.
        If the context does not contain the answer, say "I cannot answer this based on the provided documents."

        Context:
        {context}

        Query: {query}
        Answer:"""

        step("Generating answer in query language…")
        return services.stream_llm(prompt, on_token=lambda t: on_step and on_step(("token", t)))
