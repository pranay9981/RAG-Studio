import uuid
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from core.shared_services import services


class MultimodalRAGPipeline:
    def __init__(self):
        self.collection_name = "multimodal_rag_collection"
        try:
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

        existing = self.collection.count()
        texts = [doc.page_content for doc in documents]
        ids = [f"multi_mod_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        metadatas = [
            {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            for doc in documents
        ]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "No documents ingested yet."

        step("Embedding query…")
        query_embedding = services.embeddings.embed_query(query)

        step("Retrieving documents and images from ChromaDB…")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"],
        )

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        if on_step and docs:
            sources = [
                {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
                for text, meta in zip(docs, metadatas)
            ]
            on_step(("sources", sources))

        step("Building multimodal message (text + images)…")
        content = [
            {
                "type": "text",
                "text": f"Answer the user query using the following retrieved context. When comparing documents use [Source:] labels. Query: {query}\n\nContext:",
            }
        ]

        for idx, (doc_text, meta) in enumerate(zip(docs, metadatas)):
            # Use window_text for richer context when available
            ctx_text = services.get_context_text(doc_text, meta)
            content.append({"type": "text", "text": f"--- Document {idx + 1} ---\n{ctx_text}\n"})
            if meta and "image_base64" in meta:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{meta['image_base64']}"},
                })

        message = HumanMessage(content=content)

        step("Generating answer with Gemini Vision…")
        full_text = ""
        for chunk in services.llm.stream([message]):
            token = services.extract_response_text(chunk)
            if token:
                if on_step:
                    on_step(("token", token))
                full_text += token
        return full_text
