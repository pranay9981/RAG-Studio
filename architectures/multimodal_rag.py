import base64
import os
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from core.shared_services import services
from core.adaptive_db import adaptive_db


class MultimodalRAGPipeline:
    def __init__(self):
        self.arch_key = "05 Multimodal RAG (Vision + Text)"
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
        with services._chroma_lock:
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def ingest(self, documents: List[Document]):
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        ids = [f"multi_mod_{uuid.uuid4().hex}" for _ in range(len(documents))]
        metadatas = [
            {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            for doc in documents
        ]
        embeddings = services.embeddings.embed_documents(texts)
        with services._chroma_lock:
            self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        count = self.collection.count()
        if not count:
            return "No documents ingested yet."

        step("Embedding query…")
        query_embedding = services.embeddings.embed_query(query)

        step("Retrieving documents and images from ChromaDB…")
        results, self.collection = services.chroma_query(
            self.collection, self.collection_name,
            query_embeddings=[query_embedding], n_results=min(6, count),
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0] if results.get("documents") and results["documents"][0] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)
        if not docs:
            return "No documents retrieved. Please ingest a document first."

        # Feedback boost
        docs, metadatas = adaptive_db.apply_feedback_boost(docs, metadatas, self.arch_key)

        if on_step and docs:
            seen: dict = {}
            for text, meta in zip(docs, metadatas):
                src = (meta or {}).get("source", "Unknown")
                lbl = src.split("/")[-1].split("\\")[-1]
                if lbl not in seen:
                    seen[lbl] = text[:300]
            on_step(("sources", [{"text": t, "source": s} for s, t in seen.items()]))

        step("Building multimodal message (text + images)…")
        content = [
            {
                "type": "text",
                "text": f"Answer the user query using the following retrieved context. When comparing documents use [Source:] labels. Query: {query}\n\nContext:",
            }
        ]

        # Deduplicate by source so each file appears once with its richest chunk
        seen_sources: dict = {}
        for doc_text, meta in zip(docs, metadatas):
            source = (meta or {}).get("source", "Unknown")
            label = source.split("/")[-1].split("\\")[-1]
            ctx_text = services.get_context_text(doc_text, meta)
            if label not in seen_sources:
                seen_sources[label] = (ctx_text, meta)
            else:
                # Keep the longer context for the same source
                if len(ctx_text) > len(seen_sources[label][0]):
                    seen_sources[label] = (ctx_text, meta)

        for label, (ctx_text, meta) in seen_sources.items():
            content.append({"type": "text", "text": f"[Source: {label}]\n{ctx_text}\n"})
            img_path = (meta or {}).get("image_path", "")
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, "r") as fh:
                        b64_data = fh.read().strip()
                    mime = (meta or {}).get("image_mime", "image/jpeg")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                    })
                except Exception:
                    pass

        message = HumanMessage(content=content)

        step("Generating answer with Llama 4 Scout Vision…")
        try:
            full_text = ""
            for chunk in services.llm.stream([message]):
                token = services.extract_response_text(chunk)
                if token:
                    if on_step:
                        on_step(("token", token))
                    full_text += token
            return full_text
        except Exception as e:
            err_msg = str(e)
            if any(k in err_msg.lower() for k in ("vision", "image", "400", "unsupported", "invalid")):
                fallback = (
                    "⚠️ Vision processing failed for this image format. "
                    "Text context was retrieved successfully — try asking a text-only question about the document."
                )
                if on_step:
                    on_step(("token", fallback))
                return fallback
            raise
