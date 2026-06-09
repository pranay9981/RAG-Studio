import os
import uuid
import base64
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from core.shared_services import services

class MultimodalRAGPipeline:
    def __init__(self):
        self.collection_name = "multimodal_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def ingest(self, documents: List[Document]):
        """Ingests chunks into ChromaDB. Supports text and base64 images in metadata."""
        if not documents:
            return

        # Clear stale data from any previous ingest
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

        texts = [doc.page_content for doc in documents]
        ids = [f"multi_mod_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]

        # ChromaDB metadata values must be str/int/float/bool — filter out anything else
        metadatas = []
        for doc in documents:
            safe_meta = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            metadatas.append(safe_meta)

        embeddings = services.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query: str) -> str:
        """Retrieves text and images, and sends them to Gemini for a multimodal answer."""
        if not self.collection.count():
            return "No documents ingested yet."
            
        query_embedding = services.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        content = [
            {"type": "text", "text": f"Answer the user query using the following retrieved context. Query: {query}\n\nContext:"}
        ]
        
        for idx, (doc_text, meta) in enumerate(zip(docs, metadatas)):
            content.append({"type": "text", "text": f"--- Document {idx+1} ---\n{doc_text}\n"})
            # If there's an image attached to this chunk, send it to the LLM!
            if meta and "image_base64" in meta:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{meta['image_base64']}"}
                })
                
        message = HumanMessage(content=content)
        
        response = services.llm.invoke([message])
        return services.extract_response_text(response)
