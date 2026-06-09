import os
import uuid
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from core.shared_services import services

class HybridRAGPipeline:
    def __init__(self):
        self.collection_name = "hybrid_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.bm25 = None
        self.chunks = []
        
    def ingest(self, documents: List[Document]):
        """Ingests chunks into ChromaDB and builds BM25 index."""
        if not documents:
            return
            
        self.chunks = documents
        
        # 1. Prepare for Dense Retrieval (ChromaDB)
        texts = [doc.page_content for doc in documents]
        ids = [f"hybrid_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        
        # We manually embed to store in ChromaDB easily without langchain abstractions if preferred,
        # but Chroma handles it or we can use Langchain's Chroma wrapper.
        # For simplicity, let's embed using our service and add to chroma collection:
        embeddings = services.embeddings.embed_documents(texts)
            
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # 2. Prepare for Sparse Retrieval (BM25)
        # We append to self.chunks so we can retrieve full documents later
        self.chunks.extend(documents)
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k=60) -> List[Dict]:
        """Fuses ranks from multiple retrieval methods."""
        fused_scores = {}
        
        for rank, doc in enumerate(dense_results):
            doc_id = doc['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'doc': doc, 'score': 0}
            fused_scores[doc_id]['score'] += 1 / (rank + k)
            
        for rank, doc in enumerate(sparse_results):
            doc_id = doc['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'doc': doc, 'score': 0}
            fused_scores[doc_id]['score'] += 1 / (rank + k)
            
        # Sort by fused score descending
        reranked = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in reranked]

    def query(self, query: str, top_k: int = 5) -> str:
        """Runs the Hybrid RAG query."""
        if not self.bm25 or not self.chunks:
            return "Please ingest a document first!"
            
        # 1. Dense Retrieval
        query_embedding = services.embeddings.embed_query(query)
        dense_response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        dense_results = []
        for i in range(len(dense_response['ids'][0])):
            dense_results.append({
                'id': dense_response['ids'][0][i],
                'text': dense_response['documents'][0][i]
            })
            
        # 2. Sparse Retrieval
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        # Get top_k sparse results
        top_sparse_indices = sparse_scores.argsort()[-top_k:][::-1]
        sparse_results = []
        for idx in top_sparse_indices:
            sparse_results.append({
                'id': f"sparse_{idx}",
                'text': self.chunks[idx].page_content
            })
            
        # 3. Fuse Results (RRF)
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        
        # Take top_k from fused
        final_top_k = fused_results[:top_k]
        context = "\n\n".join([doc['text'] for doc in final_top_k])
        
        # 4. Generate Answer using Gemini
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.
        
        Context:
        {context}
        
        Query: {query}
        
        Answer:"""
        
        response = services.llm.invoke(prompt)
        return services.extract_response_text(response)
