import uuid
from typing import List, Dict
from langchain_core.documents import Document
from core.shared_services import services


class RAGFusionPipeline:
    def __init__(self):
        self.collection_name = "rag_fusion_collection"
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
        ids = [f"ragfusion_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def _generate_sub_queries(self, query: str, n: int = 4) -> List[str]:
        prompt = f"""You are an expert at generating multiple search queries for document retrieval.
Generate {n} different versions of the following query to retrieve relevant documents from different angles and perspectives.
Output ONLY the queries, one per line, no numbering, no bullet points, no explanations.

Original query: {query}

{n} query variations:"""
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        queries = [q.strip().lstrip("-•123456789. ") for q in text.strip().split("\n") if q.strip()]
        queries = [q for q in queries if len(q) > 5]
        if len(queries) < n:
            queries += [query] * (n - len(queries))
        return queries[:n]

    def _reciprocal_rank_fusion(self, all_ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        fused_scores: Dict[str, Dict] = {}
        for ranked_list in all_ranked_lists:
            for rank, doc in enumerate(ranked_list):
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

        if not self.collection.count():
            return "Please ingest a document first!"

        step("Generating query variations with Gemini…")
        sub_queries = self._generate_sub_queries(query, n=4)

        all_ranked_lists = []
        for i, sub_q in enumerate(sub_queries):
            preview = sub_q[:55] + "…" if len(sub_q) > 55 else sub_q
            step(f"Retrieving for sub-query {i + 1}/{len(sub_queries)}: \"{preview}\"")
            q_embedding = services.embeddings.embed_query(sub_q)
            n = min(top_k, self.collection.count())
            results = self.collection.query(
                query_embeddings=[q_embedding],
                n_results=n,
                include=["documents", "metadatas"],
            )
            ranked = [
                {
                    "id": results["ids"][0][j],
                    "text": results["documents"][0][j],
                    "source": (results["metadatas"][0][j] or {}).get("source", "Unknown"),
                }
                for j in range(len(results["ids"][0]))
            ]
            all_ranked_lists.append(ranked)

        step(f"Fusing {len(sub_queries)} ranked lists with Reciprocal Rank Fusion…")
        fused = self._reciprocal_rank_fusion(all_ranked_lists)

        if on_step and fused:
            sources = [
                {"text": doc["text"][:300], "source": doc.get("source", "Unknown")}
                for doc in fused[:top_k]
            ]
            on_step(("sources", sources))

        context = "\n\n".join(doc["text"] for doc in fused[:top_k])

        step("Generating final answer with Gemini…")
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.
The context was retrieved using multiple query variations to maximise coverage.

Context:
{context}

Original Query: {query}

Answer:"""
        return services.stream_llm(prompt, on_token=lambda t: on_step and on_step(("token", t)))
