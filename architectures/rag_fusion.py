import uuid
from typing import List, Dict
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class RAGFusionPipeline:
    def __init__(self):
        self.arch_key = "07 RAG-Fusion (Query Expansion)"
        self.collection_name = "rag_fusion_collection"
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

        texts = [doc.page_content for doc in documents]
        metadatas = [
            {k: v for k, v in doc.metadata.items()
             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 8192}
            for doc in documents
        ]
        ids = [f"ragfusion_{uuid.uuid4().hex}" for _ in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)
        with services._chroma_lock:
            self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

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
        return queries[:n] if queries else [query]

    def _reciprocal_rank_fusion(
        self, all_ranked_lists: List[List[Dict]], k: int = 60
    ) -> List[Dict]:
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

        step("Generating query variations with Llama 4 Scout…")
        sub_queries = self._generate_sub_queries(query, n=4)

        all_ranked_lists = []
        for i, sub_q in enumerate(sub_queries):
            preview = sub_q[:55] + "…" if len(sub_q) > 55 else sub_q
            step(f"Retrieving for sub-query {i + 1}/{len(sub_queries)}: \"{preview}\"")
            q_embedding = services.embeddings.embed_query(sub_q)
            n = min(top_k, self.collection.count())
            results, self.collection = services.chroma_query(
                self.collection, self.collection_name,
                query_embeddings=[q_embedding], n_results=n, include=["documents", "metadatas"],
            )
            ranked = [
                {
                    "id": results["ids"][0][j],
                    "text": results["documents"][0][j],
                    "source": (results["metadatas"][0][j] or {}).get("source", "Unknown"),
                    "window_text": (results["metadatas"][0][j] or {}).get("window_text"),
                }
                for j in range(len(results["ids"][0]))
            ]
            all_ranked_lists.append(ranked)

        step(f"Fusing {len(sub_queries)} ranked lists with Reciprocal Rank Fusion…")
        fused = self._reciprocal_rank_fusion(all_ranked_lists)
        top_docs = fused[:top_k]

        # Self-evaluation
        top_texts = [d["text"] for d in top_docs]
        quality = services.evaluate_context(query, top_texts)
        _icons = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}
        step(f"Context quality: {_icons.get(quality, '')} {quality}")

        if quality == "INCORRECT":
            step("Insufficient context — web search fallback…")
            web = services.web_search_fallback(query)
            if web:
                top_texts = web
                top_docs = [{"text": t, "source": "web", "window_text": None} for t in web]
        elif quality == "AMBIGUOUS":
            web = services.web_search_fallback(query, n=2)
            if web:
                top_texts = top_texts + web
                top_docs = top_docs + [{"text": t, "source": "web", "window_text": None} for t in web]

        # Feedback boost — reorder top_docs by feedback signal
        doc_texts = [d["text"] for d in top_docs]
        dummy_metas = [{} for _ in top_docs]
        boosted_texts, _ = adaptive_db.apply_feedback_boost(doc_texts, dummy_metas, self.arch_key)
        text_to_doc = {d["text"]: d for d in top_docs}
        top_docs = [text_to_doc.get(t, {"text": t, "source": "Unknown", "window_text": None}) for t in boosted_texts]

        if on_step and top_docs:
            sources = [
                {"text": doc["text"][:300], "source": doc.get("source", "Unknown")}
                for doc in top_docs[:top_k]
            ]
            on_step(("sources", sources))

        fused_texts = [d["text"] for d in top_docs[:top_k]]
        fused_metas = [{"source": d.get("source", "Unknown"), "parent_text": d.get("window_text")} for d in top_docs[:top_k]]
        context = services.build_sourced_context(fused_texts, fused_metas)

        step("Generating with Llama 4 Scout…")
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.
The context was retrieved using multiple query variations to maximise coverage.
When the query asks to compare documents, use the [Source: ...] labels to distinguish between them.

Context:
{context}

Original Query: {query}

Answer:"""
        return services.stream_llm(
            prompt, on_token=lambda t: on_step and on_step(("token", t))
        )
