import uuid
from typing import List
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class HyDERAGPipeline:
    """
    Hypothetical Document Embeddings (HyDE) RAG.
    Generates a hypothetical answer first, embeds it, then retrieves real
    documents whose embeddings are closest to that hypothetical — bridging
    the vocabulary gap between short queries and long documents.
    """

    def __init__(self):
        self.arch_key = "08 HyDE RAG (Hypothetical Document)"
        self.collection_name = "hyde_rag_collection"
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
        ids = [f"hyde_{uuid.uuid4().hex}" for _ in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

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

        step("Generating hypothetical answer document…")
        hypothetical_doc = self._generate_hypothetical_document(query)

        step("Embedding hypothetical document into vector space…")
        hyde_embedding = services.embeddings.embed_query(hypothetical_doc)

        step("Retrieving real documents closest to hypothetical embedding…")
        n = min(top_k, self.collection.count())
        results, self.collection = services.chroma_query(
            self.collection, self.collection_name,
            query_embeddings=[hyde_embedding], n_results=n, include=["documents", "metadatas"],
        )
        docs = results["documents"][0] if results.get("documents") and results["documents"][0] else []
        metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)

        # Supplement with direct query retrieval if only 1 source found (HyDE can miss documents)
        distinct_sources = len({(m or {}).get("source", "Unknown") for m in metas})
        total_docs = self.collection.count()
        if distinct_sources < 2 and total_docs > len(docs):
            step("Supplementing with direct query retrieval for broader source coverage…")
            direct_embedding = services.embeddings.embed_query(query)
            supp_n = min(top_k, total_docs)
            supp_results, self.collection = services.chroma_query(
                self.collection, self.collection_name,
                query_embeddings=[direct_embedding], n_results=supp_n, include=["documents", "metadatas"],
            )
            supp_docs = supp_results["documents"][0] if supp_results.get("documents") and supp_results["documents"][0] else []
            supp_metas = supp_results["metadatas"][0] if supp_results.get("metadatas") and supp_results["metadatas"][0] else [{}] * len(supp_docs)
            existing_texts = set(docs)
            for d, m in zip(supp_docs, supp_metas):
                if d not in existing_texts:
                    docs.append(d)
                    metas.append(m)
                    existing_texts.add(d)

        # Self-evaluation
        quality = services.evaluate_context(query, docs)
        _icons = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}
        step(f"Context quality: {_icons.get(quality, '')} {quality}")

        if quality == "INCORRECT":
            step("Insufficient context — web search fallback…")
            web = services.web_search_fallback(query)
            if web:
                docs = web
                metas = [{}] * len(docs)
        elif quality == "AMBIGUOUS":
            web = services.web_search_fallback(query, n=2)
            if web:
                docs = docs + web
                metas = metas + [{}] * len(web)

        # Feedback boost — surface positively-rated chunks, demote negatives
        docs, metas = adaptive_db.apply_feedback_boost(docs, metas, self.arch_key)

        if on_step and docs:
            sources = [
                {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
                for text, meta in zip(docs, metas)
            ]
            on_step(("sources", sources))

        context = services.build_sourced_context(docs, metas)

        step("Generating final answer from retrieved real context…")
        prompt = f"""You are a helpful AI assistant. Answer the user's query using ONLY the provided context.
When the query asks to compare documents, use the [Source: ...] labels to distinguish between them.

Context:
{context}

Query: {query}

Answer:"""
        return services.stream_llm(
            prompt, on_token=lambda t: on_step and on_step(("token", t))
        )
