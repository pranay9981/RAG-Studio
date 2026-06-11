import re
import json
import uuid
from typing import List, Tuple
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class SelfRAGPipeline:
    """
    Self-RAG: Retrieval-Augmented Generation with Self-Reflection.

    Simulates the tokens from the Self-RAG paper:
    - [IsRel]  per-doc relevance grading after retrieval
    - [IsSup]  faithfulness critique of the generated answer
    - [IsUse]  completeness critique of the generated answer

    If the first answer scores below threshold on either dimension, the query is
    refined towards the identified gaps and a second retrieval + generation loop
    runs before streaming the final answer.
    """

    def __init__(self):
        self.arch_key = "10 Self-RAG (Reflection + Critique)"
        self.collection_name = "self_rag_collection"
        try:
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as e:
            print(f"[{self.collection_name}] collection init failed ({e}) — recreating")
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
        ids = [f"selfrag_{uuid.uuid4().hex}" for _ in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def _retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[dict]]:
        n = min(top_k, self.collection.count())
        with services._chroma_lock:
            results = self.collection.query(
                query_embeddings=[services.embeddings.embed_query(query)],
                n_results=n,
                include=["documents", "metadatas"],
            )
        docs = results["documents"][0] if results.get("documents") and results["documents"][0] else []
        metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)
        return docs, metas

    def _grade_relevance(self, query: str, docs: List[str]) -> List[bool]:
        """[IsRel] — grade each chunk for relevance to the query in one LLM call."""
        if not docs:
            return []
        doc_list = "\n".join(f"[{i}]: {d[:250]}" for i, d in enumerate(docs))
        prompt = f"""For each document chunk, decide if it contains ANY information that could help answer the query.
Be generous — mark true if the chunk is even partially useful or topically related.
Only mark false if the chunk is completely unrelated to the query topic.
Output ONLY a JSON array of booleans, one per document in order.

Query: {query}

Documents:
{doc_list}

Output (e.g. [true, false, true]):"""
        try:
            response = services.llm.invoke(prompt)
            text = services.extract_response_text(response)
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                grades = json.loads(m.group())
                return [bool(g) for g in grades[:len(docs)]]
        except Exception as e:
            print(f"[self_rag] relevance grading failed: {e}")
        return [True] * len(docs)

    def _critique(self, query: str, answer: str, context: str) -> dict:
        """[IsSup] + [IsUse] — critique faithfulness and completeness."""
        prompt = f"""Rate this RAG answer on two dimensions.

Query: {query}
Context: {context[:1200]}
Answer: {answer[:500]}

Score each 0–10:
- faithfulness: are all claims in the answer supported by the context?
- completeness: does the answer fully address all aspects of the query?
- missing: specific information that is missing or unsupported (write "NONE" if nothing is missing)

Output ONLY valid JSON: {{"faithfulness": X, "completeness": X, "missing": "..."}}"""
        try:
            resp = services.llm.invoke(prompt)
            text = services.extract_response_text(resp)
            m = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return {
                    "faithfulness": max(0, min(10, int(data.get("faithfulness", 7)))),
                    "completeness": max(0, min(10, int(data.get("completeness", 7)))),
                    "missing": str(data.get("missing", "NONE")),
                }
        except Exception as e:
            print(f"[self_rag] critique failed: {e}")
        return {"faithfulness": 7, "completeness": 7, "missing": "NONE"}

    def _refine_query(self, original: str, missing: str) -> str:
        """Generate a focused search query targeting gaps identified by critique."""
        prompt = f"""A RAG answer was incomplete. Write a focused search query to fill the gap.

Original query: {original}
Missing information: {missing}

Output ONLY the refined query (1–2 sentences):"""
        try:
            resp = services.llm.invoke(prompt)
            refined = services.extract_response_text(resp).strip()
            return refined if len(refined) > 5 else original
        except Exception as e:
            print(f"[self_rag] query refinement failed: {e}")
            return original

    def query(self, query: str, top_k: int = 7, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "Please ingest a document first!"

        MAX_LOOPS = 2
        seen_texts: set = set()
        all_docs: List[str] = []
        all_metas: List[dict] = []
        current_query = query
        gen_prompt = ""
        draft_answer = ""

        for loop in range(MAX_LOOPS):
            label = f"{loop + 1}/{MAX_LOOPS}"
            step(f"Retrieving candidate documents (loop {label})…")
            docs, metas = self._retrieve(current_query, top_k)

            # Deduplicate across loops
            new_docs, new_metas = [], []
            for d, m in zip(docs, metas):
                if d not in seen_texts:
                    seen_texts.add(d)
                    new_docs.append(d)
                    new_metas.append(m)

            if new_docs:
                step(f"Grading {len(new_docs)} docs for relevance [IsRel]…")
                relevance = self._grade_relevance(query, new_docs)
                rel_docs = [d for d, r in zip(new_docs, relevance) if r]
                rel_metas = [m for m, r in zip(new_metas, relevance) if r]
                if not rel_docs:
                    # Grader rejected everything — use top 3 rather than falling back to web
                    rel_docs = new_docs[:3]
                    rel_metas = new_metas[:3]
                    step(f"All docs graded low-relevance — using top 3 as fallback")
                else:
                    step(f"Relevance filter: {len(rel_docs)}/{len(new_docs)} docs kept")
                all_docs.extend(rel_docs)
                all_metas.extend(rel_metas)

            if not all_docs:
                step("No relevant docs — web search fallback…")
                web = services.web_search_fallback(query)
                if web:
                    all_docs = web
                    all_metas = [{}] * len(web)
                else:
                    return "No relevant information found in ingested documents or web."

            # Feedback boost — surface positively-rated chunks, demote negatives
            all_docs, all_metas = adaptive_db.apply_feedback_boost(all_docs, all_metas, self.arch_key)

            # Emit sources once on first loop
            if loop == 0 and on_step and all_docs:
                on_step(("sources", [
                    {"text": t[:300], "source": (m or {}).get("source", "Unknown")}
                    for t, m in zip(all_docs[:5], all_metas[:5])
                ]))

            context = services.build_sourced_context(all_docs[:5], all_metas[:5])
            gen_prompt = f"""You are a precise and faithful AI assistant.
Answer using ONLY the provided context. Use [Source: ...] labels when comparing documents.
Do not add information beyond what the context supports.

Context:
{context}

Query: {query}

Answer:"""

            is_final = (loop == MAX_LOOPS - 1)

            if is_final:
                step("Streaming final answer…")
                return services.stream_llm(
                    gen_prompt, on_token=lambda t: on_step and on_step(("token", t))
                )

            # Non-final loop: generate silently, then self-critique
            step("Generating draft answer…")
            resp = services.llm.invoke(gen_prompt)
            draft_answer = services.extract_response_text(resp)

            step("Self-critique: checking faithfulness [IsSup] + completeness [IsUse]…")
            critique = self._critique(query, draft_answer, context)
            faith = critique["faithfulness"]
            comp = critique["completeness"]
            missing = critique["missing"]
            step(f"Critique → Faithfulness: {faith}/10 · Completeness: {comp}/10")

            if faith >= 7 and comp >= 7:
                step("Quality threshold met — streaming final answer…")
                return services.stream_llm(
                    gen_prompt, on_token=lambda t: on_step and on_step(("token", t))
                )

            if missing and missing.upper() != "NONE":
                step(f"Gaps found: {missing[:80]}… — refining query…")
                current_query = self._refine_query(query, missing)
            else:
                step("Scores below threshold — broadening retrieval…")
                top_k = top_k + 3  # _retrieve() already caps to collection.count()

        if draft_answer and gen_prompt:
            step("Streaming best available answer…")
            return services.stream_llm(
                gen_prompt, on_token=lambda t: on_step and on_step(("token", t))
            )
        return "Unable to generate a satisfactory answer for this query."
