from typing import List, Dict, TypedDict
import uuid
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from core.shared_services import services


class CRAGState(TypedDict):
    query: str
    documents: List[str]
    evaluation: str
    rewritten_query: str


class CorrectiveRAGPipeline:
    def __init__(self):
        self._on_step = None
        self.collection_name = "crag_collection"
        try:
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as e:
            print(f"[{self.collection_name}] init failed ({e}) — recreating")
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self._doc_sources: List[str] = []
        self.graph = self._build_graph()

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self._doc_sources = []

    def ingest(self, documents: List[Document]):
        if not documents:
            return
        existing = self.collection.count()
        texts = [doc.page_content for doc in documents]
        ids = [f"crag_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        self._doc_sources.extend([doc.metadata.get("source", "Unknown") for doc in documents])

    def retrieve_node(self, state: CRAGState) -> Dict:
        query = state["query"]
        if not self.collection.count():
            return {"documents": []}
        query_embedding = services.embeddings.embed_query(query)
        n = min(4, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"][0] else []
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        if self._on_step and docs:
            sources = [
                {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
                for text, meta in zip(docs, metas)
            ]
            self._on_step(("sources", sources))
        # Use windowed context for generation
        windowed_docs = [
            services.get_context_text(text, meta)
            for text, meta in zip(docs, metas)
        ]
        return {"documents": windowed_docs}

    def evaluate_node(self, state: CRAGState) -> Dict:
        query = state["query"]
        docs = state["documents"]
        # Use shared evaluate_context for consistency
        evaluation = services.evaluate_context(query, docs)
        _icons = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}
        if self._on_step:
            self._on_step(("step", f"Evaluator: {_icons.get(evaluation, '')} docs judged as `{evaluation}`"))
        return {"evaluation": evaluation}

    def rewrite_node(self, state: CRAGState) -> Dict:
        if self._on_step:
            self._on_step(("step", "Rewriting query for better web search…"))
        prompt = f"""Rewrite this query to be clearer and better suited for a web search.
Query: {state['query']}
Output ONLY the rewritten query."""
        response = services.llm.invoke(prompt)
        return {"rewritten_query": services.extract_response_text(response).strip()}

    def web_search_node(self, state: CRAGState) -> Dict:
        if self._on_step:
            self._on_step(("step", "Web search fallback — fetching external knowledge…"))
        query = state.get("rewritten_query") or state["query"]
        web_docs = services.web_search_fallback(query, n=3)
        return {"documents": state["documents"] + web_docs}

    def generate_node(self, state: CRAGState) -> Dict:
        if self._on_step:
            self._on_step(("step", "Generating answer with Gemini…"))
        docs = "\n\n".join(state["documents"])
        prompt = f"""Answer the user's query using the following context.
When the query asks to compare documents, use any [Source: ...] labels present to distinguish between them.

Context:
{docs}

Query: {state['query']}
Answer:"""

        def tok(t):
            if self._on_step:
                self._on_step(("token", t))

        text = services.stream_llm(prompt, on_token=tok)
        return {"documents": [text]}

    def route_evaluation(self, state: CRAGState) -> str:
        if state["evaluation"] == "CORRECT":
            return "generate_node"
        elif state["evaluation"] == "AMBIGUOUS":
            return "rewrite_node"
        else:
            return "web_search_node"

    def _build_graph(self):
        workflow = StateGraph(CRAGState)
        workflow.add_node("retrieve_node", self.retrieve_node)
        workflow.add_node("evaluate_node", self.evaluate_node)
        workflow.add_node("rewrite_node", self.rewrite_node)
        workflow.add_node("web_search_node", self.web_search_node)
        workflow.add_node("generate_node", self.generate_node)
        workflow.set_entry_point("retrieve_node")
        workflow.add_edge("retrieve_node", "evaluate_node")
        workflow.add_conditional_edges(
            "evaluate_node",
            self.route_evaluation,
            {
                "generate_node": "generate_node",
                "rewrite_node": "rewrite_node",
                "web_search_node": "web_search_node",
            },
        )
        workflow.add_edge("rewrite_node", "web_search_node")
        workflow.add_edge("web_search_node", "generate_node")
        workflow.add_edge("generate_node", END)
        return workflow.compile()

    def query(self, query: str, on_step=None) -> str:
        self._on_step = on_step
        initial_state = {
            "query": query,
            "documents": [],
            "evaluation": "",
            "rewritten_query": "",
        }
        trace = []
        final_answer = ""

        _EVAL_ICON = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}

        try:
            for s in self.graph.stream(initial_state, {"recursion_limit": 10}):
                for node_name, node_state in s.items():
                    if node_name == "evaluate_node":
                        ev = node_state.get("evaluation", "")
                        icon = _EVAL_ICON.get(ev, "❓")
                        trace.append(f"**Evaluator** {icon} ->Retrieved docs judged as `{ev}`")
                    elif node_name == "rewrite_node":
                        rq = node_state.get("rewritten_query", "")
                        trace.append(f"**Query Rewriter** ->`{rq}`")
                    elif node_name == "web_search_node":
                        trace.append("**Web Search** ->Fetched external knowledge")
                    elif node_name == "generate_node":
                        docs = node_state.get("documents", [])
                        if docs:
                            final_answer = docs[0]

            if not final_answer:
                final_answer = "Could not generate an answer. Please ingest a document first."

            if on_step:
                return final_answer

            trace_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(trace))
            return f"**CRAG Trace:**\n{trace_str}\n\n---\n\n{final_answer}"

        except Exception as e:
            return f"CRAG pipeline failed: {str(e)}"
        finally:
            self._on_step = None
