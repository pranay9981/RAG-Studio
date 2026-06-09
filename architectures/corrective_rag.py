from typing import List, Dict, TypedDict
import uuid
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from core.shared_services import services


class CRAGState(TypedDict):
    query: str
    documents: List[str]
    evaluation: str
    rewritten_query: str


class CorrectiveRAGPipeline:
    def __init__(self):
        self.collection_name = "crag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.graph = self._build_graph()

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

    def ingest(self, documents: List[Document]):
        if not documents:
            return
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        texts = [doc.page_content for doc in documents]
        ids = [f"crag_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def retrieve_node(self, state: CRAGState) -> Dict:
        query = state["query"]
        if not self.collection.count():
            return {"documents": []}
        query_embedding = services.embeddings.embed_query(query)
        n = min(3, self.collection.count())
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        return {"documents": results["documents"][0] if results["documents"][0] else []}

    def evaluate_node(self, state: CRAGState) -> Dict:
        query = state["query"]
        docs = "\n".join(state["documents"])
        if not docs.strip():
            return {"evaluation": "INCORRECT"}

        prompt = f"""You are a Relevance Evaluator.
Does the following context contain sufficient information to answer the query?

Context: {docs}
Query: {query}

Output exactly one word: CORRECT, AMBIGUOUS, or INCORRECT"""

        response = services.llm.invoke(prompt)
        result = services.extract_response_text(response).strip().upper()
        if "CORRECT" in result and "INCORRECT" not in result:
            evaluation = "CORRECT"
        elif "AMBIGUOUS" in result:
            evaluation = "AMBIGUOUS"
        else:
            evaluation = "INCORRECT"
        return {"evaluation": evaluation}

    def rewrite_node(self, state: CRAGState) -> Dict:
        prompt = f"""Rewrite this query to be clearer and better suited for a web search.
Query: {state['query']}
Output ONLY the rewritten query."""
        response = services.llm.invoke(prompt)
        return {"rewritten_query": services.extract_response_text(response).strip()}

    def web_search_node(self, state: CRAGState) -> Dict:
        query = state.get("rewritten_query") or state["query"]
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                search_texts = [f"{r['title']}: {r['body']}" for r in results]
                return {"documents": state["documents"] + search_texts}
        except Exception:
            return {"documents": state["documents"]}

    def generate_node(self, state: CRAGState) -> Dict:
        docs = "\n\n".join(state["documents"])
        prompt = f"""Answer the user's query using the following context.

Context:
{docs}

Query: {state['query']}
Answer:"""
        response = services.llm.invoke(prompt)
        return {"documents": [services.extract_response_text(response)]}

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
            {"generate_node": "generate_node", "rewrite_node": "rewrite_node", "web_search_node": "web_search_node"},
        )
        workflow.add_edge("rewrite_node", "web_search_node")
        workflow.add_edge("web_search_node", "generate_node")
        workflow.add_edge("generate_node", END)
        return workflow.compile()

    def query(self, query: str) -> str:
        initial_state = {"query": query, "documents": [], "evaluation": "", "rewritten_query": ""}
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

            trace_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(trace))
            return f"**CRAG Trace:**\n{trace_str}\n\n---\n\n{final_answer}"

        except Exception as e:
            return f"CRAG pipeline failed: {str(e)}"
