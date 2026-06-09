from typing import List, Dict, Any, Annotated, TypedDict
import operator
import uuid
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
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

    def ingest(self, documents: List[Document]):
        """Ingests chunks into ChromaDB for CRAG retrieval."""
        if not documents:
            return
            
        texts = [doc.page_content for doc in documents]
        ids = [f"crag_chunk_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )

    # --- Graph Nodes ---
    def retrieve_node(self, state: CRAGState) -> Dict:
        """Retrieves initial documents from Vector DB."""
        query = state["query"]
        if not self.collection.count():
            return {"documents": ["No documents ingested yet."]}
            
        query_embedding = services.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        if not results['documents'][0]:
            return {"documents": []}
            
        return {"documents": results['documents'][0]}

    def evaluate_node(self, state: CRAGState) -> Dict:
        """Evaluates if the retrieved documents are relevant to the query."""
        query = state["query"]
        docs = "\n".join(state["documents"])
        
        prompt = f"""You are a Relevance Evaluator. 
        Analyze if the following documents contain sufficient information to answer the user query.
        
        Documents: {docs}
        Query: {query}
        
        Output exactly one of the following words: CORRECT, AMBIGUOUS, INCORRECT"""
        
        response = services.llm.invoke(prompt)
        eval_result = services.extract_response_text(response).strip().upper()
        if "CORRECT" in eval_result and "INCORRECT" not in eval_result:
            evaluation = "CORRECT"
        elif "AMBIGUOUS" in eval_result:
            evaluation = "AMBIGUOUS"
        else:
            evaluation = "INCORRECT"
            
        return {"evaluation": evaluation}

    def rewrite_node(self, state: CRAGState) -> Dict:
        """Rewrites the query if evaluation is AMBIGUOUS."""
        query = state["query"]
        prompt = f"""Rewrite the following user query to make it clearer and better suited for a web search engine.
        Query: {query}
        Output ONLY the rewritten query."""
        
        response = services.llm.invoke(prompt)
        rewritten = services.extract_response_text(response).strip()
        return {"rewritten_query": rewritten}

    def web_search_node(self, state: CRAGState) -> Dict:
        """Performs a web search fallback."""
        query = state.get("rewritten_query", state["query"])
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                search_texts = [f"{r['title']}: {r['body']}" for r in results]
                # Combine original valid docs with web search docs
                return {"documents": state["documents"] + search_texts}
        except Exception:
            return {"documents": state["documents"]}

    def generate_node(self, state: CRAGState) -> Dict:
        """Generates the final answer."""
        query = state["query"]
        docs = "\n\n".join(state["documents"])
        
        prompt = f"""Answer the user's query using the following context.
        
        Context:
        {docs}
        
        Query: {query}
        Answer:"""
        
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        return {"documents": [text]} # We piggyback the answer in the documents field to extract easily

    # --- Routing logic ---
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
                "web_search_node": "web_search_node"
            }
        )
        
        workflow.add_edge("rewrite_node", "web_search_node")
        workflow.add_edge("web_search_node", "generate_node")
        workflow.add_edge("generate_node", END)
        
        return workflow.compile()

    def query(self, query: str) -> str:
        """Runs the CRAG pipeline."""
        initial_state = {"query": query, "documents": [], "evaluation": "", "rewritten_query": ""}
        
        trace = []
        final_answer = ""
        
        try:
            for s in self.graph.stream(initial_state, {"recursion_limit": 10}):
                for node_name, node_state in s.items():
                    if node_name == "evaluate_node":
                        trace.append(f"[Evaluator] Judged retrieved docs as: {node_state['evaluation']}")
                    elif node_name == "rewrite_node":
                        trace.append(f"[Rewriter] Rewrote query to: {node_state['rewritten_query']}")
                    elif node_name == "web_search_node":
                        trace.append(f"[Web Search] Fetched external knowledge.")
                    elif node_name == "generate_node":
                        final_answer = node_state["documents"][0]
            
            trace_str = "\n> ".join(trace)
            return f"### CRAG Execution Trace:\n> {trace_str}\n\n### Final Answer:\n{final_answer}"
        except Exception as e:
            return f"CRAG loop failed: {str(e)}"
