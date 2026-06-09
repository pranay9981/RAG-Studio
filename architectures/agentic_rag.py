from typing import List, Dict, Any, Annotated, TypedDict
import operator
import uuid
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from core.shared_services import services

# Define the State for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: str
    is_sufficient: bool

class AgenticRAGPipeline:
    def __init__(self):
        self.collection_name = "agentic_rag_collection"
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.graph = self._build_graph()

    def ingest(self, documents: List[Document]):
        """Ingests chunks into ChromaDB for the Vector Search Tool."""
        if not documents:
            return

        # Clear stale data from any previous ingest
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)

        texts = [doc.page_content for doc in documents]
        ids = [f"agentic_chunk_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        embeddings = services.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
        )

    # --- Tool Definitions ---
    def _vector_search(self, query: str) -> str:
        """Search the internal database for documents."""
        if not self.collection.count():
            return "No documents have been ingested into the database yet."
            
        query_embedding = services.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        if not results['documents'][0]:
            return "No relevant documents found in the database."
        return "\n\n".join(results['documents'][0])

    def _web_search(self, query: str) -> str:
        """Search the web for information."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if not results:
                    return "No web results found."
                return "\n".join([f"{r['title']}: {r['body']}" for r in results])
        except Exception as e:
            return f"Web search failed: {e}"

    # --- Graph Nodes ---
    def planner_node(self, state: AgentState) -> Dict:
        """Plans which tool to use based on the query."""
        last_msg = state['messages'][-1].content
        prompt = f"""You are a Planner Agent. 
        Analyze this query: '{last_msg}'
        You must decide how to answer it.
        Choose exactly one option: "VECTOR_SEARCH", "WEB_SEARCH", or "ANSWER"
        - Use VECTOR_SEARCH if it asks about internal documents or the provided text.
        - Use WEB_SEARCH if it asks about current events, code, or general knowledge not in the documents.
        - Use ANSWER if you already have enough information from previous steps.
        
        Output ONLY the option name."""
        
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response).strip()
        # Clean up output just in case
        if "VECTOR_SEARCH" in text: plan = "VECTOR_SEARCH"
        elif "WEB_SEARCH" in text: plan = "WEB_SEARCH"
        else: plan = "ANSWER"
        
        return {"plan": plan, "messages": [AIMessage(content=f"[Planner thought] I decided to route this to: {plan}")]}

    def tool_executor_node(self, state: AgentState) -> Dict:
        """Executes the chosen tool."""
        query = state['messages'][0].content # Original user query
        plan = state['plan']
        
        if plan == "VECTOR_SEARCH":
            result = self._vector_search(query)
            tool_msg = f"[Tool Output: Vector Search]\n{result}"
        else:
            result = self._web_search(query)
            tool_msg = f"[Tool Output: Web Search]\n{result}"
            
        return {"messages": [AIMessage(content=tool_msg)]}

    def reasoner_node(self, state: AgentState) -> Dict:
        """Synthesizes the final answer using the collected context."""
        context = "\n".join([m.content for m in state['messages'] if m.content.startswith("[Tool Output")])
        query = state['messages'][0].content
        
        prompt = f"""You are the Reasoner Agent. Use the tool outputs provided below to answer the user's query.
        
        Tool Outputs:
        {context}
        
        User Query: {query}
        
        Answer directly and comprehensively:"""
        
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        return {"messages": [AIMessage(content=text)]}

    # --- Routing logic ---
    def should_continue(self, state: AgentState) -> str:
        """Decide next node based on the plan."""
        if state['plan'] == "ANSWER":
            return "reasoner_node"
        return "tool_executor_node"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner_node", self.planner_node)
        workflow.add_node("tool_executor_node", self.tool_executor_node)
        workflow.add_node("reasoner_node", self.reasoner_node)
        
        workflow.set_entry_point("planner_node")
        
        workflow.add_conditional_edges(
            "planner_node",
            self.should_continue,
            {
                "tool_executor_node": "tool_executor_node",
                "reasoner_node": "reasoner_node"
            }
        )
        
        workflow.add_edge("tool_executor_node", "reasoner_node")
        workflow.add_edge("reasoner_node", END)
        
        return workflow.compile()

    def query(self, query: str) -> str:
        """Runs the LangGraph Agent pipeline."""
        initial_state = {"messages": [HumanMessage(content=query)], "plan": "", "is_sufficient": False}
        
        # We capture intermediate states to show the "Agent is thinking..." process
        final_answer = ""
        trace = []
        
        try:
            for s in self.graph.stream(initial_state, {"recursion_limit": 10}):
                for node_name, node_state in s.items():
                    if 'messages' in node_state and len(node_state['messages']) > 0:
                        latest_msg = node_state['messages'][-1].content
                        trace.append(latest_msg)
                        final_answer = latest_msg
            
            # Combine the trace into the final output so the user sees the reasoning
            reasoning_trace = "\n\n".join(trace[:-1]) # Everything except the final answer
            return f"### Agent Reasoning Trace:\n{reasoning_trace}\n\n### Final Answer:\n{final_answer}"
        except Exception as e:
            return f"Agent loop failed: {str(e)}"
