from typing import List, Dict, Annotated, TypedDict
import operator
import uuid
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from core.shared_services import services


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: str
    is_sufficient: bool


class AgenticRAGPipeline:
    def __init__(self):
        self._on_step = None
        self.collection_name = "agentic_rag_collection"
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
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
        existing = self.collection.count()
        texts = [doc.page_content for doc in documents]
        ids = [f"agentic_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def _vector_search(self, query: str) -> tuple:
        if not self.collection.count():
            return "No documents have been ingested yet.", []
        query_embedding = services.embeddings.embed_query(query)
        n = min(4, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"][0] else []
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        sources = [
            {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
            for text, meta in zip(docs, metas)
        ]
        context = "\n\n".join(
            services.get_context_text(text, meta)
            for text, meta in zip(docs, metas)
        ) if docs else "No relevant documents found."
        return context, sources

    def _web_search(self, query: str) -> str:
        results = services.web_search_fallback(query)
        return "\n".join(results) if results else "No web results found."

    def _maybe_decompose(self, query: str) -> List[str]:
        """For complex multi-part queries, decomposes into sub-questions."""
        if query.count("?") <= 1 and " and " not in query.lower():
            return [query]
        prompt = f"""If this query requires multiple distinct pieces of information, break it into 2-3 sub-questions.
If it is a single question, output it unchanged.
Output only the questions, one per line, no numbering.

Query: {query}

Questions:"""
        try:
            response = services.llm.invoke(prompt)
            text = services.extract_response_text(response)
            subs = [q.strip().lstrip("0123456789.-) ") for q in text.strip().split("\n") if q.strip()]
            subs = [q for q in subs if len(q) > 5]
            return subs[:3] if len(subs) > 1 else [query]
        except Exception:
            return [query]

    def planner_node(self, state: AgentState) -> Dict:
        last_msg = state["messages"][-1].content
        has_docs = self.collection.count() > 0
        prompt = f"""You are a Planner Agent. Analyze this query and decide how to answer it.
Query: '{last_msg}'
Documents available in database: {'YES' if has_docs else 'NO'}

Choose exactly one:
- VECTOR_SEARCH  ->query asks about the uploaded document / provided text
- WEB_SEARCH     ->query needs current events or general knowledge not in the document
- ANSWER         ->you already have enough context from previous steps

Output ONLY the option name, nothing else."""

        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response).strip()
        if "VECTOR_SEARCH" in text:
            plan = "VECTOR_SEARCH"
        elif "WEB_SEARCH" in text:
            plan = "WEB_SEARCH"
        else:
            plan = "ANSWER"
        if self._on_step:
            self._on_step(("step", f"Planner → {plan}"))
        return {"plan": plan, "messages": [AIMessage(content=f"__plan__{plan}")]}

    def tool_executor_node(self, state: AgentState) -> Dict:
        query = state["messages"][0].content
        plan = state["plan"]

        if plan == "VECTOR_SEARCH":
            # Multi-hop: decompose complex queries into sub-questions
            sub_queries = self._maybe_decompose(query)
            if len(sub_queries) > 1:
                if self._on_step:
                    self._on_step(("step", f"Multi-hop query — decomposed into {len(sub_queries)} sub-questions…"))
                all_contexts = []
                all_sources = []
                for sub in sub_queries:
                    ctx, srcs = self._vector_search(sub)
                    all_contexts.append(f"Sub-question '{sub}':\n{ctx}")
                    all_sources.extend(srcs)
                if self._on_step and all_sources:
                    self._on_step(("sources", all_sources[:6]))
                tool_msg = f"__tool__vector\n" + "\n\n---\n\n".join(all_contexts)
            else:
                if self._on_step:
                    self._on_step(("step", "Vector Search — querying ingested document…"))
                result, sources = self._vector_search(query)
                if self._on_step and sources:
                    self._on_step(("sources", sources))
                tool_msg = f"__tool__vector\n{result}"
        else:
            if self._on_step:
                self._on_step(("step", "Web Search — fetching external knowledge…"))
            result = self._web_search(query)
            tool_msg = f"__tool__web\n{result}"

        return {"messages": [AIMessage(content=tool_msg)]}

    def reasoner_node(self, state: AgentState) -> Dict:
        query = state["messages"][0].content
        context_parts = []
        for m in state["messages"]:
            if m.content.startswith("__tool__"):
                raw = m.content.split("\n", 1)
                if len(raw) > 1:
                    context_parts.append(raw[1])
        context = "\n\n".join(context_parts) if context_parts else "No retrieved context available."

        prompt = f"""Answer the user's query using the context below.
When the query asks to compare documents, use any [Source: ...] labels to distinguish between them.

Context:
{context}

User Query: {query}

Answer directly and comprehensively:"""
        if self._on_step:
            self._on_step(("step", "Reasoner — generating final answer…"))

        def tok(t):
            if self._on_step:
                self._on_step(("token", t))

        text = services.stream_llm(prompt, on_token=tok)
        return {"messages": [AIMessage(content=text)]}

    def should_continue(self, state: AgentState) -> str:
        if state["plan"] == "ANSWER":
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
            {"tool_executor_node": "tool_executor_node", "reasoner_node": "reasoner_node"},
        )
        workflow.add_edge("tool_executor_node", "reasoner_node")
        workflow.add_edge("reasoner_node", END)
        return workflow.compile()

    def query(self, query: str, on_step=None) -> str:
        self._on_step = on_step
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "plan": "",
            "is_sufficient": False,
        }
        trace_steps = []
        final_answer = ""

        try:
            for s in self.graph.stream(initial_state, {"recursion_limit": 10}):
                for node_name, node_state in s.items():
                    if "messages" not in node_state or not node_state["messages"]:
                        continue
                    msg = node_state["messages"][-1].content

                    if node_name == "planner_node":
                        if "VECTOR_SEARCH" in msg:
                            trace_steps.append("**Planner** ->`VECTOR_SEARCH` *(checking ingested document)*")
                        elif "WEB_SEARCH" in msg:
                            trace_steps.append("**Planner** ->`WEB_SEARCH` *(fetching from the web)*")
                        else:
                            trace_steps.append("**Planner** ->`ANSWER` *(sufficient context available)*")

                    elif node_name == "tool_executor_node":
                        if msg.startswith("__tool__vector"):
                            lines = [l for l in msg.split("\n")[1:] if l.strip()]
                            trace_steps.append(f"**Vector Search** ->Retrieved {len(lines)} chunk(s) from document")
                        else:
                            trace_steps.append("**Web Search** ->Retrieved external context")

                    elif node_name == "reasoner_node":
                        final_answer = msg

            if not final_answer:
                final_answer = "Could not generate an answer. Please ingest a document first."

            if on_step:
                return final_answer

            trace_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(trace_steps))
            return f"**Agent Trace:**\n{trace_str}\n\n---\n\n{final_answer}"

        except Exception as e:
            return f"Agent pipeline failed: {str(e)}"
        finally:
            self._on_step = None
