import os
import networkx as nx
import json
import uuid
from typing import List, Dict
from langchain_core.documents import Document
from core.shared_services import services
from core.adaptive_db import adaptive_db


class GraphRAGPipeline:
    def __init__(self):
        self.arch_key = "02 Graph RAG (Knowledge Graphs)"
        self.graph = nx.Graph()
        self.collection_name = "graph_rag_collection"
        try:
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as e:
            print(f"[{self.collection_name}] init failed ({e}) — recreating")
            try:
                services.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self._load_graph()

    @property
    def _graph_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db", "graph_rag.json"
        )

    def _save_graph(self):
        try:
            os.makedirs(os.path.dirname(self._graph_path), exist_ok=True)
            data = {
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {"source": u, "target": v, "relationship": d.get("relationship", "")}
                    for u, v, d in self.graph.edges(data=True)
                ],
            }
            with open(self._graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[graph_rag] Saved {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            print(f"[graph_rag] Failed to save graph: {e}")

    def _load_graph(self):
        if not os.path.exists(self._graph_path):
            return
        try:
            with open(self._graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph.clear()
            for edge in data.get("edges", []):
                self.graph.add_edge(
                    edge["source"], edge["target"],
                    relationship=edge.get("relationship", ""),
                )
            print(f"[graph_rag] Loaded {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges from disk")
        except Exception as e:
            print(f"[graph_rag] Failed to load graph: {e}")

    def reset(self):
        try:
            services.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = services.chroma_client.get_or_create_collection(self.collection_name)
        self.graph.clear()
        try:
            if os.path.exists(self._graph_path):
                os.remove(self._graph_path)
        except Exception:
            pass

    def _extract_entities_and_relationships(self, text: str) -> List[Dict]:
        prompt = f"""Extract key entities and relationships from the following text.
Output ONLY a valid JSON array of objects. Each object must have exactly three keys: "source", "target", "relationship".
Do not include markdown, backticks, or any explanation — just the raw JSON array.

Text:
{text}
"""
        try:
            response = services.llm.invoke(prompt)
            content = services.extract_response_text(response).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                content = content[start:end]
            triples = json.loads(content)
            return triples if isinstance(triples, list) else []
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def ingest(self, documents: List[Document]):
        if not documents:
            return

        existing = self.collection.count()
        texts = [doc.page_content for doc in documents]
        ids = [f"graph_{uuid.uuid4().hex[:8]}_{existing + i}" for i in range(len(documents))]
        metadatas = [
            {k: v for k, v in doc.metadata.items()
             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 8192}
            for doc in documents
        ]
        embeddings = services.embeddings.embed_documents(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

        for doc in documents:
            triples = self._extract_entities_and_relationships(doc.page_content)
            for triple in triples:
                if all(k in triple for k in ("source", "target", "relationship")):
                    self.graph.add_edge(
                        triple["source"],
                        triple["target"],
                        relationship=triple["relationship"],
                    )
        self._save_graph()

    def _extract_query_entities(self, query: str) -> List[str]:
        prompt = f"""Extract the main entities from this query. Output ONLY a comma-separated list of entity names, nothing else.
Query: {query}"""
        response = services.llm.invoke(prompt)
        text = services.extract_response_text(response)
        return [e.strip() for e in text.split(",") if e.strip()]

    def render_graph_html(self, max_nodes: int = 80) -> str:
        from pyvis.network import Network
        import tempfile, os

        if not self.graph.nodes():
            return ""

        net = Network(
            height="520px", width="100%",
            bgcolor="#0e1117", font_color="white",
            directed=True, notebook=False,
        )
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 120},
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
          },
          "edges": {
            "color": {"color": "#4a9eff"},
            "smooth": {"type": "curvedCW", "roundness": 0.15}
          }
        }
        """)

        sorted_nodes = sorted(
            self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True
        )
        nodes_to_show = set(sorted_nodes[:max_nodes])

        for node in nodes_to_show:
            degree = self.graph.degree(node)
            size = max(12, min(40, 12 + degree * 4))
            color = "#4a9eff" if degree > 2 else "#a0c4ff"
            net.add_node(
                str(node), label=str(node),
                title=f"{node}\n{degree} connection(s)", size=size, color=color,
            )

        for u, v, data in self.graph.edges(data=True):
            if u in nodes_to_show and v in nodes_to_show:
                rel = data.get("relationship", "")
                net.add_edge(str(u), str(v), label=rel, title=rel, arrows="to")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            net.write_html(f.name, notebook=False)
            tmp = f.name
        with open(tmp, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp)
        return html

    def query(self, query: str, on_step=None) -> str:
        def step(msg):
            if on_step:
                on_step(("step", msg))

        if not self.collection.count():
            return "Please ingest a document first!"

        step("Extracting query entities…")
        query_entities = self._extract_query_entities(query)

        step("Searching knowledge graph for matching nodes…")
        subgraph_lines = []
        for entity in query_entities:
            for node in self.graph.nodes():
                if entity.lower() in str(node).lower():
                    for neighbor in self.graph.neighbors(node):
                        rel = self.graph[node][neighbor].get("relationship", "associated with")
                        subgraph_lines.append(f"{node} --[{rel}]--> {neighbor}")
        graph_text = "\n".join(set(subgraph_lines))

        step("Dense retrieval from ChromaDB…")
        query_embedding = services.embeddings.embed_query(query)
        n = min(4, self.collection.count())
        dense_response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        dense_docs = dense_response["documents"][0] if dense_response["documents"][0] else []
        dense_metas = dense_response["metadatas"][0] if dense_response["metadatas"] else [{}] * len(dense_docs)

        # Self-evaluation
        quality = services.evaluate_context(query, dense_docs)
        _icons = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}
        step(f"Context quality: {_icons.get(quality, '')} {quality}")

        if quality == "INCORRECT":
            step("Insufficient context — web search fallback…")
            web = services.web_search_fallback(query)
            if web:
                dense_docs = web
                dense_metas = [{}] * len(dense_docs)
        elif quality == "AMBIGUOUS":
            web = services.web_search_fallback(query, n=2)
            if web:
                dense_docs = dense_docs + web
                dense_metas = dense_metas + [{}] * len(web)

        # Feedback boost — surface positively-rated chunks, demote negatives
        dense_docs, dense_metas = adaptive_db.apply_feedback_boost(dense_docs, dense_metas, self.arch_key)

        if on_step and dense_docs:
            sources = [
                {"text": text[:300], "source": (meta or {}).get("source", "Unknown")}
                for text, meta in zip(dense_docs, dense_metas)
            ]
            on_step(("sources", sources))

        vector_text = services.build_sourced_context(dense_docs, dense_metas)

        prompt = f"""You are GraphRAG. Answer the user's query using the Knowledge Graph relationships and semantic text below.
When the query asks to compare documents, use the [Source: ...] labels to distinguish between them.

Knowledge Graph Subgraph:
{graph_text if graph_text else "No specific relationships found."}

Semantic Text:
{vector_text if vector_text else "No semantic context available."}

Query: {query}

Answer:"""

        step("Generating with Llama 4 Scout…")
        return services.stream_llm(
            prompt, on_token=lambda t: on_step and on_step(("token", t))
        )
